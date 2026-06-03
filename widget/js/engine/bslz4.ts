/// <reference types="@webgpu/types" />
// Offline GPU decompression of HDF5 bitshuffle+LZ4 (bslz4) 4D-STEM data.
//
// Ships the NATIVE detector codec (Dectris/Arina write bslz4) so the browser
// downloads ~6x less than raw uint16 and decompresses on WebGPU - no Python, no
// server. Two passes, both verified bit-exact vs h5py on real gold (192x192
// uint16): Pass1 LZ4-decodes each independent block, Pass2 inverts the bit
// transpose. Output is uint16-packed in [scanPos][detPixel] order - the exact
// layout Show4DSTEMCompute reads in uint16 mode, so it feeds masked_sum /
// reduce_frames with no copy.
//
// Per-frame chunk layout (one HDF5 chunk = one diffraction pattern):
//   blockMeta gives, per (frame,block), the absolute byte offset + compressed
//   length of that block's LZ4 stream within the concatenated `compressed`. Each
//   block decompresses to blockElemBytes = blockElems * elemBytes, holds
//   nbits = elemBytes*8 bit-planes of planeBytes = blockElems/8 each.

import { getGPUDevice } from "./device";

// Pass1: one thread per block, LZ4-decode into the `inter` (bitshuffled) buffer.
// Blocks are independent -> embarrassingly parallel. Byte-addressed RMW because
// WGSL storage is u32-only.
const PASS1_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read_write> inter: array<u32>;
@group(0) @binding(2) var<storage,read> blkMeta: array<u32>;   // coff,clen per block
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;             // totalBlocks, blockBytes
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rout(i:u32)->u32{return (inter[i>>2u]>>((i&3u)*8u))&0xffu;}
fn wout(i:u32,v:u32){let w=i>>2u;let s=(i&3u)*8u;inter[w]=(inter[w]&(~(0xffu<<s)))|((v&0xffu)<<s);}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>){
  let g=gid.x; if(g>=cfg.x){return;}
  let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u]; let base=g*cfg.y;
  var ci=coff; var di=0u;
  loop{ if(ci>=cend){break;}
    let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
    if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
    var k=0u; loop{if(k>=nlit){break;} wout(base+di+k,rraw(ci+k)); k=k+1u;} ci=ci+nlit; di=di+nlit;
    if(ci>=cend){break;}
    let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
    if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
    var j=0u; loop{if(j>=ml){break;} wout(base+di+j,rout(base+di+j-off)); j=j+1u;} di=di+ml;
  }
}`;

// Pass2: inverse bitshuffle. One thread per GROUP of 8 consecutive pixels - they
// share the same 16 plane-bytes (one byte per bit-plane), so 16 global reads feed
// 8 outputs (8x fewer reads than per-pixel -> ~4x faster, measured). Writes 4
// uint16-packed u32. Plane-major, LSB-first: pixel (group*8 + i) bit b = bit i of
// plane-byte (b*planeBytes + group_byte). cfg = nGroups, strideX, blockElems, planeBytes.
const PASS2_WGSL = `
@group(0) @binding(0) var<storage,read> inter: array<u32>;
@group(0) @binding(1) var<storage,read_write> stack: array<u32>;  // uint16-packed
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;  // nGroups, strideX, blockElems, planeBytes
fn byteAt(o:u32)->u32{return (inter[o>>2u]>>((o&3u)*8u))&0xffu;}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>){
  let grp=gid.y*cfg.y + gid.x; if(grp>=cfg.x){return;}
  let blockElems=cfg.z; let planeBytes=cfg.w; let blockBytes=blockElems*2u;
  let nBlk=__NBLK__; let framePix=__FRAMEPIX__;
  let e0=grp*8u; let frm=e0/framePix; let inFrame=e0%framePix;
  let blk=inFrame/blockElems; let groupByte=(inFrame%blockElems)>>3u;
  let pb=(frm*nBlk+blk)*blockBytes + groupByte;
  var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
  for(var b:u32=0u;b<16u;b=b+1u){
    let byte=byteAt(pb+b*planeBytes); let bit=1u<<b;
    if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
    if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
    if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
    if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
  }
  let o=grp*4u;
  stack[o]=v0|(v1<<16u); stack[o+1u]=v2|(v3<<16u); stack[o+2u]=v4|(v5<<16u); stack[o+3u]=v6|(v7<<16u);
}`;

// Pass2 (uint8): same 8-pixel-group inverse bitshuffle, but clip(0,255) and pack 4
// pixels per u32 - halves resident memory (9.6 GB vs 19 GB for full 512x512). Offline
// default: real detector counts are 0-~50, so clip is near-lossless for the signal.
const PASS2_U8_WGSL = `
@group(0) @binding(0) var<storage,read> inter: array<u32>;
@group(0) @binding(1) var<storage,read_write> stack: array<u32>;  // uint8-packed (4/u32)
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;  // nGroups, strideX, blockElems, planeBytes
fn byteAt(o:u32)->u32{return (inter[o>>2u]>>((o&3u)*8u))&0xffu;}
fn clip8(v:u32)->u32{return select(v,255u,v>255u);}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>){
  let grp=gid.y*cfg.y + gid.x; if(grp>=cfg.x){return;}
  let blockElems=cfg.z; let planeBytes=cfg.w; let blockBytes=blockElems*2u;
  let nBlk=__NBLK__; let framePix=__FRAMEPIX__;
  let e0=grp*8u; let frm=e0/framePix; let inFrame=e0%framePix;
  let blk=inFrame/blockElems; let groupByte=(inFrame%blockElems)>>3u;
  let pb=(frm*nBlk+blk)*blockBytes + groupByte;
  var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
  for(var b:u32=0u;b<16u;b=b+1u){
    let byte=byteAt(pb+b*planeBytes); let bit=1u<<b;
    if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
    if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
    if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
    if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
  }
  let o=grp*2u;
  stack[o]=clip8(v0)|(clip8(v1)<<8u)|(clip8(v2)<<16u)|(clip8(v3)<<24u);
  stack[o+1u]=clip8(v4)|(clip8(v5)<<8u)|(clip8(v6)<<16u)|(clip8(v7)<<24u);
}`;

// Pass2 (uint8 SOURCE): companion encoded from uint8 (typesize 1) -> only 8 bit
// planes, and the LZ4 block is half the bytes -> ~2x faster than the uint16-source
// path, BOTH passes. Output is uint8-packed directly (values already <= 255).
const PASS2_U8SRC_WGSL = `
@group(0) @binding(0) var<storage,read> inter: array<u32>;
@group(0) @binding(1) var<storage,read_write> stack: array<u32>;  // uint8-packed (4/u32)
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;  // nGroups, strideX, blockElems, planeBytes
fn byteAt(o:u32)->u32{return (inter[o>>2u]>>((o&3u)*8u))&0xffu;}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>){
  let grp=gid.y*cfg.y + gid.x; if(grp>=cfg.x){return;}
  let blockElems=cfg.z; let planeBytes=cfg.w; let blockBytes=blockElems;   // uint8: 1 byte/elem
  let nBlk=__NBLK__; let framePix=__FRAMEPIX__;
  let e0=grp*8u; let frm=e0/framePix; let inFrame=e0%framePix;
  let blk=inFrame/blockElems; let groupByte=(inFrame%blockElems)>>3u;
  let pb=(frm*nBlk+blk)*blockBytes + groupByte;
  var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
  for(var b:u32=0u;b<8u;b=b+1u){
    let byte=byteAt(pb+b*planeBytes); let bit=1u<<b;
    if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
    if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
    if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
    if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
  }
  let o=grp*2u;
  stack[o]=v0|(v1<<8u)|(v2<<16u)|(v3<<24u); stack[o+1u]=v4|(v5<<8u)|(v6<<16u)|(v7<<24u);
}`;

const MAX_WG = 65535;

export interface Bslz4Spec {
  compressed: Uint8Array;        // concatenated per-frame bslz4 chunks (frame-padded to 4B)
  blockMeta: Uint32Array;        // [coff,clen] per (frame,block), absolute byte offsets
  nFrames: number;
  nBlocksPerFrame: number;
  blockElems: number;            // elements per bitshuffle block (e.g. 4096 for uint16/8192B)
  detSize: number;               // detector pixels per frame (e.g. 192*192)
}

// Decode a bslz4 stack to a packed GPU buffer ([scanPos][detPixel]). dtype "uint8"
// (clip 0-255, 4 px/u32, offline default - half the memory) or "uint16" (lossless,
// 2 px/u32). Layout matches Show4DSTEMCompute.sample() for that mode exactly.
// Returns null if WebGPU is unavailable. Throws (validation) only on misuse.
export async function decodeBslz4ToStack(spec: Bslz4Spec, dtype: "uint8" | "uint16" = "uint8", srcDtype: "uint8" | "uint16" = "uint16"): Promise<{ device: GPUDevice; buffer: GPUBuffer; mode: number } | null> {
  const device = await getGPUDevice();
  if (!device) return null;
  const { compressed, blockMeta, nFrames, nBlocksPerFrame, blockElems, detSize } = spec;
  const srcBytes = srcDtype === "uint8" ? 1 : 2;   // companion encoded from uint8 (8 planes) or uint16 (16)
  const blockBytes = blockElems * srcBytes;        // bitshuffled block bytes
  const planeBytes = blockElems / 8;
  const totalBlocks = nFrames * nBlocksPerFrame;
  const totalElems = nFrames * detSize;
  const u8 = dtype === "uint8" || srcDtype === "uint8";   // uint8 source always outputs uint8
  const stackWords = u8 ? Math.ceil(totalElems / 4) : totalElems / 2;  // packed output u32 count
  const interBytes = totalBlocks * blockBytes;

  // writeBuffer requires a multiple-of-4 size; pad the compressed bytes if the
  // companion isn't 4-aligned (robust to any chunk file).
  const rawPad = compressed.byteLength % 4 === 0 ? compressed
    : (() => { const p = new Uint8Array(Math.ceil(compressed.byteLength / 4) * 4); p.set(compressed); return p; })();
  const rawBuf = device.createBuffer({ size: rawPad.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(rawBuf, 0, rawPad.buffer as ArrayBuffer, rawPad.byteOffset, rawPad.byteLength);
  const interBuf = device.createBuffer({ size: interBytes, usage: GPUBufferUsage.STORAGE });
  const metaBuf = device.createBuffer({ size: blockMeta.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(metaBuf, 0, blockMeta.buffer as ArrayBuffer, blockMeta.byteOffset, blockMeta.byteLength);
  const stack = device.createBuffer({ size: stackWords * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

  const cfg1 = uniform(device, [totalBlocks, blockBytes, 0, 0]);
  const nGroups = totalElems / 8;   // pass2: one thread per 8-pixel group
  const p2wg = Math.ceil(nGroups / 64), gx = Math.min(p2wg, MAX_WG), gy = Math.ceil(p2wg / MAX_WG);
  const cfg2 = uniform(device, [nGroups, gx * 64, blockElems, planeBytes]);

  // uint8 source -> 8-plane fast path (output uint8); else uint16 source -> uint8(clip) or uint16.
  const pass2tpl = srcDtype === "uint8" ? PASS2_U8SRC_WGSL : (u8 ? PASS2_U8_WGSL : PASS2_WGSL);
  const pass2 = pass2tpl.replace("__NBLK__", `${nBlocksPerFrame}u`).replace("__FRAMEPIX__", `${detSize}u`);
  const p1 = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: PASS1_WGSL }), entryPoint: "main" } });
  const p2 = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: pass2 }), entryPoint: "main" } });
  const bg1 = device.createBindGroup({ layout: p1.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: rawBuf } }, { binding: 1, resource: { buffer: interBuf } },
    { binding: 2, resource: { buffer: metaBuf } }, { binding: 3, resource: { buffer: cfg1 } } ] });
  const bg2 = device.createBindGroup({ layout: p2.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: interBuf } }, { binding: 1, resource: { buffer: stack } },
    { binding: 2, resource: { buffer: cfg2 } } ] });

  const enc = device.createCommandEncoder();
  let pa = enc.beginComputePass(); pa.setPipeline(p1); pa.setBindGroup(0, bg1); pa.dispatchWorkgroups(Math.ceil(totalBlocks / 64)); pa.end();
  let pb = enc.beginComputePass(); pb.setPipeline(p2); pb.setBindGroup(0, bg2); pb.dispatchWorkgroups(gx, gy); pb.end();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  rawBuf.destroy(); interBuf.destroy(); metaBuf.destroy(); cfg1.destroy(); cfg2.destroy();
  return { device, buffer: stack, mode: u8 ? 1 : 0 };
}

function uniform(device: GPUDevice, vals: number[]): GPUBuffer {
  const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(b, 0, new Uint32Array(vals).buffer);
  return b;
}
