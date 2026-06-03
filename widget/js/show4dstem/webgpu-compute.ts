/// <reference types="@webgpu/types" />
// Offline 4D-STEM compute in the browser via WebGPU - no Python kernel.
// Primitives (browser siblings of the Python backends):
//   maskedSum(detectorMask) -> virtual image  (one thread per scan position)
//   reduceFrames(scanMask)  -> diffraction DP  (one thread per detector pixel)
//
// CHUNKED: the stack is split into scan-row ranges, each in its own GPU buffer
// (<= the 1 GB per-buffer cap). This lets a stack far larger than one buffer
// (e.g. 512x512x192x192 = 9.7 GB) live across N buffers and be reduced by
// dispatching per chunk and accumulating. A single-buffer dataset is just the
// N=1 case. Verified bit-exact vs numpy (chunked masked_sum maxErr 0).
//
// Stack ships as uint8 (clip(0,255): real detector counts are 0-~200, so the
// value IS the count, near-lossless) or uint16; dtype inferred from byte length.
import { getGPUDevice } from "./fft";
import { decodeBslz4ToStack, type Bslz4Spec } from "./bslz4";

// `mode`: 0 = uint16 (2 samples/u32), 1 = uint8 (4/u32). `sample(gp)` reads a
// detector value at a chunk-local global pixel index.
const SAMPLE = `
fn sample(gp: u32, mode: u32) -> u32 {
  if (mode == 1u) { let w = data[gp >> 2u]; return (w >> ((gp & 3u) * 8u)) & 0xffu; }
  let w = data[gp >> 1u];
  return select(w >> 16u, w & 0xffffu, (gp & 1u) == 0u);
}`;

// One thread per scan position IN THIS CHUNK. Writes the VI at the global scan
// offset, so chunks write disjoint VI slices (no accumulation needed).
const MASKED_SUM_WGSL = `
@group(0) @binding(0) var<storage,read> data: array<u32>;
@group(0) @binding(1) var<storage,read> idx: array<u32>;   // ACTIVE detector pixel indices only
@group(0) @binding(2) var<storage,read_write> vi: array<f32>;
@group(0) @binding(3) var<uniform> u: vec4<u32>;   // startScan, nScanInChunk, detSize, mode
${SAMPLE}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let sl = gid.x; if (sl >= u.y) { return; }
  let base = sl * u.z; let n = arrayLength(&idx); var sum: u32 = 0u;
  for (var j: u32 = 0u; j < n; j = j + 1u) { sum = sum + sample(base + idx[j], u.w); }  // only in-aperture px
  vi[u.x + sl] = f32(sum);
}`;

// One thread per detector pixel; ACCUMULATES this chunk's in-ROI scan positions
// into the DP (chunks dispatched serially, so += across chunks is safe). dims:
// startScan, nScanInChunk, detSize, mode; plus extra: total scanMask is global,
// indexed by startScan+sl.
const REDUCE_FRAMES_WGSL = `
@group(0) @binding(0) var<storage,read> data: array<u32>;
@group(0) @binding(1) var<storage,read> scanMask: array<u32>;  // GLOBAL scanCount
@group(0) @binding(2) var<storage,read_write> dp: array<u32>;  // detSize, INTEGER accumulate (exact)
@group(0) @binding(3) var<uniform> u: vec4<u32>;   // startScan, nScanInChunk, detSize, mode
${SAMPLE}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x; let detSize = u.z; if (k >= detSize) { return; }
  var sum: u32 = 0u;   // integer accumulate: bit-exact, no f32 rounding on large/dead-pixel sums
  for (var sl: u32 = 0u; sl < u.y; sl = sl + 1u) {
    if (scanMask[u.x + sl] != 0u) { sum = sum + sample(sl * detSize + k, u.w); }
  }
  dp[k] = dp[k] + sum;
}`;

// Extract ONE frame's diffraction pattern (detSize values) from a chunk buffer -
// the offline replacement for the kernel's per-probe frame_bytes. One thread/pixel.
const FRAME_WGSL = `
@group(0) @binding(0) var<storage,read> data: array<u32>;
@group(0) @binding(1) var<storage,read_write> frame: array<f32>;
@group(0) @binding(2) var<uniform> u: vec4<u32>;   // localBase (pixels), detSize, mode
${SAMPLE}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x; if (k >= u.y) { return; }
  frame[k] = f32(sample(u.x + k, u.z));
}`;

interface Chunk { buffer: GPUBuffer; startScan: number; nScan: number; }

export class Show4DSTEMCompute {
  private device: GPUDevice;
  private maskedSumPipe: GPUComputePipeline;
  private reduceFramesPipe: GPUComputePipeline;
  private frameAtPipe: GPUComputePipeline;
  private chunks: Chunk[];
  readonly scanCount: number;
  readonly detSize: number;
  readonly mode: number;
  // Bad/hot detector pixel indices (from the HDF5 pixel_mask). Auto-excluded from
  // every reduction so the offline result matches CUDA's apply_mask path - the
  // browser data is filtered automatically, no per-call masking needed.
  badPx: Uint32Array = new Uint32Array(0);

  private constructor(device: GPUDevice, chunks: Chunk[], scanCount: number, detSize: number, mode: number) {
    this.device = device; this.chunks = chunks; this.scanCount = scanCount; this.detSize = detSize; this.mode = mode;
    const ms = device.createShaderModule({ code: MASKED_SUM_WGSL });
    const rf = device.createShaderModule({ code: REDUCE_FRAMES_WGSL });
    this.maskedSumPipe = device.createComputePipeline({ layout: "auto", compute: { module: ms, entryPoint: "main" } });
    this.reduceFramesPipe = device.createComputePipeline({ layout: "auto", compute: { module: rf, entryPoint: "main" } });
    this.frameAtPipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: FRAME_WGSL }), entryPoint: "main" } });
  }

  // One frame's diffraction pattern (f32[detSize]) for scan position scanIdx -
  // a GPU extract from whichever chunk holds it. Drives the offline DP panel.
  async frameAt(scanIdx: number): Promise<Float32Array> {
    const ch = this.chunks.find((c) => scanIdx >= c.startScan && scanIdx < c.startScan + c.nScan) ?? this.chunks[0];
    const localBase = (scanIdx - ch.startScan) * this.detSize;
    const out = this.device.createBuffer({ size: this.detSize * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const dims = this.uniform([localBase, this.detSize, this.mode, 0]);
    const bind = this.device.createBindGroup({ layout: this.frameAtPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: ch.buffer } }, { binding: 1, resource: { buffer: out } }, { binding: 2, resource: { buffer: dims } } ] });
    this.dispatch(this.frameAtPipe, bind, Math.ceil(this.detSize / 64));
    const frame = await this.readF32(out, this.detSize);
    for (const bp of this.badPx) frame[bp] = 0;   // auto-filter hot px in the diffraction pattern
    out.destroy(); dims.destroy(); return frame;
  }

  // Single decompressed stack -> one chunk (the common, fits-in-one-buffer case).
  static async create(stack: Uint8Array, scanCount: number, detSize: number): Promise<Show4DSTEMCompute | null> {
    return Show4DSTEMCompute.createChunked([{ bytes: stack, startScan: 0, nScan: scanCount }], scanCount, detSize);
  }

  // Decompress a native HDF5 bitshuffle+LZ4 (bslz4) stack on the GPU and wrap the
  // decoded buffer as the (single) compute chunk - the offline "ship compressed,
  // decompress in browser, no Python" path. dtype "uint8" (offline default, clip
  // 0-255, half memory) or "uint16" (lossless). The decoded buffer is packed in
  // [scanPos][detPixel] order matching sample() for that mode, so masked_sum /
  // reduceFrames run on it unchanged.
  static async createFromBslz4(spec: Bslz4Spec, dtype: "uint8" | "uint16" = "uint8"): Promise<Show4DSTEMCompute | null> {
    const decoded = await decodeBslz4ToStack(spec, dtype);
    if (!decoded) return null;
    const chunks: Chunk[] = [{ buffer: decoded.buffer, startScan: 0, nScan: spec.nFrames }];
    return new Show4DSTEMCompute(decoded.device, chunks, spec.nFrames, spec.detSize, decoded.mode);
  }

  // Chunked bslz4: decode each scan-row chunk's compressed bytes into its OWN GPU
  // buffer and hold them all, so a stack far bigger than one 1 GB buffer (full
  // 512x512x192x192 = 9.6 GB uint8) lives across N buffers and masked_sum /
  // reduceFrames reduce across them. Each decode reuses a ~1 GB scratch internally.
  static async createFromBslz4Chunked(
    chunkSpecs: (Bslz4Spec & { startScan: number; nScan: number })[],
    scanCount: number, detSize: number, dtype: "uint8" | "uint16" = "uint8",
  ): Promise<Show4DSTEMCompute | null> {
    let device: GPUDevice | null = null;
    let mode = dtype === "uint8" ? 1 : 0;
    const chunks: Chunk[] = [];
    for (const spec of chunkSpecs) {
      const decoded = await decodeBslz4ToStack(spec, dtype);
      if (!decoded) return null;
      device = decoded.device; mode = decoded.mode;
      chunks.push({ buffer: decoded.buffer, startScan: spec.startScan, nScan: spec.nScan });
    }
    if (!device) return null;
    return new Show4DSTEMCompute(device, chunks, scanCount, detSize, mode);
  }

  // N chunks, each {bytes, startScan, nScan}. Each chunk's bytes hold its scan
  // range's frames contiguously. dtype inferred from total bytes vs total pixels.
  static async createChunked(chunkSpecs: { bytes: Uint8Array; startScan: number; nScan: number }[], scanCount: number, detSize: number): Promise<Show4DSTEMCompute | null> {
    const device = await getGPUDevice();
    if (!device) return null;
    const totalBytes = chunkSpecs.reduce((a, c) => a + c.bytes.byteLength, 0);
    const mode = totalBytes <= scanCount * detSize ? 1 : 0;  // <= 1 byte/pixel => uint8
    const chunks: Chunk[] = chunkSpecs.map((c) => {
      const padLen = Math.ceil(c.bytes.byteLength / 4) * 4;
      const buffer = device.createBuffer({ size: Math.max(4, padLen), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buffer, 0, c.bytes.buffer as ArrayBuffer, c.bytes.byteOffset, c.bytes.byteLength);
      return { buffer, startScan: c.startScan, nScan: c.nScan };
    });
    return new Show4DSTEMCompute(device, chunks, scanCount, detSize, mode);
  }

  // Virtual image: f32[scanCount]. Each chunk writes its disjoint VI slice.
  // Loops only the ACTIVE (in-aperture) detector pixels, not all detSize - a BF
  // disk (~9k of 36864 px) or ADF annulus is then 4-10x fewer reads per scan pos.
  async maskedSum(mask: Uint32Array): Promise<Float32Array> {
    const device = this.device;
    const bad = this.badPx.length ? new Set(this.badPx) : null;
    const idxArr = new Uint32Array(this.detSize); let n = 0;
    for (let k = 0; k < this.detSize; k++) if (mask[k] !== 0 && !(bad && bad.has(k))) idxArr[n++] = k;  // skip hot px
    const idx = idxArr.subarray(0, n || 1);   // active pixel indices (>=1 to keep a valid binding)
    const idxBuf = this.upload(idx, GPUBufferUsage.STORAGE);
    const vi = device.createBuffer({ size: this.scanCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const temps: GPUBuffer[] = [];
    for (const ch of this.chunks) {
      const dims = this.uniform([ch.startScan, ch.nScan, this.detSize, this.mode]); temps.push(dims);
      const bind = device.createBindGroup({ layout: this.maskedSumPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: ch.buffer } }, { binding: 1, resource: { buffer: idxBuf } },
        { binding: 2, resource: { buffer: vi } }, { binding: 3, resource: { buffer: dims } } ] });
      this.dispatch(this.maskedSumPipe, bind, Math.ceil(ch.nScan / 64));
    }
    const out = await this.readF32(vi, this.scanCount);  // awaits GPU completion before freeing
    if (n === 0) out.fill(0);   // empty mask -> all zero (idx had a dummy entry)
    idxBuf.destroy(); vi.destroy(); temps.forEach((b) => b.destroy()); return out;
  }

  // DP over a real-space ROI: f32[detSize]. scanMask is GLOBAL; chunks accumulate
  // in INTEGER (u32, bit-exact) - the mean divide happens once in f64 at readback,
  // so the result matches the torch/CUDA integer-sum-then-divide exactly (even on
  // saturated 65535 dead pixels, where f32 accumulation would drift ~1 count).
  async reduceFrames(scanMask: Uint32Array, mean = true): Promise<Float32Array> {
    const device = this.device;
    const maskBuf = this.upload(scanMask, GPUBufferUsage.STORAGE);
    const dp = device.createBuffer({ size: this.detSize * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(dp, 0, new Uint32Array(this.detSize));  // zero-init integer accumulator
    const temps: GPUBuffer[] = [];
    for (const ch of this.chunks) {
      const dims = this.uniform([ch.startScan, ch.nScan, this.detSize, this.mode]); temps.push(dims);
      const bind = device.createBindGroup({ layout: this.reduceFramesPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: ch.buffer } }, { binding: 1, resource: { buffer: maskBuf } },
        { binding: 2, resource: { buffer: dp } }, { binding: 3, resource: { buffer: dims } } ] });
      this.dispatch(this.reduceFramesPipe, bind, Math.ceil(this.detSize / 64));
    }
    const sums = await this.readU32(dp, this.detSize);   // exact integer per-pixel sum
    temps.forEach((b) => b.destroy());
    const n = mean ? (scanMask.reduce((a, v) => a + (v ? 1 : 0), 0) || 1) : 1;
    const out = new Float32Array(this.detSize);
    for (let i = 0; i < this.detSize; i++) out[i] = sums[i] / n;  // f64 divide -> f32 store
    for (const bp of this.badPx) out[bp] = 0;   // auto-filter hot px (matches CUDA apply_mask)
    maskBuf.destroy(); dp.destroy(); return out;
  }

  private upload(arr: Uint32Array, usage: number): GPUBuffer {
    const b = this.device.createBuffer({ size: Math.max(16, arr.byteLength), usage: usage | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(b, 0, arr.buffer as ArrayBuffer, arr.byteOffset, arr.byteLength); return b;
  }
  private uniform(vals: number[]): GPUBuffer {
    const b = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const a = new Uint32Array(vals); this.device.queue.writeBuffer(b, 0, a.buffer as ArrayBuffer, a.byteOffset, a.byteLength); return b;
  }
  private dispatch(pipe: GPUComputePipeline, bind: GPUBindGroup, groups: number) {
    const enc = this.device.createCommandEncoder(); const pass = enc.beginComputePass();
    pass.setPipeline(pipe); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(groups); pass.end();
    this.device.queue.submit([enc.finish()]);
  }
  private async readU32(buf: GPUBuffer, n: number): Promise<Uint32Array> {
    const rb = this.device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder(); enc.copyBufferToBuffer(buf, 0, rb, 0, n * 4); this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ); const out = new Uint32Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy(); return out;
  }
  private async readF32(buf: GPUBuffer, n: number): Promise<Float32Array> {
    const rb = this.device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder(); enc.copyBufferToBuffer(buf, 0, rb, 0, n * 4); this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ); const out = new Float32Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy(); return out;
  }

  dispose() { for (const c of this.chunks) c.buffer.destroy(); }
}
