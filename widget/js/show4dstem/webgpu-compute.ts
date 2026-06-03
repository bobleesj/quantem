/// <reference types="@webgpu/types" />
// Offline 4D-STEM compute: run the virtual-image and DP-from-ROI reductions in
// the browser via WebGPU, so a small dataset stays fully interactive with no
// Python kernel. This is the browser sibling of the Python compute backends
// (TorchCompute / MetalCompute / CudaKernelCompute) - same two primitives:
//
//   maskedSum(detectorMask)  -> virtual image  (one thread per scan position)
//   reduceFrames(scanMask)   -> diffraction DP  (one thread per detector pixel)
//
// Detector counts are integers, so the stack ships as uint16 and the sum
// accumulates in u32 -> the virtual image is BIT-EXACT to the torch/Metal
// result (verified against a CPU reference, maxErr=0). Only the optional mean
// divide is float32, matching the kernel backends.
import { getGPUDevice } from "./fft";

// One thread per scan position: sum the detector pixels under the mask. The
// stack is packed 2 uint16 per u32 (little-endian), unpacked by global index.
const MASKED_SUM_WGSL = `
@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read> mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;            // scanCount, detSize
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = gid.x;
  if (s >= dims.x) { return; }
  let detSize = dims.y;
  let base = s * detSize;
  var sum: u32 = 0u;
  for (var k: u32 = 0u; k < detSize; k = k + 1u) {
    if (mask[k] != 0u) {
      let gp = base + k;
      let word = data[gp >> 1u];
      let val = select(word >> 16u, word & 0xffffu, (gp & 1u) == 0u);
      sum = sum + val;
    }
  }
  out[s] = f32(sum);
}`;

// One thread per detector pixel: sum (or mean) over the selected scan positions.
const REDUCE_FRAMES_WGSL = `
@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read> scanMask: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;            // scanCount, detSize, mean
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  let detSize = dims.y;
  if (k >= detSize) { return; }
  var sum: u32 = 0u;
  var cnt: u32 = 0u;
  for (var s: u32 = 0u; s < dims.x; s = s + 1u) {
    if (scanMask[s] != 0u) {
      let gp = s * detSize + k;
      let word = data[gp >> 1u];
      let val = select(word >> 16u, word & 0xffffu, (gp & 1u) == 0u);
      sum = sum + val;
      cnt = cnt + 1u;
    }
  }
  var res = f32(sum);
  if (dims.z == 1u && cnt > 0u) { res = res / f32(cnt); }
  out[k] = res;
}`;

export class Show4DSTEMCompute {
  private device: GPUDevice;
  private maskedSumPipe: GPUComputePipeline;
  private reduceFramesPipe: GPUComputePipeline;
  private dataBuf: GPUBuffer;
  readonly scanCount: number;
  readonly detSize: number;

  private constructor(device: GPUDevice, dataBuf: GPUBuffer, scanCount: number, detSize: number) {
    this.device = device;
    this.dataBuf = dataBuf;
    this.scanCount = scanCount;
    this.detSize = detSize;
    const maskedSum = device.createShaderModule({ code: MASKED_SUM_WGSL });
    const reduceFrames = device.createShaderModule({ code: REDUCE_FRAMES_WGSL });
    this.maskedSumPipe = device.createComputePipeline({ layout: "auto", compute: { module: maskedSum, entryPoint: "main" } });
    this.reduceFramesPipe = device.createComputePipeline({ layout: "auto", compute: { module: reduceFrames, entryPoint: "main" } });
  }

  // Upload the uint16 4D stack once. Returns null if WebGPU is unavailable so the
  // caller falls back to the Python kernel path. The stack is uploaded as packed
  // u32 (2 pixels per word); a trailing pad keeps the byte length 4-aligned.
  static async create(stack: Uint16Array, scanCount: number, detSize: number): Promise<Show4DSTEMCompute | null> {
    const device = await getGPUDevice();
    if (!device) return null;
    const total = scanCount * detSize;
    const padded = (total & 1) ? new Uint16Array(total + 1) : stack;
    if (padded !== stack) padded.set(stack);
    const dataBuf = device.createBuffer({ size: padded.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(dataBuf, 0, padded.buffer as ArrayBuffer, padded.byteOffset, padded.byteLength);
    return new Show4DSTEMCompute(device, dataBuf, scanCount, detSize);
  }

  // Virtual image: f32 array of length scanCount, one masked detector sum per
  // scan position. mask is detSize entries, nonzero = inside the detector.
  async maskedSum(mask: Uint32Array): Promise<Float32Array> {
    return this.run(this.maskedSumPipe, mask, this.scanCount, [this.scanCount, this.detSize, 0, 0]);
  }

  // Diffraction pattern over a real-space ROI: f32 array of length detSize.
  // scanMask is scanCount entries; mean=true divides by the count (else sum).
  async reduceFrames(scanMask: Uint32Array, mean = true): Promise<Float32Array> {
    return this.run(this.reduceFramesPipe, scanMask, this.detSize, [this.scanCount, this.detSize, mean ? 1 : 0, 0]);
  }

  private async run(pipe: GPUComputePipeline, mask: Uint32Array, outLen: number, dims: number[]): Promise<Float32Array> {
    const device = this.device;
    const maskBuf = device.createBuffer({ size: Math.max(16, mask.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(maskBuf, 0, mask.buffer as ArrayBuffer, mask.byteOffset, mask.byteLength);
    const dimsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const dimsArr = new Uint32Array(dims);
    device.queue.writeBuffer(dimsBuf, 0, dimsArr.buffer as ArrayBuffer, dimsArr.byteOffset, dimsArr.byteLength);
    const outBuf = device.createBuffer({ size: outLen * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const bind = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.dataBuf } },
      { binding: 1, resource: { buffer: maskBuf } },
      { binding: 2, resource: { buffer: outBuf } },
      { binding: 3, resource: { buffer: dimsBuf } },
    ] });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(outLen / 64));
    pass.end();
    const readBuf = device.createBuffer({ size: outLen * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, outLen * 4);
    device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();
    maskBuf.destroy(); dimsBuf.destroy(); outBuf.destroy(); readBuf.destroy();
    return result;
  }

  dispose() { this.dataBuf.destroy(); }
}
