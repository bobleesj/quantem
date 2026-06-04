/// <reference types="@webgpu/types" />
// Shared WebGPU device for the 4D-STEM compute engine. ONE source of the GPU
// device + adapter limits, imported by every consumer (Show4DSTEM widget, the
// offline browser app, FFT). Keeps no UI deps so the engine is framework-agnostic.

let gpuDevice: GPUDevice | null = null;
let gpuInfo = "GPU";
const lostCallbacks: Array<() => void> = [];

// Register a reset to run when the GPU device is lost (process crash, tab suspend).
// Consumers (e.g. the FFT cache) use this to drop their device-bound state.
export function onGPULost(cb: () => void): void { lostCallbacks.push(cb); }

export async function getGPUDevice(): Promise<GPUDevice | null> {
  if (gpuDevice) return gpuDevice;
  if (!navigator.gpu) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return null;
    try {
      // Newer Chrome exposes the sync `adapter.info`; older builds used the async
      // requestAdapterInfo(). Prefer the sync one, fall back to async.
      // @ts-ignore - info / requestAdapterInfo are not in all type definitions
      const info = adapter.info || (await adapter.requestAdapterInfo?.());
      if (info) {
        gpuInfo = info.description || `${info.vendor || ""} ${info.architecture || ""} ${info.device || ""}`.trim() || "Generic WebGPU Adapter";
      }
    } catch (_e) { /* adapter info not available */ }
    // Raise device limits to the adapter max. Defaults are conservative
    // (maxStorageBufferBindingSize 128 MB, maxTextureDimension2D 8192); without
    // this, buffers > 128 MB silently invalidate bind groups and wide panels fail.
    const requiredLimits: Record<string, number> = {};
    for (const key of ["maxBufferSize", "maxStorageBufferBindingSize", "maxTextureDimension2D"] as const) {
      const v = adapter.limits[key] || 0;
      if (v > 0) requiredLimits[key] = v;
    }
    gpuDevice = await adapter.requestDevice({ requiredFeatures: [], requiredLimits });
    gpuDevice.lost.then(() => { gpuDevice = null; lostCallbacks.forEach((cb) => cb()); });
    return gpuDevice;
  } catch { return null; }
}

export function getGPUInfo(): string { return gpuInfo; }
