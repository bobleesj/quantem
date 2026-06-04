# WebGPU 4D-STEM load optimization: 16.8s -> 1.3-4.4s, and the uint32 parity bug (2026-06-03)

## Question

The standalone WebGPU browser app (`widget/web/`) loaded a full Arina dataset far slower
than Python's `quantem.widget.load` (~2s). How fast can a native `.h5` 4D-STEM dataset go
from a picked folder to a rendered virtual image, all in-browser, and where are the floors?

## Setup

- App: `widget/web/` (quantem.live Browse GUI + the `js/engine/` WGSL engine).
- Real data, multiple users + dtypes: gold04 (3 GB, **uint16**), gold06 (6.6 GB, **uint32**),
  karen SiN (3.9 GB, uint32), wmill dggg (5.5 GB), steph lamella (801 MB), george gold_10
  (743 MB). Driven over CDP via `DOM.setFileInputFiles` (the picker's exact `File.arrayBuffer`
  disk path) on headed Chrome + NVIDIA Vulkan (Blackwell, NOT SwiftShader - asserted).
- Parity reference: h5py mean DP (clip uint8, integer sum, pixel_mask bad px zeroed).

## The critical bug: uint32 datasets decoded 16x wrong

`h5reader` detected only uint8/uint16; uint32 detector data (`<u4`, common for high dynamic
range) fell through to uint16. The decode then read 16 bit-planes from a 32-plane bitshuffle
-> every value ~16x off. The IMAGE looked right (relative contrast preserved) so it passed
vision checks; only a numerical mean-DP-vs-h5py check caught it. **Lesson: vision is not
parity. Always check a numerical reference on real data of every dtype.**

Fix: detect `<u1`/`<u2`/`<u4` -> 8/16/32 planes (`srcBytes` 1/2/4 -> `blockElems`,
`nBlocksPerFrame`, `blockMeta` all correct per dtype), threaded `srcDtype` store -> kernel
(templated `__NBITS__` in the fused kernel). Verified bit-exact after:
- gold04 uint16: browser 147010.54 == h5py 147010.55
- gold06 uint32: browser 2529764.38 == h5py 2529764.50 (diff = float32 summation rounding)

## Results (per-stage, measured)

| Stage | Before | After | How |
|---|---|---|---|
| File read | 3.0s (2.3 GB/s, main thread) | **0.7s (10 GB/s)** | **8 Web Workers** each call File.arrayBuffer |
| jsfive parse | 2.8s | **0.25s** | custom DataView HDF5 v1 B-tree reader (no per-node alloc) |
| GPU upload | mappedAtCreation 4s | staging pool + Float64 wide copy | reuse MAP_WRITE buffers (kills 1.7s alloc/zero); f64 copy = 20 GB/s |
| GPU kernel | "1.9s" (mismeasured) | **459ms** (timestamp) | fused shared-mem decode (no interBuf); the 1.9s was upload-DMA |
| bitshuffle | 16/32-plane transpose | OR-fold | uint8 output: low-8 transpose + OR high planes (2x uint16 / 4x uint32 less) |

End-to-end (picker disk path, parity-verified):

| Dataset | Size | Load |
|---|---|---|
| george gold_10 | 743 MB | 1.3s |
| steph lamella | 801 MB | 1.5s |
| karen SiN (uint32) | 3.9 GB | 3.3s |
| wmill dggg | 5.5 GB | 3.7s |
| gold06 (uint32) | 6.6 GB | 4.3s |

**Sub-GB hits the 1-2s Python sweet spot.** Big datasets ~1s/GB.

## The breakthrough: File.arrayBuffer is main-thread-bound, not bandwidth-bound

Main-thread `File.arrayBuffer` (sequential or `Promise.all`) caps at ~2.3 GB/s. Measured
alternatives:
- `Blob.slice()` 8x parallel ranges: **2.0 GB/s** (slower - more overhead)
- `File.stream()`: 2.1 GB/s (slower)
- OPFS `createSyncAccessHandle` (worker): 3.5 GB/s (and needs an OPFS copy first)
- **8 Web Workers each `File.arrayBuffer` + transfer buffer back: 10.1 GB/s** <- 4.4x

So the 2.3 GB/s was the single main thread (file delivery + ArrayBuffer allocation), not the
disk or an IPC pipe. A worker pool parallelizes it to near disk speed. `readWorker.ts` reads
+ runs the fast btree parse, transfers `{buffer, blockMeta}` back zero-copy.

## Floors (what's left)

- File read: ~10 GB/s with workers (near Python's C disk read).
- GPU kernel: 459ms (uint16), decode-bound on the **serial LZ4** (the OR-fold proved the
  bitshuffle is not the bottleneck). Memory-traffic floor is ~11ms (16 GB / 1.47 TB/s), so
  ~40x compute headroom remains in the serial decode.
- WebGPU upload: spec mandates a staging copy on discrete GPUs (gpuweb#2388); cudaMemcpy's
  64 GB/s is unreachable, realistic ~5-10 GB/s.

For big datasets the wall is now the **decode** (serial LZ4 + upload of the GBs). The path to
a ~100ms kernel is the silx warp-cooperative decode (parallel literal/match copies), but
WGSL's u32-only shared memory forces `atomic<u32>` for the byte-strided parallel writes - a
complex, parity-sensitive rewrite, not yet done.

## Rejected / dead ends

- **Worker PARSE with buffer transferred IN**: regressed to 7.6s - transferring 7.5 GB to
  workers + reading all upfront cost more than the parallel parse saved. The win is workers
  reading the File THEMSELVES (only the result transfers back).
- **All-28-reads upfront** (no grouping): 7.4s - memory pressure, lost the decode pipeline.
- **mappedAtCreation per file**: the alloc+zero of host-visible memory is ~3.9 GB/s (1.7s for
  6.6 GB) and scales with bytes; a reused staging pool amortizes it.
- **OR-fold high-plane STORAGE skip** (4096-byte shared): risks parity (LZ4 matches may cross
  the low/high plane boundary); only the high-plane bitshuffle is OR-folded, never the decode.

## Files

`js/engine/h5reader.ts` (fast btree + uint32 detect + zero-copy), `js/engine/bslz4.ts` (fused
kernel + `__NBITS__` + staging pool + wide copy + OR-fold), `js/engine/compute.ts`
(`fromGpuChunks`), `web/src/local/{store,readWorker}.ts` (worker read pool + srcDtype + LRU),
`web/src/App.tsx` (File/handle source for workers).
