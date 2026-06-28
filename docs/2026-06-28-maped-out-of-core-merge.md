# MAPED out-of-core merge for small-VRAM GPUs

2026-06-28

## Problem

A no-bin 7-tilt MAPED series is ~135 GB. The merge accumulates a float32 output
(`num`, 38.6 GB at no-bin) while streaming one uint16 tilt (19.3 GB) through the
GPU at a time. Accumulator + one tilt is ~58 GB: that fits a 96 GB card, but not
a 24 GB one. We want a microscopist with a small GPU (24 GB) plus a big CPU RAM
to run the full no-bin merge anyway.

## Three audiences

| | Hardware | Merge strategy |
|---|---|---|
| A1 MPS | Mac, big unified memory | merge on MPS; unified memory holds it |
| A2 big VRAM | RTX 6000 (96 GB) | full merge on GPU, per-tilt stream |
| A3 small VRAM + big CPU RAM | 24 GB GPU + 64+ GB RAM | out-of-core: accumulator in CPU RAM, one tilt streamed through the GPU |

Pure CPU (no GPU at all) is explicitly NOT a target. Even A3 loads each tilt with
the GPU LZ4 decompress (~1.0 s/tilt); the CPU never decompresses.

## Out-of-core design (A3)

`merge_datasets(accumulator_device="cpu")` keeps the 38.6 GB `num` accumulator in
CPU RAM. Per tilt: load (GPU decompress) -> stream scan-row batches through the
GPU (warp + sub-pixel shift) -> copy each batch's weighted product to CPU and add
into `num`. The GPU only ever holds one tilt (19.3 GB) plus the batch buffers,
never the accumulator. The divide (num/den) runs band-by-band on the CPU; `den`
is kept factorized (tiny per-tilt weight maps), never a second full accumulator.

### Auto-trigger

`merge_datasets` reads free VRAM and, when `accumulator + one tilt` will not fit,
auto-switches the accumulator to CPU RAM with no user flag. A 96 GB card keeps the
fast in-VRAM path; a 24 GB card auto-goes out-of-core. The same
`cuda_maped_merge` notebook therefore serves both A2 and A3. Override with an
explicit `accumulator_device=` ('cpu' or a second GPU).

## Numbers (2 tilts, full-res 512x512x192x192, RTX 6000, det no-bin)

| | merge time | vs A2 |
|---|---|---|
| A2 in-VRAM | 4.2 s | 1.0x |
| A3 out-of-core, pageable accumulator | 40.8 s | 9.81x |
| A3 out-of-core, pinned staging | 19.9 s | 4.79x |

Load (GPU decompress) is ~1.0 s/tilt on every path.

### Where the time went (torch.profiler, pageable accumulator)

| op | time | share |
|---|---|---|
| cudaMemcpyAsync / copy_ (GPU -> CPU) | ~30 s | ~73% |
| add_ (CPU accumulate, num += batch) | 4.9 s | 12% |
| warp + sub-pixel shift (GPU) | ~4 s | 10% |
| div_ + einsum + mean (CPU) | ~4 s | 8% |

The dominant cost was the GPU -> CPU transfer, and at ~2.7 GB/s it was running at
the pageable-memory rate, far below PCIe 4.0's ~20 GB/s ceiling. It was NOT the
CPU compute (a common wrong guess).

### The pinned-staging fix

The accumulator was a plain `torch.zeros(device="cpu")` = pageable. Copying each
batch's weighted product into a reused page-locked (`pin_memory=True`) staging
buffer first hits PCIe peak: 40.8 s -> 19.9 s (9.81x -> 4.79x). It is the same
copy then the same add, so the result stays bit-exact and the parity test still
passes. The cross-GPU split (accumulator on a second GPU) keeps its already-fast
peer `.to()` path; only the CPU accumulator uses the staging buffer.

## Parity

`tests/diffraction/test_maped_merge_parity.py::test_out_of_core_cpu_accumulator_matches_in_vram`
asserts the out-of-core merge is bit-for-bit identical (`torch.equal`) to the
in-VRAM merge on real Samsung tilts, via the production `from_files` (uint16,
bilinear) path. Run it with `--runslow` and
`MAPED_TEST_DIR=<tilt-series-dir> MAPED_TEST_PREFIX=<master-prefix>`.

Note: the cupy float32 `from_datasets` baseline path with `shift_method="fourier"`
is NOT bit-exact across accumulator devices (the GPU fused `addcmul_` vs the CPU
separate multiply-then-add diverge by ~1 ULP on that data); the production
`from_files`/bilinear path is bit-exact. The test uses the production path.

## Rejected / future

- Tilt-on-CPU plus chunk (for <19 GB cards): unnecessary at 24 GB, since one tilt
  fits the GPU at 19 GB. Deferred until a sub-19 GB card is a real target.
- Async overlap (`non_blocking=True` plus pipelining the copy behind the next
  batch's GPU compute): would hide the remaining ~4-8 s transfer and take 4.79x
  toward ~3x. Not done yet.
- Divide on the GPU (stream num chunks back for num/den): removes the ~1.5 s CPU
  divide; minor next to the transfer.

## Evidence

- `e79213a5` feat(maped): auto out-of-core merge for small-VRAM GPUs
- `741ec940` perf(maped): pin CPU accumulator staging, out-of-core merge 9.8x -> 4.8x
- `071d4bf1` test(maped): out-of-core merge bit-exact vs in-VRAM on real data
