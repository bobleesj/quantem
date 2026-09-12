---
name: maped-demo
description: Run and verify seven-tilt MAPED with encoded GPU inputs, bounded float32 merging, scaled-uint16 storage, and packed viewing on CUDA, Torch MPS, or native Metal.
---

# MAPED demo

Keep code, tutorials, notebooks, qualifications, and this skill in the QuantEM
repository. The active feature branch is `maped-cuda-mps-ans`; QuantEM.GPU
infrastructure tracks its `main`. Use the selected current checkouts, not a
stale installed wheel. Public representation spelling is **`encoded`**, not
`ans`. ANS describes the underlying lossless count codec.

## Existing workflow and ownership

Use `MAPEDTorch.from_files(files, device="mps")` on Phil or `device="cuda:0"`
on mjgoat physical GPU0. The normal sequence is `preprocess`,
`diffraction_origin`, `diffraction_align`, `real_space_align`, and
`merge_datasets(dtype="scaled_uint16", plot_result=False)`. The result stays packed;
`maped.show()` provides Show4DSTEM when a viewer is requested. Do not add custom
readers, binning, or a second workflow API to the ordinary seven-tilt demo.

MAPED's stage order, alignment, weights, interpolation, and combination law
belong in QuantEM. Python MAPED uses Torch, with no CuPy, CUDA/Metal kernels,
or native allocator management. QuantEM.GPU supplies generic encoded residency,
bounded reads, reductions, precision conversion, and file writing. Never move
the MAPED algorithm into an IO helper or a GPU-library MAPED executor.

All scientific array operations stay on the selected GPU. CPU file IO,
compressed-byte handling, scalar metadata, and independent small NumPy test
oracles are allowed. Do not enable MPS CPU fallback.

Verify all seven source representations are `encoded`, each
`source_read_passes` is one, and each hot-pixel correction method is `median`
and applied. The loader corrects stored-mask pixels by their local integer
3x3 median, excluding invalid neighbors, before encoding. Raw files are
unchanged. MAPED BF is the arithmetic mean over the complete detector, including
corrected pixels, with the full detector pixel count as divisor. Saved
`maped_summary` metadata must preserve that definition and correction provenance.

Retain all seven inputs through both merge passes. Generate at most 4096 frames
per region for the qualified geometry. First measure the complete merged range;
then recompute float32 regions, convert with one global scale, and save. Never
allocate the whole 36 GiB float32 output. Release owned inputs before reopening
the full packed result; preserve caller-owned inputs from `from_resident`.

## Qualification and timing

Hardware: **Rodman is an Apple M5 Mac (Mac17,2) with 24 GiB unified memory
and Torch MPS support**, not an NVIDIA CUDA host. Use Rodman for physical
24 GiB Mac qualification. Phil is an Apple M5 Max with 128 GiB; a memory cap
on Phil does not replace a physical Rodman run. Verify hardware and available
memory before each qualification, and record system swap before and after.
Rodman's local seven-tilt acquisition differs from Phil's reference data;
do not compare their timings or error measurements as a paired experiment.

Read [the current Torch MPS inspection and performance qualification](../../mps-maped-inspection-performance.md)
for commands, exact timing boundaries, and evidence. On physical Phil M5 Max,
the complete seven-tilt workflow now measured **44.19-47.53 s**, versus a fresh
**75.44 s** control. A paired complete float32 merge measured **19.93 -> 10.02 s**.
All 9.66 billion float32 values and all 262,144 saved DP chunks were exact.
GPU allocations peaked below **13.8 GiB**, process footprint below **14.8 GiB**,
with a 24 GiB allocation cap on a 128 GB machine. This is not a physical 24 GB
laptop run or a promised latency. Input residency is 6.998 GiB, output 5.679 GiB.
The [earlier qualification](../../mps-maped-merge-performance.md) retains the
77-90 second implementation as historical evidence.

QuantEM.GPU now decodes resident count regions directly into Torch-owned MPS
storage. The native queue waits before returning an independently owned tensor;
never reintroduce a decoded-count copy through a NumPy view or release a
Torch-owned buffer manually. MAPED fuses its four ordered scan-interpolation
taps with Torch compilation for large interior regions. Keep shift indices and
weights as tensor inputs, and use fixed shapes within the pass: dynamic-shape
compilation was substantially slower. Preserve eager edge handling and the
`compile_merge=False` control. The compiler needs one exact float32 cast from
native counts; do not reinterpret signed integers or loosen the parity gate.
GPU IO uses one MPS min/max reduction to measure the range, validated against
nonfinite and strided inputs, with the established bounded writer queue.

## Current CUDA processing qualification

Read [CUDA processing before export](../../cuda-maped-processing-performance.md).
The pure-Torch first-tap initialization change removes a full zero-fill and
readback. Two paired complete CUDA merges improved 6.27 s to 5.79–5.83 s,
with all 9.66 billion float32 values exact in each audit. Complete merge,
BF/mean-DP summaries, and global range took 5.72–5.92 s after preparation
(1.66–1.68 s). Loading measured 8.46–13.14 s, not the desired 3–4 s.
Peak process memory was at most 13.21 GiB including the paired audit.

CUDA keeps eager interpolation: the MPS compiled form failed exact CUDA
parity and was rejected. Smaller 2048-frame regions were slower. Preserve
4096-frame automatic scheduling and the zero-copy DLPack read already used
on CUDA. This update changes no QuantEM.GPU API or implementation. Keep
complete overview benchmark timing separate from full scaled-uint16 viewing
and from actual browser rendering. The notebook includes exact float32 region
inspection before optional full export.

## Inspect before saving

After alignment, use the existing merge method with a selected scan region:

```python
patch = maped.merge_datasets(
    scan_region=(252, 260, 252, 260), plot_result=False,
)
maped.show()
# Keep the same seven resident inputs for another patch or full export.
merged = maped.merge_datasets(save_to="merged_master.h5", plot_result=False)
```

Regions use full aligned-scan coordinates and exclusive stops. The patch retains
all detector pixels and float32 precision. No file, global scaling, or reopening
is needed. A bounded inspection contains at most 4096 scan positions. It does
not certify the uninspected scan area. The public result records `scan_region`
in its merge metadata, and all owned inputs remain alive until full saving or
`maped.close()`. Existing save calls and scientific parameters are unchanged.
The final measured 8x8 patch took 0.085 s after alignment, with 0.126 s for
Show4DSTEM construction and 16.74 s loading through the first patch. Quote
construction separately from actual browser interaction. A full BF/mean-DP
one-pass overview is currently a benchmark experiment, not another public API.

Distinguish **Torch MPS** from **native Swift/Metal**. Native Metal now completes
loading through saved scaled-uint16 packed GPU
reopening in **22.56–22.87 s** on Phil, with **9.73 GiB peak Metal allocation**.
See [native processing performance](../../native-maped-processing-performance.md);
the earlier 67.56 s run is historical. The 75 parameter cases and 45 sensitivity
intervals are described in
[native parameter qualification](../../native-maped-parameters.md). Do not use
old 37-second MPS measurements from earlier algorithm ownership arrangements as
the timing of the current Torch implementation. Windows remains unqualified.

Before claiming parity or speed, preserve frozen fixtures and compare summaries,
origins, shifts, float32 merge values, saved/restored values, precision metrics,
ownership, and full workflow time. Never loosen numerical gates to obtain a
speedup. Distinguish float32 execution error from approximate uint16 storage
error. Report loading, merging, saving/reopening, and viewer construction
separately when those stages are requested. A CUDA regression test is not a
fresh CUDA performance measurement.

The local notebook belongs under
`notebooks/maped/cuda_maped_merge (1).ipynb`; the interactive tutorial is
`docs/tutorials/maped_interactive.html`. Keep private data paths and dataset
identities out of tracked artifacts. For native port changes, apply the adjacent
`port-torch-scientific-workflows` skill. Check GPU ownership before compute and
release only resources belonging to this run when finished.

For reusable execution lessons, use the adjacent
[Torch resident optimization skill](../torch-resident-optimization/SKILL.md).

Native optimization keeps prepared sampling geometry on the GPU, reuses bounded
decode/encode/merge/compression storage, and overlaps one compressed-byte file
write with subsequent GPU work. Preserve the 32-thread prepared sampling dispatch,
ordered float32 operations, owned public region results, and invalidation after
shift changes. Never sum overlapping file-write and GPU phase timings. Both
encoded and bit-packed borrowed sources use the same corrected count contract.

## Native IO follow-up

Use [native IO performance](../../native-maped-io-performance.md) for the latest
Metal behavior: one bounded compressed read ahead, and direct retention through
`MetalPackedSource.append` while saving. Do not reread the output just to view it.
Keep the same public stage sequence and both precise float32 merge passes.
The native count codec expanded exact float byte planes in the tested cache
probe; it does not qualify a single-merge float ANS cache.

Same-session totals were 28.82 s retained versus 30.02/32.58 s reopening under
active host services; an initial retained run took 25.65 s. Do not compare those
absolute totals to the earlier 22.56–22.87 s runs without matching conditions.
Peak Metal allocation is 15.424 GiB and measured process footprint 16.734 GiB,
so this trades the previous approximate 14 GiB target for less IO while staying
below 24 GiB in the Phil measurement. Physical Rodman qualification is separate.
The output's saved chunks and precision report remain exact. Keep the lower
memory reference diagnostic available for comparisons and larger acquisitions.

## Experimental single-merge regional storage

The [regional storage experiment](../../native-maped-regional-storage.md) retains
all inputs encoded, merges each 4096-frame region once in float32, then keeps
that region's scaled uint16 codes packed with its own precision metadata.
On Phil it took 13.26–13.76 s load-through-GPU-ready, excluding saving and UI.
RMSE is 0.00544976 versus global 0.00695111; output is 6.196 GiB and peak process
footprint 15.277 GiB. Full packed restoration was audited for all 9.66 billion
values. That native experiment did not change production defaults. The Python
CUDA/MPS implementation described below now supplies calibrated cross-region
reads, reductions and versioned files. Never treat regional codes as globally scaled.

The [scaled-storage contract](../../maped-scaled-storage-api-plan.md) is now
implemented for CUDA and Torch MPS using only `dtype="scaled_uint16"`.
There is no `scaling` keyword. The full output stays packed without saving;
`save_to` is optional and does not require reopening. Generic file saving streams
regions in one pass. Preserve calibrated reads, metadata and errors in tests.
The old global-format files remain readable. Native Swift regional-file loading
and Live4DSTEM integration are separate work, not implied by Python MPS support.
See [current timings](../../maped-scaled-storage-performance.md).

## Precision vocabulary

Document `dtype` as the output storage choice, not MAPED calculation precision.
Resident MAPED supports complete `scaled_uint16` output and bounded `float32`
inspection. QuantEM.GPU IO also supports `float16`; do not advertise it as a
resident MAPED merge keyword. Keep `scaled_uint16` as the single spelling.
Scaled reads reconstruct float32 calibrated intensities; they do not recover
rounding losses. Keep storage RMSE distinct from scientific algorithm parity.


## Smaller MPS machines

Use the existing `merge_datasets(save_to="merged_master.h5",
dtype="scaled_uint16", plot_result=False)` when input/output overlap is too
large. Owned inputs are released before the full packed result is reopened;
borrowed sources remain caller-owned. Keep float32 computation and original
storage-calibration boundaries even when computation batches are smaller.
Measure native loading peaks independently: a Torch cap does not cap all Metal
allocations. The 11.89 GiB larger-machine experiment is not physical 16 GB Mac
qualification. See [the memory investigation](../../maped-16gb-memory.md).


## Scaled output uses ANS

Python CUDA and Torch MPS `dtype="scaled_uint16"` now default to ANS-resident
codes (`representation="encoded"`, `resident_codec="ans"`). Do not describe
current scaled output as bit-packed. Earlier measurements above retain their
historical codec. Float16 still uses bit packing. HDF5 disk compression remains
GPU bitshuffle/LZ4; reopening scaled files creates ANS residency with unchanged
codes and calibration. Native Swift integration requires separate qualification.
See [ANS output evidence](../../maped-ans-output.md) for memory and latency;
ANS is not guaranteed to be smaller or faster for every intensity distribution.


## ANS kernel timing boundaries

Use the existing public API; scaled uint16 ANS output selects optimized native
MPS range checks and direct ANS mean-DP accumulation automatically. Keep MAPED
science in Torch. Separate pending Torch producer time from conversion timing;
otherwise synchronization makes conversion appear to own earlier computation.
Do not add nested GPU/wall counters or claim saved/reopened time is the baseline
for a no-file result. Recent all-seven retained measurements: about 28 s through
a selected DP, about 13.4-13.7 s merge/conversion/ANS/summaries, and about 1.1 s
output ANS encoding on M5 Max. These are individual runs, not universal targets.
See [kernel evidence](../../maped-ans-output.md#follow-up-native-mps-range-and-ans-mean-kernels).
