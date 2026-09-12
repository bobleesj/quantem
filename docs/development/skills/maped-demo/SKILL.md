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
`merge_datasets(save_to=..., plot_result=False)`. The result reopens packed;
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

Distinguish **Torch MPS** from **native Swift/Metal**. Native Metal's separately
qualified run was 67.56 s, 8.81 GiB Metal allocation and 10.13 GiB process
footprint; its 75 parameter cases and 45 sensitivity intervals are described in
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
