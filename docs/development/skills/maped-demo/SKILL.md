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

Read [the current Torch MPS qualification](../../mps-maped-merge-performance.md)
for exact commands, evidence, and timing boundaries. On physical Phil M5 Max,
the current seven-tilt workflow measured **77.26-89.67 s**, versus **120.52 s**
before. A paired complete float32 merge measured **48.37 -> 34.50 s**. All
9.66 billion float32 values and all 262,144 saved DP chunks were exact.
GPU allocations peaked below **13.8 GiB**, process footprint below **14.6 GiB**,
with a 24 GiB allocation cap on a 128 GB machine. This is not a physical 24 GB
laptop run or a promised latency. Input residency was 6.998 GiB, output 5.679 GiB.

Torch already prepared weights once. The new speed comes primarily from
converting native counts during float32 accumulation instead of materializing
four overlapping float copies, plus reusable working storage and fewer waits.
The GPU IO writer uses bounded backpressure and overlaps writing with compute.
Do not reintroduce per-tilt scalar GPU reads, per-tap float copies, or periodic
full write-queue drains without a measured reason.

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
