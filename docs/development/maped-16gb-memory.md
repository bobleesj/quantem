# MAPED saved workflow for smaller Macs

This is a candidate for 16 GB unified-memory Macs. Measurements were made on
an M5 Max with more physical memory, a 12 GiB Torch allocation cap and sampled
native driver memory. They are not physical 16 GB qualification. OS memory,
other apps, swap and browser interaction remain outside this claim.

## Existing notebook API

After loading with `MAPEDTorch.from_files` and completing alignment:

```python
merged = maped.merge_datasets(
    save_to="merged_master.h5", dtype="scaled_uint16", plot_result=False
)
maped.show()
```

Seven encoded inputs remain on the GPU during processing. MAPED evaluates
2048-frame row-aligned computation blocks and assembles the original
4096-frame storage regions. All interpolation and accumulation remain Torch
float32. The storage scales and boundaries are preserved. QuantEM.GPU writes
calibrated uint16 regions with GPU compression, without retaining the full
packed output. MAPED releases owned inputs and reopens the complete packed
result, approximately 6.20 GiB, for viewing. No CPU scientific fallback is used.

The existing no-file call remains faster but retains input and output residency
simultaneously. Borrowed `from_resident` inputs are not released; users retaining
other references or other GPU datasets need additional memory. The saved path
requires disk space for the complete HDF5 output and its shards.

## Measured first public run

| Stage | Time | Sampled driver peak |
|---|---:|---:|
| Load seven inputs | 12.93 s | 8.67 GiB |
| Preparation and alignment | 3.89 s | 8.31 GiB |
| Merge, conversion, saving, reopening, summaries | 24.95 s | 11.89 GiB |
| Selected DP read | 0.40 ms | bounded read |
| Viewer construction | 0.22 s | 6.23 GiB |

Loading through the selected DP took 41.76 s. The generated float32 merge
traversal recorded 12.88 s inside the combined stage. That internal timer
excludes consumer IO and does not isolate every asynchronous GPU operation.
Earlier prototypes separated merge/conversion/saving (20–22 s) and reopening
(2.3–2.6 s); those are separate runs, not an exact decomposition of this run.
Browser rendering, dragging and physical-machine pressure were not measured.

The earlier no-file MPS peak was 17.79 GiB. Reducing ANS load batches from
32768 to 8192 frames removes large overlapping decoded and encoding buffers.
Both are multiples of the 512-frame codec interval. The retained inputs remain
approximately 7 GiB. No-file output precision and the scientific API are unchanged.

## Checks and rejected trials

The first public run's entire saved precision report equals the previous MPS
no-file baseline, including every regional scale, count, maximum error and
RMSE. This is not by itself a whole-volume bitwise comparison. Tests also check
exact float32 equality for fractional shifts across uneven work/storage block
boundaries, saved and unsaved median-corrected data, native ANS reads, precision
conversion against NumPy, and CUDA saved-output behavior. No numerical pins or
tolerances were changed. The lifecycle assertion now requires owned sources to
be released before reopening, matching the intentional behavior change.

Initial disk-streaming runs with 4096-frame computation exhausted 11 and 12 GiB
Torch caps. With smaller computation, merging fit but the old loader peaked at
13.7 GiB. Flushing Torch cache after every output region saved only about
0.33 GiB during merging and slowed it, so it was not adopted. A cap alone did
not stop native allocations exceeding the cap; direct driver sampling found
the remaining loading peak. Only final public runs should be used for the
shipped implementation's memory claim.

[Benchmark and anonymous reports](benchmarks/2026-09-12-maped-16gb/) record the
full seven-tilt workflow. Keep the notebook call simple; internal scheduling
and generic GPU IO own the memory behavior.

The repeat public run peaked at 11.81 GiB and took 38.44 s through the selected
DP. Its complete precision report again matched the baseline.
