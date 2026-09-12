# MPS MAPED merge qualification, 2026-09-12

The later [inspection and performance qualification](mps-maped-inspection-performance.md)
adds no-save region inspection and reduces full execution to 44-48 seconds.
The measurements below describe the earlier implementation and remain retained
as its historical baseline.

The Torch MPS workflow retains all seven corrected encoded inputs, merges
bounded float32 regions twice, saves globally scaled uint16, and reopens the
complete packed output. Public methods and scientific parameters are unchanged.
This measurement is for **Torch MPS**, separately from native Swift/Metal MAPED.

## Measured performance

Physical Apple M5 Max, 40 GPU cores, 128 GB unified memory, Torch 2.13.0.
Seven real experimental acquisitions, each `(512, 512, 192, 192)`, with the
same median correction and alignment settings. No binning, cropped benchmark,
CPU scientific fallback, or full float32 output allocation was used.

| Measurement | Before | Optimized | Repeat |
|---|---:|---:|---:|
| Load through saved-result reopening and summaries | 120.52 s | 89.67 s | 77.26 s |
| Merge, save, reopen, and output summaries | 104.76 s | 72.54 s | 63.43 s |
| Load seven encoded inputs | 11.52 s | 12.58 s | 10.41 s |
| Input summaries | 3.32 s | 3.71 s | 2.79 s |
| First merge and global range | not separately instrumented | 34.27 s | 26.80 s |
| Second merge, conversion, compression, and writing | not separately instrumented | 33.57 s | 32.78 s |
| Packed reopening | not separately instrumented | 4.03 s | 3.22 s |
| Peak Metal allocation | 15.98 GiB | 13.68 GiB | 13.78 GiB |
| Peak process footprint | not recorded | 14.55 GiB | 14.59 GiB |

End-to-end time fell 26-36% in these two optimized runs. The two merge passes
still run in sequence because the second needs the completed global scale.
The first-pass total includes its range reductions. The second-pass total
includes writing, with overlap, so these are not individual kernel timings.

An additional paired test alternated the original and optimized implementation
on the **same already-resident inputs**, comparing every output strip. One
complete float32 merge pass took **48.37 s before and 34.50 s after**, a **28.7%
reduction**. This is the controlled evidence for faster merging independent of
file loading and saving. Per-strip medians were 0.7566 and 0.5374 seconds.

OS file-cache state was not forced cold, and separate runs varied. These are
measured workflow times, not a hardware-limit claim or a fixed latency promise.
Compilation/imports and subsequent parity checks are outside the workflow timer.
`/usr/bin/time -l` includes them and reported zero swaps in both final runs.

The process imposed a 24 GiB MPS allocation cap on the 128 GB machine. This is
not a physical 24 GB Mac test. Measured process footprint stayed below 14.6 GiB.
All seven inputs occupied 6.998 GiB; the reopened output occupied 5.679 GiB.
Owned inputs were released before reopening. Both allocation and footprint
measurements describe overlapping memory and must not be added together.

## What changed

QuantEM keeps the MAPED calculation in Torch:

- Read the 14 fixed scan-shift scalars once instead of repeatedly waiting on
  accelerator scalar reads inside the strip/tilt loops.
- Reuse a bounded scan-sampling workspace. Each yielded merged result remains
  independently owned and valid when subsequent strips are computed.
- Let Torch's `add_` convert native uint8/uint16 counts during float32
  accumulation. The previous code materialized four overlapping float32 copies
  for the four interpolation taps. Tap order and seven-tilt accumulation order
  are unchanged. Other input dtypes still convert to float32 first.

QuantEM.GPU supplies generic infrastructure:

- Queue MPS precision-report reductions before transferring their small scalar
  results, preserving the same reductions and integer counters.
- Bound the HDF5 background queue to two pending byte batches and let it apply
  backpressure. Remove the unconditional drain after every fourth conversion
  block. File boundaries and final completion still drain and surface errors.

Weights and detector grids were already computed once in the Torch MPS path;
that is not a new optimization here. No MAPED algorithm, native kernel, or new
public API was added to QuantEM.GPU. Native Swift/Metal MAPED was not changed.

The initial synchronization/workspace-only trial took 132.38 seconds, with
18.54 seconds loading. It did not demonstrate a speedup and is retained as
negative evidence. A focused full-strip test then showed the direct-count
accumulation step taking 24-28 ms instead of 37-41 ms, with exact float32
results. The full qualification below follows that change.

## Numerical and lifecycle qualification

- All **9,663,676,416 float32 merged values** matched the original MPS code
  exactly across all 64 strips, including boundaries. Float32 RMSE and maximum
  difference are both zero for this comparison.
- Both final saved outputs matched **all 262,144 compressed HDF5 DP chunks**
  byte-for-byte. This compares stored bytes without CPU decompression or
  scientific array computation.
- All input BF/DP summary hashes, origins, both shift arrays, and every saved
  precision-report field matched the baseline exactly.
- Scaled-uint16 storage remains approximate: RMS error `0.006951109748621411`,
  maximum error `0.01318359375`, scale `0.0261908817211223`, offset zero. There
  were no clipped or overflowing values. These are storage errors, separately
  from the zero before/after float32 computation error.
- Every source was read once, kept `encoded`, and received median correction.
  All owned inputs were released before full packed reopening.
- Physical MPS MAPED/lifetime and backend-boundary tests: 7 passed.
- Physical MPS precision/NumPy-oracle tests: 7 passed.
- CUDA GPU0 MAPED and boundary regression tests: 5 passed.
- CUDA GPU0 precision and compressed-HDF5 contracts: 25 passed.
- QuantEM Ruff checks and both diff checks passed. The three touched GPU IO
  files retain 17 pre-existing Ruff findings; comparison against the parent
  revision showed no new findings.

These exact-output results qualify this acquisition and the tested cases. The
existing native-port floating tolerances and parameter fixtures remain frozen.

## Reproduce

Use the existing public workflow with `device="mps"`. No new tuning switch is
needed:

```python
maped = MAPEDTorch.from_files(files, device="mps")
maped.preprocess(plot_summary=False)
maped.diffraction_origin(sigma=1, plot_origins=False)
maped.diffraction_align(edge_blend=2, plot_aligned=False)
maped.real_space_align(
    num_iter=20, edge_blend=5, padding=2,
    hanning_filter=True, plot_aligned=False,
)
merged = maped.merge_datasets(save_to="merged_master.h5", plot_result=False)
```

The [benchmark runner](../../tests/diffraction/benchmark_maped_mps.py) accepts
an input directory, a private output directory, `--mode`, and `--reference`.
The reference module is an unchanged copy of
`src/quantem/diffraction/_maped_resident.py` from QuantEM `519a5a3f`.
For `--mode baseline`, also put the original GPU
`src/quantem/gpu/io/backends/mps/precision.py` from `6c171565` beside it as
`reference_precision.py`. Use that original GPU checkout on `PYTHONPATH` to
reproduce the complete original IO scheduling as well as the original math.
`--mode parity` compares every float32 value and records paired merge timings.
`--mode optimized --compare-with BASELINE_DIRECTORY` verifies the complete
saved output against a prior baseline after measuring the public workflow.
`--mode profile` measures four bounded strips with per-stage synchronization;
those diagnostic timings are not end-to-end performance measurements.

Inherited `merge_generation_pass_seconds` metadata measures host call duration
and can exclude pending GPU work. Use the runner's synchronized range/workflow
timers and paired merge timings for performance comparisons.

Machine-readable evidence is in [benchmarks/2026-09-12-mps-maped](benchmarks/2026-09-12-mps-maped/qualification.json).
