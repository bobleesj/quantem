# Scaled uint16 output defaults to ANS

Python CUDA and Torch MPS now keep scaled output in ANS residency, matching
encoded input tilts. The notebook API remains:

```python
merged = maped.merge_datasets(dtype="scaled_uint16", plot_result=False)
maped.show()
```

`save_to="merged_master.h5"` retains the lower-memory saved workflow. Saving
still writes GPU bitshuffle/LZ4 HDF5 with uint16 codes and calibration; reopening
creates ANS residency. Files themselves are not ANS archives. Float16 still
uses bit packing. Native Swift workflow changes are outside this qualification.

## What changed

QuantEM.GPU encodes the existing scaled uint16 codes with its native count-ANS
implementations. CUDA calibrated queries use bounded ANS reads and GPU
reductions. Metal calibrated queries decode only needed ranges before applying
the same existing compensated calibration and reduction kernels. Mean/BF
queries traverse bounded regions. Returned Torch reads retain independent
ownership after later queries and source closure. MAPED float32 algorithm code
and storage calibration are unchanged. No CPU scientific fallback was added.

Loaded scaled output reports `representation="encoded"` and
`metadata["resident_codec"] == "ans"`. Explicit `representation="encoded"`
is accepted on precision loading; omitting it selects the same default. This
also applies when reopening older globally scaled files. Compression is exact
relative to scaled codes, not relative to pre-conversion float32 intensities.

## Measurements

| Full seven-tilt workflow | Output residency | Sampled peak | Merge/conversion/summaries | Load through selected DP |
|---|---:|---:|---:|---:|
| CUDA, no file | 6.285 GiB | 19.30 GiB | 6.04 s | 15.62 s |
| MPS, no file | 6.106 GiB | 17.28 GiB | 14.22 s | 28.49 s |
| MPS, saved/reopened | 6.124 GiB | 11.81 GiB | 32.07 s including save/reload | 48.51 s |

MPS selected DP reads took 1.3–1.7 ms and viewer construction 0.31–0.42 s.
The final CUDA run used 4096-frame query-reduction batches; an initial 512-frame
trial took 6.39 s for merging/conversion/summaries. Codec intervals stay at
512 frames. These are individual full runs with uncontrolled filesystem caches
and background services, not repeatable comparative speedup claims.

Previous bit-packed output was approximately 6.194 GiB on CUDA and 6.196 GiB on
MPS. ANS is slightly larger on CUDA and slightly smaller on MPS for this data.
Reopened region boundaries can change compression overhead without changing
codes or calibration. The saved MPS path is slower than the previous 38–42 s
bit-packed end-to-end measurements; ANS is the requested default, not a claim
of universally better compression or latency.

Memory uses process NVML allocation on CUDA and Torch-reported native driver
allocation on MPS. The saved MPS run applied a 12 GiB Torch cap on a larger Mac.
It does not qualify physical 16 GB hardware, browser rendering, drag latency,
OS pressure or swap behavior.

## Verification

50 tests passed across CUDA/MPS precision and MAPED resident/boundary suites.
They cover exact uint16 codes and restored values against NumPy, regional
crops, means and BF, explicit encoded loading, saving/reopening, legacy global
files, float16 conversion, and returned-read ownership. No scientific tolerance
or numerical baseline was relaxed.

Both complete MPS precision reports, resident and saved/reopened, equal the
previous baseline. CUDA differences were confined to regional RMSE reduction
roundoff (up to 4.34e-18 in the first run); all other report fields matched.
Report equality alone is not a full-volume bitwise comparison. The identical
float32 MAPED path and independent codec/conversion tests provide complementary
checks.

[Compact evidence and reproduction commands](benchmarks/2026-09-12-ans-output/)
are retained alongside the original benchmark scripts and notebook.
