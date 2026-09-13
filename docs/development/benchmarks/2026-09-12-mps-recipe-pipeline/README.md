# MPS recipe reconstruction and precision conversion

This experiment does **not** achieve a two-to-three-second full MAPED reconstruction. It tests the same seven full-size acquisitions on the Apple M5 Max test host, with CPU fallback disabled, saved alignment parameters, encoded inputs resident on MPS, Torch float32 merging, and a complete ANS scaled uint16 viewing volume. Loading is timed separately. No alignment search, output-file save, or browser rendering is included in processing time.

## Measurements

| Configuration | Merge | Conversion, reporting, ANS | Processing through selected DP |
|---|---:|---:|---:|
| Original, eight scan rows | 9.48 s | 3.75 s | 13.24 s |
| Original, sixteen scan rows | 8.80 s | 3.76 s | 12.56 s |
| Original, eight-row repeat | 9.19 s | 3.99 s | 13.18 s |
| GPU statistics reduction, eight-row later repeat | 14.39 s | 4.10 s | 18.49 s |
| GPU statistics + occupancy tuning, eight rows | 13.98 s | 3.29 s | 17.28 s |
| GPU statistics + occupancy tuning, repeat | 12.82 s | 3.20 s | 16.02 s |

The unchanged merge varied substantially during later runs. These numbers do not establish an end-to-end speedup from the new precision kernel. They retain negative evidence rather than comparing only the best cases. No other process was stopped. Input loading took approximately 11.6–15.0 seconds and is excluded above. Compiler setup before the first yielded region and framework initialization are not uniformly covered by these processing timers; these are warmed processing results, not cold-start promises.

A same-process conversion probe alternated the original 8192-partial topology with larger configurations on the same real merged region. The original took approximately 6.3–6.7 ms; 65536 partials took approximately 5.1 ms. The first and final original trials bracketed the candidate trials. This supports a conversion-stage improvement, not a whole-workflow claim. A 131072-partial trial was slightly faster but doubles statistics scratch compared with the selected configuration.

The full encoded viewing result occupies 6,680,914,680 bytes. Eight-row trials sampled approximately 16.62 GiB of Metal driver allocation while retaining all seven inputs and the viewing volume. Sixteen rows used approximately 19.66 GiB. These samples are at region boundaries, not guaranteed peaks, and this is a 128 GB Mac, not qualification for a physical 16 GB laptop.

## Changes

MAPED science stays in QuantEM/Torch. QuantEM.GPU changes only generic precision infrastructure:

- Reduce partial precision statistics on Metal, returning scalar statistics instead of copying thousands of partial rows into Python.
- Use a compensated pair of floats for squared-error aggregation and 64-bit integer counters. Preserve infinity reporting.
- Increase independent conversion work from 8192 to 65536 partials, improving occupancy while keeping scratch bounded.

Calibration remains fixed at 1024 frames for this recipe comparison, independent of the merge batch size. This matches the preceding CUDA recipe experiment; it is not a change to public MAPED defaults. No new public parameter was added. No CUDA implementation was changed.

## Numerical evidence

Three complete eight-versus-sixteen-row MPS comparisons each checked all 9,663,676,416 float32 values bit-for-bit and found zero mismatches. These are same-MPS comparisons, not proof of exact CUDA/MPS interchangeability.

A separate full-volume conversion comparison checked 8192 versus 65536 partials: all 9,663,676,416 uint16 codes matched exactly. Scale, offset, maximum error and integer counters matched in every region. The largest regional RMSE-report difference was 2.76487773648304e-12, from a changed compensated reduction topology. This does not change the stored codes or reconstructed float32 science. The test bounds relative RMSE-report differences at 1e-7; it does not relax output-code parity.

The GPU-only statistics edge test passed on the Apple M5 Max test host: tiny contributions beside larger partial sums, counters exceeding uint32, and infinity reporting. Command in the QuantEM.GPU checkout:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 QUANTEM_MPS_PRECISION_TEST=1 PYTHONPATH=src python -m pytest -q tests/hardware/mps/test_precision_statistics.py
```

## Bottlenecks and next target

A diagnostic full pass inserted synchronization at stage boundaries. It measured approximately 2.17 s decoding, 2.71 s scan interpolation, 3.35 s detector interpolation, 0.68 s range measurement, 1.63 s conversion/error measurement (including 0.084 s final statistics reduction), and 1.36 s ANS encoding. Instrumentation changes scheduling; these are attribution measurements, not additive predictions for an uninstrumented run.

The next substantial gains must address Torch interpolation and native ANS decoding, while retaining exact same-backend arithmetic and Torch-owned decoded storage. The precision change alone cannot produce a few-second full-volume result. Returning selected regions sooner is a separate latency target and must not be substituted for full-volume timing.

## Reproduce

Use an MPS Python environment, with the tested QuantEM and QuantEM.GPU source trees on `PYTHONPATH` and `PYTORCH_ENABLE_MPS_FALLBACK=0`:

```sh
python run.py INPUT_DIRECTORY RECIPE_JSON results.json --rows 8 16 4 8
python profile_stages.py INPUT_DIRECTORY RECIPE_JSON profile.json
python conversion_probe.py INPUT_DIRECTORY RECIPE_JSON conversion.json
python verify_conversion.py INPUT_DIRECTORY RECIPE_JSON conversion-parity.json
```

The input directory and recipe are private local artifacts, supplied explicitly. The script files and compact benchmark evidence are kept in this repository. Baseline JSON retains full precision reports. Local source changes have not been pushed or published.

## Follow-up: native command timestamps explain conversion overhead

`profile_commands.py` wraps existing command completion calls and records Metal
GPUStartTime/GPUEndTime separately from host wall time. It separates initial
input encoding from processing commands. The full-volume diagnostic releases
each encoded output part after measurement, unlike the retained-output benchmark;
its 2.62 s conversion/reporting/ANS total is therefore not a replacement for the
3.20 s retained-output measurement or proof of a new end-to-end speedup.

| Processing component | Stage wall time | Metal command execution |
|---|---:|---:|
| Range/calibration measurement | 0.546 s | 0.381 s |
| Code conversion and error reporting | 0.900 s | 0.311 s |
| ANS construction, encoding and compaction | 1.170 s | 0.483 s |
| Total | 2.616 s | 1.175 s |

Error-reduction time is included in conversion, not added twice. GPU commands
are waited on sequentially here; timestamps report their execution intervals.
The difference of approximately 1.44 s includes allocation, Python work, command
construction/dispatch, queue wait and synchronization. It is not a measurement
of pure Python execution alone. All science remained on MPS/Metal.

There are 256 calibration regions. Each submits range, encoding/error,
statistics-reduction, ANS-table initialization, ANS encoding and compaction
commands. ANS probability tables are rebuilt 256 times despite being invariant.
That initialization alone is only about 0.071 s completion wall time; caching it
cannot explain or remove the full overhead. Native encoding plus compaction
account for approximately 0.459 s of GPU execution.

The next useful work is to reduce synchronous round trips and per-region
allocation, reuse immutable ANS resources with correct lifetime management, and
optimize the remaining range and encoding kernels. Batching must preserve the
1024-frame calibration boundaries and exact integer codes; changing calibration
size would be a different precision policy. Generic queue/codec improvements
belong in QuantEM.GPU, with MAPED math remaining in QuantEM/Torch.

A 0.1–0.2 s target is not established by these results. Current GPU execution
alone is approximately 1.17 s. An illustrative current-path traffic estimate is
36 GiB range reads + 36 GiB conversion reads + 18 GiB code writes + 18 GiB ANS
reads + about 6 GiB output writes, before scratch and compaction traffic. At
Apple's advertised 614 GB/s peak for the 40-core M5 Max, that is around 0.20 s
under idealized bandwidth-only assumptions. This is neither a measured bandwidth
nor a universal lower bound: cache reuse, fusion, and a different implementation
change the traffic. It shows why 0.2 s for the complete existing path leaves
almost no allowance for arithmetic, entropy coding, extra passes or dispatch.
Hardware source: https://www.apple.com/macbook-pro/specs/

Committed JSON reports retain aggregate statistics and region counts. Full per-region precision reports are archived with the private run; regenerate them with the included scripts when needed.
