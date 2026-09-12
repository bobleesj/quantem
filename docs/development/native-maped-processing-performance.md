# Native Metal MAPED processing performance

On the physical Apple M5 Max (128 GiB), the complete seven-tilt workflow now
finishes in **22.56 and 22.87 seconds**, versus the retained **85.71-second**
diagnostic baseline. This includes input preparation, alignment, both bounded
float32 merge passes, globally scaled uint16 saving, and packed GPU reopening.
It excludes validation exports and application rendering. No Live4DSTEM UI
integration is included.

The acquisition has seven `(512, 512, 192, 192)` uint16 inputs. All seven remain
encoded on the GPU during merging. Each input file is read once. The second
merge pass reads resident counts again; it does not reread the input HDF5 files.

| Stage | Before (s) | Current run 1 (s) | Current run 2 (s) |
|---|---:|---:|---:|
| Load, GPU median correction, encode, summaries | 11.28 | 5.80 | 5.70 |
| Alignment and preparation after load | 0.73 | 0.64 | 0.62 |
| First merge and global range | 29.86 | 4.80 | 4.75 |
| Second merge, conversion, compression and writing | 39.6 | 8.05 | 8.56 |
| Reopen complete packed output on GPU | 4.02 | 3.19 | 3.17 |
| Complete wall time | 85.71 | 22.56 | 22.87 |

Within the current second pass, merge generation takes 4.72–4.75 s,
GPU uint16 conversion and error measurement 0.61–0.62 s, GPU compression
1.83–1.86 s, and file writing 2.21–2.70 s. File writing overlaps GPU work:
these component durations must not be added to reconstruct wall time.
The first full float32 merge/range is available after roughly 11 seconds;
this is a processing milestone, not a measured first displayed image.

## What changed

- Prepare scan/detector sampling indices and weights once per shape and shift,
  instead of recalculating coordinates for billions of output samples. The
  cache checks shift bit patterns, including edits to exposed shift buffers.
- Reuse neighboring SIMD lanes' interpolated values when their exact indices
  agree. Preserve boundary fallbacks and ordered float32 accumulation. A
  32-thread group improved the prepared sampling path over the tested 128-thread
  configuration (29.85 and 31.06 s complete runs).
- Reuse bounded GPU decoding, ANS encoding, merge, and HDF5 compression buffers.
  Public region reads still own their results; internal scratch does not escape.
- Overlap one compressed-byte HDF5 write with subsequent GPU processing. Drain
  that write before buffer reuse, publication, cancellation, or destruction.
  Scientific array operations remain on Metal; the CPU handles file bytes.

QuantEM owns MAPED's stage order and scientific weighting. QuantEM.GPU owns
reusable count reads, sampling, buffers, precision conversion, and IO. Existing
scientific method and parameter names remain unchanged. `from_resident` accepts
both encoded and bit-packed count sources through their shared count contract;
masked borrowed sources receive a GPU median-corrected view without altering
caller-owned storage. Python MAPED remains Torch-based.

## Precision and memory

The prepared implementation matched **all 9,663,676,416 float32 values bit for
bit** against the retained native sampling path in a GPU comparison. Real input
regions also passed GPU byte-for-byte ANS round-trip verification. The optimized
native parameter qualification passed **75 cases, 45 sensitivity comparisons,
and 7 rejected-option checks** against Torch MPS without changing tolerances.
Float32 agreement with Torch uses the established cross-backend tolerances;
it is not a claim of universal bit identity across backends.

The final saved-output audit passed for all **262,144 compressed HDF5 chunks**
and identical precision metadata against the original output, without host array decoding.
All 11 native workflow tests and six shared ANS count tests passed.
Globally scaled uint16 storage remains approximate: RMSE **0.00695110997**,
maximum absolute error **0.0131225586**, and zero clipped values. This storage
error is unchanged by the optimization.

Inputs occupy **7.008 GiB** and the complete packed output **5.679 GiB**.
Measured peak Metal allocation is **9.735 GiB**. Owned input residents are
released before output reopening. This allocation measurement is not total
system memory or a physical 24 GiB Mac qualification; Rodman still needs a
separate uncontended full-workflow run.

## Reproduction and limits

Build with the QuantEM.GPU source override when testing local infrastructure:

```sh
QUANTEM_GPU_PACKAGE=/path/to/quantem.gpu swift build -c release \
  --product maped-native-benchmark
.build/release/maped-native-benchmark INPUT_DIRECTORY REPORT.json OUTPUT.h5
```

Run `swift test -c release` with the same dependency override. For a full GPU
float32 before/after comparison, set `MAPED_FULL_PARITY=1`; for input count
round-trip auditing set `QUANTEM_GPU_VALIDATE_COUNTS=1`. These diagnostic modes
add work and must not be used as ordinary performance measurements.
`native/Tests/compare_saved_chunks.py REFERENCE.h5 CANDIDATE.h5` audits serialized
output without decompressing scientific arrays on the CPU.

These are repeated loads with an existing index and uncontrolled OS file cache,
not cold-storage guarantees. Leave adequate output space: an earlier repeat
filled the benchmark disk and failed with ENOSPC; its incomplete output was
removed. Earlier write-heavy repeats ranged from 33.6 to 39.9 s under that
storage condition. A materialized two-stage interpolation experiment also
regressed and was rejected. Keep such negative measurements distinct from the
completed runs above. No speed claim is made for CUDA, Torch MPS, or Rodman from
these native Phil measurements.

Sanitized reports: [run 1](../../native/Benchmarks/results/2026-09-12-metal-load/phil-optimized-1.json),
[run 2](../../native/Benchmarks/results/2026-09-12-metal-load/phil-optimized-2.json),
and [before](../../native/Benchmarks/results/2026-09-12-metal-load/phil-end-to-end-diagnostic.json).
