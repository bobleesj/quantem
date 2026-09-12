# Removing native MAPED IO waits

The native Metal workflow now reads ahead one compressed input region and retains
the scaled uint16 codes in the existing packed-source builder during saving.
No Python or Torch runtime is involved. Scientific stages, parameters, global
scaling, and the saved file format are unchanged. There is no new public API.

## Measured non-computational costs

The earlier sequential seven-tilt loader spent 1.062 s reading compressed bytes,
0.022 s copying index metadata, and 0.378 s in decoder wall time beyond measured
GPU execution. GPU decompression itself took 1.724 s; the encoding, median, and
summary consumer took 2.983 s including its synchronization. Setup was 0.012 s.
These timings come from repeated loads with uncontrolled OS cache.

The existing QuantEM.GPU read-ahead helper reduced exposed read waiting to
0.070 s. Reader total fell from 6.232 to 5.367 s in the profile. File reading
still occurs, overlapping GPU decoding and encoding; it is not eliminated.
The helper keeps one pending compressed region, preserves ordered consumption,
propagates errors, and drains on cancellation or consumer failure. It does not
stage a second dense acquisition or perform CPU scientific calculations.

Previously the output was GPU-scaled, compressed and saved, then reread,
decompressed and packed again. `MetalPackedSource.append` already accepts the
GPU codes produced during conversion. MAPED now calls it while saving and
returns that resident after successful file publication. Packing costs
0.383–0.415 s; the unnecessary 2.91–3.03 s reopening step is gone.

## End-to-end comparison and memory

A same-executable reference/candidate/reference sequence on Phil measured:

| Path | Total (s) | Load (s) | Range merge (s) | Second pass/save (s) | Reopen (s) |
|---|---:|---:|---:|---:|---:|
| Reference A1 | 30.02 | 6.20 | 8.09 | 11.86 | 3.03 |
| Retained output B | 28.82 | 6.38 | 8.15 | 13.43 | 0 |
| Reference A2 | 32.58 | 6.45 | 8.72 | 13.69 | 2.91 |

All three used read-ahead. The first independent retained-output run took
25.65 s. The candidate was 4–12% faster than the surrounding references,
although its file writes were slower than A1. These runs had active screen
sharing, WindowServer, WebKit GPU, and indexing processes. Their contribution
was not isolated, and no unrelated process was terminated. Do not compare the
absolute totals directly to the earlier 22.56–22.87 s measurements or claim a
new sub-20-second qualification. Timings exclude validation exports and UI
rendering. File writes overlap GPU work.

Retaining packed output alongside encoded inputs trades memory for less IO:
peak Metal allocation increases from 9.735 to **15.424 GiB**. The separately
measured process footprint was **16.734 GiB** with zero recorded swaps.
Inputs are released after the second pass. The complete result remains
5.679 GiB. This fits below 24 GiB in the Phil measurement, but exceeds the
previous approximate 14 GiB optimization target. A physical 24 GiB Mac still
requires its own system-memory qualification. Larger acquisitions may require
the lower-memory save/reopen strategy.

## Why this still merges twice

The exact complete-source maximum is needed before applying the established
global uint16 scale. A single merge would require storing all intermediate
float32 results losslessly, or changing the scale/precision contract.

We probed the existing native count-ANS encoder on exact float32 byte planes
in three 4096-frame regions. All byte round trips were exact, but each
603,979,776-byte region expanded to 1,149,779,578–1,189,957,099 resident bytes.
This count codec uses uint16 literal streams even for declared uint8 input;
it is not a specialized floating-point entropy codec. Linear projection is
roughly 69–71 GiB for the merged cache alone, and encoding plus reading adds
about 6.1–6.4 s before scaling. The second merge costs about 4.7 s in the earlier
uncontended measurements. This specific reuse is rejected, not evidence that
all possible float ANS schemes must fail. A purpose-built exact float codec
would need its own compression, speed, and memory qualification before enabling
a single-merge workflow. No float16 intermediate, clipping, approximate range,
or changed scientific weights were introduced.

## Validation and reproduction

All 262,144 compressed output chunks and precision metadata match the frozen
original output exactly. Every region of all seven input sources passed the
GPU count round-trip check with read-ahead enabled. All 11 native workflow
tests pass, including retained-versus-reopened selected frames, partial final
regions, cancellation, and consumer exceptions. Float32 merge arithmetic is
unchanged from the prior full 9.66-billion-value parity qualification.

Use the existing benchmark command from
[native processing performance](native-maped-processing-performance.md).
Diagnostic controls, not scientist-facing parameters:

- `QUANTEM_GPU_LOAD_PROFILE=1`: report file/GPU/consumer phase timings.
- `QUANTEM_GPU_LOAD_REFERENCE=1`: disable read-ahead for comparison.
- `MAPED_REOPEN_REFERENCE=1`: use the previous lower-memory save/reopen path.
- `MAPED_FLOAT_CACHE_PROBE=1`: run the rejected exact-byte count-codec experiment.
- `QUANTEM_GPU_VALIDATE_COUNTS=1`: GPU-verify every input region, adding audit work.

[Sanitized evidence](../../native/Benchmarks/results/2026-09-12-metal-io/README.md)
retains the slower references and rejected probe alongside the candidate.
