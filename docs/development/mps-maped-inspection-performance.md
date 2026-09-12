# Faster MPS MAPED and inspection before saving

Historical qualification. The current full resident workflow is documented in
[scaled uint16 storage](maped-scaled-storage-api-plan.md); it no longer needs saving or reopening.

The seven-tilt Torch MPS workflow took about 75 seconds because exporting
required two complete float32 merges, precision measurement, compression,
writing, and reopening. A compressed 7 GiB input is not a 7 GiB calculation:
one pass combines 67.65 billion detector values before interpolation taps.
An inspection should not require that entire export.

The current implementation provides a small exact float32 inspection through
the existing `merge_datasets` method, and accelerates full export. MAPED's
scientific math remains Torch in QuantEM. QuantEM.GPU owns count residency,
GPU reads, precision conversion, and writing. Native Swift/Metal MAPED and
Live4DSTEM UI integration are unchanged.

## Inspect, then save when needed

After the ordinary load, preprocessing, and alignment sequence:

```python
patch = maped.merge_datasets(
    scan_region=(252, 260, 252, 260), plot_result=False,
)
maped.show()

# When satisfied, save the complete acquisition using the existing inputs.
merged = maped.merge_datasets(save_to="merged_master.h5", plot_result=False)
```

`scan_region` uses full aligned-scan coordinates, with exclusive row/column
stops. The returned patch retains all detector pixels and float32 intensities.
It is a region of the same merge, not a differently aligned or binned estimate.
Its metadata records the original scan region. The aligned seven encoded
inputs remain available for another patch or full export. Inspection writes
no data file and needs no global scale, uint16 conversion, or reopening.
A single in-memory selection is limited to 4096 scan positions. A complete
small acquisition also works without `save_to`; a large full acquisition still
requires the bounded saved-output path.

Inspection does not prove that every scan position is correct. The separate
full-volume parity audit below checks every merged value. The benchmark also
measures an exact full-scan BF/mean-DP overview in one bounded pass; that is an
experiment, not a new public viewer API.

## What was consuming time

1. MPS count reads decoded a native Metal buffer, copied its complete decoded
   contents into Torch, and waited. The new generic `FourDSTEMData.read()` path
   decodes directly into independently owned Torch MPS storage. It waits for
   previous Torch use before writing recycled storage and for the native
   command before returning. It neither releases Torch-owned buffers nor
   lends an output that a later read overwrites.
2. Scan interpolation made four large float32 update passes. For large interior
   regions, Torch now compiles those four ordered taps together. Detector
   interpolation and seven-tilt accumulation order remain unchanged. Shift
   indices and weights are tensor inputs, so changing fractional shifts does
   not compile a different program. Edge regions use the established eager
   calculation. `compile_merge=False` retains that calculation throughout.
3. Range measurement separately scanned for minima, maxima, and finiteness.
   MPS now uses `aminmax` and checks its two scalar endpoints. Its NaN and
   infinity behavior is covered by tests, including strided arrays.

The compiled interpolation uses one exact native-count-to-float32 cast because
this Torch compiler does not accept uint16 inputs. The working-region shape is
fixed within a pass, so the compiler can simplify its indexing. It does not
change scientific precision or introduce a hand-written kernel into MAPED.

## Measured full workflow

Physical Apple M5 Max, 40 GPU cores, 128 GB unified memory, Torch 2.13.0.
Seven original acquisitions, each `(512, 512, 192, 192)`, median correction,
no binning, all seven encoded inputs resident, and a 24 GiB MPS allocation cap.
This is not a physical 24 GB laptop test. OS file-cache state was not forced
cold. Imports, later parity comparisons, and browser rendering are excluded
from workflow timings; compilation needed during merging is included.

| Measurement | Fresh control | Optimized |
|---|---:|---:|
| Load through saved-result reopening and summaries | 75.44 s | 44.19 s |
| Load seven inputs | 10.92 s | 10.50 s |
| Input summaries and alignment | 3.81 s | 3.31 s |
| First merge and complete range | 25.78 s | 10.08 s |
| Second merge, conversion, compression, writing | 31.41 s | 16.68 s |
| Packed reopening | 2.89 s | 2.98 s |
| Merge/save/reopen and output summaries | 60.70 s | 30.37 s |
| Peak Metal allocation | 13.78 GiB | 13.78 GiB |
| Peak process footprint | 14.57 GiB | 14.54 GiB |

The repeat completed in **47.53 seconds**, with a 10.28-second first merge,
17.52-second second merge/conversion/write, and 3.34-second reopen. Its peak
Metal allocation was 13.68 GiB and process footprint 14.80 GiB. The two full
runs are **37–41% faster end to end** than the fresh control. GPU allocation and
process footprint are overlapping measurements; do not add them. Input
residency is 6.998 GiB and reopened packed output is 5.679 GiB. Owned inputs
are released before reopening; borrowed inputs stay caller-owned.

A paired full-volume test alternated the original and optimized read/merge
paths on the same resident sources. It measured **19.93 -> 10.02 seconds**
for one complete float32 merge, including the first compiled invocation.
That is about 50% less time. The first optimized strip took 0.875 seconds;
the optimized strip median was 0.144 seconds versus 0.311 seconds before.
These synchronized comparisons are distinct from inherited
`merge_generation_pass_seconds`, which measures host calls and can exclude
queued work.

## Inspection measurements

The final no-save qualification measured an 8x8 patch with the complete
192x192 detector at **0.085 seconds**, followed by Show4DSTEM construction at
0.126 seconds. Loading through the first patch was **16.74 seconds**. Other
selected regions took 0.020–0.042 seconds. A DP read from the already-merged
patch took 0.196 milliseconds; that is not a newly merged DP latency or a
browser frame rate. Each 8x8 float32 patch occupies 9 MiB.

The complete BF/mean-DP overview experiment took **14.77 seconds** without
saving. The earlier dynamic-shape candidate took 21.22 seconds. These overview
measurements include GPU reductions and differ from the merge-only audit.
No claim of whole-acquisition instantaneous viewing or live browser gesture
verification follows from these construction/compute timings.

## Precision and negative trials

- All 9,663,676,416 float32 values matched the frozen original merge and original
  read implementation exactly across 64 strips. No tolerance was widened.
- Every one of the 262,144 saved compressed DP chunks matched the existing
  baseline byte-for-byte. Input BF/DP hashes, origins, both shift arrays, and
  all precision-report fields also matched.
- Selected-region values match their full-merge counterparts exactly, including
  edge patches and patches crossing working-region boundaries.
- Scaled-uint16 storage error is unchanged: RMSE 0.006951109748621411, maximum
  0.01318359375, zero clipping/overflow. Inspection retains float32, so it does
  not incur this storage conversion error.
- Smaller 1/2/4-row batches did not consistently beat the existing 8-row work
  region. They were not promoted as a speedup.
- Compiling native uint16 directly failed in the installed compiler. The single
  exact float32 cast is required; reinterpreting signed counts was not used.
- A flattened four-tap gather did not provide a reliable improvement. A
  dynamic-shape column gather improved little end to end: 76.71 seconds versus
  the 75.44-second control, with identical saved output. The fixed-shape column
  gather and direct Torch read are the qualified combination.

## Reproduce

The control revisions are QuantEM `d37c1615` and QuantEM.GPU `af7e4988`.
Freeze `_maped_resident.py` and GPU IO `_read.py` from those revisions, and
`backends/mps/precision.py` as `reference_precision.py` beside the former.
Run `tests/diffraction/benchmark_maped_mps.py` with input/output directories,
`--reference`, `--reference-read`, and `--mode baseline|parity|optimized`.
Use `--compare-with` to compare all saved DP chunks and metadata against an
existing untouched output after the workflow timer stops.

`tests/diffraction/benchmark_maped_preview_mps.py` measures the public selected
region and Show4DSTEM construction, then the explicitly separate full-overview
experiment. Both runners use private output directories and retain only bounded
GPU working tensors. Test numerical/library behavior on physical MPS and run
CUDA regression checks before publishing. The portable source remains Torch;
no CUDA performance or new native Swift/Metal speedup is claimed here.

## Retained evidence and regression checks

[Benchmark JSON records](benchmarks/2026-09-12-mps-maped-inspection/) retain the
fresh control, two final exports, full float32 audit, inspection, and rejected
dynamic-shape trial. These records omit private source identities.

Physical MPS: QuantEM 9 passed/1 skipped, QuantEM.GPU 10 passed. CUDA GPU0:
QuantEM 6 passed/1 skipped, QuantEM.GPU precision/save contracts 25 passed.
The MPS read tests cover full-range uint16, partial columns, repeated reads,
and retained output ownership after source closure. No Swift implementation
or native UI performance was changed or requalified by these Python checks.
