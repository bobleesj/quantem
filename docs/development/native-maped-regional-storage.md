# Single-merge regional scaled uint16 experiment

The complete seven-tilt acquisition reached packed GPU residency in **13.26 and
13.76 seconds** on an Apple M5 Max. The benchmark retains every region and reads
selected diffraction patterns using its region's scale. It does not save a file
or render an application UI. It is an experimental benchmark, not a changed
MAPED default or a new loader/file format.

## Workflow

1. Load all seven native count inputs into lossless ANS residents, with the
   existing GPU median correction and summaries.
2. Run the unchanged float32 alignment. Merge each 4096-frame region once using
   the established float32 interpolation and accumulation.
3. Measure that region's complete range on Metal, convert to scaled uint16,
   measure restored-value errors, and retain the codes in `MetalPackedSource`.
4. Keep each region's precision metadata with its packed source. Release input
   residents after processing. Selected DP reads restore intensities on Metal
   using the corresponding scale, without a file reread or another merge.

The experiment composes existing public QuantEM.GPU operations. No uint16
accumulation, float16 intermediates, binning, clipping, or changed alignment
parameters are used. Regions are eight scan rows each for this acquisition.

## Measurements

Seven inputs each have shape `(512, 512, 192, 192)`; there are 64 output regions
and 9,663,676,416 output values. Repeated loads use an existing index and
uncontrolled OS cache; these are not cold-storage guarantees.

| Stage | Run 1 (s) | Run 2 (s) |
|---|---:|---:|
| Input load and preparation | 4.91 | 4.96 |
| Alignment after loading | 0.65 | 0.69 |
| One complete float32 merge | 6.27 | 6.65 |
| Regional range, conversion and error reports | 1.01 | 1.03 |
| Packing | 0.33 | 0.34 |
| Processing after alignment, including orchestration | 7.61 | 8.01 |
| Complete load-through-GPU-ready wall time | 13.26 | 13.76 |

Selected DP restoration had median latency **0.149–0.156 ms**. First-use latency
was included, with maxima 1.21–1.39 ms. These are synchronized native GPU reads,
not measured viewer presentation latency. Do not compare these totals directly
with earlier save-inclusive global-scaling runs as a controlled speedup.

Packed output uses **6.196 GiB**, compared with **5.679 GiB** for global scaling.
Finer local scales use more packed bits. Peak measured Metal allocation is
**14.811 GiB**; maximum process footprint is **15.277 GiB**, with zero recorded
swaps. This is below 24 GiB on Phil but does not replace physical 24 GiB Mac
qualification. Each region also owns a small precision workspace, included in
the measured peak but not in the packed-payload count.

## Precision against the same float32 values

| Metric | Global scale reference | Regional scales |
|---|---:|---:|
| Whole-output RMSE | 0.00695110997 | 0.00544976257 |
| Maximum absolute error | 0.0131225586 | 0.0131225586 |
| Positive values rounded to zero | 1,813,736,950 | 1,531,128,400 |
| Overflow | 0 | 0 |

RMSE improves **21.6%**, and **15.6% fewer positive samples** round to zero.
That count includes small interpolated intensities and is not a percentage of
lost total signal. The brightest region still determines the maximum error;
regional scale steps range from 0.0124925381 to 0.0261908836 intensity units.

The audit compares both storage choices with the same freshly merged float32
region. The global reference reproduces all previously recorded precision
metrics exactly. Origins, shifts, source-read counts, regional scales, error
reports and packed sizes match exactly across the audit and two timing runs.
All **9,663,676,416 restored packed values** match direct GPU restoration of the
regional codes bit-for-bit. Each region's error satisfies its half-step bound
plus float32 rounding. The existing converter's independent NumPy fixtures
remain the arithmetic oracle; the new full-data audit checks composition and
storage, rather than replacing those fixtures.

The audit takes extra time for the global comparison and packed restoration;
its 16.48 s total is separate from the two uninstrumented timing runs above.
All large scientific operations run on Metal. The host combines only the
GPU-generated per-region error scalars to report whole-output RMSE.

## Running the experiment

Build the existing native benchmark with the local QuantEM.GPU override:

```sh
QUANTEM_GPU_PACKAGE=/path/to/quantem.gpu swift build -c release \
  --product maped-native-benchmark
MAPED_REGIONAL_STORAGE=1 .build/release/maped-native-benchmark INPUT_DIRECTORY REPORT.json
```

For the full audit, also set `MAPED_REGIONAL_REFERENCE_REPORT` to a previous
complete global-scaling benchmark JSON containing its `precision` object.
No output HDF5 argument is used by this experimental mode.

Before production adoption, the generic resident reader and persisted format
must explicitly support per-region scales, including reads across boundaries,
calibrated reductions and intensity restoration. A viewer must apply those
scales before scientific reductions or display normalization. Never write these
codes with a single global scale or present the raw code values as intensities.
The current public notebook and native merge defaults remain globally scaled;
Live4DSTEM UI integration remains a separate task.

Evidence: [audit](../../native/Benchmarks/results/2026-09-12-regional-storage/audit.json)
and [timing repeats](../../native/Benchmarks/results/2026-09-12-regional-storage/timings.json).
