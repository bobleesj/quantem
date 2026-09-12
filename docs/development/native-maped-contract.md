# Native MAPED contract, version 1

The native workflow aligns and merges the same encoded seven-tilt acquisitions
as `MAPEDTorch`, without embedding Python or Torch in a native application.
QuantEM owns stage order and scientific parameter choices. QuantEM.GPU provides
native loading, exact encoded residency, image operations, sampling, reductions,
and bounded scientific file writing. Native algorithm code calls these public
operations without compiling kernels or managing Metal device state.

The native methods retain `from_files`, `preprocess`, `diffraction_origin`,
`diffraction_align`, `real_space_align`, and `merge_datasets`, including the
existing scientific keyword names. Presentation belongs to the embedding app.

Counts have axes `(scan_row, scan_column, detector_row, detector_column)`.
Hot pixels use the existing GPU local 3x3 median with invalid neighbors excluded;
even neighbor counts use the integer average of the central pair. No detector
binning, clipping, or intensity normalization is introduced. Exact count reads
and integer reductions must agree bit-for-bit with independent fixtures.

The floating-point reference is the current resident `MAPEDTorch` workflow.
Preserve its zero-padded interpolation, periodic Hann windows, reflect-padded
Gaussian/Sobel filters, Fourier correlation with local DFT refinement, and
mean-centered shifts. In particular, detector/image translations retain the
existing `(dimension - 1) / dimension` effective displacement from its normalized
grid; scan interpolation in the resident merge uses the requested displacement
directly. Do not silently correct this convention during a backend port.
Construct the normalized grid with separately rounded float32 multiplication
and subtraction. Compiler contraction shifts boundary coordinates by one ULP
and breaks strict merge parity. The frozen NumPy boundary fixture protects this
operation order.

Some established parameters affect only optional plots (`padding`, `pad_val`,
and real-space `shift_method`). `weight_scale` is calculated but currently not
applied to diffraction correlation. `max_shift` selects correlation padding but
does not constrain its peak search. Preserve these meanings and document them;
changing them requires a separate scientific change.

The resident merge supports the same bounded subset: bilinear shifts, zero scan
and detector padding, scan edge blend 1, detector edge blend 0, and globally
scaled uint16 output. Other merge choices must fail explicitly. Compute the
global output range, then recompute and write bounded regions. Retain all inputs
until both passes complete. Build the packed output from the existing scaled
codes during saving, then release owned inputs. The diagnostic lower-memory
reference releases inputs before reopening the saved file.

Qualification compares intermediate summaries, origins, shifts, sampled merge
regions, saved/restored intensities, precision metrics, and source lifecycle.
Keep frozen expected values unchanged when diagnosing differences. Measure full
seven-tilt runtime and process/Metal peak separately from small correctness tests.
An implementation is not qualified until both the public native run and its
independent parity checks have executed on a physical Apple GPU.

The real-data gates are exact count summaries and origins, maximum shift
difference below 0.01 pixel, same-shift merge `rtol=3e-6, atol=2e-5`, and
independently aligned merged-region RMSE below 0.001 in native intensity units.
These gates apply to the retained seven-tilt qualification acquisition; they
are not an absolute intensity tolerance for every possible future dataset.
Saved selected DPs must match a float64 NumPy scale/round/restore oracle exactly.
The [2026-09-12 physical Metal run](../../native/Benchmarks/results/2026-09-12-metal/README.md)
records the achieved errors and memory measurements.

## Measuring input preparation

Use `MAPED_LOAD_PASSES=3 maped-native-benchmark INPUT_DIRECTORY REPORT_JSON`
to time each tilt and the seven-tilt total while retaining all seven encoded
inputs. This benchmark uses the existing sequential public loader. Each tilt's
resident-preparation time includes file reading, GPU decompression, median
correction, ANS encoding, and summary generation. Indexing is reported separately;
alignment, merging, saving, and viewing are excluded.

On Phil's Apple M5 Max, the 2026-09-12 repeats took **14.73, 13.69, and 13.80 s**
for all seven inputs. Individual tilts in the later two repeats took
**1.89–2.07 s**. Input residency was **7.008 GiB** and peak Metal allocation was
**7.901 GiB**. Every source was read once. These are repeated loads with an
existing index and uncontrolled OS file cache, not a cold-storage guarantee.
The [per-tilt measurements](../../native/Benchmarks/results/2026-09-12-metal-load/phil-load.json)
also record process footprint and timing boundaries. This loading benchmark
does not establish full-workflow memory or parity qualification on a 24 GiB Mac.

A subsequent [full native diagnostic run](../../native/Benchmarks/results/2026-09-12-metal-load/phil-end-to-end-diagnostic.json)
took **85.71 s** through packed GPU reopening: loading 11.28 s, alignment and
preparation 0.73 s, range pass 29.86 s, second-pass generation 28.80 s,
scaled-uint16 conversion/error measurement 1.30 s, GPU HDF5 compression 5.73 s,
file writing 3.86 s, and packed reopening 4.02 s. Total excludes validation-file
export and UI rendering. The sampled 150,994,944 float32 values matched the
frozen native reference bit-for-bit.

This historical run exposed a performance difference from the earlier
15–16 s merge passes. Restoring separate decode and sampling submissions
still measured 26.85 and 27.27 s per pass, so combining command buffers does
not explain most of the difference. Keep these diagnostic timings distinct
from the earlier measurements; no new speedup or physical 24 GiB qualification
is established by this run.

The subsequent [native Metal optimization](native-maped-processing-performance.md)
reduced the complete workflow to **22.56 and 22.87 s**, with unchanged saved
precision metrics. Use those current measurements; retain the diagnostic run
above as the before baseline.

The latest [IO optimization](native-maped-io-performance.md) removes input read
waits and the output reread using existing infrastructure. It preserves both
precise merge passes; retaining packed output increases peak Metal allocation
to 15.424 GiB on the qualification acquisition.

An [experimental regional-scale single merge](native-maped-regional-storage.md)
reaches complete packed GPU residency in 13.26–13.76 s without saving. It changes
storage precision intentionally and does not change the default global-scale
contract; production readers and formats must support per-region calibration
before adoption.

See the [scaled-storage API proposal](maped-scaled-storage-api-plan.md) for the
experimental regional precision policy and its remaining qualification gates.
