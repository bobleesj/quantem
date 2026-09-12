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
until both passes complete; release owned inputs before packed reopening.

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
