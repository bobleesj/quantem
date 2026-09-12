# MAPED parameter parity

The reference is `MAPEDTorch` with encoded resident inputs and plotting disabled.
The native methods keep its scientific keyword names. Swift uses `Double` for
scalar choices that determine integer kernel widths or padding, then float32
GPU arrays for the calculation. This matters just above and below a rounding
boundary. Both backends use `(row, column)` coordinates.

## Active and inactive controls

| Method / parameter | Default | Actual reference behavior |
| --- | --- | --- |
| `preprocess(scale)` | one per tilt | Scalar or per-tilt nonzero factors for summary display; does not scale input counts or alignment. |
| `diffraction_origin(origins)` | automatic | One integer pair broadcast to all tilts, or one pair per tilt. Chooses recorded origins. |
| `diffraction_origin(sigma)` | none | Gaussian sigma in detector pixels before automatic peak finding; nonpositive values bypass filtering. Does not change downstream alignment in the current reference. |
| `diffraction_align(edge_blend)` | 16 | Detector-pixel Tukey taper; reaches periodic Hann at half the detector dimension. |
| `diffraction_align(upsample_factor)` | 100 | Local correlation refinement; factors 1 and 2 use the existing half-pixel coarse branch, greater factors use a local DFT. |
| `diffraction_align(weight_scale)` | 0.125 | Currently calculated but not applied to correlation. Changing it has no numerical effect. |
| `diffraction_align(padding, pad_val)` | none, `min` | Presentation only; do not affect native scientific arrays. `pad_val` accepts a named statistic or a number. |
| `real_space_align(num_images)` | all | Align the first requested number, capped at the tilt count. Unselected tilts retain zero shifts. Must be positive. |
| `real_space_align(num_iter)` | 3 | Positive count of correlation refinements; recomputes alignment from zero on each public call. |
| `real_space_align(edge_blend)` | 1 | Without `max_shift`, correlation padding is `ceil(edge_blend) + 4` scan pixels. |
| `real_space_align(max_shift)` | none | Overrides padding with `ceil(max_shift) + 4`; **does not limit the correlation peak search**. |
| `real_space_align(upsample_factor)` | 100 | Same coarse/refined correlation convention as diffraction alignment. |
| `real_space_align(edge_filter)` | true | Apply reflected Sobel gradients, Gaussian smoothing, then gradient magnitude. |
| `real_space_align(edge_sigma)` | 2 | Scan-pixel Gaussian sigma; width `2 * int(2 * sigma) + 1`. Must be positive when filtering is enabled; ignored when disabled. |
| `real_space_align(hanning_filter)` | false | Apply a periodic Hann window during mean centering and correlation. |
| `real_space_align(padding, pad_val, shift_method)` | none, `median`, `bilinear` | Presentation only. The scientific alignment still uses the established bilinear interpolation. |

An application should expose only numerically active settings as scientific
sliders. It must not imply that an inactive weight controls alignment or that
`max_shift` constrains the displacement. Discrete counts require integer
controls. Sigma and padding can cause discrete changes at kernel-size or
ceiling boundaries; smooth slider motion does not imply a smooth result.

The native resident merge has the same supported subset as the encoded Torch
resident merge:

| Parameter | Supported value |
| --- | --- |
| `real_space_padding`, `diffraction_padding` | 0 |
| `real_space_edge_blend` | 1 |
| `diffraction_edge_blend` | 0 |
| `shift_method` | `bilinear`, ignoring surrounding whitespace and case |
| `dtype` | omitted or `scaled_uint16` |
| `scale_output` | false |
| `diffraction_pad_val` | retained in provenance, inactive with zero padding |
| `save_to` | a new HDF5 output path |

Other scientific merge choices raise an error in both resident workflows.
Swift represents named or numeric padding with the literal-convertible
`MAPEDPadValue` type, so both `pad_val: "median"` and `pad_val: 0.5` retain
the existing keyword.

Torch's separate dense workflow, plotting options, and Torch-specific execution
hints are not additional native scientific controls. Native device allocation
limits remain real limits: very large padding, filtering kernels, or refinement
matrices cannot be made unlimited by an API. The native implementation checks
allocation limits instead of imposing an arbitrary refinement-factor ceiling.

## Reproduce the qualification

One [case manifest](../../native/Tests/parameter_cases.json) drives both public
workflows. It includes defaults, low/high choices, coarse/refined branches,
Gaussian and padding boundaries, manual origins, first-N tilts, inactive
settings, coupled settings, and returning to defaults on the same seven inputs.
Nine output DPs include scan corners, edges, and the center. The drivers retain
pointwise errors, response differences relative to defaults, and secant
sensitivities for both shifts and intensities. Alignment is not assumed to be
monotonic or differentiable.

```bash
export QUANTEM_GPU_PACKAGE=/path/to/quantem.gpu
swift run -c release maped-parameter-parity \
  /path/to/seven/tilts native/Tests/parameter_cases.json /path/to/new/run
PYTHONPATH=src:/path/to/quantem.gpu/src python native/Tests/compare_parameters.py \
  /path/to/seven/tilts native/Tests/parameter_cases.json /path/to/new/run
swift test -c release --filter MAPEDNativeTests
```

Fixed gates are the same as the [native contract](native-maped-contract.md):
exact origins, shifts within 0.01 pixel, same-shift merge `rtol=3e-6, atol=2e-5`,
and independently aligned sampled-DP RMSE below 0.001 intensity units. A response
subtracts two outputs, so its bound is twice the corresponding pointwise bound;
a secant divides that bound by the parameter interval. No gate is fitted after
observing a failure. These tolerances qualify the retained acquisition and
case matrix, not every possible dataset or point on a continuous slider.

The native tests additionally check scalar broadcasting, A → B → A equality,
borrowed input lifetime, result invalidation, repeated saving, saved scientific
settings, exact counts/median correction, NumPy precision oracles, and frozen
Torch intermediate samples. Float32 execution parity and optional scaled-uint16
storage error are separate measurements. The saved merge records all stage
settings alongside both shift arrays and the precision report.

The reusable [Torch porting skill](skills/port-torch-scientific-workflows/SKILL.md)
requires this discipline for native Metal and future Windows GPU work. Windows
is a platform; qualification must identify and execute its actual GPU backend.
No Windows hardware qualification is implied by these Metal tests.


## Measured qualification

The [retained physical Metal run](../../native/Benchmarks/results/2026-09-12-metal-parameters/README.md)
passed all 75 cases, 45 shift/intensity sensitivity intervals, and seven rejected
merge options on the seven-tilt acquisition. Both diffraction and real-space
shifts matched Torch MPS bit-for-bit for every case. Worst sampled merged-DP
RMSE was 1.47e-6 with unchanged gates. Native allocation peaked at 7.90 GiB across
the sweep, and each source was read exactly once. These rounded figures are
backed by the complete per-case JSON records.
