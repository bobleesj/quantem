---
name: port-torch-scientific-workflows
description: Port or qualify an existing PyTorch scientific workflow on native Metal or a Windows GPU backend, preserving parameter meaning, numerical sensitivity, precision, and resident-data lifecycle.
---

# Port a Torch scientific workflow

Treat the existing Torch implementation as the scientific reference. A port
that matches one default run or copies parameter names is not parameter parity.
Keep application UI integration separate when the user asks for code only.
Windows identifies a platform, not an execution backend; identify the actual
native GPU runtime before claiming Windows qualification.

## Establish the observable contract

- Pin the reference revision before edits. Inventory each existing entry point,
  default, unit, accepted scalar/vector form, conditional dependency, output
  dtype, and supported resident/dense workflow. Read executed code, not only
  docstrings. Classify each parameter as numerically active, presentation-only,
  currently inactive, or explicitly unsupported in the selected workflow.
- Preserve actual scientific behavior. Do not silently activate an unused
  weight, turn correlation padding into a maximum-shift constraint, change
  interpolation coordinates, or widen the resident workflow to a different
  algorithm. Record discrepancies between names/docs and executed semantics.
- Use the existing method and keyword names. Map language-level types without
  introducing a second scientific vocabulary. Scalar broadcasting, per-input
  values, null defaults, and integer coercion require explicit comparison.
- In QuantEM, keep scientific sequencing and weights in QuantEM. Put generic
  residency, codecs, image operations, sampling, precision, and file writing
  in QuantEM.GPU. A native consumer calls public infrastructure operations;
  it does not copy backend kernels into the algorithm or move a named
  scientific workflow into the infrastructure package.

## Measure parameter responses

Keep one repository-owned case manifest shared by the reference and native
test drivers. Both drivers call their public scientific methods. First execute
the true default call without supplying a second copy of its default values.

For each active parameter, test representative low/default/high settings and
branch boundaries. Include Boolean alternatives, scalar/vector forms, coupled
parameters, and meaningful combinations. Gaussian kernel-radius transitions,
Tukey-to-Hann transitions, and coarse-to-refined correlation are examples.
Also test options expected to have no effect and options expected to fail.

Compare both outputs and changes from a baseline. For numeric sweeps, retain
finite-difference or secant responses with units and intervals. Do not assume
alignment sensitivity is monotonic or differentiable: peak selection and
integer geometry can jump. Near zero response, use an absolute bound rather
than a relative error with a tiny denominator. A difference-of-two-outputs
bound may be derived from the already fixed pointwise tolerance; never choose
a new tolerance after inspecting failures merely to get a pass.

Test A → B → A using the same resident data and compare the final A with both
the first A and a fresh run. Check source-read counts, unchanged encoded
inputs, downstream result invalidation, ownership, and release after saving.
Parameter changes must not silently show an old result.

## Separate precision claims

- Exact integer counts, masks, codec reconstruction, and declared integer
  reductions require exact equality.
- Compare float32/complex64 intermediate images, shifts, and merged values
  with explicit scientific tolerances. A shift tolerance alone does not
  establish intensity parity.
- Distinguish execution error from optional storage precision loss. Test
  float32-to-scaled-uint16 codes and restoration against an independent
  float64 NumPy oracle on small test arrays, including midpoint rounding,
  signed ranges, zero range, small values, clipping, and saved coefficients.
- Freeze fixtures and existing gates. Reproduce failures twice and inspect
  dtype changes, reduction order, FFT scaling, tie handling, normalized-grid
  construction, and fused arithmetic before changing a kernel. Do not alter
  Torch's scientific implementation or frozen expectations to match a port.
- Preserve scalar decision precision separately from array precision. A Python
  float can select a different Gaussian radius or padding ceiling just below
  a boundary than a prematurely rounded Swift Float. Use Double for these
  choices and test both sides before converting numerical buffers to float32.
- Equivalent formulas are not always equivalent float32 programs. Check full
  two-dimensional versus separable convolution, normalization reductions,
  complex versus real-input FFT plans, matrix-multiply accumulation, and
  interpolation FMA order. Keep fixes inside generic GPU infrastructure.
- Keep production array computation on the requested GPU. Disable automatic
  CPU fallback. NumPy may be an explicitly identified test oracle; it is not
  a production fallback.

## Reproducibility and qualification

Persist the actual scientific parameter settings with the result, including
resolved per-input settings and shift arrays. Reopen through the existing
public loader and compare restored values and provenance.

After correctness, measure a real workflow and selected parameter extremes.
Report hardware, backend, dtype, shape, source revision, cache state, timing
boundaries, and both resident and peak memory. A fast selected-region preview
does not establish fast full export. One low-memory default run does not prove
every advanced setting fits the same memory budget.

Keep implementation, evidence, and platform claims separate. A shared test
driver that supports CUDA is not evidence that it ran on Windows. List
unexecuted platforms and unsupported options directly. Retain compact case
results and reproduction commands with the repository; keep large private
data and transient outputs out of public commits.

The MAPED example lives in the QuantEM repository under
native/Tests/parameter_cases.json, its native hardware runner under
native/Benchmarks/MAPEDParameterParity, and its Torch comparison under
native/Tests/compare_parameters.py. These are examples of the discipline,
not universal parameter ranges or tolerances for other algorithms.
