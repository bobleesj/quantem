# Native Metal qualification, 2026-09-12

This record covers seven real experimental acquisitions, each shaped
`(512, 512, 192, 192)`, on an Apple M5 Max with 40 GPU cores and 128 GB unified
memory. Native runtime: Swift 6.3.3, release build, macOS 26. The source identity
and original scientific files remain private; no source filenames or images
are included here.

QuantEM.GPU infrastructure revision: `7e5a95293252f47681d699faeaa08e402a1440c3`.
The unchanged Python MAPED scientific reference is from QuantEM `2fea14cb`.
The QuantEM algorithm, test scripts, and frozen observations are committed
together with this record. Defensive argument guards added after the timed
run do not change the measured valid-input computation; the final native
and Python MPS test suites also passed with those guards present.

- [Native load → align → merge → save → full packed reopen](native.json)
- [Current Torch MPS and NumPy parity observations](torch-parity.json)
- [Reproduction commands and supported API](../../../README.md)

The complete native workflow took **48.3356 seconds**. A separately timed
0.5736-second export of bounded validation observations is excluded. Process
wall time including those exports and executable setup was 48.95 seconds.
GPU operations synchronize before timings are read. Compression and writing
subtimings are nested inside `merge_write`; do not add them twice.

The seven inputs were read once each. Two merge passes operate on encoded
GPU residency: one measures a single global range; the other recomputes
float32 regions, converts them to scaled uint16, measures restored-value
errors, and writes standard compressed HDF5. Owned inputs are released before
the full packed output is reopened. No Python or CPU scientific fallback is
used by the native executable. CPU work handles orchestration, file I/O,
metadata, and scalar scale coefficients.

Peak Metal allocation: **9,429,860,352 bytes**. `/usr/bin/time -l` reported
**10,834,382,976 bytes** peak process footprint and **7,280,934,912 bytes**
maximum resident set size. These metrics overlap. Zero swaps were reported.
The run fits the requested 24 GB allocation budget, but this is not a physical
24 GB machine qualification. Filesystem cache state was not forced cold.

Every diffraction/bright-field summary pixel and every origin is identical
to current `MAPEDTorch` on MPS. Shift differences are below 0.01 pixel. The
same-shift merged region covers eight scan rows (150,994,944 values), passes
`rtol=3e-6, atol=2e-5`, and has RMSE `6.1732783e-7`. Separate native/Torch
alignment gives merged-region RMSE `0.0006570862`, below the fixed 0.001 gate.
This is bounded-region floating-point parity, not a comparison of all merged
float32 values. Conversion metrics themselves cover **every saved value**.

Three selected DPs from the native HDF5 were reopened with the existing Python
GPU loader and checked exactly against an independent float64 NumPy
scale/round/restore oracle. Native unit tests separately cover exact integer
count reconstruction, median correction, NumPy means/Gaussian filtering,
normalized-grid boundaries, FFT/correlation, uint16 rounding, restored
intensities, coefficient persistence, and packed HDF5 reopening.

Validation also passed the Python MPS precision/encoded-resident tests
(10 tests), QuantEM's backend-boundary tests (3), and QuantEM.GPU backend
status/resource tests (30). The native suite contains 5 tests. This qualifies
the tested bilinear, zero-padding, scaled-uint16 resident workflow. Native
float16 export, other detector layouts, app integration, and iOS performance
are not covered. No claim is made that floating-point alignment is bit-identical
across backends or that the current measurement is a controlled speed ratio
against historical runs.
