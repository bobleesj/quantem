# Native Metal MAPED parameter qualification

Physical Apple M5 Max, 40 GPU cores, 128 GB unified memory, 2026-09-12.
The acquisition has seven 512×512 scans with 192×192 detector pixels. All seven
corrected native-count inputs stay encoded on the GPU during alignment and both
merge passes. No Python or Torch runtime is used by the native implementation.

The [shared manifest](../../../Tests/parameter_cases.json) exercises **75 public
parameter cases**, **45 shift/intensity sensitivity intervals**, and **seven
unsupported merge choices**. Every check passed, including actual default
calls, returning to defaults, low/high values, conditional settings, scalar and
per-tilt forms, numeric padding values, and values immediately on either side
of Gaussian-radius and padding-ceiling boundaries.

| Measurement | Observed |
| --- | ---: |
| Diffraction shifts, all cases | bit-identical to Torch MPS |
| Real-space shifts, all cases | bit-identical to Torch MPS |
| Origins | exact |
| Worst sampled float32 merged-DP RMSE | 1.46462696e-6 |
| Worst same-shift absolute intensity error | 0.00048828125 |
| Peak Metal allocation during parameter sweep | 7.893 GiB |
| Input reads | one per tilt |

Intensity differences pass the original combined `rtol=3e-6, atol=2e-5` gate;
maximum absolute intensity error alone is not the allclose criterion. Nine
sampled DPs span scan corners, boundaries, and the center. Response and secant
bounds derive from the unchanged pointwise gates. See [parameters.json](parameters.json)
for every comparison and sensitivity interval, [native.json](native.json) for
actual native settings and shifts, and [validation.json](validation.json) for
native rejection messages. This finite test matrix does not prove equality
for every acquisition or continuous parameter value.

## Full export and reopen

The [end-to-end record](end-to-end.json) uses the established seven-tilt settings
(`sigma=1`, diffraction `edge_blend=2`, real-space `num_iter=20`, `edge_blend=5`,
`padding=2`, `hanning_filter=true`). Source reads and exact summaries, alignment,
both float32 merge passes, scaled uint16 conversion, compressed HDF5 saving,
and packed reopening are included. Bounded validation exports are timed
separately and excluded from the workflow total.

| Stage | Seconds |
| --- | ---: |
| Load, median correction, encoding and summaries | 11.271 |
| Diffraction alignment | 0.040 |
| Real-space alignment | 0.642 |
| Merge to measure the global range | 21.772 |
| Merge, convert, compress and write | 30.554 |
| Reopen complete packed output | 3.218 |
| **Total workflow** | **67.563** |

The write stage includes 20.602 s of merge generation, 1.034 s of GPU conversion
and precision measurement, 4.602 s of compression, and 4.313 s of HDF5 writing.
These are included components, not extra time. Native alignment takes 0.682 s;
the separate Torch MPS comparison measured 0.591 s. A bounded merge region took
0.492 s natively and 0.579 s in Torch in this run.

Full-workflow peak Metal allocation was **8.813 GiB**; `/usr/bin/time -l`
reported **10.128 GiB** peak process footprint. Encoded inputs occupied 7.008 GiB
and the final packed output occupied 5.679 GiB. Owned inputs were released
before reopening. These observed footprints fit within 24 GB; the hardware
used for testing has 128 GB. No physical 24 GB Mac qualification is claimed.

The [independent comparison](end-to-end-parity.json) found exact BF/DP means,
origins and both shift arrays. Both same-shift and independently aligned
center-region RMSE were 6.23964581e-7. Three saved DPs matched the float64 NumPy
scale/round/restore oracle exactly through the Python loader. Whole-output
scaled-storage RMSE was 0.00695111, maximum error 0.01312256, with zero clipping
or overflow. This storage error is separate from native float32 execution error.

This measurement followed the qualification workload. It is not a controlled
cold/warm comparison with the earlier 48.34 s run retained in the neighboring
historical result directory. Do not infer a full-workflow speedup from these
records.

## Changes and reproduction

The native port now preserves Torch's full 2D Gaussian normalization, complex
FFT input path, matrix-based local DFT, reduction order, periodic windows,
phase blending, and bilinear FMA order. Generic numerical implementations live
in QuantEM.GPU. MAPED retains the scientific sequence and existing keyword names.
The Torch reference only received a GPU-device fix for manual-origin coordinate
grids and removal of an unused CPU allocation; its scientific arithmetic was
not changed to match the port.

The native tests check exact count/median and NumPy storage oracles, frozen
Torch intermediate samples, parameter mutation and A → B → A, saved provenance,
failed-save preservation, result invalidation, and borrowed-source ownership.
The complete executed matrix and source hashes are recorded in
[qualification.json](qualification.json). See the [parameter guide](../../../../docs/development/native-maped-parameters.md)
for commands and the reusable porting skill. No application UI integration or
Windows GPU execution is included in this qualification.
