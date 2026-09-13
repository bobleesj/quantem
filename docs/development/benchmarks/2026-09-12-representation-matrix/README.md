# Seven-tilt input representation qualification

Both `encoded` (ANS, default) and `packed` are supported resident-input contracts.
Input representation is separate from `dtype="scaled_uint16"` for display.
No MAPED production API or scientific arithmetic was changed in this audit.

| Backend | ANS merge/display | Packed merge/display | Qualification |
| --- | ---: | ---: | --- |
| CUDA, RTX PRO 6000 GPU0 | 5.24 s | 4.98 s | Complete real-data workflow; parity below |
| Torch MPS, the Apple M5 Max test host | 19.80 s | unavailable through ordinary HDF5 load | Prepared packed sources only; public load gap remains |
| Native Metal, the Apple M5 Max test host | 9.29–14.90 s | 13.52 s | Complete real-data workflow; qualification limits below |

Timings exclude loading, preprocessing, alignment, saving and UI rendering.
Native ANS measurements are in `native/Benchmarks/results/2026-09-12-metal-resident`.
These are single-run measurements (two native ANS runs), not a statistically
controlled cross-hardware performance guarantee. The CUDA and Metal test hardware differs.

CUDA load-through-DP took 14.12 s for ANS on the repeated run and 12.92 s for
packed. Native packed took 32.49 s including 7.17 s loading and approximately
11.8 s preprocessing/alignment; native ANS took 18–25 s total. Correcting packed
pixels during reads and computing summaries after loading remain extra costs.
Native packed input residency was 11.23 GiB and sampled peak Metal allocation
19.29 GiB, versus ANS inputs 7.01 GiB and peak 14.76 GiB. This does not qualify
packed operation on a physical 16 GB Mac.

## Bugs found and fixed

CUDA's compact detector adapter reapplied the original stored pixel mask to
already median-corrected packed counts. It now honors the correction record;
explicit detector exclusions remain in force. The new real-data GPU regression
is `quantem.gpu/tests/hardware/cuda/test_corrected_packed_summaries.py`.

Native original-HDF5 packed loading preserved raw counts but lost original
hot-pixel provenance because its active detector exclusion mask was empty.
That provenance is now retained separately, so the existing generic median
wrapper corrects bounded reads without changing the caller-owned storage.
Twelve native tests passed after the fix.

## Numerical evidence

CUDA packed and ANS agree bit-for-bit for one sampled DP from each input,
every mean-DP and bright-field summary value, both shift arrays, and the
selected scaled-display DP. The initial failed comparison and diagnosis are
not treated as passing results. No baseline or tolerance was relaxed.

Native packed and ANS shift arrays were compared exactly on MPS and matched;
whole-output display RMSE was 0.005449763 for both. Existing native unit tests
cover both count types. This is not a new full-volume bitwise native merge
comparison or a 75-case packed sensitivity sweep; the earlier 75-case/45-interval
qualification used ANS. Those expanded packed qualifications remain outstanding.

## Remaining MPS work

`MAPEDTorch.from_resident` accepts prepared packed sources, but
`io.load(h5, backend="mps", representation="packed")` raises NotImplementedError.
Reuse generic GPU packing/loading infrastructure to wire that path, preserving
median correction and source ownership. Do not solve this by silently returning
ANS, by adding MAPED science to QuantEM.GPU, or by using CPU encoding. Until it
is implemented and tested, do not advertise equivalent ordinary-file loading
or matched performance across all six combinations.

Use the benchmark `run.py --device cuda:0|mps --representation encoded|packed`
with the seven-input folder and a new report path. Run encoded first to create
the GPU parity reference. Native benchmarking uses `MAPED_NATIVE_DISPLAY=1`
and `MAPED_INPUT_REPRESENTATION=encoded|packed`. Outputs remain ANS-scaled views.
