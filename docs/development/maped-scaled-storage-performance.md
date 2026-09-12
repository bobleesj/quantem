# CUDA and Torch MPS full resident scaled output

QuantEM.GPU implementation: `062b7d19` on `main`.

Seven original 512×512 scans with 192×192 detectors, encoded inputs, median
hot-pixel correction and unchanged float32 MAPED stages. The public call is
`merge_datasets(dtype="scaled_uint16", plot_result=False)`, followed by `show()`.
No output file, global range prepass, second merge, or reopening is required.

| Stage | CUDA GPU0 | Torch MPS on Phil |
|---|---:|---:|
| Load seven inputs | 20.59–24.14 s | 10.35–10.74 s |
| Preparation/alignment after loading | 1.58–2.89 s | 3.26–3.32 s |
| Merge, conversion, packing, output summaries | 6.19–10.34 s | 11.79–12.26 s |
| Load through selected DP | 31.91–33.82 s | 25.85–25.87 s |
| Selected resident DP | 0.74–2.94 ms | 0.27–0.47 ms |
| Show4DSTEM constructor | 0.28–0.74 s | 0.24–0.38 s |
| Packed merged output | 6.194 GiB | 6.196 GiB |
| Sampled memory peak | 19.27–20.31 GiB | 17.64–17.79 GiB |

CUDA is the RTX PRO 6000 Blackwell on mjgoat; MPS is Phil's M5 Max. CUDA
memory is the sampled NVML allocation for this process; MPS memory is the
sampled driver allocation reported by Torch. These are not identical accounting
methods or physical 24 GB laptop qualification. Torch allocations were capped
at 24 GiB; external native allocations were included through the stated sampling.
Device services and filesystem caches were not controlled. CUDA loading is
substantially slower than earlier cached measurements; do not quote a 3–4 s
load or extrapolate speedups between these sessions. One additional CUDA run
overlapping small tests is retained but excluded from the headline range.

Each result records exactly one complete float32 generation pass. Reported
storage RMSE was 0.00545180 on CUDA and 0.00544976 on MPS. These compare each
backend's own float32 merge to its stored result. Different MAPED float32
rounding is separate from precision conversion parity on identical inputs.

The shared conversion fixture tests exact uint16 codes and restored float32
values against NumPy on both GPUs, including ties and wide ranges. Calibrated
reads, selected-region and detector crops, BF/mean-DP, center of mass, one-pass
saving, fresh/saved equality, and legacy globally scaled input are covered.
The unchanged MAPED scientific algorithm retains its dense/resident comparison
and median-correction tests. Browser rendering and pointer interaction were not
measured; constructor timing is not an end-to-end browser latency claim.

Evidence and the reproducible public-API benchmark are in
[the benchmark directory](benchmarks/2026-09-12-scaled-storage/).
See the [storage contract](maped-scaled-storage-api-plan.md) for current API,
file metadata and native Swift integration boundaries.
