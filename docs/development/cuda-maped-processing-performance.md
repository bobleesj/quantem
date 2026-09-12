# CUDA MAPED processing before export

The notebook uses the same `MAPEDTorch` sequence on CUDA and MPS. Change
`device="cuda:0"` to `device="mps"`; representation and region scheduling stay
automatic. Encoded inputs use the ANS count codec. Scientific processing stays
in Torch inside QuantEM. This optimization makes no QuantEM.GPU changes.

## Notebook API

```python
from pathlib import Path
from quantem.diffraction import MAPEDTorch

files = sorted(Path("input-directory").glob("*_master.h5"))
maped = MAPEDTorch.from_files(files, device="cuda:0")
maped.preprocess(plot_summary=False)
maped.diffraction_origin(sigma=1, plot_origins=False)
maped.diffraction_align(edge_blend=2, plot_aligned=False)
maped.real_space_align(
    num_iter=20, hanning_filter=True, padding=2, edge_blend=5,
    pad_val="median", shift_method="bilinear", plot_aligned=False,
)

# Inspect without saving. Coordinates have exclusive stops.
patch = maped.merge_datasets(
    scan_region=(252, 260, 252, 260), plot_result=False,
)
patch_viewer = maped.show()
patch_viewer
```

All seven inputs remain available for another region. Inspection returns exact
float32 with the complete detector, up to 4096 scan positions. It does not
materialize the whole acquisition. When the full packed scaled-uint16 result
is wanted, use the existing optional export path:

```python
merged = maped.merge_datasets(save_to="merged_master.h5", plot_result=False)
viewer = maped.show()
viewer
```

That path finds one global scale, saves bounded regions, releases owned inputs,
and reopens packed. It remains the current full-acquisition viewing workflow.
A complete unsaved scaled resident viewer is not implemented by this change.
Close both viewers, then `maped.close()` when finished.

The [canonical notebook](../../notebooks/maped/cuda_maped_merge%20%281%29.ipynb)
now includes inspection before its optional full export. Use the current
QuantEM and QuantEM.GPU checkouts in the notebook kernel environment. A shell's
relative `PYTHONPATH=src` does not select that checkout after a kernel changes
its working directory; configure absolute source paths or install the checkout.

## Measured processing

Physical GPU0: NVIDIA RTX PRO 6000 Blackwell, Torch 2.10.0+cu130. Seven original
512x512x192x192 inputs, median hot-pixel correction, no binning, one source-file
read each, all seven encoded residents present throughout processing. Torch
allocations were capped at 24 GiB; NVML also measured the complete process,
including non-Torch allocations. Desktop and an unrelated resident service
were present and left alone. Cache state was not controlled.

| Stage | Original | Optimized |
|---|---:|---:|
| Preprocessing and alignment after loading | 1.72 s | 1.66–1.68 s |
| Complete float32 merge + BF + mean DP + global range | 6.12–6.20 s | 5.72–5.92 s |
| Paired merge alone, first audit | 6.270 s | 5.788 s |
| Paired merge alone, repeat audit | 6.277 s | 5.834 s |
| Newly merged selected DP | 15.7 ms | 13.8–16.4 ms |
| Selected 8x8 patch | 104 ms | 103–116 ms |
| Show4DSTEM construction for patch | 233 ms | 204–208 ms |

The paired full-merge improvement is 7.1–7.7%. Overview timings are three
passes in each of two optimized sessions. Selected-region timings do not
establish a patch speedup. The complete overview is a benchmark experiment,
not an additional public method. Browser rendering and pointer interactions
are outside these backend/constructor timings.

Loading took 8.46–13.14 s, so 3–4 s loading remains a target. Excluding loading,
the current processing reaches a selected DP in about 1.7 s and a complete
overview in about 7.4–7.6 s. If loading later reaches 3–4 s, those estimates
become about 4.7–5.7 s and 10.4–11.6 s respectively, before browser rendering.
The complete globally scaled uint16 resident view needs additional preparation;
these one-pass overview times do not include its second merge/conversion pass.

Inputs occupy 7.76 GiB. Peak process allocation was 12.66 GiB for the first
optimized processing run and 13.21 GiB for the repeat with paired parity.
These measurements satisfy the budget on a 96 GB GPU; they are not a physical
24 GB laptop performance test. They are also not CUDA/MPS numerical identity
or identical-hardware speed comparisons.

## Why this helps and what was rejected

The eager scan interpolation previously zeroed the entire float32 workspace,
then read those zeros while adding its first contribution. It now writes the
first valid contribution directly with `torch.mul(..., out=...)` and zeros only
uncovered borders. Remaining additions keep their original order. This removes
one full zero-fill and the first contribution's unnecessary output read.
Empty overlaps still produce all zeros. MPS's compiled interior remains
unchanged; its eager boundary and inspection paths use the same improvement.

Two independent full CUDA audits each compared all **9,663,676,416 float32
values** against frozen QuantEM revision `a71d41fd`, with exact equality.
Global extrema and full BF/DP summary scalars also matched at the same region
size. Storage precision rules and conversion code are unchanged.

The MPS compiled four-tap form was also tried on CUDA. It failed exact parity
on full-range uint16 synthetic inputs (maximum merged error 0.01171875).
Disabling compiler floating-point fusion did not repair that trial. It was
rejected, and CUDA retains eager interpolation. No tolerance was widened.
Reducing regions from 4096 to 2048 frames took 6.33–6.39 s and changed the mean
DP reduction's rounding order; this alternative was not promoted either.

## Reproduce and evidence

[Retained JSON records](benchmarks/2026-09-12-cuda-maped-processing/) contain
baseline, optimized, repeated, all-values parity, and smaller-region trials.
`tests/diffraction/benchmark_maped_cuda.py INPUT_DIRECTORY OUTPUT_DIRECTORY`
measures loading, alignment, inspection, and three complete overview/range
passes. Add `--reference FROZEN_MAPED_RESIDENT.py` for alternating-order,
synchronized, all-values comparison after those timings. Freeze the reference
from `a71d41fd`; do not regenerate it from the candidate.

Regression tests include ordered-tap interpolation with border, fractional,
integer, and empty overlaps on both GPUs, plus resident save/inspection and
backend-boundary checks. No handwritten CUDA, Metal, or CuPy code was added to
MAPED. CUDA already shares decoded buffers with Torch through DLPack; this
change does not claim to transfer native allocation ownership to Torch.

The updated notebook also executed through real Jupyter kernel cells, including
inspection, complete scaled-uint16 save/reopen, and cleanup: load 9.21 s,
alignment 1.81 s, patch 0.225 s, export/reopen 33.77 s. Its conversion report
was RMSE 0.006954265 and maximum error 0.01318359, with zero clipping. These
notebook timings are separate from the processing-only benchmarks above.
The first test launch selected an older installed QuantEM because of a relative
source path; the successful run used absolute checkout paths. No browser
gestures or frontend frame rate were verified by the kernel execution.

Checks: CUDA 7 passed/2 skipped; MPS resident/preview/boundary 9 passed/1 skipped,
plus MPS ordered-tap parity 1 passed/1 skipped. Skips select the other hardware.
