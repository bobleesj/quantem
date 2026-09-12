# MAPED on a 24 GiB GPU

The practical question is whether all seven tilts can remain available while
MAPED aligns and merges them without allocating the complete float32 result.
The answer is yes for the qualified CUDA workflow: encoded inputs occupy about
7.76 GiB, and bounded regions keep the measured process peak near 11.15 GiB.

Start with the [interactive MAPED tutorial](maped_interactive.html) if you want
to see how the alignment, region size, memory, and output precision relate.

## What stays on the GPU?

`MAPEDTorch.from_files` loads each source once into the encoded resident
representation. Hot pixels are corrected with the GPU median path during that
load. All seven sources then remain available while MAPED computes summaries,
alignment shifts, output range, and bounded merge regions.

```python
maped = MAPEDTorch.from_files(files, device="cuda:0")
maped.preprocess(plot_summary=False)
maped.diffraction_origin(sigma=1, plot_origins=False)
maped.diffraction_align(edge_blend=2, plot_aligned=False)
maped.real_space_align(...)
merged = maped.merge_datasets(
    save_to="merged_master.h5",
    plot_result=False,
)
viewer = maped.show()
viewer
```

The public sequence is the same on CUDA and Torch MPS. Encoded residency,
bounded reads, and output packing are infrastructure choices selected beneath
this API.

## Why are there two merge passes?

One global scale must describe every output value. MAPED therefore makes a
range pass to find the global minimum and maximum, followed by a bounded
merge-and-write pass that transforms each region directly to uint16. It never
allocates the complete merged float32 tensor.

For a value `x`, the saved code is approximately

```text
code = round((x - offset) / scale)
restored = offset + scale * code
```

The HDF5 metadata stores `offset` and `scale`. Show4DSTEM can request selected
regions and restore their physical values without expanding the whole file in
memory. The precision report records RMSE, maximum absolute error, and clipped
values so the display representation remains auditable.

## Qualified seven-tilt CUDA reference

The reference dataset has shape `(512, 512, 192, 192)` with seven tilts and
9,663,676,416 merged values.

| Measurement | Result |
|---|---:|
| Encoded inputs | 7.756 GiB |
| Process GPU peak | 11.146 GiB |
| Total GPU0 peak | 12.076 GiB |
| Isolated end-to-end time | 31.39 s |
| Merge, save, and reopen | 20.78 s |
| GPU uint16 encoding | 0.176 s |
| Packed reopen | 2.53 s |
| Show4DSTEM construction | 0.249 s |
| Selected diffraction pattern | 0.36 ms |
| RMSE | 0.00695145 |
| Maximum absolute error | 0.0131836 |
| Clipped values | 0 |

The executable notebook includes a full smoke run. Its latest measured phases
were 10.21 s for encoded loading, 2.03 s for alignment, and 30.90 s for merge,
save, and reopen. Notebook overhead puts that complete run near 43-45 s.

## What changes on a 24 GiB Mac?

The Python API and scientific sequence stay the same with `device="mps"`.
Torch MPS uses unified memory, so the OS and applications share the 24 GiB
budget. The current seven-tilt reference measured 6.998 GiB of encoded inputs
and a 16.387 GiB MPS driver peak. It remains within 24 GiB, although the current
87.16 s end-to-end reference is slower than CUDA and still needs optimization.

The scientist should not choose a private CUDA or Metal implementation. MAPED
owns the scientific sequence; QuantEM.GPU owns encoded residency, regional
access, reductions, scaled output, and backend execution.

## Where are the runnable files?

- [Interactive tutorial](maped_interactive.html)
- [CUDA notebook](../../notebooks/maped/cuda_maped_merge%20\(1\).ipynb)
- [Selected diffraction patterns](../../notebooks/maped/selected_diffraction_patterns.png)

Keep these files together in the QuantEM repository so the documentation,
notebook, figures, and implementation evolve on the same branch.
