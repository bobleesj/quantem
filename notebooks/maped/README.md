# MAPED on CUDA

Open [maped_cuda.ipynb](maped_cuda.ipynb) for an existing HDF5 tilt series.
Set `folder` and a new `output` filename, then run the cells in order. The
example inspection positions assume 512 × 512 scans.

Use an activated conda environment with compatible development builds of
`quantem.gpu` and `quantem.widget`. From this repository's root:

```bash
python -m pip install -e .
python -m jupyterlab notebooks/maped/maped_cuda.ipynb
```

If the GPU loader and viewer are missing, install the widget extra from
TestPyPI; it includes the GPU package:

```bash
python -m pip install --upgrade --pre -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "quantem.widget[cuda]" pandas jupyterlab
```

These commands do not replace the development-build prerequisite with a
release-only guarantee. Restart the kernel after changing packages.

## One workflow across the packages

| Package | Responsibility |
| --- | --- |
| `quantem.gpu` | Load HDF5, retain encoded data on the GPU, and save/reopen results. |
| `quantem.diffraction.MAPED` | Align diffraction and specimen images, merge, and report shifts. |
| `quantem.widget` | Inspect the loaded or merged data with `Show4DSTEM`. |
| `quantem.live` | Orchestrate acquisitions and dashboard workflows using the shared loaders. |

```python
maped = MAPED.from_datasets(tilts)
maped.align_diffraction(show=False)
maped.align_images(iterations=160, show=False)
merged = maped.merge(crop=True)
```

`maped.run(iterations=160)` performs the alignment and merge together.
The notebook uses bounded scaled-uint16 HDF5 export for the full-size series;
its rounding error and clipping are reported. GPU inputs are released after
the merge, so reload them before merging again. Native CPU datasets remain
supported by MAPED without GPU or widget dependencies.

The implementation preserves Henry Bell's MAPED contribution and the
MAPED-live updates. The notebook follows the
[MAPED tutorials](https://github.com/bobleesj/quantem-tutorials/tree/maped/tutorials/diffraction/maped).
Private data, executed outputs, and local validation records are not included.
