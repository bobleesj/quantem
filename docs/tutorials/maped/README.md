# MAPED on CPU, MPS, and CUDA

These tutorials build on Henry Bell's Torch implementation in
[QuantEM PR #256](https://github.com/electronmicroscopy/quantem/pull/256).
The additions connect native CPU loading and encoded GPU acquisitions to that
workflow, with bounded merging, saved-result checks, and individual-pattern views.

Three notebooks use the same public synthetic input and the same API:

- `maped_cpu.ipynb`: native QuantEM loading, CPU processing, plotting and saving;
  neither quantem.gpu nor quantem.widget is required.
- `maped_mps.ipynb`: Apple Silicon.
- `maped_cuda.ipynb`: NVIDIA CUDA.

These are **development tutorials**, based on PR #256 plus the loaded-dataset
MAPED work and accompanying IO fixes. They do not yet run against an unchanged
released installation. Use the validated source checkouts in an activated conda
environment; select their `src` directories on `PYTHONPATH` before starting
Jupyter. Do not change device inside the alignment workflow.

```bash
export PYTHONPATH=/path/to/quantem/src
python make_data.py synthetic-tilts
jupyter lab
```

For the GPU notebooks, also include the validated quantem.gpu and quantem.widget
source directories. `quantem.widget[cuda]` and `quantem.widget[mps]` include
the matching `quantem.gpu` extra. Plain `quantem.widget` and its `cpu` extra do
not include it. The CPU notebook uses `Dataset4dstem.from_file` and the native
QuantEM `save`/`load` methods. Package releases must contain the APIs used here
before these become released-package installation instructions.

## Data

The generator adapts the analytic objects and known shifts from the
[validation notebook attached to PR #256](https://github.com/user-attachments/files/29561645/maped_torch_validation.ipynb).
It creates five 64 × 64 scans with 64 × 64 diffraction patterns. It evaluates
shifted Gaussians directly and uses Poisson integer counts instead of the
original additive Gaussian noise. No experimental pixels, acquisition paths,
or microscope metadata are included. Coordinates are pixels; no physical beam
tilt or scattering calibration is claimed.

The fixture checks registration and IO, not experimental MAPED performance.
Its `ground_truth.json` records applied shifts and the PR's 0.2-pixel tolerance.
Raw generated HDF5 files remain outside version control.

A crop estimates alignment from less image context than a full scan. Comparing
it with an earlier merge requires matching the input region, detector corrections,
alignment settings, and output precision. Inspect individual patterns at matching
specimen positions; an average diffraction pattern suppresses local count noise.

## Workflow

Native CPU:

```python
tilts = [Dataset4dstem.from_file(file, "arina") for file in files]
maped = MAPED.from_datasets(tilts)
merged = maped.run(iterations=80, show=False)
maped.show(positions=[(15, 15), (25, 25), (40, 40)], tilt_indices=[0])
```

MPS or CUDA:

```python
tilts = io.load(files, device="cuda:0")
maped = MAPED.from_datasets(tilts)
merged = maped.run(iterations=80, dtype="float32", show=False)
Show4DSTEM(merged)
```

The notebooks expand `run` into its alignment stages so each correction can be
inspected. The small result stays float32. CPU saves through QuantEM as a Zarr
ZIP; GPU notebooks use lossless HDF5 through quantem.gpu.
The CPU comparison marks selected scan positions and shows their individual
diffraction patterns from the first tilt and the merge. Input patterns use
the same registration as the merge; coordinates refer to the cropped result.
For a full series, write exact float32 blocks before reopening the result:

```python
merged = maped.merge(
    dtype="float32", save_to="merged_master.h5", release_tilts=True,
)
io.save("merged.qem", merged)
```

This avoids the 4 GiB in-memory merge limit; the reopened result must still
fit on the device. Saving and reopening include disk IO and are not real-time.
Scaled uint16 storage is quantized and is not used in these tutorials.

Multiple input files load as a list, so `stack=False` is unnecessary. Stacking
would combine acquisitions along an additional tilt axis. MAPED keeps each
acquisition separate until registration. `hot_pixel_correction="zero"` means
replace detector-flagged pixels with zero, not disable correction or zero the
alignment shifts. The GPU default uses median replacement; native CPU Arina
`apply_mask=True` uses zero masking.

To use your own recordings, replace the folder path and use the recorded
calibration. Input corrections are disabled only for this known synthetic
fixture. A real recording must be reviewed before choosing its corrections.
