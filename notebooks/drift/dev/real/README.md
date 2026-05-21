# Drift Forward-Model Development Notes

This folder contains the active real-data known-drift workflow. Use the two
unnumbered notebooks here for interactive development runs:

1. `known_4dstem_scan_drift_forward_model.ipynb`
2. `known_4dstem_scan_drift_locked_ssb.ipynb`

The old numbered notebooks have been moved to `archive/` as development
history.

## Forward Model

The forward model applies drift on the scan axes:

- detector pixels inside a diffraction pattern are not rolled or warped
- each output scan pixel samples the clean 4D-STEM dataset at a drifted probe
  position
- subpixel drift uses interpolation over neighboring scan positions, so an
  output diffraction pattern can be a weighted mix of nearby clean diffraction
  patterns

For BTO_18, the known drift vectors are pure right drift in shared specimen
coordinates:

```text
right30: [0, 30] px
right60: [0, 60] px
right90: [0, 90] px
```


## Code Organization

The QuantEM drift module is the source of truth for the forward model and
metadata contract. The notebooks/scripts in this folder should stay thin:
load the real 4D-STEM master, call QuantEM drift helpers, export H5 files, and
optionally run downstream diagnostics.

Canonical pieces now live in `src/quantem/imaging/`:

- `drift_simulation.py`: scan-direction geometry, scan-time drift fields,
  scan-axis forward simulation, valid-position masks, and crop selection
- `drift_io.py`: `entry/quantem/drift` H5 metadata read/write helpers
- `drift.py` / `drift_4dstem.py`: correction workflows, separate from the
  known-drift forward model

The export H5 files are the bridge to any later tool, including
`quantem.live`. `quantem.live` should not define the drift vector convention;
it should only consume the QuantEM metadata when we add that bridge.

## Current BTO_18 Artifacts

Raw exports live on the SSD:

```text
/home/owner/ssd/data/dasol/20260415_BTOSTO/quantem/drift/real/
  BTO_18_known_right30_crop400_detbin2_u16/
  BTO_18_known_right60_crop400_detbin2_u16/
  BTO_18_known_right90_crop400_detbin2_u16/
```

Each export directory contains:

```text
BTO_18_ground_truth_crop400_detbin2_master.h5
BTO_18_rightXX_image_0_crop400_detbin2_master.h5
BTO_18_rightXX_image_1_crop400_detbin2_master.h5
```

Each master has drift metadata under `entry/quantem/drift/`:

```text
probe_positions_px
positions_offset_px
known_drift_total_px_down_right
scan_crop_rows
scan_crop_cols
```

The locked regular-raster SSB outputs live under:

```text
/home/owner/repos/quantem/notebooks/drift/dev/outputs/
  real_14_bto18_known_right_drift_live_ssb_baseline/
  real_bto18_known_right60_live_ssb_locked/
  real_bto18_known_right90_live_ssb_locked/
```

Important SSB convention:

- clean0 supplies the microscope calibration
- drift0 keeps the same `C10/C12/phi12` and same rotation
- drift90 keeps the same `C10/C12/phi12`, tries `clean0 - 90` and
  `clean0 + 90`, and picks the lower-loss branch
- 90-degree phases are rotated into the clean0 display frame with
  `np.rot90(phase, k=-1)` for figures and comparisons

## Active Workflow Notebooks

The active workflow uses only two unnumbered notebooks in this folder:

1. `known_4dstem_scan_drift_forward_model.ipynb`
2. `known_4dstem_scan_drift_locked_ssb.ipynb`

The numbered notebooks in this `dev/real` folder are development history and
should not be treated as the active workflow entry point.

## Reproducible Commands

Generate or refresh one drift strength:

```bash
env CUDA_VISIBLE_DEVICES=0 /home/owner/miniforge3/envs/cuda-env/bin/python \
  notebooks/drift/dev/real/bto18_known_right_drift_batch.py --right-px 60 --gpu 0

env CUDA_VISIBLE_DEVICES=1 /home/owner/miniforge3/envs/cuda-env/bin/python \
  notebooks/drift/dev/real/bto18_known_right_drift_batch.py --right-px 90 --gpu 1
```

Build the manifest used to select inputs for ptychography:

```bash
/home/owner/miniforge3/envs/cuda-env/bin/python \
  notebooks/drift/dev/real/bto18_known_right_drift_manifest.py
```

The manifest is written to:

```text
notebooks/drift/dev/outputs/bto18_known_right_drift_manifest.csv
notebooks/drift/dev/outputs/bto18_known_right_drift_manifest.json
```

## Current Caveat

The right30 and right60 exports use the same 400 x 400 raw crop for clean0,
drift0, and drift90. The exploratory right90 export had to shift crops to keep
all drifted probe positions inside the source field. That is valid as a stress
test, but for final paper comparisons we should regenerate right30/right60/right90
as a common displayed-FOV series before comparing ptychography quality.

## Next Runner Inputs

Use the manifest rows for these acquisition names:

```text
clean0
drift0
drift90
```

For the next stage, treat the manifest as the source of drift vector, crop,
and output provenance. The H5 files also retain `probe_positions_px` and
`positions_offset_px` for a future position-aware bridge, but the current
organization work does not require `quantem.live` to understand those fields.


## Active Ptychography Drift Control

The current narrow ptychography control isolates image-0 drift before adding
the 90-degree acquisition:

1. clean 0 degree, no added drift, regular raster positions
2. 0 degree with known right30 drift, regular raster positions
3. the same 0 degree right30 drift diffraction patterns, known drift-corrected
   probe positions

Run one case with:

```bash
env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python \
  notebooks/drift/dev/real/run_bto18_crop400_drift0_three_way.py \
  --case drift0_corrected --trial-id-base 2030 --iters 10 \
  --slices 6 --slice-thickness 18 --probes 8 \
  --obj-lr 0.2 --probe-lr 0.2 --batch-size 4096
```

Then build the comparison manifest and QA figure:

```bash
PYTHONPATH=src python notebooks/drift/dev/real/make_bto18_crop400_drift0_three_way_qa.py
```

This control is intentionally explicit: case 2 and case 3 use the same drifted
H5 cube, and only the probe-position source changes.


Ptychography position correction is not another bilinear interpolation step.
The drifted H5 cube is loaded as-is. When explicit positions are supplied, the
fused CUDA ptychography path consumes them through `scan_positions_px`: rounded
integer positions choose the object patch centers, while subpixel fractions are
handled by Fourier phase ramps on the probe.

## 0/90 Position Geometry

Known-drift export metadata stores `probe_positions_px` in one shared physical
sample row/column frame. It is a global specimen frame, not detector pixels and
not a per-image display frame. The ptychography runners subtract the scan crop
origin to get crop-local coordinates, but image 0 and image 1 remain in the
same physical frame.

Use this rule to avoid double-rotating the 90-degree acquisition:

- regular image-0 raster without explicit positions: lock rotation to `clean0`
- regular image-1/90 raster without explicit global positions: lock rotation to
  `clean0 + 90`
- image-1/90 with explicit global/sample-frame probe positions: those positions
  already encode the 90-degree scan geometry, so lock the reconstruction in the
  common image-0/sample frame
- 0/90 combined corrected runs should stack the same DPs as the uncorrected
  combined run and change only the stacked probe-position array

If a 90-degree-only reconstruction is intended to stay in the raw 90 display
frame, first convert the probe positions into that raw frame and then use the
`clean0 + 90` calibration branch. Do not use global-frame positions and
`clean0 + 90` together.


Run the 90-degree counterpart with explicit sample-frame image1 positions:

```bash
env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python \
  notebooks/drift/dev/real/run_bto18_crop400_drift90_three_way.py \
  --case drift90_corrected --trial-id-base 2040 --iters 10 \
  --slices 6 --slice-thickness 18 --probes 8 \
  --obj-lr 0.2 --probe-lr 0.2 --batch-size 4096
```

Then build the 90-degree comparison manifest and QA figure:

```bash
PYTHONPATH=src python notebooks/drift/dev/real/make_bto18_crop400_drift90_three_way_qa.py
```

The 90-degree runner uses explicit image1 probe positions for clean, drifted,
and corrected cases, and locks the common base rotation. It does not combine
sample-frame positions with `clean0 + 90`.
