# MAPED API

Merge a 4D-STEM tilt series into one dataset. Runs unchanged on an NVIDIA GPU
(CUDA) or an Apple Silicon Mac (MPS). Nothing to configure: `io.load` picks the
GPU, and MAPED runs where the tilts are. With several NVIDIA GPUs the first is
used; `io.load(files, device="cuda:1")` picks another. On a Mac the one GPU is
`mps`. `quantem.gpu.device.profile()` lists what a machine has.

```python
from quantem.gpu import io
from quantem.diffraction import MAPED
from quantem.widget import Show4DSTEM

files = io.discover("/path/to/sample/maped")   # one *_master.h5 per tilt
tilts = io.load(files)             # compressed on the GPU, bad pixels replaced by their median
maped = MAPED(tilts, files)                     # file names carry the tilt, used in plot titles
merged = maped.run()                            # all stages, one figure per stage
io.save("/path/to/sample/maped_merged_master.h5", merged)
Show4DSTEM(merged)                              # from quantem.widget
```

Install `quantem.gpu` 0.0.1rc11 or newer from
[TestPyPI](https://test.pypi.org/project/quantem-gpu/) for calibrated merge storage.
The [notebooks](../../notebooks/maped/README.md) include installation commands.
File-name tilt labels retain the recorded values; they do not infer degrees,
milliradians, or a conversion to image row/column coordinates.

`run()` prints one line per stage (what was done, what it found, how long it
took) and draws four pictures. In a script or on the command line use
`run(show=False)`; `run(verbose=False)` silences the lines. Every stage takes
the same `show` flag.

## Stages

`run()` shows the tilts, then calls these in order. Call them yourself to change a setting or to look
between steps; `run(iterations=80)` passes any stage setting through.

| Call | What it does | Settings |
|---|---|---|
| `show_tilts()` | Draws each tilt's bright-field image and mean diffraction pattern (computed when `MAPED(tilts, files)` is built; every stage works on them). Titles read the tilt from the file name (`-17.0x 0.0y`), or say `Tilt 0`, `Tilt 1`, ... without `files`. | `plot_scale`: divides the bright-field plots only |
| `find_beam_center(blur=1)` | Brightest pixel of each blurred mean pattern, marked on the plot. Feeds nothing downstream. | `blur` px; `centers=(row, col)` to set them yourself |
| `align_diffraction(border_taper=2)` | One detector shift per tilt, by cross-correlating the mean patterns. | `border_taper` px faded at the detector border; `precision` px, default 0.01 |
| `align_real_space(iterations=20, hann_window=True, max_shift=5)` | One scan shift per tilt. Aligns each tilt to the average image and rebuilds the average, `iterations` times. | see below |
| `merge(dtype="scaled_uint16")` | Weighted mean of the tilts at every position, read at the shifted positions. | `scan_region`, `save_to`, `release_tilts` |

### `align_real_space`

Two tilts of the same area do not have the same brightness: contrast changes
with tilt, often as a slow ramp across the scan. Matching brightness would chase
that ramp, so the alignment matches edge strength, how fast the brightness
changes, blurred over `edge_blur` scan pixels. Edges sit where the features
are, whatever the overall brightness does.

- `iterations`: raise until doubling it no longer changes `real_space_shifts`. On the reference data 20 was 0.6 px short of settled, 80 was settled.
- `hann_window`: fade the scan border to zero so the frame of the scan is not a feature. Keep it on.
- `edge_blur`: width of an edge in scan pixels, default 2; `None` aligns on brightness.
- `max_shift`: the largest scan shift you expect, in pixels. Pads the correlation so shifted features do not wrap around. Not a limit.
- `precision`: smallest shift step measured, in pixels.

### `merge`

- Computes in float32. The complete result is stored as scaled 16-bit integers
  (`dtype="scaled_uint16"`), each region with its own calibration; the printed
  line reports the rounding error. Exact float32 is available for a small patch:
  `merge(scan_region=(row0, row1, col0, col1))`, at most 4096 positions.
- `save_to=path` writes the result to disk region by region instead of holding
  it on the GPU during the merge, then reopens the file.
- `release_tilts=True` closes the tilts once the merge no longer needs them.

## Results

`maped.mean_pattern`, `maped.bright_field` (one per tilt), `maped.beam_centers`, `maped.diffraction_shifts`, `maped.real_space_shifts`
(one `(row, col)` per tilt), `maped.merged` (the result, also returned by
`merge` and `run`). `maped.close()` releases the merged result; the
tilts stay with you.

## Memory, seven tilts of 512 x 512 x 192 x 192

| Recipe | Resident on the GPU | Measured on an M5 Max |
|---|---|---|
| `maped.run()` | tilts 7.8 GiB + result 6.0 GiB + about 3 GiB during the merge | 40 s load through saved file |
| `maped.run(save_to=path, release_tilts=True)` | tilts 7.8 GiB during the merge, then only the 6.0 GiB result | 35 s load through reopened file |

Use the second form on a 24 GB Mac. Decoding and merging never hold the
dense data: 135 GB of counts stream through a 2 GB window.
