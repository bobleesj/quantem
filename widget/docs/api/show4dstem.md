# Show4DSTEM

A 4D-STEM viewer with live virtual detectors: a movable aperture over the
diffraction stack and the resulting virtual image. See the
[Show4DSTEM tutorial](../tutorials/show4dstem).

```{note}
`quantem.widget.Show4DSTEM` is a thin dispatcher that picks the right backend
viewer from what you pass (raw array / tensor / `Dataset4dstem` / `load(...)`
output) and the device (CUDA, Apple Metal, CPU). The constructor parameters
below are the universal Torch viewer documented as the base class.
```

## Reference

```{autodoc2-object} quantem.widget.show4dstem.Show4DSTEM
render_plugin = "myst"
```

## Interactive controls

With a running kernel these recompute on the GPU backend (CUDA / MPS / CPU). For
a small dataset passed `offline=True`, the same controls run entirely in the
browser via WebGPU (bit-exact, sub-millisecond) with no kernel - see
[Performance](../perf/index).

| Control | Trait | Expected effect |
|---|---|---|
| Detector position (drag on diffraction) | `pos_row`, `pos_col` | Virtual image recomputes for that probe position |
| BF aperture radius | `bf_radius` | Bright-field disk grows/shrinks; virtual image updates |
| Aperture center | `center_row`, `center_col` | Recenters the detector on the unscattered beam |
| Detector ROI mode | `roi_mode`, `roi_active` | Switch BF / annular / rectangular detector |
| Annular inner / outer | `roi_radius_inner`, `roi_radius` | ADF annulus geometry |
| Virtual-image ROI | `vi_roi_mode`, `vi_roi_center_row`, `vi_roi_center_col` | Pick a real-space region to average its diffraction |
| FFT toggle | `show_fft`, `fft_window` | Power spectrum of the virtual image |
| Scan-path playback | `path_playing`, `path_index`, `path_interval_ms` | Sweeps the probe across the scan |
| k-space calibration | `k_pixel_size`, `k_calibrated` | Diffraction axes read in mrad when calibrated |
