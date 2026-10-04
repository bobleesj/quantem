# MAPED

Merge a beam-tilt series into one calibrated 4D-STEM dataset using
`quantem.gpu` for loading and storage, and MAPED for alignment and merging.

- [Automated workflow](maped_automated.ipynb): load, align, merge, and save.
- [Step-by-step workflow](maped_step_by_step.ipynb): inspect each alignment stage.
- [Load a saved result](maped_load.ipynb): reopen and inspect the merge.
- [API reference](../../docs/tutorials/maped_api.md): stage settings and outputs.

Set the example input and output paths to your own files. The notebooks are
distributed without acquisition data, saved figures, or widget state.

Pass `MAPED(tilts, files)` to label plots with the tilt values encoded as
`<x>x_<y>y` in each file name. These are acquisition labels: no angular unit
or coordinate conversion is inferred. Unmatched names use `Tilt 0`, `Tilt 1`,
and so on.
