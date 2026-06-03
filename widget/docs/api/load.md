# load

Reads compressed 4D-STEM data straight onto the GPU (CUDA / Apple Metal) or CPU
and returns a `LoadResult` you hand to [`Show4DSTEM`](show4dstem). Public import:

```python
from quantem.widget import load
```

## Reference

```{autodoc2-object} quantem.widget.io.hdf5.load
render_plugin = "myst"
```

```{tip}
`det_bin=2` (or `4`) bins the detector on load to cut memory and speed first
paint; pass a list of file paths to stack several datasets behind a single
"Dataset" slider.
```
