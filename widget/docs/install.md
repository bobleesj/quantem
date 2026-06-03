# Installation

```bash
pip install quantem.widget
```

That single line works on every backend; the widget picks the fastest path it
finds at runtime.

## Backends

- **NVIDIA CUDA** - the universal Torch viewer runs on GPU. The integer-reduction
  detector path uses CuPy. We do not pin a CuPy wheel (a fixed `cuda12x`/`cuda13x`
  would collide with one your environment already ships); a real CUDA workflow
  already has the matching CuPy installed.
- **Apple Silicon (Metal / MPS)** - a dedicated raw-Metal viewer powers
  `Show4DSTEM` on the MacBook, with full-resolution CBED and a fast virtual-image
  path. The tiny `pyobjc-framework-Metal` wheel installs automatically on macOS.
- **CPU** - everything still runs, just slower. This is the path used to build
  these docs.

## Verify

```python
import quantem.widget as qw
print(qw.__version__)
print(qw.__all__)   # ['Show2D', 'Show3D', 'Show3DSlices', 'Show4DSTEM', 'load']
```
