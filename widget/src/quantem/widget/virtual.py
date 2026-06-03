"""Virtual images (DP / BF / ABF / ADF / HAADF / DF) with automatic probe fitting.

One call - ``virtual(data, "BF")`` - finds the probe (BF disk center + radius) from
the mean diffraction pattern, builds the detector mask for the requested mode, and
reduces over every scan position. MacBook (MPS) runs the raw-Metal masked-sum over
chunked uint16 buffers; CUDA / CPU runs torch. **No binning** on either path. The
result is a 2D array ready for ``Show2D``.

Modes (annular bands in units of the auto-detected BF radius ``r``):
  DP    - the mean diffraction pattern itself (detector space)
  BF    - bright field, disk ``<= r``
  ABF   - annular bright field, ``0.5r .. r``
  ADF   - annular dark field, ``r .. 2r``
  HAADF - high-angle ADF, ``2r .. 4r``
  DF    - dark field, ``> r``
  annular(inner=, outer=) - custom band in BF-radius units

Usage::

    from quantem.widget import load, virtual, Show2D
    Show2D(virtual(load("master.h5"), "ADF"))   # probe found automatically
"""
from __future__ import annotations

import numpy as np


def _resolve_backend(data):
    """Return a compute backend (MetalCompute on MPS chunks, TorchCompute on array)."""
    if getattr(data, "_qw_dataset", False):  # Dataset4dstemGPU - backend already resolved
        return data.compute
    if hasattr(data, "_fields") and "data" in getattr(data, "_fields", ()):
        data = data.data
    # raw MPS chunks -> wrap so compute_backend sees a _is_gpu_frames source
    if hasattr(data, "chunks") and not getattr(data, "_is_gpu_frames", False):
        from quantem.widget.kernels.compute.mps import ChunkedFrames
        data = ChunkedFrames(data)
    from quantem.widget.kernels.compute.backends import compute_backend
    return compute_backend(data)


def auto_probe(mean_dp):
    """Detect the probe (BF disk) from the mean diffraction pattern.

    Threshold at ``mean + std``, take the centroid of the bright disk for the
    center, and ``radius = sqrt(area / pi)``. Matches Show4DSTEM.auto_detect_center.
    Returns ``((center_row, center_col), bf_radius)``.
    """
    dp = np.asarray(mean_dp, dtype=np.float32)
    thr = float(dp.mean()) + float(dp.std())
    mask = dp > thr
    total = int(mask.sum())
    if total == 0:
        h, w = dp.shape
        return (h / 2.0, w / 2.0), min(h, w) * 0.25
    rows = np.arange(dp.shape[0], dtype=np.float32)[:, None]
    cols = np.arange(dp.shape[1], dtype=np.float32)[None, :]
    cy = float((rows * mask).sum() / total)
    cx = float((cols * mask).sum() / total)
    radius = float(np.sqrt(total / np.pi))
    return (cy, cx), radius


def _detector_mask(mode, center, bf_radius, det_shape, inner, outer):
    """Boolean ``(det_row, det_col)`` mask for a virtual-detector mode."""
    cy, cx = center
    r = float(max(1.0, bf_radius))
    rows = np.arange(det_shape[0], dtype=np.float32)[:, None]
    cols = np.arange(det_shape[1], dtype=np.float32)[None, :]
    dist = np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2)
    bands = {
        "BF": (0.0, r),
        "ABF": (0.5 * r, r),
        "ADF": (r, 2.0 * r),
        "HAADF": (2.0 * r, 4.0 * r),
        "DF": (r, np.inf),
    }
    if mode == "ANNULAR":
        lo, hi = (inner if inner is not None else 0.0) * r, (outer if outer is not None else np.inf) * r
    else:
        lo, hi = bands[mode]
    return (dist >= lo) & (dist <= hi)


class VirtualImageAccessor:
    """``ds.virtual_image.bf()`` / ``.adf()`` / ``.df()`` - cached virtual images.

    The probe (bright-disk center + radius) is auto-fit once from the mean
    diffraction pattern, so ``inner`` / ``outer`` are in **BF-radius units**
    (``1.0`` = the bright-disk edge) and need no calibration. Every result is
    cached by ``(detector, center, radius, inner, outer)``; a repeat call is
    instant. Override the probe with ``.center`` / ``.bf_radius`` (clears the
    cache), or the default band with ``.adf_inner`` / ``.adf_outer`` / ``.df_inner``.

        ds.virtual_image.bf()                  # bright field, disk <= r
        ds.virtual_image.adf()                 # annular dark field, r .. 2r
        ds.virtual_image.adf(inner=1.5, outer=6)
        ds.virtual_image.df()                  # all dark field, > r
    """

    def __init__(self, data):
        self._backend = _resolve_backend(data)
        self._scan_shape = tuple(data.scan_shape) if getattr(data, "_qw_dataset", False) else None
        self._mean_dp = None
        self._center = None       # auto-fit lazily; user-set wins
        self._bf_radius = None
        self.adf_inner = 1.0      # BF-radius units; default ADF band r .. 2r
        self.adf_outer = 2.0
        self.df_inner = 1.0       # default DF: everything beyond the disk
        self._cache = {}

    @property
    def mean_dp(self) -> np.ndarray:
        """Mean diffraction pattern (computed once)."""
        if self._mean_dp is None:
            self._mean_dp = np.asarray(self._backend.mean_dp(), dtype=np.float32)
        return self._mean_dp

    def _probe(self):
        if self._center is None or self._bf_radius is None:
            center, radius = auto_probe(self.mean_dp)
            if self._center is None:
                self._center = center
            if self._bf_radius is None:
                self._bf_radius = radius
        return self._center, self._bf_radius

    @property
    def center(self):
        """Bright-disk center ``(row, col)`` in detector pixels (auto-fit if unset)."""
        return self._probe()[0]

    @center.setter
    def center(self, value):
        self._center = None if value is None else (float(value[0]), float(value[1]))
        self._cache.clear()

    @property
    def bf_radius(self):
        """Bright-disk radius in detector pixels (auto-fit if unset)."""
        return self._probe()[1]

    @bf_radius.setter
    def bf_radius(self, value):
        self._bf_radius = None if value is None else float(value)
        self._cache.clear()

    def _image(self, name, lo, hi):
        center, radius = self._probe()
        key = (name, round(center[0], 3), round(center[1], 3), round(radius, 3),
               round(float(lo), 4), round(float(hi), 4))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        mask = _detector_mask("ANNULAR", center, radius, self.mean_dp.shape, lo, hi)
        img = np.asarray(self._backend.masked_sum(mask), dtype=np.float32)
        self._cache[key] = img
        return img

    def bf(self) -> np.ndarray:
        """Bright-field image: detector disk ``<= r`` (the unscattered probe)."""
        return self._image("bf", 0.0, 1.0)

    def adf(self, inner: float | None = None, outer: float | None = None) -> np.ndarray:
        """Annular-dark-field image: band ``inner .. outer`` in BF-radius units
        (default ``r .. 2r``). Override the defaults via ``.adf_inner`` /
        ``.adf_outer``."""
        lo = self.adf_inner if inner is None else float(inner)
        hi = self.adf_outer if outer is None else float(outer)
        return self._image("adf", lo, hi)

    def df(self, inner: float | None = None) -> np.ndarray:
        """Dark-field image: everything beyond ``inner`` (BF-radius units, default
        ``> r``). Override the default via ``.df_inner``."""
        lo = self.df_inner if inner is None else float(inner)
        return self._image("df", lo, np.inf)


def virtual(data, mode="BF", *, center=None, bf_radius=None, inner=None, outer=None):
    """Virtual image for ``mode`` with automatic probe fitting. See module docstring.

    ``mode`` is case-insensitive (DP/BF/ABF/ADF/HAADF/DF/annular). ``center`` and
    ``bf_radius`` override the auto-detected probe; ``inner``/``outer`` (BF-radius
    units) define a custom band when ``mode="annular"``. Returns a 2D float array
    (detector-space for DP, scan-space otherwise) for ``Show2D``.
    """
    backend = _resolve_backend(data)
    mean_dp = np.asarray(backend.mean_dp(), dtype=np.float32)
    mode = str(mode).strip().upper()
    if mode == "DP":
        return mean_dp
    if center is None or bf_radius is None:
        c_auto, r_auto = auto_probe(mean_dp)
        center = center if center is not None else c_auto
        bf_radius = bf_radius if bf_radius is not None else r_auto
    mask = _detector_mask(mode, center, bf_radius, mean_dp.shape, inner, outer)
    return np.asarray(backend.masked_sum(mask), dtype=np.float32)
