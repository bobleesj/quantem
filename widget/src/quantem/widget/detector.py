"""Virtual detectors (bright / annular-dark / dark field) for 4D-STEM.

Primary API - place a virtual detector on a dataset and get its image, with
collection angles in **mrad**::

    from quantem.widget import load, Dataset4dstemGPU, Show2D
    ds = Dataset4dstemGPU(load("master.h5"))
    Show2D(ds.detector.bf())                       # bright field (the bright disk)
    Show2D(ds.detector.adf())                       # annular dark field (auto band)
    Show2D(ds.detector.adf(inner=50, outer=180))    # collection angles in mrad
    Show2D(ds.detector.df())                        # outside the bright disk

The probe (disk center + size) auto-fits from the mean diffraction pattern;
``semiangle_mrad`` (from the load metadata, or set on the accessor) calibrates
mrad because the bright disk spans exactly the convergence semi-angle. MacBook
(MPS) runs the raw-Metal masked-sum over chunked uint16 buffers; CUDA / CPU runs
torch. **No binning** on either path.

The lower-level :func:`virtual` function (below) is mode-based
(DP/BF/ABF/ADF/HAADF/DF, bands measured in the auto-detected disk radius) and is
mainly the reference path the parity tests pin; ``ds.detector`` is the API to use.
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


class VirtualDetector:
    """``ds.detector.bf()`` / ``.adf()`` / ``.df()`` - place a virtual detector,
    get its (cached) image.

    bf / adf / df are virtual-detector geometries; each call integrates that
    geometry over every scan position and returns the 2D image. Collection
    angles are in **mrad** - the convergence semi-angle (``semiangle_mrad``,
    read from the load metadata) calibrates the detector, because the bright
    disk spans exactly that angle. The probe (disk center + size) auto-fits
    once from the mean diffraction pattern. Every result is cached; a repeat
    call is instant. Override the probe with ``.center`` / ``.bf_radius``.

        ds.detector.bf()                       # bright field (the bright disk)
        ds.detector.adf()                       # annular dark field, automatic band
        ds.detector.adf(inner=50, outer=180)    # collection angles in mrad
        ds.detector.df()                        # everything outside the bright disk
        ds.detector.df(inner=40)                # dark field beyond 40 mrad
    """

    def __init__(self, data):
        self._backend = _resolve_backend(data)
        self._scan_shape = tuple(data.scan_shape) if getattr(data, "_qw_dataset", False) else None
        # convergence semi-angle in mrad: the bright disk spans exactly this
        # angle, so it converts every mrad collection angle to a detector pixel
        # radius. From the load metadata; settable when the file lacks it.
        self.semiangle_mrad = getattr(data, "semiangle_mrad", None)
        self._mean_dp = None
        self._center = None       # auto-fit lazily; user-set wins
        self._bf_radius = None
        self._cache = {}

    def _mrad_to_px(self, mrad: float) -> float:
        """Collection angle in mrad -> detector pixel radius. The bright disk
        radius in pixels spans ``semiangle_mrad``, so a mrad angle maps to
        ``mrad / semiangle_mrad * bf_radius_px``."""
        if not self.semiangle_mrad:
            raise ValueError(
                "inner / outer are collection angles in mrad, but the convergence "
                "semi-angle is unknown for this dataset. Set it explicitly:\n"
                "    ds.detector.semiangle_mrad = <convergence semi-angle in mrad>")
        _, bf_radius_px = self._probe()
        return float(mrad) / float(self.semiangle_mrad) * bf_radius_px

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

    def _image(self, name, lo_px, hi_px):
        """Masked-sum image over the annulus ``lo_px .. hi_px`` detector pixels."""
        center, _ = self._probe()
        key = (name, round(center[0], 3), round(center[1], 3),
               round(float(lo_px), 3), round(float(hi_px), 3))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        cy, cx = center
        rows = np.arange(self.mean_dp.shape[0], dtype=np.float32)[:, None]
        cols = np.arange(self.mean_dp.shape[1], dtype=np.float32)[None, :]
        dist = np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2)
        mask = (dist >= lo_px) & (dist <= hi_px)
        img = np.asarray(self._backend.masked_sum(mask), dtype=np.float32)
        self._cache[key] = img
        return img

    def bf(self) -> np.ndarray:
        """Bright-field image: the bright disk (the unscattered probe)."""
        _, radius = self._probe()
        return self._image("bf", 0.0, radius)

    def adf(self, inner: float | None = None, outer: float | None = None) -> np.ndarray:
        """Annular-dark-field image collected between ``inner`` and ``outer`` mrad.

        ``inner`` / ``outer`` are collection angles in **mrad** (needs
        ``semiangle_mrad``). Omit either for the automatic band: ``inner`` =
        the bright-disk edge, ``outer`` = twice that.
        """
        _, radius = self._probe()
        lo_px = radius if inner is None else self._mrad_to_px(inner)
        hi_px = 2.0 * radius if outer is None else self._mrad_to_px(outer)
        return self._image("adf", lo_px, hi_px)

    def df(self, inner: float | None = None) -> np.ndarray:
        """Dark-field image: everything collected beyond ``inner`` mrad.

        ``inner`` is a collection angle in **mrad** (needs ``semiangle_mrad``).
        Omit it for the automatic edge: everything outside the bright disk.
        """
        _, radius = self._probe()
        lo_px = radius if inner is None else self._mrad_to_px(inner)
        return self._image("df", lo_px, np.inf)


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
