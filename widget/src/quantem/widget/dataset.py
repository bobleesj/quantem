"""``Dataset4dstemGPU`` - the widget's own GPU-resident 4D-STEM container.

One simple type over BOTH backends: a torch tensor (CUDA / MPS / CPU) or raw Apple
Metal uint16 chunks (MacBook no-bin). It wraps the shared compute backend
(``MetalCompute`` / ``TorchCompute``) and the scan/detector shape + calibration, so
user code never branches on hardware:

    from quantem.widget import load, Dataset4dstemGPU, Show4DSTEM, Show2D
    ds = Dataset4dstemGPU(load("master.h5"))   # torch on CUDA, Metal chunks on Mac
    Show4DSTEM(ds)                               # raw 4D viewer
    Show2D(ds.bf())                               # bright field (cached, auto probe)
    Show2D(ds.adf(inner=50, outer=180))          # annular dark field, mrad
    Show2D(ds.com.col)                            # horizontal CoM (also ds.com.row)
    Show2D(ds.idpc())                             # iDPC phase (CoM -> rotation -> integrate)

It is deliberately NOT ``quantem.core.Dataset4dstem`` (torch-only, can't hold Metal
chunks / trips the MPS INT_MAX ceiling on no-bin, and re-adds the quantem dep). This
one is self-contained and MPS-aware, and it's thin - all the math lives in the
backend; this is the friendly face over it.
"""
from __future__ import annotations

import numpy as np


def _resolve_compute(data):
    """Compute backend (MetalCompute on Metal chunks, TorchCompute on array) for raw load output."""
    if hasattr(data, "_fields") and "data" in getattr(data, "_fields", ()):
        data = data.data
    if hasattr(data, "chunks") and not getattr(data, "_is_gpu_frames", False):
        from quantem.widget.kernels.compute.mps import ChunkedFrames
        data = ChunkedFrames(data)
    from quantem.widget.kernels.compute.backends import compute_backend
    return compute_backend(data)


class Dataset4dstemGPU:
    """GPU-resident 4D-STEM dataset over either backend. Holds the compute backend +
    scan/detector shape + optional sampling/units; methods delegate to the backend."""

    _qw_dataset = True  # duck-type flag so dpc()/virtual() route via .compute, no import cycle

    def __init__(self, data, *, scan_shape=None, sampling=None, units=None, name="",
                 semiangle_mrad=None):
        # carry calibration straight off a LoadResult's metadata when present
        if hasattr(data, "_fields") and "metadata" in getattr(data, "_fields", ()):
            meta = data.metadata or {}
            if sampling is None:
                sampling = meta.get("scan_sampling_A") and (meta["scan_sampling_A"],) * 2
            if name == "":
                name = meta.get("name", "")
            if semiangle_mrad is None:
                # convergence semi-angle: calibrates ds.adf()/df() mrad collection
                # angles. Optional - the automatic bf/adf/df bands work without it.
                semiangle_mrad = meta.get("semiangle_mrad") or meta.get("semi_angle_mrad")
        self._compute = _resolve_compute(data)
        self.scan_shape = tuple(scan_shape) if scan_shape is not None else tuple(self._compute.scan_shape)
        self.det_shape = tuple(self._compute.det_shape)
        self.sampling = sampling
        self.units = units
        self.name = name
        self.semiangle_mrad = float(semiangle_mrad) if semiangle_mrad else None
        self._raw = data  # kept so Show4DSTEM can take the underlying tensor / chunks
        # probe (auto-fit lazily; user-set wins) + caches for the derived API
        self._mean_dp_cache = None
        self._center = None
        self._bf_radius = None
        self._detector_cache = {}
        self._dpc_cache = {}

    # --- backend identity ---
    @property
    def compute(self):
        return self._compute

    @property
    def backend(self) -> str:
        cls = self._compute.__class__.__name__
        return {"MetalCompute": "mps", "TorchCompute": str(getattr(self._compute, "device", "cpu")),
                "CudaKernelCompute": "cuda"}.get(cls, cls)

    @property
    def shape(self):
        return (*self.scan_shape, *self.det_shape)

    @property
    def n_frames(self) -> int:
        return int(self._compute.n_frames)

    # --- primitive reads (delegate to backend) ---
    def frame(self, idx: int) -> np.ndarray:
        return np.asarray(self._compute.frame(int(idx)))

    def mean_dp(self) -> np.ndarray:
        if self._mean_dp_cache is None:
            self._mean_dp_cache = np.asarray(self._compute.mean_dp(), dtype=np.float32)
        return self._mean_dp_cache

    def masked_sum(self, det_mask) -> np.ndarray:
        return np.asarray(self._compute.masked_sum(det_mask)).reshape(self.scan_shape)

    # --- probe (auto-fit bright disk; override clears the detector cache) ---
    def _probe(self):
        if self._center is None or self._bf_radius is None:
            from quantem.widget.detector import auto_probe
            center, radius = auto_probe(self.mean_dp())
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
        self._detector_cache.clear()

    @property
    def bf_radius(self):
        """Bright-disk radius in detector pixels (auto-fit if unset)."""
        return self._probe()[1]

    @bf_radius.setter
    def bf_radius(self, value):
        self._bf_radius = None if value is None else float(value)
        self._detector_cache.clear()

    # --- derived API: virtual detectors + DPC (thin over the compute backend) ---
    def bf(self) -> np.ndarray:
        """Bright-field image (the bright disk). See :func:`quantem.widget.detector.bf`."""
        from quantem.widget.detector import bf
        return bf(self)

    def adf(self, inner=None, outer=None, unit="mrad") -> np.ndarray:
        """Annular-dark-field image; ``inner``/``outer`` in mrad (default) or
        ``unit='px'`` (auto band if omitted). See :func:`quantem.widget.detector.adf`."""
        from quantem.widget.detector import adf
        return adf(self, inner, outer, unit)

    def df(self, inner=None, unit="mrad") -> np.ndarray:
        """Dark-field image beyond ``inner`` mrad (default) or ``unit='px'``
        (outside the bright disk if omitted). See :func:`quantem.widget.detector.df`."""
        from quantem.widget.detector import df
        return df(self, inner, unit)

    def _dpc(self):
        """Cached full DPC pipeline (CoM -> auto rotation -> Fourier integrate)."""
        result = self.__dict__.get("_dpc_result")
        if result is None:
            from quantem.widget.dpc import dpc
            result = dpc(self)
            self.__dict__["_dpc_result"] = result
        return result

    @property
    def com(self):
        """Center-of-mass vector field (cached): ``ds.com.row`` / ``ds.com.col``."""
        accessor = self.__dict__.get("_com_result")
        if accessor is None:
            from quantem.widget.dpc import com
            accessor = com(self)
            self.__dict__["_com_result"] = accessor
        return accessor

    def idpc(self) -> np.ndarray:
        """Integrated-DPC phase image (CoM -> auto rotation -> integrate), cached."""
        return self._dpc().phase

    @property
    def rotation_deg(self) -> float:
        """Auto-found scan/detector rotation used by :meth:`idpc` (degrees)."""
        return self._dpc().rotation_deg

    def __repr__(self) -> str:
        s = "x".join(str(x) for x in self.shape)
        return f"Dataset4dstemGPU({s}, backend={self.backend})"
