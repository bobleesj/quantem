"""``Dataset4dstemGPU`` - the widget's own GPU-resident 4D-STEM container.

One simple type over BOTH backends: a torch tensor (CUDA / MPS / CPU) or raw Apple
Metal uint16 chunks (MacBook no-bin). It wraps the shared compute backend
(``MetalCompute`` / ``TorchCompute``) and the scan/detector shape + calibration, so
user code never branches on hardware:

    from quantem.widget import load, Dataset4dstemGPU, Show4DSTEM, Show2D
    ds = Dataset4dstemGPU(load("master.h5"))   # torch on CUDA, Metal chunks on Mac
    Show4DSTEM(ds)                               # raw 4D viewer
    Show2D(ds.detector.bf())                      # bright field (cached, auto probe)
    Show2D(ds.detector.adf())                     # annular dark field
    Show2D(ds.dpc().phase)                       # CoM -> rotation -> iDPC (cached)

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
                # convergence semi-angle: calibrates ds.detector mrad collection
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
        return np.asarray(self._compute.mean_dp())

    def masked_sum(self, det_mask) -> np.ndarray:
        return np.asarray(self._compute.masked_sum(det_mask)).reshape(self.scan_shape)

    # --- derived properties (the friendly API) ---
    @property
    def detector(self):
        """Virtual detectors: ``.bf()`` / ``.adf()`` / ``.df()`` (cached images).

        See :class:`quantem.widget.detector.VirtualDetector`. Built once per
        dataset; the probe auto-fits and every detector result is memoized.
        """
        accessor = self.__dict__.get("_detector")
        if accessor is None:
            from quantem.widget.detector import VirtualDetector
            accessor = VirtualDetector(self)
            self.__dict__["_detector"] = accessor
        return accessor

    def center_of_mass(self, mask=None):
        from quantem.widget.dpc import center_of_mass
        return center_of_mass(self, mask=mask)

    def dpc(self, **kwargs):
        """Center-of-mass -> rotation -> iDPC (cached). See :func:`dpc`.

        The CoM pass over the 4D block is the cost; the result is memoized per
        kwargs so a repeat ``ds.dpc()`` is instant. A custom ``mask=`` array
        bypasses the cache (arrays aren't hashable, and it's a one-off anyway).
        """
        cache = self.__dict__.setdefault("_dpc_cache", {})
        if "mask" in kwargs and kwargs["mask"] is not None:
            from quantem.widget.dpc import dpc
            return dpc(self, **kwargs)
        key = tuple(sorted((k, v) for k, v in kwargs.items() if k != "mask"))
        result = cache.get(key)
        if result is None:
            from quantem.widget.dpc import dpc
            result = dpc(self, **kwargs)
            cache[key] = result
        return result

    def __repr__(self) -> str:
        s = "x".join(str(x) for x in self.shape)
        return f"Dataset4dstemGPU({s}, backend={self.backend})"
