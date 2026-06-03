"""``Dataset4dstemGPU`` - the widget's own GPU-resident 4D-STEM container.

One simple type over BOTH backends: a torch tensor (CUDA / MPS / CPU) or raw Apple
Metal uint16 chunks (MacBook no-bin). It wraps the shared compute backend
(``MetalCompute`` / ``TorchCompute``) and the scan/detector shape + calibration, so
user code never branches on hardware:

    from quantem.widget import load, Dataset4dstemGPU, Show2D
    ds = Dataset4dstemGPU(load("master.h5"))   # torch on CUDA, Metal chunks on Mac
    Show2D(ds.virtual("ADF"))                    # auto probe-fit virtual image
    Show2D(ds.dpc().phase)                       # CoM -> rotation -> iDPC

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

    def __init__(self, data, *, scan_shape=None, sampling=None, units=None, name=""):
        # carry calibration straight off a LoadResult's metadata when present
        if hasattr(data, "_fields") and "metadata" in getattr(data, "_fields", ()):
            meta = data.metadata or {}
            if sampling is None:
                sampling = meta.get("scan_sampling_A") and (meta["scan_sampling_A"],) * 2
            if name == "":
                name = meta.get("name", "")
        self._compute = _resolve_compute(data)
        self.scan_shape = tuple(scan_shape) if scan_shape is not None else tuple(self._compute.scan_shape)
        self.det_shape = tuple(self._compute.det_shape)
        self.sampling = sampling
        self.units = units
        self.name = name
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
    def virtual(self, mode: str = "BF", **kwargs) -> np.ndarray:
        """Auto-probe-fit virtual image (DP/BF/ABF/ADF/HAADF/DF). See :func:`virtual`."""
        from quantem.widget.virtual import virtual
        return virtual(self, mode, **kwargs)

    def center_of_mass(self, mask=None):
        from quantem.widget.dpc import center_of_mass
        return center_of_mass(self, mask=mask)

    def dpc(self, **kwargs):
        """Center-of-mass -> rotation -> iDPC. See :func:`dpc`."""
        from quantem.widget.dpc import dpc
        return dpc(self, **kwargs)

    def show4dstem(self, **kwargs):
        """Open the raw 4D viewer on the underlying data."""
        from quantem.widget import Show4DSTEM
        return Show4DSTEM(self._raw, **kwargs)

    def __repr__(self) -> str:
        s = "x".join(str(x) for x in self.shape)
        return f"Dataset4dstemGPU({s}, backend={self.backend})"
