from importlib.metadata import PackageNotFoundError, version

from quantem.widget.show2d import Show2D
from quantem.widget.show3d import Show3D
from quantem.widget.show3dslices import Show3DSlices
from quantem.widget.show4dstem import Show4DSTEM as _Show4DSTEMBase
from quantem.widget.io import load


def Show4DSTEM(data, **kwargs):
    """Open a 4D-STEM viewer over ``load(...)`` output, on any backend.

    One mental model, single + multi, CUDA + MacBook::

        from quantem.widget import load, Show4DSTEM
        Show4DSTEM(load("a.h5", det_bin=2))                 # one dataset
        Show4DSTEM(load(["a.h5", "b.h5", ...], det_bin=4))  # many (dataset slider)

    Dispatch is automatic from what ``load`` returns:
      - MacBook (MPS) single -> the raw-Metal real-time viewer (full-res CBED +
        bin2 virtual-image fast path). torch.mps is not fast enough on Apple
        Silicon, which is why the dedicated Metal path exists.
      - MacBook (MPS) many -> a lazy handle; dataset 0 shows now, 1..N fill in the
        background behind the dataset slider.
      - CUDA / CPU single or many -> the universal torch viewer (a 5D array gives
        an instant dataset slider on big-VRAM boxes).
    """
    # MacBook lazy multi-dataset handle -> build the viewer + start background fill.
    from quantem.widget.multidataset_mps import LazyMacbookDatasets
    if isinstance(data, LazyMacbookDatasets):
        return data.build_viewer(**kwargs)
    # MacBook single raw-Metal load (LoadResult wrapping MPSChunked4DSTEM, or an
    # already-wrapped ChunkedFrames) -> the specialized Metal viewer with sampling
    # pulled from metadata. CUDA/CPU loads fall through to the universal viewer.
    is_loadresult = hasattr(data, "_fields") and "data" in getattr(data, "_fields", ())
    payload = data.data if is_loadresult else data
    if getattr(payload, "_is_gpu_frames", False) or hasattr(payload, "chunks"):
        from quantem.widget.show4dstem_mps import Show4DSTEM_MACBOOK
        return Show4DSTEM_MACBOOK(payload, **kwargs)
    # CUDA/CPU multi-dataset stack (a list load gives a 5D array): label the slider
    # "Dataset" and name each slot from the source files, so it reads as a list of
    # datasets rather than a generic frame axis.
    if is_loadresult and getattr(payload, "ndim", 0) == 5:
        meta = getattr(data, "metadata", {}) or {}
        kwargs.setdefault("frame_dim_label", "Dataset")
        names = meta.get("file_names")
        if names is not None:
            kwargs.setdefault("frame_labels", list(names))
    return _Show4DSTEMBase(data, **kwargs)


try:
    __version__ = version("quantem.widget")
except PackageNotFoundError:
    # Source-tree imports (e.g. `PYTHONPATH=src pytest`) skip pip install.
    __version__ = "0.0.0+local"

__all__ = ["Show2D", "Show3D", "Show3DSlices", "Show4DSTEM", "load"]
