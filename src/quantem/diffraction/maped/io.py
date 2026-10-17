"""Optional MAPED input preparation through QuantEM's existing readers."""

from pathlib import Path

import h5py
import numpy as np

from quantem.core.datastructures import Dataset4dstem
from quantem.core.io import read_4dstem


def load(
    file_path: str | Path,
    file_type: str | None = None,
    *,
    apply_mask: bool = False,
    **kwargs,
) -> Dataset4dstem:
    """Load one CPU acquisition and retain its file name for MAPED tilt labels.

    Uses ``read_4dstem`` without changing its behavior. Masking is opt-in;
    ordinary ``Dataset4dstem.from_file`` inputs also work directly with MAPED.
    Encoded GPU acquisitions use ``quantem.gpu.io.load``.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Acquisition file.
    file_type : str, optional
        Reader name accepted by ``read_4dstem``, such as ``"arina"``.
    apply_mask : bool, optional
        Replace Arina detector-mask pixels with zero in every pattern. Requires
        native detector sampling. The input file is unchanged. Default False.
    **kwargs
        Options passed to ``read_4dstem``, including calibration and binning.

    Returns
    -------
    Dataset4dstem
        Native CPU dataset with source and optional correction metadata.

    Examples
    --------
    >>> from quantem.diffraction.maped import MAPED, io
    >>> tilts = [io.load(file, "arina") for file in files]
    >>> merged = MAPED.from_datasets(tilts).run(show=False)
    """
    if apply_mask and (
        (file_type or "").lower() not in {"arina", "rsciio.arina"}
        or kwargs.get("rebin_diffraction", 1) != 1
    ):
        raise ValueError(
            "apply_mask requires file_type='arina' and rebin_diffraction=1; "
            "load the native detector dimensions before applying the mask."
        )
    dataset = read_4dstem(file_path, file_type, **kwargs)
    dataset.file_path = file_path
    dataset.metadata.update(source_path=str(file_path), representation="dense")
    if apply_mask:
        with h5py.File(file_path, "r") as source:
            mask = source["entry/instrument/detector/detectorSpecific/pixel_mask"][()] != 0
        if mask.shape != dataset.shape[-2:]:
            raise ValueError(
                f"Detector mask shape {mask.shape} differs from {dataset.shape[-2:]}; "
                "load the native detector dimensions before applying the mask."
            )
        dataset.array[..., mask] = 0
        dataset.metadata["hot_pixel_correction"] = {
            "method": "zero",
            "applied": True,
            "pixel_count": int(mask.sum()),
            "coordinates_row_column": np.argwhere(mask).tolist(),
        }
    return dataset
