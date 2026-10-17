"""Read Arina counts independently of optional microscope calibration files."""

import math
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - register detector compression filters
import numpy as np


def read_arina(
    filename,
    *,
    navigation_shape=None,
    rebin_diffraction=1,
    dtype=None,
    flatfield=None,
    lazy=False,
):
    """Read native counts; use physical scan sampling only when supplied by a file."""
    if lazy:
        raise NotImplementedError("Arina loading is eager; omit lazy=True.")
    path = Path(filename)
    with h5py.File(path, "r") as source:
        datasets = [source["entry/data"][key] for key in source["entry/data"]]
        frames = sum(item.shape[0] for item in datasets)
        detector_shape = datasets[0].shape[1:]
        if any(item.shape[1:] != detector_shape for item in datasets):
            raise ValueError("Arina data chunks have different detector shapes.")
        if navigation_shape is None:
            navigation_shape = source.attrs.get("scan_shape", (math.isqrt(frames),) * 2)
        elif np.isscalar(navigation_shape):
            navigation_shape = (int(navigation_shape), "auto")
        scan_rows, scan_cols = navigation_shape
        if scan_rows == "auto":
            scan_rows = frames // scan_cols
        if scan_cols == "auto":
            scan_cols = frames // scan_rows
        if scan_rows * scan_cols != frames:
            raise ValueError(
                f"navigation_shape={navigation_shape} does not match {frames} patterns; "
                "supply (scan_rows, scan_cols)."
            )
        output_dtype = np.dtype(
            np.float32 if flatfield is not None else (dtype or datasets[0].dtype)
        )
        counts = np.empty((frames, *detector_shape), dtype=output_dtype)
        offset = 0
        for item in datasets:
            item.read_direct(counts, dest_sel=np.s_[offset : offset + item.shape[0]])
            offset += item.shape[0]

    if flatfield is not None:
        flatfield = np.asarray(flatfield)
        factors = np.ones(detector_shape, dtype=np.float32)
        np.divide(np.median(flatfield), flatfield, out=factors, where=flatfield != 0)
        counts *= factors
    if rebin_diffraction != 1:
        factor = int(rebin_diffraction)
        if factor != rebin_diffraction or factor < 1 or any(n % factor for n in detector_shape):
            raise ValueError("rebin_diffraction must be a positive divisor of both detector axes.")
        rows, cols = detector_shape
        counts = counts.reshape(frames, rows // factor, factor, cols // factor, factor)
        counts = counts.mean(axis=(2, 4)).astype(output_dtype)

    sampling = [1.0] * 4
    units = ["pixels"] * 4
    companion = path.with_name(path.name.removesuffix("_master.h5") + ".h5")
    calibration = {"available": False, "reason": "No microscope metadata companion"}
    if companion.exists():
        with h5py.File(companion, "r") as metadata:
            group = metadata.get("STEM Metadata")
            pixel_size = None if group is None else group.attrs.get("Pixel Size")
        if pixel_size is not None:
            step = float(np.asarray(pixel_size).reshape(-1)[0]) * 10  # nm to Angstrom
            if not np.isfinite(step) or step <= 0:
                raise ValueError(
                    f"Invalid scan pixel size in {companion}; correct its calibration."
                )
            sampling[:2], units[:2] = [step, step], ["A", "A"]
            calibration = {"available": True, "source": "microscope metadata companion"}
        else:
            calibration["reason"] = "Microscope metadata has no Pixel Size"
    data = counts.reshape(int(scan_rows), int(scan_cols), *counts.shape[-2:])
    return [
        {
            "data": data,
            "axes": [
                {"name": name, "size": size, "scale": step, "offset": 0.0, "units": unit}
                for name, size, step, unit in zip(
                    ("scan_row", "scan_col", "detector_row", "detector_col"),
                    data.shape,
                    sampling,
                    units,
                    strict=True,
                )
            ],
            "metadata": {"scan_calibration": calibration},
        }
    ]
