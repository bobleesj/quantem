"""Read Velox EMD images, spectrum images, and scan metadata for drift correction."""

from pathlib import Path

import numpy as np


def _image_dataset(stream, path, metadata):
    from quantem.core.datastructures.dataset2d import Dataset2d

    axes = stream.get("axes", [])
    title = stream.get("metadata", {}).get("General", {}).get("title")
    image = Dataset2d.from_array(
        np.ascontiguousarray(stream["data"], dtype=np.float32),
        name=title or Path(path).stem,
        sampling=tuple(float(axis.get("scale", 1.0)) for axis in axes[:2]),
        units=tuple((axis.get("units") or "pixels") for axis in axes[:2]),
    )
    image.file_path = path
    image.metadata.update(metadata)
    return image


def read_emd(path: str | Path):
    """Read the HAADF image and acquisition geometry from a Velox EMD file.

    Drift correction needs one calibrated scan image plus its recorded scan
    direction. RosettaSciIO selects the image stream, while QuantEM's normalized
    metadata reader supplies rotation, stage position, magnification, and
    acquisition time without duplicating Velox decoding here.

    Parameters
    ----------
    path : str or Path
        Velox EMD acquisition.

    Returns
    -------
    Dataset2d
        Calibrated HAADF image carrying the normalized acquisition metadata.

    Examples
    --------
    >>> image = read_emd("scan_0.emd")
    >>> image.metadata["scan_rotation_deg"]
    0.0
    """
    from rsciio.emd import file_reader

    from quantem.core.io.file_readers import read_emd_metadata

    streams = file_reader(str(path), select_type="images")
    stream = next(
        (
            item
            for item in streams
            if item["data"].ndim == 2
            and item.get("metadata", {}).get("General", {}).get("title") == "HAADF"
        ),
        next((item for item in streams if item["data"].ndim == 2), None),
    )
    if stream is None:
        raise ValueError(f"{path} contains no two-dimensional image stream")

    return _image_dataset(stream, path, read_emd_metadata(path))


def _energy_axis_kev(axes, n_channels):
    energy = next(
        (
            axis
            for axis in axes
            if (axis.get("units") or "") in ("keV", "eV")
            or "energy" in (axis.get("name") or "").lower()
        ),
        None,
    )
    if energy is None:
        return None
    to_keV = 1e-3 if energy.get("units") == "eV" else 1.0
    scale = float(energy.get("scale", 1.0)) * to_keV
    offset = float(energy.get("offset", 0.0)) * to_keV
    return offset + np.arange(n_channels) * scale


def read_emd_eds(
    path: str | Path,
    *,
    load_spectrum: bool = False,
    verbose: bool = True,
) -> dict[str, object]:
    """Read images from a Velox EDS/EELS spectrum-image EMD.

    Encapsulates the raw rsciio stream traversal so callers never write a
    ``for ds in file_reader(...)`` loop: the simultaneously-acquired HAADF
    survey and any Velox pre-quantified 2-D element maps are separated here
    and returned by name. The full spectrum is opt-in because expanding it can
    require tens of gigabytes while drift correction needs only the HAADF.
    The scan angle comes from the EMD metadata (never typed).

    Parameters
    ----------
    path : str or Path
        Velox spectrum-image EMD file.
    load_spectrum : bool, optional
        Load the full ``(row, col, energy)`` spectrum. The default ``False``
        loads only the HAADF and stored 2-D elemental maps.
    verbose : bool, optional
        Print a compact summary of the loaded images.

    Returns
    -------
    dict[str, object]
        Calibrated HAADF image, optional spectrum and energy axis, stored
        element maps, and the acquisition geometry needed for correction.

    Examples
    --------
    >>> acquisition = read_emd_eds("spectrum_image.emd")
    >>> acquisition["haadf"].shape
    (2048, 2048)
    """
    from rsciio.emd import file_reader

    from quantem.core.io.file_readers import read_emd_metadata

    streams = [
        (np.asarray(ds["data"]),
         ds.get("metadata", {}).get("General", {}).get("title", ""),
         ds.get("axes", []))
        for ds in file_reader(
            str(path),
            select_type=None if load_spectrum else "images",
        )
    ]
    cube, cube_axes = next(
        ((data, axes) for data, _, axes in streams if data.ndim == 3), (None, None))
    if load_spectrum and cube is None:
        raise ValueError(f"{path} contains no 3-D spectrum")
    haadf_entry = next(
        ((data, axes) for data, title, axes in streams
         if data.ndim == 2 and title == "HAADF"), None)
    element_maps = {
        title: data.astype(np.float32) for data, title, _ in streams
        if data.ndim == 2 and title != "HAADF"
    }
    energy_axis = (
        None if cube is None else _energy_axis_kev(cube_axes, cube.shape[-1])
    )
    metadata = read_emd_metadata(path)
    scan_rot = metadata["scan_rotation_deg"]
    if scan_rot is None:
        scan_rot = 0.0
    px_nm = metadata["pixel_size_nm"]
    px_nm = float("nan") if px_nm is None else float(px_nm)
    if haadf_entry is None:
        haadf = None
    else:
        data, axes = haadf_entry
        haadf = _image_dataset(
            {
                "data": data,
                "axes": axes,
                "metadata": {"General": {"title": Path(path).stem}},
            },
            path,
            metadata | {
                "scan_rotation_deg": float(scan_rot),
                "pixel_size_nm": px_nm,
            },
        )
    image = haadf if haadf is not None else next(iter(element_maps.values()), None)
    if image is None:
        raise ValueError(f"{path} contains no 2-D HAADF or elemental maps")
    shape = tuple(image.shape)
    if verbose:
        spectrum = "not loaded" if cube is None else str(cube.shape)
        print(
            f"HAADF {None if haadf is None else haadf.shape}  "
            f"elements {list(element_maps)}  spectrum {spectrum}  "
            f"scan {scan_rot:.1f}°"
        )
    return {
        "haadf": haadf,
        "cube": cube,
        "energy_axis_keV": energy_axis,
        "element_maps": element_maps,
        "scan_rotation_deg": scan_rot,
        "pixel_size_nm": px_nm,
        "shape": shape,
        "path": str(path),
    }


def scan_pairs(
    folder: str | Path,
    *,
    max_rotation_tolerance_deg: float = 5.0,
):
    """Pair orthogonal Velox scans acquired from the same specimen area.

    Stage position identifies the shared field of view; scan rotation identifies
    the orthogonal acquisition. Shape, pixel calibration, field of view, and
    nominal magnification reject incompatible acquisitions when those metadata
    are present. Only unique mutual matches are paired. The returned inventory
    includes every file and a reason for every acquisition that is not included.

    Parameters
    ----------
    folder : str or Path
        Session folder containing Velox EMD files.
    max_rotation_tolerance_deg : float, optional
        Angular tolerance around 0 and ±90 degrees. Default is 5 degrees.

    Returns
    -------
    pandas.DataFrame
        Acquisition metadata and pair assignments. ``pair_order=0`` marks the
        scan closest to 0 degrees and ``pair_order=1`` its orthogonal partner.

    Examples
    --------
    >>> pairs = scan_pairs("~/data/session")
    >>> pairs[pairs.pair_order == 0][["file", "partner"]]
    """
    import pandas as pd

    from quantem.core.io.file_readers import read_emd_metadata

    folder = Path(folder).expanduser()
    records = []
    for path in sorted(folder.iterdir()):
        if path.name.startswith("._"):
            continue
        if path.suffix.lower() == ".npy":
            records.append({"file": path.name, "shape": tuple(np.load(path, mmap_mode="r").shape)})
            continue
        if path.suffix.lower() != ".emd":
            continue

        metadata = read_emd_metadata(path)
        shape = metadata["scan_shape"]
        pixel_size_nm = metadata["pixel_size_nm"]
        stage = metadata["stage_xy_m"]
        records.append(
            {
                "file": path.name,
                "shape": shape,
                "pixel_size_nm": pixel_size_nm,
                "fov_nm": None if metadata["fov_m"] is None else metadata["fov_m"] * 1e9,
                "magnification": metadata["magnification"],
                "rotation_deg": metadata["scan_rotation_deg"],
                "stage_x_m": None if stage is None else stage[0],
                "stage_y_m": None if stage is None else stage[1],
                "fov_m": metadata["fov_m"],
                "acquired": metadata["acquisition_timestamp"],
                "acquisition_context": metadata.get("acquisition_context", "image"),
            }
        )

    table = pd.DataFrame(records)
    if table.empty:
        return table
    for column in (
        "pixel_size_nm",
        "fov_nm",
        "magnification",
        "rotation_deg",
        "stage_x_m",
        "stage_y_m",
        "fov_m",
        "acquired",
        "acquisition_context",
    ):
        if column not in table:
            table[column] = None

    table = table.reset_index(drop=True)
    table["pair"] = ""
    table["partner"] = ""
    table["pair_order"] = pd.array([pd.NA] * len(table), dtype="Int64")
    table["partner_rotation_deg"] = np.nan
    table["relative_partner_rotation_deg"] = np.nan
    table["stage_distance_nm"] = np.nan
    table["pair_tolerance_nm"] = np.nan
    table["pair_status"] = "not_included"
    table["pair_reason"] = ""

    tolerance = float(max_rotation_tolerance_deg)
    zero_indices = [
        index
        for index, angle in table.rotation_deg.items()
        if pd.notna(angle)
        and abs(float(angle)) < tolerance
        and table.at[index, "acquisition_context"] != "spectrum_image"
    ]
    ninety_indices = [
        index
        for index, angle in table.rotation_deg.items()
        if pd.notna(angle) and abs(abs(float(angle)) - 90.0) < tolerance
        and table.at[index, "acquisition_context"] != "spectrum_image"
    ]
    candidates_by_zero = {zero: [] for zero in zero_indices}
    zeros_by_ninety = {ninety: [] for ninety in ninety_indices}
    for zero in zero_indices:
        for ninety in ninety_indices:
            incompatible = False
            if table.at[zero, "shape"] and table.at[ninety, "shape"]:
                incompatible = tuple(table.at[zero, "shape"]) != tuple(
                    table.at[ninety, "shape"]
                )
            for column, relative_tolerance in (
                ("pixel_size_nm", 0.02),
                ("fov_m", 0.02),
                ("magnification", 0.02),
            ):
                first, second = table.loc[[zero, ninety], column]
                if pd.notna(first) and pd.notna(second):
                    scale = max(abs(float(first)), abs(float(second)), 1e-30)
                    incompatible |= abs(float(first) - float(second)) > (
                        relative_tolerance * scale
                    )
            if incompatible:
                continue
            stage_values = table.loc[
                [zero, ninety], ["stage_x_m", "stage_y_m"]
            ].to_numpy(dtype=float)
            if np.isfinite(stage_values).all():
                distance = float(np.linalg.norm(stage_values[0] - stage_values[1]))
                fovs = [
                    float(value)
                    for value in table.loc[[zero, ninety], "fov_m"]
                    if pd.notna(value) and float(value) > 0
                ]
                pair_tolerance = max(0.25 * min(fovs), 10e-9) if fovs else 10e-9
                if distance > pair_tolerance:
                    continue
            else:
                distance, pair_tolerance = float("inf"), float("nan")
            candidate = (distance, ninety, pair_tolerance)
            candidates_by_zero[zero].append(candidate)
            zeros_by_ninety[ninety].append((distance, zero, pair_tolerance))

    for index, angle in table.rotation_deg.items():
        if table.at[index, "acquisition_context"] == "spectrum_image":
            table.at[index, "pair_reason"] = (
                "Spectrum-image acquisition belongs in the EDS/EELS reference workflow."
            )
        elif pd.isna(angle):
            table.at[index, "pair_reason"] = "Missing scan-rotation metadata."
        elif index not in zero_indices and index not in ninety_indices:
            table.at[index, "pair_reason"] = (
                f"Scan rotation is not within {tolerance:g}° of 0° or ±90°."
            )

    pair_count = 0
    for zero in zero_indices:
        candidates = candidates_by_zero[zero]
        if not candidates:
            table.at[zero, "pair_reason"] = (
                "No orthogonal scan has compatible shape, calibration, field of view, and stage position."
            )
            continue
        if len(candidates) > 1:
            table.at[zero, "pair_reason"] = (
                f"Ambiguous: {len(candidates)} compatible ±90° scans match this 0° acquisition."
            )
            for _, ninety, _ in candidates:
                table.at[ninety, "pair_reason"] = (
                    "Ambiguous: this ±90° scan is one of multiple candidates for the same 0° acquisition."
                )
            continue

        distance, ninety, pair_tolerance = candidates[0]
        reverse_candidates = zeros_by_ninety[ninety]
        if len(reverse_candidates) != 1:
            table.at[zero, "pair_reason"] = (
                "Ambiguous: the compatible ±90° scan also matches multiple 0° acquisitions."
            )
            table.at[ninety, "pair_reason"] = (
                f"Ambiguous: {len(reverse_candidates)} compatible 0° scans match this ±90° acquisition."
            )
            continue

        pair_count += 1
        pair_name = f"P{pair_count:02d}"
        rotations = table.loc[[zero, ninety], "rotation_deg"].astype(float).to_numpy()
        table.loc[[zero, ninety], "pair"] = pair_name
        table.loc[[zero, ninety], "pair_status"] = "confident"
        table.loc[[zero, ninety], "pair_reason"] = ""
        table.at[zero, "partner"] = table.at[ninety, "file"]
        table.at[ninety, "partner"] = table.at[zero, "file"]
        table.at[zero, "pair_order"] = 0
        table.at[ninety, "pair_order"] = 1
        table.at[zero, "partner_rotation_deg"] = rotations[1]
        table.at[ninety, "partner_rotation_deg"] = rotations[0]
        table.at[zero, "relative_partner_rotation_deg"] = (
            rotations[1] - rotations[0] + 180.0
        ) % 360.0 - 180.0
        table.at[ninety, "relative_partner_rotation_deg"] = (
            rotations[0] - rotations[1] + 180.0
        ) % 360.0 - 180.0
        if np.isfinite(distance):
            table.loc[[zero, ninety], "stage_distance_nm"] = distance * 1e9
            table.loc[[zero, ninety], "pair_tolerance_nm"] = pair_tolerance * 1e9

    for ninety in ninety_indices:
        if not table.at[ninety, "pair"] and not table.at[ninety, "pair_reason"]:
            table.at[ninety, "pair_reason"] = (
                "No 0° scan has compatible shape, calibration, field of view, and stage position."
            )

    return table.sort_values("acquired", na_position="last").reset_index(drop=True)
