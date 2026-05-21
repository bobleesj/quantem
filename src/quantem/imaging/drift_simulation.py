"""Forward models for synthetic scan drift experiments.

These helpers generate drifted scan-axis-leading datasets from a clean
reference dataset. They are intentionally separate from drift correction:
the functions here create known synthetic acquisitions so correction and
ptychography code can be tested against a controlled ground truth.

Conventions
-----------
All drift vectors use ``(down_px, right_px)`` order in a shared specimen
row/column frame. A positive lab-frame drift means the specimen moved
down/right during the scan, so the clean data are sampled at
``rotated_scan_position - drift``. The detector axes are never shifted,
rolled, or warped. Subpixel drift is handled by interpolating over the
scan axes, which can mix neighboring whole diffraction patterns but does
not resample pixels inside any diffraction pattern.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _right_angle_index(scan_direction_degrees: float) -> int:
    """Return ``scan_direction_degrees / 90`` modulo 4.

    The synthetic forward model is intentionally limited to right-angle
    raster scans because that is the microscope use case we want to test
    here: 0, 90, -90, and 180 degree scan directions.
    """
    turns = round(float(scan_direction_degrees) / 90.0)
    if not np.isclose(float(scan_direction_degrees), 90.0 * turns, atol=1e-6):
        raise ValueError(
            "scan_direction_degrees must be one of the right-angle scan "
            f"directions 0, 90, -90, or 180; got {scan_direction_degrees!r}"
        )
    return int(turns) % 4


def rotated_scan_positions(
    scan_shape: tuple[int, int],
    scan_direction_degrees: float = 0.0,
) -> np.ndarray:
    """Return raw scan positions in a shared row/column specimen frame.

    Parameters
    ----------
    scan_shape : tuple[int, int]
        Scan shape ``(scan_rows, scan_cols)``.
    scan_direction_degrees : float
        Right-angle scan direction. ``0`` returns the usual raster grid.
        ``90`` means the raw image appears counterclockwise relative to the
        ``0`` degree image. A raw 90 degree image therefore rotates back into
        the image 0 display frame with ``np.rot90(image, k=-1)``.

    Returns
    -------
    np.ndarray
        Position map with shape ``(scan_rows, scan_cols, 2)`` in
        row/column scan-pixel units.
    """
    scan_h, scan_w = map(int, scan_shape)
    row, col = np.meshgrid(
        np.arange(scan_h, dtype=np.float32),
        np.arange(scan_w, dtype=np.float32),
        indexing="ij",
    )
    angle = _right_angle_index(scan_direction_degrees)
    if angle == 0:
        pos_row, pos_col = row, col
    elif angle == 1:
        if scan_h != scan_w:
            raise ValueError("90 degree synthetic scan geometry requires a square scan")
        pos_row, pos_col = col, scan_w - 1 - row
    elif angle == 2:
        pos_row, pos_col = scan_h - 1 - row, scan_w - 1 - col
    else:
        if scan_h != scan_w:
            raise ValueError("-90 degree synthetic scan geometry requires a square scan")
        pos_row, pos_col = scan_h - 1 - col, row
    return np.stack([pos_row, pos_col], axis=-1).astype(np.float32)


def scan_time_drift_field(
    scan_shape: tuple[int, int],
    *,
    drift_per_scanline_px: tuple[float, float] = (0.001, 0.1),
    total_drift_px: tuple[float, float] | None = None,
    jitter_sigma_px: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Build an original-style scan-time drift field.

    Drift is one row/column vector per slow-scan line, broadcast across the
    fast axis. This mirrors ``notebooks/drift/reference/drift_original.ipynb``:
    the whole fast-scan line is shifted by the drift accumulated at that
    scan time.

    Parameters
    ----------
    scan_shape : tuple[int, int]
        Scan shape ``(scan_rows, scan_cols)``.
    drift_per_scanline_px : tuple[float, float]
        Row/column drift added per slow-scan line when ``total_drift_px`` is
        not supplied.
    total_drift_px : tuple[float, float] or None
        Total row/column drift from the first to last scan line. If supplied,
        this overrides ``drift_per_scanline_px``.
    jitter_sigma_px : float
        Optional independent Gaussian line jitter in scan pixels.
    seed : int or None
        Random seed for jitter.

    Returns
    -------
    np.ndarray
        Drift field with shape ``(scan_rows, scan_cols, 2)`` in
        ``(down_px, right_px)`` order.
    """
    scan_h, scan_w = map(int, scan_shape)
    slow_line = np.arange(scan_h, dtype=np.float32)
    if total_drift_px is None:
        per_line = np.asarray(drift_per_scanline_px, dtype=np.float32)
        drift_line = slow_line[:, None] * per_line[None, :]
    else:
        total = np.asarray(total_drift_px, dtype=np.float32)
        denom = max(scan_h - 1, 1)
        drift_line = (slow_line / denom)[:, None] * total[None, :]

    if jitter_sigma_px:
        rng = np.random.default_rng(seed)
        drift_line = drift_line + rng.normal(
            0.0, float(jitter_sigma_px), drift_line.shape,
        ).astype(np.float32)

    return np.broadcast_to(
        drift_line[:, None, :], (scan_h, scan_w, 2),
    ).copy().astype(np.float32)


def integrate_virtual_detector_image(
    data: np.ndarray | torch.Tensor,
    detector_mask: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Integrate a detector mask into a scan-space virtual image.

    The first two axes are treated as scan axes. The mask must match the
    trailing detector axes. This is intentionally small: it is the virtual
    bright-field/dark-field operation used by synthetic 4D-STEM drift tests.
    """
    mask_shape = tuple(detector_mask.shape)
    if len(mask_shape) == 0:
        raise ValueError("detector_mask must have at least one dimension")
    if tuple(data.shape[-len(mask_shape):]) != mask_shape:
        raise ValueError(
            "detector_mask shape must match the trailing data axes, got "
            f"{mask_shape} for data shape {tuple(data.shape)}"
        )

    if isinstance(data, torch.Tensor):
        mask = torch.as_tensor(detector_mask, device=data.device, dtype=torch.bool).reshape(-1)
        flat = data.reshape(*data.shape[:-len(mask_shape)], -1)
        return flat[..., mask].sum(dim=-1)

    mask = np.asarray(detector_mask, dtype=bool).reshape(-1)
    flat = np.asarray(data).reshape(*data.shape[:-len(mask_shape)], -1)
    return np.asarray(flat[..., mask].sum(axis=-1), dtype=np.float32)


def raw_raster_drift_effect(
    drift_field_px: np.ndarray | torch.Tensor,
    scan_direction_degrees: float = 0.0,
) -> np.ndarray:
    """Return apparent drift displacement in the raw raster display frame.

    ``drift_field_px`` is in lab-frame ``(down_px, right_px)`` order. The
    returned array has the same shape and uses raw-display
    ``(down_px, right_px)`` order. This is the apparent displacement of image
    features in the raw raster. The correction offset that cancels this drift
    has the opposite sign.
    """
    drift = np.asarray(
        drift_field_px.detach().cpu().numpy() if isinstance(drift_field_px, torch.Tensor) else drift_field_px,
        dtype=np.float32,
    )
    if drift.ndim < 1 or drift.shape[-1] != 2:
        raise ValueError(
            f"drift_field_px must have trailing dimension 2, got shape {drift.shape}"
        )

    angle = _right_angle_index(scan_direction_degrees)
    down = drift[..., 0]
    right = drift[..., 1]
    if angle == 0:
        effect = np.stack([down, right], axis=-1)
    elif angle == 1:
        effect = np.stack([-right, down], axis=-1)
    elif angle == 2:
        effect = np.stack([-down, -right], axis=-1)
    else:
        effect = np.stack([right, -down], axis=-1)
    return effect.astype(np.float32, copy=False)


def valid_scan_position_mask(
    positions_px: np.ndarray | torch.Tensor,
    source_shape: tuple[int, int],
) -> np.ndarray:
    """Return where probe positions fall inside a source scan field.

    Parameters
    ----------
    positions_px : ndarray or Tensor
        Probe positions with trailing dimension ``(down_px, right_px)``.
    source_shape : tuple[int, int]
        Source scan shape ``(rows, cols)`` that can be sampled.

    Returns
    -------
    np.ndarray
        Boolean mask with shape ``positions_px.shape[:-1]``. A true value
        means the corresponding probe position is inside the source scan.
    """
    positions = np.asarray(
        positions_px.detach().cpu().numpy()
        if isinstance(positions_px, torch.Tensor)
        else positions_px,
        dtype=np.float32,
    )
    if positions.ndim < 1 or positions.shape[-1] != 2:
        raise ValueError(
            f"positions_px must have trailing dimension 2, got shape {positions.shape}"
        )
    source_h, source_w = map(int, source_shape)
    return (
        (positions[..., 0] >= 0)
        & (positions[..., 0] <= source_h - 1)
        & (positions[..., 1] >= 0)
        & (positions[..., 1] <= source_w - 1)
    )


def find_valid_square_scan_crop(
    valid_mask: np.ndarray | torch.Tensor,
    size: int,
    *,
    target_origin: tuple[float, float] | None = None,
) -> tuple[slice, slice]:
    """Find the valid square scan crop closest to a target origin.

    This is the crop selector used by known-drift forward-model exports.
    It keeps the policy in QuantEM instead of notebook code: choose a
    square crop whose every scan position is valid, preferring the crop
    closest to the centered scan window unless ``target_origin`` is given.

    Parameters
    ----------
    valid_mask : ndarray or Tensor
        Boolean map of valid raw scan pixels.
    size : int
        Square crop size in scan pixels.
    target_origin : tuple[float, float], optional
        Preferred ``(row_start, col_start)`` for the returned crop.

    Returns
    -------
    tuple[slice, slice]
        Row and column slices for the selected crop.
    """
    valid = np.asarray(
        valid_mask.detach().cpu().numpy()
        if isinstance(valid_mask, torch.Tensor)
        else valid_mask,
        dtype=bool,
    )
    if valid.ndim != 2:
        raise ValueError(f"valid_mask must be 2-D, got shape {valid.shape}")
    crop_size = int(size)
    if crop_size < 1:
        raise ValueError("size must be >= 1")
    scan_h, scan_w = valid.shape
    if crop_size > scan_h or crop_size > scan_w:
        raise ValueError(
            f"size {crop_size} cannot exceed valid_mask shape {valid.shape}"
        )
    if target_origin is None:
        target_row = (scan_h - crop_size) / 2.0
        target_col = (scan_w - crop_size) / 2.0
    else:
        target_row, target_col = map(float, target_origin)

    valid_u8 = valid.astype(np.uint8, copy=False)
    integral = np.pad(valid_u8, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    needed = crop_size * crop_size
    best: tuple[slice, slice] | None = None
    best_score: float | None = None
    for row0 in range(0, scan_h - crop_size + 1):
        row1 = row0 + crop_size
        for col0 in range(0, scan_w - crop_size + 1):
            col1 = col0 + crop_size
            count = (
                integral[row1, col1]
                - integral[row0, col1]
                - integral[row1, col0]
                + integral[row0, col0]
            )
            if count != needed:
                continue
            score = (row0 - target_row) ** 2 + (col0 - target_col) ** 2
            if best is None or score < best_score:
                best = (slice(row0, row1), slice(col0, col1))
                best_score = float(score)
    if best is None:
        raise RuntimeError(f"could not find a valid {crop_size} x {crop_size} crop")
    return best


def plot_lab_drift_vectors(
    sample: np.ndarray,
    labels: list[str] | tuple[str, ...],
    total_drifts_px: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    *,
    scan_shape: tuple[int, int],
    scan_origin_px: tuple[float, float] | None = None,
    axsize: tuple[float, float] = (5.0, 5.0),
    cmap: str = "gray",
):
    """Plot lab-frame drift arrows on the full physical sample.

    ``total_drifts_px`` is in ``(down_px, right_px)`` order.
    """
    import matplotlib.pyplot as plt

    sample_np = np.asarray(sample)
    if len(labels) != len(total_drifts_px):
        raise ValueError("labels and total_drifts_px must have the same length")
    scan_h, scan_w = map(int, scan_shape)
    if scan_origin_px is None:
        origin = ((sample_np.shape[0] - scan_h) / 2.0, (sample_np.shape[1] - scan_w) / 2.0)
    else:
        origin = tuple(map(float, scan_origin_px))

    fig, axes = plt.subplots(
        1, len(labels), figsize=(axsize[0] * len(labels), axsize[1]), constrained_layout=True
    )
    axes_arr = np.asarray(axes).reshape(-1)
    row0, col0 = origin
    for ax, label, drift in zip(axes_arr, labels, total_drifts_px):
        down_px, right_px = map(float, drift)
        ax.imshow(sample_np, cmap=cmap, origin="upper")
        ax.add_patch(
            plt.Rectangle((col0, row0), scan_w, scan_h, fill=False, color="tab:blue", linewidth=2)
        )
        start_col = col0 + 0.11 * scan_w
        start_row = row0 + 0.11 * scan_h
        if down_px or right_px:
            ax.arrow(
                start_col,
                start_row,
                right_px,
                down_px,
                color="tab:red",
                width=0.8,
                head_width=4,
                length_includes_head=True,
            )
            ax.text(start_col + right_px + 3, start_row + down_px + 3, "lab drift", color="tab:red")
        else:
            ax.text(start_col, start_row + 4, "no lab drift", color="tab:red")
        ax.set_title(f"{label}\nphysical sample frame: down={down_px:.0f}, right={right_px:.0f} px")
        ax.set_xlim(0, sample_np.shape[1] - 1)
        ax.set_ylim(sample_np.shape[0] - 1, 0)
        ax.set_aspect("equal")
        ax.set_xlabel("right / column")
        ax.set_ylabel("down / row")
    return fig, axes_arr


def plot_raw_raster_drift_effects(
    backgrounds: list[list[np.ndarray]] | tuple[tuple[np.ndarray, ...], ...],
    drift_fields_px: list[np.ndarray] | tuple[np.ndarray, ...],
    row_labels: list[str] | tuple[str, ...],
    *,
    scan_direction_degrees: tuple[float, ...] = (0.0, 90.0),
    stride: int = 8,
    axsize: tuple[float, float] = (5.0, 5.0),
    cmap: str = "gray",
):
    """Plot apparent raw-raster drift vectors over virtual images.

    ``backgrounds[row][col]`` should match ``row_labels[row]`` and
    ``scan_direction_degrees[col]``. This is meant for synthetic drift
    diagnostics: image 0 and image 1 can use the same lab drift field but show
    different raw-raster drift-vector directions. Red arrows show apparent
    raw-image feature drift. The small blue arrow in each panel shows the
    opposite correction direction that would cancel that apparent drift.
    """
    import matplotlib.pyplot as plt

    if stride < 1:
        raise ValueError("stride must be >= 1")
    if len(backgrounds) != len(drift_fields_px) or len(backgrounds) != len(row_labels):
        raise ValueError("backgrounds, drift_fields_px, and row_labels must have the same length")
    n_rows = len(backgrounds)
    n_cols = len(scan_direction_degrees)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(axsize[0] * n_cols, axsize[1] * n_rows), constrained_layout=True
    )
    axes_arr = np.asarray(axes).reshape(n_rows, n_cols)
    for row_idx, (row_backgrounds, drift, label) in enumerate(
        zip(backgrounds, drift_fields_px, row_labels)
    ):
        if len(row_backgrounds) != n_cols:
            raise ValueError("each background row must match scan_direction_degrees length")
        for col_idx, angle in enumerate(scan_direction_degrees):
            ax = axes_arr[row_idx, col_idx]
            background = np.asarray(row_backgrounds[col_idx])
            effect = raw_raster_drift_effect(drift, angle)
            rr = np.arange(0, effect.shape[0], stride)
            cc = np.arange(0, effect.shape[1], stride)
            grid_r, grid_c = np.meshgrid(rr, cc, indexing="ij")
            ax.imshow(background, cmap=cmap, origin="upper", alpha=0.55)
            ax.quiver(
                grid_c,
                grid_r,
                effect[grid_r, grid_c, 1],
                effect[grid_r, grid_c, 0],
                angles="xy",
                scale_units="xy",
                scale=1,
                color="tab:red",
                width=0.004,
            )
            center_r = background.shape[0] * 0.16
            center_c = background.shape[1] * 0.16
            mean_effect = effect.reshape(-1, 2).mean(axis=0)
            if np.linalg.norm(mean_effect) > 0:
                ax.arrow(
                    center_c,
                    center_r,
                    -mean_effect[1],
                    -mean_effect[0],
                    color="tab:blue",
                    width=0.5,
                    head_width=3,
                    length_includes_head=True,
                )
                ax.text(
                    center_c - mean_effect[1] + 2,
                    center_r - mean_effect[0] + 2,
                    "correction",
                    color="tab:blue",
                )
            max_offset = np.linalg.norm(effect, axis=-1).max()
            ax.set_title(
                f"{label}\n{angle:g} degree apparent raw drift, max={max_offset:.2f} px"
            )
            ax.set_aspect("equal")
            ax.set_xlim(0, background.shape[1] - 1)
            ax.set_ylim(background.shape[0] - 1, 0)
            ax.set_xlabel("right / column")
            ax.set_ylabel("down / row")
    return fig, axes_arr


def correct_scalar_image_from_positions(
    image: np.ndarray | torch.Tensor,
    positions_px: np.ndarray | torch.Tensor,
    *,
    output_shape: tuple[int, int],
    output_origin_px: tuple[float, float] = (0.0, 0.0),
    device: str | torch.device | None = None,
) -> dict[str, np.ndarray | torch.Tensor]:
    """Splat a raw scalar scan image onto a known physical-position grid.

    This is the scalar equivalent of correcting a 4D-STEM dataset by known
    probe positions and then integrating a detector mask. It is useful for
    synthetic parity checks because detector integration, bilinear scan
    correction, and scalar averaging are linear.

    Parameters
    ----------
    image : ndarray or Tensor
        Raw scalar scan image, such as BF or DF, with shape ``(rows, cols)``.
    positions_px : ndarray or Tensor
        Physical sample positions for each raw scan pixel, shape
        ``(rows, cols, 2)`` in ``(down_px, right_px)`` order.
    output_shape : tuple[int, int]
        Output corrected image shape.
    output_origin_px : tuple[float, float]
        Physical sample coordinate of output pixel ``(0, 0)``.
    device : str or torch.device, optional
        Device used for the splat. Defaults to the input tensor device, CUDA
        when available, otherwise CPU.

    Returns
    -------
    dict
        ``image`` is the corrected scalar image. ``weight`` is the bilinear
        coverage weight used for normalization.
    """
    if tuple(image.shape) != tuple(positions_px.shape[:2]):
        raise ValueError(
            f"image shape {tuple(image.shape)} must match positions shape {tuple(positions_px.shape[:2])}"
        )
    input_is_torch = isinstance(image, torch.Tensor)
    if device is None:
        if input_is_torch:
            device_t = image.device
        else:
            device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    image_t = torch.as_tensor(image, device=device_t, dtype=torch.float32)
    positions_t = torch.as_tensor(positions_px, device=device_t, dtype=torch.float32)
    origin_t = torch.as_tensor(output_origin_px, device=device_t, dtype=torch.float32)
    out_h, out_w = map(int, output_shape)

    row = positions_t[..., 0] - origin_t[0]
    col = positions_t[..., 1] - origin_t[1]
    row0 = torch.floor(row)
    col0 = torch.floor(col)
    drow = row - row0
    dcol = col - col0
    row0_i = row0.to(torch.int64)
    col0_i = col0.to(torch.int64)

    values = image_t.reshape(-1)
    image_out = torch.zeros(out_h * out_w, device=device_t, dtype=torch.float32)
    weight_out = torch.zeros_like(image_out)

    for dr, dc, w in (
        (0, 0, (1.0 - drow) * (1.0 - dcol)),
        (0, 1, (1.0 - drow) * dcol),
        (1, 0, drow * (1.0 - dcol)),
        (1, 1, drow * dcol),
    ):
        rr = row0_i + dr
        cc = col0_i + dc
        valid = (rr >= 0) & (rr < out_h) & (cc >= 0) & (cc < out_w)
        idx = (rr * out_w + cc).reshape(-1)
        ww = w.reshape(-1)
        valid_flat = valid.reshape(-1)
        if valid_flat.any():
            idx_valid = idx[valid_flat]
            weighted = values[valid_flat] * ww[valid_flat]
            image_out.scatter_add_(0, idx_valid, weighted)
            weight_out.scatter_add_(0, idx_valid, ww[valid_flat])

    valid_weight = weight_out > 0
    corrected = torch.zeros_like(image_out)
    corrected[valid_weight] = image_out[valid_weight] / weight_out[valid_weight]
    corrected = corrected.reshape(out_h, out_w)
    weight = weight_out.reshape(out_h, out_w)

    if input_is_torch:
        return {"image": corrected, "weight": weight}
    return {
        "image": corrected.cpu().numpy().astype(np.float32, copy=False),
        "weight": weight.cpu().numpy().astype(np.float32, copy=False),
    }


@torch.inference_mode()
def simulate_drifted_4dstem(
    dataset: np.ndarray | torch.Tensor,
    *,
    scan_shape: tuple[int, int] | None = None,
    scan_origin_px: tuple[float, float] | None = None,
    scan_direction_degrees: float = 0.0,
    drift_field_px: np.ndarray | torch.Tensor | None = None,
    drift_per_scanline_px: tuple[float, float] = (0.001, 0.1),
    total_drift_px: tuple[float, float] | None = None,
    jitter_sigma_px: float = 0.0,
    seed: int | None = None,
    mode: str = "bilinear",
    channel_chunk: int | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Generate a synthetic drifted 4D-STEM acquisition.

    The input ``dataset`` is a clean scan-axis-leading dataset
    ``(source_rows, source_cols, ...detector_or_channel_axes)``. Each raw
    scan pixel contains the diffraction pattern sampled from the clean
    dataset at the rotated and drifted row/column probe position.

    By default the output scan shape matches the source real-space shape.
    Supplying ``scan_shape`` and ``scan_origin_px`` simulates a smaller scan
    window inside a larger physical specimen field. If ``scan_shape`` is
    supplied and ``scan_origin_px`` is omitted, the scan window is centered
    in the source field.

    Detector pixels are not shifted. The interpolation is only over the
    real-space scan axes.

    Returns a dictionary with:

    ``data``
        Drifted 4D-STEM data as float32. Numpy input returns numpy output;
        torch input returns a torch tensor.
    ``positions``
        Actual probe positions in the shared row/column specimen frame.
    ``positions_offset_px``
        ``positions - nominal_raw_raster``. This is the value passed to
        iterative ptychography for drift-updated probe positions.
    ``drift_field_px``
        The lab-frame sample drift field used for synthesis, in
        ``(down_px, right_px)`` order. A positive down/right drift means the
        sample moved down/right in the lab frame, so the clean data are
        sampled at ``rotated_scan_position - drift``.
    """
    if mode not in {"bilinear", "nearest", "bicubic"}:
        raise ValueError(f"mode must be 'bilinear', 'nearest', or 'bicubic', got {mode!r}")
    if len(dataset.shape) < 3:
        raise ValueError(f"dataset must have at least 3 dimensions, got {tuple(dataset.shape)}")

    input_is_torch = isinstance(dataset, torch.Tensor)
    original_shape = tuple(dataset.shape)
    source_h, source_w = original_shape[:2]
    detector_shape = original_shape[2:]
    output_scan_shape = tuple(map(int, scan_shape or original_shape[:2]))
    scan_h, scan_w = output_scan_shape
    n_channels = int(np.prod(detector_shape))

    if scan_origin_px is None:
        origin = np.array(
            [(source_h - scan_h) / 2.0, (source_w - scan_w) / 2.0],
            dtype=np.float32,
        )
    else:
        origin = np.asarray(scan_origin_px, dtype=np.float32)
    if origin.shape != (2,):
        raise ValueError(f"scan_origin_px must have shape (2,), got {origin.shape}")

    nominal = rotated_scan_positions(output_scan_shape, 0.0) + origin
    rotated = rotated_scan_positions(output_scan_shape, scan_direction_degrees) + origin
    if drift_field_px is None:
        drift_field = scan_time_drift_field(
            output_scan_shape,
            drift_per_scanline_px=drift_per_scanline_px,
            total_drift_px=total_drift_px,
            jitter_sigma_px=jitter_sigma_px,
            seed=seed,
        )
    else:
        drift_field = np.asarray(
            drift_field_px.detach().cpu().numpy() if isinstance(drift_field_px, torch.Tensor) else drift_field_px,
            dtype=np.float32,
        )
    if drift_field.shape != (scan_h, scan_w, 2):
        raise ValueError(
            f"drift_field_px shape must be {(scan_h, scan_w, 2)}, got {drift_field.shape}"
        )
    positions = (rotated - drift_field).astype(np.float32)
    positions_offset_px = (positions - nominal).astype(np.float32)

    if device is None:
        if input_is_torch:
            device_t = dataset.device
        else:
            device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    source = (
        dataset.reshape(source_h, source_w, n_channels)
        if input_is_torch
        else torch.as_tensor(dataset.reshape(source_h, source_w, n_channels))
    ).to(device=device_t, dtype=torch.float32)

    row = torch.as_tensor(positions[..., 0], device=device_t, dtype=torch.float32)
    col = torch.as_tensor(positions[..., 1], device=device_t, dtype=torch.float32)
    grid = torch.stack(
        [
            2.0 * col / (source_w - 1) - 1.0,
            2.0 * row / (source_h - 1) - 1.0,
        ],
        dim=-1,
    )[None]

    if channel_chunk is None:
        channel_chunk = n_channels
    out = torch.empty(scan_h, scan_w, n_channels, device=device_t, dtype=torch.float32)
    for start in range(0, n_channels, channel_chunk):
        end = min(start + channel_chunk, n_channels)
        chunk = source[:, :, start:end].permute(2, 0, 1).contiguous()[None]
        sampled = F.grid_sample(
            chunk,
            grid,
            mode=mode,
            padding_mode="border",
            align_corners=True,
        )[0].permute(1, 2, 0)
        out[:, :, start:end] = sampled

    data = out.reshape(output_scan_shape + detector_shape)
    if not input_is_torch:
        data = data.cpu().numpy().astype(np.float32, copy=False)

    return {
        "data": data,
        "positions": positions,
        "positions_offset_px": positions_offset_px,
        "drift_field_px": drift_field,
    }
