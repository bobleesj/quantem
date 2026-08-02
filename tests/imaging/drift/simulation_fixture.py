"""Forward models for synthetic drift-correction tests.

These helpers create known scan distortions from clean data so correction can
be checked against controlled ground truth without expanding the production API.

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

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


def make_synthetic_drift_data(scale=1, seed=42):
    """Generate a chevron image and two orthogonally drifted scans.

    The two scans share linear row/column drift and independent scanline
    jitter. This gives the affine and non-rigid workflows a deterministic
    distortion with known clean-image ground truth.
    """
    np.random.seed(seed)
    shape = (200 * scale, 200 * scale)
    row_grid, col_grid = np.meshgrid(
        np.arange(-shape[0] / 2, shape[0] / 2),
        np.arange(-shape[0] / 2, shape[0] / 2),
        indexing="ij",
    )
    base_image = (
        np.mod(np.abs(row_grid) + np.abs(col_grid), 16 * scale) < 8 * scale
    ).astype(float)
    base_image[np.logical_and(row_grid > 0, col_grid > 0)] += 0.5
    base_image[np.maximum(np.abs(row_grid), np.abs(col_grid)) < 20 * scale] = 2
    base_image = gaussian_filter(base_image, sigma=0.667 * scale)

    scan_size = 128 * scale
    scan_positions = np.arange(scan_size)
    row_drift = scan_positions * 0.001 * scale
    col_drift = scan_positions * 0.1 * scale
    jitter0 = np.random.randn(2, scan_size) * 0.5 * scale
    jitter1 = np.random.randn(2, scan_size) * 0.5 * scale

    image_0 = np.zeros((scan_size, scan_size))
    image_1 = np.zeros((scan_size, scan_size))
    for row_index in range(scan_size):
        row_0 = 40 * scale + row_index + row_drift[row_index] + jitter0[0, row_index]
        col_0 = 30 * scale + col_drift[row_index] + jitter0[1, row_index]
        image_0[row_index] = bilinear_sample(
            base_image,
            row_0 + scan_positions * 0,
            col_0 + scan_positions,
        )

        row_1 = 170 * scale + row_drift[row_index] + jitter1[0, row_index]
        col_1 = 30 * scale + row_index + col_drift[row_index] + jitter1[1, row_index]
        image_1[row_index] = bilinear_sample(
            base_image,
            row_1 - scan_positions,
            col_1 + scan_positions * 0,
        )
    return image_0, image_1, base_image


def bilinear_sample(image, row, column):
    """Sample a synthetic image at floating-point row/column coordinates."""
    row = np.clip(row, 0, image.shape[0] - 2)
    column = np.clip(column, 0, image.shape[1] - 2)
    row_floor = np.floor(row).astype(int)
    column_floor = np.floor(column).astype(int)
    row_fraction = row - row_floor
    column_fraction = column - column_floor
    return (
        image[row_floor, column_floor]
        * (1 - row_fraction)
        * (1 - column_fraction)
        + image[row_floor + 1, column_floor]
        * row_fraction
        * (1 - column_fraction)
        + image[row_floor, column_floor + 1]
        * (1 - row_fraction)
        * column_fraction
        + image[row_floor + 1, column_floor + 1]
        * row_fraction
        * column_fraction
    )


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
) -> dict[str, object]:
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
