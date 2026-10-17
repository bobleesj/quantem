from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import torch
from matplotlib.patches import Polygon

from quantem.core.visualization import show_2d


# TODO update sampling to allow for 3D and to plot with appropriate units along row/col/depth
# TODO proper normalization for line going through edges/corner with finite linewidth
def linescan(
    image: np.ndarray | torch.Tensor,
    center: tuple[int, int] | None = None,
    phi: float = 0,
    linewidth: int = 1,
    line_len: int = -1,
    show: bool = False,
    sampling: tuple[float, float] | np.ndarray | None = None,
    sampling_units: str | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a line scan through an image.

    Torch input is sampled on its current device and returns Torch positions
    and profiles. With show=True, only the mean image and profile transfer
    to Matplotlib. NumPy input retains the established array behavior.

    Args:
        image (np.ndarray | torch.Tensor): 2D or 3D image array. For 3D arrays, shape should be (depth, num_rows, num_cols).
        center (tuple, optional): Center point for the line scan (center_row, center_col). Defaults to image center.
        phi (float): Angle for the line scan in degrees.
        linewidth (int): Line width for the scan.
        line_len (int): Length of the line scan. If -1, uses full intersection with image bounds.
        show (bool): Whether to display the plot.
        **kwargs: Additional keyword arguments for plotting.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (positions, profile) where:
            - positions: 1D array of positions along the line, units are pixels if sampling is None
             otherwise units are the same as the sampling
            - profile: 1D array (for 2D input) or 2D array (for 3D input) of intensity values
    """
    if isinstance(image, torch.Tensor):
        return _linescan_tensor(
            image, center, phi, linewidth, line_len, show, sampling, sampling_units, **kwargs
        )
    image = np.asarray(image)

    if image.ndim == 2:
        scan_image = image
        image_shape = image.shape
    elif image.ndim == 3:
        # For 3D input, we'll scan along each depth slice
        scan_image = np.mean(image, axis=0)  # Use mean for visualization
        image_shape = image.shape[1:]  # (num_rows, num_cols)
    else:
        raise ValueError(f"Input image must be 2D or 3D, got {image.ndim}D")

    num_rows, num_cols = image_shape

    if center is None:
        center = (num_rows // 2, num_cols // 2)

    center_row, center_col = int(round(center[0])), int(round(center[1]))

    # Calculate start and end points
    start_point, end_point = _calculate_line_endpoints(
        image_shape, (center_row, center_col), phi, line_len
    )

    if image.ndim == 2:
        # For 2D images, use the original approach
        profile = skimage.measure.profile_line(
            scan_image,
            start_point,
            end_point,
            linewidth=linewidth,
            mode="constant",
            reduce_func=np.mean,
        )
    else:
        # For 3D images, extract profile from each depth slice
        profiles = []
        for depth_slice in image:
            slice_profile = skimage.measure.profile_line(
                depth_slice,
                start_point,
                end_point,
                linewidth=linewidth,
                mode="constant",
                reduce_func=np.mean,
            )
            profiles.append(slice_profile)
        profile = np.array(profiles)  # Shape: (depth, line_length)

    # Trim profile if line_len is specified
    if line_len > 0 and profile.shape[-1] > line_len:
        excess = profile.shape[-1] - line_len
        start_idx = excess // 2
        end_idx = start_idx + line_len
        profile = profile[..., start_idx:end_idx]

    # Create position array
    positions = np.arange(profile.shape[-1])
    if sampling is not None:
        if isinstance(sampling, float | int):
            s = sampling
        elif len(sampling) == 1:
            s = sampling[0]
        elif len(sampling) == 2:
            if sampling[0] != sampling[1]:
                raise ValueError("Sampling must be uniform across dimensions")
            s = sampling[0]
        else:
            raise ValueError(f"Sampling must be a single value or of len 2, got: {sampling}")
        positions = positions * s
        sampling_units = "A" if sampling_units is None else sampling_units
    else:
        sampling_units = "pixels"

    if show:
        _fig, _axs = _show_linescan(
            scan_image,
            profile,
            positions,
            start_point,
            end_point,
            linewidth,
            sampling_units,
            **kwargs,
        )

    return positions, profile


def _calculate_line_endpoints(
    image_shape: tuple[int, int], center: tuple[int, int], phi: float, line_len: int = -1
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Calculate the start and end points of a line within image boundaries.

    Args:
        image_shape (tuple): Image dimensions (num_rows, num_cols).
        center (tuple): Center point (center_row, center_col).
        phi (float): Angle in degrees.
        line_len (int): Desired line length. If -1, extends to image boundaries.

    Returns:
        Tuple of start and end points: ((y1, x1), (y2, x2)).
    """
    num_rows, num_cols = image_shape
    center_row, center_col = center
    phi_rad = np.deg2rad(phi)

    if line_len > 0:
        # Use specified line length
        half_len = line_len / 2
        delta_row = np.sin(phi_rad) * half_len
        delta_col = np.cos(phi_rad) * half_len

        start_point = (center_row - delta_row, center_col - delta_col)
        end_point = (center_row + delta_row, center_col + delta_col)
    else:
        # Extend to image boundaries
        start_point, end_point = _find_boundary_intersections(image_shape, center, phi_rad)

    start_point = tuple(np.clip(start_point, 0, np.array(image_shape) - 1))
    end_point = tuple(np.clip(end_point, 0, np.array(image_shape) - 1))
    return start_point, end_point


def _find_boundary_intersections(
    image_shape: tuple[int, int], center: tuple[int, int], phi_rad: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Find intersections of line with image boundaries."""
    num_rows, num_cols = image_shape
    center_row, center_col = center

    # Direction vector
    delta_col = np.cos(phi_rad)
    delta_row = np.sin(phi_rad)

    # Find intersections with all four boundaries
    intersections = []

    # Left boundary (col = 0)
    if delta_col != 0:
        t = -center_col / delta_col
        row = center_row + t * delta_row
        if 0 <= row < num_rows:
            intersections.append((row, 0))

    # Right boundary (col = num_cols - 1)
    if delta_col != 0:
        t = (num_cols - 1 - center_col) / delta_col
        row = center_row + t * delta_row
        if 0 <= row < num_rows:
            intersections.append((row, num_cols - 1))

    # Top boundary (row = 0)
    if delta_row != 0:
        t = -center_row / delta_row
        col = center_col + t * delta_col
        if 0 <= col < num_cols:
            intersections.append((0, col))

    # Bottom boundary (row = num_rows - 1)
    if delta_row != 0:
        t = (num_rows - 1 - center_row) / delta_row
        col = center_col + t * delta_col
        if 0 <= col < num_cols:
            intersections.append((num_rows - 1, col))

    if len(intersections) < 2:
        raise ValueError("Could not find valid line intersections with image boundaries")

    # Sort intersections by parameter t to get start and end points
    intersections_with_t = []
    for row, col in intersections:
        if delta_col != 0:
            t = (col - center_col) / delta_col
        else:
            t = (row - center_row) / delta_row
        intersections_with_t.append((t, (row, col)))

    intersections_with_t.sort()
    return intersections_with_t[0][1], intersections_with_t[-1][1]


def _show_linescan(
    image: np.ndarray,
    profile: np.ndarray,
    positions: np.ndarray,
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    linewidth: int,
    sampling_units: str,
    **kwargs,
) -> tuple[Any, tuple[Any, Any]]:
    """Display the linescan results."""
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    if profile.ndim == 1:
        ax0.plot(positions, profile, linewidth=2)
        title = "Image"
    else:
        # For 3D input, show as image
        # im = ax0.imshow(profile, aspect="equal", origin="upper")
        im = ax0.matshow(profile, aspect="auto", origin="upper")
        plt.colorbar(im, ax=ax0)
        title = "Mean of depth slices"

    ax0.set_xlabel(f"Position ({sampling_units})")
    ax0.set_ylabel("Intensity" if profile.ndim == 1 else "Depth (pixels)")
    ax0.set_title("Line Profile")

    show_2d(image, figax=(fig, ax1), **kwargs)
    color = kwargs.get("color", "red")

    if linewidth > 1:
        _draw_line_with_width(ax1, start_point, end_point, linewidth, color)
    else:
        ax1.plot(
            [start_point[1], end_point[1]],
            [start_point[0], end_point[0]],
            color=color,
            linewidth=2,
        )

    ax1.set_xlim(0, image.shape[1] - 1)
    ax1.set_ylim(image.shape[0] - 1, 0)
    ax1.set_title(title)

    plt.tight_layout()
    return fig, (ax0, ax1)


def _draw_line_with_width(
    ax,
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    linewidth: int,
    color: str,
) -> None:
    """Draw a line with specified width as a filled polygon."""
    start_row, start_col = start_point
    end_row, end_col = end_point

    # Calculate perpendicular direction
    line_vec = np.array([end_col - start_col, end_row - start_row])
    line_length = np.linalg.norm(line_vec)

    if line_length > 0:
        perp_vec = np.array([-line_vec[1], line_vec[0]]) / line_length * (linewidth / 2)

        # Create polygon points
        p1 = (start_col - perp_vec[0], start_row - perp_vec[1])
        p2 = (end_col - perp_vec[0], end_row - perp_vec[1])
        p3 = (end_col + perp_vec[0], end_row + perp_vec[1])
        p4 = (start_col + perp_vec[0], start_row + perp_vec[1])

        polygon = Polygon([p1, p2, p3, p4], alpha=0.3, facecolor=color, edgecolor=None)
        ax.add_patch(polygon)

    # Draw center line
    ax.plot([start_col, end_col], [start_row, end_row], color=color, linewidth=1)


@torch.no_grad()
def _linescan_tensor(
    image_t, center, phi, linewidth, line_len, show, sampling, sampling_units, **kwargs
):
    """Sample a depth profile on device; transfer only plotted reductions."""
    import math

    from torch.nn.functional import grid_sample

    shape = image_t.shape[-2:]
    center = (
        tuple(round(value) for value in center)
        if center is not None
        else tuple(size // 2 for size in shape)
    )
    start, end = _calculate_line_endpoints(shape, center, phi, line_len)
    delta_row, delta_col = end[0] - start[0], end[1] - start[1]
    length = int(math.ceil(math.hypot(delta_row, delta_col) + 1))
    angle = math.atan2(delta_row, delta_col)
    fraction_t = torch.linspace(0, 1, length, device=image_t.device, dtype=image_t.dtype)
    offsets_t = torch.linspace(
        -(linewidth - 1) / 2,
        (linewidth - 1) / 2,
        linewidth,
        device=image_t.device,
        dtype=image_t.dtype,
    )
    row_t = start[0] + fraction_t[:, None] * delta_row + offsets_t * math.cos(angle)
    col_t = start[1] + fraction_t[:, None] * delta_col - offsets_t * math.sin(angle)
    grid_t = torch.stack(
        (2 * col_t / max(1, shape[1] - 1) - 1, 2 * row_t / max(1, shape[0] - 1) - 1), dim=-1
    )
    stack_t = image_t[None] if image_t.ndim == 2 else image_t
    sampled_t = grid_sample(
        stack_t[:, None],
        grid_t[None].expand(len(stack_t), -1, -1, -1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0]
    outside_t = (row_t < 0) | (row_t > shape[0] - 1) | (col_t < 0) | (col_t > shape[1] - 1)
    profile_t = sampled_t.masked_fill(outside_t, 0).mean(-1)
    if line_len > 0 and profile_t.shape[-1] > line_len:
        first = (profile_t.shape[-1] - line_len) // 2
        profile_t = profile_t[:, first : first + line_len]
    if image_t.ndim == 2:
        profile_t = profile_t[0]
    positions_t = torch.arange(profile_t.shape[-1], device=image_t.device, dtype=image_t.dtype)
    if sampling is not None:
        steps = np.atleast_1d(sampling)
        if not np.all(steps == steps[0]):
            raise ValueError("Line profiles require equal row and col sampling.")
        positions_t *= float(steps[0])
    else:
        sampling_units = "pixels"
    if show:
        _show_linescan(
            stack_t.mean(0).cpu().numpy(),
            profile_t.cpu().numpy(),
            positions_t.cpu().numpy(),
            start,
            end,
            linewidth,
            sampling_units,
            **kwargs,
        )
    return positions_t, profile_t
