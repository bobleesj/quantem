"""Every MAPED picture.

Each function takes the MAPED object and draws a scientific comparison;
none computes anything MAPED keeps. Diffraction patterns use a square-root
scale (``power_sqrt``) so the weak spots show next to the direct beam.
"""

from collections.abc import Sequence
from math import ceil

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import stack

from quantem.core.visualization import show_2d
from quantem.diffraction.maped.core.align import blend_shifted_images, shift_images


@torch.inference_mode()
def alignment(
    maped,
    positions: Sequence[tuple[int, int]] | None = None,
    crop_size: int = 64,
    *,
    num_cols: int | None = None,
    size: str = "medium",
    reference: int = 0,
    images: Sequence[np.ndarray | torch.Tensor] | None = None,
) -> Figure:
    """Compare every tilt before and after the existing scan alignment.

    Columns are tilts; rows show before, after, and the signed difference
    between each aligned image and the aligned reference. Blue/white/red means
    negative/zero/positive on one shared scale, without intensity normalization
    before subtraction. Differences include tilt-dependent contrast and noise,
    not just alignment error.
    No shifts are fitted, no data are merged, and workflow state is unchanged.

    Parameters
    ----------
    positions : sequence of (int, int), optional
        Explicit centers in the full aligned scan canvas, in (row, col) pixels,
        not in the cropped merged scan. Show a three-row group per position.
        Omit to show the common interior of the entire scan.
    crop_size : int
        Width and height of each requested region in native scan pixels. No
        binning is applied. Regions outside the common interior are rejected.
    num_cols : int, optional
        Number of tilts across. Omit to show all tilts across; smaller values
        wrap into further Before/After/Difference groups in the same figure.
    size : {"small", "medium", "large"}
        Panel size, shared with ``show_tilts``. Changes presentation only.
    reference : int
        Acquisition index for the difference reference. Defaults to the first
        tilt; specify the zero-tilt index when that is the desired reference.
    images : sequence of 2D arrays or tensors, optional
        One image per tilt in acquisition order. Defaults to ``bright_field``.
        If alignment used external images, pass those same images here.

    Returns
    -------
    matplotlib.figure.Figure
        A notebook-ready figure. Before and after use the same intensity range
        per tilt and region (2nd to 98th percentiles of the unaligned image).
        Differences retain image intensity units and share symmetric limits
        spanning all displayed differences. White denotes zero.
        Subpixel translations use the existing bilinear alignment operator.

    Raises
    ------
    RuntimeError
        Scan shifts have not been fitted yet.
    ValueError
        Positions do not fit the common interior, or image shapes do not match.

    Examples
    --------
    >>> maped.align_images(show=False)
    >>> maped.show_alignment(reference=3)
    >>> maped.show_alignment(num_cols=4, size="large", reference=3)
    >>> maped.show_alignment(positions=[(96, 96), (256, 256)], crop_size=64, reference=3)
    """
    if maped.real_space_shifts is None:
        raise RuntimeError("Run align_images() before show_alignment().")
    num_tilts = len(maped.tilt_names)
    if not isinstance(reference, (int, np.integer)) or not 0 <= reference < num_tilts:
        raise ValueError(
            f"Choose reference between 0 and {num_tilts - 1}; got {reference}."
        )
    selected = maped.im_bf if images is None else images
    if len(selected) != num_tilts:
        raise ValueError("Supply one image per tilt, in the same order as the datasets.")
    selected_t = [torch.as_tensor(image, device=maped.device) for image in selected]
    if any(image.shape != maped.shape[:2] or image.is_complex() for image in selected_t):
        raise ValueError(f"Supply real 2D images with scan shape {maped.shape[:2]}.")
    images_t = stack(selected_t).to(torch.float32)
    if not torch.isfinite(images_t).all():
        raise ValueError("Images contain NaN or infinity; supply finite image values.")
    shifts_t = maped.real_space_shifts
    # Exclude translation padding from every displayed image.
    margin = ceil(float(shifts_t.abs().max())) + 1
    rows, cols = maped.shape[:2]
    if min(rows, cols) <= 2 * margin:
        raise ValueError(
            "No common interior remains; use larger scan images or check fitted shifts."
        )
    regions = [(margin, rows - margin, margin, cols - margin)]
    if positions is not None:
        if not isinstance(crop_size, (int, np.integer)) or crop_size < 1:
            raise ValueError(f"Choose a positive integer crop_size; got {crop_size}.")
        positions = list(positions)
        if not positions or any(
            len(position) != 2
            or any(not isinstance(value, (int, np.integer)) for value in position)
            for position in positions
        ):
            raise ValueError("Supply explicit integer (row, col) scan positions.")
        regions = [
            (
                row - crop_size // 2,
                row - crop_size // 2 + crop_size,
                col - crop_size // 2,
                col - crop_size // 2 + crop_size,
            )
            for row, col in positions
        ]
        if any(
            row0 < margin or row1 > rows - margin or col0 < margin or col1 > cols - margin
            for row0, row1, col0, col1 in regions
        ):
            raise ValueError(
                f"Requested {crop_size}-pixel crops must fit rows {margin}:{rows - margin} "
                f"and columns {margin}:{cols - margin}; move positions inward or reduce crop_size."
            )
    aligned_t = shift_images(images_t, shifts_t)
    before, after = images_t.cpu().numpy(), aligned_t.cpu().numpy()
    differences = after - after[reference]
    difference_limit = (
        max(
            float(np.abs(differences[:, row0:row1, col0:col1]).max())
            for row0, row1, col0, col1 in regions
        )
        or 1.0
    )
    figure, axes = _tilt_grid(num_tilts, 3, num_cols, size, len(regions))
    num_cols = axes.shape[1]
    num_groups = ceil(num_tilts / num_cols)
    for group, (row0, row1, col0, col1) in enumerate(regions):
        region = (slice(row0, row1), slice(col0, col1))
        for index, name in enumerate(maped.tilt_names):
            raw = before[index][region]
            low, high = np.quantile(raw, [0.02, 0.98])
            if high <= low:
                high = low + 1
            first = 3 * (group * num_groups + index // num_cols)
            column = index % num_cols
            panels = (
                (raw, "gray", low, high),
                (after[index][region], "gray", low, high),
                (differences[index][region], "bwr", -difference_limit, difference_limit),
            )
            for offset, (values, cmap, lower, upper) in enumerate(panels):
                difference_image = axes[first + offset, column].imshow(
                    values, cmap=cmap, vmin=lower, vmax=upper, interpolation="none"
                )
            axes[first, column].set_title(name, fontsize=9)
        label = f"Rows {row0}:{row1}\nCols {col0}:{col1}"
        for block in range(num_groups):
            for offset, stage in enumerate(("Before", "After", "Difference")):
                axes[3 * (group * num_groups + block) + offset, 0].set_ylabel(
                    f"{stage}\n{label}" if offset == 0 else stage
                )
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"Scan alignment | difference = aligned tilt minus {maped.tilt_names[reference]}",
        fontsize=11,
    )
    figure.colorbar(
        difference_image,
        ax=axes[-1, :].tolist(),
        orientation="horizontal",
        fraction=0.06,
        pad=0.02,
        shrink=0.35,
        aspect=55,
        label="Image intensity difference",
    )
    plt.close(figure)
    return figure


def _tilt_grid(
    num_tilts: int,
    rows_per_group: int,
    num_cols: int | None,
    size: str,
    num_regions: int = 1,
) -> tuple[Figure, np.ndarray]:
    """Lay out tilt columns consistently across overview and alignment figures."""
    panel_sizes = {"small": 1.4, "medium": 2.0, "large": 2.7}
    if size not in panel_sizes:
        raise ValueError(f"Choose size='small', 'medium', or 'large'; got {size!r}.")
    if num_tilts < 1:
        raise ValueError("Load at least one tilt before plotting.")
    if num_cols is None:
        num_cols = num_tilts
    if not isinstance(num_cols, (int, np.integer)) or num_cols < 1:
        raise ValueError(f"Choose a positive integer num_cols; got {num_cols}.")
    num_cols = min(num_cols, num_tilts)
    num_groups = ceil(num_tilts / num_cols)
    num_rows = rows_per_group * num_groups * num_regions
    panel_size = panel_sizes[size]
    figure, axes = plt.subplots(
        num_rows,
        num_cols,
        squeeze=False,
        layout="compressed",
        figsize=(panel_size * num_cols + 0.7, panel_size * num_rows + 0.7),
    )
    # Two adjacent half-gutters give a five-pixel gap at the figure's native DPI.
    figure.get_layout_engine().set(
        w_pad=2.5 / figure.dpi, h_pad=0.02, wspace=0, hspace=0.02
    )
    for region in range(num_regions):
        for index in range(num_tilts, num_groups * num_cols):
            first = rows_per_group * (region * num_groups + index // num_cols)
            for axis in axes[first : first + rows_per_group, index % num_cols]:
                axis.set_visible(False)
    return figure, axes


def tilts(
    maped,
    plot_scale: float | Sequence[float] | None = None,
    show_detector_center: bool = True,
    bright_field_cmap: str = "gray",
    bright_field_contrast="linear_auto",
    pattern_cmap: str = "inferno",
    pattern_contrast="power_sqrt",
    *,
    num_cols: int | None = None,
    size: str = "medium",
    **plot_kwargs,
) -> Figure:
    """
    Show each tilt: its bright-field image and its mean diffraction pattern.

    Parameters
    ----------
    plot_scale : float or sequence of float, optional
        Divides the bright-field images in this picture only, one value for all
        tilts or one per tilt, so tilts of different dose compare by eye.
    show_detector_center : bool
        Mark the detector center with a red cross, so each beam's offset from
        it shows at a glance.
    bright_field_cmap, pattern_cmap : str
        Matplotlib colormaps of the two columns.
    bright_field_contrast, pattern_contrast : str or dict
        Contrast as ``show_2d`` takes it: ``"linear_auto"``, ``"log_auto"``,
        ``"power_sqrt"`` (2nd to 98th percentile), or a dict such as
        ``{"lower_quantile": 0.01, "upper_quantile": 0.999, "power": 0.5}``.
    num_cols : int, optional
        Number of tilts across, as in ``show_alignment``. Each group has a
        scan-image row and a diffraction-pattern row. Omit for all tilts across.
    size : {"small", "medium", "large"}
        Panel size, shared with ``show_alignment``. Changes presentation only.
    **plot_kwargs
        Passed to ``show_2d``.

    Returns
    -------
    matplotlib.figure.Figure
        One notebook-ready figure, rather than the workflow object.

    Examples
    --------
    >>> maped.show_tilts(num_cols=7, size="medium")
    >>> maped.show_tilts(num_cols=4, size="large")
    """
    names = maped.tilt_names
    num_tilts = len(names)
    scales = plot_scale if isinstance(plot_scale, Sequence) else [plot_scale or 1] * num_tilts
    bright_fields = [image / scale for image, scale in zip(maped.im_bf, scales)]
    figure, axes = _tilt_grid(num_tilts, 2, num_cols, size)
    num_cols = axes.shape[1]
    for index, (name, image, pattern) in enumerate(zip(names, bright_fields, maped.dp_mean)):
        first, column = 2 * (index // num_cols), index % num_cols
        show_2d(
            [[image], [pattern]],
            figax=(figure, axes[first : first + 2, column : column + 1]),
            title=[[name], [None]],
            cmap=[[bright_field_cmap], [pattern_cmap]],
            norm=[[bright_field_contrast], [pattern_contrast]],
            **plot_kwargs,
        )
        axes[first, column].set_title(name, fontsize=9)
        if show_detector_center:
            detector_rows, detector_cols = pattern.shape
            _cross(axes[first + 1, column], detector_rows / 2, detector_cols / 2, size=14)
    for first in range(0, axes.shape[0], 2):
        axes[first, 0].set_ylabel("Scan image")
        axes[first + 1, 0].set_ylabel("Mean diffraction")
    plt.close(figure)
    return figure


def beam_centers(maped, **plot_kwargs) -> None:
    """Each mean diffraction pattern with its beam center as a red cross, four per row.

    Four per row keeps each pattern large enough to read; the title only names
    the tilt, since the cross is the result.
    """
    per_row = 4
    patterns, names = maped.dp_mean, maped.tilt_names
    _, axes = show_2d(
        [patterns[start : start + per_row] for start in range(0, len(patterns), per_row)],
        title=[names[start : start + per_row] for start in range(0, len(names), per_row)],
        cmap="inferno",
        norm="power_sqrt",
        returnfig=True,
        **plot_kwargs,
    )
    axes = np.ravel(np.asarray(axes, dtype=object))
    for axis, (row, col) in zip(axes, maped.beam_centers.tolist()):
        _cross(axis, row, col, size=16)


def aligned_patterns(maped, **plot_kwargs) -> None:
    """Mean of the patterns, each moved by its detector shift: sharp spots mean a good alignment."""
    aligned = shift_images(stack(maped.dp_mean), maped.diffraction_shifts).mean(0)
    title = f"Mean diffraction pattern, {len(maped.dp_mean)} tilts aligned on the detector"
    show_2d(aligned, title=title, cmap="inferno", norm="power_sqrt", **plot_kwargs)


def aligned_images(maped, images, edge_width: float, **plot_kwargs) -> None:
    """Blend the alignment images after applying their measured scan shifts."""
    blended = blend_shifted_images(stack(images), maped.real_space_shifts, edge_width)
    title = f"Images, {len(images)} tilts aligned in the scan"
    show_2d(blended, title=title, **plot_kwargs)


def merged(maped, **plot_kwargs) -> None:
    """The merged bright field and merged mean diffraction pattern, side by side."""
    show_2d(
        [[maped.im_bf_merged, maped.dp_mean_merged]],
        title=[["Merged bright field", "Merged mean diffraction pattern"]],
        cmap=[["gray", "inferno"]],
        norm=[[None, "power_sqrt"]],
        **plot_kwargs,
    )


def _cross(axis, row: float, col: float, size: int) -> None:
    """Why: a red cross reads at a glance on the inferno colormap, where a dot would vanish in the beam."""
    axis.plot([col], [row], marker="+", color="red", markersize=size, markeredgewidth=2)


def positions(
    maped,
    positions: Sequence[tuple[int, int]],
    *,
    tilt_indices: Sequence[int] | None = None,
):
    """Compare aligned patterns at matching specimen positions."""
    from quantem.diffraction.maped.core.merge import MergedRegions
    from quantem.diffraction.maped.io import read_region

    if getattr(maped, "merged", None) is None or maped.im_bf_merged is None:
        raise RuntimeError("Run merge() before inspecting positions.")
    positions = list(positions)
    rows, columns = maped.merged.shape[:2]
    if not positions or any(
        not (0 <= row < rows and 0 <= col < columns)
        or not isinstance(row, (int, np.integer))
        or not isinstance(col, (int, np.integer))
        for row, col in positions
    ):
        raise ValueError(f"Choose integer (row, col) positions inside {(rows, columns)}.")
    indices = list(range(len(maped.tilts))) if tilt_indices is None else list(tilt_indices)
    if any(
        not isinstance(index, (int, np.integer)) or not 0 <= index < len(maped.tilts)
        for index in indices
    ):
        raise ValueError(f"Choose tilt_indices between 0 and {len(maped.tilts) - 1}.")
    record = maped.merged.metadata["maped_merge"]
    region = record.get("scan_region", record["scan_box"])
    offset_row, _, offset_col, _ = region
    grid, markers, names = [], [], []
    for index in indices:
        source = MergedRegions(
            [maped.tilts[index]],
            maped.real_space_shifts[index : index + 1],
            maped.diffraction_shifts[index : index + 1],
            compile_merge=False,
        )
        panels = [maped.im_bf[index].detach().cpu().numpy()]
        for row, col in positions:
            scan_row, scan_col = row + offset_row, col + offset_col
            block = next(source.blocks((scan_row, scan_row + 1, scan_col, scan_col + 1)))
            panels.append(block[0].detach().cpu().numpy().copy())
        shift_row, shift_col = maped.real_space_shifts[index].detach().cpu().tolist()
        markers.append(
            [
                (row + offset_row - shift_row, col + offset_col - shift_col)
                for row, col in positions
            ]
        )
        grid.append(panels)
        names.append(maped.tilt_names[index])
    panels = [maped.im_bf_merged.detach().cpu().numpy()]
    for row, col in positions:
        block = read_region(maped.merged, (row, row + 1, col, col + 1))
        panels.append(block[0, 0].detach().cpu().numpy().copy())
    grid.append(panels)
    markers.append(positions)
    names.append("Merged")
    labels = [str(index + 1) for index in range(len(positions))]
    pattern_max = float(np.quantile(np.stack([p for row in grid for p in row[1:]]), 0.999))
    scan_limits = np.quantile(np.concatenate([row[0].ravel() for row in grid]), [0.02, 0.98])
    norms = [
        {"vmin": float(scan_limits[0]), "vmax": float(scan_limits[1])},
        *[{"vmin": 0, "vmax": pattern_max, "power": 0.5}] * len(positions),
    ]
    titles = [
        [f"{name}: scan image"]
        + [f"{name}: {label} ({row}, {col})" for label, (row, col) in zip(labels, positions)]
        for name in names
    ]
    figure, axes = show_2d(
        grid,
        title=titles,
        cmap=[["gray"] + ["inferno"] * len(positions)] * len(grid),
        norm=[norms] * len(grid),
        axsize=(3, 3),
    )
    axes = np.asarray(axes, dtype=object).reshape(len(grid), -1)
    for axis, points in zip(axes[:, 0], markers, strict=True):
        for label, (row, col) in zip(labels, points, strict=True):
            axis.plot(col, row, "+", color="cyan", markersize=8)
            axis.annotate(
                label,
                (col, row),
                xytext=(5, 5),
                textcoords="offset points",
                color="cyan",
                bbox={"facecolor": "black", "alpha": 0.6, "pad": 1},
            )
    plt.close(figure)
    return figure
