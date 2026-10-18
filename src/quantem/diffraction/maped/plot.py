"""Rendering for the public MAPED inspection methods."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from quantem.core.visualization import show_2d

from .alignment import shift_images_torch

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .maped import MAPED


def positions(
    maped: MAPED,
    positions: Sequence[tuple[int, int]],
    *,
    tilt_indices: Sequence[int] | None = None,
) -> Figure:
    """Compare aligned patterns at matching specimen positions."""
    from matplotlib import pyplot as plt

    from .merge import MergeSource, _read_scan_region

    if getattr(maped, "merged", None) is None or maped.im_bf_merged is None:
        raise RuntimeError("Run merge(compute_summaries=True) before inspecting positions.")
    positions = list(positions)
    rows, columns = maped.merged.shape[:2]
    if not positions or any(
        not (0 <= row < rows and 0 <= col < columns)
        or not isinstance(row, (int, np.integer))
        or not isinstance(col, (int, np.integer))
        for row, col in positions
    ):
        raise ValueError(f"Choose integer (row, col) positions inside {(rows, columns)}.")
    indices = list(range(len(maped.datasets))) if tilt_indices is None else list(tilt_indices)
    if any(
        not isinstance(index, (int, np.integer)) or not 0 <= index < len(maped.datasets)
        for index in indices
    ):
        raise ValueError(f"Choose tilt_indices between 0 and {len(maped.datasets) - 1}.")
    record = maped.merged.metadata["maped_merge"]
    region = record.get("scan_region", record["scan_box"])
    offset_row, _, offset_col, _ = region
    grid, markers, names = [], [], []
    for index in indices:
        source = MergeSource(
            [maped.datasets[index]],
            maped.real_space_shifts[index : index + 1],
            maped.diffraction_shifts[index : index + 1],
            close_sources_before_reopen=False,
            compile_merge=False,
        )
        panels = [maped.im_bf[index].detach().cpu().numpy()]
        try:
            for row, col in positions:
                scan_row, scan_col = row + offset_row, col + offset_col
                block = next(source.blocks((scan_row, scan_row + 1, scan_col, scan_col + 1)))
                panels.append(block[0].detach().cpu().numpy().copy())
        finally:
            source.close()
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
        block = _read_scan_region(maped.merged, (row, row + 1, col, col + 1))
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


def tilts(
    maped: MAPED,
    show_detector_center: bool,
    bright_field_cmap: str,
    bright_field_contrast,
    pattern_cmap: str,
    pattern_contrast,
    **plot_kwargs,
) -> None:
    """Draw the scan image and mean diffraction pattern for every tilt."""
    n = len(maped.datasets)
    tiles = [[(maped.im_bf[i] / maped.scales[i]), maped.dp_mean[i]] for i in range(n)]
    titles = [
        [f"{name}: bright field", f"{name}: mean diffraction pattern"] for name in maped.tilt_names
    ]
    fig, ax = show_2d(
        tiles,
        title=titles,
        cmap=[[bright_field_cmap, pattern_cmap]] * n,
        norm=[[bright_field_contrast, pattern_contrast]] * n,
        returnfig=True,
        **plot_kwargs,
    )
    if show_detector_center:
        k_rows, k_cols = maped.dp_mean[0].shape
        axs = np.asarray(ax, dtype=object).reshape(n, 2)
        for i in range(n):
            axs[i, 1].plot(
                [k_cols / 2],
                [k_rows / 2],
                marker="+",
                color="red",
                markersize=14,
                markeredgewidth=2,
            )


def beam_centers(maped: MAPED, indices: Sequence[int], **plot_kwargs) -> None:
    """Mark the direct-beam center in each selected diffraction pattern."""
    arrays = [np.asarray(maped.dp_mean[i].cpu()) for i in indices]
    titles = [maped.tilt_names[i] for i in indices]
    columns = 4
    fig, ax = show_2d(
        [arrays[i : i + columns] for i in range(0, len(arrays), columns)],
        title=[titles[i : i + columns] for i in range(0, len(titles), columns)],
        cmap="inferno",
        norm="power_sqrt",
        returnfig=True,
        **plot_kwargs,
    )
    axs = np.ravel(np.asarray(ax, dtype=object))
    for j, i in enumerate(indices):
        r, c = maped.beam_centers[i].cpu().numpy()
        axs[j].plot([c], [r], marker="+", color="red", markersize=16, markeredgewidth=2)


def diffraction_alignment(maped: MAPED, border_taper: float, **plot_kwargs) -> None:
    """Draw the mean of the aligned diffraction patterns."""
    n = len(maped.dp_mean)
    im_aligned = shift_images_torch(
        images=torch.stack(maped.dp_mean),
        shifts_rc=maped.diffraction_shifts,
        edge_blend=float(border_taper),
        pad_val="min",
    )
    show_2d(
        im_aligned.mean(0),
        title=f"Mean diffraction pattern, {n} tilts aligned on the detector",
        cmap="inferno",
        norm="power_sqrt",
        **plot_kwargs,
    )


def scan_alignment(maped: MAPED, num_tilts: int, max_shift: float, **plot_kwargs) -> None:
    """Draw the aligned scan images."""
    im_aligned = shift_images_torch(
        images=torch.stack(maped.im_bf[:num_tilts]),
        shifts_rc=maped.real_space_shifts[:num_tilts, :],
        edge_blend=float(max_shift),
        pad_val="median",
        blend=True,
    )
    show_2d(
        im_aligned,
        title=f"Bright field, {num_tilts} tilts aligned in the scan",
        **plot_kwargs,
    )


def merged(maped: MAPED, in_memory: bool, **plot_kwargs) -> None:
    """Draw the merged scan image and mean diffraction pattern."""
    label = "Merged Region" if in_memory else "Merged"
    show_2d(
        [[maped.im_bf_merged, maped.dp_mean_merged]],
        title=[[f"{label} Bright Field", f"{label} Mean Diffraction Pattern"]],
        cmap=[["gray", "inferno"]],
        norm=[[None, "power_sqrt"]],
        **plot_kwargs,
    )


def scan_correction(
    dataset: torch.Tensor,
    shifted_dps: torch.Tensor,
    diffraction_shifts: torch.Tensor,
    diffraction_shifts_old: torch.Tensor | None,
) -> None:
    """Compare measured and fitted shifts and the corrected diffraction patterns."""
    if diffraction_shifts_old is not None:
        show_2d(
            [
                [
                    diffraction_shifts_old[:, :, 0],
                    diffraction_shifts[:, :, 0],
                    diffraction_shifts[:, :, 0] - diffraction_shifts_old[:, :, 0],
                ],
                [
                    diffraction_shifts_old[:, :, 1],
                    diffraction_shifts[:, :, 1],
                    diffraction_shifts[:, :, 1] - diffraction_shifts_old[:, :, 1],
                ],
            ],
            title=[
                ["Shifts x", "Fit x", "Residual x"],
                ["Shifts y", "Fit y", "Residual y"],
            ],
            cmap="RdBu_r",
        )
    dp_mean_before = dataset.mean(dim=(0, 1))
    dp_mean = shifted_dps.mean(dim=(0, 1))
    dp_max = torch.max(
        torch.max(shifted_dps, dim=0, keepdim=False).values, dim=0, keepdim=False
    ).values
    show_2d(
        [dp_mean_before, dp_mean, dp_max],
        vmax=0.75,
    )
