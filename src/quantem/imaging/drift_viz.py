"""Standalone plot functions for DriftCorrection.

All public functions take a ``DriftCorrection`` instance as their first
argument (``dc``).  ``DriftCorrection`` delegates its ``plot_*`` methods
to these functions via one-line wrappers in ``drift.py``, keeping
alignment logic and visualization code in separate files.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

if TYPE_CHECKING:
    from quantem.imaging.drift import DriftCorrection

from quantem.core.visualization import show_2d


# --- private helpers (used only by the plot functions below) ----------------

def _znorm(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    return (a - a.mean()) / (a.std() + 1e-8)


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((_znorm(a) - _znorm(b)) ** 2).mean()))


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    an, bn = _znorm(a), _znorm(b)
    return float(np.corrcoef(an.ravel(), bn.ravel())[0, 1])


def _log_fft(img: np.ndarray, mask_radius: int = 5) -> np.ndarray:
    """Log-magnitude FFT with a zero-frequency mask for display."""
    n_h, n_w = img.shape
    hann = np.outer(np.hanning(n_h), np.hanning(n_w))
    f = np.fft.fftshift(np.fft.fft2(img * hann))
    mag = np.log1p(np.abs(f))
    center_row, center_col = n_h // 2, n_w // 2
    row_offset, col_offset = np.ogrid[-center_row:n_h - center_row, -center_col:n_w - center_col]
    mag[row_offset ** 2 + col_offset ** 2 < mask_radius ** 2] = 0
    return mag


def _center_crop_slice(
    ref_np: np.ndarray, crop: int | None,
) -> tuple[tuple[slice, slice], int]:
    """Return a centre-crop slice and the crop size."""
    h, w = ref_np.shape
    if crop is None:
        crop = int(min(h, w) * 0.8) // 2 * 2
    r0, c0 = (h - crop) // 2, (w - crop) // 2
    return (slice(r0, r0 + crop), slice(c0, c0 + crop)), crop


# --- public plot functions ---------------------------------------------------

def plot_correction_summary(
    dc: DriftCorrection,
    corrected: torch.Tensor | np.ndarray | None = None,
    reference_index: int = 0,
    target_index: int = -1,
    crop: int | None = None,
    mode: str = "bicubic",
    show_fft: bool = True,
    show_diff: bool = True,
    fft_mask_radius: int = 5,
    axsize: tuple[float, float] = (3.5, 3.5),
    **kwargs,
) -> tuple[Figure, np.ndarray]:
    """One-liner before/after comparison of drift correction.

    Shows the reference, raw (drifted) input, and corrected image
    side by side with RMS and NCC metrics.  Optionally appends rows
    for FFT magnitudes and difference maps.

    The image and FFT rows are rendered via :func:`show_2d`, so all
    of its keyword arguments (``cmap``, ``norm``, ``scalebar``,
    ``cbar``, ``show_ticks``, ...) are forwarded.

    Parameters
    ----------
    dc : DriftCorrection
        Drift correction instance after alignment.
    corrected : torch.Tensor or np.ndarray, optional
        Pre-computed corrected image.  If *None*, calls
        ``dc.apply_correction(mode=mode)`` automatically.
    reference_index : int, default 0
        Index of the reference image (typically the fixed HAADF).
    target_index : int, default -1
        Index of the target image (the one being corrected).
    crop : int, optional
        Centre-crop size in pixels.  *None* uses 80% of shorter dimension.
    mode : str, default "bicubic"
        Interpolation mode if *corrected* is not provided.
    show_fft : bool, default True
        Append a row with log-FFT magnitudes.
    show_diff : bool, default True
        Append a row with difference maps (``seismic`` colourmap).
    fft_mask_radius : int, default 5
        Pixel radius of the zero-frequency mask in FFT panels.
    axsize : tuple, default (3.5, 3.5)
        Size of each subplot panel.
    **kwargs
        Extra keyword arguments forwarded to ``show_2d``.

    Returns
    -------
    fig : Figure
    axes : np.ndarray of Axes, shape (nrows, 3)
    """
    ref_np = dc.images[reference_index].array
    idx = target_index % len(dc.images)
    raw_np = dc.images[idx].array
    if corrected is None:
        corrected = dc.apply_correction(image_index=target_index, mode=mode)
    if isinstance(corrected, torch.Tensor):
        corrected_np = corrected.cpu().numpy()
    else:
        corrected_np = np.asarray(corrected)
    s, crop = _center_crop_slice(ref_np, crop)
    h, w = ref_np.shape
    ref_c = ref_np[s].astype(np.float32)
    raw_c = raw_np[s].astype(np.float32)
    cor_c = corrected_np[s].astype(np.float32)
    raw_rms = _rms(raw_c, ref_c)
    cor_rms = _rms(cor_c, ref_c)
    raw_ncc = _ncc(raw_c, ref_c)
    cor_ncc = _ncc(cor_c, ref_c)
    nrows = 1 + int(show_fft) + int(show_diff)
    ncols = 3
    img_titles = [
        "Reference",
        f"Raw (RMS={raw_rms:.3f}, NCC={raw_ncc:.3f})",
        f"Corrected (RMS={cor_rms:.3f}, NCC={cor_ncc:.3f})",
    ]
    if not show_fft and not show_diff:
        fig, axs = show_2d([ref_c, raw_c, cor_c], title=img_titles, axsize=axsize, **kwargs)
        if not isinstance(axs, np.ndarray):
            axs = np.array([[axs]])
        elif axs.ndim == 1:
            axs = axs.reshape(1, -1)
    else:
        fw, fh = axsize
        fig, axes_grid = plt.subplots(
            nrows, ncols, figsize=(fw * ncols, fh * nrows), squeeze=False,
        )
        show_2d([ref_c, raw_c, cor_c], title=img_titles,
                figax=(fig, axes_grid[0]), axsize=axsize, **kwargs)
        row = 1
        if show_fft:
            fft_kwargs = {k: v for k, v in kwargs.items() if k not in ("cmap",)}
            show_2d(
                [_log_fft(ref_c, fft_mask_radius),
                 _log_fft(raw_c, fft_mask_radius),
                 _log_fft(cor_c, fft_mask_radius)],
                title=["FFT: Reference", "FFT: Raw", "FFT: Corrected"],
                figax=(fig, axes_grid[row]),
                axsize=axsize,
                **fft_kwargs,
            )
            row += 1
        if show_diff:
            ref_n = _znorm(ref_c)
            diff_raw = _znorm(raw_c) - ref_n
            diff_cor = _znorm(cor_c) - ref_n
            vmax = float(max(np.abs(diff_raw).max(), np.abs(diff_cor).max()) * 0.8)
            axes_grid[row][0].axis("off")
            axes_grid[row][0].text(
                0.5, 0.5,
                f"Crop: {crop}\u00d7{crop}\nfrom {h}\u00d7{w}",
                transform=axes_grid[row][0].transAxes,
                ha="center", va="center", fontsize=10, color="gray",
            )
            show_2d(
                [diff_raw, diff_cor],
                title=[
                    f"Raw \u2212 Ref (RMS={raw_rms:.3f})",
                    f"Corrected \u2212 Ref (RMS={cor_rms:.3f})",
                ],
                cmap="seismic",
                figax=(fig, axes_grid[row, 1:]),
                axsize=axsize,
                vmin=-vmax, vmax=vmax,
            )
        fig.tight_layout()
        axs = axes_grid
    print(f"{'':>12s}   RMS     NCC")
    print("-" * 35)
    print(f"{'Raw':>12s}  {raw_rms:.4f}  {raw_ncc:.4f}")
    print(f"{'Corrected':>12s}  {cor_rms:.4f}  {cor_ncc:.4f}")
    reduction = (1 - cor_rms / raw_rms) * 100 if raw_rms > 0 else 0
    print(f"  RMS reduction: {reduction:.1f}%")
    return fig, axs


def plot_correction_comparison(
    dc: DriftCorrection,
    crop: int | None = None,
    target_index: int = -1,
    axsize: tuple[float, float] = (3.5, 3.5),
    show_fft: bool = True,
    **kwargs,
) -> tuple[Figure, np.ndarray, dict]:
    """Compare all correction modes in a single figure.

    Automatically applies affine-only and nonrigid corrections with
    both ``bilinear`` and ``bicubic`` interpolation, then shows them
    alongside the reference in a grid with RMS / NCC metrics.

    Requires :meth:`~DriftCorrection.align_affine` (and optionally
    :meth:`~DriftCorrection.align_nonrigid`) to have been called.

    Parameters
    ----------
    dc : DriftCorrection
    crop : int, optional
        Centre-crop size.  *None* uses 80% of shorter dimension.
    target_index : int, default -1
        Which image to correct.
    axsize : tuple, default (3.5, 3.5)
        Size of each panel.
    show_fft : bool, default True
        Show FFT magnitude row below the images.
    **kwargs
        Forwarded to :func:`show_2d`.

    Returns
    -------
    fig : Figure
    axes : np.ndarray
    metrics : dict
        ``{method_name: (rms, ncc)}``
    """
    idx = target_index % len(dc.knots)
    ref_np = dc.images[0].array
    raw_np = dc.images[idx].array
    s, crop = _center_crop_slice(ref_np, crop)
    has_nonrigid = hasattr(dc, "_knots_after_affine") and not torch.equal(
        dc.knots[idx], dc._knots_after_affine[idx]
    )
    saved_knots = dc.knots[idx].clone()
    results = {}
    with torch.no_grad():
        if has_nonrigid:
            dc.knots[idx] = dc._knots_after_affine[idx]
            results["affine bilinear"] = dc.apply_correction(
                image_index=target_index, mode="bilinear").cpu().numpy()
            results["affine bicubic"] = dc.apply_correction(
                image_index=target_index, mode="bicubic").cpu().numpy()
            dc.knots[idx] = saved_knots
        results["nonrigid bilinear"] = dc.apply_correction(
            image_index=target_index, mode="bilinear").cpu().numpy()
        results["nonrigid bicubic"] = dc.apply_correction(
            image_index=target_index, mode="bicubic").cpu().numpy()
    ref_c = ref_np[s].astype(np.float32)
    metrics = {}
    labels = ["HAADF ref", "raw (drifted)"]
    panels = [ref_c, raw_np[s].astype(np.float32)]
    for name, img in results.items():
        labels.append(name)
        panels.append(img[s].astype(np.float32))
        metrics[name] = (_rms(img[s], ref_c), _ncc(img[s], ref_c))
    raw_rms, raw_ncc = _rms(raw_np[s], ref_c), _ncc(raw_np[s], ref_c)
    metrics["raw (drifted)"] = (raw_rms, raw_ncc)
    titles = ["HAADF ref", f"raw (RMS={raw_rms:.3f})"]
    for name in results:
        r, n = metrics[name]
        titles.append(f"{name}\nRMS={r:.3f} NCC={n:.3f}")
    nrows = 1 + int(show_fft)
    ncols = len(panels)
    fw, fh = axsize
    fig, axes_grid = plt.subplots(
        nrows, ncols, figsize=(fw * ncols, fh * nrows), squeeze=False,
    )
    show_2d(panels, title=titles, figax=(fig, axes_grid[0]), axsize=axsize, **kwargs)
    if show_fft:
        fft_panels = [_log_fft(p) for p in panels]
        fft_titles = [f"FFT: {label}" for label in labels]
        fft_kw = {k: v for k, v in kwargs.items() if k != "cmap"}
        show_2d(fft_panels, title=fft_titles,
                figax=(fig, axes_grid[1]), axsize=axsize, **fft_kw)
    fig.tight_layout()
    h, w = ref_np.shape
    print(f"--- RMS / NCC vs reference (center {crop}\u00d7{crop} "
          f"crop from {h}\u00d7{w}) ---")
    print(f"{'Method':>20s}   RMS    NCC")
    print("-" * 45)
    for name in ["raw (drifted)", *results]:
        r, n = metrics[name]
        print(f"{name:>20s}  {r:.4f}  {n:.4f}")
    return fig, axes_grid, metrics


def plot_radial_power(
    dc: DriftCorrection,
    methods: dict[str, np.ndarray] | None = None,
    crop: int | None = None,
    target_index: int = -1,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[Figure, Axes]:
    """Radial FFT power spectrum comparing correction methods.

    Higher power at high spatial frequencies indicates sharper
    features.  Useful for verifying that drift correction preserves
    (or recovers) lattice fringe resolution.

    Parameters
    ----------
    dc : DriftCorrection
    methods : dict, optional
        ``{label: image_ndarray}``.  If *None*, auto-generates from
        the current pipeline state (raw, affine, nonrigid).
    crop : int, optional
        Centre-crop size.  *None* uses 80% of shorter dimension.
    target_index : int, default -1
        Which image to correct when auto-generating methods.
    figsize : tuple, default (10, 6)

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    idx = target_index % len(dc.knots)
    ref_np = dc.images[0].array
    raw_np = dc.images[idx].array
    s, crop = _center_crop_slice(ref_np, crop)
    if methods is None:
        methods = {"HAADF ref": ref_np[s], "raw (drifted)": raw_np[s]}
        has_nonrigid = (
            hasattr(dc, "_knots_after_affine")
            and not torch.equal(dc.knots[idx], dc._knots_after_affine[idx])
        )
        saved = dc.knots[idx].clone()
        with torch.no_grad():
            if has_nonrigid:
                dc.knots[idx] = dc._knots_after_affine[idx]
                methods["affine bicubic"] = dc.apply_correction(
                    image_index=target_index, mode="bicubic"
                ).cpu().numpy()[s]
                dc.knots[idx] = saved
            methods["nonrigid bicubic"] = dc.apply_correction(
                image_index=target_index, mode="bicubic"
            ).cpu().numpy()[s]
    else:
        methods = {k: v[s] if v.shape != (crop, crop) else v
                   for k, v in methods.items()}

    def _radial(img: np.ndarray):
        n = img.shape[0]
        hann = np.outer(np.hanning(n), np.hanning(n))
        f = np.fft.fftshift(np.fft.fft2(img.astype(np.float32) * hann))
        power = np.abs(f) ** 2
        center_row, center_col = n // 2, n // 2
        row_offset, col_offset = np.ogrid[-center_row:n - center_row, -center_col:n - center_col]
        r = np.sqrt(row_offset ** 2 + col_offset ** 2).astype(int).ravel()
        max_r = min(center_row, center_col)
        valid = r < max_r
        count = np.bincount(r[valid], minlength=max_r)
        total = np.bincount(r[valid], weights=power.ravel()[valid], minlength=max_r)
        radial = np.where(count > 0, total / count, 0.0)
        freqs = np.arange(max_r) / n
        return freqs, radial

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for label, img in methods.items():
        freqs, power = _radial(img)
        ax.semilogy(freqs[1:], power[1:], label=label, alpha=0.8)
    ax.set_xlabel("Spatial frequency (cycles/pixel)")
    ax.set_ylabel("Power (log scale)")
    ax.set_title("Radial FFT power spectrum")
    ax.legend()
    ax.set_xlim(0, 0.5)
    fig.tight_layout()
    return fig, ax


def plot_warped_images(
    dc: DriftCorrection,
    show_knots: bool = True,
    **kwargs,
) -> tuple[Figure, np.ndarray]:
    """Plot each warped image with optional knot overlays.

    Parameters
    ----------
    dc : DriftCorrection
    show_knots : bool, default True
        Overlay current knot positions on each image.
    **kwargs
        Forwarded to :func:`show_2d`.

    Returns
    -------
    fig : Figure
    axes : np.ndarray of Axes
    """
    dc._ensure_warped_images()
    fig, ax = show_2d(list(dc.imgs_warped.array), **kwargs)
    if show_knots:
        for img_idx in range(dc.shape[0]):
            knots_np = dc.knots[img_idx].cpu().numpy()
            ax[img_idx].plot(knots_np[1], knots_np[0], color="r")
    return fig, ax


def plot_convergence(
    dc: DriftCorrection,
    figsize: tuple[float, float] = (8, 3),
    **kwargs,
) -> tuple[Figure, np.ndarray]:
    """Plot the convergence of drift correction over iterations.

    Plots affine and nonrigid error curves side by side. Useful for
    diagnosing whether the optimizer converged or needs more iterations.

    Parameters
    ----------
    dc : DriftCorrection
    figsize : tuple, default (8, 3)
    **kwargs
        Forwarded to ``ax.plot``.

    Returns
    -------
    fig : Figure
    axes : np.ndarray of Axes, shape (2,)
    """
    is_nonrigid = dc.error_track[:, 0] == 2
    error = dc.error_track[:, 1]
    it = np.arange(error.shape[0])
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    color = (1, 0, 0)
    if np.any(~is_nonrigid):
        ax[0].plot(
            it[~is_nonrigid], 100 * error[~is_nonrigid],
            marker="o", color=color, linestyle="-", label="Affine", **kwargs,
        )
        ax[0].set_xlabel("Affine Iterations")
        ax[0].set_ylabel("Mean Error [%]")
        ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    else:
        ax[0].axis("off")
    if np.any(is_nonrigid):
        first_true = np.argmax(is_nonrigid)
        if first_true > 0:
            is_nonrigid[first_true - 1] = True
        ax[1].plot(
            it[is_nonrigid], 100 * error[is_nonrigid],
            marker="o", color=color, linestyle="-", label="nonrigid", **kwargs,
        )
        ax[1].set_xlabel("nonrigid iterations")
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    else:
        ax[1].axis("off")
    fig.tight_layout()
    return fig, ax


def plot_merged_images(
    dc: DriftCorrection,
    show_knots: bool = True,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Plot the mean of all warped images with optional knot overlays.

    Parameters
    ----------
    dc : DriftCorrection
    show_knots : bool, default True
        Overlay current knot positions for each image.
    **kwargs
        Forwarded to :func:`show_2d`.

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    dc._ensure_warped_images()
    fig, ax = show_2d(dc.imgs_warped.array.mean(0), **kwargs)
    if show_knots:
        for img_idx in range(dc.shape[0]):
            knots_np = dc.knots[img_idx].cpu().numpy()
            ax.plot(knots_np[1], knots_np[0])
    return fig, ax


def plot_knots(
    dc: DriftCorrection,
    figsize: tuple[int, int] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot knot trajectories before and after correction plus the per-scanline delta field.

    Two panels per image:
    - Top: mean warped image with initial knots (dashed) and corrected knots (solid)
      overlaid. A third dotted line shows the affine-only state when available, so
      the affine vs. nonrigid contributions are visible side by side.
    - Bottom: per-scanline correction delta (row and col components) in pixels.
      A smooth curve means the correction field is physically reasonable. Rapid
      oscillations indicate regularization_sigma_px is too small and the optimizer
      is fitting noise rather than real drift.

    Parameters
    ----------
    dc : DriftCorrection
    figsize : (width, height), optional
        Figure size in inches. Defaults to (7 * num_images, 6).

    Returns
    -------
    fig : Figure
    axes : np.ndarray of Axes, shape (2, num_images)
    """
    dc._ensure_warped_images()
    num_images = dc.shape[0]
    merged = dc.imgs_warped.array.mean(0)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if figsize is None:
        figsize = (7 * num_images, 6)
    fig, axes = plt.subplots(
        2, num_images, figsize=figsize,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    if num_images == 1:
        axes = axes[:, None]
    scanlines = np.arange(dc.knots[0].shape[1])
    for img_idx in range(num_images):
        color = colors[img_idx % len(colors)]
        initial = dc._initial_knots[img_idx].cpu().numpy()
        current = dc.knots[img_idx].cpu().numpy()
        delta = current - initial
        ax_img = axes[0, img_idx]
        ax_img.imshow(merged, cmap="gray", origin="upper", aspect="equal",
                      vmin=float(merged.min()), vmax=float(merged.max()))
        ax_img.plot(initial[1, :, 0], initial[0, :, 0],
                    "--", color=color, lw=1.2, alpha=0.7, label="initial")
        ax_img.plot(current[1, :, 0], current[0, :, 0],
                    "-", color=color, lw=1.5, label="corrected")
        if hasattr(dc, "_knots_after_affine"):
            affine_np = dc._knots_after_affine[img_idx].cpu().numpy()
            ax_img.plot(affine_np[1, :, 0], affine_np[0, :, 0],
                        ":", color=color, lw=1.0, alpha=0.6, label="after affine")
        ax_img.legend(fontsize=8, loc="upper right")
        ax_img.set_title(f"image {img_idx} — knot trajectory", fontsize=10)
        ax_img.axis("off")
        ax_delta = axes[1, img_idx]
        ax_delta.plot(scanlines, delta[0, :, 0], lw=1.2, label="row \u0394")
        ax_delta.plot(scanlines, delta[1, :, 0], lw=1.2, label="col \u0394")
        if hasattr(dc, "_knots_after_affine"):
            aff_delta = affine_np - initial
            ax_delta.plot(scanlines, aff_delta[0, :, 0],
                          ":", lw=1.0, alpha=0.6, label="affine row \u0394")
            ax_delta.plot(scanlines, aff_delta[1, :, 0],
                          ":", lw=1.0, alpha=0.6, label="affine col \u0394")
        ax_delta.axhline(0, color="k", lw=0.5, ls="--")
        ax_delta.set_xlabel("scanline")
        ax_delta.set_ylabel("correction (px)")
        ax_delta.set_title(f"image {img_idx} — delta field", fontsize=10)
        ax_delta.legend(fontsize=8)
        ax_delta.grid(alpha=0.3)
    fig.tight_layout()
    return fig, axes


# --- 4D-STEM correction visualization ----------------------------------------


def plot_4dstem_correction(
    dc: DriftCorrection,
    cube_raw: torch.Tensor,
    cube_corrected: torch.Tensor,
    vdf_raw: np.ndarray | None = None,
    vdf_corrected: np.ndarray | None = None,
    ref_image: np.ndarray | None = None,
    vdf_mask: torch.Tensor | None = None,
    sample_positions: list[tuple[int, int]] | None = None,
    n_samples: int = 4,
    crop: int | None = None,
    axsize: tuple[float, float] = (3.0, 3.0),
) -> tuple[Figure, np.ndarray, dict]:
    """Visualize 4D-STEM drift correction results.

    Shows three rows of evidence that the correction worked:

    1. **VDF comparison** — virtual dark field before/after vs HAADF
       reference, with NCC metrics.
    2. **Mean diffraction pattern** — averaged CBED before vs after,
       confirming reciprocal-space content is preserved/sharpened.
    3. **Individual CBED patterns** — before / after / difference at
       sampled scan positions, showing per-pixel resampling.

    All reductions are computed on GPU; only small 2D arrays are
    transferred to CPU for plotting.

    Parameters
    ----------
    dc : DriftCorrection
        Must have ``align_nonrigid`` completed.
    cube_raw : torch.Tensor
        Raw 4D-STEM data ``(H, W, det_h, det_w)`` on GPU.
    cube_corrected : torch.Tensor
        Drift-corrected data ``(H, W, det_h, det_w)`` on GPU.
    vdf_raw, vdf_corrected : np.ndarray, optional
        Pre-computed VDF images.  If ``None``, computed from the cubes
        using *vdf_mask* or a default annular mask.
    ref_image : np.ndarray, optional
        HAADF reference image.  If ``None``, uses ``dc.images[0].array``.
    vdf_mask : torch.Tensor, optional
        Boolean mask ``(det_h, det_w)`` for VDF computation.
        Default: annular dark-field (radius > det_h/4).
    sample_positions : list of (row, col), optional
        Scan positions for CBED comparison.  ``None`` auto-selects.
    n_samples : int, default 4
        Number of CBED positions when auto-selecting.
    crop : int, optional
        Centre-crop for VDF panels.  ``None`` → 80% of shorter dim.
    axsize : tuple, default (3.0, 3.0)
        Size of each subplot.

    Returns
    -------
    fig : Figure
    axes : np.ndarray
    metrics : dict
        ``{'vdf_raw_ncc': float, 'vdf_corrected_ncc': float, ...}``
    """
    scan_h, scan_w = cube_raw.shape[:2]
    det_h, det_w = cube_raw.shape[2], cube_raw.shape[3]
    device = cube_raw.device

    if ref_image is None:
        ref_image = dc.images[0].array

    # --- VDF computation (on GPU, transfer only the 2D result) ---
    if vdf_mask is None:
        qy, qx = torch.meshgrid(
            torch.arange(det_h, device=device) - det_h // 2,
            torch.arange(det_w, device=device) - det_w // 2,
            indexing="ij",
        )
        vdf_mask = (qy ** 2 + qx ** 2) > (det_h // 4) ** 2

    if vdf_raw is None:
        vdf_raw = _compute_vdf(cube_raw, vdf_mask)
    if vdf_corrected is None:
        vdf_corrected = _compute_vdf(cube_corrected, vdf_mask)

    # --- Mean diffraction pattern (GPU reduction → small 2D) ---
    mean_dp_raw = cube_raw.float().mean(dim=(0, 1)).cpu().numpy()
    mean_dp_corr = cube_corrected.float().mean(dim=(0, 1)).cpu().numpy()

    # --- Sample positions ---
    if sample_positions is None:
        sample_positions = _auto_sample_positions(
            dc, scan_h, scan_w, n_samples,
        )

    # --- Extract CBEDs (small: just n_samples × det_h × det_w) ---
    cbeds_raw, cbeds_corr = [], []
    for r, c in sample_positions:
        cbeds_raw.append(cube_raw[r, c].float().cpu().numpy())
        cbeds_corr.append(cube_corrected[r, c].float().cpu().numpy())

    # --- Crop for VDF panels ---
    s, crop_val = _center_crop_slice(ref_image, crop)

    # --- Metrics ---
    ref_c = ref_image[s].astype(np.float32)
    vdf_raw_c = vdf_raw[s]
    vdf_corr_c = vdf_corrected[s]
    metrics = {
        "vdf_raw_ncc": _ncc(vdf_raw_c, ref_c),
        "vdf_corrected_ncc": _ncc(vdf_corr_c, ref_c),
        "vdf_raw_rms": _rms(vdf_raw_c, ref_c),
        "vdf_corrected_rms": _rms(vdf_corr_c, ref_c),
    }

    # === Build figure ===
    n_pos = len(sample_positions)
    # Row 1: VDF raw, VDF corrected, HAADF ref (3 panels)
    # Row 2: mean DP raw, mean DP corrected, mean DP difference (3 panels)
    # Row 3+: per-position CBED raw, corrected, difference
    ncols = 3
    nrows = 2 + n_pos
    fw, fh = axsize
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fw * ncols, fh * nrows), squeeze=False,
    )

    # --- Row 1: VDF comparison ---
    raw_ncc = metrics["vdf_raw_ncc"]
    corr_ncc = metrics["vdf_corrected_ncc"]
    vdf_panels = [vdf_raw_c, vdf_corr_c, ref_c]
    vdf_titles = [
        f"VDF raw (NCC={raw_ncc:.3f})",
        f"VDF corrected (NCC={corr_ncc:.3f})",
        "HAADF reference",
    ]
    for j, (panel, title) in enumerate(zip(vdf_panels, vdf_titles)):
        ax = axes[0, j]
        ax.imshow(panel, cmap="gray", origin="upper", aspect="equal")
        # Mark CBED sample positions
        for idx, (r, c) in enumerate(sample_positions):
            r_crop = r - s[0].start
            c_crop = c - s[1].start
            if 0 <= r_crop < crop_val and 0 <= c_crop < crop_val:
                ax.plot(c_crop, r_crop, "o", color=f"C{idx}", ms=6, mew=1.5,
                        mfc="none")
                ax.annotate(f"P{idx}", (c_crop + 3, r_crop - 3),
                            color=f"C{idx}", fontsize=7, fontweight="bold")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # --- Row 2: mean diffraction pattern ---
    dp_diff = mean_dp_corr - mean_dp_raw
    dp_panels = [mean_dp_raw, mean_dp_corr, dp_diff]
    dp_titles = ["Mean DP (raw)", "Mean DP (corrected)", "Mean DP (Δ)"]
    dp_cmaps = ["inferno", "inferno", "seismic"]
    for j, (panel, title, cmap) in enumerate(
        zip(dp_panels, dp_titles, dp_cmaps)
    ):
        ax = axes[1, j]
        vkw = {}
        if cmap == "seismic":
            vlim = max(abs(panel.min()), abs(panel.max()))
            vkw = {"vmin": -vlim, "vmax": vlim}
        ax.imshow(np.log1p(np.abs(panel)) if cmap != "seismic" else panel,
                  cmap=cmap, origin="upper", aspect="equal", **vkw)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # --- Rows 3+: individual CBED patterns ---
    for i, (r, c) in enumerate(sample_positions):
        cbed_r = cbeds_raw[i]
        cbed_c = cbeds_corr[i]
        cbed_diff = cbed_c - cbed_r
        row_idx = 2 + i

        ax_raw = axes[row_idx, 0]
        ax_corr = axes[row_idx, 1]
        ax_diff = axes[row_idx, 2]

        # Shared scale for raw/corrected
        vmin = min(cbed_r.min(), cbed_c.min())
        vmax = max(cbed_r.max(), cbed_c.max())
        ax_raw.imshow(np.log1p(np.maximum(cbed_r, 0)), cmap="inferno",
                      origin="upper", aspect="equal")
        ax_raw.set_title(f"P{i} ({r},{c}) raw", fontsize=8)
        ax_raw.axis("off")

        ax_corr.imshow(np.log1p(np.maximum(cbed_c, 0)), cmap="inferno",
                       origin="upper", aspect="equal")
        ax_corr.set_title(f"P{i} ({r},{c}) corrected", fontsize=8)
        ax_corr.axis("off")

        vlim = max(abs(cbed_diff.min()), abs(cbed_diff.max()))
        if vlim < 1e-8:
            vlim = 1.0
        ax_diff.imshow(cbed_diff, cmap="seismic", origin="upper",
                       aspect="equal", vmin=-vlim, vmax=vlim)
        ax_diff.set_title(f"P{i} Δ (corr − raw)", fontsize=8)
        ax_diff.axis("off")

    fig.tight_layout()

    # Print summary
    print(f"--- 4D-STEM drift correction summary ---")
    print(f"  Scan: {scan_h}×{scan_w}, Detector: {det_h}×{det_w}")
    print(f"  VDF vs ref (centre {crop_val}×{crop_val} crop):")
    print(f"    Raw:       NCC={raw_ncc:.4f}  RMS={metrics['vdf_raw_rms']:.4f}")
    print(f"    Corrected: NCC={corr_ncc:.4f}  RMS={metrics['vdf_corrected_rms']:.4f}")
    return fig, axes, metrics


def _compute_vdf(
    cube: torch.Tensor, mask: torch.Tensor,
) -> np.ndarray:
    """Compute VDF from 4D cube on GPU, return as numpy."""
    scan_h, scan_w = cube.shape[:2]
    vdf = torch.zeros(scan_h, scan_w, device=cube.device, dtype=torch.float32)
    for r in range(scan_h):
        vdf[r] = cube[r].float()[:, mask].sum(dim=-1)
    return vdf.cpu().numpy()


def _auto_sample_positions(
    dc: DriftCorrection,
    scan_h: int,
    scan_w: int,
    n_samples: int,
) -> list[tuple[int, int]]:
    """Pick scan positions that highlight drift correction.

    Selects: centre, max-drift row, and evenly-spaced extras.
    """
    # Centre
    positions = [(scan_h // 2, scan_w // 2)]

    # Row with maximum drift magnitude
    try:
        idx = -1 % len(dc.knots)
        delta = (dc.knots[idx] - dc._initial_knots[idx]).cpu().numpy()
        drift_mag = np.sqrt(delta[0, :, 0] ** 2 + delta[1, :, 0] ** 2)
        max_row = int(np.argmax(drift_mag))
        positions.append((max_row, scan_w // 2))
    except Exception:
        positions.append((scan_h - scan_h // 4, scan_w // 2))

    # Fill remaining with evenly spaced positions
    margin = max(scan_h // 10, 5)
    step = max(1, (scan_h - 2 * margin) // max(1, n_samples - 1))
    for i in range(n_samples - 2):
        r = margin + i * step
        c = margin + (i * scan_w // max(1, n_samples)) % (scan_w - 2 * margin)
        if (r, c) not in positions:
            positions.append((r, c))
        if len(positions) >= n_samples:
            break

    return positions[:n_samples]
