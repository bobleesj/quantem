"""Torch helper functions for drift correction."""

import math

import numpy as np
import torch
from numpy.typing import NDArray
from torch.fft import fft2, fftfreq, ifft2, ifftshift


# ---------------------------------------------------------------------------
# Public API - called by DriftCorrection in drift.py
# ---------------------------------------------------------------------------


def initialize_scanline_knots(
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    scan_fast: NDArray,
    scan_slow: NDArray,
    number_knots: int,
) -> NDArray:
    """Build the initial knot grid used by ``DriftCorrection.preprocess``.

    The knot anchors define where each scanline starts on the padded canvas
    before any affine or non-rigid optimization. For ``number_knots == 1``,
    this is a vertical line of anchors at the fast-scan start edge of the
    centered footprint. The full scanline width is then added later by
    ``transform_coordinates_single_knot``.

    Parameters
    ----------
    input_shape : tuple[int, int]
        Raw image shape ``(num_rows, num_cols)``.
    output_shape : tuple[int, int]
        Padded canvas shape ``(num_rows, num_cols)``.
    scan_fast : NDArray
        Unit vector of the fast scan direction in ``(row, col)`` order.
    scan_slow : NDArray
        Unit vector of the slow scan direction in ``(row, col)`` order.
    number_knots : int
        Number of control knots per scanline.

    Returns
    -------
    NDArray
        Initial knot array with shape ``(2, input_rows, number_knots)``.
    """
    v_slow = np.linspace(-(input_shape[0] - 1) / 2, (input_shape[0] - 1) / 2, input_shape[0])
    u_fast = np.linspace(-(input_shape[1] - 1) / 2, (input_shape[1] - 1) / 2, number_knots)
    row_knots = ((output_shape[0] - 1) / 2
                 + u_fast[None, :] * scan_fast[0]
                 + v_slow[:, None] * scan_slow[0])
    col_knots = ((output_shape[1] - 1) / 2
                 + u_fast[None, :] * scan_fast[1]
                 + v_slow[:, None] * scan_slow[1])
    return np.stack([row_knots, col_knots], axis=0)


def bilinear_kde_batch(
    row_coords: torch.Tensor,
    col_coords: torch.Tensor,
    source_image: torch.Tensor,
    output_shape: tuple[int, int],
    kde_sigma: float,
    pad_value: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched bilinear KDE: scatter N source images onto an output canvas.

    Each pixel scatters its value to its 4 nearest grid neighbors with
    bilinear weights ``(1-dr)·(1-dc)``, ``dr·(1-dc)``, ``(1-dr)·dc``,
    ``dr·dc`` where ``dr, dc`` are fractional row/col distances.
    Accumulated counts and values are Gaussian-smoothed, then normalized:
    ``output = pad_value·(1-coverage) + coverage·(values/counts)``.

    Used by both the affine grid search (N = candidate drift vectors,
    single source image broadcast across drifts) and the nonrigid loop
    (N = stacked source images, one per drift).

    Parameters
    ----------
    row_coords : torch.Tensor
        Row coordinates of input pixels, shape ``(N, rows, cols)``.
    col_coords : torch.Tensor
        Column coordinates of input pixels, shape ``(N, rows, cols)``.
    source_image : torch.Tensor
        Pixel values to scatter. Either ``(rows, cols)`` (same image used
        for all N drifts - affine grid search) or ``(N, rows, cols)``
        (different image per drift - multi-image batched warping).
    output_shape : tuple[int, int]
        Canvas size ``(num_rows, num_cols)`` for the output images.
    kde_sigma : float
        Gaussian smoothing sigma in pixels.
    pad_value : float or torch.Tensor
        Fill value where pixel coverage is below threshold. If a tensor of
        shape ``(N,)``, applies a different pad value per drift.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(warped_images, sum_weights)`` - warped images and smoothed pixel coverage,
        both shape ``(N, num_rows, num_cols)``.
    """
    num_test_drifts = row_coords.shape[0]
    num_rows, num_cols = output_shape
    coverage_threshold = 1e-3
    # Flatten spatial dims - scatter_add_ works on 1D buffers
    row_flat = row_coords.flatten(1)
    col_flat = col_coords.flatten(1)
    # Stay in float for fractional distance, convert to int only for scatter indices
    row_floor = row_flat.floor()
    col_floor = col_flat.floor()
    frac_row = row_flat - row_floor
    frac_col = col_flat - col_floor
    row_floor = row_floor.int()
    col_floor = col_floor.int()
    if source_image.dim() == 3:
        # Per-drift source images: each drift scatters its own pixel values
        source_values_flat = source_image.flatten()
    else:
        source_values_flat = source_image.flatten().repeat(num_test_drifts)
    # All N batch entries scatter into one flat buffer - offset separates them
    batch_offsets = (
        torch.arange(num_test_drifts, device=row_coords.device, dtype=torch.int32)
        * num_rows * num_cols
    )[:, None]
    # Float32 accumulators - scatter_add_ requires source dtype to match,
    # so all input tensors must be float32 (raises on float64).
    sum_weights = torch.zeros(
        num_test_drifts * num_rows * num_cols, dtype=torch.float32, device=row_coords.device
    )
    sum_values = torch.zeros_like(sum_weights)
    # Periodic wrapping so pixels near edges scatter to the opposite side
    row_wrapped = row_floor % num_rows
    col_wrapped = col_floor % num_cols
    row_next = (row_wrapped + 1) % num_rows
    col_next = (col_wrapped + 1) % num_cols
    # Each pixel distributes its value to the 4 nearest grid neighbors
    # weighted by bilinear distance: (1-dr)(1-dc), dr(1-dc), (1-dr)dc, dr·dc
    for corner_row, corner_col, corner_weight in [
        (row_wrapped, col_wrapped, ((1 - frac_row) * (1 - frac_col)).flatten()),
        (row_next, col_wrapped, (frac_row * (1 - frac_col)).flatten()),
        (row_wrapped, col_next, ((1 - frac_row) * frac_col).flatten()),
        (row_next, col_next, (frac_row * frac_col).flatten()),
    ]:
        flat_indices = (corner_row * num_cols + corner_col + batch_offsets).flatten()
        sum_weights.scatter_add_(0, flat_indices, corner_weight)
        sum_values.scatter_add_(0, flat_indices, corner_weight * source_values_flat)
    sum_weights = sum_weights.reshape(num_test_drifts, num_rows, num_cols)
    sum_values = sum_values.reshape(num_test_drifts, num_rows, num_cols)
    # Smooth the scattered counts and values to fill gaps between pixels
    sum_weights = gaussian_smooth_batch(sum_weights, kde_sigma)
    sum_values = gaussian_smooth_batch(sum_values, kde_sigma)
    # Blend between pad_value (uncovered) and normalized values (covered),
    # ramping linearly with coverage to avoid hard edges at the boundary
    coverage_weight = torch.clamp(sum_weights / coverage_threshold, max=1.0)
    if isinstance(pad_value, torch.Tensor) and pad_value.dim() == 1:
        # Per-drift pad value: reshape (N,) → (N, 1, 1) for broadcasting
        pad_value = pad_value[:, None, None]
    warped_images = pad_value * (1 - coverage_weight) + coverage_weight * (
        sum_values / torch.clamp(sum_weights, min=1e-8)
    )
    return warped_images, sum_weights


def cross_corr_batch(
    ref_images: torch.Tensor,
    mov_images: torch.Tensor,
    upsample_factor: int,
    max_shift_mask: torch.Tensor | None = None,
    freq_grids: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Score test drift vectors by cross-correlation alignment cost.

    Core cost function of the affine grid search. For each test drift,
    measures how well the warped image pairs align after sub-pixel
    translation correction. Without this, the grid search has no way
    to rank test drifts - it is the signal that drives drift estimation.

    Pipeline: FFT cross-correlation → parabolic coarse peak →
    DFT upsample for sub-pixel refinement → Fourier-domain shift →
    MAE between reference and aligned image.

    Parameters
    ----------
    ref_images : torch.Tensor
        Reference images, shape ``(N, num_rows, num_cols)``.
    mov_images : torch.Tensor
        Images to align, shape ``(N, num_rows, num_cols)``.
    upsample_factor : int
        Sub-pixel upsampling factor for DFT refinement.
    max_shift_mask : torch.Tensor or None
        Precomputed boolean mask, shape ``(num_rows, num_cols)``.
        True where correlation peaks should be zeroed (beyond max shift).
    freq_grids : tuple[torch.Tensor, torch.Tensor] or None, optional
        Precomputed ``(freq_row, freq_col)`` from ``torch.fft.fftfreq``,
        shapes ``(num_rows, 1)`` and ``(1, num_cols)``. Avoids
        recomputing the same grids each call. Default is None.

    Returns
    -------
    torch.Tensor
        MAE cost per pair, shape ``(N,)``.

    Examples
    --------
    >>> ref = torch.randn(5, 64, 64, dtype=torch.float64)
    >>> mov = torch.randn(5, 64, 64, dtype=torch.float64)
    >>> cost = cross_corr_batch(ref, mov, 8, 32.0)
    >>> cost.shape
    torch.Size([5])
    """
    num_test_drifts, num_rows, num_cols = ref_images.shape
    dtype = ref_images.dtype
    mov_fft = fft2(mov_images)
    cross_corr_fft = fft2(ref_images) * mov_fft.conj()
    cross_corr = ifft2(cross_corr_fft).real
    # Reject correlation peaks beyond max shift to avoid locking onto
    # periodic lattice repeats or noise peaks far from the true shift
    if max_shift_mask is not None:
        cross_corr.masked_fill_(max_shift_mask[None], 0.0)
    # Find best-matching shift: integer peak → parabola (~0.1 px) → DFT (~0.01 px)
    peak_flat_idx = cross_corr.flatten(1).argmax(dim=1)
    peak_row = peak_flat_idx // num_cols
    peak_col = peak_flat_idx % num_cols
    batch_idx = torch.arange(num_test_drifts, device=ref_images.device)
    refined_row, refined_col = _parabolic_peak_2d(
        cross_corr, peak_row, peak_col, num_rows, num_cols, batch_idx
    )
    image_shifts = _dft_refine_shifts(
        cross_corr_fft, refined_row, refined_col, upsample_factor
    )
    # Wrap from [0, N) to [-N/2, N/2) so shifts represent actual displacement
    image_shifts[:, 0] = ((image_shifts[:, 0] + num_rows / 2) % num_rows) - num_rows / 2
    image_shifts[:, 1] = ((image_shifts[:, 1] + num_cols / 2) % num_cols) - num_cols / 2
    if freq_grids is not None:
        freq_row, freq_col = freq_grids
    else:
        freq_row = fftfreq(num_rows, device=ref_images.device, dtype=dtype)[:, None]
        freq_col = fftfreq(num_cols, device=ref_images.device, dtype=dtype)[None, :]
    phase = -2j * math.pi * (
        freq_row[None] * image_shifts[:, 0, None, None]
        + freq_col[None] * image_shifts[:, 1, None, None]
    )
    aligned_images = ifft2(mov_fft * torch.exp(phase)).real
    return torch.mean(torch.abs(ref_images - aligned_images), dim=(1, 2))


def translate_align(
    warped_images: torch.Tensor,
    upsample_factor: int,
    max_image_shift: float | None,
) -> torch.Tensor:
    """Pairwise translation alignment of warped images via cross-correlation.

    Called by ``_warp_and_translate_torch`` after each affine warp to
    remove residual translational misalignment between the image pair.
    Without this step, the merged image would be blurred by the remaining
    translation offset even after the affine drift is corrected.

    Starting from image 0 as reference, sequentially aligns each image
    using FFT cross-correlation with parabolic + DFT sub-pixel refinement.
    The reference is updated as a running Fourier-domain average.

    Parameters
    ----------
    warped_images : torch.Tensor
        Warped images, shape ``(num_images, num_rows, num_cols)``.
    upsample_factor : int
        Sub-pixel precision (1/N pixel) for DFT refinement.
    max_image_shift : float or None
        Maximum allowed shift in pixels. Peaks beyond this radius are masked.

    Returns
    -------
    torch.Tensor
        Zero-mean shifts, shape ``(num_images, 2)`` in (row, col) order.
    """
    num_images, num_rows, num_cols = warped_images.shape
    dtype = warped_images.dtype
    device = warped_images.device
    image_shifts = torch.zeros(num_images, 2, dtype=dtype, device=device)
    ref_fft = fft2(warped_images[0])
    # Reject bad correlation peaks from noise or periodicity
    # by zeroing everything beyond max_image_shift pixels from origin
    shift_mask = None
    if max_image_shift is not None:
        dist_row = fftfreq(num_rows, 1.0 / num_rows, device=device, dtype=dtype)
        dist_col = fftfreq(num_cols, 1.0 / num_cols, device=device, dtype=dtype)
        shift_mask = dist_row[:, None] ** 2 + dist_col[None, :] ** 2 >= max_image_shift ** 2
    freq_row = fftfreq(num_rows, device=device, dtype=dtype)[:, None]
    freq_col = fftfreq(num_cols, device=device, dtype=dtype)[None, :]
    for img_idx in range(1, num_images):
        mov_fft = fft2(warped_images[img_idx])
        cross_corr_fft = ref_fft * mov_fft.conj()
        cross_corr = ifft2(cross_corr_fft).real
        if shift_mask is not None:
            cross_corr.masked_fill_(shift_mask, 0.0)
        # Find the integer peak, refine with parabola to ~0.1 px
        peak_flat_idx = cross_corr.flatten().argmax()
        peak_row = peak_flat_idx[None] // num_cols
        peak_col = peak_flat_idx[None] % num_cols
        batch_idx = torch.zeros(1, dtype=torch.long, device=device)
        refined_row, refined_col = _parabolic_peak_2d(
            cross_corr[None], peak_row, peak_col, num_rows, num_cols, batch_idx
        )
        # Zoom into peak neighborhood with DFT to get ~0.01 px precision,
        # then wrap from [0, N) to centered [-N/2, N/2) convention
        refined_shift = _dft_refine_shifts(
            cross_corr_fft[None], refined_row, refined_col, upsample_factor
        )
        image_shifts[img_idx, 0] = ((refined_shift[0, 0] + num_rows / 2) % num_rows) - num_rows / 2
        image_shifts[img_idx, 1] = ((refined_shift[0, 1] + num_cols / 2) % num_cols) - num_cols / 2
        # Apply the recovered shift to current image via Fourier shift theorem,
        # then blend into running average so later images align to the cumulative mean
        phase = torch.exp(
            -2j * math.pi * (
                freq_row * image_shifts[img_idx, 0] + freq_col * image_shifts[img_idx, 1]
            )
        )
        ref_fft = ref_fft * img_idx / (img_idx + 1) + mov_fft * phase / (img_idx + 1)
    # Remove mean so shifts are relative (no absolute reference frame)
    image_shifts -= image_shifts.mean(dim=0)
    return image_shifts


def transform_coordinates_single_knot(
    knots: torch.Tensor,
    scan_fast: torch.Tensor,
    input_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-knot fast path: map source pixels to canvas coordinates.

    **Single-knot only.** Each scanline has exactly one (row, col) anchor;
    the fast-scan-direction position is filled in by linear interpolation
    along the scanline. Multi-knot Bezier interpolation is intentionally
    not supported here - that's the scipy backend's job. The pytorch path
    optimizes for the common single-knot case (≥95% of real STEM workflows).

    Called by ``preprocess``, ``_affine_grid_search_batch``, and
    ``_warp_and_translate_torch`` to map source image pixels onto the
    padded output canvas. Without this, the warped images would have
    no spatial mapping and the grid search couldn't score test drifts.

    Each input row maps to a line on the canvas:
    ``row = knot_row + fraction * scan_fast[0] * (num_rows - 1)``
    ``col = knot_col + fraction * scan_fast[1] * (num_cols - 1)``
    where row and col dimensions scale independently for non-square images.

    Parameters
    ----------
    knots : torch.Tensor
        Knot positions, shape ``(2, num_rows, 1)``. First dim is (row, col).
        The trailing 1 is the single-knot dimension; multi-knot inputs are
        rejected by the caller before reaching this function.
    scan_fast : torch.Tensor
        Fast scan direction vector, shape ``(2,)``.
    input_shape : tuple[int, int]
        Original image shape ``(num_rows, num_cols)``.

    Returns
    -------
    row_coords : torch.Tensor
        Row coordinates on canvas, shape ``(num_rows, num_cols)``.
    col_coords : torch.Tensor
        Column coordinates on canvas, shape ``(num_rows, num_cols)``.

    Examples
    --------
    >>> knots = torch.zeros(2, 64, 1)
    >>> scan_fast = torch.tensor([0.0, 1.0])
    >>> r, c = transform_coordinates_single_knot(knots, scan_fast, (64, 64))
    >>> r.shape
    torch.Size([64, 64])
    """
    num_rows, num_cols = input_shape
    fast_fraction = torch.linspace(0, 1, num_cols, dtype=knots.dtype, device=knots.device)
    row_coords = knots[0, :, 0:1] + fast_fraction[None, :] * scan_fast[0] * (num_rows - 1)
    col_coords = knots[1, :, 0:1] + fast_fraction[None, :] * scan_fast[1] * (num_cols - 1)
    return row_coords, col_coords


def backward_warp(
    images: torch.Tensor,
    drift: tuple[float, float] | torch.Tensor,
    rigid_shift: tuple[float, float] = (0.0, 0.0),
    mode: str = "bicubic",
) -> torch.Tensor:
    """Apply drift correction via backward interpolation (``grid_sample``).

    Builds a per-scanline sampling grid that undoes the estimated drift
    and optional rigid translation, then resamples with the chosen
    interpolation kernel.

    Parameters
    ----------
    images : torch.Tensor
        Images to correct, shape ``(N, H, W)`` or ``(H, W)``.
        For 4D-STEM, pass detector-pixel slices in chunks.
    drift : tuple[float, float] | torch.Tensor
        **Affine mode** — ``(row_slope, col_slope)`` scalar drift rate
        in pixels per scan line, as returned by ``align_affine``.

        **Per-row mode** — a ``torch.Tensor`` of shape ``(2, H)`` or
        ``(H, 2)`` giving the per-scanline shift in ``(row, col)``
        order.  Typically obtained from ``align_nonrigid`` knot deltas::

            delta = dc.knots[1] - knots_initial  # (2, H, 1)
            drift_per_row = delta[:, :, 0]        # (2, H)

        In per-row mode *rigid_shift* is ignored (it is already
        incorporated in the per-row values).
    rigid_shift : tuple[float, float], default (0.0, 0.0)
        Global ``(row, col)`` translation to apply (affine mode only).
    mode : str, default "bicubic"
        Interpolation kernel passed to ``grid_sample``.
        ``"bicubic"`` preserves more high-frequency content than
        ``"bilinear"`` (e.g. retains ~93% at 0.3× Nyquist vs ~69%
        for bilinear with a 0.3 px sub-pixel shift).

    Returns
    -------
    torch.Tensor
        Corrected images, same shape as *images*.
    """
    squeeze = images.dim() == 2
    if squeeze:
        images = images[None]
    n, h, w = images.shape
    device, dtype = images.device, images.dtype

    if isinstance(drift, torch.Tensor):
        # Per-row mode: drift is (2, H) or (H, 2)
        if drift.shape == (h, 2):
            drift = drift.T  # → (2, H)
        if drift.shape != (2, h):
            msg = f"Per-row drift must be (2, {h}) or ({h}, 2), got {drift.shape}"
            raise ValueError(msg)
        drift = drift.to(device=device, dtype=dtype)
        sample_r = (
            torch.arange(h, device=device, dtype=dtype)[:, None].expand(-1, w)
            - drift[0][:, None]
        )
        sample_c = (
            torch.arange(w, device=device, dtype=dtype)[None, :].expand(h, -1)
            - drift[1][:, None]
        )
    else:
        # Affine mode: drift is (row_slope, col_slope)
        offset = torch.arange(h, device=device, dtype=dtype) - (h - 1) / 2
        drift_row, drift_col = drift
        shift_row, shift_col = rigid_shift
        sample_r = (
            torch.arange(h, device=device, dtype=dtype)[:, None].expand(-1, w)
            - drift_row * offset[:, None]
            - shift_row
        )
        sample_c = (
            torch.arange(w, device=device, dtype=dtype)[None, :].expand(h, -1)
            - drift_col * offset[:, None]
            - shift_col
        )

    grid_row = 2.0 * sample_r / (h - 1) - 1.0
    grid_col = 2.0 * sample_c / (w - 1) - 1.0
    # Pass as (1, N, H, W) with a single (1, H, W, 2) grid so grid_sample applies
    # one grid to all N channels in one kernel call. The alternative (N, 1, H, W)
    # with (N, H, W, 2) materialises N identical grids — 4096× more memory for EDS.
    grid = torch.stack([grid_col, grid_row], dim=-1)[None]  # (1, H, W, 2) — col first per grid_sample convention

    out = torch.nn.functional.grid_sample(
        images[None], grid, mode=mode,
        align_corners=True, padding_mode="border",
    )[0]  # (N, H, W)
    return out[0] if squeeze else out


@torch.inference_mode()
def fourier_shift_warp(
    images: torch.Tensor,
    drift: tuple[float, float],
    rigid_shift: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    """Apply drift correction via per-row Fourier phase shifts.

    For each scanline the column-direction shift is applied as a phase
    ramp in Fourier space (mathematically exact, **zero interpolation
    blur**).  The row-direction shift is handled by linearly mixing
    adjacent source rows before the Fourier step — only one axis uses
    spatial interpolation, and with typical STEM drift rates the row
    component is small.

    This preserves high-frequency features (lattice fringes, Bragg
    spots) better than ``backward_warp`` with any spatial interpolation
    kernel, at the cost of ~2× runtime due to per-row FFTs.

    Parameters
    ----------
    images : torch.Tensor
        Images to correct, shape ``(N, H, W)`` or ``(H, W)``.
    drift : tuple[float, float]
        Affine drift rate ``(row_slope, col_slope)`` in pixels per scan
        line.
    rigid_shift : tuple[float, float], default (0.0, 0.0)
        Global ``(row, col)`` translation.

    Returns
    -------
    torch.Tensor
        Corrected images, same shape as *images*.
    """
    squeeze = images.dim() == 2
    if squeeze:
        images = images[None]
    n, h, w = images.shape
    device, dtype = images.device, images.dtype

    offset = torch.arange(h, device=device, dtype=dtype) - (h - 1) / 2
    drift_row, drift_col = drift
    shift_row, shift_col = rigid_shift

    # Per-row shifts to undo
    row_shifts = drift_row * offset + shift_row  # (H,)
    col_shifts = drift_col * offset + shift_col  # (H,)

    # Phase 1: row-direction correction via linear mixing of adjacent rows
    # Source row for output row r: r_src = r - row_shifts[r]
    src_rows = torch.arange(h, device=device, dtype=dtype) - row_shifts
    src_rows_floor = src_rows.floor().long()
    frac = (src_rows - src_rows.floor()).to(dtype)  # (H,)

    # Clamp to valid range
    r0 = src_rows_floor.clamp(0, h - 1)
    r1 = (src_rows_floor + 1).clamp(0, h - 1)

    # Linear mix: images[:, r0[r], :] * (1-f) + images[:, r1[r], :] * f
    row_corrected = images[:, r0] * (1.0 - frac)[None, :, None] + images[:, r1] * frac[None, :, None]

    # Phase 2: column-direction correction via Fourier phase ramp (exact)
    freq_col = torch.fft.fftfreq(w, device=device, dtype=dtype)  # (W,)
    # To undo a column shift of +d pixels, we apply phase exp(+2πi·k·d)
    # which shifts the signal by -d in the spatial domain.
    phase = torch.exp(2j * torch.pi * freq_col[None, :] * col_shifts[:, None].to(torch.complex64))  # (H, W)

    # Apply batched FFT → phase shift → IFFT (vectorised across batch)
    spectrum = torch.fft.fft(row_corrected.to(torch.complex64), dim=-1)  # (N, H, W)
    spectrum *= phase[None, :, :]  # broadcast phase (H, W) across batch
    out = torch.fft.ifft(spectrum, dim=-1).real.to(dtype)  # (N, H, W)

    return out[0] if squeeze else out


@torch.inference_mode()
def backward_warp_grid_search(
    ref_image: torch.Tensor,
    mov_image: torch.Tensor,
    drift_vectors: torch.Tensor,
    upsample_factor: int,
    max_image_shift: float | None,
    chunk_size: int | None = None,
) -> tuple[int, torch.Tensor]:
    """Score drift candidates by backward-warping the moving image.

    For each candidate ``(dr, dc)``, builds a sampling grid that undoes the
    drift::

        sample_row[i, j] = i - dr * (i - center)
        sample_col[i, j] = j - dc * (i - center)

    then backward-warps the moving image with ``grid_sample`` (bicubic) and
    scores alignment with the reference via ``cross_corr_batch``.

    This avoids forward-scatter KDE artifacts that bias the cost when only
    one image's geometry changes (``fixed_indices`` mode).  The periodic
    wrapping in ``bilinear_kde_batch`` creates geometry-dependent seam
    artifacts that differ between the fixed reference and the drift-shifted
    moving image, making the MAE minimum diverge from the true drift.
    Backward-warp scoring eliminates this by working at original resolution
    without any canvas or KDE.

    Parameters
    ----------
    ref_image : torch.Tensor
        Reference image, shape ``(H, W)``.
    mov_image : torch.Tensor
        Moving image to correct, shape ``(H, W)``.
    drift_vectors : torch.Tensor
        Candidate drift rates, shape ``(N, 2)`` — columns ``(row_rate, col_rate)``.
    upsample_factor : int
        Sub-pixel precision for cross-correlation refinement.
    max_image_shift : float or None
        Maximum allowed translational shift in pixels.
    chunk_size : int or None
        Candidates per GPU pass.  ``None`` auto-selects based on free memory.

    Returns
    -------
    tuple[int, torch.Tensor]
        Index of the best candidate and full cost tensor of shape ``(N,)``.
    """
    device = ref_image.device
    dtype = ref_image.dtype
    h, w = ref_image.shape
    num_candidates = drift_vectors.shape[0]
    center = (h - 1) / 2.0

    rows = torch.arange(h, device=device, dtype=dtype)
    cols = torch.arange(w, device=device, dtype=dtype)
    offset = rows - center  # (H,)

    shift_mask = None
    if max_image_shift is not None:
        dist_r = fftfreq(h, 1.0 / h, device=device, dtype=dtype)
        dist_c = fftfreq(w, 1.0 / w, device=device, dtype=dtype)
        shift_mask = dist_r[:, None] ** 2 + dist_c[None, :] ** 2 >= max_image_shift ** 2
    freq_grids = (
        fftfreq(h, device=device, dtype=dtype)[:, None],
        fftfreq(w, device=device, dtype=dtype)[None, :],
    )

    if chunk_size is None:
        if device.type == "cuda":
            bytes_per_element = torch.finfo(dtype).bits // 8
            # grid (H*W*2) + warped (H*W) + FFT buffers (H*W*8*2)
            per_cand_bytes = h * w * bytes_per_element * (2 + 1 + 16)
            free_bytes, _ = torch.cuda.mem_get_info(device)
            chunk_size = max(1, int(free_bytes * 0.4 / per_cand_bytes))
            chunk_size = min(chunk_size, num_candidates)
        else:
            chunk_size = num_candidates

    all_costs = []
    for chunk_start in range(0, num_candidates, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_candidates)
        n_chunk = chunk_end - chunk_start
        drift_chunk = drift_vectors[chunk_start:chunk_end]

        drift_row_rate = drift_chunk[:, 0]  # (n_chunk,)
        drift_col_rate = drift_chunk[:, 1]  # (n_chunk,)

        row_shift = drift_row_rate[:, None] * offset[None, :]  # (n_chunk, H)
        col_shift = drift_col_rate[:, None] * offset[None, :]  # (n_chunk, H)

        sample_rows = rows[None, :, None].expand(n_chunk, h, w) - row_shift[:, :, None]
        sample_cols = cols[None, None, :].expand(n_chunk, h, w) - col_shift[:, :, None]

        grid_row = 2.0 * sample_rows / (h - 1) - 1.0
        grid_col = 2.0 * sample_cols / (w - 1) - 1.0
        grid = torch.stack([grid_col, grid_row], dim=-1)  # col first per grid_sample convention

        warped = torch.nn.functional.grid_sample(
            mov_image[None, None].expand(n_chunk, 1, h, w), grid,
            mode="bicubic", align_corners=True, padding_mode="border",
        )[:, 0]

        ref_batch = ref_image[None].expand(n_chunk, -1, -1)
        costs = cross_corr_batch(
            ref_batch, warped, upsample_factor,
            max_shift_mask=shift_mask, freq_grids=freq_grids,
        )
        all_costs.append(costs)

    all_costs = torch.cat(all_costs)
    return torch.argmin(all_costs).item(), all_costs


def gaussian_smooth_batch(
    field_stack: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Batched 2D Gaussian smoothing matching ``scipy.ndimage.gaussian_filter``.

    Used by ``bilinear_kde_batch`` to smooth scattered counts and
    values before normalization. Without smoothing, the warped images
    have salt-and-pepper artifacts from the scatter step.

    Parameters
    ----------
    field_stack : torch.Tensor
        Input tensor of shape ``(N, num_rows, num_cols)``.
    sigma : float
        Standard deviation of the Gaussian kernel in pixels.

    Returns
    -------
    torch.Tensor
        Smoothed tensor of shape ``(N, num_rows, num_cols)``.

    """
    kernel, radius = _gaussian_kernel_1d(sigma, field_stack.dtype, field_stack.device)
    # Separable kernel: column pass then row pass to halve FLOPs vs full 2D conv
    kernel_col = kernel[None, None, None, :]
    kernel_row = kernel[None, None, :, None]
    field_stack = field_stack[:, None]
    field_stack = torch.nn.functional.conv2d(_symmetric_pad(field_stack, 0, radius), kernel_col)
    field_stack = torch.nn.functional.conv2d(_symmetric_pad(field_stack, radius, 0), kernel_row)
    return field_stack[:, 0]


def gaussian_smooth_1d(
    signal: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """1D Gaussian smoothing matching ``scipy.ndimage.gaussian_filter``.

    Smooths each row of the input independently using a separable 1D kernel.
    Used for regularizing knot displacement vectors in the nonrigid loop,
    where the signal is 1D (one value per scan line).

    Parameters
    ----------
    signal : torch.Tensor
        Input tensor of shape ``(N, L)`` - N channels, L samples.
    sigma : float
        Standard deviation of the Gaussian kernel in pixels.

    Returns
    -------
    torch.Tensor
        Smoothed tensor of shape ``(N, L)``.
    """
    kernel, radius = _gaussian_kernel_1d(sigma, signal.dtype, signal.device)
    signal_padded = _symmetric_pad_1d(signal[:, None], radius)
    return torch.nn.functional.conv1d(signal_padded, kernel[None, None, :])[:, 0]


# ---------------------------------------------------------------------------
# Building blocks - used internally by the public API functions above
# ---------------------------------------------------------------------------


def _dft_refine_shifts(
    cross_corr_fft: torch.Tensor,
    peak_row: torch.Tensor,
    peak_col: torch.Tensor,
    upsample_factor: int,
) -> torch.Tensor:
    """Refine coarse sub-pixel shifts using DFT upsampling + parabolic fit.

    After ``_parabolic_peak_2d`` gives a coarse sub-pixel position, this
    function zooms into a small neighborhood via the matrix-multiply DFT
    and applies a second parabolic refinement on the upsampled patch.
    The result is sub-pixel shifts with ``1 / upsample_factor`` precision.

    Without this step, shifts would have only ~0.1 px precision from
    parabolic fitting alone. With ``upsample_factor=8``, precision
    improves to ~0.01 px.

    Parameters
    ----------
    cross_corr_fft : torch.Tensor
        Complex cross-correlation in Fourier domain, ``(N, num_rows, num_cols)``.
    peak_row, peak_col : torch.Tensor
        Coarse sub-pixel peak positions in [0, N) from ``_parabolic_peak_2d``.
    upsample_factor : int
        Sub-pixel precision factor.

    Returns
    -------
    image_shifts : torch.Tensor
        Sub-pixel shifts in [0, N) coordinates, shape ``(N, 2)``.
    """
    num_test_drifts = cross_corr_fft.shape[0]
    dtype = peak_row.dtype
    batch_idx = torch.arange(num_test_drifts, device=cross_corr_fft.device)
    # Evaluate the correlation surface at 1/upsample_factor pixel spacing
    # in a small window around each coarse peak - gives actual values,
    # not the parabolic approximation from step 1
    upsampled_corr = _dft_upsample_batch(
        cross_corr_fft, upsample_factor, torch.stack([peak_row, peak_col], dim=1)
    )
    upsample_size = upsampled_corr.shape[1]
    peak_flat_idx = upsampled_corr.flatten(1).argmax(dim=1)
    local_row = peak_flat_idx // upsample_size
    local_col = peak_flat_idx % upsample_size
    # Final parabolic fit on the dense grid for last fraction of precision.
    # Peaks at the edge of the upsampled window can't use the 3-point stencil
    # (no neighbor on one side), so those are masked and kept at integer position
    can_refine = (
        (local_row >= 1)
        & (local_row < upsample_size - 1)
        & (local_col >= 1)
        & (local_col < upsample_size - 1)
    )
    peak_val = upsampled_corr[batch_idx, local_row, local_col]
    d_row_fine = _parabolic_sub_pixel(
        upsampled_corr[batch_idx, (local_row - 1).clamp(min=0), local_col],
        peak_val,
        upsampled_corr[batch_idx, (local_row + 1).clamp(max=upsample_size - 1), local_col],
        mask=can_refine,
    )
    d_col_fine = _parabolic_sub_pixel(
        upsampled_corr[batch_idx, local_row, (local_col - 1).clamp(min=0)],
        peak_val,
        upsampled_corr[batch_idx, local_row, (local_col + 1).clamp(max=upsample_size - 1)],
        mask=can_refine,
    )
    # Convert upsampled-grid position back to image-pixel coordinates:
    # patch center is at index patch_radius in the upsampled grid,
    # so (local_row - patch_radius) / upsample_factor = offset from coarse peak
    patch_radius = math.ceil(1.5 * upsample_factor)
    image_shifts = torch.zeros(num_test_drifts, 2, dtype=dtype, device=cross_corr_fft.device)
    # local_row/col are int from argmax - cast to float for sub-pixel arithmetic
    image_shifts[:, 0] = peak_row + (local_row.to(dtype) - patch_radius + d_row_fine) / upsample_factor
    image_shifts[:, 1] = peak_col + (local_col.to(dtype) - patch_radius + d_col_fine) / upsample_factor
    return image_shifts


def _dft_upsample_batch(
    cross_corr_fft: torch.Tensor,
    upsample_factor: int,
    peak_positions: torch.Tensor,
) -> torch.Tensor:
    """Sub-pixel peak refinement for all test drifts in one pass.

    After the coarse FFT cross-correlation finds integer-pixel peaks,
    zooms into a small neighborhood using the Guizar-Sicairos
    matrix-multiply DFT. Without DFT upsampling, shift precision is
    limited to ~0.1 px from parabolic fitting alone. With
    ``upsample_factor=8``, precision improves to ~0.01 px.

    Parameters
    ----------
    cross_corr_fft : torch.Tensor
        Complex 2D cross-correlation in Fourier domain, shape ``(N, num_rows, num_cols)``.
    upsample_factor : int
        Upsampling factor (typically 8).
    peak_positions : torch.Tensor
        Coarse peak locations ``(row, col)`` per test drift, shape ``(N, 2)``.

    Returns
    -------
    torch.Tensor
        Real-valued upsampled correlation neighborhoods, shape ``(N, P, P)``
        where ``P = 2 * ceil(1.5 * upsample_factor) + 1``.

    """
    num_test_drifts, num_rows, num_cols = cross_corr_fft.shape
    real_dtype = torch.float32 if cross_corr_fft.dtype == torch.complex64 else torch.float64
    # 1.5x radius ensures the patch captures the true peak after parabolic shift
    patch_radius = math.ceil(1.5 * upsample_factor)
    # Upsampled grid positions centered at zero: [-radius, ..., 0, ..., +radius]
    upsample_grid = torch.arange(-patch_radius, patch_radius + 1, dtype=real_dtype, device=cross_corr_fft.device)
    # ifftshift reorders [0,1,...,N-1] to match FFT output ordering,
    # then subtract N//2 to center at zero
    freq_row_base = ifftshift(
        torch.arange(num_rows, dtype=real_dtype, device=cross_corr_fft.device)
    ) - num_rows // 2
    freq_col_base = ifftshift(
        torch.arange(num_cols, dtype=real_dtype, device=cross_corr_fft.device)
    ) - num_cols // 2
    freq_row = freq_row_base[None, :] + (peak_positions[:, 0] - num_rows // 2)[:, None]
    freq_col = freq_col_base[None, :] + (peak_positions[:, 1] - num_cols // 2)[:, None]
    # Guizar-Sicairos matrix-multiply DFT: K_row @ CC @ K_col
    kern_row = torch.exp(
        -2j * math.pi / (num_rows * upsample_factor)
        * upsample_grid[None, :, None] * freq_row[:, None, :]
    ).to(cross_corr_fft.dtype)  # real → complex for matrix multiply
    kern_col = torch.exp(
        -2j * math.pi / (num_cols * upsample_factor)
        * freq_col[:, :, None] * upsample_grid[None, None, :]
    ).to(cross_corr_fft.dtype)  # real → complex for matrix multiply
    # (N,P,M) @ (N,M,K) @ (N,K,P) -> (N,P,P)
    return (kern_row @ cross_corr_fft @ kern_col).real

# ---------------------------------------------------------------------------
# Primitives - lowest-level operations
# ---------------------------------------------------------------------------


def _parabolic_peak_2d(
    cross_corr: torch.Tensor,
    peak_row: torch.Tensor,
    peak_col: torch.Tensor,
    num_rows: int,
    num_cols: int,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refine an integer cross-correlation peak to sub-pixel precision.

    Extracts the 3-point stencil along each axis and fits a parabola.
    Without this, the DFT upsample window would be centered on the
    integer peak which may be up to 0.5 px away from the true peak,
    causing the upsampled patch to miss the true maximum.

    Parameters
    ----------
    cross_corr : torch.Tensor
        Batched correlation map, shape ``(N, num_rows, num_cols)``.
    peak_row, peak_col : torch.Tensor
        Integer peak positions, shape ``(N,)``.
    num_rows, num_cols : int
        Dimensions for periodic wrapping.
    batch_idx : torch.Tensor
        Batch indices, ``torch.arange(N)``.

    Returns
    -------
    refined_row, refined_col : torch.Tensor
        Sub-pixel peak positions in [0, N) coordinates.
    """
    dtype = cross_corr.dtype
    val_center = cross_corr[batch_idx, peak_row, peak_col]
    val_row_m1 = cross_corr[batch_idx, (peak_row - 1) % num_rows, peak_col]
    val_row_p1 = cross_corr[batch_idx, (peak_row + 1) % num_rows, peak_col]
    val_col_m1 = cross_corr[batch_idx, peak_row, (peak_col - 1) % num_cols]
    val_col_p1 = cross_corr[batch_idx, peak_row, (peak_col + 1) % num_cols]
    # peak_row/col are int from argmax - cast to float for sub-pixel addition.
    # Double modulo handles tiny negative offsets from float32 rounding
    # that would otherwise wrap to N instead of 0 (e.g. -4e-8 % 64 = 64.0)
    refined_row = ((peak_row.to(dtype) + _parabolic_sub_pixel(val_row_m1, val_center, val_row_p1)) % num_rows) % num_rows
    refined_col = ((peak_col.to(dtype) + _parabolic_sub_pixel(val_col_m1, val_center, val_col_p1)) % num_cols) % num_cols
    return refined_row, refined_col


def _parabolic_sub_pixel(
    val_m1: torch.Tensor,
    val_0: torch.Tensor,
    val_p1: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sub-pixel offset from a 3-point stencil via parabolic interpolation.

    Cross-correlation peaks fall on integer pixel positions, but the true
    shift is usually between pixels. Fitting a parabola through the peak
    and its two neighbors gives ~0.1 px precision cheaply:
    ``offset = (val_p1 - val_m1) / (4·val_0 - 2·val_p1 - 2·val_m1)``.
    Without this, the DFT upsample window may be centered on the wrong
    pixel and miss the true peak.
    """
    denom = 4 * val_0 - 2 * val_p1 - 2 * val_m1
    valid = denom != 0
    if mask is not None:
        valid = valid & mask
    return torch.where(valid, (val_p1 - val_m1) / denom, torch.zeros_like(denom))


def _symmetric_pad_1d(signal: torch.Tensor, pad: int) -> torch.Tensor:
    """Symmetric 1D padding matching scipy's reflect mode.

    Same edge-repeat semantics as ``_symmetric_pad`` but for 1D signals.
    Used by ``gaussian_smooth_1d`` for regularization of knot vectors.
    """
    left = signal[:, :, :pad].flip(-1)
    right = signal[:, :, -pad:].flip(-1)
    return torch.cat([left, signal, right], dim=-1)


def _symmetric_pad(
    field_stack: torch.Tensor,
    pad_rows: int,
    pad_cols: int,
) -> torch.Tensor:
    """Symmetric padding matching scipy's reflect mode for parity.

    Without this, the torch and numpy Gaussian smoothing paths produce
    different results near canvas edges, breaking numerical parity.
    
    Scipy's ``mode='reflect'`` repeats the edge pixel
    (``[1,2,3]`` → ``[2,1,1,2,3,3,2]``), but PyTorch's
    ``F.pad(mode='reflect')`` does not (``[1,2,3]`` → ``[3,2,1,2,3,2,1]``).

    Parameters
    ----------
    field_stack : torch.Tensor
        Input tensor of shape ``(N, C, num_rows, num_cols)``.
    pad_rows : int
        Number of rows to pad on top and bottom.
    pad_cols : int
        Number of columns to pad on left and right.

    Returns
    -------
    torch.Tensor
        Padded tensor.

    Examples
    --------
    >>> t = torch.tensor([[[[1., 2., 3.]]]])
    >>> _symmetric_pad(t, 0, 2)
    tensor([[[[2., 1., 1., 2., 3., 3., 2.]]]])
    """
    if pad_cols > 0:
        left = field_stack[:, :, :, :pad_cols].flip(-1)
        right = field_stack[:, :, :, -pad_cols:].flip(-1)
        field_stack = torch.cat([left, field_stack, right], dim=-1)
    if pad_rows > 0:
        top = field_stack[:, :, :pad_rows, :].flip(-2)
        bottom = field_stack[:, :, -pad_rows:, :].flip(-2)
        field_stack = torch.cat([top, field_stack, bottom], dim=-2)
    return field_stack


def _gaussian_kernel_1d(sigma: float, dtype: torch.dtype, device: torch.device, _cache: dict = {}) -> torch.Tensor:
    """Normalized 1D Gaussian ``exp(-0.5*(x/sigma)^2)``, radius ``4*sigma``.

    Cached via mutable default arg - the grid search calls this ~800 times
    with the same sigma, saving ~44ms of redundant kernel construction.
    """
    key = (sigma, dtype, device)
    if key not in _cache:
        radius = int(4 * sigma + 0.5)
        offsets = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
        kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
        _cache[key] = (kernel / kernel.sum(), radius)
    return _cache[key]
