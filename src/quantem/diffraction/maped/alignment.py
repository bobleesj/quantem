"""Fourier correlation and scan-shift routines for MAPED."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def tukey_torch(N, alpha=0.5, device=None, dtype=torch.float32):
    """
    Creates a 1D Tukey window of length N and shape parameter alpha.

    Parameters
    ----------
    N : int
        Length of the window.
    alpha : float
        Shape parameter for the Tukey window.
    device : torch.device | str
        Device on which to create the window.
    dtype : torch.dtype
        torch.dtype, Data type of the window.

    Returns
    -------
    window : torch.Tensor
        1D Tukey window of length N.
    """
    n = torch.arange(N, device=device, dtype=dtype)
    w = torch.ones(N, device=device, dtype=dtype)

    if alpha <= 0:
        return w
    if alpha >= 1:
        return torch.hann_window(N, device=device, dtype=dtype)

    edge = alpha * (N - 1) / 2

    left = n < edge
    right = n >= (N - 1 - edge)

    w[left] = 0.5 * (1 + torch.cos(torch.pi * (2 * n[left] / (alpha * (N - 1)) - 1)))

    w[right] = 0.5 * (1 + torch.cos(torch.pi * (2 * n[right] / (alpha * (N - 1)) - 2 / alpha + 1)))

    return w


def shift_images_torch(
    images,
    shifts_rc,
    mode="bilinear",
    blend: bool = False,
    edge_blend: float = 8.0,
    padding=None,
    pad_val: str | float = 0.0,
):
    """
    Shift (and optionally blend) a stack of 2D images by per-image (dr, dc) pixel shifts using grid_sample.

    Parameters
    ----------
    images : torch.Tensor, shape (n, H, W) or (H, W)
        Stack of images (or a single image).
    shifts_rc : torch.Tensor, shape (n, 2) or (2,)
        Per-image shifts as (row_shift, col_shift) in pixels.
    mode : 'bilinear' or 'nearest'
    blend : bool, whether to blend the shifted images using a Tukey window
    edge_blend : float, Tukey edge width in pixels used when blending
    padding : int or None, canvas padding. If None, computed from max shift + edge_blend
    pad_val : float or one of 'min','max','mean','median', fill value outside support

    Returns
    -------
    torch.Tensor
        Shifted (and blended) images. If the input was a single image, returns an array
        of shape (Hp, Wp). Otherwise returns (n, Hp, Wp) for blended result or (n, H, W)
        for the non-blended case.
    """
    single = images.dim() == 2
    if single:
        images = images.unsqueeze(0)
        shifts_rc = shifts_rc.unsqueeze(0)

    n, H, W = images.shape

    shifts_rc = shifts_rc.to(dtype=torch.float32, device=images.device)

    if not blend:
        # simple shift per-image without padding/blending, keep original behavior
        imgs = images.float().unsqueeze(1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=images.device),
            torch.linspace(-1, 1, W, device=images.device),
            indexing="ij",
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        grid = base_grid.unsqueeze(0).expand(n, -1, -1, -1).clone()  # (n, H, W, 2)
        grid[..., 0] -= 2.0 * shifts_rc[:, 1].view(n, 1, 1) / W  # col shift → x
        grid[..., 1] -= 2.0 * shifts_rc[:, 0].view(n, 1, 1) / H  # row shift → y

        shifted = F.grid_sample(imgs, grid, mode=mode, padding_mode="zeros", align_corners=True)
        result = shifted[:, 0]  # (n, H, W)
        return result[0] if single else result

    # --- blending path ---
    # determine pad_val numeric
    if isinstance(pad_val, str):
        s = pad_val.strip().lower()
        v = images.reshape(-1)
        if s == "min":
            pad_val_f = float(torch.min(v).item())
        elif s == "max":
            pad_val_f = float(torch.max(v).item())
        elif s == "mean":
            pad_val_f = float(torch.mean(v).item())
        elif s == "median":
            pad_val_f = float(torch.median(v).item())
        else:
            raise ValueError("pad_val must be a float or one of {'min','max','mean','median'}")
    else:
        pad_val_f = float(pad_val)

    # padding (compute from max shift if not provided)
    max_shift = float(torch.max(torch.abs(shifts_rc)).item()) if shifts_rc.numel() else 0.0
    if padding is None:
        padding = int(np.ceil(max_shift + float(edge_blend))) + 2
    padding = int(padding)

    alpha_r = min(1.0, 2.0 * float(edge_blend) / float(H)) if edge_blend > 0 else 0.0
    alpha_c = min(1.0, 2.0 * float(edge_blend) / float(W)) if edge_blend > 0 else 0.0

    w = (
        tukey_torch(H, alpha=alpha_r, device=images.device, dtype=torch.float32)[:, None]
        * tukey_torch(W, alpha=alpha_c, device=images.device, dtype=torch.float32)[None, :]
    )

    Hp = H + 2 * padding
    Wp = W + 2 * padding
    r0 = padding
    c0 = padding

    # build padded stacks
    stack = torch.zeros((n, Hp, Wp), dtype=torch.float32, device=images.device)
    stack_w = torch.zeros_like(stack)
    for ind in range(n):
        stack[ind, r0 : r0 + H, c0 : c0 + W] = images[ind].to(dtype=torch.float32) * w
        stack_w[ind, r0 : r0 + H, c0 : c0 + W] = w
    # shift both stack and stack_w using grid_sample on (n,1,Hp,Wp)
    imgs = stack.unsqueeze(1)
    imgs_w = stack_w.unsqueeze(1)

    # Build base normalized grid for Hp, Wp
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, Hp, device=images.device),
        torch.linspace(-1, 1, Wp, device=images.device),
        indexing="ij",
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (Hp, Wp, 2)
    grid = base_grid.unsqueeze(0).expand(n, -1, -1, -1).clone()  # (n, Hp, Wp, 2)
    grid[..., 0] -= 2.0 * shifts_rc[:, 1].view(n, 1, 1) / Wp  # col shift → x
    grid[..., 1] -= 2.0 * shifts_rc[:, 0].view(n, 1, 1) / Hp  # row shift → y

    shifted = F.grid_sample(imgs, grid, mode=mode, padding_mode="zeros", align_corners=True)
    shifted_w = F.grid_sample(imgs_w, grid, mode=mode, padding_mode="zeros", align_corners=True)

    shifted = shifted[:, 0]
    shifted_w = shifted_w[:, 0]

    shifted_w = torch.clamp(shifted_w, 0.0, 1.0)

    edge_w = torch.clamp(1.0 - torch.sum(shifted_w, dim=0), 0.0, 1.0)

    num = torch.sum(shifted, dim=0) + edge_w * pad_val_f
    den = torch.sum(shifted_w, dim=0) + edge_w

    out = torch.empty_like(num)
    mask = den != 0.0
    out[mask] = num[mask] / den[mask]
    out[~mask] = 0.0

    return out


def cross_correlation_shift_torch(
    im_ref: torch.Tensor,
    im: torch.Tensor,
    upsample_factor: int = 2,
    fft_input: bool = False,
) -> torch.Tensor:
    """
    Align two real images using Fourier cross-correlation and DFT upsampling.

    Supports a single image pair with shape (H, W) or a batch of image pairs with
    shape (N, H, W). When batched, returns a tensor of shape (N, 2).
    """
    if im_ref.shape != im.shape:
        raise ValueError("im_ref and im must have the same shape")

    if im_ref.ndim == 2:
        if fft_input:
            G1 = im_ref
            G2 = im
        else:
            G1 = torch.fft.fft2(im_ref)
            G2 = torch.fft.fft2(im)

        xy_shift = align_images_fourier_torch(G1, G2, upsample_factor)
        M, N = im_ref.shape
        dx = ((xy_shift[0] + M / 2) % M) - M / 2
        dy = ((xy_shift[1] + N / 2) % N) - N / 2
        return torch.tensor([dx, dy], device=G1.device)

    if im_ref.ndim == 3:
        if fft_input:
            G1 = im_ref
            G2 = im
        else:
            G1 = torch.fft.fft2(im_ref, dim=(-2, -1))
            G2 = torch.fft.fft2(im, dim=(-2, -1))

        xy_shift = align_images_fourier_torch_batched(G1, G2, upsample_factor)
        M, N = im_ref.shape[-2:]
        dx = ((xy_shift[..., 0] + M / 2) % M) - M / 2
        dy = ((xy_shift[..., 1] + N / 2) % N) - N / 2
        return torch.stack([dx, dy], dim=-1)

    raise ValueError("im_ref and im must be 2D or 3D tensors")


def align_images_fourier_torch(
    G1: torch.Tensor,
    G2: torch.Tensor,
    upsample_factor: int,
) -> torch.Tensor:
    """
    Alignment using DFT upsampling of cross correlation.
    G1, G2: torch tensors representing FTs of images (complex)
    Returns: xy_shift (tensor length 2)
    """
    device = G1.device
    cc = G1 * G2.conj()
    cc_real = torch.fft.ifft2(cc).real

    flat_idx = torch.argmax(cc_real)
    x0 = (flat_idx // cc_real.shape[1]).to(torch.long).item()
    y0 = (flat_idx % cc_real.shape[1]).to(torch.long).item()

    M, N = cc_real.shape
    x_inds = [((x0 + dx) % M) for dx in (-1, 0, 1)]
    y_inds = [((y0 + dy) % N) for dy in (-1, 0, 1)]

    vx = cc_real[x_inds, y0]
    vy = cc_real[x0, y_inds]

    denom_x = 4.0 * vx[1] - 2.0 * vx[2] - 2.0 * vx[0]
    denom_y = 4.0 * vy[1] - 2.0 * vy[2] - 2.0 * vy[0]
    dx = (vx[2] - vx[0]) / denom_x if denom_x != 0 else torch.tensor(0.0, device=device)
    dy = (vy[2] - vy[0]) / denom_y if denom_y != 0 else torch.tensor(0.0, device=device)

    x0 = torch.round((x0 + dx) * 2.0) / 2.0
    y0 = torch.round((y0 + dy) * 2.0) / 2.0

    xy_shift = torch.tensor([x0, y0], device=device)

    if upsample_factor > 2:
        xy_shift = upsampled_correlation_torch(cc, upsample_factor, xy_shift)

    return xy_shift


def align_images_fourier_torch_batched(
    G1: torch.Tensor,
    G2: torch.Tensor,
    upsample_factor: int,
) -> torch.Tensor:
    """
    Batched version of align_images_fourier_torch.

    G1 and G2 must have shape (N, H, W), where N is the batch size.
    Returns a tensor of shape (N, 2) with unwrapped peak locations.
    """
    if G1.shape != G2.shape:
        raise ValueError("G1 and G2 must have the same shape")
    if G1.ndim != 3:
        raise ValueError("G1 and G2 must have shape (N, H, W)")

    device = G1.device
    cc = G1 * G2.conj()
    cc_real = torch.fft.ifft2(cc, dim=(-2, -1)).real

    batch, M, N = cc_real.shape
    flat_idx = torch.argmax(cc_real.reshape(batch, -1), dim=1)
    x0 = flat_idx // N
    y0 = flat_idx % N

    offsets = torch.tensor([-1, 0, 1], device=device, dtype=torch.long)
    x_inds = (x0[:, None] + offsets[None, :]) % M
    y_inds = (y0[:, None] + offsets[None, :]) % N

    batch_inds = torch.arange(batch, device=device)[:, None]
    vx = cc_real[batch_inds, x_inds, y0[:, None].expand(-1, 3)]
    vy = cc_real[batch_inds, x0[:, None].expand(-1, 3), y_inds]

    denom_x = 4.0 * vx[:, 1] - 2.0 * vx[:, 2] - 2.0 * vx[:, 0]
    denom_y = 4.0 * vy[:, 1] - 2.0 * vy[:, 2] - 2.0 * vy[:, 0]
    dx = torch.where(denom_x != 0, (vx[:, 2] - vx[:, 0]) / denom_x, torch.zeros_like(denom_x))
    dy = torch.where(denom_y != 0, (vy[:, 2] - vy[:, 0]) / denom_y, torch.zeros_like(denom_y))

    x0 = torch.round((x0.to(cc_real.dtype) + dx) * 2.0) / 2.0
    y0 = torch.round((y0.to(cc_real.dtype) + dy) * 2.0) / 2.0
    xy_shift = torch.stack([x0, y0], dim=-1)

    if upsample_factor > 2:
        xy_shift = upsampled_correlation_torch(cc, upsample_factor, xy_shift)

    return xy_shift


def upsampled_correlation_torch(
    imageCorr: torch.Tensor,
    upsampleFactor: int,
    xyShift: torch.Tensor,
) -> torch.Tensor:
    """
    Refine the correlation peak of imageCorr around xyShift by DFT upsampling.

    Supports a single correlation image or a batch of them.
    """
    assert upsampleFactor > 2

    squeeze_output = imageCorr.ndim == 2
    if squeeze_output:
        imageCorr = imageCorr.unsqueeze(0)
    if xyShift.ndim == 1:
        xyShift = xyShift.unsqueeze(0)

    if imageCorr.ndim != 3 or xyShift.ndim != 2:
        raise ValueError("imageCorr must have shape (H, W) or (N, H, W), and xyShift must match")
    if imageCorr.shape[0] != xyShift.shape[0]:
        raise ValueError("imageCorr and xyShift batch dimensions must match")

    xyShift = torch.round(xyShift * float(upsampleFactor)) / float(upsampleFactor)
    globalShift = float(math.floor(math.ceil(upsampleFactor * 1.5) / 2.0))
    upsampleCenter = globalShift - (upsampleFactor * xyShift)

    conj_input = imageCorr.conj()
    im_up = dftUpsample_torch(conj_input, upsampleFactor, upsampleCenter)
    imageCorrUpsample = im_up.conj()

    batch, _, out_w = imageCorrUpsample.real.shape
    flat_idx = torch.argmax(imageCorrUpsample.real.reshape(batch, -1), dim=1)
    r = flat_idx // out_w
    c = flat_idx % out_w

    padded = F.pad(imageCorrUpsample.real, (1, 1, 1, 1), mode="circular")
    batch_inds = torch.arange(batch, device=imageCorr.device)

    center = padded[batch_inds, r + 1, c + 1]
    top = padded[batch_inds, r, c + 1]
    bottom = padded[batch_inds, r + 2, c + 1]
    left = padded[batch_inds, r + 1, c]
    right = padded[batch_inds, r + 1, c + 2]

    denom_x = 4.0 * center - 2.0 * bottom - 2.0 * top
    denom_y = 4.0 * center - 2.0 * right - 2.0 * left
    dx = torch.where(denom_x != 0, (bottom - top) / denom_x, torch.zeros_like(denom_x))
    dy = torch.where(denom_y != 0, (right - left) / denom_y, torch.zeros_like(denom_y))

    xySubShift = torch.stack([r, c], dim=-1).to(dtype=xyShift.dtype) - globalShift
    xyShift = xyShift + (xySubShift + torch.stack([dx, dy], dim=-1)) / float(upsampleFactor)

    return xyShift[0] if squeeze_output else xyShift


def dftUpsample_torch(
    imageCorr: torch.Tensor,
    upsampleFactor: int,
    xyShift: torch.Tensor,
) -> torch.Tensor:
    """
    Matrix-multiply DFT upsampling for a single correlation image or a batch.
    """
    squeeze_output = imageCorr.ndim == 2
    if squeeze_output:
        imageCorr = imageCorr.unsqueeze(0)
    if xyShift.ndim == 1:
        xyShift = xyShift.unsqueeze(0)

    if imageCorr.ndim != 3 or xyShift.ndim != 2:
        raise ValueError("imageCorr must have shape (M, N) or (B, M, N), and xyShift must match")
    if imageCorr.shape[0] != xyShift.shape[0]:
        raise ValueError("imageCorr and xyShift batch dimensions must match")

    device = imageCorr.device
    _, M, N = imageCorr.shape
    pixelRadius = 1.5
    numRow = int(math.ceil(pixelRadius * upsampleFactor))
    numCol = numRow

    col_freq = torch.fft.ifftshift(torch.arange(N, device=device)) - math.floor(N / 2)
    row_freq = torch.fft.ifftshift(torch.arange(M, device=device)) - math.floor(M / 2)

    col_coords = (
        torch.arange(numCol, device=device, dtype=torch.get_default_dtype())[None, :]
        - (xyShift[:, 1:2])
    )
    row_coords = (
        torch.arange(numRow, device=device, dtype=torch.get_default_dtype())[None, :]
        - (xyShift[:, 0:1])
    )

    factor_col = -2j * math.pi / (N * float(upsampleFactor))
    colKern = torch.exp(factor_col * (col_freq[None, :, None] * col_coords[:, None, :])).to(
        imageCorr.dtype
    )

    factor_row = -2j * math.pi / (M * float(upsampleFactor))
    rowKern = torch.exp(factor_row * (row_coords[:, :, None] * row_freq[None, None, :])).to(
        imageCorr.dtype
    )

    # one small matmul per item: a batched complex matmul picks a different cuBLAS
    # kernel whose rounding differs from the single-image path by 1e-7 relative,
    # which an 80-cycle alignment amplifies to 3e-3 px; per item both paths match
    imageUpsample = torch.cat(
        [
            torch.matmul(torch.matmul(rowKern[i : i + 1], imageCorr[i : i + 1]), colKern[i : i + 1])
            for i in range(imageCorr.shape[0])
        ]
    )

    result = imageUpsample.real
    return result[0] if squeeze_output else result


def fit_surface_lstsq(img, mode="linear"):
    """
    Fits an image with a linear or quadratic function

    Parameters
    ----------
    img : torch.Tensor
        Image to fit, of shape (H, W)
    mode : str
        Fitting mode, either "linear" or "quadratic"

    Returns
    ------
    fitted : torch.Tensor
        Array of shape (H, W) of the fit function over the image
    coeffs : torch.Tensor
        fitting coefficients
    """
    H, W = img.shape
    x_1d = torch.arange(img.shape[1], device=img.device, dtype=torch.float32)
    y_1d = torch.arange(img.shape[0], device=img.device, dtype=torch.float32)

    xx, yy = torch.meshgrid(x_1d, y_1d)

    x = xx.flatten()
    y = yy.flatten()
    z = img.flatten()

    if mode == "linear":
        A = torch.stack([x, y, torch.ones_like(x)], dim=1)
    elif mode == "quadratic":
        A = torch.stack([x**2, y**2, x * y, x, y, torch.ones_like(x)], dim=1)

    coeffs, _, _, _ = torch.linalg.lstsq(A, z.unsqueeze(1))

    fitted = (A @ coeffs).reshape(H, W)
    return fitted, coeffs


def dscan_correct(
    dataset,
    iterations,
    upsample_factor: int = 100,
    plot: bool = True,
    edge_blend: float = 2.0,
    device="cpu",
    method="autocorrelation",
    fit_shifts=True,
    mode="linear",
    batch_size: int | None = None,
):
    """
    Align diffraction patterns using autocorrelation.

    Parameters
    ----------
    dataset : torch.Tensor
        Input 4D dataset
    iterations : int
        Number of refinement iterations
    upsample_factor : int
        Upsampling factor for sub-pixel accuracy
    plot : bool
        Whether to plot results after each iteration
    edge_blend : float
        Edge blending parameter for Tukey window
    device : torch.device
        Device to use
    fit_shifts : bool
        Whether to fit shifts to a smooth surface
    mode : str
        "linear" or "quadratic" for surface fitting

    Returns
    -------
    tuple
        A tuple ``(diffraction_shifts, shifted_dps, G_ref_final)`` where
        ``diffraction_shifts`` is a ``torch.Tensor`` of shape (H_rs, W_rs, 2) with
        per-scan-position shifts, ``shifted_dps`` is the aligned dataset (same shape
        as ``dataset``), and ``G_ref_final`` is the final complex Fourier-domain
        reference (torch.Tensor).
    """
    H_rs, W_rs, H_dp, W_dp = dataset.shape
    n_pos = H_rs * W_rs
    if batch_size is None:
        batch_size = max(1, min(n_pos, 256))

    w = (
        tukey_torch(
            H_dp,
            alpha=2.0 * float(edge_blend) / float(H_dp),
            device=device,
            dtype=torch.float32,
        )[:, None]
        * tukey_torch(
            W_dp,
            alpha=2.0 * float(edge_blend) / float(W_dp),
            device=device,
            dtype=torch.float32,
        )[None, :]
    )

    diffraction_shifts = torch.zeros((H_rs, W_rs, 2), device=device, dtype=torch.float32)
    shifted_dps = dataset.clone()

    kr = torch.fft.fftfreq(H_dp, device=device)[:, None]
    kc = torch.fft.fftfreq(W_dp, device=device)[None, :]

    for iteration in range(iterations):
        G_ref = torch.fft.fft2(shifted_dps.mean(dim=(0, 1)) * w)

        if method == "cross_correlation":
            for h_rs in tqdm(range(H_rs), desc=f"Iteration {iteration + 1}/{iterations}"):
                for w_rs in range(W_rs):
                    ind = w_rs + h_rs * H_rs
                    dp = shifted_dps[h_rs, w_rs]  # <-- Read from current shifted_dps, not original
                    G = torch.fft.fft2(w * dp)
                    shift = cross_correlation_shift_torch(
                        G_ref, G, upsample_factor=upsample_factor, fft_input=True
                    )
                    diffraction_shifts[h_rs, w_rs] = shift

                    phase_ramp = torch.exp(-1j * torch.pi * (kr * shift[0] + kc * shift[1]))
                    G_shift = G * phase_ramp

                    shifted_dps[h_rs, w_rs, :, :] = torch.fft.ifft2(G_shift).real
                    G_ref = G_ref * (ind / (ind + 1)) + G_shift / (ind + 1)

        if method == "autocorrelation":
            shifts_flat = torch.zeros((n_pos, 2), device=device, dtype=torch.float32)
            shifted_dps_flat = shifted_dps.reshape(n_pos, H_dp, W_dp)

            for batch_start in tqdm(
                range(0, n_pos, batch_size),
                desc=f"Iteration {iteration + 1}/{iterations} (autocorrelation)",
            ):
                batch_end = min(batch_start + batch_size, n_pos)
                dp_b = w * shifted_dps_flat[batch_start:batch_end]
                G_b = torch.fft.fft2(dp_b, dim=(-2, -1))
                G_flipped = torch.conj(G_b)
                shifts_flat[batch_start:batch_end] = (
                    -cross_correlation_shift_torch(
                        G_b,
                        G_flipped,
                        upsample_factor=upsample_factor,
                        fft_input=True,
                    )
                    / 2.0
                )
                del dp_b, G_b, G_flipped
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            diffraction_shifts[:, :, :] = shifts_flat.reshape(H_rs, W_rs, 2)

        if method == "direct_fitting":
            centers = torch.zeros((n_pos, 2), device=device, dtype=torch.float32)
            shifted_dps_flat = shifted_dps.reshape(n_pos, H_dp, W_dp)

            for batch_start in tqdm(
                range(0, n_pos, batch_size),
                desc=f"Iteration {iteration + 1}/{iterations} (direct fitting)",
            ):
                batch_end = min(batch_start + batch_size, n_pos)
                dp_b = shifted_dps_flat[batch_start:batch_end].float()
                B = dp_b.shape[0]
                batch_idx = torch.arange(B, device=device)

                # argmax: integer center estimate
                flat_idx = torch.argmax(dp_b.reshape(B, -1), dim=1)
                row_peak = flat_idx // W_dp
                col_peak = flat_idx % W_dp

                # log-parabolic sub-pixel refinement, row direction
                row_safe = row_peak.clamp(1, H_dp - 2)
                vr_m = dp_b[batch_idx, row_safe - 1, col_peak].clamp(min=1e-6).log()
                vr_0 = dp_b[batch_idx, row_safe, col_peak].clamp(min=1e-6).log()
                vr_p = dp_b[batch_idx, row_safe + 1, col_peak].clamp(min=1e-6).log()
                denom_r = vr_m + vr_p - 2.0 * vr_0
                dr = torch.where(
                    (denom_r < -1e-6) & (row_peak > 0) & (row_peak < H_dp - 1),
                    ((vr_m - vr_p) / (2.0 * denom_r)).clamp(-1.0, 1.0),
                    torch.zeros(B, device=device),
                )

                # log-parabolic sub-pixel refinement, col direction
                col_safe = col_peak.clamp(1, W_dp - 2)
                vc_m = dp_b[batch_idx, row_peak, col_safe - 1].clamp(min=1e-6).log()
                vc_0 = dp_b[batch_idx, row_peak, col_safe].clamp(min=1e-6).log()
                vc_p = dp_b[batch_idx, row_peak, col_safe + 1].clamp(min=1e-6).log()
                denom_c = vc_m + vc_p - 2.0 * vc_0
                dc = torch.where(
                    (denom_c < -1e-6) & (col_peak > 0) & (col_peak < W_dp - 1),
                    ((vc_m - vc_p) / (2.0 * denom_c)).clamp(-1.0, 1.0),
                    torch.zeros(B, device=device),
                )

                centers[batch_start:batch_end, 0] = row_peak.float() + dr
                centers[batch_start:batch_end, 1] = col_peak.float() + dc

                del dp_b, flat_idx, row_peak, col_peak, batch_idx
                del vr_m, vr_0, vr_p, vc_m, vc_0, vc_p, dr, dc
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # fit a plane to centers across the real-space scan grid
            centers_2d = centers.reshape(H_rs, W_rs, 2)
            centers_fit_r, _ = fit_surface_lstsq(centers_2d[:, :, 0], mode="linear")
            centers_fit_c, _ = fit_surface_lstsq(centers_2d[:, :, 1], mode="linear")

            # shifts = mean_center - fitted_center: moves each DP toward the global mean
            diffraction_shifts[:, :, 0] = H_dp / 2 - centers_fit_r
            diffraction_shifts[:, :, 1] = W_dp / 2 - centers_fit_c

        if fit_shifts:
            diffraction_shifts_1, _ = fit_surface_lstsq(diffraction_shifts[:, :, 0], mode=mode)
            diffraction_shifts_2, _ = fit_surface_lstsq(diffraction_shifts[:, :, 1], mode=mode)
            diffraction_shifts_old = diffraction_shifts.clone()
            diffraction_shifts = torch.stack((diffraction_shifts_1, diffraction_shifts_2), dim=2)

            # Apply fitted shifts in batches over all scan positions.
            shifted_dps_flat = shifted_dps.reshape(n_pos, H_dp, W_dp)
            shifts_flat = diffraction_shifts.reshape(n_pos, 2)

            for batch_start in range(0, n_pos, batch_size):
                batch_end = min(batch_start + batch_size, n_pos)
                G_b = torch.fft.fft2(w * shifted_dps_flat[batch_start:batch_end], dim=(-2, -1))
                s_b = shifts_flat[batch_start:batch_end]
                phase_ramp = torch.exp(
                    -1j
                    * torch.pi
                    * (
                        kr.unsqueeze(0) * s_b[:, 0][:, None, None]
                        + kc.unsqueeze(0) * s_b[:, 1][:, None, None]
                    )
                )
                shifted_dps_flat[batch_start:batch_end] = torch.fft.ifft2(
                    G_b * phase_ramp, dim=(-2, -1)
                ).real
                del G_b, s_b, phase_ramp
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if plot:
            from .plot import scan_correction

            scan_correction(
                dataset, shifted_dps, diffraction_shifts,
                diffraction_shifts_old if fit_shifts else None,
            )

    return diffraction_shifts, shifted_dps
