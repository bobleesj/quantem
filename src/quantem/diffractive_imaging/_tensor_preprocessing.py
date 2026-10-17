"""Torch preprocessing for raster ptychography detector measurements."""

import math

import torch


def _fit_com(measured_t, mask_t, mode, detector_shape):
    """Fit the same constant, plane or parabola used by the array path."""
    if mode == "none":
        return measured_t.clone()
    if mode == "no_shift":
        center_t = measured_t.new_tensor(detector_shape) / 2
        return center_t[:, None, None].expand_as(measured_t).clone()
    valid_t = mask_t & torch.isfinite(measured_t).all(0)
    if not valid_t.any():
        raise ValueError("No finite selected diffraction centers; select nonzero patterns.")
    if mode == "constant":
        selected_t = measured_t[:, valid_t]
        mean_t = _numpy_pairwise_sum(selected_t) / selected_t.new_tensor(selected_t.shape[-1])
        return mean_t[:, None, None].expand_as(measured_t).clone()
    if mode not in ("plane", "parabola"):
        raise ValueError(
            f"Unknown CoM fit {mode!r}; use constant, plane, parabola, none or no_shift."
        )
    row_t, col_t = torch.meshgrid(
        torch.arange(measured_t.shape[1], device=measured_t.device, dtype=torch.float64),
        torch.arange(measured_t.shape[2], device=measured_t.device, dtype=torch.float64),
        indexing="ij",
    )
    columns = [torch.ones_like(row_t), row_t, col_t]
    if mode == "parabola":
        columns.extend([row_t.square(), col_t.square(), row_t * col_t])
    design_t = torch.stack(columns, dim=-1)
    coefficients_t = torch.linalg.lstsq(
        design_t[valid_t], measured_t[:, valid_t].T.double()
    ).solution
    return (design_t @ coefficients_t).permute(2, 0, 1).to(measured_t.dtype)


def _numpy_pairwise_sum(values_t, dim=-1):
    """Reduce with NumPy's contiguous pairwise accumulation order on device."""
    values_t = values_t.movedim(dim, -1)
    # NumPy uses eight accumulators below its 128-element pairwise cutoff.
    length = values_t.shape[-1]
    if length >= 128 and length % 128 == 0 and (length // 128).bit_count() == 1:
        blocks_t = values_t.reshape(*values_t.shape[:-1], length // 128, 16, 8)
        lanes_t = blocks_t[..., 0, :].clone()
        for index in range(1, 16):
            lanes_t = lanes_t + blocks_t[..., index, :]
        result_t = (
            (lanes_t[..., 0] + lanes_t[..., 1])
            + (lanes_t[..., 2] + lanes_t[..., 3])
        ) + (
            (lanes_t[..., 4] + lanes_t[..., 5])
            + (lanes_t[..., 6] + lanes_t[..., 7])
        )
        while result_t.shape[-1] > 1:
            result_t = result_t[..., 0::2] + result_t[..., 1::2]
        return result_t[..., 0]
    if length < 8:
        result_t = torch.zeros_like(values_t[..., 0])
        for index in range(length):
            result_t = result_t + values_t[..., index]
        return result_t
    if length <= 128:
        lanes_t = values_t[..., :8].clone()
        stop = length - length % 8
        for index in range(8, stop, 8):
            lanes_t = lanes_t + values_t[..., index : index + 8]
        result_t = (
            (lanes_t[..., 0] + lanes_t[..., 1])
            + (lanes_t[..., 2] + lanes_t[..., 3])
        ) + (
            (lanes_t[..., 4] + lanes_t[..., 5])
            + (lanes_t[..., 6] + lanes_t[..., 7])
        )
        for index in range(stop, length):
            result_t = result_t + values_t[..., index]
        return result_t
    midpoint = (length // 2) // 8 * 8
    return _numpy_pairwise_sum(values_t[..., :midpoint]) + _numpy_pairwise_sum(values_t[..., midpoint:])


def _measure_com(intensities_t, dtype):
    """Match NumPy detector moments, retaining doubles until the origin is fitted."""

    scan_shape = intensities_t.shape[:2]
    detector_shape = intensities_t.shape[-2:]
    frames_t = intensities_t.reshape(-1, *detector_shape)
    centers_t = torch.empty((len(frames_t), 2), device=frames_t.device, dtype=torch.float64)
    row_t = torch.arange(detector_shape[0], device=frames_t.device, dtype=torch.float64)
    col_t = torch.arange(detector_shape[1], device=frames_t.device, dtype=torch.float64)
    for first in range(0, len(frames_t), 128):
        block_t = frames_t[first : first + 128].to(dtype)
        totals_t = _numpy_pairwise_sum(block_t.flatten(1))
        row_moment_t = _numpy_pairwise_sum((block_t.double() * row_t[:, None]).flatten(1))
        col_moment_t = _numpy_pairwise_sum((block_t.double() * col_t[None, :]).flatten(1))
        result_t = torch.stack((row_moment_t, col_moment_t), dim=-1) / totals_t[:, None]
        result_t[totals_t == 0] = torch.nan
        centers_t[first : first + len(block_t)] = result_t
    return centers_t.T.reshape(2, *scan_shape)


def _rotate_com(com_t, angles_t, transpose=False):
    """Rotate scan-row/scan-column vector fields at one or more angles."""
    row_t, col_t = com_t.flip(0) if transpose else com_t
    cosine_t = angles_t.cos()[:, None, None]
    sine_t = angles_t.sin()[:, None, None]
    return torch.stack(
        (cosine_t * row_t - sine_t * col_t, sine_t * row_t + cosine_t * col_t), dim=1
    )


def _select_rotation(com_t, mask_t, forced_angle, forced_transpose):
    """Minimize the mean absolute CoM curl with the array path's tie policy."""
    angles_t = (
        torch.arange(-89, 90, device=com_t.device, dtype=torch.float64)
        if forced_angle is None
        else com_t.new_tensor([forced_angle]).double()
    )
    masked_t = com_t.double().masked_fill(~mask_t[None], torch.nan)
    curls = []
    for transpose in (False, True):
        rotated_t = _rotate_com(masked_t, angles_t.deg2rad(), transpose)
        curl_t = (
            rotated_t[:, 1, 2:, 1:-1]
            - rotated_t[:, 1, :-2, 1:-1]
            - rotated_t[:, 0, 1:-1, 2:]
            + rotated_t[:, 0, 1:-1, :-2]
        )
        curls.append(curl_t.abs().nanmean((-2, -1)))
    if forced_transpose is None:
        transpose = bool(curls[1].min() < curls[0].min())
    else:
        transpose = forced_transpose
    index = int(curls[int(transpose)].argmin())
    angle = float(angles_t[index])
    return angle, transpose, angles_t, curls


def _center_amplitudes(intensities_t, com_fit_t, mask_t, bilinear, dtype):
    """Center amplitudes with periodic interpolation, clipping after the shift."""
    detector_shape = intensities_t.shape[-2:]
    frames_t = intensities_t.reshape(-1, *detector_shape)
    indices_t = mask_t.flatten().nonzero().flatten()
    shifts_t = -com_fit_t.reshape(2, -1).T[indices_t]
    centered_t = torch.empty(
        (len(indices_t), *detector_shape), device=frames_t.device, dtype=dtype
    )
    frame_totals_t = torch.empty((len(indices_t), 2), device=frames_t.device, dtype=dtype)
    freq_row_t = torch.fft.fftfreq(detector_shape[0], device=frames_t.device, dtype=torch.float64)
    freq_col_t = torch.fft.fftfreq(detector_shape[1], device=frames_t.device, dtype=torch.float64)
    grid_row_t = torch.arange(detector_shape[0], device=frames_t.device)[None, :, None]
    grid_col_t = torch.arange(detector_shape[1], device=frames_t.device)[None, None, :]
    phase_cache_t = None
    if len(indices_t) and not bilinear and torch.equal(
        shifts_t.view(torch.uint8), shifts_t[:1].expand_as(shifts_t).view(torch.uint8)
    ):
        from ._reference_phase import _phase_factor

        first_shift_t = shifts_t[0].double()
        phase_cache_t = _phase_factor(
            -2
            * math.pi
            * (first_shift_t[0] * freq_row_t[:, None] + first_shift_t[1] * freq_col_t[None, :])
        )
    # Bound the larger batch to the measured detector and available memory.
    chunk_size = 128
    if (frames_t.is_cuda and dtype == torch.float32
            and detector_shape == (128, 128) and phase_cache_t is not None):
        free_bytes, _ = torch.cuda.mem_get_info(frames_t.device)
        if free_bytes >= 4 * 1024**3:
            chunk_size = 512
    for first in range(0, len(indices_t), chunk_size):
        selection_t = indices_t[first : first + chunk_size]
        intensity_t = frames_t[selection_t].to(dtype).clamp_min(0)
        amplitude_t = intensity_t.sqrt()
        frame_totals_t[first : first + len(selection_t), 0] = _numpy_pairwise_sum(
            intensity_t.flatten(1)
        )
        frame_totals_t[first : first + len(selection_t), 1] = _numpy_pairwise_sum(
            amplitude_t.flatten(1)
        )
        shift_t = shifts_t[first : first + len(selection_t)].double()
        if bilinear:
            lower_t = shift_t.floor().long()
            weight_t = shift_t - lower_t
            shifted_t = torch.zeros_like(amplitude_t)
            batch_t = torch.arange(len(selection_t), device=frames_t.device)[:, None, None]
            for row_offset in (0, 1):
                for col_offset in (0, 1):
                    rows_t = (
                        grid_row_t - lower_t[:, 0, None, None] - row_offset
                    ) % detector_shape[0]
                    cols_t = (
                        grid_col_t - lower_t[:, 1, None, None] - col_offset
                    ) % detector_shape[1]
                    row_weight_t = weight_t[:, 0] if row_offset else 1 - weight_t[:, 0]
                    col_weight_t = weight_t[:, 1] if col_offset else 1 - weight_t[:, 1]
                    shifted_t += (
                        amplitude_t[batch_t, rows_t, cols_t]
                        * (row_weight_t * col_weight_t)[:, None, None]
                    )
        else:
            phase_t = shift_t[:, 0, None, None] * freq_row_t[None, :, None]
            phase_t = phase_t + shift_t[:, 1, None, None] * freq_col_t[None, None, :]
            # NumPy rounds each forward FFT axis to the input's complex dtype.
            # Retain double arithmetic within an axis, then reproduce that boundary.
            complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
            if detector_shape == (128, 128):
                from ._reference_fft import _fft128

                spectrum_t = _fft128(_fft128(amplitude_t, dim=-1), dim=-2)
            else:
                spectrum_t = torch.fft.fft(amplitude_t.double(), dim=-1).to(complex_dtype)
                spectrum_t = torch.fft.fft(spectrum_t.cdouble(), dim=-2).to(complex_dtype)
            spectrum_t = spectrum_t.cdouble()
            from ._reference_phase import _phase_factor

            phase_factor_t = (
                _phase_factor(-2 * math.pi * phase_t) if phase_cache_t is None else phase_cache_t
            )
            product_t = torch.complex(
                torch.addcmul(
                    -spectrum_t.imag * phase_factor_t.imag,
                    spectrum_t.real,
                    phase_factor_t.real,
                ),
                torch.addcmul(
                    spectrum_t.imag * phase_factor_t.real,
                    spectrum_t.real,
                    phase_factor_t.imag,
                ),
            )
            if detector_shape == (128, 128):
                shifted_t = _fft128(
                    _fft128(product_t, dim=-1, inverse=True), dim=-2, inverse=True
                ).real
            else:
                shifted_t = torch.fft.ifft(torch.fft.ifft(product_t, dim=-1), dim=-2).real
        centered_t[first : first + len(selection_t)] = torch.fft.fftshift(
            shifted_t.clamp_min(0), dim=(-2, -1)
        )
    # The reference accumulates frame totals sequentially in the configured dtype.
    if frame_totals_t.is_cuda and dtype == torch.float32 and len(frame_totals_t):
        totals_t = frame_totals_t.cumsum(0)[-1]
    else:
        totals_t = frame_totals_t.new_zeros(2)
        for frame_total_t in frame_totals_t.unbind(0):
            totals_t.add_(frame_total_t)
    return centered_t, totals_t / totals_t.new_tensor(len(indices_t))


def _scan_positions(model, padding):
    """Construct raster positions on the measurement device in (row, col) order."""
    dtype = model.scan_positions_px.dtype
    row_t, col_t = torch.meshgrid(
        torch.arange(int(model.gpts[0]), device=model.device, dtype=torch.float64)
        * float(model.scan_sampling[0]),
        torch.arange(int(model.gpts[1]), device=model.device, dtype=torch.float64)
        * float(model.scan_sampling[1]),
        indexing="ij",
    )
    # NumPy multiplies its integer raster by the calibration in float64 before
    # converting positions to the reconstruction dtype.
    positions_t = torch.stack((row_t.flatten(), col_t.flatten()), dim=-1)[
        model.positions_mask
    ].to(dtype)
    angle = model.com_rotation_rad
    if angle:
        rotation_t = positions_t.new_tensor(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        # NumPy's strided axis-0 mean accumulates each coordinate sequentially.
        # The device outer-dimension scan follows that order, whereas Torch's
        # CPU cumsum widens its accumulator and would change subpixel positions.
        if positions_t.device.type == "cpu":
            total_t = positions_t.new_zeros(2)
            for position_t in positions_t.unbind(0):
                total_t.add_(position_t)
        else:
            total_t = positions_t.cumsum(0)[-1]
        origin_t = total_t / positions_t.new_tensor(len(positions_t))
        translated_t = positions_t - origin_t
        # The reference affine transform promotes its translated result to float64.
        # Keep that precision through anchoring, calibration and padding so values
        # near half pixels select the same integer patches. Translation cancels
        # when anchoring at the minimum, so no origin addition is needed here.
        # Separate products preserve the reference's unfused two-term dot product.
        positions_t = (
            translated_t[:, 0, None] * rotation_t[0]
            + translated_t[:, 1, None] * rotation_t[1]
        ).double()
    sampling_t = positions_t.new_tensor(model.obj_sampling)
    if model.com_transpose:
        positions_t = positions_t.flip(1)
        sampling_t = sampling_t.flip(0)
    positions_t = (positions_t - positions_t.amin(0)) / sampling_t
    return (positions_t + positions_t.new_tensor((0, 0) if padding is None else padding)).to(dtype)


def _object_fov_mask(overlap_t: torch.Tensor, dilation: int, sigma: float) -> torch.Tensor:
    """Build the original closed, dilated, reflect-smoothed illumination support."""
    import torch.nn.functional as functional

    cross_t = overlap_t.new_tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])[None, None]
    mask_t = overlap_t > overlap_t.max() * 0.3
    for _ in range(5):
        mask_t = (
            functional.conv2d(mask_t[None, None].to(overlap_t.dtype), cross_t, padding=1)[0, 0] > 0
        )
    for _ in range(5):
        mask_t = (
            functional.conv2d(mask_t[None, None].to(overlap_t.dtype), cross_t, padding=1)[0, 0]
            == 5
        )
    for _ in range(dilation):
        mask_t = (
            functional.conv2d(mask_t[None, None].to(overlap_t.dtype), cross_t, padding=1)[0, 0] > 0
        )
    values_t = mask_t.to(overlap_t.dtype)
    if sigma <= 0:
        return values_t
    radius = int(4 * sigma + 0.5)
    coordinates_t = torch.arange(-radius, radius + 1, device=overlap_t.device, dtype=torch.float64)
    kernel_t = torch.exp((-0.5 / (sigma * sigma)) * coordinates_t.square())
    kernel_t = kernel_t / kernel_t.sum()
    for axis in (0, 1):
        size = values_t.shape[axis]
        indices_t = torch.arange(-radius, size + radius, device=overlap_t.device) % (2 * size)
        indices_t = torch.minimum(indices_t, 2 * size - 1 - indices_t)
        padded_t = values_t.index_select(axis, indices_t)
        # SciPy accumulates its symmetric filter in double precision, starting
        # at the center and adding reflected pairs from the outside inward.
        # Round to the image dtype between the two separable passes.
        padded_t = padded_t.to(torch.float64)
        filtered_t = padded_t.narrow(axis, radius, size) * kernel_t[radius]
        for offset in range(radius):
            pair_t = padded_t.narrow(axis, offset, size) + padded_t.narrow(
                axis, 2 * radius - offset, size
            )
            filtered_t = filtered_t + pair_t * kernel_t[offset]
        values_t = filtered_t.to(overlap_t.dtype)
    return values_t
