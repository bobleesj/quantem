import json
import math
import re
import time
from pathlib import Path
from typing import Any, Self, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from quantem.core import config
from quantem.core.io.serialize import AutoSerialize
from quantem.core.visualization import show_2d


def _resident_summaries(sources, device: str):
    """Compute MAPED summaries from encoded residents without dense tilts."""
    from quantem.gpu.detector import prepare

    sources = list(sources)
    native_sources = [getattr(source, "data", source) for source in sources]
    if all(
        callable(getattr(source, "mean_dp_device", None))
        and callable(getattr(source, "detector_mean_device", None))
        for source in native_sources
    ):
        mean_dp_torch = []
        detector_mean_torch = []
        for source in native_sources:
            mean_dp = source.mean_dp_device()
            detector_mean = source.detector_mean_device()
            try:
                mean_dp_torch.append(mean_dp.to_torch())
                detector_mean_torch.append(detector_mean.to_torch())
            finally:
                mean_dp.release()
                detector_mean.release()
        return mean_dp_torch, detector_mean_torch
    session = prepare(sources[0] if len(sources) == 1 else sources)
    mean_dp = session.mean_dp(output="native")
    detector_mask = np.ones(session.detector_shape, dtype=bool)
    detector_sum = session.masked_sum(detector_mask, output="native")
    if (
        callable(getattr(mean_dp, "to_torch", None))
        and callable(getattr(detector_sum, "to_torch", None))
    ):
        try:
            mean_dp_torch = mean_dp.to_torch()
            detector_sum_torch = detector_sum.to_torch()
            detector_mean_torch = detector_sum_torch / float(
                math.prod(session.detector_shape)
            )
        finally:
            mean_dp.release()
            detector_sum.release()
        if mean_dp_torch.ndim == 2:
            mean_dp_torch = mean_dp_torch[None]
            detector_mean_torch = detector_mean_torch[None]
        return list(mean_dp_torch.unbind(0)), list(detector_mean_torch.unbind(0))
    detector_mean = (
        detector_sum / float(math.prod(session.detector_shape))
    ).astype(mean_dp.dtype)
    mean_dp_torch = torch.from_dlpack(mean_dp).to(device=device)
    detector_mean_torch = torch.from_dlpack(detector_mean).to(device=device)
    if mean_dp_torch.ndim == 2:
        mean_dp_torch = mean_dp_torch[None]
        detector_mean_torch = detector_mean_torch[None]
    return list(mean_dp_torch.unbind(0)), list(detector_mean_torch.unbind(0))


def _tilt_name(index: int, file: str | Path | None) -> str:
    """
    Plot label of one tilt: ``-17.0x 0.0y`` read from its file name, else ``Tilt 3``.

    Acquisition software writes the beam tilt into the file name as
    ``<x>x_<y>y`` (``sample_-17.0x_0.0y_master.h5``); the HDF5 metadata does
    not carry it. Plots labelled by position in the list make the reader
    count files to know which tilt they are looking at.
    """
    if file is not None:
        match = re.search(r"(-?\d+(?:\.\d+)?)x_(-?\d+(?:\.\d+)?)y", Path(file).name)
        if match:
            return f"{match.group(1)}x {match.group(2)}y"
    return f"Tilt {index}"


class TiltShifts(torch.Tensor):
    """(n, 2) tensor of one (row, col) value per tilt that prints as a table.

    Behaves as a plain tensor in arithmetic (results are ordinary tensors); only
    the printout differs, so a notebook cell ending in ``maped.real_space_shifts``
    shows labelled, rounded numbers instead of a raw tensor dump.
    """

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def wrap(values: torch.Tensor, label: str, unit: str) -> "TiltShifts":
        shifts = torch.Tensor._make_subclass(TiltShifts, values.detach())
        shifts.label, shifts.unit = label, unit
        return shifts

    def __repr__(self) -> str:
        rows = self.detach().cpu().tolist()
        digits = 0 if self.dtype in (torch.int32, torch.int64) else 2
        lines = [f"{self.label} ({self.unit}), one (row, col) per tilt", "tilt      row      col"]
        lines += [f"{i:4d} {row:8.{digits}f} {col:8.{digits}f}" for i, (row, col) in enumerate(rows)]
        return "\n".join(lines)


class MAPED(AutoSerialize):
    """
    Merge-Averaged Precession Electron Diffraction (MAPED) helper coded in PyTorch.

    This class manages a set of 4D-STEM datasets and provides utilities to:
    - compute mean BF and mean DP summaries,
    - find the direct beam in each mean diffraction pattern,
    - align diffraction space and real space,
    - merge the tilts into a single 4D-STEM dataset.
    """

    def __init__(self, tilts: Sequence, files: Sequence[str | Path] | None = None):
        """
        Start MAPED from tilts loaded by ``quantem.gpu.io``.

        Each tilt stays encoded on the GPU it was loaded on, and MAPED runs
        there, so no device is chosen here. The tilts remain owned by the
        caller. Without encoded residency a seven-tilt full-detector series
        does not fit on one GPU.

        Parameters
        ----------
        tilts : Sequence
            One loaded acquisition per tilt, all with the same shape
            (scan_row, scan_col, k_row, k_col) and on the same device.
        files : Sequence of paths, optional
            The files the tilts were loaded from, in the same order. Only the
            names are used: a tilt written into the name as ``-17.0x_0.0y``
            labels that tilt ``-17.0x 0.0y`` in every plot, so a reader can
            tell the tilts apart without counting. Without it, or when a name
            carries no tilt, plots say ``Tilt 0``, ``Tilt 1``, ...

        Examples
        --------
        >>> from quantem.gpu import io
        >>> tilts = io.load(files)
        >>> maped = MAPED(tilts, files)
        >>> merged = maped.run()
        """
        super().__init__()
        tilts = list(tilts)
        from quantem.gpu.io import Dataset4dstemGPU

        if not tilts or any(not isinstance(tilt, Dataset4dstemGPU) for tilt in tilts):
            raise TypeError("Load the tilts with quantem.gpu.io.load(files).")
        if any(tilt.shape != tilts[0].shape or str(tilt.device) != str(tilts[0].device) for tilt in tilts):
            raise ValueError("Tilts must share one shape and one device.")
        if files is not None and len(files) != len(tilts):
            raise ValueError("files must list one path per tilt, in load order.")
        self.datasets = tilts
        self.tilt_names = [_tilt_name(i, files[i] if files is not None else None) for i in range(len(tilts))]
        self.metadata: dict[str, Any] = {}
        self.device = str(tilts[0].device)
        config.set_device(self.device)
        # every stage works on these two small images per tilt: the mean diffraction
        # pattern (average over the scan) and the bright-field image (average over
        # the detector); one pass over the encoded counts, nothing is cleaned
        self.dp_mean, self.im_bf = _resident_summaries(self.datasets, self.device)
        self.scales = torch.ones(len(self.datasets), dtype=torch.float32, device=self.device)

    @property
    def mean_pattern(self) -> list[torch.Tensor]:
        """Mean diffraction pattern of each tilt, (k_row, k_col)."""
        return self.dp_mean

    @property
    def bright_field(self) -> list[torch.Tensor]:
        """Bright-field image of each tilt, (scan_row, scan_col)."""
        return self.im_bf

    @property
    def device(self) -> str:
        if hasattr(self, "_device"):
            return self._device
        return config.get_device()

    @device.setter
    def device(self, device: str | int | None) -> None:
        if device is not None:
            dev, _id = config.validate_device(device)
            self._device = dev
        # if None, leave unset so the property falls back to config.get_device()

    def __repr__(self) -> str:
        """Name the tilt count and device, e.g. ``MAPED(7 tilts on cuda:0)``.

        Every stage returns ``self`` for chaining, so a notebook cell ending in a
        stage call prints this. The default object address tells a reader nothing.
        """
        return f"MAPED({len(self.datasets)} tilts on {self.device})"

    def show_tilts(
        self,
        plot_scale: float | Sequence[float] | None = None,
        show_detector_center: bool = True,
        bright_field_cmap: str = "gray",
        bright_field_contrast: Any = "linear_auto",
        pattern_cmap: str = "inferno",
        pattern_contrast: Any = "power_sqrt",
        **plot_kwargs: Any,
    ) -> Self:
        """
        Show each tilt: its bright-field image and its mean diffraction pattern.

        Both images are computed when the MAPED is built (``bright_field`` and
        ``mean_pattern``, one per tilt); this only draws them.

        Parameters
        ----------
        plot_scale : float or sequence of float or None, optional
            Per-tilt divisor applied to the bright-field images in this plot
            only, so tilts of different dose can be compared by eye. The counts
            and every stage are unchanged (default None).
        show_detector_center : bool
            If True, mark the detector center with a red cross on every mean
            diffraction pattern, so the direct beam's offset from it is visible
            at a glance (default True).
        bright_field_cmap, pattern_cmap : str
            Matplotlib colormap of the bright-field images and of the patterns.
        bright_field_contrast, pattern_contrast
            Contrast of each column, as ``show_2d`` takes it: a preset name
            (``"linear_auto"``, ``"log_auto"``, ``"power_sqrt"``, all clipped
            to the 2nd to 98th percentile) or a dict such as
            ``{"lower_quantile": 0.01, "upper_quantile": 0.999, "power": 0.5}``.
        **plot_kwargs
            Passed to show_2d.

        Returns
        -------
        MAPED
            self
        """
        n = len(self.datasets)
        if plot_scale is None:
            self.scales = torch.ones(n, dtype=torch.float32, device=self.device)
        elif isinstance(plot_scale, (int, float, np.floating)):
            self.scales = torch.full((n,), float(plot_scale), dtype=torch.float32, device=self.device)
        else:
            self.scales = torch.tensor(plot_scale, dtype=torch.float32, device=self.device)
            if self.scales.shape != (n,):
                raise ValueError("plot_scale must be a scalar or a sequence with one entry per tilt.")
        if torch.any(self.scales == 0):
            raise ValueError("plot_scale entries must be nonzero.")
        tiles = [[(self.im_bf[i] / self.scales[i]), self.dp_mean[i]] for i in range(n)]
        titles = [[f"{name}: bright field", f"{name}: mean diffraction pattern"] for name in self.tilt_names]
        fig, ax = show_2d(
            tiles, title=titles,
            cmap=[[bright_field_cmap, pattern_cmap]] * n,
            norm=[[bright_field_contrast, pattern_contrast]] * n,
            returnfig=True, **plot_kwargs,
        )
        if show_detector_center:
            k_rows, k_cols = self.dp_mean[0].shape
            axs = np.asarray(ax, dtype=object).reshape(n, 2)
            for i in range(n):
                axs[i, 1].plot(
                    [k_cols / 2], [k_rows / 2], marker="+", color="red", markersize=14, markeredgewidth=2
                )
        return self

    def find_beam_center(
        self,
        centers: tuple | list | None = None,
        blur: float | None = 1,
        show: bool = True,
        plot_indices: list | None = None,
        **plot_kwargs: Any,
    ) -> Self:
        """
        Find the direct beam in each mean diffraction pattern, or take it as given.

        Parameters
        ----------
        centers : tuple or list, optional
            Beam centers to use instead of searching. Can be:
            - a single (row, col) tuple, applied to all datasets
            - a list of (row, col) tuples of length n (one per dataset)
        blur : float, optional
            Gaussian blur width (standard deviation, in detector pixels) applied
            to each mean DP before its brightest pixel is taken as the beam
            center. It keeps one noisy pixel from outshining the direct beam.
        show : bool, optional
            If True, plot mean diffraction patterns with the beam centers marked.
        plot_indices : list, optional
            Optional indices to plot. If None, plots all datasets.
        **plot_kwargs
            Passed to show_2d.

        Attributes
        ----------
        beam_centers : torch.Tensor
            Array of shape (n, 2) with integer (row, col) beam centers.

        Returns
        -------
        MAPED
            self (updated instance)
        """
        n = len(self.datasets)

        if plot_indices is None:
            plot_indices_list = list(range(n))
        else:
            plot_indices_list = list(plot_indices)
            for i in plot_indices_list:
                if i < 0 or i >= n:
                    raise IndexError("plot_indices contains an out-of-range index.")

        if blur is not None and float(blur) > 0:
            gaussian_filter_torch = torchvision.transforms.GaussianBlur(
                kernel_size=[2 * int(2 * float(blur)) + 1] * 2,
                sigma=[blur, blur],
            )

            dp_means_use = gaussian_filter_torch(torch.stack(self.dp_mean))
        else:
            dp_means_use = torch.stack(self.dp_mean)

        if centers is None:
            centers_t = torch.zeros((n, 2), dtype=torch.int)
            for i in range(n):
                dp_use = dp_means_use[i]

                r, c = torch.unravel_index(torch.argmax(dp_use), dp_use.shape)
                centers_t[i, 0] = int(r)
                centers_t[i, 1] = int(c)
        else:
            if isinstance(centers, tuple) and len(centers) == 2:
                centers_t = torch.tile(
                    torch.tensor(centers, dtype=torch.int, device=self.device)[None, :], (n, 1)
                )
            else:
                centers_list = list(centers)
                if len(centers_list) != n:
                    raise ValueError(
                        "centers must be a single (row, col) tuple or a list with one per tilt."
                    )
                centers_t = torch.tensor(centers_list, dtype=torch.int, device=self.device)
                if centers_t.shape != (n, 2):
                    raise ValueError("centers must have shape (n, 2) after conversion.")

        self.beam_centers = TiltShifts.wrap(centers_t, "beam centers", "detector px")

        if show:
            arrays = [np.asarray(self.dp_mean[i].cpu()) for i in plot_indices_list]
            # the red cross is the beam center; the title only names the tilt
            titles = [self.tilt_names[i] for i in plot_indices_list]
            # at most four patterns per row, so each stays large enough to read
            columns = 4
            fig, ax = show_2d(
                [arrays[i : i + columns] for i in range(0, len(arrays), columns)],
                title=[titles[i : i + columns] for i in range(0, len(titles), columns)],
                cmap="inferno", norm="power_sqrt", returnfig=True,
                **plot_kwargs,
            )
            axs = np.ravel(np.asarray(ax, dtype=object))
            for j, i in enumerate(plot_indices_list):
                r, c = self.beam_centers[i].cpu().numpy()
                axs[j].plot([c], [r], marker="+", color="red", markersize=16, markeredgewidth=2)

        return self

    def align_diffraction(
        self,
        border_taper: float = 2,
        precision: float = 0.01,
        show: bool = True,
        **plot_kwargs: Any,
    ) -> Self:
        """
        Align mean diffraction patterns using weighted cross-correlation in Fourier space.

        Parameters
        ----------
        border_taper : float
            Width (detector pixels) over which each mean DP fades to zero at
            the detector border (Tukey window), so the border itself is not
            correlated.
        precision : float
            Smallest shift step measured, in detector pixels. 0.01 measures to
            a hundredth of a pixel; 1 measures whole pixels only.
        show : bool
            If True, plot aligned mean diffraction patterns.
        **plot_kwargs
            Passed to show_2d when plotting.

        Attributes
        ----------
        diffraction_shifts : np.ndarray
            Array of shape (n, 2) with (row, col) shifts to align diffraction patterns.

        Returns
        -------
        MAPED
            self (updated instance)
        """
        if not hasattr(self, "beam_centers"):
            raise RuntimeError("Run find_beam_center() first so self.beam_centers exists.")

        H, W = self.dp_mean[0].shape

        w = (
            tukey_torch(
                H,
                alpha=2.0 * float(border_taper) / float(H),
                device=self.device,
                dtype=torch.float32,
            )[:, None]
            * tukey_torch(
                W,
                alpha=2.0 * float(border_taper) / float(W),
                device=self.device,
                dtype=torch.float32,
            )[None, :]
        )

        n = len(self.dp_mean)
        self.diffraction_shifts = torch.zeros((n, 2), device=self.device, dtype=torch.float32)

        G_ref = torch.fft.fft2(w * self.dp_mean[0])

        kr = torch.fft.fftfreq(H, device=self.device)[:, None]
        kc = torch.fft.fftfreq(W, device=self.device)[None, :]

        for ind in range(1, n):
            G = torch.fft.fft2(w * self.dp_mean[ind])
            shift_rc = cross_correlation_shift_torch(
                im_ref=G_ref,
                im=G,
                upsample_factor=round(1 / precision),
                fft_input=True,
            )

            phase_ramp = torch.exp(-2j * torch.pi * (kr * shift_rc[0] + kc * shift_rc[1]))

            G_shift = G * phase_ramp
            self.diffraction_shifts[ind, :] = shift_rc.clone()

            G_ref = G_ref * (ind / (ind + 1)) + G_shift / (ind + 1)

        self.diffraction_shifts -= torch.mean(self.diffraction_shifts, dim=0)[None, :]
        self.diffraction_shifts = TiltShifts.wrap(
            self.diffraction_shifts, "detector shifts", "detector px"
        )
        if show:
            im_aligned = shift_images_torch(
                images=torch.stack(self.dp_mean),
                shifts_rc=self.diffraction_shifts,
                edge_blend=float(border_taper),
                pad_val="min",
            )
            show_2d(
                im_aligned.mean(0),
                title=f"Mean diffraction pattern, {n} tilts aligned on the detector",
                cmap="inferno", norm="power_sqrt",
                **plot_kwargs,
            )

        return self

    def align_real_space(
        self,
        iterations: int = 20,
        max_shift: float = 5,
        edge_blur: float | None = 2,
        hann_window: bool = True,
        precision: float = 0.01,
        num_tilts: int | None = None,
        show: bool = True,
        **plot_kwargs: Any,
    ) -> Self:
        """
        Align real-space mean BF images using iterative average-reference correlation.

        Parameters
        ----------
        iterations : int
            Number of refinement iterations. One cycle aligns every tilt to the
            current average image and rebuilds that average. Raise it until
            doubling it no longer changes ``real_space_shifts``.
        max_shift : float
            Largest scan shift you expect between tilts, in scan pixels. It is
            an expectation, not a limit: the images are padded by this much
            (plus 4 pixels) so a shifted feature does not wrap around to the
            opposite side, and larger shifts are still measured.
        edge_blur : float or None
            Align on edge strength (gradient magnitude) instead of raw
            intensity, so slow brightness changes across the scan do not drive
            the alignment. The value is the Gaussian blur width (standard
            deviation, in scan pixels) applied to the gradients: how wide a
            feature must be to count as an edge. ``None`` aligns on intensity.
        hann_window : bool
            If True, fade each image to zero at the scan border (Hann window)
            before correlating, so the frame of the scan is not a feature.
        precision : float
            Smallest shift step measured, in scan pixels. 0.01 measures to a
            hundredth of a pixel; 1 measures whole pixels only.
        num_tilts : int, optional
            If provided, align only the first ``num_tilts`` tilts; the rest
            keep a zero shift.
        show : bool
            If True, plot aligned mean BF images.
        **plot_kwargs
            Passed to show_2d when plotting.

        Attributes
        ----------
        real_space_shifts : np.ndarray
            Array of shape (n_total, 2) with (row, col) shifts for aligned datasets.

        Returns
        -------
        MAPED
            self (updated instance)
        """
        if len(self.im_bf) == 0:
            raise RuntimeError("No images found in self.im_bf.")

        H, W = self.im_bf[0].shape
        for im in self.im_bf:
            if im.shape != (H, W):
                raise ValueError("all self.im_bf images must have the same shape")

        n_total = len(self.im_bf)
        if num_tilts is None:
            n = n_total
        else:
            n = int(num_tilts)
            if n <= 0:
                raise ValueError("num_tilts must be positive")
            n = min(n, n_total)

        if int(iterations) < 1:
            raise ValueError("iterations must be >= 1")

        pad_cc = int(np.ceil(float(max_shift))) + 4

        Hp = H + 2 * pad_cc
        Wp = W + 2 * pad_cc
        r0 = pad_cc
        c0 = pad_cc

        w_h = torch.ones((H, W), dtype=torch.float32, device=self.device)
        if hann_window:
            w_h = (
                torch.hann_window(H, dtype=torch.float32, device=self.device)[:, None]
                * torch.hann_window(W, dtype=torch.float32, device=self.device)[None, :]
            )
        w_h_pad = torch.zeros((Hp, Wp), dtype=torch.float32, device=self.device)
        w_h_pad[r0 : r0 + H, c0 : c0 + W] = w_h
        w_h_sum = torch.sum(w_h_pad)
        if w_h_sum <= 0:
            raise RuntimeError("hann window sum is zero")

        if edge_blur is not None:
            wx = torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            wx = None

        base_pad = torch.zeros((n, Hp, Wp), dtype=torch.float32, device=self.device)
        for i in range(n):
            im0 = self.im_bf[i].float()

            if edge_blur is not None:
                pad_symmetric = wx.shape[-1] // 2
                im0_pad = F.pad(
                    im0[None, None],
                    pad=(pad_symmetric, pad_symmetric, pad_symmetric, pad_symmetric),
                    mode="reflect",
                )

                gx = F.conv2d(im0_pad, wx[None, None])[0, 0]
                gy = F.conv2d(im0_pad, wx.T[None, None])[0, 0]

                gaussian_filt = torchvision.transforms.GaussianBlur(
                    kernel_size=[2 * int(2 * float(edge_blur)) + 1] * 2,
                    sigma=[edge_blur, edge_blur],
                )
                gx = gaussian_filt(gx[None])
                gy = gaussian_filt(gy[None])
                im_use = torch.sqrt(gx * gx + gy * gy)
            else:
                im_use = im0

            base_pad[i, r0 : r0 + H, c0 : c0 + W] = im_use

        shifts = torch.zeros((n, 2), dtype=torch.float32, device=self.device)

        upsample_factor = round(1 / precision)
        for _ in range(int(iterations)):
            # shift images to current guess
            ims_a = shift_images_torch(base_pad, shifts)
            ims_mean = torch.sum(ims_a * w_h_pad, dim=(1, 2)) / w_h_sum
            G_list = torch.fft.fft2((ims_a - ims_mean[:, None, None]) * w_h_pad[None])
            G_ref = torch.mean(G_list, dim=0)
            # correlate every tilt against the reference in one batched call, so a
            # cycle costs one launch sequence instead of n with a host sync each
            if n > 1:
                shifts[1:] += cross_correlation_shift_torch(
                    im_ref=G_ref[None].expand(n - 1, -1, -1),
                    im=G_list[1:],
                    upsample_factor=upsample_factor,
                    fft_input=True,
                ).to(shifts.dtype)
            shifts -= shifts[0:1].clone()

        shifts -= torch.mean(shifts, dim=0)[None, :]

        self.real_space_shifts = torch.zeros((n_total, 2), dtype=torch.float32, device=self.device)
        self.real_space_shifts[:n, :] = shifts
        self.real_space_shifts = TiltShifts.wrap(self.real_space_shifts, "scan shifts", "scan px")

        if show:
            im_aligned = shift_images_torch(
                images=torch.stack(self.im_bf[:n]),
                shifts_rc=self.real_space_shifts[:n, :],
                edge_blend=float(max_shift),
                pad_val="median",
                blend=True,
            )
            show_2d(
                im_aligned,
                title=f"Bright field, {n} tilts aligned in the scan",
                **plot_kwargs,
            )

        return self

    def merge(
        self,
        dtype=None,
        save_to: str | Path | None = None,
        scan_region: tuple[int, int, int, int] | None = None,
        crop: bool = False,
        release_tilts: bool = False,
        show: bool = True,
        verbose: bool = True,
        compile_merge: bool | None = None,
        compute_summaries: bool = True,
        profile_timings: dict[str, Any] | None = None,
        **plot_kwargs: Any,
    ) -> Any:
        """
        Merge the aligned tilts into one 4D-STEM dataset.

        Every merged diffraction pattern is the weighted mean of the tilts,
        each shifted bilinearly by its scan and detector alignment. The merge
        always computes in float32; ``dtype`` only chooses how the result is
        stored. A complete float32 result does not fit on the device, so the
        complete merge is stored as scaled uint16 and float32 is offered for a
        scan region.

        Notes
        -----
        Requires the following attributes to be present on ``self``:

        self.real_space_shifts
            From ``align_real_space()``.
        self.diffraction_shifts
            From ``align_diffraction()``.

        Parameters
        ----------
        dtype : str or torch.dtype, optional
            Output storage dtype. ``"scaled_uint16"`` computes the merge in
            float32 once, then retains calibrated, ANS-encoded uint16 regions on
            the GPU. Calibration and region sizes are automatic. Saving uses
            this storage by default. This does not reduce the precision of
            alignment, interpolation, or float32 accumulation. Reads reconstruct
            float32 intensities from the stored codes and calibration; rounding
            error is recorded in the result's ``metadata["precision"]``.
            ``None`` without ``save_to`` and explicit ``"float32"`` retain exact
            float32 scan-region inspection (at most 4096 scan positions).
            ``None`` with ``save_to`` selects scaled uint16.
        save_to : str, optional
            Output HDF5 path. The merge is written in bounded regions with their
            intensity calibration, then reopened encoded for viewing. This holds
            less in memory than merging first and saving afterwards, at the cost
            of disk IO during the merge.
        scan_region : tuple of int, optional
            Inspect this region of the merge without saving:
            ``(row_start, row_stop, column_start, column_stop)``, exclusive stops
            in the full aligned scan coordinates. The tilts and the alignment
            remain available for another inspection or the complete merge.
            The returned region retains the complete detector and float32
            intensities. Select at most 4096 scan positions for bounded memory.
            This option is only available without ``save_to``.
        crop : bool
            If True, keep only the scan positions every tilt covers after its
            shift. Border positions see fewer tilts and so average a different
            tilt set; cropping makes the result uniform and smaller. The box is
            printed and recorded in ``metadata["maped_merge"]["scan_box"]``.
        release_tilts : bool
            If True, close the tilts once the complete merge no longer needs
            them, so the result does not share the device with its inputs. Load
            the tilts again to rerun. Default False keeps them for another merge.
        show : bool
            If True, plot merged BF and merged mean DP.
        verbose : bool
            If True, print the storage precision report.
        compile_merge : bool, optional
            Compile the scan interpolation with ``torch.compile``. If None, this
            fuses large interior scan regions on MPS.
        compute_summaries : bool
            If True, compute merged BF and mean-DP summaries after merging.
        profile_timings : dict, optional
            If provided, populated with phase timings for profiling.
        **plot_kwargs
            Passed to show_2d.

        Returns
        -------
        quantem.gpu.io.Dataset4dstemGPU
            Merged dataset on the device: the complete scaled result, or a
            float32 scan region.

        Examples
        --------
        >>> patch = maped.merge(scan_region=(252, 260, 252, 260), show=False)
        >>> merged = maped.merge(dtype="scaled_uint16")
        """
        from quantem.gpu import io as gpu_io

        from ._maped_resident import ResidentMergeSource

        if not hasattr(self, "real_space_shifts"):
            raise RuntimeError("Run align_real_space() first so self.real_space_shifts exists.")
        if not hasattr(self, "diffraction_shifts"):
            raise RuntimeError("Run align_diffraction() first so self.diffraction_shifts exists.")
        if dtype not in (None, "scaled_uint16") and (
            save_to is not None or dtype not in ("float32", torch.float32)
        ):
            raise ValueError("dtype must be 'scaled_uint16', or 'float32' without save_to.")
        profile_start = time.perf_counter()
        scan_rows, scan_cols = self.im_bf[0].shape
        k_rows, k_cols = self.dp_mean[0].shape
        backend = torch.device(self.device).type

        if save_to is None and dtype != "scaled_uint16":
            generated = ResidentMergeSource(
                self.datasets,
                self.real_space_shifts,
                self.diffraction_shifts,
                close_sources_before_reopen=False,
                crop=crop,
            )
            region = generated.scan_box if scan_region is None else scan_region
            if (
                len(region) != 4
                or any(not isinstance(value, (int, np.integer)) for value in region)
                or not (0 <= region[0] < region[1] <= scan_rows)
                or not (0 <= region[2] < region[3] <= scan_cols)
            ):
                raise ValueError(
                    f"scan_region={region} must be (row_start, row_stop, "
                    f"column_start, column_stop) inside {(scan_rows, scan_cols)}."
                )
            row0, row1, column0, column1 = map(int, region)
            if (row1 - row0) * (column1 - column0) > 4096:
                raise ValueError(
                    "A complete float32 merge is too large to keep on the device. "
                    "Use dtype='scaled_uint16' for the complete merge, or select "
                    "scan_region with at most 4096 scan positions for exact float32."
                )
            try:
                parts = list(generated.blocks((row0, row1, column0, column1)))
                values = parts[0] if len(parts) == 1 else torch.cat(parts)
                values = values.reshape(row1 - row0, column1 - column0, k_rows, k_cols)
                metadata = {
                    key.removeprefix("quantem_").removesuffix("_v1"): json.loads(value)
                    for key, value in generated.save_metadata.items()
                }
                metadata.update(
                    representation="dense",
                    residency="device",
                    working_shape=tuple(values.shape),
                    working_dtype="float32",
                )
                metadata["maped_merge"]["scan_region"] = list(region)
                metadata["maped_merge"]["scan_box"] = list(generated.scan_box)
                result = gpu_io.Dataset4dstemGPU(values, metadata)
            finally:
                generated.close()
            self.merged = result
            if compute_summaries or show:
                self.im_bf_merged = values.mean(dim=(-2, -1))
                self.dp_mean_merged = values.flatten(0, 1).mean(dim=0)
            else:
                self.im_bf_merged = self.dp_mean_merged = None
            titles = ["Merged Region Bright Field", "Merged Region Mean Diffraction Pattern"]
        else:
            if scan_region is not None:
                raise ValueError(
                    "scan_region selects float32 inspection; omit it for the "
                    "complete scaled_uint16 result."
                )
            generated = ResidentMergeSource(
                self.datasets,
                self.real_space_shifts,
                self.diffraction_shifts,
                close_sources_before_reopen=save_to is not None and release_tilts,
                compile_merge=compile_merge,
                compute_region_frames=2048 if save_to is not None else None,
                crop=crop,
            )
            if crop and verbose:
                row0, row1, column0, column1 = generated.scan_box
                print(
                    f"Cropped to rows {row0}:{row1}, columns {column0}:{column1} "
                    f"({row1 - row0} x {column1 - column0} of {scan_rows} x {scan_cols}), "
                    f"where all {len(self.datasets)} tilts overlap. crop=False keeps the full scan."
                )
            try:
                if save_to is None:
                    result = gpu_io.load(
                        generated, dtype="scaled_uint16", backend=backend, verbose=verbose
                    )
                else:
                    # stream completed regions to disk so the complete result never
                    # shares the device with the float32 regions being merged
                    gpu_io.save(
                        save_to, generated, dtype="scaled_uint16", backend=backend, verbose=verbose
                    )
            finally:
                generated.close()
            if release_tilts:
                for tilt in self.datasets:
                    tilt.close()
                self.datasets = []
                if backend == "cuda":
                    torch.cuda.empty_cache()
                else:
                    torch.mps.empty_cache()
            if save_to is not None:
                result = gpu_io.load(
                    save_to, dtype="scaled_uint16", backend=backend, verbose=verbose
                )
            result.metadata.setdefault("maped_merge", {})["scan_box"] = list(generated.scan_box)
            self.merged = result
            if compute_summaries or show:
                # accumulated from the float32 regions during the merge pass, so the
                # stored result is not decoded a second time just for two pictures
                self.dp_mean_merged = generated.detector_sum / math.prod(generated.shape[:2])
                self.im_bf_merged = generated.scan_sum / (k_rows * k_cols)
            else:
                self.dp_mean_merged = self.im_bf_merged = None
            if profile_timings is not None:
                profile_timings.update(result.metadata.get("maped_merge", {}))
            titles = ["Merged Bright Field", "Merged Mean Diffraction Pattern"]
        if profile_timings is not None:
            profile_timings["total_profiled_seconds"] = time.perf_counter() - profile_start
        if show:
            show_2d(
                [[self.im_bf_merged, self.dp_mean_merged]], title=[titles],
                cmap=[["gray", "inferno"]], norm=[[None, "power_sqrt"]], **plot_kwargs,
            )
        return result

    def run(
        self,
        *,
        blur: float = 1,
        border_taper: float = 2,
        iterations: int = 20,
        hann_window: bool = True,
        max_shift: float = 5,
        edge_blur: float | None = 2,
        precision: float = 0.01,
        dtype: str = "scaled_uint16",
        crop: bool = True,
        save_to: str | Path | None = None,
        release_tilts: bool = False,
        show: bool = True,
        verbose: bool = True,
    ) -> Any:
        """
        Run every MAPED stage in order and return the merged dataset.

        Calls ``find_beam_center``, ``align_diffraction``, ``align_real_space``
        and ``merge`` with the settings of the qualified seven-tilt workflow, so
        one call replaces four. Each stage is the same method a step-by-step
        workflow calls, so the two give identical shifts and identical merged
        values.

        Parameters
        ----------
        blur, border_taper, iterations, hann_window, max_shift, edge_blur, precision
            Settings of the stages, with the same meaning as on each stage.
        dtype, crop, save_to, release_tilts
            Storage settings of ``merge``. ``crop`` keeps only the scan positions
            every tilt covers, and says so. ``scaled_uint16`` stores the complete
            merge compactly; the progress line reports its rounding error.
        show : bool
            If True, draw the four pictures: the tilts, the aligned mean
            diffraction pattern, the aligned bright field, the merged result.
            Set False in scripts and on the command line.
        verbose : bool
            If True, print one line per stage: what was done, what was found,
            and how long it took.

        Returns
        -------
        quantem.gpu.io.Dataset4dstemGPU
            Merged dataset, as returned by ``merge``.

        Examples
        --------
        >>> merged = maped.run()
        >>> merged = maped.run(iterations=80, show=False)
        """
        scan_rows, scan_cols = self.im_bf[0].shape
        k_rows, k_cols = self.dp_mean[0].shape
        started = stage_started = time.perf_counter()

        def report(step: int, done: str, found: str = "") -> None:
            """Print one finished stage: what was done, what was found, how long it took."""
            nonlocal stage_started
            if verbose:
                elapsed = time.perf_counter() - stage_started
                print(f"[{step}/4] {done:<64s} {elapsed:5.1f} s   {found}".rstrip())
            stage_started = time.perf_counter()

        if verbose:
            print(
                f"MAPED: {len(self.datasets)} tilts, {scan_rows} x {scan_cols} scan, "
                f"{k_rows} x {k_cols} detector, on {self.device}"
            )
        self.find_beam_center(blur=blur, show=False)
        if show:
            self.show_tilts()
        report(1, "Beam centers found on the mean diffraction patterns")
        self.align_diffraction(border_taper=border_taper, precision=precision, show=show)
        # reading the largest shift also waits for the device, so the time is real
        report(
            2, "Detector alignment: mean diffraction patterns aligned",
            f"largest detector shift {float(self.diffraction_shifts.abs().max()):.2f} px",
        )
        self.align_real_space(
            iterations=iterations, hann_window=hann_window, max_shift=max_shift,
            edge_blur=edge_blur, precision=precision, show=show,
        )
        report(
            3, f"Real-space alignment: bright-field images aligned, {iterations} iterations",
            f"largest scan shift {float(self.real_space_shifts.abs().max()):.2f} px",
        )
        merged = self.merge(
            dtype=dtype, crop=crop, save_to=save_to, release_tilts=release_tilts, show=show,
            verbose=verbose,
        )
        precision_report = merged.metadata.get("precision")
        if precision_report is None:
            report(4, "Merged, exact float32")
        else:
            report(
                4, "Merged, stored as scaled 16-bit",
                f"{merged.resident_bytes / 2**30:.2f} GiB, rounding RMSE "
                f"{precision_report['rmse']:.4f} and at most {precision_report['max_abs_error']:.4f} counts",
            )
            if verbose:
                print(
                    "      Exact float32 is available for a patch: "
                    "maped.merge(scan_region=(row0, row1, col0, col1))"
                )
        if verbose:
            print(f"Done in {time.perf_counter() - started:.1f} s")
        return merged

    def close(self) -> None:
        """Release the merged result. The tilts stay with their caller."""
        merged = getattr(self, "merged", None)
        if merged is not None and hasattr(merged, "close"):
            merged.close()
            self.merged = None
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
            if fit_shifts:
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
                    # vmax=3,
                    # vmin=-3,
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

    return diffraction_shifts, shifted_dps
