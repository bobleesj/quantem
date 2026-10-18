"""Public MAPED workflow for alignment, merging, and result inspection."""

import json
import math
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from quantem.core import config
from quantem.core.datastructures import Dataset4dstem
from quantem.core.io.serialize import AutoSerialize

from . import plot
from .alignment import align_images_fourier_torch as align_images_fourier_torch
from .alignment import align_images_fourier_torch_batched as align_images_fourier_torch_batched
from .alignment import cross_correlation_shift_torch as cross_correlation_shift_torch
from .alignment import dftUpsample_torch as dftUpsample_torch
from .alignment import dscan_correct as dscan_correct
from .alignment import fit_surface_lstsq as fit_surface_lstsq
from .alignment import shift_images_torch as shift_images_torch
from .alignment import tukey_torch as tukey_torch
from .alignment import upsampled_correlation_torch as upsampled_correlation_torch


def _summaries(sources, device: str):
    """Compute mean diffraction patterns and scan images in bounded memory."""
    sources = list(sources)
    if torch.device(device).type == "cpu":
        from .merge import _read_scan_region

        patterns, images = [], []
        for source in sources:
            rows, columns, detector_rows, detector_columns = source.shape
            pattern_t = torch.zeros((detector_rows, detector_columns), dtype=torch.float64)
            image_t = torch.empty((rows, columns), dtype=torch.float32)
            for row in range(0, rows, 16):
                stop = min(row + 16, rows)
                counts_t = _read_scan_region(source, (row, stop, 0, columns)).to(torch.float64)
                pattern_t += counts_t.sum(dim=(0, 1))
                image_t[row:stop] = counts_t.mean(dim=(2, 3))
            patterns.append((pattern_t / (rows * columns)).to(torch.float32))
            images.append(image_t)
        return patterns, images
    from quantem.gpu.detector import prepare
    from quantem.gpu.io.read import torch_value

    native_sources = [getattr(source, "data", source) for source in sources]
    if all(
        callable(getattr(source, "mean_dp_device", None))
        and callable(getattr(source, "detector_mean_device", None))
        for source in native_sources
    ):
        # Encoded count means round once after exact reduction, including on MPS.
        return (
            [torch_value(source.mean_dp_device()) for source in native_sources],
            [torch_value(source.detector_mean_device()) for source in native_sources],
        )
    session = prepare(sources[0] if len(sources) == 1 else sources)
    mean_dp_t = torch_value(session.mean_dp(output="native"))
    detector_mask = np.ones(session.detector_shape, dtype=bool)
    detector_sum_t = torch_value(session.masked_sum(detector_mask, output="native"))
    if not detector_sum_t.is_floating_point():
        # Divide exact integer totals before rounding to the float32 summary.
        detector_sum_t = detector_sum_t.to(torch.float64)
    detector_mean_t = (detector_sum_t / math.prod(session.detector_shape)).to(mean_dp_t.dtype)
    return (
        list(mean_dp_t.reshape(-1, *session.detector_shape).unbind(0)),
        list(detector_mean_t.reshape(-1, *session.scan_shape).unbind(0)),
    )


def _tilt_name(index: int, file: str | Path | None) -> str:
    """
    Plot label of one tilt: ``-17.0x 0.0y`` read from its file name, else ``Tilt 3``.

    Acquisition software writes the beam tilt into the file name as
    ``<x>x_<y>y`` (``sample_-17.0x_0.0y_master.h5``); the HDF5 metadata does
    not carry it. ``maped.io.load`` and ``quantem.gpu.io.load`` record the file in
    ``metadata["source_path"]``. Plots labelled by position in the list make
    the reader count files to know which tilt they are looking at.
    """
    if file is not None:
        match = re.search(r"(-?\d+(?:\.\d+)?)x_(-?\d+(?:\.\d+)?)y", Path(file).name)
        if match:
            return f"{match.group(1)}x {match.group(2)}y"
    return f"Tilt {index}"


class MAPED(AutoSerialize):
    """
    Merge-Averaged Precession Electron Diffraction (MAPED) helper coded in PyTorch.

    This class manages a set of 4D-STEM datasets and provides utilities to:
    - compute mean BF and mean DP summaries,
    - find the direct beam in each mean diffraction pattern,
    - align diffraction space and real space,
    - merge the tilts into a single 4D-STEM dataset.
    """

    def __init__(self, tilts: Sequence):
        """
        Start MAPED from native QuantEM datasets or encoded GPU acquisitions.

        MAPED uses the device selected while loading. GPU inputs stay encoded;
        small CPU examples use dense arrays. The tilts remain owned by the
        caller. Full experimental tilt series should use encoded GPU storage.

        Parameters
        ----------
        tilts : Sequence
            One loaded acquisition per tilt, all with the same shape
            (scan_row, scan_col, k_row, k_col) and on the same device.
            CPU inputs are native ``Dataset4dstem`` objects; they do not require
            ``quantem.gpu``. Encoded inputs use ``quantem.gpu.io.load``. Plots
            name each tilt from ``metadata["source_path"]`` when available: a file named
            ``..._-17.0x_0.0y_master.h5`` is labelled ``-17.0x 0.0y``; a name
            without a tilt or source path is labelled ``Tilt 0``, ``Tilt 1``, ...

        Examples
        --------
        >>> from quantem.gpu import io
        >>> tilts = io.load(files)
        >>> maped = MAPED.from_datasets(tilts)
        >>> merged = maped.run()
        """
        super().__init__()
        tilts = list(tilts)
        if not tilts:
            raise ValueError("Provide at least one loaded 4D-STEM acquisition.")
        if all(isinstance(tilt, Dataset4dstem) for tilt in tilts):
            devices = ["cpu" if tilt.array is not None else str(tilt.tensor.device) for tilt in tilts]
            if any(device != "cpu" for device in devices):
                raise ValueError(
                    "Native MAPED datasets must be on CPU; load encoded GPU inputs "
                    "with quantem.gpu.io.load(files, device='cuda:0' or 'mps')."
                )
        else:
            from quantem.gpu.io import Dataset4dstemGPU

            if any(not isinstance(tilt, Dataset4dstemGPU) for tilt in tilts):
                raise TypeError("Provide native Dataset4dstem objects or quantem.gpu.io datasets.")
            devices = [str(tilt.device) for tilt in tilts]
        if any(tilt.shape != tilts[0].shape for tilt in tilts) or len(set(devices)) != 1:
            raise ValueError("Tilts must share one shape and one device.")
        self.datasets = tilts
        self.tilt_names = [_tilt_name(i, tilt.metadata.get("source_path")) for i, tilt in enumerate(tilts)]
        self.metadata: dict[str, object] = {}
        self.device = devices[0]
        config.set_device(self.device)
        # every stage works on these two small images per tilt: the mean diffraction
        # pattern (average over the scan) and the bright-field image (average over
        # the detector); one pass over the encoded counts, nothing is cleaned
        self.dp_mean, self.im_bf = _summaries(self.datasets, self.device)
        self.scales = torch.ones(len(self.datasets), dtype=torch.float32, device=self.device)

    @classmethod
    def from_datasets(cls, datasets: Sequence) -> Self:
        """Align acquisitions already loaded on CPU, CUDA, or MPS.

        Parameters
        ----------
        datasets : Sequence
            Loaded 4D-STEM acquisitions with matching shapes and devices.
            Compressed acquisitions remain encoded during processing.

        Returns
        -------
        MAPED
            Workflow borrowing the supplied acquisitions.

        Examples
        --------
        >>> tilt = Dataset4dstem.from_file("tilt_master.h5", "arina")
        >>> tilts = [tilt]
        >>> maped = MAPED.from_datasets(tilts)
        >>> merged = maped.run(show=False)
        """
        return cls(datasets)

    def report(self) -> dict[str, list[str] | list[float]]:
        """Return measured detector and scan shifts in pixels, grouped by column.

        Returns
        -------
        dict
            Tilt labels and available row/column shifts as lists in acquisition
            order. All values are plain Python objects.

        Examples
        --------
        >>> maped.align_diffraction(show=False)
        >>> maped.align_real_space(show=False)
        >>> maped.report()
        """
        columns: dict[str, list[str] | list[float]] = {"Tilt": self.tilt_names.copy()}
        for attribute, label in (
            ("diffraction_shifts", "Detector"),
            ("real_space_shifts", "Scan"),
        ):
            shifts_t = getattr(self, attribute, None)
            if shifts_t is not None:
                columns[f"{label} row (px)"] = shifts_t[:, 0].detach().cpu().tolist()
                columns[f"{label} col (px)"] = shifts_t[:, 1].detach().cpu().tolist()
        return columns

    def show(
        self,
        positions: Sequence[tuple[int, int]],
        *,
        tilt_indices: Sequence[int] | None = None,
    ):
        """Compare individual diffraction patterns at matching specimen positions.

        Each row shows one tilt or the merged result. The scan image marks
        the requested positions; the following panels contain their diffraction
        patterns, not averages over the scan. Input patterns use the same scan
        and detector interpolation as the merge. Display ranges are shared.

        Parameters
        ----------
        positions : sequence of (int, int)
            Positions in the cropped merged scan, in (row, col) pixels.
        tilt_indices : sequence of int, optional
            Input tilts to compare with the merge. By default, show every tilt.

        Returns
        -------
        matplotlib.figure.Figure
            Scan images with marked positions and their diffraction patterns.

        Examples
        --------
        >>> merged = maped.merge(crop=True, show=False)
        >>> maped.show(positions=[(20, 30), (40, 50)], tilt_indices=[0])
        """
        return plot.positions(self, positions, tilt_indices=tilt_indices)

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
        bright_field_contrast= "linear_auto",
        pattern_cmap: str = "inferno",
        pattern_contrast= "power_sqrt",
        **plot_kwargs,
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
        plot.tilts(
            self, show_detector_center, bright_field_cmap, bright_field_contrast,
            pattern_cmap, pattern_contrast, **plot_kwargs,
        )
        return self

    def find_beam_center(
        self,
        centers: tuple | list | None = None,
        blur: float | None = 1,
        show: bool = True,
        plot_indices: list | None = None,
        **plot_kwargs,
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

        self.beam_centers = centers_t

        if show:
            plot.beam_centers(self, plot_indices_list, **plot_kwargs)

        return self

    def align_diffraction(
        self,
        border_taper: float = 2,
        precision: float = 0.01,
        show: bool = True,
        **plot_kwargs,
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
        if show:
            plot.diffraction_alignment(self, border_taper, **plot_kwargs)

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
        **plot_kwargs,
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

        if show:
            plot.scan_alignment(self, n, max_shift, **plot_kwargs)

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
        profile_timings: dict[str, object] | None = None,
        **plot_kwargs,
    ):
        """
        Merge the aligned tilts into one 4D-STEM dataset.

        Every merged diffraction pattern is the weighted mean of the tilts,
        each shifted bilinearly by its scan and detector alignment. The merge
        always computes in float32; ``dtype`` only chooses how the result is
        stored. Large acquisitions can exceed device memory. Use ``save_to``
        to write an exact float32 result in bounded regions,
        then reopen it as lossless ANS. The reopened result must fit in memory;
        ``release_tilts=True`` frees input residents first.

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
            Output storage dtype; omitted or ``"float32"`` retains exact
            float32 values on every backend. In-memory results are limited to
            4 GiB; ``save_to`` writes bounded blocks without that allocation.
            Explicit ``"scaled_uint16"`` retains calibrated, ANS-encoded uint16
            regions on the GPU. It leaves alignment and float32 accumulation
            unchanged but rounds the stored result. Reads reconstruct float32
            intensities from the codes and calibration; rounding error is
            recorded in ``metadata["precision"]``. Alternatively, keep the
            exact merge and choose scaled storage when saving it with
            ``quantem.gpu.io.save(..., dtype="scaled_uint16")`` as HDF5.
        save_to : str, optional
            Output HDF5 path. The merge is written in bounded regions with the
            requested storage precision, then reopened encoded for viewing. This holds
            less in memory than merging first and saving afterwards, at the cost
            of disk IO during the merge.
        scan_region : tuple of int, optional
            Inspect this region of the merge without saving:
            ``(row_start, row_stop, column_start, column_stop)``, exclusive stops
            in the full aligned scan coordinates. The tilts and the alignment
            remain available for another inspection or the complete merge.
            The returned region retains the complete detector and float32
            intensities. The complete float32 result must fit within 4 GiB.
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
        Dataset4dstem or quantem.gpu.io.Dataset4dstemGPU
            Native CPU inputs return a native ``Dataset4dstem``. Encoded inputs
            return the dataset wrapper supplied by ``quantem.gpu``.

        Examples
        --------
        >>> patch = maped.merge(scan_region=(252, 260, 252, 260), show=False)
        >>> merged = maped.merge(save_to="merged_master.h5", release_tilts=True)
        >>> merged = maped.merge(dtype="scaled_uint16")
        """
        from .merge import MergeSource

        native = isinstance(self.datasets[0], Dataset4dstem)
        if not native:
            from quantem.gpu import io as gpu_io

        if not hasattr(self, "real_space_shifts"):
            raise RuntimeError("Run align_real_space() first so self.real_space_shifts exists.")
        if not hasattr(self, "diffraction_shifts"):
            raise RuntimeError("Run align_diffraction() first so self.diffraction_shifts exists.")
        if dtype not in (None, "scaled_uint16", "float32", torch.float32):
            raise ValueError("dtype must be 'scaled_uint16' or 'float32'.")
        profile_start = time.perf_counter()
        scan_rows, scan_cols = self.im_bf[0].shape
        k_rows, k_cols = self.dp_mean[0].shape
        backend = torch.device(self.device).type
        if backend == "cpu" and (dtype == "scaled_uint16" or save_to is not None):
            raise ValueError(
                "CPU MAPED retains float32 values. Use merge() "
                "on a small scan, then save the returned dataset."
            )

        in_memory = save_to is None and dtype != "scaled_uint16"
        storage_dtype = "scaled_uint16" if dtype == "scaled_uint16" else "float32"
        if scan_region is not None and not in_memory:
            raise ValueError(
                "scan_region selects float32 inspection; omit it for the "
                "complete saved result."
            )
        generated = MergeSource(
            self.datasets,
            self.real_space_shifts,
            self.diffraction_shifts,
            close_sources_before_reopen=save_to is not None and release_tilts,
            compile_merge=compile_merge,
            compute_region_frames=2048 if save_to is not None else None,
            crop=crop,
        )
        try:
            if in_memory:
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
                shape = (row1 - row0, column1 - column0, k_rows, k_cols)
                output_bytes = math.prod(shape) * 4
                if output_bytes > 4 * 1024**3:
                    raise ValueError(
                        f"This float32 merge needs {output_bytes / 1024**3:.2f} GiB; "
                        "the in-memory limit is 4 GiB. Select a smaller scan_region "
                        "for exact float32, use save_to for a bounded float32 archive, "
                        "or explicitly choose dtype='scaled_uint16' "
                        "to accept calibrated rounding for the complete merge."
                    )
                values = torch.empty(shape, dtype=torch.float32, device=self.device)
                destination = values.flatten(0, 1)
                cursor = 0
                for block in generated.blocks((row0, row1, column0, column1)):
                    destination[cursor : cursor + len(block)].copy_(block)
                    cursor += len(block)
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
                if native:
                    source = self.datasets[0]
                    origin = np.asarray(source.origin) + np.asarray(source.sampling) * [
                        row0, column0, 0, 0
                    ]
                    result = Dataset4dstem.from_array(
                        values.numpy(), name="MAPED merged",
                        origin=origin, sampling=source.sampling, units=source.units,
                        signal_units=source.signal_units,
                    )
                    metadata["residency"] = "host"
                    result.metadata.update(metadata)
                else:
                    result = gpu_io.Dataset4dstemGPU(values, metadata)
            else:
                if crop and verbose:
                    row0, row1, column0, column1 = generated.scan_box
                    print(
                        f"Cropped to rows {row0}:{row1}, columns {column0}:{column1} "
                        f"({row1 - row0} x {column1 - column0} of {scan_rows} x {scan_cols}), "
                        f"where all {len(self.datasets)} tilts overlap. crop=False keeps the full scan."
                    )
                if save_to is None:
                    result = gpu_io.load(
                        generated, dtype=storage_dtype, backend=backend, verbose=verbose
                    )
                else:
                    # Write completed regions without allocating the full float32 result.
                    gpu_io.save(
                        save_to, generated, dtype=storage_dtype, backend=backend, verbose=verbose
                    )
        finally:
            generated.close()

        if not in_memory:
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
                    save_to, dtype="native" if storage_dtype == "float32" else storage_dtype,
                    backend=backend, verbose=verbose, hot_pixel_correction="none",
                )
            result.metadata.setdefault("maped_merge", {})["scan_box"] = list(generated.scan_box)
            if profile_timings is not None:
                profile_timings.update(result.metadata.get("maped_merge", {}))
        self.merged = result
        self.im_bf_merged = self.dp_mean_merged = None
        if compute_summaries or show:
            if in_memory:
                self.im_bf_merged = values.mean(dim=(-2, -1))
                self.dp_mean_merged = values.flatten(0, 1).mean(dim=0)
            else:
                # Reuse the sums from the merge pass instead of decoding stored values.
                self.im_bf_merged = generated.scan_sum / (k_rows * k_cols)
                self.dp_mean_merged = generated.detector_sum / math.prod(generated.shape[:2])
        if profile_timings is not None:
            profile_timings["total_profiled_seconds"] = time.perf_counter() - profile_start
        if show:
            plot.merged(self, in_memory, **plot_kwargs)
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
        dtype: str | None = None,
        crop: bool = True,
        save_to: str | Path | None = None,
        release_tilts: bool = False,
        show: bool = True,
        verbose: bool = True,
    ):
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
            Omitted ``dtype`` retains exact float32 on every backend, including
            when saving. Choose ``save_to`` for results larger than 4 GiB.
        show : bool
            If True, draw the four pictures: the tilts, the aligned mean
            diffraction pattern, the aligned bright field, the merged result.
            Set False in scripts and on the command line.
        verbose : bool
            If True, print one line per stage: what was done, what was found,
            and how long it took.

        Returns
        -------
        Dataset4dstem or quantem.gpu.io.Dataset4dstemGPU
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
