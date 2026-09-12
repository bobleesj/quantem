from __future__ import annotations

import json
import math
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Self, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as ndi_shift
from scipy.signal import convolve2d
from scipy.signal.windows import tukey
from tqdm import tqdm

from quantem.core import config
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.imaging_utils import weighted_cross_correlation_shift
from quantem.core.visualization import show_2d


class MAPED(AutoSerialize):
    """
    Merge-Averaged Precession Electron Diffraction (MAPED) helper.

    This class manages a set of 4D-STEM datasets and provides utilities to:
    - compute mean BF and mean DP summaries,
    - choose/find diffraction origins,
    - align diffraction space and real space,
    - merge datasets into a single composite Dataset4dstem.
    """

    _token = object()

    def __init__(self, datasets: list[Dataset4dstem], _token: object | None = None):
        if _token is not self._token:
            raise RuntimeError("Use MAPED.from_datasets() to instantiate this class.")
        super().__init__()
        self.datasets = datasets
        self.metadata: dict[str, Any] = {}

    @classmethod
    def from_datasets(cls, datasets: Sequence[Dataset4dstem]) -> MAPED:
        """
        Construct a MAPED instance from a non-empty sequence of Dataset4dstem.

        Parameters
        ----------
        datasets
            Sequence of Dataset4dstem instances.

        Returns
        -------
        MAPED
            New MAPED instance.
        """
        if not isinstance(datasets, Sequence) or isinstance(datasets, (str, bytes)):
            raise TypeError("MAPED.from_datasets expects a sequence of Dataset4dstem instances.")
        ds_list: list[Dataset4dstem] = []
        for d in datasets:
            if not isinstance(d, Dataset4dstem):
                raise TypeError(
                    "MAPED.from_datasets expects a sequence of Dataset4dstem instances."
                )
            ds_list.append(d)
        if not ds_list:
            raise ValueError(
                "MAPED.from_datasets expects a non-empty sequence of Dataset4dstem instances."
            )
        return cls(datasets=ds_list, _token=cls._token)

    def preprocess(
        self,
        plot_summary: bool = True,
        scale: float | Sequence[float] | None = None,
        **plot_kwargs: Any,
    ) -> MAPED:
        """
        Compute dataset summary images.

        Parameters
        ----------
        plot_summary : bool, optional
            If True, display summary plots (default True).
        scale : float or sequence of float or None, optional
            Per-dataset scaling factor(s) (default None).

        Attributes
        ----------
        scales : np.ndarray
            Per-dataset scaling factors (n,).
        dp_mean : list[np.ndarray]
            Mean diffraction patterns (H, W), one per dataset.
        im_bf : list[np.ndarray]
            Mean bright-field images (R, C), one per dataset.

        Returns
        -------
        MAPED
            self (updated instance)
        """
        n = len(self.datasets)
        if scale is None:
            self.scales = np.ones(n, dtype=float)
        elif isinstance(scale, (int, float, np.floating)):
            self.scales = np.full(n, float(scale), dtype=float)
        else:
            self.scales = np.asarray(list(scale), dtype=float)
            if self.scales.shape != (n,):
                raise ValueError(
                    "scale must be a scalar or a sequence with the same length as datasets."
                )
        if np.any(self.scales == 0):
            raise ValueError("scale entries must be nonzero.")

        self.dp_mean: list[np.ndarray] = []
        self.im_bf: list[np.ndarray] = []

        for d in self.datasets:
            if hasattr(d, "get_dp_mean"):
                try:
                    d.get_dp_mean()
                except TypeError:
                    try:
                        d.get_dp_mean(returnval=False)
                    except Exception:
                        pass

            dp = getattr(d, "dp_mean", None)
            if dp is None:
                arr = np.asarray(d.array)
                dp_arr = np.mean(arr, axis=(0, 1))
            else:
                dp_arr = np.asarray(dp.array if hasattr(dp, "array") else dp)

            arr = np.asarray(d.array)
            im_bf_arr = np.mean(arr, axis=(2, 3))

            self.dp_mean.append(np.asarray(dp_arr))
            self.im_bf.append(np.asarray(im_bf_arr))

        if plot_summary:
            tiles = [[(self.im_bf[i] / self.scales[i]), self.dp_mean[i]] for i in range(n)]
            titles = [
                [f"{i} - Mean Bright Field", f"{i} - Mean Diffraction Pattern"] for i in range(n)
            ]
            show_2d(tiles, title=titles, **plot_kwargs)

        return self

    def diffraction_origin(
        self,
        origins=None,
        sigma=None,
        plot_origins: bool = True,
        plot_indices=None,
        **plot_kwargs: Any,
    ) -> MAPED:
        """
        Choose or automatically find the origin in diffraction space.

        Parameters
        ----------
        origins : tuple or sequence, optional
            Optional manual origins. Can be:
            - a single (row, col) tuple, applied to all datasets
            - a list of (row, col) tuples of length n (one per dataset)
        sigma : float, optional
            Optional low-pass smoothing sigma (pixels) applied to each mean DP prior to peak finding.
        plot_origins : bool, optional
            If True, plot mean diffraction patterns with overlaid origin markers.
        plot_indices : sequence of int, optional
            Optional indices to plot. If None, plots all datasets.
        **plot_kwargs
            Passed to show_2d.

        Attributes
        ----------
        diffraction_origins : np.ndarray
            Array of shape (n, 2) with integer (row, col) origins.

        Returns
        -------
        MAPED
            self (updated instance)
        """
        n = len(self.datasets)
        if not hasattr(self, "dp_mean"):
            raise RuntimeError("Run preprocess() first so self.dp_mean exists.")

        if plot_indices is None:
            plot_indices_list = list(range(n))
        else:
            plot_indices_list = list(plot_indices)
            for i in plot_indices_list:
                if i < 0 or i >= n:
                    raise IndexError("plot_indices contains an out-of-range index.")

        if origins is None:
            origins_arr = np.zeros((n, 2), dtype=int)
            for i in range(n):
                dp = np.asarray(self.dp_mean[i])
                if sigma is not None and float(sigma) > 0:
                    dp_use = gaussian_filter(
                        dp.astype(float, copy=False), float(sigma), mode="nearest"
                    )
                else:
                    dp_use = dp
                r, c = np.unravel_index(int(np.argmax(dp_use)), dp_use.shape)
                origins_arr[i, 0] = int(r)
                origins_arr[i, 1] = int(c)
        else:
            if isinstance(origins, tuple) and len(origins) == 2:
                origins_arr = np.tile(np.asarray(origins, dtype=int)[None, :], (n, 1))
            else:
                origins_list = list(origins)
                if len(origins_list) != n:
                    raise ValueError(
                        "origins must be a single (row,col) tuple or a list of length n."
                    )
                origins_arr = np.asarray(origins_list, dtype=int)
                if origins_arr.shape != (n, 2):
                    raise ValueError("origins must have shape (n, 2) after conversion.")

        self.diffraction_origins = origins_arr

        if plot_origins:
            arrays = [np.asarray(self.dp_mean[i]) for i in plot_indices_list]
            titles = [f"{i} - Mean Diffraction Pattern" for i in plot_indices_list]
            fig, ax = show_2d(arrays, title=titles, returnfig=True, **plot_kwargs)
            axs = np.ravel(np.asarray(ax, dtype=object))
            for j, i in enumerate(plot_indices_list):
                r, c = self.diffraction_origins[i]
                axs[j].plot([c], [r], marker="+", color="red", markersize=16, markeredgewidth=2)

        return self

    def diffraction_align(
        self,
        edge_blend: float = 16.0,
        padding=None,
        pad_val: str | float = "min",
        upsample_factor: int = 100,
        weight_scale: float = 1 / 8,
        plot_aligned: bool = True,
        **plot_kwargs: Any,
    ) -> MAPED:
        """
        Align mean diffraction patterns using weighted cross-correlation in Fourier space.

        Parameters
        ----------
        edge_blend : float
            Tukey window edge taper (pixels).
        padding : int or None
            Passed to shift_images for plotting.
        pad_val : str or float
            Passed to shift_images for plotting.
        upsample_factor : int
            Subpixel upsampling factor for correlation peak estimation.
        weight_scale : float
            Radial weight falloff scale (fraction of mean DP size).
        plot_aligned : bool
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
        if not hasattr(self, "dp_mean"):
            raise RuntimeError("Run preprocess() first so self.dp_mean exists.")
        if not hasattr(self, "diffraction_origins"):
            raise RuntimeError(
                "Run diffraction_origin() first so self.diffraction_origins exists."
            )

        H, W = np.asarray(self.dp_mean[0]).shape

        w = (
            tukey(H, alpha=2.0 * float(edge_blend) / float(H))[:, None]
            * tukey(W, alpha=2.0 * float(edge_blend) / float(W))[None, :]
        )

        r = np.fft.fftfreq(H, 1.0 / float(H))[:, None]
        c = np.fft.fftfreq(W, 1.0 / float(W))[None, :]

        n = len(self.dp_mean)
        self.diffraction_shifts = np.zeros((n, 2), dtype=float)

        G_ref = np.fft.fft2(w * np.asarray(self.dp_mean[0]))
        xy0 = np.asarray(self.diffraction_origins[0], dtype=float)

        for ind in range(1, n):
            G = np.fft.fft2(w * np.asarray(self.dp_mean[ind]))
            xy = np.asarray(self.diffraction_origins[ind], dtype=float)

            dr2 = (r - xy0[0] + xy[0]) ** 2 + (c - xy0[1] + xy[1]) ** 2
            im_weight = np.clip(
                1.0 - np.sqrt(dr2) / float(np.mean((H, W))) / float(weight_scale),
                0.0,
                1.0,
            )
            im_weight = np.sin(im_weight * np.pi / 2.0) ** 2

            shift_rc, G_shift = weighted_cross_correlation_shift(
                im_ref=G_ref,
                im=G,
                weight_real=im_weight * 0.0 + 1.0,
                upsample_factor=int(upsample_factor),
                fft_input=True,
                fft_output=True,
                return_shifted_image=True,
            )
            self.diffraction_shifts[ind, :] = np.asarray(shift_rc, dtype=float)

            G_ref = G_ref * (ind / (ind + 1)) + G_shift / (ind + 1)

        self.diffraction_shifts -= np.mean(self.diffraction_shifts, axis=0)[None, :]

        if plot_aligned:
            im_aligned = shift_images(
                images=self.dp_mean,
                shifts_rc=self.diffraction_shifts,
                edge_blend=float(edge_blend),
                padding=padding,
                pad_val=pad_val,
            )
            show_2d(im_aligned, **plot_kwargs)

        return self

    def real_space_align(  # torch.grid_sample
        self,
        num_images=None,
        num_iter: int = 3,
        edge_blend: float = 1.0,
        padding=None,
        pad_val: str | float = "median",
        upsample_factor: int = 100,
        max_shift=None,
        shift_method: str = "bilinear",
        edge_filter: bool = True,
        edge_sigma: float = 2.0,
        hanning_filter: bool = False,
        plot_aligned: bool = True,
        **plot_kwargs: Any,
    ) -> MAPED:
        """
        Align real-space mean BF images using iterative average-reference correlation.

        Parameters
        ----------
        num_images : int, optional
            If provided, align only the first num_images images.
        num_iter : int
            Number of refinement iterations.
        edge_blend : float
            Used to set default correlation padding when max_shift is None.
        padding : int or None
            Passed to shift_images for plotting.
        pad_val : str or float
            Passed to shift_images for plotting.
        upsample_factor : int
            Subpixel upsampling factor for correlation peak estimation.
        max_shift : float, optional
            Optional maximum shift constraint passed to weighted_cross_correlation_shift.
        shift_method : str
            Passed to shift_images for plotting ('bilinear' or 'fourier').
        edge_filter : bool
            If True, correlate on gradient magnitude instead of raw intensity.
        edge_sigma : float
            Gaussian sigma applied to gradients when edge_filter is True.
        hanning_filter : bool
            If True, apply a Hanning window prior to FFT.
        plot_aligned : bool
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
        if not hasattr(self, "im_bf"):
            raise RuntimeError("Run preprocess() first so self.im_bf exists.")
        if len(self.im_bf) == 0:
            raise RuntimeError("No images found in self.im_bf.")

        H, W = self.im_bf[0].shape
        for im in self.im_bf:
            if im.shape != (H, W):
                raise ValueError("all self.im_bf images must have the same shape")

        n_total = len(self.im_bf)
        if num_images is None:
            n = n_total
        else:
            n = int(num_images)
            if n <= 0:
                raise ValueError("num_images must be positive")
            n = min(n, n_total)

        if int(num_iter) < 1:
            raise ValueError("num_iter must be >= 1")

        if max_shift is not None:
            pad_cc = int(np.ceil(float(max_shift))) + 4
        else:
            pad_cc = int(np.ceil(float(edge_blend))) + 4

        Hp = H + 2 * pad_cc
        Wp = W + 2 * pad_cc
        r0 = pad_cc
        c0 = pad_cc

        w_h = np.ones((H, W), dtype=float)
        if hanning_filter:
            w_h = np.hanning(H)[:, None] * np.hanning(W)[None, :]
        w_h_pad = np.zeros((Hp, Wp), dtype=float)
        w_h_pad[r0 : r0 + H, c0 : c0 + W] = w_h
        w_h_sum = float(np.sum(w_h_pad))
        if w_h_sum <= 0:
            raise RuntimeError("hanning window sum is zero")

        if edge_filter:
            wx = np.array(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                dtype=float,
            )
        else:
            wx = None

        base_pad = np.zeros((n, Hp, Wp), dtype=float)
        for i in range(n):
            im0 = np.asarray(self.im_bf[i], dtype=float)

            if edge_filter:
                gx = convolve2d(im0, wx, mode="same", boundary="symm")
                gy = convolve2d(im0, wx.T, mode="same", boundary="symm")
                gx = gaussian_filter(gx, float(edge_sigma), mode="nearest")
                gy = gaussian_filter(gy, float(edge_sigma), mode="nearest")
                im_use = np.sqrt(gx * gx + gy * gy)
            else:
                im_use = im0

            base_pad[i, r0 : r0 + H, c0 : c0 + W] = im_use

        shifts = np.zeros((n, 2), dtype=float)

        for _ in range(int(num_iter)):
            G_list = np.empty((n, Hp, Wp), dtype=np.complex128)

            for i in range(n):
                im_a = ndi_shift(
                    base_pad[i],
                    shift=(shifts[i, 0], shifts[i, 1]),
                    order=1,
                    mode="constant",
                    cval=0.0,
                    prefilter=False,
                )
                im_mean = float(np.sum(im_a * w_h_pad) / w_h_sum)
                im_win = (im_a - im_mean) * w_h_pad
                G_list[i] = np.fft.fft2(im_win)

            G_ref = np.mean(G_list, axis=0)

            for i in range(1, n):
                drc = weighted_cross_correlation_shift(
                    im_ref=G_ref,
                    im=G_list[i],
                    weight_real=None,
                    upsample_factor=int(upsample_factor),
                    max_shift=max_shift,
                    fft_input=True,
                    fft_output=False,
                    return_shifted_image=False,
                )
                shifts[i, 0] += float(drc[0])
                shifts[i, 1] += float(drc[1])

            shifts -= shifts[0][None, :]

        shifts -= np.mean(shifts, axis=0)[None, :]

        self.real_space_shifts = np.zeros((n_total, 2), dtype=float)
        self.real_space_shifts[:n, :] = shifts

        if plot_aligned:
            im_aligned = shift_images(
                images=self.im_bf[:n],
                shifts_rc=self.real_space_shifts[:n, :],
                edge_blend=float(edge_blend),
                padding=padding,
                pad_val=pad_val,
                shift_method=shift_method,
            )
            show_2d(im_aligned, **plot_kwargs)

        return self

    def merge_datasets(
        self,
        real_space_padding=0,
        real_space_edge_blend=1.0,
        diffraction_padding=0,
        diffraction_edge_blend=0.0,
        diffraction_pad_val="min",
        shift_method: str = "bilinear",
        dtype=None,
        scale_output: bool = False,
        plot_result: bool = True,
        **plot_kwargs: Any,
    ) -> Dataset4dstem:
        """
        Merge aligned datasets into a single Dataset4dstem.

        Notes
        -----
        Requires the following attributes to be present on ``self``:

        self.real_space_shifts
            From ``real_space_align()``.
        self.diffraction_shifts
            From ``diffraction_align()``.

        Parameters
        ----------
        real_space_padding
            Output scan padding in pixels (adds border to scan grid).
        real_space_edge_blend
            Tukey taper width for scan-space interpolation weights.
        diffraction_padding
            Output diffraction padding in pixels (adds border around DPs).
        diffraction_edge_blend
            Tukey taper width for diffraction-space weights.
        diffraction_pad_val
            Pad value for diffraction padding ('min','max','mean','median' or float).
        shift_method
            Diffraction shift method: 'bilinear' or 'fourier'.
        dtype
            Output dtype. If None, uses parent dtype.
        scale_output
            If True and dtype is integer, scale to full dynamic range using global max.
        plot_result
            If True, plot merged BF and merged mean DP.
        **plot_kwargs
            Passed to show_2d.

        Returns
        -------
        Dataset4dstem
            Merged dataset.
        """
        if not hasattr(self, "real_space_shifts"):
            raise RuntimeError("Run real_space_align() first so self.real_space_shifts exists.")
        if not hasattr(self, "diffraction_shifts"):
            raise RuntimeError("Run diffraction_align() first so self.diffraction_shifts exists.")

        arrays = [np.asarray(d.array) for d in self.datasets]
        n = len(arrays)
        if n == 0:
            raise RuntimeError("No datasets found in self.datasets.")

        Rs, Cs, H, W = arrays[0].shape
        for a in arrays:
            if a.shape != (Rs, Cs, H, W):
                raise ValueError("All dataset arrays must have the same shape (Rs, Cs, H, W).")

        rs_shifts = np.asarray(self.real_space_shifts, dtype=float)
        dp_shifts = np.asarray(self.diffraction_shifts, dtype=float)
        if rs_shifts.shape != (n, 2):
            raise ValueError("self.real_space_shifts must have shape (n, 2).")
        if dp_shifts.shape != (n, 2):
            raise ValueError("self.diffraction_shifts must have shape (n, 2).")

        if dtype is None:
            dtype_out = np.asarray(arrays[0]).dtype
            warnings.warn(f"dtype=None; using parent dtype {dtype_out}.", RuntimeWarning)
        else:
            dtype_out = np.dtype(dtype)

        real_space_padding = int(real_space_padding)
        diffraction_padding = int(diffraction_padding)

        Rout = Rs + 2 * real_space_padding
        Cout = Cs + 2 * real_space_padding

        Hp = H + 2 * diffraction_padding
        Wp = W + 2 * diffraction_padding
        rp0 = diffraction_padding
        cp0 = diffraction_padding

        method = str(shift_method).strip().lower()
        if method not in {"bilinear", "fourier"}:
            raise ValueError("shift_method must be 'bilinear' or 'fourier'.")

        if real_space_edge_blend and float(real_space_edge_blend) > 0:
            alpha_r = min(1.0, 2.0 * float(real_space_edge_blend) / float(Rs))
            alpha_c = min(1.0, 2.0 * float(real_space_edge_blend) / float(Cs))
            w_rs = tukey(Rs, alpha=alpha_r)[:, None] * tukey(Cs, alpha=alpha_c)[None, :]
        else:
            w_rs = np.ones((Rs, Cs), dtype=float)
        w_rs = w_rs.astype(float, copy=False)

        if diffraction_edge_blend and float(diffraction_edge_blend) > 0:
            alpha_dr = min(1.0, 2.0 * float(diffraction_edge_blend) / float(H))
            alpha_dc = min(1.0, 2.0 * float(diffraction_edge_blend) / float(W))
            w_dp = tukey(H, alpha=alpha_dr)[:, None] * tukey(W, alpha=alpha_dc)[None, :]
        else:
            w_dp = np.ones((H, W), dtype=float)
        w_dp = w_dp.astype(float, copy=False)

        dp_means = [np.mean(a, axis=(0, 1), dtype=np.float64) for a in arrays]
        v = np.stack(dp_means, axis=0).reshape(-1)

        if isinstance(diffraction_pad_val, str):
            s = diffraction_pad_val.strip().lower()
            if s == "min":
                pad_val_dp = float(np.min(v))
            elif s == "max":
                pad_val_dp = float(np.max(v))
            elif s == "mean":
                pad_val_dp = float(np.mean(v))
            elif s == "median":
                pad_val_dp = float(np.median(v))
            else:
                raise ValueError(
                    "diffraction_pad_val must be a float or one of {'min','max','mean','median'}."
                )
        else:
            pad_val_dp = float(diffraction_pad_val)

        wdp_pad = np.zeros((Hp, Wp), dtype=float)
        wdp_pad[rp0 : rp0 + H, cp0 : cp0 + W] = w_dp

        wdp_shifted = np.zeros((n, Hp, Wp), dtype=float)
        if method == "fourier":
            kr = np.fft.fftfreq(Hp)[:, None]
            kc = np.fft.fftfreq(Wp)[None, :]
            Fw = np.fft.fft2(wdp_pad)
            ramps: list[np.ndarray] = []
            for i in range(n):
                dr, dc = dp_shifts[i, 0], dp_shifts[i, 1]
                ramp = np.exp(-2j * np.pi * (kr * dr + kc * dc))
                ramps.append(ramp)
                w_i = np.fft.ifft2(Fw * ramp).real
                wdp_shifted[i] = np.clip(w_i, 0.0, 1.0)
        else:
            for i in range(n):
                w_i = ndi_shift(
                    wdp_pad,
                    shift=(dp_shifts[i, 0], dp_shifts[i, 1]),
                    order=1,
                    mode="constant",
                    cval=0.0,
                    prefilter=False,
                )
                wdp_shifted[i] = np.clip(w_i, 0.0, 1.0)
            ramps = []

        coverage = np.clip(np.sum(wdp_shifted, axis=0), 0.0, 1.0)
        edge_w_dp = 1.0 - coverage

        merged = np.zeros((Rout, Cout, Hp, Wp), dtype=np.float64)

        dp_local = np.zeros((H, W), dtype=np.float64)
        dp_pad = np.zeros((Hp, Wp), dtype=np.float64)
        dp_shifted_tmp = np.zeros((Hp, Wp), dtype=np.float64)
        num_tmp = np.zeros((Hp, Wp), dtype=np.float64)
        den_tmp = np.zeros((Hp, Wp), dtype=np.float64)

        for ro in tqdm(range(Rout), desc="Merging (rows)"):
            r_base = ro - real_space_padding
            for co in range(Cout):
                c_base = co - real_space_padding

                num_tmp.fill(0.0)
                den_tmp.fill(0.0)
                max_wi = 0.0

                for i in range(n):
                    r_in = r_base - rs_shifts[i, 0]
                    c_in = c_base - rs_shifts[i, 1]

                    r0 = int(np.floor(r_in))
                    c0 = int(np.floor(c_in))
                    if r0 < 0 or r0 >= Rs - 1 or c0 < 0 or c0 >= Cs - 1:
                        continue

                    dr = r_in - r0
                    dc = c_in - c0

                    w00 = (1.0 - dr) * (1.0 - dc)
                    w10 = dr * (1.0 - dc)
                    w01 = (1.0 - dr) * dc
                    w11 = dr * dc

                    wi = (
                        w00 * w_rs[r0, c0]
                        + w10 * w_rs[r0 + 1, c0]
                        + w01 * w_rs[r0, c0 + 1]
                        + w11 * w_rs[r0 + 1, c0 + 1]
                    )
                    if wi <= 0.0:
                        continue
                    if wi > max_wi:
                        max_wi = wi

                    a = arrays[i]
                    dp_local[:] = (
                        w00 * a[r0, c0]
                        + w10 * a[r0 + 1, c0]
                        + w01 * a[r0, c0 + 1]
                        + w11 * a[r0 + 1, c0 + 1]
                    )

                    dp_pad.fill(0.0)
                    dp_pad[rp0 : rp0 + H, cp0 : cp0 + W] = dp_local * w_dp

                    if method == "fourier":
                        ramp = ramps[i]
                        dp_shifted_tmp[:] = np.fft.ifft2(np.fft.fft2(dp_pad) * ramp).real
                    else:
                        dp_shifted_tmp[:] = ndi_shift(
                            dp_pad,
                            shift=(dp_shifts[i, 0], dp_shifts[i, 1]),
                            order=1,
                            mode="constant",
                            cval=0.0,
                            prefilter=False,
                        )

                    num_tmp += wi * dp_shifted_tmp
                    den_tmp += wi * wdp_shifted[i]

                if max_wi <= 0.0:
                    merged[ro, co] = 0.0
                    continue

                num = num_tmp + edge_w_dp * pad_val_dp
                den = den_tmp + edge_w_dp

                out = np.empty_like(num)
                np.divide(num, den, out=out, where=den != 0.0)
                out[den == 0.0] = 0.0
                merged[ro, co] = out

        self.im_bf_merged = np.mean(merged, axis=(2, 3), dtype=np.float64)
        self.dp_mean_merged = np.mean(merged, axis=(0, 1), dtype=np.float64)

        if np.issubdtype(dtype_out, np.integer):
            info = np.iinfo(dtype_out)
            dmin = float(info.min)
            dmax = float(info.max)

            merged_f = merged

            if scale_output:
                peak = float(np.max(merged_f))
                if peak <= 0.0:
                    merged_scaled = merged_f
                else:
                    merged_scaled = merged_f * (dmax / peak)

                if np.issubdtype(dtype_out, np.unsignedinteger):
                    lo, hi = 0.0, dmax
                else:
                    lo, hi = dmin, dmax

                merged_out = np.rint(np.clip(merged_scaled, lo, hi)).astype(dtype_out)
            else:
                below = float(np.min(merged_f))
                above = float(np.max(merged_f))
                if below < dmin or above > dmax:
                    warnings.warn(
                        f"Output overflow for dtype {dtype_out}: data range [{below}, {above}] exceeds "
                        f"[{dmin}, {dmax}]. Values will be clipped.",
                        RuntimeWarning,
                    )
                merged_out = np.rint(np.clip(merged_f, dmin, dmax)).astype(dtype_out)
        else:
            merged_out = merged.astype(dtype_out, copy=False)

        dataset_merged = Dataset4dstem.from_array(array=merged_out)
        dataset_merged.im_bf_merged = self.im_bf_merged
        dataset_merged.dp_mean_merged = self.dp_mean_merged

        if plot_result:
            show_2d(
                [[self.im_bf_merged, self.dp_mean_merged]],
                title=[["Merged Bright Field", "Merged Mean Diffraction Pattern"]],
                **plot_kwargs,
            )

        return dataset_merged


class _TiltFiles:
    """A tilt series that lives on disk and is read one tilt at a time.

    The whole point of MAPED is to merge tilts that are individually large
    (a no-bin 4D tilt is 19.3 GB; seven of them is 135 GB - more than a GPU
    holds). So when the tilts come from files, MAPED keeps only the one it is
    currently working on resident and releases it before reading the next.
    ``preprocess`` and ``merge_datasets`` index this exactly like a list of
    in-memory tilts, so the rest of the pipeline is unaware of the difference.

    ``read(path)`` returns one tilt as a uint16 torch tensor on the GPU.

    Releasing matters: the pipeline keeps a torch *view* of each tilt (a dlpack
    share of the underlying GPU buffer), and a bare ``del`` does not always drop
    that view's refcount to zero on the same Python tick - so without a gc the
    freed buffer never returns to the pool and a second tilt accumulates. A
    gc.collect() before the pool reclaim makes the release deterministic.
    """

    def __init__(self, paths: Sequence[str], read):
        self.paths = list(paths)
        self.read = read
        self._current = None

    def __len__(self) -> int:
        return len(self.paths)

    def release(self, *, reclaim_cache: bool = True) -> None:
        """Drop and free the tilt currently in memory (idempotent)."""
        had_current = self._current is not None
        self._current = None
        if not reclaim_cache or not had_current:
            return
        import gc

        gc.collect()
        try:
            from quantem.gpu.device import release_cached_memory

            release_cached_memory()
        except (ImportError, RuntimeError):
            pass
        # Apple GPUs need the same courtesy: the Metal loader hands back buffers
        # that torch does not own, so dropping the tensor alone leaves them on the
        # shared pool and the next tilt stacks on top of the last one.
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def __getitem__(self, i: int):
        self.release()
        self._current = self.read(self.paths[i])
        return self._current

    def __setitem__(self, i: int, value):
        raise NotImplementedError(
            "Cannot modify file-backed tilts (from_files) in place - they stream "
            "read-only from disk one at a time. Steps that rewrite each tilt (e.g. "
            "dscan_align) need the tilts in memory; build the MAPED with "
            "from_datasets instead."
        )

    def __iter__(self):
        for i in range(len(self.paths)):
            yield self[i]


class _ResidentTilts:
    """Borrow exact encoded tilt owners without materializing dense tilts."""

    def __init__(self, sources, *, owns_sources: bool = False):
        self.sources = list(sources)
        self.owns_sources = bool(owns_sources)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        return self.sources[index]

    def __iter__(self):
        return iter(self.sources)

    def close(self):
        """Release sources loaded by MAPED while preserving borrowed sources."""
        if not self.owns_sources:
            return
        for source in self.sources:
            source.close()
        self.sources.clear()


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


def _shift_diffraction_batch(
    dp_padded: torch.Tensor,
    method: str,
    ramp: torch.Tensor,
    shift_rc: torch.Tensor,
    batch_n: int,
    cout: int,
    hp: int,
    wp: int,
) -> torch.Tensor:
    """Rigidly shift a ``(batch, Cout, Hp, Wp)`` block of diffraction patterns

    Both merge paths - the streaming tilt-outer path and the in-memory batch-outer
    path - shift every pattern of a tilt by the same per-tilt ``(dr, dc)`` before
    accumulating, so the shift lives here once (a fix to either method lands in one
    place). ``"fourier"`` multiplies by a precomputed phase ramp in frequency space:
    exact sub-pixel, but its sinc kernel rings on the sharp direct beam and leaves a
    faint center cross. ``"bilinear"`` interpolates in real space via the canonical
    ``shift_images_torch`` (batched over all ``batch * Cout`` images): no FFT, no
    ringing, a slight blur.

    Parameters
    ----------
    dp_padded : torch.Tensor
        ``(batch, Cout, Hp, Wp)`` diffraction patterns to shift.
    method : str
        ``"fourier"`` or ``"bilinear"``.
    ramp : torch.Tensor
        Precomputed ``exp(-2i pi (kr*dr + kc*dc))`` phase ramp for this tilt
        (used only by ``"fourier"``).
    shift_rc : torch.Tensor
        ``(1, 2)`` real-space ``(dr, dc)`` shift for this tilt (used only by
        ``"bilinear"``).
    batch_n, cout, hp, wp : int
        The dimensions of ``dp_padded``.

    Returns
    -------
    torch.Tensor
        The shifted ``(batch, Cout, Hp, Wp)`` patterns.
    """
    if method == "fourier":
        fft_result = torch.fft.fft2(dp_padded)
        fft_result.mul_(ramp[None, None])
        return torch.fft.ifft2(fft_result).real
    return _shift_constant_bilinear_grid_sample_convention(
        dp_padded.reshape(batch_n * cout, hp, wp),
        shift_rc,
    ).reshape(batch_n, cout, hp, wp)


def _shift_constant_bilinear_grid_sample_convention(
    images: torch.Tensor,
    shift_rc: torch.Tensor,
) -> torch.Tensor:
    """Bilinear shift for one shared shift using the existing grid_sample convention.

    ``shift_images_torch`` builds a full per-image grid even though MAPED shifts all
    diffraction patterns for one tilt by the same ``(row, col)`` offset. With
    ``align_corners=True`` its normalized ``2 * shift / size`` convention samples
    source coordinates ``out - shift * (size - 1) / size``. This helper evaluates
    that same separable bilinear interpolation with four slice adds and zero
    padding, avoiding the large random-gather grid.
    """
    n, height, width = images.shape
    out = torch.zeros_like(images)
    shift = shift_rc.reshape(-1)
    row_offset = -float(shift[0]) * float(height - 1) / float(height)
    col_offset = -float(shift[1]) * float(width - 1) / float(width)
    row_floor = math.floor(row_offset)
    col_floor = math.floor(col_offset)
    row_frac = row_offset - row_floor
    col_frac = col_offset - col_floor

    row_taps = ((row_floor, 1.0 - row_frac), (row_floor + 1, row_frac))
    col_taps = ((col_floor, 1.0 - col_frac), (col_floor + 1, col_frac))
    for row_delta, row_weight in row_taps:
        if row_weight == 0.0:
            continue
        row_start = max(0, -row_delta)
        row_stop = min(height, height - row_delta)
        if row_start >= row_stop:
            continue
        src_row_start = row_start + row_delta
        src_row_stop = row_stop + row_delta
        for col_delta, col_weight in col_taps:
            weight = row_weight * col_weight
            if weight == 0.0:
                continue
            col_start = max(0, -col_delta)
            col_stop = min(width, width - col_delta)
            if col_start >= col_stop:
                continue
            src_col_start = col_start + col_delta
            src_col_stop = col_stop + col_delta
            out[:, row_start:row_stop, col_start:col_stop] += (
                images[:, src_row_start:src_row_stop, src_col_start:src_col_stop]
                * weight
            )
    return out


@lru_cache(maxsize=4)
def _compiled_merge_tail(method: str):
    if method == "fourier":

        def _tail(
            num_band: torch.Tensor,
            dp_padded: torch.Tensor,
            wi: torch.Tensor,
            ramp: torch.Tensor,
            shift_rc: torch.Tensor,
        ) -> torch.Tensor:
            fft_result = torch.fft.fft2(dp_padded)
            shifted = torch.fft.ifft2(fft_result * ramp[None, None]).real
            return num_band + wi[..., None, None] * shifted

    else:

        def _tail(
            num_band: torch.Tensor,
            dp_padded: torch.Tensor,
            wi: torch.Tensor,
            ramp: torch.Tensor,
            shift_rc: torch.Tensor,
        ) -> torch.Tensor:
            batch_n, cout, hp, wp = dp_padded.shape
            n_det_images = batch_n * cout
            shifted = shift_images_torch(
                dp_padded.reshape(n_det_images, hp, wp),
                shift_rc.expand(n_det_images, 2),
                mode="bilinear",
            ).reshape(batch_n, cout, hp, wp)
            return num_band + wi[..., None, None] * shifted

    return torch.compile(_tail, backend="inductor", dynamic=False)


class MAPEDTorch(AutoSerialize):
    """
    Merge-Averaged Precession Electron Diffraction (MAPED) helper coded in PyTorch.

    This class manages a set of 4D-STEM datasets and provides utilities to:
    - compute mean BF and mean DP summaries,
    - choose/find diffraction origins,
    - align diffraction space and real space,
    - merge datasets into a single composite Dataset4dstem.
    """

    _token = object()

    def __init__(
        self,
        datasets: list[torch.Tensor],
        dtype: str | Any,
        device: str | int | None = None,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use MAPED.from_datasets() to instantiate this class.")
        super().__init__()
        self.datasets = datasets
        self.metadata: dict[str, Any] = {}
        self.device = device
        self.dtype = dtype

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

    @classmethod
    def from_datasets(cls, datasets: Sequence[torch.Tensor]) -> MAPED:
        """
        Construct a MAPED instance from a non-empty sequence of Dataset4dstem.

        Parameters
        ----------
        datasets
            Sequence of Dataset4dstem instances.

        Returns
        -------
        MAPED
            New MAPED instance.
        """
        if not isinstance(datasets, Sequence) or isinstance(datasets, (str, bytes)):
            raise TypeError("MAPED.from_datasets expects a sequence of Torch tensor instances.")
        ds_list: list[torch.Tensor] = []
        for d in datasets:
            if not isinstance(d, torch.Tensor):
                raise TypeError(
                    "MAPED.from_datasets expects a sequence of Torch tensor instances."
                )
            ds_list.append(d)

        dtypes = [dataset.dtype for dataset in datasets]

        # check that all datasets have the same dtype
        if len(set(str(d) for d in dtypes)) > 1:
            raise TypeError("All datasets need to have the same type")
        # Frames MAY span devices (a sharded multi-GPU 5D-STEM series, e.g. 7 tilts
        # distributed across 2 GPUs): preprocess + merge move each frame to the
        # compute device on demand, so the storage device need not be uniform.
        # The compute device is the configured device, not the frames' storage.
        from quantem.core.config import get_device

        compute_device = get_device()

        if not ds_list:
            raise ValueError(
                "MAPED.from_datasets expects a non-empty sequence of Torch tensor instances."
            )
        return cls(datasets=ds_list, _token=cls._token, device=compute_device, dtype=dtypes[0])

    @classmethod
    def from_files(
        cls,
        paths: Sequence[str],
        read: Callable[[str], torch.Tensor] | None = None,
        device: str | torch.device | None = None,
        det_bin: int | None = None,
        backend: str | None = None,
    ) -> Self:
        """Build MAPED from files using native-count encoded residency by default.

        The CUDA path keeps all encoded acquisitions resident and computes summaries
        and bounded merge regions directly from them. It never constructs a complete
        dense input tilt. Stored detector-mask pixels receive the loader's default
        GPU local-median replacement before encoded storage. A custom reader, detector
        binning, or another backend keeps the established one-file-at-a-time path.

        Parameters
        ----------
        paths : Sequence[str]
            One ``*_master.h5`` tilt file per tilt, e.g. from
            ``quantem.widget.io.discover_masters``.
        read : callable, optional
            ``read(path) -> uint16 torch tensor`` for a single tilt. Supplying this
            selects the established dense streaming path.
        device : str or torch.device, optional
            Compute device, e.g. ``"cuda:0"``. Default picks the least-busy GPU so the
            large merge does not land on a card already running something else.
        det_bin : int, optional
            Bin the detector by this factor in the dense streaming path. Default
            ``None`` preserves the complete detector and selects encoded residency.
        backend : str, optional
            Decode backend. Default ``None`` selects the current accelerator encoded
            compute device.

        Returns
        -------
        Self
            A MAPED backed by encoded resident sources on CUDA.

        Examples
        --------
        >>> from quantem.widget.io import discover_masters
        >>> files = discover_masters("/data/sample/maped")
        >>> maped = MAPEDTorch.from_files(files)
        >>> maped.preprocess(); maped.diffraction_origin(); maped.diffraction_align()
        >>> maped.real_space_align()
        >>> merged = maped.merge_datasets(save_to="merged_master.h5")
        """
        from quantem.core.config import get_device, set_device

        paths = list(paths)
        if not paths:
            raise ValueError("MAPEDTorch.from_files expects a non-empty sequence of paths.")
        if device is None and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            from quantem.gpu.device import least_busy_cuda_device

            device = f"cuda:{least_busy_cuda_device(torch.cuda.device_count())}"
        if device is not None:
            set_device(device)

        selected_device = torch.device(get_device())
        selected_backend = selected_device.type if backend in (None, "auto") else backend
        use_encoded = (
            read is None
            and selected_device.type in {"cuda", "mps"}
            and selected_backend == selected_device.type
            and det_bin in (None, 1)
        )
        if use_encoded:
            from quantem.gpu import io as gpu_io

            sources = []
            try:
                for path in paths:
                    sources.append(
                        gpu_io.load(
                            path,
                            backend=selected_backend,
                            representation="encoded",
                            dtype="native",
                            apply_mask=False,
                            **(
                                {"device": selected_device.index or 0}
                                if selected_device.type == "cuda"
                                else {}
                            ),
                            verbose=False,
                        )
                    )
            except BaseException:
                for source in sources:
                    source.close()
                raise
            return cls.from_resident(
                sources,
                device=str(selected_device),
                _owns_sources=True,
            )

        if read is None:
            from quantem.gpu import io as gpu_io

            def read(path):
                # det_bin / backend let a small box (Mac MPS 24 GB, or a no-GPU CPU
                # machine) bin the detector and force a non-cuda decode so the merge
                # fits. Default (None) keeps the full-res cuda path bit-identical.
                kw = {"verbose": False, "representation": "dense"}
                if det_bin is not None:
                    kw["det_bin"] = det_bin
                if backend is not None:
                    kw["backend"] = backend
                data = gpu_io.load(path, **kw).data
                if torch.is_tensor(data):
                    return data
                if hasattr(data, "__dlpack__"):
                    return torch.from_dlpack(data)
                if hasattr(data, "chunks") and hasattr(data, "scan_shape"):
                    scan_rows, scan_cols = data.scan_shape
                    n_frames, k_row, k_col = data.shape
                    target = device if device is not None else get_device()
                    frames = torch.empty(
                        (n_frames, k_row, k_col),
                        dtype=torch.uint16,
                        device=target,
                    )
                    offset = 0
                    for chunk in data.chunks:
                        n_chunk = int(chunk.shape[0])
                        frames[offset : offset + n_chunk].copy_(torch.from_numpy(chunk))
                        offset += n_chunk
                    if hasattr(data, "free"):
                        data.free()
                    return frames.view(scan_rows, scan_cols, k_row, k_col)
                raise TypeError(
                    "quantem.gpu.io.load returned data that cannot be converted "
                    "to a torch.Tensor. Pass a custom read= callable to "
                    "MAPEDTorch.from_files()."
                )

        return cls(
            datasets=_TiltFiles(paths, read),
            _token=cls._token,
            device=get_device(),
            dtype=torch.uint16,
        )

    @classmethod
    def from_resident(
        cls,
        sources: Sequence,
        *,
        device: str = "cuda:0",
        apply_mask: bool = True,
        _owns_sources: bool = False,
    ) -> Self:
        """Build MAPED over encoded native-count tilts on one accelerator.

        Every source stays resident and caller-owned. MAPED computes directly from
        its exact encoding and does not materialize a complete dense tilt. Close
        borrowed sources only after the final MAPED or detector consumer finishes.

        Parameters
        ----------
        sources
            Loaded ``quantem.gpu.io.FourDSTEMData`` objects with encoded or packed
            uint8/uint16 counts and identical native four-dimensional shapes.
        device
            Device containing every source and running the merge.
        apply_mask
            Honor each source's recorded detector policy during GPU computation.
            Loader-corrected pixels remain valid; uncorrected stored-mask pixels are
            excluded. This compatibility switch must remain ``True``.

        Returns
        -------
        Self
            MAPED with reusable resident tilt owners.

        Examples
        --------
        >>> maped = MAPEDTorch.from_resident(tilts, device="cuda:0")
        >>> maped.preprocess(plot_summary=False)
        >>> maped.diffraction_origin(); maped.diffraction_align()
        >>> maped.real_space_align()
        >>> merged = maped.merge_datasets(save_to="merged_master.h5")
        """
        from quantem.core.config import set_device
        from quantem.gpu.io import FourDSTEMData

        sources = list(sources)
        if not sources:
            raise ValueError("Provide at least one encoded tilt from quantem.gpu.io.")
        if not apply_mask:
            raise ValueError(
                "Resident MAPED always honors each source's recorded detector-mask "
                "policy."
            )
        selected = torch.device(device)
        if selected.type not in {"cuda", "mps"} or (
            selected.type == "cuda" and selected.index is None
        ):
            raise ValueError("Specify a resident accelerator such as 'cuda:0' or 'mps'.")
        for loaded in sources:
            if (
                not isinstance(loaded, FourDSTEMData)
                or loaded.representation.value not in {"encoded", "packed"}
            ):
                raise TypeError(
                    "Load native counts with representation='encoded' or 'packed'."
                )
            source_device = getattr(
                loaded.data, "device", getattr(loaded.data, "_device_id", None)
            )
            expected_device = selected.index if selected.type == "cuda" else selected
            if loaded.shape != sources[0].shape or source_device != expected_device:
                raise ValueError("Resident tilts must share their shape and the requested device.")
        set_device(device)
        return cls(
            datasets=_ResidentTilts(sources, owns_sources=_owns_sources),
            dtype=torch.uint16,
            device=device,
            _token=cls._token,
        )

    def preprocess(
        self,
        plot_summary: bool = True,
        scale: float | Sequence[float] | None = None,
        **plot_kwargs: Any,
    ) -> MAPED:
        """
        Compute dataset summary images.

        Parameters
        ----------
        plot_summary : bool, optional
            If True, display summary plots (default True).
        scale : float or sequence of float or None, optional
            Per-dataset scaling factor(s) (default None).

        Attributes
        ----------
        scales : torch.tensor
            Per-dataset scaling factors (n,).
        dp_mean : list[torch.tensor]
            Mean diffraction patterns (H, W), one per dataset.
        im_bf : list[torch.tensor]
            Mean bright-field images (R, C), one per dataset.

        Returns
        -------
        MAPED
            self (updated instance)
        """
        n = len(self.datasets)

        # Scales are a per-tilt multiplicative factor, so they are float even when
        # the tilts themselves are a native integer dtype. Building them as uint16
        # (self.dtype) also left `scales == 0` with no MPS kernel
        # (eq_dense_scalar_cast_bool_ushort), which broke preprocess on Apple GPUs.
        if scale is None:
            self.scales = torch.ones(n, dtype=torch.float32, device=self.device)
        elif isinstance(scale, (int, float, np.floating)):
            self.scales = torch.full(
                (n,), float(scale), dtype=torch.float32, device=self.device
            )
        else:
            self.scales = torch.tensor(scale, dtype=torch.float32, device=self.device)
            if self.scales.shape != (n,):
                raise ValueError(
                    "scale must be a scalar or a sequence with the same length as datasets."
                )
        if torch.any(self.scales == 0):
            raise ValueError("scale entries must be nonzero.")

        self.dp_mean: list[torch.Tensor] = []
        self.im_bf: list[torch.Tensor] = []

        if isinstance(self.datasets, _ResidentTilts):
            self.dp_mean, self.im_bf = _resident_summaries(
                self.datasets.sources, self.device
            )
        else:
            for d in self.datasets:
                d = d.to(self.device)
                rows, cols, k_row, k_col = d.shape
                if (
                    torch.device(self.device).type == "mps"
                    and d.numel() > _MAX_MPS_FUSED_ELEMENTS
                ):
                    dp_sum = torch.zeros((k_row, k_col), dtype=torch.float32, device=self.device)
                    im_bf = torch.empty((rows, cols), dtype=torch.float32, device=self.device)
                    for r0 in range(0, rows, _MPS_PREPROCESS_ROW_BAND):
                        r1 = min(r0 + _MPS_PREPROCESS_ROW_BAND, rows)
                        band = d[r0:r1].contiguous()
                        band = band if band.is_floating_point() else band.float()
                        flat = band.reshape((r1 - r0) * cols, k_row * k_col)
                        dp_sum += flat.sum(0).reshape(k_row, k_col)
                        im_bf[r0:r1] = flat.mean(1).reshape(r1 - r0, cols)
                        del band, flat
                    self.dp_mean.append(dp_sum / float(rows * cols))
                    self.im_bf.append(im_bf)
                    del dp_sum
                else:
                    d = d if d.is_floating_point() else d.float()
                    flat = d.reshape(rows * cols, k_row * k_col)
                    self.dp_mean.append(flat.mean(0).reshape(k_row, k_col))
                    self.im_bf.append(flat.mean(1).reshape(rows, cols))
                    del flat
                del d

        if plot_summary:
            tiles = [[(self.im_bf[i] / self.scales[i]), self.dp_mean[i]] for i in range(n)]
            titles = [
                [f"{i} - Mean Bright Field", f"{i} - Mean Diffraction Pattern"] for i in range(n)
            ]
            show_2d(tiles, title=titles, **plot_kwargs)

        return self

    def diffraction_origin(
        self,
        origins: tuple | list | None = None,
        sigma: float | None = None,
        plot_origins: bool = True,
        plot_indices: list | None = None,
        **plot_kwargs: Any,
    ) -> MAPED:
        """
        Choose or automatically find the origin in diffraction space.

        Parameters
        ----------
        origins : tuple or list, optional
            Optional manual origins. Can be:
            - a single (row, col) tuple, applied to all datasets
            - a list of (row, col) tuples of length n (one per dataset)
        sigma : float, optional
            Optional low-pass smoothing sigma (pixels) applied to each mean DP prior to peak finding.
        plot_origins : bool, optional
            If True, plot mean diffraction patterns with overlaid origin markers.
        plot_indices : list, optional
            Optional indices to plot. If None, plots all datasets.
        **plot_kwargs
            Passed to show_2d.

        Attributes
        ----------
        diffraction_origins : np.ndarray
            Array of shape (n, 2) with integer (row, col) origins.

        Returns
        -------
        MAPED
            self (updated instance)
        """
        n = len(self.datasets)
        if not hasattr(self, "dp_mean"):
            raise RuntimeError("Run preprocess() first so self.dp_mean exists.")

        if plot_indices is None:
            plot_indices_list = list(range(n))
        else:
            plot_indices_list = list(plot_indices)
            for i in plot_indices_list:
                if i < 0 or i >= n:
                    raise IndexError("plot_indices contains an out-of-range index.")

        if sigma is not None and float(sigma) > 0:
            gaussian_filter_torch = torchvision.transforms.GaussianBlur(
                kernel_size=[2 * int(2 * float(sigma)) + 1, 2 * int(2 * float(sigma)) + 1],
                sigma=[sigma, sigma],
            )

            dp_means_use = gaussian_filter_torch(torch.stack(self.dp_mean))
        else:
            dp_means_use = torch.stack(self.dp_mean)

        if origins is None:
            origins_arr = torch.zeros((n, 2), dtype=torch.int)
            for i in range(n):
                dp_use = dp_means_use[i]

                r, c = torch.unravel_index(torch.argmax(dp_use), dp_use.shape)
                origins_arr[i, 0] = int(r)
                origins_arr[i, 1] = int(c)
        else:
            if isinstance(origins, tuple) and len(origins) == 2:
                origins_arr = torch.tile(
                    torch.tensor(origins, dtype=torch.int, device=self.device)[None, :], (n, 1)
                )
            else:
                origins_list = list(origins)
                if len(origins_list) != n:
                    raise ValueError(
                        "origins must be a single (row,col) tuple or a list of length n."
                    )
                origins_arr = torch.tensor(origins_list, dtype=torch.int, device=self.device)
                if origins_arr.shape != (n, 2):
                    raise ValueError("origins must have shape (n, 2) after conversion.")

        self.diffraction_origins = origins_arr

        if plot_origins:
            arrays = [np.asarray(self.dp_mean[i].cpu()) for i in plot_indices_list]
            titles = [f"{i} - Mean Diffraction Pattern" for i in plot_indices_list]
            fig, ax = show_2d(arrays, title=titles, returnfig=True, **plot_kwargs)
            axs = np.ravel(np.asarray(ax, dtype=object))
            for j, i in enumerate(plot_indices_list):
                r, c = self.diffraction_origins[i].cpu().numpy()
                axs[j].plot([c], [r], marker="+", color="red", markersize=16, markeredgewidth=2)

        return self

    def dscan_align(
        self,
        iterations: int,
        upsample_factor: int = 100,
        method: str = "autocorrelation",
        plot: bool = True,
        edge_blend: float = 2.0,
        fit_shifts: bool = True,
        mode: str = "linear",
        batch_size: int | None = None,
    ):
        for i, dataset in enumerate(self.datasets):
            _, aligned_dataset = dscan_correct(
                dataset,
                iterations,
                method=method,
                upsample_factor=upsample_factor,
                plot=plot,
                edge_blend=edge_blend,
                device=self.device,
                fit_shifts=fit_shifts,
                mode=mode,
                batch_size=batch_size,
            )
            self.datasets[i] = aligned_dataset

        return self

    def diffraction_align(
        self,
        edge_blend: float = 16.0,
        padding=None,
        pad_val: str | float = "min",
        upsample_factor: int = 100,
        weight_scale: float = 1 / 8,
        plot_aligned: bool = True,
        **plot_kwargs: Any,
    ) -> MAPED:
        """
        Align mean diffraction patterns using weighted cross-correlation in Fourier space.

        Parameters
        ----------
        edge_blend : float
            Tukey window edge taper (pixels).
        padding : int or tuple, optional
            Passed to shift_images for plotting.
        pad_val : str or float
            Passed to shift_images for plotting.
        upsample_factor : int
            Subpixel upsampling factor for correlation peak estimation.
        weight_scale : float
            Radial weight falloff scale (fraction of mean DP size).
        plot_aligned : bool
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
        if not hasattr(self, "dp_mean"):
            raise RuntimeError("Run preprocess() first so self.dp_mean exists.")
        if not hasattr(self, "diffraction_origins"):
            raise RuntimeError(
                "Run diffraction_origin() first so self.diffraction_origins exists."
            )

        H, W = self.dp_mean[0].shape

        w = (
            tukey_torch(
                H,
                alpha=2.0 * float(edge_blend) / float(H),
                device=self.device,
                dtype=torch.float32,
            )[:, None]
            * tukey_torch(
                W,
                alpha=2.0 * float(edge_blend) / float(W),
                device=self.device,
                dtype=torch.float32,
            )[None, :]
        )

        r = torch.fft.fftfreq(H, 1.0 / float(H), device=self.device)[:, None]
        c = torch.fft.fftfreq(W, 1.0 / float(W), device=self.device)[None, :]

        n = len(self.dp_mean)
        self.diffraction_shifts = torch.zeros((n, 2), device=self.device, dtype=torch.float32)

        G_ref = torch.fft.fft2(w * self.dp_mean[0])
        xy0 = self.diffraction_origins[0]

        kr = torch.fft.fftfreq(H, device=self.device)[:, None]
        kc = torch.fft.fftfreq(W, device=self.device)[None, :]

        for ind in range(1, n):
            G = torch.fft.fft2(w * self.dp_mean[ind])
            xy = self.diffraction_origins[ind]

            dr2 = (r - xy0[0] + xy[0]) ** 2 + (c - xy0[1] + xy[1]) ** 2
            im_weight = torch.clip(
                1.0
                - torch.sqrt(dr2)
                / ((H + W) / 2.0)
                / float(weight_scale),
                0.0,
                1.0,
            )
            im_weight = torch.sin(im_weight * torch.pi / 2.0) ** 2
            shift_rc = cross_correlation_shift_torch(  # not torchified yet
                im_ref=G_ref,
                im=G,
                # weight_real=im_weight * 0.0 + 1.0,
                upsample_factor=int(upsample_factor),
                fft_input=True,
            )

            phase_ramp = torch.exp(-2j * torch.pi * (kr * shift_rc[0] + kc * shift_rc[1]))

            G_shift = G * phase_ramp
            self.diffraction_shifts[ind, :] = shift_rc.clone()

            G_ref = G_ref * (ind / (ind + 1)) + G_shift / (ind + 1)

        self.diffraction_shifts -= torch.mean(self.diffraction_shifts, dim=0)[None, :]
        if plot_aligned:
            im_aligned = shift_images_torch(
                images=torch.stack(self.dp_mean),
                shifts_rc=self.diffraction_shifts,
                edge_blend=float(edge_blend),
                padding=padding,
                pad_val=pad_val,
            )
            show_2d(im_aligned.unbind(0), **plot_kwargs)

        return self

    def real_space_align(
        self,
        num_images=None,
        num_iter: int = 3,
        edge_blend: float = 1.0,
        padding=None,
        pad_val: str | float = "median",
        upsample_factor: int = 100,
        max_shift=None,
        shift_method: str = "bilinear",
        edge_filter: bool = True,
        edge_sigma: float = 2.0,
        hanning_filter: bool = False,
        plot_aligned: bool = True,
        **plot_kwargs: Any,
    ) -> MAPED:
        """
        Align real-space mean BF images using iterative average-reference correlation.

        Parameters
        ----------
        num_images : int, optional
            If provided, align only the first num_images images.
        num_iter : int
            Number of refinement iterations.
        edge_blend : float
            Used to set default correlation padding when max_shift is None.
        padding : int or tuple, optional
            Passed to shift_images for plotting.
        pad_val : float
            Passed to shift_images for plotting.
        upsample_factor  : int
            Subpixel upsampling factor for correlation peak estimation.
        max_shift : float
            Optional maximum shift constraint passed to weighted_cross_correlation_shift.
        shift_method : 'bilinear' or 'fourier'
            Passed to shift_images for plotting ('bilinear' or 'fourier').
        edge_filter : bool
            If True, correlate on gradient magnitude instead of raw intensity.
        edge_sigma : float
            Gaussian sigma applied to gradients when edge_filter is True.
        hanning_filter : bool
            If True, apply a Hanning window prior to FFT.
        plot_aligned : bool
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
        if not hasattr(self, "im_bf"):
            raise RuntimeError("Run preprocess() first so self.im_bf exists.")
        if len(self.im_bf) == 0:
            raise RuntimeError("No images found in self.im_bf.")

        H, W = self.im_bf[0].shape
        for im in self.im_bf:
            if im.shape != (H, W):
                raise ValueError("all self.im_bf images must have the same shape")

        n_total = len(self.im_bf)
        if num_images is None:
            n = n_total
        else:
            n = int(num_images)
            if n <= 0:
                raise ValueError("num_images must be positive")
            n = min(n, n_total)

        if int(num_iter) < 1:
            raise ValueError("num_iter must be >= 1")

        if max_shift is not None:
            pad_cc = int(np.ceil(float(max_shift))) + 4
        else:
            pad_cc = int(np.ceil(float(edge_blend))) + 4

        Hp = H + 2 * pad_cc
        Wp = W + 2 * pad_cc
        r0 = pad_cc
        c0 = pad_cc

        w_h = torch.ones((H, W), dtype=torch.float32, device=self.device)
        if hanning_filter:
            w_h = (
                torch.hann_window(H, dtype=torch.float32, device=self.device)[:, None]
                * torch.hann_window(W, dtype=torch.float32, device=self.device)[None, :]
            )
        w_h_pad = torch.zeros((Hp, Wp), dtype=torch.float32, device=self.device)
        w_h_pad[r0 : r0 + H, c0 : c0 + W] = w_h
        w_h_sum = torch.sum(w_h_pad)
        if w_h_sum <= 0:
            raise RuntimeError("hanning window sum is zero")

        if edge_filter:
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

            if edge_filter:
                pad_symmetric = wx.shape[-1] // 2
                im0_pad = F.pad(
                    im0[None, None],
                    pad=(pad_symmetric, pad_symmetric, pad_symmetric, pad_symmetric),
                    mode="reflect",
                )

                gx = F.conv2d(im0_pad, wx[None, None])[0, 0]
                gy = F.conv2d(im0_pad, wx.T[None, None])[0, 0]

                gaussian_filt = torchvision.transforms.GaussianBlur(
                    kernel_size=[
                        2 * int(2 * float(edge_sigma)) + 1,
                        2 * int(2 * float(edge_sigma)) + 1,
                    ],
                    sigma=[edge_sigma, edge_sigma],
                )
                gx = gaussian_filt(gx[None])
                gy = gaussian_filt(gy[None])
                im_use = torch.sqrt(gx * gx + gy * gy)
            else:
                im_use = im0

            base_pad[i, r0 : r0 + H, c0 : c0 + W] = im_use

        shifts = torch.zeros((n, 2), dtype=torch.float32, device=self.device)

        for _ in range(int(num_iter)):
            # shift images to current guess
            ims_a = shift_images_torch(base_pad, shifts)
            ims_mean = torch.sum(ims_a * w_h_pad, dim=(1, 2)) / w_h_sum

            ims_win = (ims_a - ims_mean[:, None, None]) * w_h_pad[None]
            G_list = torch.fft.fft2(ims_win)

            G_ref = torch.mean(G_list, dim=0)

            # perform cross correlation again
            for i in range(1, n):
                drc = cross_correlation_shift_torch(
                    im_ref=G_ref,
                    im=G_list[i],
                    # weight_real=None,
                    upsample_factor=int(upsample_factor),
                    # max_shift=max_shift,
                    fft_input=True,
                    # fft_output=False,
                    # return_shifted_image=False,
                )

                shifts[i, 0] += float(drc[0])
                shifts[i, 1] += float(drc[1])

            shifts -= shifts[0:1].clone()

        shifts -= torch.mean(shifts, dim=0)[None, :]

        self.real_space_shifts = torch.zeros((n_total, 2), dtype=torch.float32, device=self.device)
        self.real_space_shifts[:n, :] = shifts

        if plot_aligned:
            im_aligned = shift_images_torch(
                images=torch.stack(self.im_bf[:n]),
                shifts_rc=self.real_space_shifts[:n, :],
                edge_blend=float(edge_blend),
                padding=padding,
                pad_val=pad_val,
                mode=shift_method,
                blend=True,
            )
            show_2d(im_aligned, **plot_kwargs)

        return self

    def merge_datasets(
        self,
        real_space_padding: int = 0,
        real_space_edge_blend: float = 1.0,
        diffraction_padding: int = 0,
        diffraction_edge_blend: float = 0.0,
        diffraction_pad_val: str | float = "min",
        shift_method: str = "bilinear",
        dtype=None,
        save_to: str | Path | None = None,
        scale_output: bool = False,
        plot_result: bool = True,
        verbose: bool = True,
        batch_size: int | None = None,
        cast_dtype: torch.dtype | None = None,
        accumulator_device: str | torch.device | None = None,
        compile_merge: bool | None = None,
        compile_uint16_as_int16: bool = True,
        compute_summaries: bool = True,
        prefetch_tilts: bool = False,
        profile_timings: dict[str, Any] | None = None,
        scan_region: tuple[int, int, int, int] | None = None,
        **plot_kwargs: Any,
    ) -> Any:
        """
        Merge aligned datasets into a single Dataset4dstem.

        Notes
        -----
        Requires the following attributes to be present on ``self``:

        self.real_space_shifts
            From ``real_space_align()``.
        self.diffraction_shifts
            From ``diffraction_align()``.

        Parameters
        ----------
        real_space_padding : int
            Output scan padding in pixels (adds border to scan grid).
        real_space_edge_blend : float
            Tukey taper width for scan-space interpolation weights.
        diffraction_padding : int
            Output diffraction padding in pixels (adds border around DPs).
        diffraction_edge_blend : float
            Tukey taper width for diffraction-space weights.
        diffraction_pad_val : str | float
            Pad value for diffraction padding ('min','max','mean','median' or float).
        shift_method : str
            How each tilt's diffraction patterns are sub-pixel shifted to align
            their direct beams before averaging. 'bilinear' (default) interpolates
            in real space - no ringing, slight blur. 'fourier' is exact sub-pixel
            but its sinc kernel rings on the sharp direct-beam disk, leaving a faint
            dotted 'cross' through the center of the merged mean diffraction pattern.
        dtype : str or torch.dtype, optional
            Output storage dtype. ``"scaled_uint16"`` computes the merge in
            float32 once, then retains calibrated, packed uint16 regions on
            the GPU. Calibration and region sizes are automatic. Saving uses
            this storage by default. Float32 scan-region inspection remains available.
        save_to : str, optional
            Output HDF5 path for resident encoded sources. The merge is written in
            bounded regions with their intensity calibration. The already-resident
            packed result is retained for viewing without reopening the file.
            Omit this for resident ``scaled_uint16`` output or a small float32
            ``scan_region`` inspection.
        scale_output : bool
            If True and dtype is integer, scale to full dynamic range using global max.
        plot_result : bool
            If True, plot merged BF and merged mean DP.
        batch_size : int, optional
            Number of rows to process per batch. If None, auto-sized from free VRAM
            (8-48 rows) so the merge fits the card without tuning.
        cast_dtype : torch.dtype, optional
            dtype the streamed tilts are cast to for sub-pixel warping. If None, uses
            the parent float dtype.
        accumulator_device : str or torch.device, optional
            Where the float32 output accumulator lives. If None, auto-picks: the
            compute device when it fits, else CPU RAM (out-of-core) for a small-VRAM
            card. Pass 'cpu' or a second GPU to force it.
        compile_merge : bool, optional
            Compile experimental merge kernels with ``torch.compile``. If None, this
            fuses large interior scan regions for encoded MPS inputs. It also
            enables MPS Fourier-shift merges. Other dense and streamed bilinear
            paths remain eager by default.
        compile_uint16_as_int16 : bool
            When compiling a uint16 tilt, convert it to int16 after verifying the
            counts fit in int16. Inductor does not support uint16, and int16 is
            lossless for low-count MAPED data.
        compute_summaries : bool
            If True, compute merged BF and mean-DP summaries after merging. Disable
            for latency-critical live viewing when only the merged 4D tensor is
            needed.
        prefetch_tilts : bool
            If True for file-backed streams, load the next tilt on a background
            thread while the current tilt is being accumulated. This can hide HDF5
            decode/disk latency on unified-memory MPS machines, but increases peak
            memory by up to one resident tilt.
        profile_timings : dict, optional
            If provided, populated with synchronized phase timings for profiling.
        scan_region : tuple of int, optional
            Inspect this region of an encoded resident merge without saving:
            ``(row_start, row_stop, column_start, column_stop)``, exclusive stops
            in the full aligned scan coordinates. All resident inputs and the full
            alignment remain available for another inspection or later saving.
            The returned region retains the complete detector and float32
            intensities. Select at most 4096 scan positions for bounded memory.
            This option is only available without ``save_to``.
        **plot_kwargs
            Passed to show_2d.

        Returns
        -------
        Dataset4dstem or quantem.gpu.io.FourDSTEMData
            Merged dataset, or an accelerator-resident selected float32 region.

        Examples
        --------
        Inspect aligned resident inputs before saving the complete result:

        >>> patch = maped.merge_datasets(scan_region=(252, 260, 252, 260), plot_result=False)
        >>> maped.show()
        >>> merged = maped.merge_datasets(save_to="merged_master.h5", plot_result=False)
        """

        if shift_method == "fourier":
            warnings.warn(
                "shift_method='fourier' is exact sub-pixel but leaves a faint dotted "
                "'cross' (sinc ringing) through the center of the merged diffraction "
                "pattern, on the sharp direct-beam disk. The default 'bilinear' avoids "
                "it at the cost of a slight interpolation blur.",
                stacklevel=2,
            )
        if not hasattr(self, "real_space_shifts"):
            raise RuntimeError("Run real_space_align() first so self.real_space_shifts exists.")
        if not hasattr(self, "diffraction_shifts"):
            raise RuntimeError("Run diffraction_align() first so self.diffraction_shifts exists.")

        arrays = self.datasets
        if scan_region is not None and not isinstance(arrays, _ResidentTilts):
            raise ValueError(
                "scan_region inspection requires encoded resident inputs; "
                "load them with MAPEDTorch.from_files()."
            )
        n = len(arrays)
        if n == 0:
            raise RuntimeError("No datasets found in self.datasets.")
        _profile_t0 = time.perf_counter()

        def _profile_now() -> float:
            if profile_timings is not None:
                if torch.device(self.device).type == "mps" and torch.backends.mps.is_available():
                    torch.mps.synchronize()
                elif torch.device(self.device).type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
            return time.perf_counter()

        def _profile_elapsed(start: float) -> float:
            return _profile_now() - start

        # Shapes come from the preprocess summaries already in memory - im_bf[i] is
        # (Rs, Cs) and dp_mean[i] is (H, W) - NOT by re-reading a tilt. For file-backed
        # tilts (from_files) reloading every 19 GB tilt just to read .shape would
        # dominate the merge. Only validate cross-tilt shape agreement for in-memory
        # lists, where it is free; for streaming the per-tilt grid_sample fails loudly
        # on a mismatch anyway.
        Rs, Cs = self.im_bf[0].shape
        H, W = self.dp_mean[0].shape
        if not isinstance(arrays, _TiltFiles):
            for a in arrays:
                if a.shape != (Rs, Cs, H, W):
                    raise ValueError("All dataset arrays must have the same shape (Rs, Cs, H, W).")

        rs_shifts = self.real_space_shifts
        dp_shifts = self.diffraction_shifts
        if rs_shifts.shape != (n, 2):
            raise ValueError("self.real_space_shifts must have shape (n, 2).")
        if dp_shifts.shape != (n, 2):
            raise ValueError("self.diffraction_shifts must have shape (n, 2).")

        if isinstance(arrays, _ResidentTilts):
            unsupported = []
            if int(real_space_padding) != 0:
                unsupported.append("real_space_padding=0")
            if float(real_space_edge_blend) != 1.0:
                unsupported.append("real_space_edge_blend=1")
            if int(diffraction_padding) != 0:
                unsupported.append("diffraction_padding=0")
            if float(diffraction_edge_blend) != 0.0:
                unsupported.append("diffraction_edge_blend=0")
            if str(shift_method).strip().lower() != "bilinear":
                unsupported.append("shift_method='bilinear'")
            if save_to is None:
                if dtype not in (None, "float32", torch.float32, "scaled_uint16"):
                    unsupported.append("dtype='float32' or 'scaled_uint16'")
            elif dtype not in (None, "scaled_uint16"):
                unsupported.append("dtype='scaled_uint16'")
            if scale_output:
                unsupported.append("scale_output=False")
            if unsupported:
                raise ValueError(
                    "The encoded resident merge currently requires "
                    + ", ".join(unsupported)
                    + "."
                )
            from quantem.gpu import io as gpu_io

            from ._maped_resident import ResidentMergeSource

            if save_to is None and dtype != "scaled_uint16":
                region = (0, Rs, 0, Cs) if scan_region is None else scan_region
                if (
                    len(region) != 4
                    or any(not isinstance(value, (int, np.integer)) for value in region)
                    or not (0 <= region[0] < region[1] <= Rs)
                    or not (0 <= region[2] < region[3] <= Cs)
                ):
                    raise ValueError(
                        f"scan_region={region} must be (row_start, row_stop, "
                        f"column_start, column_stop) inside {(Rs, Cs)}."
                    )
                row0, row1, column0, column1 = map(int, region)
                if (row1 - row0) * (column1 - column0) > 4096:
                    raise ValueError(
                        "A full float32 merge is too large for bounded inspection. "
                        "Select scan_region with at most 4096 scan positions, or "
                        "provide save_to='merged_master.h5' for the complete output."
                    )
                generated = ResidentMergeSource(
                    arrays.sources, rs_shifts, dp_shifts,
                    close_sources_before_reopen=False,
                )
                try:
                    parts = list(generated.blocks((row0, row1, column0, column1)))
                    values = parts[0] if len(parts) == 1 else torch.cat(parts)
                    values = values.reshape(row1 - row0, column1 - column0, H, W)
                    metadata = {
                        key.removeprefix("quantem_").removesuffix("_v1"): json.loads(value)
                        for key, value in generated.save_metadata.items()
                    }
                    metadata.update(
                        representation="dense", residency="device",
                        working_shape=tuple(values.shape), working_dtype="float32",
                    )
                    metadata["maped_merge"]["scan_region"] = list(region)
                    result = gpu_io.FourDSTEMData(values, metadata)
                finally:
                    generated.close()
                self.merged = result
                if compute_summaries or plot_result:
                    self.im_bf_merged = values.mean(dim=(-2, -1))
                    self.dp_mean_merged = values.flatten(0, 1).mean(dim=0)
                else:
                    self.im_bf_merged = self.dp_mean_merged = None
                if profile_timings is not None:
                    profile_timings["total_profiled_seconds"] = _profile_elapsed(_profile_t0)
                if plot_result:
                    show_2d(
                        [[self.im_bf_merged, self.dp_mean_merged]],
                        title=[["Merged Region Bright Field", "Merged Region Mean Diffraction Pattern"]],
                        **plot_kwargs,
                    )
                return result
            if scan_region is not None:
                raise ValueError(
                    "scan_region selects float32 inspection; omit it for the "
                    "complete scaled_uint16 result."
                )
            generated = ResidentMergeSource(
                arrays.sources,
                rs_shifts,
                dp_shifts,
                close_sources_before_reopen=False,
                compile_merge=compile_merge,
            )
            result = gpu_io.load(
                generated,
                dtype="scaled_uint16",
                backend=torch.device(self.device).type,
                verbose=verbose,
            )
            if save_to is not None:
                try:
                    gpu_io.save(
                        save_to, result, backend=torch.device(self.device).type,
                        verbose=verbose,
                    )
                except BaseException:
                    result.close()
                    raise
            if arrays.owns_sources:
                for source in arrays.sources:
                    source.close()
                arrays.sources = []
                if torch.device(self.device).type == "cuda":
                    torch.cuda.empty_cache()
                else:
                    torch.mps.empty_cache()
            self.merged = result
            if compute_summaries or plot_result:
                summaries_dp, summaries_bf = _resident_summaries([result], self.device)
                self.dp_mean_merged = summaries_dp[0]
                self.im_bf_merged = summaries_bf[0]
            else:
                self.dp_mean_merged = None
                self.im_bf_merged = None
            if profile_timings is not None:
                profile_timings.update(result.metadata.get("maped_merge", {}))
                profile_timings["total_profiled_seconds"] = time.perf_counter() - _profile_t0
            if plot_result:
                show_2d(
                    [[self.im_bf_merged, self.dp_mean_merged]],
                    title=[["Merged Bright Field", "Merged Mean Diffraction Pattern"]],
                    **plot_kwargs,
                )
            return result

        if dtype is None:
            # The merged dataset is an interpolated + accumulated quantity, so it
            # is inherently floating point. When the inputs are kept in a native
            # integer dtype (e.g. uint16, to halve VRAM vs float32), default the
            # output to float32 instead of rounding the result back to the integer
            # parent dtype. Float inputs keep their own dtype unchanged. Read the tilt
            # dtype from self.dtype for file-backed tilts (no reload); a plain list is
            # already in memory so arrays[0].dtype is free.
            dtype_out = self.dtype if isinstance(arrays, _TiltFiles) else arrays[0].dtype
            if not torch.empty(0, dtype=dtype_out, device=self.device).is_floating_point():
                dtype_out = torch.float32
        else:
            dtype_out = dtype

        real_space_padding = int(real_space_padding)
        diffraction_padding = int(diffraction_padding)

        Rout = Rs + 2 * real_space_padding
        Cout = Cs + 2 * real_space_padding

        Hp = H + 2 * diffraction_padding
        Wp = W + 2 * diffraction_padding
        rp0 = diffraction_padding
        cp0 = diffraction_padding

        method = str(shift_method).strip().lower()
        if method not in {"bilinear", "fourier"}:
            raise ValueError("shift_method must be 'bilinear' or 'fourier'.")

        # set up real space edge blending weights
        if real_space_edge_blend and float(real_space_edge_blend) > 0:
            alpha_r = min(1.0, 2.0 * float(real_space_edge_blend) / float(Rs))
            alpha_c = min(1.0, 2.0 * float(real_space_edge_blend) / float(Cs))
            w_rs = (
                tukey_torch(Rs, alpha=alpha_r, device=self.device, dtype=torch.float32)[:, None]
                * tukey_torch(Cs, alpha=alpha_c, device=self.device, dtype=torch.float32)[None, :]
            )
        else:
            w_rs = torch.ones((Rs, Cs), dtype=torch.float32, device=self.device)

        # set up diffraction space edge blending weights
        if diffraction_edge_blend and float(diffraction_edge_blend) > 0:
            alpha_dr = min(1.0, 2.0 * float(diffraction_edge_blend) / float(H))
            alpha_dc = min(1.0, 2.0 * float(diffraction_edge_blend) / float(W))
            w_dp = (
                tukey_torch(H, alpha=alpha_dr, device=self.device, dtype=torch.float32)[:, None]
                * tukey_torch(W, alpha=alpha_dc, device=self.device, dtype=torch.float32)[None, :]
            )
        else:
            w_dp = torch.ones((H, W), dtype=torch.float32, device=self.device)

        v = torch.stack(self.dp_mean, dim=0).flatten()

        if isinstance(diffraction_pad_val, str):
            s = diffraction_pad_val.strip().lower()
            if s == "min":
                pad_val_dp = float(torch.min(v))
            elif s == "max":
                pad_val_dp = float(torch.max(v))
            elif s == "mean":
                pad_val_dp = float(torch.mean(v))
            elif s == "median":
                pad_val_dp = float(torch.median(v))
            else:
                raise ValueError(
                    "diffraction_pad_val must be a float or one of {'min','max','mean','median'}."
                )
        else:
            pad_val_dp = float(diffraction_pad_val)

        wdp_pad = torch.zeros((Hp, Wp), dtype=torch.float32, device=self.device)
        wdp_pad[rp0 : rp0 + H, cp0 : cp0 + W] = w_dp

        wdp_shifted = torch.zeros((n, Hp, Wp), dtype=torch.float32, device=self.device)
        if method == "fourier":
            kr = torch.fft.fftfreq(Hp, device=self.device)[:, None]
            kc = torch.fft.fftfreq(Wp, device=self.device)[None, :]
            Fw = torch.fft.fft2(wdp_pad)
            ramps: list[torch.Tensor] = []
            for i in range(n):
                dr, dc = dp_shifts[i, 0], dp_shifts[i, 1]

                ramp = torch.exp(-2j * torch.pi * (kr * dr + kc * dc))
                ramps.append(ramp)
                w_i = torch.fft.ifft2(Fw * ramp).real
                wdp_shifted[i] = torch.clip(w_i, 0.0, 1.0)
        else:
            for i in range(n):
                w_i = shift_images_torch(
                    wdp_pad,
                    shifts_rc=dp_shifts[i, :],
                    mode="bilinear",
                )
                wdp_shifted[i] = torch.clip(w_i, 0.0, 1.0)

        coverage = torch.clip(torch.sum(wdp_shifted, dim=0), 0.0, 1.0)
        edge_w_dp = 1.0 - coverage

        # Where the accumulator lives - auto-picked from free VRAM. If the full
        # float32 accumulator (Rout*Cout*Hp*Wp*4, ~38.6 GB at no-bin) plus one
        # streamed uint16 tilt won't fit the compute GPU, put the accumulator in
        # CPU RAM (the _split path): one tilt stays on the GPU (19 GB fits a 24 GB
        # card), the 38.6 GB output lives in RAM. A big card (96 GB) keeps both on
        # the GPU - bit-identical to before. Pass accumulator_device explicitly
        # ('cpu' or another GPU) to override the auto-pick.
        row_elems = Cout * Hp * Wp
        acc_bytes = Rout * row_elems * 4
        tilt_bytes = Rs * Cs * H * W * 2
        _dev = torch.device(self.device)
        _free_bytes = None
        if _dev.type == "cuda" and torch.cuda.is_available():
            # Reclaim what preprocess/align left cached (and free the streaming
            # loader's stranded tilt) BEFORE reading free memory - otherwise the
            # budget reads artificially low and both the accumulator pick and the
            # auto batch-size collapse.
            if hasattr(arrays, "release"):
                arrays.release(reclaim_cache=False)
            torch.cuda.empty_cache()
            _free_bytes, _ = torch.cuda.mem_get_info(_dev)
        if accumulator_device is not None:
            _acc_device = accumulator_device
        elif _free_bytes is not None and _free_bytes < (acc_bytes + tilt_bytes) * 1.2:
            _acc_device = "cpu"
            if verbose:
                print(
                    f"  merge: out-of-core - accumulator -> CPU RAM "
                    f"(free {_free_bytes / 1e9:.0f} GB < need "
                    f"{(acc_bytes + tilt_bytes) / 1e9:.0f} GB; one tilt stays on GPU)"
                )
        else:
            _acc_device = self.device
        _split = torch.device(_acc_device) != torch.device(self.device)
        if compile_merge is None:
            compile_merge = (
                torch.device(self.device).type == "mps"
                and hasattr(torch, "compile")
                and method == "fourier"
            )
        compile_merge = bool(compile_merge)

        # Determine batch size. Auto-pick from free VRAM so the merge fits without
        # the caller tuning it. Bigger batches are faster; smaller cut peak memory.
        # The result is bit-identical regardless of batch_size.
        if batch_size is None:
            if _free_bytes is not None:
                # Resident GPU set: one uint16 tilt (always on the compute card) +,
                # when NOT split, the float32 accumulator too. The per-batch buffers
                # (slab cast, FFT pair, dp_padded) are reused across batches, so the
                # peak barely grows with batch size - the complex FFT pair (x8) is
                # the per-row cost. Cap at 48 (the empirically safe no-bin value).
                fixed_bytes = tilt_bytes if _split else acc_bytes + tilt_bytes
                per_row_bytes = row_elems * 8
                budget = _free_bytes - fixed_bytes
                batch_size = max(8, min(48, int(budget / per_row_bytes)))
                if verbose:
                    print(
                        f"  merge: auto batch_size={batch_size} "
                        f"(free {_free_bytes / 1e9:.1f} GB, ~{per_row_bytes / 1e9:.3f} GB/row)"
                    )
            else:
                batch_size = max(1, min(32, Rout // 2))

        c_base = torch.arange(Cout, dtype=torch.float32, device=self.device) - real_space_padding

        # The merge ALWAYS streams: keep a float32 accumulator num (the full output,
        # 38.6 GB at no-bin) on the accumulator card, read ONE uint16 tilt at a time
        # onto the compute card, warp + shift + accumulate it, free it. It never holds
        # all tilts (135 GB) nor a second full den. Data arrives sequentially anyway,
        # so there is no in-memory "hold everything" path. float32 is exact for the
        # integer counts (max ~6764, well under 2^24); only the final divide differs
        # from float64 at ~1e-7 - negligible for count data. accumulator_device='cpu'
        # or a second GPU keeps the 38.6 GB output off the compute card when even that
        # won't fit beside a streamed tilt (the _split path below).

        # File-backed tilts (from_files) leave the last preprocess tilt in memory;
        # release it before allocating the accumulator so the card starts clean. On
        # CUDA this already ran above (before the free-memory read); only the non-CUDA
        # path (no _free_bytes) still needs it here. No-op for in-memory dataset lists.
        if _free_bytes is None and hasattr(arrays, "release"):
            arrays.release()
        # When accumulator_device differs from the compute device (_split), num/den
        # live on the accumulator card and tilts are cast ONE AT A TIME on the compute
        # card - so neither card holds both. Each tilt is cast once (not re-cast per
        # batch), which kills the dominant merge cost.
        num = torch.zeros((Rout, Cout, Hp, Wp), dtype=torch.float32, device=_acc_device)
        # den FACTORIZES: den = sum_i wi_i (x) wdp_i, an outer product of the
        # per-tilt real-space weight (Rout, Cout) and diffraction weight (Hp, Wp).
        # So a full (Rout,Cout,Hp,Wp) den would be a SECOND 38.6 GB accumulator we
        # never need: keep only the tiny per-tilt wi maps (n*Rout*Cout ~ 7 MB) and
        # rebuild den one output-row band at a time at divide. This halves the
        # accumulator (77 -> 38.6 GB) - the second piece (with the uint16 tilts and
        # detector-slab cast) that lets no-bin merge run on a single 96 GB GPU.
        wi_all = torch.zeros((n, Rout, Cout), dtype=torch.float32, device=_acc_device)
        c_base_b = c_base[None]
        w_rs_reshaped = w_rs[None, None]
        _cast = cast_dtype if cast_dtype is not None else torch.float32
        # Page-locked staging for a CPU accumulator: a pageable GPU->CPU copy of each
        # batch's weighted product runs at ~2.7 GB/s and is ~73% of the out-of-core
        # merge time; copying into a pinned buffer instead hits PCIe peak (~20 GB/s),
        # ~7x faster. One buffer sized to the largest batch, reused every batch. A
        # second-GPU accumulator (cross-GPU split) keeps the already-fast .to() path.
        _acc_is_cpu = torch.device(_acc_device).type == "cpu"
        # Only the Metal backend has the fused-multiply-add ceiling; CUDA keeps
        # its addcmul_ so the frozen baseline stays bit-exact.
        _acc_over_int_max = (
            torch.device(_acc_device).type == "mps"
            and Rout * row_elems > _MAX_MPS_FUSED_ELEMENTS
        )
        _compiled_tail = (
            _compiled_merge_tail(method)
            if compile_merge and not _split and method == "fourier"
            else None
        )
        # pin_memory needs a CUDA context; pin only when the compute device is CUDA
        # (the normal out-of-core path). A non-CUDA box that forces
        # accumulator_device='cpu' falls back to a pageable buffer, not a crash.
        _stage = (
            torch.empty(
                (batch_size, Cout, Hp, Wp), dtype=torch.float32,
                pin_memory=_dev.type == "cuda" and torch.cuda.is_available(),
            )
            if _acc_is_cpu
            else None
        )
        if profile_timings is not None:
            profile_timings.clear()
            profile_timings["setup_seconds"] = _profile_elapsed(_profile_t0)
            profile_timings["tilts"] = []
        _tilt_loop_t0 = _profile_now()
        _prefetch_executor: ThreadPoolExecutor | None = None
        _prefetch_future: Future[Any] | None = None
        _use_prefetch = (
            bool(prefetch_tilts)
            and isinstance(arrays, _TiltFiles)
            and torch.device(self.device).type != "mps"
        )
        if prefetch_tilts and isinstance(arrays, _TiltFiles) and not _use_prefetch:
            warnings.warn(
                "prefetch_tilts=True is disabled for MPS because overlapping "
                "background MPS loads with active MPS kernels is not thread-safe "
                "in the current backend. Use a chunk-direct MPS reader instead.",
                RuntimeWarning,
                stacklevel=2,
            )
        if _use_prefetch:
            _prefetch_executor = ThreadPoolExecutor(max_workers=1)
            _prefetch_future = _prefetch_executor.submit(arrays.read, arrays.paths[0])
        try:
            for i in tqdm(range(n), desc="Merging tilts"):
                _tilt_profile: dict[str, float | int | bool] = {"index": i}
                _tilt_t0 = _profile_now()
                # Keep the tilt in native dtype (uint16 = 19.3 GB) on the compute
                # device; the whole 38.6 GB float copy never exists - detector slabs
                # are cast to float inside grid_sample below.
                _load_t0 = _profile_now()
                if _use_prefetch:
                    assert _prefetch_future is not None
                    a_raw = _prefetch_future.result()
                    if i + 1 < n:
                        assert _prefetch_executor is not None
                        _prefetch_future = _prefetch_executor.submit(
                            arrays.read,
                            arrays.paths[i + 1],
                        )
                    else:
                        _prefetch_future = None
                else:
                    a_raw = arrays[i]
                a = a_raw.to(device=self.device)
                if compile_merge and a.dtype == torch.uint16:
                    if not compile_uint16_as_int16:
                        raise ValueError(
                            "compile_merge=True cannot consume uint16 tilts because "
                            "torch.compile/inductor does not support torch.uint16. "
                            "Pass compile_uint16_as_int16=True for count data known to "
                            "fit in int16, or disable compile_merge."
                        )
                    max_count = int(torch.max(a).item())
                    if max_count > torch.iinfo(torch.int16).max:
                        raise ValueError(
                            "compile_merge=True would need to reinterpret a uint16 tilt "
                            f"as int16, but the maximum count is {max_count}, above "
                            "32767. Disable compile_merge or load/cast to a supported "
                            "floating dtype."
                        )
                    a_i16 = a.to(torch.int16)
                    del a
                    a = a_i16
                _tilt_profile["load_to_device_seconds"] = _profile_elapsed(_load_t0)
                _tilt_profile["chunked"] = False
                a_reshaped = a.view(Rs, Cs, H * W).permute(2, 0, 1)[None]
                _band_t0 = _profile_now()
                _fallback_batches = 0
                for batch_start in range(0, Rout, batch_size):
                    batch_end = min(batch_start + batch_size, Rout)
                    batch_rows = torch.arange(
                        batch_start, batch_end, dtype=torch.float32, device=self.device
                    )
                    _fallback_batches += 1
    
                    r_in = (
                        (batch_rows.unsqueeze(1) - real_space_padding).expand(-1, Cout)
                        - rs_shifts[i, 0]
                    )
                    c_in = c_base_b.expand(batch_end - batch_start, -1) - rs_shifts[i, 1]
                    c_norm = 2.0 * c_in / (Cs - 1) - 1.0
                    r_norm = 2.0 * r_in / (Rs - 1) - 1.0
                    grid_full = torch.stack([c_norm, r_norm], dim=-1).unsqueeze(0)
                    dp_sample = _sample_tilt_constant_bilinear(
                        a_reshaped,
                        batch_start,
                        batch_end,
                        Cout,
                        real_space_padding,
                        rs_shifts[i],
                        _cast,
                    )
                    wi_sample = torch.nn.functional.grid_sample(
                        w_rs_reshaped, grid_full,
                        mode="bilinear", padding_mode="zeros", align_corners=True,
                    )
                    dp_interp = dp_sample.squeeze(0).view(
                        H, W, batch_end - batch_start, Cout
                    ).permute(2, 3, 0, 1)
                    wi = wi_sample.squeeze(0).squeeze(0)
                    # dp_interp is already float32 (grid_sample output); the weight
                    # mul stays float32, so no .float() copy. When there is no
                    # diffraction padding (Hp==H), the weighted interp IS the FFT
                    # input - skip allocating + zeroing + scattering a full padded
                    # buffer every batch (pure waste that was overwritten anyway).
                    dp_weighted = dp_interp * w_dp[None, None]
                    if Hp == H and Wp == W:
                        dp_padded = dp_weighted
                    else:
                        dp_padded = torch.zeros(
                            (batch_end - batch_start, Cout, Hp, Wp),
                            dtype=torch.float32, device=self.device,
                        )
                        dp_padded[:, :, rp0 : rp0 + H, cp0 : cp0 + W] = dp_weighted
                    del dp_sample, dp_interp, wi_sample, dp_weighted
                    if _compiled_tail is not None:
                        num[batch_start:batch_end] = _compiled_tail(
                            num[batch_start:batch_end],
                            dp_padded,
                            wi,
                            ramps[i] if method == "fourier" else torch.empty(0, device=self.device),
                            dp_shifts[i : i + 1],
                        )
                        wi_all[i, batch_start:batch_end] = wi.to(_acc_device)
                        del dp_padded, wi
                        continue
    
                    dp_shifted = _shift_diffraction_batch(
                        dp_padded, method,
                        ramps[i] if method == "fourier" else None,
                        dp_shifts[i : i + 1],
                        batch_end - batch_start, Cout, Hp, Wp,
                    )
                    wi_exp = wi[..., None, None]
                    if _split:
                        prod = wi_exp * dp_shifted  # weighted product on the compute device
                        if _acc_is_cpu:
                            # page-locked staging -> ~7x faster GPU->CPU copy than a
                            # pageable .to(cpu); same values, so the accumulate stays
                            # bit-exact vs the in-VRAM addcmul_ on the production path.
                            stg = _stage[: batch_end - batch_start]
                            stg.copy_(prod)
                            num[batch_start:batch_end] += stg
                        else:
                            # cross-GPU: a GPU->GPU peer copy is already fast.
                            num[batch_start:batch_end] += prod.to(_acc_device)
                        del prod  # ~3.6 GB batch buffer - free before the next batch
                    elif _acc_over_int_max:
                        # MPSGraph cannot build a fused multiply-add against an
                        # accumulator holding more than 2**31 elements (a no-bin
                        # output is 9.7e9), though it handles the same shapes as two
                        # separate kernels. Costs one batch-sized temporary.
                        num[batch_start:batch_end] += wi_exp * dp_shifted
                    else:
                        # single device: fuse weight-mul + accumulate into one addcmul_
                        # kernel (no full-size weighted temporary, no identity .to copy).
                        num[batch_start:batch_end].addcmul_(wi_exp, dp_shifted)
                    wi_all[i, batch_start:batch_end] = wi.to(_acc_device)
                    del dp_padded, dp_shifted, wi, wi_exp
                _tilt_profile["band_accumulate_seconds"] = _profile_elapsed(_band_t0)
                _tilt_profile["fallback_batches"] = _fallback_batches
                _cleanup_t0 = _profile_now()
                del a, a_reshaped, a_raw
                if isinstance(arrays, _TiltFiles) and not _use_prefetch:
                    arrays.release()
                # On a real cross-GPU split the contribution adds copy compute ->
                # accumulator asynchronously; sync both cards before the next tilt
                # reuses the compute device, else the writes race the next tilt's
                # kernels (CUDA "unspecified launch failure"). Single-GPU stream
                # stays on one stream, so the sync is only needed when _split.
                if _split:
                    # _acc_device is a second GPU (cross-GPU split) OR CPU RAM
                    # (out-of-core). Sync only CUDA devices: a CPU accumulator's blocking
                    # copy already synchronized, and self.device may be MPS/CPU if a
                    # caller forces accumulator_device='cpu' off a CUDA box.
                    if torch.device(self.device).type == "cuda":
                        torch.cuda.synchronize(self.device)
                    if torch.device(_acc_device).type == "cuda":
                        torch.cuda.synchronize(_acc_device)
                    # empty_cache helps ONLY the memory-tight split path; on the in-VRAM
                    # path it returns cached blocks and forces the next tilt to
                    # re-cudaMalloc (a per-tilt stall), so gate it here.
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                _tilt_profile["cleanup_seconds"] = _profile_elapsed(_cleanup_t0)
                _tilt_profile["total_seconds"] = _profile_elapsed(_tilt_t0)
                if profile_timings is not None:
                    profile_timings["tilts"].append(_tilt_profile)
        finally:
            if _prefetch_executor is not None:
                _prefetch_executor.shutdown(wait=True)
        if profile_timings is not None:
            profile_timings["tilt_loop_seconds"] = _profile_elapsed(_tilt_loop_t0)
        del _stage  # free the (pinned) staging buffer before the divide + scaling
        # Edge contribution + factorized-den divide, one output-row band at a
        # time so den is only ever materialized batch-sized, never full.
        _edge_t0 = _profile_now()
        edge = edge_w_dp.to(_acc_device)  # (Hp, Wp)
        num += edge[None, None] * pad_val_dp
        wdp_acc = wdp_shifted.to(_acc_device)  # (n, Hp, Wp)
        edge_b = edge[None, None]
        if profile_timings is not None:
            profile_timings["edge_fill_seconds"] = _profile_elapsed(_edge_t0)
        _normalize_t0 = _profile_now()
        for bs in range(0, Rout, batch_size):
            be = min(bs + batch_size, Rout)
            # den_band[r,c,h,w] = sum_i wi_all[i,r,c] * wdp[i,h,w] + edge[h,w]
            den_band = torch.einsum("nrc,nhw->rchw", wi_all[:, bs:be], wdp_acc)
            den_band += edge_b
            nb = num[bs:be]
            mask = den_band == 0.0
            nb.div_(den_band)
            nb.masked_fill_(mask, 0.0)
            del den_band, mask
        merged = num
        del wi_all, wdp_acc
        if profile_timings is not None:
            profile_timings["normalize_seconds"] = _profile_elapsed(_normalize_t0)

        _summary_t0 = _profile_now()
        if compute_summaries:
            # MPS reductions over the full no-bin tensor can silently produce
            # incorrect values once the flattened reduction exceeds the Metal
            # indexing limit. Keep reductions in scan-row bands; this also mirrors
            # the preprocess summary path used before merge.
            summary_band = min(max(1, int(batch_size)), _MPS_PREPROCESS_ROW_BAND)
            im_bf = torch.empty((Rout, Cout), dtype=torch.float32, device=merged.device)
            dp_sum = torch.zeros((Hp, Wp), dtype=torch.float32, device=merged.device)
            for r0 in range(0, Rout, summary_band):
                r1 = min(r0 + summary_band, Rout)
                band = merged[r0:r1].contiguous()
                flat_band = band.reshape((r1 - r0) * Cout, Hp * Wp)
                im_bf[r0:r1] = flat_band.mean(1).reshape(r1 - r0, Cout)
                dp_sum += flat_band.sum(0).reshape(Hp, Wp)
                del band, flat_band
            self.im_bf_merged = im_bf
            self.dp_mean_merged = dp_sum / float(Rout * Cout)
        else:
            self.im_bf_merged = None
            self.dp_mean_merged = None
        if profile_timings is not None:
            profile_timings["summary_seconds"] = _profile_elapsed(_summary_t0)

        # dtype scaling and clipping
        _dtype_t0 = _profile_now()
        try:
            info = torch.iinfo(dtype_out)
            is_int_dtype = True
        except TypeError:
            is_int_dtype = False

        if is_int_dtype:
            dmin = float(info.min)
            dmax = float(info.max)

            merged_f = merged

            if scale_output:
                peak = torch.max(merged_f).item()
                if peak <= 0.0:
                    merged_scaled = merged_f
                else:
                    merged_scaled = merged_f * (dmax / peak)

                lo, hi = (0.0, dmax) if dtype_out == torch.uint8 else (dmin, dmax)
                merged_out = torch.round(torch.clamp(merged_scaled, lo, hi)).to(dtype=dtype_out)
            else:
                below = torch.min(merged_f).item()
                above = torch.max(merged_f).item()
                if below < dmin or above > dmax:
                    warnings.warn(
                        f"Output overflow for dtype {dtype_out}: data range [{below}, {above}] exceeds "
                        f"[{dmin}, {dmax}]. Values will be clipped.",
                        RuntimeWarning,
                    )
                merged_out = torch.round(torch.clamp(merged_f, dmin, dmax)).to(dtype=dtype_out)
        else:
            merged_out = merged.to(dtype=dtype_out)
        if profile_timings is not None:
            profile_timings["dtype_output_seconds"] = _profile_elapsed(_dtype_t0)

        _wrap_t0 = _profile_now()
        dataset_merged = Dataset4dstem.from_tensor(tensor=merged_out)
        if compute_summaries:
            dataset_merged.im_bf_merged = self.im_bf_merged
            dataset_merged.dp_mean_merged = self.dp_mean_merged
        self.merged = dataset_merged
        if profile_timings is not None:
            profile_timings["dataset_wrap_seconds"] = _profile_elapsed(_wrap_t0)
            profile_timings["total_profiled_seconds"] = _profile_elapsed(_profile_t0)

        if plot_result:
            show_2d(
                [[self.im_bf_merged, self.dp_mean_merged]],
                title=[["Merged Bright Field", "Merged Mean Diffraction Pattern"]],
                **plot_kwargs,
            )

        return dataset_merged

    @staticmethod
    def _as_torch(obj: Any) -> torch.Tensor:
        """Pull a Torch tensor out of a dataset or accelerator array."""
        if torch.is_tensor(obj):
            return obj
        tensor = getattr(obj, "tensor", None)
        if torch.is_tensor(tensor):
            return tensor
        data = getattr(obj, "data", obj)
        if torch.is_tensor(data):
            return data
        if hasattr(data, "__dlpack__"):
            return torch.from_dlpack(data)
        return torch.as_tensor(data)

    def close(self) -> None:
        """Release MAPED-owned encoded inputs, output, and viewer resources."""
        viewer = getattr(self, "viewer", None)
        if viewer is not None and hasattr(viewer, "close"):
            viewer.close()
            self.viewer = None
        merged = getattr(self, "merged", None)
        if merged is not None and hasattr(merged, "close"):
            merged.close()
            self.merged = None
        if isinstance(self.datasets, _ResidentTilts):
            self.datasets.close()

    def show(
        self,
        reference: int | None = None,
        *,
        labels: Sequence[str] | None = None,
        dp_scale_mode: str = "log",
        verbose: bool = False,
        **show_kwargs: Any,
    ):
        """Open the merged result in Show4DSTEM.

        Encoded resident workflows display the packed scaled result or an unsaved
        float32 scan-region inspection directly. Dense and file-streamed workflows retain
        the two-panel reference-tilt versus merge view.

        Parameters
        ----------
        reference
            Reference tilt for dense and file-streamed workflows.
        labels
            Two frame labels for dense and file-streamed workflows.
        dp_scale_mode
            Forwarded to Show4DSTEM (default ``"log"``).
        verbose
            Show4DSTEM chatter.
        **show_kwargs
            Extra Show4DSTEM kwargs (e.g. ``view_mode="multiple"``).

        Returns
        -------
        Show4DSTEM
            Live viewer (also stored as ``self.viewer``).
        """
        if not hasattr(self, "merged") or self.merged is None:
            raise RuntimeError("Run merge_datasets() (or run()) before show().")

        from quantem.widget import Show4DSTEM

        if isinstance(self.datasets, _ResidentTilts):
            viewer = Show4DSTEM(
                self.merged,
                dp_scale_mode=dp_scale_mode,
                verbose=verbose,
                **show_kwargs,
            )
            self.viewer = viewer
            return viewer

        # The widget's own Dataset5dstem, not quantem.diffraction's: Show4DSTEM is
        # built for it, and the sharded multi-GPU class the merge uses internally
        # is not the type its viewer paths accept.
        from quantem.widget.data import Dataset5dstem

        idx = len(self.datasets) // 2 if reference is None else reference
        merged_t = self._as_torch(self.merged)
        dev = torch.device(self.device)

        # Drop any still-resident stream tilt, then load the reference onto the
        # same card as the merge (both stay on GPU).
        if hasattr(self.datasets, "release"):
            self.datasets.release()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ref_t = self._as_torch(self.datasets[idx]).to(device=dev, non_blocking=False)
        if merged_t.device != dev:
            merged_t = merged_t.to(device=dev, non_blocking=False)

        if labels is None:
            labels = ("BEFORE - single tilt", "AFTER - merged")
        if len(labels) != 2:
            raise ValueError(f"labels must have length 2; got {len(labels)}")

        show_kwargs = dict(show_kwargs)
        show_kwargs.setdefault("view_mode", "multiple")
        show_kwargs.setdefault("columns", 2)
        show_kwargs.setdefault("page_size", 2)

        # Both panels must share a dtype, and the viewer reads the merge through
        # numpy, which has no uint16 torch equivalent to interpret. Match the
        # merge's float32 so the before/after pair is directly comparable.
        # from_frames requires a shared dtype, and the merge is float32 while a
        # streamed tilt is native uint16; matching also makes the two panels
        # directly comparable on screen.
        # `series` is the numeric series axis, so the human labels go to the
        # viewer's frame_labels instead.
        ds5 = Dataset5dstem.from_frames(
            [ref_t.to(merged_t.dtype), merged_t],
            series_type="generic",
            name="MAPED before/after",
        )
        viewer = Show4DSTEM(
            ds5,
            dp_scale_mode=dp_scale_mode,
            frame_dim_label="view",
            frame_labels=list(labels),
            verbose=verbose,
            **show_kwargs,
        )
        self.viewer = viewer
        return viewer


def shift_images(
    images: list[np.ndarray],
    shifts_rc: np.ndarray,
    edge_blend: float = 8.0,
    padding: int | None = None,
    pad_val: str | float = 0.0,
    shift_method: str = "bilinear",
):
    """
    Shift and blend a stack of 2D images into a common padded canvas.

    Parameters
    ----------
    images : list of np.ndarray
        Sequence of (H, W) arrays.
    shifts_rc : np.ndarray
        Array-like of shape (n, 2) with (row, col) shifts for each image.
    edge_blend : float, optional
        Tukey taper width in pixels for image blending.
    padding : int
        Output padding. If None, set from max shift and edge_blend.
    pad_val : str | float optional
        Fill value outside support ('min','max','mean','median' or float).
    shift_method : str
        'bilinear' or 'fourier'.

    Returns
    -------
    np.ndarray
        Blended image of shape (H + 2*padding, W + 2*padding).
    """
    images = [np.asarray(im, dtype=float) for im in images]
    if len(images) == 0:
        raise ValueError("images must be non-empty")

    H, W = images[0].shape
    for im in images:
        if im.shape != (H, W):
            raise ValueError("all images must have the same shape")

    shifts_rc = np.asarray(shifts_rc, dtype=float)
    if shifts_rc.shape != (len(images), 2):
        raise ValueError("shifts_rc must have shape (len(images), 2)")

    if isinstance(pad_val, str):
        s = pad_val.strip().lower()
        v = np.stack(images, axis=0).reshape(-1)
        if s == "min":
            pad_val_f = float(np.min(v))
        elif s == "max":
            pad_val_f = float(np.max(v))
        elif s == "mean":
            pad_val_f = float(np.mean(v))
        elif s == "median":
            pad_val_f = float(np.median(v))
        else:
            raise ValueError("pad_val must be a float or one of {'min','max','mean','median'}")
    else:
        pad_val_f = float(pad_val)

    if padding is None:
        max_shift = float(np.max(np.abs(shifts_rc))) if shifts_rc.size else 0.0
        padding = int(np.ceil(max_shift + float(edge_blend))) + 2
    padding = int(padding)

    alpha_r = min(1.0, 2.0 * float(edge_blend) / float(H)) if edge_blend > 0 else 0.0
    alpha_c = min(1.0, 2.0 * float(edge_blend) / float(W)) if edge_blend > 0 else 0.0
    w = tukey(H, alpha=alpha_r)[:, None] * tukey(W, alpha=alpha_c)[None, :]
    w = w.astype(float, copy=False)

    Hp = H + 2 * padding
    Wp = W + 2 * padding

    stack_w = np.zeros((len(images), Hp, Wp), dtype=float)
    stack = np.zeros_like(stack_w)

    r0 = padding
    c0 = padding
    stack_w[:, r0 : r0 + H, c0 : c0 + W] = w[None, :, :]
    for ind, im in enumerate(images):
        stack[ind, r0 : r0 + H, c0 : c0 + W] = im * w

    method = str(shift_method).strip().lower()
    if method not in {"bilinear", "fourier"}:
        raise ValueError("shift_method must be 'bilinear' or 'fourier'")

    if method == "fourier":
        kr = np.fft.fftfreq(Hp)[:, None]
        kc = np.fft.fftfreq(Wp)[None, :]
        for ind in range(len(images)):
            dr, dc = shifts_rc[ind, 0], shifts_rc[ind, 1]
            ramp = np.exp(-2j * np.pi * (kr * dr + kc * dc))

            F = np.fft.fft2(stack[ind])
            stack[ind] = np.fft.ifft2(F * ramp).real

            Fw = np.fft.fft2(stack_w[ind])
            stack_w[ind] = np.fft.ifft2(Fw * ramp).real
            stack_w[ind] = np.clip(stack_w[ind], 0.0, 1.0)
    else:
        for ind in range(len(images)):
            stack[ind] = ndi_shift(
                stack[ind],
                shift=(shifts_rc[ind, 0], shifts_rc[ind, 1]),
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            stack_w[ind] = ndi_shift(
                stack_w[ind],
                shift=(shifts_rc[ind, 0], shifts_rc[ind, 1]),
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            stack_w[ind] = np.clip(stack_w[ind], 0.0, 1.0)

    edge_w = np.clip(1.0 - np.sum(stack_w, axis=0), 0.0, 1.0)

    num = np.sum(stack, axis=0) + edge_w * pad_val_f
    den = np.sum(stack_w, axis=0) + edge_w

    out = np.empty_like(num)
    np.divide(num, den, out=out, where=den != 0.0)
    out[den == 0.0] = 0.0

    return out


# grid_sample indexes with 32 bits: a single call over more than 2**31 elements
# is rejected by MPSGraph and is where the no-bin merge fails on an Apple GPU.
_MAX_GRID_SAMPLE_ELEMENTS = 2**31 - 1


# MPSGraph addresses tensor dims with 32 bits, so a fused multiply-add against an
# accumulator larger than this raises "does not support tensor dims larger than
# INT_MAX" even though the same shapes work as two separate kernels.
_MAX_MPS_FUSED_ELEMENTS = 2**31 - 1
_MPS_PREPROCESS_ROW_BAND = 16


@lru_cache(maxsize=8)
def _compiled_grid_sample(cast_dtype: torch.dtype):
    def _warp(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.grid_sample(
            x.to(cast_dtype),
            grid.to(cast_dtype),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    return torch.compile(_warp, backend="inductor", dynamic=False)


def _grid_sample_tilt(a_reshaped, grid_full, cast_dtype, *, compile_torch: bool = False):
    """Real-space-warp one tilt's diffraction patterns onto the output grid.

    ``a_reshaped`` is ``(1, H*W, Rs, Cs)`` - the whole tilt, with the detector
    pixels as channels. grid_sample needs a float input; how we supply it depends
    on whether a cast is required:

    - Input ALREADY the target dtype (e.g. a float32 tilt that fits in memory):
      one grid_sample over all channels. This is the reference path - bit-for-bit
      identical to the pre-streaming merge, so the frozen baseline holds exactly.
    - Input a different dtype (uint16 no-bin tilt = 19.3 GB, whose float copy is
      38.6 GB and would not fit beside the 38.6 GB accumulator): cast ONE detector
      slab to float at a time and stitch. The H*W detector channels are sampled
      independently, so the stitched result is the same values - but the cast makes
      each slab contiguous, so grid_sample picks a different kernel and the extreme
      tail can differ at ~1e-4 (sum/mean/std stay within 1e-5). This float-reorder
      is the price of fitting no-bin on one GPU, and only the uint16 path pays it.
    """
    grid_sample = _compiled_grid_sample(cast_dtype) if compile_torch else None
    det, rows, cols = a_reshaped.shape[1], a_reshaped.shape[2], a_reshaped.shape[3]
    # grid_sample indexes its input with 32 bits, so one call must stay under
    # 2**31 elements. At no-bin a quarter of the detector is 9216 * 512 * 512 =
    # 2.4e9, which CUDA tolerates but MPSGraph rejects outright ("does not
    # support tensor dims larger than INT_MAX"). Take whichever is more slabs:
    # the historic quarter, or the fewest that fit under the limit.
    min_slabs = -(-det * rows * cols // _MAX_GRID_SAMPLE_ELEMENTS)
    if a_reshaped.dtype == cast_dtype and min_slabs <= 1:
        if grid_sample is not None:
            return grid_sample(a_reshaped, grid_full)
        return torch.nn.functional.grid_sample(
            a_reshaped, grid_full.to(cast_dtype),
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )
    slab = max(1, -(-det // max(4, min_slabs)))
    grid_cast = grid_full.to(cast_dtype)
    # Pre-allocate the full (1, det, batch, Cout) output ONCE and write each slab's
    # grid_sample result into its channel slice. The previous torch.cat held all
    # slab buffers live, then allocated a second full-detector buffer to reassemble
    # them; writing into one pre-allocated buffer keeps only one slab + the output
    # live (lower allocator churn). Same kernel + same values -> bit-identical.
    batch_n, cout_n = grid_cast.shape[1], grid_cast.shape[2]
    out = torch.empty((1, det, batch_n, cout_n), dtype=cast_dtype, device=a_reshaped.device)
    for c0 in range(0, det, slab):
        a_slab = a_reshaped[:, c0 : c0 + slab]
        if grid_sample is None:
            out[:, c0 : c0 + slab] = torch.nn.functional.grid_sample(
                a_slab.to(cast_dtype), grid_cast,
                mode="bilinear", padding_mode="zeros", align_corners=True,
            )
        else:
            out[:, c0 : c0 + slab] = grid_sample(a_slab, grid_cast)
        del a_slab
    return out


def _sample_tilt_constant_bilinear(
    a_reshaped: torch.Tensor,
    batch_start: int,
    batch_end: int,
    cout: int,
    real_space_padding: int,
    shift_rc: torch.Tensor,
    cast_dtype: torch.dtype,
) -> torch.Tensor:
    """Sample a tilt for one row band using MAPED's constant real-space shift.

    The merge's real-space warp is a rigid sub-pixel translation for every
    detector channel in a tilt. ``grid_sample`` treats it as a general gather over
    ``H*W`` channels, which is the dominant MPS cost. This evaluates the same
    bilinear interpolation by slicing the scan axes directly:

    ``source_row = output_row - real_space_padding - shift_row``
    ``source_col = output_col - real_space_padding - shift_col``

    The return shape matches ``grid_sample``: ``(1, det, batch, cout)``.
    """
    _, det, rows, cols = a_reshaped.shape
    batch_n = batch_end - batch_start
    out = torch.zeros((1, det, batch_n, cout), dtype=cast_dtype, device=a_reshaped.device)
    shift = shift_rc.reshape(-1)
    row_offset = float(batch_start - real_space_padding) - float(shift[0])
    col_offset = float(-real_space_padding) - float(shift[1])
    row_floor = math.floor(row_offset)
    col_floor = math.floor(col_offset)
    row_frac = row_offset - row_floor
    col_frac = col_offset - col_floor

    row_taps = ((row_floor, 1.0 - row_frac), (row_floor + 1, row_frac))
    col_taps = ((col_floor, 1.0 - col_frac), (col_floor + 1, col_frac))
    for row_delta, row_weight in row_taps:
        if row_weight == 0.0:
            continue
        row_start = max(0, -row_delta)
        row_stop = min(batch_n, rows - row_delta)
        if row_start >= row_stop:
            continue
        src_row_start = row_start + row_delta
        src_row_stop = row_stop + row_delta
        for col_delta, col_weight in col_taps:
            weight = row_weight * col_weight
            if weight == 0.0:
                continue
            col_start = max(0, -col_delta)
            col_stop = min(cout, cols - col_delta)
            if col_start >= col_stop:
                continue
            src_col_start = col_start + col_delta
            src_col_stop = col_stop + col_delta
            out[:, :, row_start:row_stop, col_start:col_stop] += (
                a_reshaped[:, :, src_row_start:src_row_stop, src_col_start:src_col_stop].to(
                    cast_dtype
                )
                * weight
            )
    return out


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

    imageUpsample = torch.matmul(torch.matmul(rowKern, imageCorr), colKern)

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
