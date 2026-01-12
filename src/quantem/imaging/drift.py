import warnings
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, fields, replace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.optimize import minimize
from tqdm import tqdm

from quantem.core.config import get_device
from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.compound_validators import (
    validate_list_of_dataset2d,
    validate_pad_value,
)
from quantem.core.utils.imaging_utils import (
    bilinear_kde,
    cross_correlation_shift,
    fourier_cropping,
)
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization import show_2d


@dataclass
class NonrigidAlignmentParams:
    """Configuration for non-rigid alignment."""

    # Shared parameters
    num_iterations: int = 8
    regularization_sigma_px: float = 16.0
    regularization_update_step_size: float | None = 0.8
    min_image_shift: float | None = None
    max_image_shift: float | None = 32.0
    translation_interval: int = 1
    translation_upsample_factor: int = 8
    translation_downsample_factor: int = 1
    # PyTorch parameters
    adam_steps: int = 50
    lr: float = 0.02
    pytorch_use_amp: bool = False
    pytorch_normalize_loss: bool = True
    pytorch_row_stride: int | None = None
    pytorch_fast_schedule: bool = True
    pytorch_fast_iterations: int = 2
    pytorch_fast_steps: int = 8
    pytorch_schedule: str | None = None
    pytorch_learn_translation: bool = False
    pytorch_translation_center: bool = True
    pytorch_translation_penalty: float | None = None
    pytorch_learn_affine: bool = False
    pytorch_affine_center: bool = True
    pytorch_affine_penalty: float | None = None
    pytorch_reference_mode: str = "auto"
    pytorch_multiscale: bool = True
    pytorch_multiscale_scales: Sequence[float] | None = None
    pytorch_multiscale_steps: int | None = None
    pytorch_multiscale_row_stride: int | None = None
    pytorch_refine_steps: int = 0
    pytorch_refine_lr_scale: float = 0.5
    pytorch_refine_normalize_loss: bool | None = None
    pytorch_refine_row_stride: int = 1
    # SciPy parameters
    max_optimize_iterations: int = 10
    regularization_poly_order: int = 1
    regularization_max_image_shift_px: float | None = None
    solve_individual_rows: bool = True
    # Display parameters
    show_merged: bool = True
    show_images: bool = False
    show_knots: bool = True


_NONRIGID_PARAM_NAMES = {field.name for field in fields(NonrigidAlignmentParams)}


def _split_nonrigid_overrides(
    overrides: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    param_overrides: dict[str, object] = {}
    plot_kwargs: dict[str, object] = {}
    for key, value in overrides.items():
        if key in _NONRIGID_PARAM_NAMES:
            param_overrides[key] = value
        else:
            plot_kwargs[key] = value
    return param_overrides, plot_kwargs


def _coerce_nonrigid_params(
    params: NonrigidAlignmentParams | dict[str, object] | None,
    overrides: dict[str, object],
) -> tuple[NonrigidAlignmentParams, dict[str, object]]:
    if params is None:
        params_obj = NonrigidAlignmentParams()
    elif isinstance(params, NonrigidAlignmentParams):
        params_obj = params
    elif isinstance(params, dict):
        try:
            params_obj = NonrigidAlignmentParams(**params)
        except TypeError as exc:
            raise ValueError("Unknown nonrigid alignment parameter in params.") from exc
    else:
        raise TypeError("params must be NonrigidAlignmentParams, dict, or None.")
    param_overrides, plot_kwargs = _split_nonrigid_overrides(overrides)
    if param_overrides:
        try:
            params_obj = replace(params_obj, **param_overrides)
        except TypeError as exc:
            raise ValueError("Unknown nonrigid alignment parameter in overrides.") from exc
    return params_obj, plot_kwargs


class DriftCorrection(AutoSerialize):
    """
    DriftCorrection provides translation, affine, and non-rigid drift correction for
    sequential 2D images using scan direction metadata and flexible spatial interpolation.

    This class supports input data as numpy arrays, Dataset2d, or Dataset3d instances,
    with various padding strategies and configurable spline interpolation of scanline
    trajectories via Bézier knot control.

    Features
    --------
    - Load data from arrays or files
    - Apply initial scanline resampling using Bézier curves
    - Align images using translation, affine, or non-rigid optimization
    - Visualize intermediate and final results with optional knot overlays
    - Serialize state with `.save()` and restore with `.load()`

    Parameters (via `from_data` or `from_file`)
    -------------------------------------------
    images : list of 2D arrays, Dataset2d, Dataset3d, or file names, or a 3D numpy array
        The image stack to correct for drift.
    scan_direction_degrees : list of float
        The scan direction angle (in degrees) for each image, measured relative to vertical.
    pad_fraction : float, default 0.25
        Fraction of padding to add around each image during interpolation.
    pad_value : str, float, or list of float, default 'median'
        How to pad outside the image area during warping. Can be:
        - One of: 'median', 'mean', 'min', 'max'
        - A float quantile value (e.g., 0.25)
        - A list of per-image float values
    number_knots : int, default 1
        Number of knots to use for Bézier interpolation of scanline trajectories.
        We strongly recommend using `number_knots = 1` unless the fast scan direction is
        expected to vary within the image.

    Example
    -------
    Instantiate the DriftCorrection class, run preprocessing and alignment, and save/load results:

    >>> drift = DriftCorrection.from_data(
    ...     images=[
    ...         image0,  # 2D numpy array or Dataset2d
    ...         image1,
    ...     ],
    ...     scan_direction_degrees=[0, 90],
    ... ).preprocess(
    ...     pad_fraction=0.25,
    ...     pad_value='median',
    ...     number_knots=1,
    ... )

    >>> drift.align_affine()
    >>> drift.align_nonrigid()
    >>> drift.plot_merged_images()
    >>> image_corr = drift.generate_corrected_image()

    >>> drift.save("drift_result.zip")
    >>> drift_reloaded = quantem.io.load("drift_result.zip")

    >>> image_corr.save("image_corrected.zip")
    >>> image_corr_reloaded = quantem.io.load("image_corrected.zip")

    Notes
    -----
    - Use `align_translation()` for rigid shifts, `align_affine()` for scan-shear or uniform drift,
      and `align_nonrigid()` for flexible per-row or per-image correction.
    - The class stores resampled images in `self.images_warped` and the control knots in `self.knots`.
    - Interactive visualization is supported through `plot_merged_images()` and `plot_transformed_images()`.
    """

    _token = object()

    def __init__(
        self,
        images: list[Dataset2d],
        scan_direction_degrees: NDArray,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use DriftCorrection.from_data() or .from_file() to instantiate this class."
            )

        self._images = images
        self.scan_direction_degrees = scan_direction_degrees

    @classmethod
    def from_file(
        cls,
        file_paths: Sequence[str],
        scan_direction_degrees: Sequence[float] | NDArray,
        file_type: str | None = None,
    ) -> "DriftCorrection":
        image_list = [Dataset2d.from_file(fp, file_type=file_type) for fp in file_paths]
        return cls.from_data(
            image_list,
            scan_direction_degrees,
        )

    @classmethod
    def from_data(
        cls,
        images: list[Dataset2d] | list[NDArray] | Dataset3d | NDArray,
        scan_direction_degrees: list[float] | NDArray,
    ) -> "DriftCorrection":
        validated_images = validate_list_of_dataset2d(images)

        return cls(
            images=validated_images,
            scan_direction_degrees=scan_direction_degrees,
            _token=cls._token,
        )

    # --- Properties ---
    @property
    def images(self) -> list[Dataset2d]:
        return self._images

    @images.setter
    def images(self, value: list[Dataset2d] | list[NDArray] | Dataset3d | NDArray):
        self._images = validate_list_of_dataset2d(value)
        self.pad_value = self.pad_value

    @property
    def pad_value(self) -> list[float]:
        return self._pad_value

    @pad_value.setter
    def pad_value(self, value: float | str | list[float]):
        self._pad_value = validate_pad_value(value, self.images)

    @property
    def scan_direction_degrees(self) -> NDArray:
        return self._scan_direction_degrees

    @scan_direction_degrees.setter
    def scan_direction_degrees(self, value: list[float] | NDArray):
        self._scan_direction_degrees = ensure_valid_array(value, ndim=1)

    @property
    def pad_fraction(self) -> float:
        return self._pad_fraction

    @pad_fraction.setter
    def pad_fraction(self, value: float):
        self._pad_fraction = float(value)

    @property
    def kde_sigma(self) -> float:
        return self._kde_sigma

    @kde_sigma.setter
    def kde_sigma(self, value: float):
        self._kde_sigma = float(value)

    @property
    def number_knots(self) -> int:
        return self._number_knots

    @number_knots.setter
    def number_knots(self, value: float):
        self._number_knots = int(value)

    def preprocess(
        self,
        pad_fraction: float = 0.25,
        pad_value: float | str | list[float] = "median",
        kde_sigma: float = 0.5,
        number_knots: int = 1,
        show_merged: bool = False,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        # Validators
        validated_pad_value = validate_pad_value(pad_value, self._images)

        # Input data
        self.pad_fraction = pad_fraction
        self._pad_value = validated_pad_value
        self.kde_sigma = kde_sigma
        self.number_knots = number_knots

        # Derived data
        self.scan_direction = np.deg2rad(self.scan_direction_degrees)
        self.scan_fast = np.stack(
            [
                np.sin(-self.scan_direction),
                np.cos(-self.scan_direction),
            ],
            axis=1,
        )
        self.scan_slow = np.stack(
            [
                np.cos(-self.scan_direction),
                -np.sin(-self.scan_direction),
            ],
            axis=1,
        )
        self.shape = (
            len(self.images),
            int(np.round(self.images[0].shape[0] * (1 + self.pad_fraction) / 2) * 2),
            int(np.round(self.images[1].shape[1] * (1 + self.pad_fraction) / 2) * 2),
        )

        # Initialize Bezier knots and scan vectors for scanlines
        self.knots = []
        for a0 in range(self.shape[0]):
            shape = self.images[a0].shape

            v_slow = np.linspace(-(shape[0] - 1) / 2, (shape[0] - 1) / 2, shape[0])
            u_fast = np.linspace(-(shape[1] - 1) / 2, (shape[1] - 1) / 2, self.number_knots)

            xa = (
                (self.shape[1] - 1) / 2
                + u_fast[None, :] * self.scan_fast[a0, 0]
                + v_slow[:, None] * self.scan_slow[a0, 0]
            )
            ya = (
                (self.shape[2] - 1) / 2
                + u_fast[None, :] * self.scan_fast[a0, 1]
                + v_slow[:, None] * self.scan_slow[a0, 1]
            )

            self.knots.append(np.stack([xa, ya], axis=0))

        # Precompute the interpolator for all images
        self.interpolator = []
        for a0 in range(self.shape[0]):
            self.interpolator.append(
                DriftInterpolator(
                    input_shape=self.images[a0].shape,
                    output_shape=self.shape[1:],
                    scan_fast=self.scan_fast[a0],
                    scan_slow=self.scan_slow[a0],
                    pad_value=self.pad_value[a0],
                    kde_sigma=self.kde_sigma,
                )
            )

        # Generate initial resampled images
        self.images_warped = Dataset3d.from_shape(self.shape)
        self.weights_warped = Dataset3d.from_shape(self.shape)
        for ind in range(self.shape[0]):
            self.images_warped.array[ind], self.weights_warped.array[ind] = self.interpolator[
                ind
            ].warp_image(
                self.images[ind].array,
                self.knots[ind],
            )

        # Error tracking
        self.calculate_error(0)

        # Plots
        kwargs.pop("title", None)
        if show_merged:
            self.plot_merged_images(show_knots=show_knots, title="Merged: initial", **kwargs)
        if show_images:
            self.plot_transformed_images(
                show_knots=show_knots,
                title=[f"Image {i}: initial" for i in range(self.shape[0])],
                **kwargs,
            )

        return self

    # Translation alignment
    def align_translation(
        self,
        upsample_factor: int = 8,
        downsample_factor: int = 1,
        min_image_shift: float | None = None,
        max_image_shift: float | None = 32,
        show_merged: bool = True,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        """
        Solve for the translation between all images in DriftCorrection.images_warped

        Parameters
        ----------
        upsample_factor : int, default 8
            Subpixel upsampling factor for the cross-correlation peak.
        downsample_factor : int, default 1
            Downsample factor for the alignment FFTs (1 = full resolution).
        """

        if not hasattr(self, "knots"):
            print("\033[91mNo knots found — running .preprocess() with default settings.\033[0m")
            self.preprocess()

        # init
        dxy = np.zeros((self.shape[0], 2))
        upsample_factor = max(1, int(upsample_factor))
        downsample_factor = max(1, int(downsample_factor))
        max_shift = None
        if max_image_shift is not None:
            max_shift = max_image_shift / downsample_factor

        # loop over images
        ref_image = self.images_warped.array[0]
        if downsample_factor > 1:
            ref_image = ref_image[::downsample_factor, ::downsample_factor]
        F_ref = np.fft.fft2(ref_image)
        for ind in range(1, self.shape[0]):
            image = self.images_warped.array[ind]
            if downsample_factor > 1:
                image = image[::downsample_factor, ::downsample_factor]
            shifts, image_shift = cross_correlation_shift(
                F_ref,
                np.fft.fft2(image),
                upsample_factor=upsample_factor,
                max_shift=max_shift,
                fft_input=True,
                fft_output=True,
                return_shifted_image=True,
            )

            dxy[ind, :] = np.array(shifts) * downsample_factor
            F_ref = F_ref * ind / (ind + 1) + image_shift / (ind + 1)

        # Normalize dxy
        dxy -= np.mean(dxy, axis=0)

        # Minimum image shift
        if min_image_shift is not None:
            small = np.linalg.norm(dxy, axis=1) < min_image_shift
            dxy[small] = 0.0

        # Apply shifts to knots
        for ind in range(self.shape[0]):
            self.knots[ind][0] += dxy[ind, 0]
            self.knots[ind][1] += dxy[ind, 1]

        # Regenerate images
        for ind in range(self.shape[0]):
            self.images_warped.array[ind], self.weights_warped.array[ind] = self.interpolator[
                ind
            ].warp_image(
                self.images[ind].array,
                self.knots[ind],
            )

        # Plots
        kwargs.pop("title", None)
        if show_merged:
            self.plot_merged_images(show_knots=show_knots, title="Merged: translation", **kwargs)
        if show_images:
            self.plot_transformed_images(
                show_knots=show_knots,
                title=[f"Image {i}: translation" for i in range(self.shape[0])],
                **kwargs,
            )

        return self

    # Affine alignment
    def align_affine(
        self,
        step: float = 0.01,
        num_tests: int = 9,
        refine: bool = True,
        upsample_factor: int = 8,
        max_image_shift: float | None = 32,
        show_merged: bool = True,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        """
        Estimate affine drift from the first 2 images.
        """

        if not hasattr(self, "knots"):
            print("\033[91mNo knots found — running .preprocess() with default settings.\033[0m")
            self.preprocess()

        if num_tests % 2 == 0:
            raise ValueError("num_tests should be odd.")

        # Potential drift vectors
        vec = np.arange(-(num_tests - 1) / 2, (num_tests + 1) / 2)
        xx, yy = np.meshgrid(vec, vec, indexing="ij")
        keep = xx**2 + yy**2 <= (num_tests / 2) ** 2
        dxy = (
            np.vstack(
                (
                    xx[keep],
                    yy[keep],
                )
            ).T
            * step
        )

        # Measure cost function for linear drift vectors
        cost = np.zeros(dxy.shape[0])
        for a0 in tqdm(range(dxy.shape[0]), desc="Solving affine drift"):
            # updated knots
            knot_0 = self.knots[0].copy()
            u = np.arange(knot_0.shape[1]) - (knot_0.shape[1] - 1) / 2
            knot_0[0] += dxy[a0, 0] * u[:, None]
            knot_0[1] += dxy[a0, 1] * u[:, None]

            knot_1 = self.knots[1].copy()
            u = np.arange(knot_1.shape[1]) - (knot_1.shape[1] - 1) / 2
            knot_1[0] += dxy[a0, 0] * u[:, None]
            knot_1[1] += dxy[a0, 1] * u[:, None]

            im0, w0 = self.interpolator[0].warp_image(
                self.images[0].array,
                knot_0,
            )
            im1, w1 = self.interpolator[1].warp_image(
                self.images[1].array,
                knot_1,
            )
            # Cross correlation alignment
            shifts, image_shift = cross_correlation_shift(
                im0,
                im1,
                upsample_factor=upsample_factor,
                fft_input=False,
                fft_output=False,
                return_shifted_image=True,
                max_shift=max_image_shift,
            )
            cost[a0] = np.mean(np.abs(im0 - image_shift))

        # update all knots
        ind = np.argmin(cost)
        for a0 in range(self.shape[0]):
            u = np.arange(self.knots[a0].shape[1]) - (self.knots[a0].shape[1] - 1) / 2
            self.knots[a0][0] += dxy[ind, 0] * u[:, None]
            self.knots[a0][1] += dxy[ind, 1] * u[:, None]

        # Regenerate images
        for ind in range(self.shape[0]):
            self.images_warped.array[ind], self.weights_warped.array[ind] = self.interpolator[
                ind
            ].warp_image(
                self.images[ind].array,
                self.knots[ind],
            )

        # Translation alignment
        self.align_translation(
            max_image_shift=max_image_shift,
            show_images=False,
            show_merged=False,
            show_knots=False,
        )

        # Error tracking
        self.calculate_error(1)

        # Affine drift refinement
        if refine:
            # Potential drift vectors
            dxy /= num_tests - 1

            # Measure cost function
            cost = np.zeros(dxy.shape[0])
            for a0 in tqdm(range(dxy.shape[0]), desc="Refining affine drift"):
                # updated knots

                knot_0 = self.knots[0].copy()
                u = np.arange(knot_0.shape[1]) - (knot_0.shape[1] - 1) / 2
                knot_0[0] += dxy[a0, 0] * u[:, None]
                knot_0[1] += dxy[a0, 1] * u[:, None]

                knot_1 = self.knots[1].copy()
                u = np.arange(knot_1.shape[1]) - (knot_1.shape[1] - 1) / 2
                knot_1[0] += dxy[a0, 0] * u[:, None]
                knot_1[1] += dxy[a0, 1] * u[:, None]

                im0, w0 = self.interpolator[0].warp_image(
                    self.images[0].array,
                    knot_0,
                )
                im1, w1 = self.interpolator[1].warp_image(
                    self.images[1].array,
                    knot_1,
                )
                # Cross correlation alignment
                shifts, image_shift = cross_correlation_shift(
                    im0,
                    im1,
                    upsample_factor=upsample_factor,
                    fft_input=False,
                    fft_output=False,
                    return_shifted_image=True,
                    max_shift=max_image_shift,
                )
                cost[a0] = np.mean(np.abs(im0 - image_shift))

            # update all knots
            ind = np.argmin(cost)
            for a0 in range(self.shape[0]):
                u = np.arange(self.knots[a0].shape[1]) - (self.knots[a0].shape[1] - 1) / 2
                self.knots[a0][0] += dxy[ind, 0] * u[:, None]
                self.knots[a0][1] += dxy[ind, 1] * u[:, None]

        # Regenerate images
        for ind in range(self.shape[0]):
            self.images_warped.array[ind], self.weights_warped.array[ind] = self.interpolator[
                ind
            ].warp_image(
                self.images[ind].array,
                self.knots[ind],
            )

        # Translation alignment
        self.align_translation(
            max_image_shift=max_image_shift,
            show_images=False,
            show_merged=False,
            show_knots=False,
        )

        # Error tracking
        self.calculate_error(1)

        # Plots
        kwargs.pop("title", None)
        if show_merged:
            self.plot_merged_images(
                show_knots=show_knots,
                title="Merged: affine",
                **kwargs,
            )
        if show_images:
            self.plot_transformed_images(
                show_knots=show_knots,
                title=[f"Image {i}: affine" for i in range(self.shape[0])],
                **kwargs,
            )

        return self

    def align(
        self,
        params: NonrigidAlignmentParams | dict[str, object] | None = None,
        **kwargs,
    ):
        """
        Joint translation + affine + non-rigid alignment in a single Adam loss.

        This uses the optimized PyTorch backend with joint parameters enabled.
        """
        if params is None:
            kwargs.setdefault("pytorch_schedule", "pytroch_joint_rmse")
            kwargs.setdefault("pytorch_normalize_loss", False)
            kwargs.setdefault("pytorch_fast_schedule", False)
        elif isinstance(params, dict):
            if "pytorch_schedule" not in params:
                kwargs.setdefault("pytorch_schedule", "pytroch_joint_rmse")
            if "pytorch_normalize_loss" not in params:
                kwargs.setdefault("pytorch_normalize_loss", False)
            if "pytorch_fast_schedule" not in params:
                kwargs.setdefault("pytorch_fast_schedule", False)
        return self.align_nonrigid(
            backend="pytorch_joint",
            params=params,
            **kwargs,
        )

    # non-rigid alignment
    def align_nonrigid(
        self,
        backend: str = "pytorch",
        params: NonrigidAlignmentParams | dict[str, object] | None = None,
        **kwargs,
    ):
        """
        Non-rigid drift correction using PyTorch (default) or SciPy backend.

        Parameters
        ----------
        backend : str, default "pytorch"
            Optimization backend. "pytorch" uses GPU-accelerated Adam optimizer.
            "pytorch_optimized" batches the optimization across images. "pytorch_joint"
            solves translation, affine, and non-rigid parameters jointly using the
            optimized backend. "scipy" uses L-BFGS row-by-row.
        params : NonrigidAlignmentParams or dict or None
            Configuration object for non-rigid alignment. Use NonrigidAlignmentParams()
            for defaults.
        **kwargs : dict
            Keys matching NonrigidAlignmentParams override config values; remaining
            kwargs are forwarded to plotting helpers.
        """
        params, plot_kwargs = _coerce_nonrigid_params(params, kwargs)
        num_iterations = params.num_iterations
        regularization_sigma_px = params.regularization_sigma_px
        regularization_update_step_size = params.regularization_update_step_size
        min_image_shift = params.min_image_shift
        max_image_shift = params.max_image_shift
        translation_interval = params.translation_interval
        translation_upsample_factor = params.translation_upsample_factor
        translation_downsample_factor = params.translation_downsample_factor
        adam_steps = params.adam_steps
        lr = params.lr
        pytorch_use_amp = params.pytorch_use_amp
        pytorch_normalize_loss = params.pytorch_normalize_loss
        pytorch_row_stride = params.pytorch_row_stride
        pytorch_fast_schedule = params.pytorch_fast_schedule
        pytorch_fast_iterations = params.pytorch_fast_iterations
        pytorch_fast_steps = params.pytorch_fast_steps
        pytorch_schedule = params.pytorch_schedule
        pytorch_learn_translation = params.pytorch_learn_translation
        pytorch_translation_center = params.pytorch_translation_center
        pytorch_translation_penalty = params.pytorch_translation_penalty
        pytorch_learn_affine = params.pytorch_learn_affine
        pytorch_affine_center = params.pytorch_affine_center
        pytorch_affine_penalty = params.pytorch_affine_penalty
        pytorch_reference_mode = params.pytorch_reference_mode
        pytorch_multiscale = params.pytorch_multiscale
        pytorch_multiscale_scales = params.pytorch_multiscale_scales
        pytorch_multiscale_steps = params.pytorch_multiscale_steps
        pytorch_multiscale_row_stride = params.pytorch_multiscale_row_stride
        pytorch_refine_steps = params.pytorch_refine_steps
        pytorch_refine_lr_scale = params.pytorch_refine_lr_scale
        pytorch_refine_normalize_loss = params.pytorch_refine_normalize_loss
        pytorch_refine_row_stride = params.pytorch_refine_row_stride
        max_optimize_iterations = params.max_optimize_iterations
        regularization_poly_order = params.regularization_poly_order
        regularization_max_image_shift_px = params.regularization_max_image_shift_px
        solve_individual_rows = params.solve_individual_rows
        show_merged = params.show_merged
        show_images = params.show_images
        show_knots = params.show_knots
        if backend not in {"pytorch", "pytorch_optimized", "pytorch_joint", "scipy"}:
            raise ValueError(
                "backend must be 'pytorch', 'pytorch_optimized', 'pytorch_joint', or 'scipy'."
            )
        if not hasattr(self, "knots"):
            print("\033[91mNo knots found — running .preprocess() with default settings.\033[0m")
            self.preprocess()
        if self.shape[0] < 2:
            raise ValueError("Non-rigid alignment requires at least 2 images.")

        optimized_backend = backend in {"pytorch_optimized", "pytorch_joint"}
        torch_cache = None
        torch_cache_scales: dict[float, dict[str, torch.Tensor | tuple[int, int]]] | None = None
        if optimized_backend:
            if self.number_knots != 1:
                raise NotImplementedError(
                    "pytorch_optimized/pytorch_joint backend only supports single-knot scanlines."
                )
            if pytorch_reference_mode not in {"auto", "leave_one_out", "mean"}:
                raise ValueError(
                    "pytorch_reference_mode must be 'auto', 'leave_one_out', or 'mean'."
                )
            if pytorch_schedule is not None and pytorch_schedule not in {
                "pytroch_optmized",
                "pytroch_joint_rmse",
            }:
                raise ValueError(
                    "pytorch_schedule must be None, 'pytroch_optmized', or 'pytroch_joint_rmse'."
                )
            min_dim = min(self.shape[1], self.shape[2])
            if pytorch_schedule == "pytroch_optmized":
                schedule = self._pytroch_optmized(min_dim)
            elif pytorch_schedule == "pytroch_joint_rmse":
                schedule = self._pytroch_joint_rmse(min_dim)
            else:
                schedule = None
            if schedule:
                num_iterations = schedule.get("num_iterations", num_iterations)
                regularization_sigma_px = schedule.get(
                    "regularization_sigma_px", regularization_sigma_px
                )
                adam_steps = schedule.get("adam_steps", adam_steps)
                lr = schedule.get("lr", lr)
                pytorch_reference_mode = schedule.get(
                    "pytorch_reference_mode", pytorch_reference_mode
                )
                pytorch_normalize_loss = schedule.get(
                    "pytorch_normalize_loss", pytorch_normalize_loss
                )
                pytorch_row_stride = schedule.get("pytorch_row_stride", pytorch_row_stride)
                pytorch_fast_schedule = schedule.get(
                    "pytorch_fast_schedule", pytorch_fast_schedule
                )
                pytorch_learn_translation = schedule.get(
                    "pytorch_learn_translation", pytorch_learn_translation
                )
                pytorch_translation_center = schedule.get(
                    "pytorch_translation_center", pytorch_translation_center
                )
                pytorch_translation_penalty = schedule.get(
                    "pytorch_translation_penalty", pytorch_translation_penalty
                )
                pytorch_learn_affine = schedule.get("pytorch_learn_affine", pytorch_learn_affine)
                pytorch_affine_center = schedule.get(
                    "pytorch_affine_center", pytorch_affine_center
                )
                pytorch_affine_penalty = schedule.get(
                    "pytorch_affine_penalty", pytorch_affine_penalty
                )
                pytorch_multiscale = schedule.get("pytorch_multiscale", pytorch_multiscale)
                pytorch_multiscale_scales = schedule.get(
                    "pytorch_multiscale_scales", pytorch_multiscale_scales
                )
                pytorch_multiscale_steps = schedule.get(
                    "pytorch_multiscale_steps", pytorch_multiscale_steps
                )
                pytorch_multiscale_row_stride = schedule.get(
                    "pytorch_multiscale_row_stride", pytorch_multiscale_row_stride
                )
                pytorch_refine_steps = schedule.get("pytorch_refine_steps", pytorch_refine_steps)
                pytorch_refine_lr_scale = schedule.get(
                    "pytorch_refine_lr_scale", pytorch_refine_lr_scale
                )
                pytorch_refine_normalize_loss = schedule.get(
                    "pytorch_refine_normalize_loss", pytorch_refine_normalize_loss
                )
                pytorch_refine_row_stride = schedule.get(
                    "pytorch_refine_row_stride", pytorch_refine_row_stride
                )
                translation_interval = schedule.get("translation_interval", translation_interval)
                translation_upsample_factor = schedule.get(
                    "translation_upsample_factor", translation_upsample_factor
                )
                translation_downsample_factor = schedule.get(
                    "translation_downsample_factor", translation_downsample_factor
                )
            if pytorch_fast_schedule and num_iterations == 8 and adam_steps == 50:
                num_iterations = pytorch_fast_iterations
                adam_steps = pytorch_fast_steps
            if pytorch_reference_mode == "auto":
                pytorch_reference_mode = "mean"
            if pytorch_row_stride is None:
                if regularization_sigma_px is None or regularization_sigma_px <= 0:
                    pytorch_row_stride = 1
                else:
                    pytorch_row_stride = max(1, int(round(regularization_sigma_px / 4)))
            if (
                pytorch_fast_schedule
                and pytorch_reference_mode == "mean"
                and min_dim <= 192
                and pytorch_row_stride > 1
            ):
                pytorch_row_stride = 1
            device = torch.device(get_device())
            target_images = torch.tensor(
                np.stack([img.array for img in self.images]),
                dtype=torch.float32,
                device=device,
            )
            row_position = torch.tensor(
                self.interpolator[0].u,
                dtype=torch.float32,
                device=device,
            )
            H, W = self.images[0].array.shape
            scan_fast = torch.tensor(self.scan_fast, dtype=torch.float32, device=device)
            scale_x = scan_fast[:, 0] * (H - 1)
            scale_y = scan_fast[:, 1] * (W - 1)
            base_x = scale_x[:, None] * row_position[None, :]
            base_y = scale_y[:, None] * row_position[None, :]
            torch_cache = {
                "target_images": target_images,
                "base_x": base_x,
                "base_y": base_y,
                "out_shape": (self.shape[1], self.shape[2]),
            }
            if pytorch_normalize_loss:
                target_mean = target_images.mean(dim=(1, 2), keepdim=True)
                target_std = target_images.std(dim=(1, 2), keepdim=True, unbiased=False)
                torch_cache["target_norm"] = (target_images - target_mean) / target_std.clamp_min(
                    1e-6
                )
            if pytorch_multiscale:
                if pytorch_multiscale_scales is None:
                    if min_dim >= 1024:
                        pytorch_multiscale_scales = (0.25, 0.5)
                    elif min_dim >= 256:
                        pytorch_multiscale_scales = (0.5,)
                    else:
                        pytorch_multiscale_scales = ()
                else:
                    pytorch_multiscale_scales = tuple(
                        s for s in pytorch_multiscale_scales if s and s < 1.0
                    )
            else:
                pytorch_multiscale_scales = ()
            torch_cache_scales = {}
            if pytorch_multiscale_scales:
                for scale in pytorch_multiscale_scales:
                    scaled = F.interpolate(
                        target_images[:, None, :, :],
                        scale_factor=scale,
                        mode="area",
                        recompute_scale_factor=False,
                    )[:, 0]
                    out_h, out_w = scaled.shape[1:]
                    row_position = torch.linspace(0, 1, out_w, device=device)
                    base_x = scan_fast[:, 0, None] * (out_h - 1) * row_position[None, :]
                    base_y = scan_fast[:, 1, None] * (out_w - 1) * row_position[None, :]
                    cache = {
                        "target_images": scaled,
                        "base_x": base_x,
                        "base_y": base_y,
                        "out_shape": (out_h, out_w),
                    }
                    if pytorch_normalize_loss:
                        target_mean = scaled.mean(dim=(1, 2), keepdim=True)
                        target_std = scaled.std(dim=(1, 2), keepdim=True, unbiased=False)
                        cache["target_norm"] = (scaled - target_mean) / target_std.clamp_min(1e-6)
                    torch_cache_scales[scale] = cache
            if backend == "pytorch_joint":
                pytorch_learn_translation = True
                pytorch_learn_affine = True
                translation_interval = 0

        def build_image_refs(reference_mode: str) -> np.ndarray:
            image_sum = np.sum(self.images_warped.array, axis=0)
            if reference_mode == "mean":
                template = image_sum / self.shape[0]
                return np.repeat(template[None, :, :], self.shape[0], axis=0)
            return (image_sum[None, :, :] - self.images_warped.array) / (self.shape[0] - 1)

        if optimized_backend and pytorch_multiscale_scales:
            for scale in pytorch_multiscale_scales:
                cache = torch_cache_scales[scale]
                out_h, out_w = cache["out_shape"]
                image_refs = build_image_refs(pytorch_reference_mode)
                image_refs_t = torch.tensor(
                    image_refs,
                    dtype=target_images.dtype,
                    device=device,
                )
                image_refs_t = F.interpolate(
                    image_refs_t[:, None, :, :],
                    size=(out_h, out_w),
                    mode="area",
                )[:, 0]
                knots_init = np.stack(self.knots, axis=0)
                rows_full = knots_init.shape[2]
                rows_scale = max(2, out_h)
                knots_scale = self._resample_knots_rows(knots_init, rows_scale) * scale
                if pytorch_multiscale_row_stride is None:
                    row_stride = max(1, int(round(pytorch_row_stride * scale)))
                else:
                    row_stride = pytorch_multiscale_row_stride
                if pytorch_multiscale_steps is None:
                    steps = max(4, int(round(adam_steps * scale)))
                else:
                    steps = pytorch_multiscale_steps
                knots_updated_stack = self._optimize_knots_pytorch_optimized(
                    image_refs_t,
                    knots_scale,
                    torch_cache=cache,
                    adam_steps=steps,
                    lr=lr,
                    use_amp=pytorch_use_amp,
                    normalize_loss=pytorch_normalize_loss,
                    row_stride=row_stride,
                    learn_translation=pytorch_learn_translation,
                    translation_center=pytorch_translation_center,
                    translation_penalty=pytorch_translation_penalty,
                    learn_affine=pytorch_learn_affine,
                    affine_center=pytorch_affine_center,
                    affine_penalty=pytorch_affine_penalty,
                )
                knots_full = self._resample_knots_rows(knots_updated_stack / scale, rows_full)
                for ind in range(self.shape[0]):
                    self.knots[ind] = knots_full[ind]
                for ind in range(self.shape[0]):
                    self.images_warped.array[ind], self.weights_warped.array[ind] = (
                        self.interpolator[ind].warp_image(self.images[ind].array, self.knots[ind])
                    )
        # Main optimization loop
        for iter_idx in tqdm(
            range(num_iterations),
            desc=f"Solving nonrigid drift ({backend})",
        ):
            if optimized_backend:
                image_refs = build_image_refs(pytorch_reference_mode)
                knots_init = np.stack(self.knots, axis=0)
                knots_updated_stack = self._optimize_knots_pytorch_optimized(
                    image_refs,
                    knots_init,
                    torch_cache=torch_cache,
                    adam_steps=adam_steps,
                    lr=lr,
                    use_amp=pytorch_use_amp,
                    normalize_loss=pytorch_normalize_loss,
                    row_stride=pytorch_row_stride,
                    learn_translation=pytorch_learn_translation,
                    translation_center=pytorch_translation_center,
                    translation_penalty=pytorch_translation_penalty,
                    learn_affine=pytorch_learn_affine,
                    affine_center=pytorch_affine_center,
                    affine_penalty=pytorch_affine_penalty,
                )
                if pytorch_refine_steps > 0 and iter_idx == num_iterations - 1:
                    refine_normalize = (
                        pytorch_refine_normalize_loss
                        if pytorch_refine_normalize_loss is not None
                        else False
                    )
                    knots_updated_stack = self._optimize_knots_pytorch_optimized(
                        image_refs,
                        knots_updated_stack,
                        torch_cache=torch_cache,
                        adam_steps=pytorch_refine_steps,
                        lr=lr * pytorch_refine_lr_scale,
                        use_amp=pytorch_use_amp,
                        normalize_loss=refine_normalize,
                        row_stride=pytorch_refine_row_stride,
                        learn_translation=pytorch_learn_translation,
                        translation_center=pytorch_translation_center,
                        translation_penalty=pytorch_translation_penalty,
                        learn_affine=pytorch_learn_affine,
                        affine_center=pytorch_affine_center,
                        affine_penalty=pytorch_affine_penalty,
                    )
                for ind in range(self.shape[0]):
                    knots_updated = knots_updated_stack[ind]
                    # Max shift regularization
                    if regularization_max_image_shift_px is not None:
                        knots_shift = knots_updated - self.knots[ind]
                        knots_dist = np.sqrt(np.sum(knots_shift**2, axis=0))
                        sub = knots_dist > regularization_max_image_shift_px
                        knots_updated[0][sub] = (
                            self.knots[ind][0][sub]
                            + knots_shift[0][sub]
                            * regularization_max_image_shift_px
                            / knots_dist[sub]
                        )
                        knots_updated[1][sub] = (
                            self.knots[ind][1][sub]
                            + knots_shift[1][sub]
                            * regularization_max_image_shift_px
                            / knots_dist[sub]
                        )
                    # Smoothness regularization
                    if regularization_sigma_px is not None and regularization_sigma_px > 0:
                        knots_smoothed = knots_updated.copy()
                        for dim in range(knots_updated.shape[0]):
                            x = np.arange(knots_updated.shape[1])
                            for knot_ind in range(knots_updated.shape[2]):
                                y = knots_updated[dim, :, knot_ind]
                                coefs = np.polyfit(x, y, deg=regularization_poly_order)
                                trend = np.polyval(coefs, x)
                                residual = y - trend
                                residual_smooth = gaussian_filter(
                                    residual, sigma=regularization_sigma_px
                                )
                                knots_smoothed[dim, :, knot_ind] = residual_smooth + trend
                        knots_updated = knots_smoothed
                    # Step size
                    if regularization_update_step_size is not None:
                        knots_updated = (
                            self.knots[ind]
                            + (knots_updated - self.knots[ind]) * regularization_update_step_size
                        )
                    self.knots[ind] = knots_updated
            else:
                for ind in range(self.shape[0]):
                    image_ref = np.delete(self.images_warped.array, ind, axis=0).mean(axis=0)
                    knots_init = self.knots[ind]
                    # Optimize knots
                    if backend == "pytorch":
                        knots_updated = self._optimize_knots_pytorch(
                            ind, image_ref, knots_init, adam_steps=adam_steps, lr=lr
                        )
                    else:
                        knots_updated = self._optimize_knots_scipy(
                            ind,
                            image_ref,
                            knots_init,
                            max_optimize_iterations=max_optimize_iterations,
                            solve_individual_rows=solve_individual_rows,
                        )
                    # Max shift regularization
                    if regularization_max_image_shift_px is not None:
                        knots_shift = knots_updated - self.knots[ind]
                        knots_dist = np.sqrt(np.sum(knots_shift**2, axis=0))
                        sub = knots_dist > regularization_max_image_shift_px
                        knots_updated[0][sub] = (
                            self.knots[ind][0][sub]
                            + knots_shift[0][sub]
                            * regularization_max_image_shift_px
                            / knots_dist[sub]
                        )
                        knots_updated[1][sub] = (
                            self.knots[ind][1][sub]
                            + knots_shift[1][sub]
                            * regularization_max_image_shift_px
                            / knots_dist[sub]
                        )
                    # Smoothness regularization
                    if regularization_sigma_px is not None and regularization_sigma_px > 0:
                        knots_smoothed = knots_updated.copy()
                        for dim in range(knots_updated.shape[0]):
                            x = np.arange(knots_updated.shape[1])
                            for knot_ind in range(knots_updated.shape[2]):
                                y = knots_updated[dim, :, knot_ind]
                                coefs = np.polyfit(x, y, deg=regularization_poly_order)
                                trend = np.polyval(coefs, x)
                                residual = y - trend
                                residual_smooth = gaussian_filter(
                                    residual, sigma=regularization_sigma_px
                                )
                                knots_smoothed[dim, :, knot_ind] = residual_smooth + trend
                        knots_updated = knots_smoothed
                    # Step size
                    if regularization_update_step_size is not None:
                        knots_updated = (
                            self.knots[ind]
                            + (knots_updated - self.knots[ind]) * regularization_update_step_size
                        )
                    self.knots[ind] = knots_updated
            # Update warped images
            for ind in range(self.shape[0]):
                self.images_warped.array[ind], self.weights_warped.array[ind] = self.interpolator[
                    ind
                ].warp_image(self.images[ind].array, self.knots[ind])
            # Translation alignment
            run_translation = False
            if translation_interval is None:
                run_translation = True
            elif translation_interval > 0:
                run_translation = (
                    iter_idx + 1
                ) % translation_interval == 0 or iter_idx == num_iterations - 1
            if run_translation:
                self.align_translation(
                    upsample_factor=translation_upsample_factor,
                    downsample_factor=translation_downsample_factor,
                    min_image_shift=min_image_shift,
                    max_image_shift=max_image_shift,
                    show_images=False,
                    show_merged=False,
                    show_knots=False,
                )
            self.calculate_error(2)

        if show_merged:
            self.plot_merged_images(
                show_knots=show_knots,
                title="Merged: non-rigid",
                **plot_kwargs,
            )

        if show_images:
            self.plot_transformed_images(
                show_knots=show_knots,
                title=[f"Image {i}: non-rigid" for i in range(self.shape[0])],
                **plot_kwargs,
            )

        return self

    def _optimize_knots_pytorch(
        self,
        idx: int,
        image_ref: np.ndarray,
        knots_init: np.ndarray,
        adam_steps: int = 5,
        lr: float = 0.02,
    ) -> np.ndarray:
        """PyTorch Adam batched optimization for one image (single knot only)."""
        # TODO: support multiple knots (requires differentiable spline interpolation)
        if knots_init.shape[2] != 1:
            raise NotImplementedError(
                f"PyTorch backend only supports single knot (got {knots_init.shape[2]}). "
                "Use backend='scipy' for multiple knots."
            )
        device = get_device()
        H, W = self.images[idx].array.shape
        # Convert to tensors
        ref_image = torch.tensor(image_ref, dtype=torch.float32, device=device)
        target_image = torch.tensor(self.images[idx].array, dtype=torch.float32, device=device)
        row_position = torch.tensor(self.interpolator[idx].u, dtype=torch.float32, device=device)
        scan_fast = self.interpolator[idx].scan_fast
        scale_x = scan_fast[0] * (H - 1)
        scale_y = scan_fast[1] * (W - 1)
        # Initialize knots as trainable tensor: shape (2, num_rows)
        knots = torch.tensor(
            knots_init[:, :, 0], dtype=torch.float32, device=device, requires_grad=True
        )
        optimizer = torch.optim.Adam([knots], lr=lr)
        # Adam optimization (batched over all rows)
        for _ in range(adam_steps):
            optimizer.zero_grad()
            # Transform: single knot = shift along scan direction
            xa = knots[0, :, None] + row_position[None, :] * scale_x
            ya = knots[1, :, None] + row_position[None, :] * scale_y
            # Bilinear interpolation (boundary clamp critical for lower RMSE than scipy's L-BFGS)
            xa_c = xa.clamp(0, H - 1.001)
            ya_c = ya.clamp(0, W - 1.001)
            # Guarantee xf+1 ≤ H-1
            xf = xa_c.floor().long().clamp(0, H - 2)
            yf = ya_c.floor().long().clamp(0, W - 2)
            dx, dy = xa_c - xf.float(), ya_c - yf.float()
            warped = (
                ref_image[xf, yf] * (1 - dx) * (1 - dy)
                + ref_image[xf + 1, yf] * dx * (1 - dy)
                + ref_image[xf, yf + 1] * (1 - dx) * dy
                + ref_image[xf + 1, yf + 1] * dx * dy
            )
            loss = ((warped - target_image) ** 2).mean()
            loss.backward()
            optimizer.step()
        return knots.detach().cpu().numpy()[:, :, None]

    @staticmethod
    def _resample_knots_rows(knots: np.ndarray, rows_out: int) -> np.ndarray:
        """Resample knot rows to a new row count using linear interpolation."""
        if rows_out < 2:
            raise ValueError("rows_out must be >= 2.")
        knots_arr = np.asarray(knots)
        squeeze = False
        if knots_arr.ndim == 3:
            knots_arr = knots_arr[None, ...]
            squeeze = True
        elif knots_arr.ndim != 4:
            raise ValueError(
                "knots must have shape (2, rows, num_knots) or (N, 2, rows, num_knots)."
            )
        rows_in = knots_arr.shape[2]
        if rows_in == rows_out:
            return knots_arr[0].copy() if squeeze else knots_arr.copy()
        row_src = np.linspace(0.0, 1.0, rows_in)
        row_dst = np.linspace(0.0, 1.0, rows_out)
        batch, dims, _, num_knots = knots_arr.shape
        resampled = np.zeros((batch, dims, rows_out, num_knots), dtype=knots_arr.dtype)
        for b in range(batch):
            for d in range(dims):
                for k in range(num_knots):
                    resampled[b, d, :, k] = np.interp(
                        row_dst,
                        row_src,
                        knots_arr[b, d, :, k],
                    )
        return resampled[0] if squeeze else resampled

    def _pytroch_optmized(self, min_dim: int) -> dict[str, object]:
        """Return a tuned parameter schedule for the optimized PyTorch backend."""
        if min_dim <= 160:
            return {
                "num_iterations": 8,
                "regularization_sigma_px": 4.0,
                "adam_steps": 10,
                "lr": 0.04,
                "pytorch_reference_mode": "leave_one_out",
                "pytorch_normalize_loss": False,
                "pytorch_row_stride": 1,
                "pytorch_multiscale": False,
                "pytorch_fast_schedule": False,
                "pytorch_learn_translation": False,
                "translation_interval": 0,
                "translation_upsample_factor": 4,
                "translation_downsample_factor": 1,
            }
        if min_dim <= 320:
            return {
                "num_iterations": 2,
                "adam_steps": 15,
                "lr": 0.02,
                "pytorch_reference_mode": "leave_one_out",
                "pytorch_normalize_loss": False,
                "pytorch_row_stride": 1,
                "pytorch_fast_schedule": False,
                "pytorch_multiscale": False,
                "pytorch_learn_translation": False,
                "translation_interval": 0,
                "translation_upsample_factor": 4,
                "translation_downsample_factor": 1,
            }
        if min_dim <= 768:
            return {
                "num_iterations": 2,
                "adam_steps": 15,
                "lr": 0.02,
                "pytorch_reference_mode": "leave_one_out",
                "pytorch_normalize_loss": False,
                "pytorch_row_stride": 1,
                "pytorch_fast_schedule": False,
                "pytorch_multiscale": False,
                "pytorch_learn_translation": False,
                "translation_interval": 0,
                "translation_upsample_factor": 4,
                "translation_downsample_factor": 2,
            }
        return {
            "num_iterations": 2,
            "adam_steps": 15,
            "lr": 0.02,
            "pytorch_reference_mode": "leave_one_out",
            "pytorch_normalize_loss": False,
            "pytorch_row_stride": 1,
            "pytorch_fast_schedule": False,
            "pytorch_multiscale": False,
            "pytorch_learn_translation": False,
            "translation_interval": 0,
            "translation_upsample_factor": 4,
            "translation_downsample_factor": 1,
        }

    def _pytroch_joint_rmse(self, min_dim: int) -> dict[str, object]:
        """Return an RMSE-focused joint schedule for the optimized backend."""
        base = {
            "regularization_sigma_px": 8.0,
            "regularization_update_step_size": 1.0,
            "pytorch_reference_mode": "leave_one_out",
            "pytorch_normalize_loss": False,
            "pytorch_row_stride": 1,
            "pytorch_fast_schedule": False,
            "pytorch_multiscale": True,
            "pytorch_refine_lr_scale": 0.5,
            "pytorch_refine_normalize_loss": False,
            "pytorch_refine_row_stride": 1,
            "translation_interval": 0,
            "translation_upsample_factor": 4,
            "translation_downsample_factor": 1,
        }
        if min_dim <= 320:
            return {
                **base,
                "num_iterations": 10,
                "adam_steps": 200,
                "lr": 0.008,
                "pytorch_refine_steps": 80,
            }
        if min_dim <= 768:
            return {
                **base,
                "num_iterations": 8,
                "adam_steps": 180,
                "lr": 0.008,
                "pytorch_refine_steps": 70,
            }
        return {
            **base,
            "num_iterations": 6,
            "adam_steps": 160,
            "lr": 0.008,
            "pytorch_refine_steps": 60,
        }

    def _optimize_knots_pytorch_optimized(
        self,
        image_refs: np.ndarray | torch.Tensor | None,
        knots_init: np.ndarray,
        torch_cache: dict[str, torch.Tensor | tuple[int, int]] | None = None,
        adam_steps: int = 5,
        lr: float = 0.02,
        use_amp: bool = False,
        normalize_loss: bool = True,
        row_stride: int = 1,
        learn_translation: bool = False,
        translation_center: bool = True,
        translation_penalty: float | None = None,
        learn_affine: bool = False,
        affine_center: bool = True,
        affine_penalty: float | None = None,
    ) -> np.ndarray:
        """Optimized PyTorch backend (batched over images, single knot only)."""
        if knots_init.shape[3] != 1:
            raise NotImplementedError(
                "pytorch_optimized/pytorch_joint backend only supports single knot per scanline."
            )
        if torch_cache is None:
            raise ValueError("torch_cache is required for pytorch_optimized backend.")
        target_images = torch_cache["target_images"]
        base_x = torch_cache["base_x"]
        base_y = torch_cache["base_y"]
        out_shape = torch_cache["out_shape"]
        out_h, out_w = out_shape
        device = target_images.device
        dtype = target_images.dtype

        if image_refs is None:
            raise ValueError("image_refs is required for pytorch_optimized backend.")
        if isinstance(image_refs, torch.Tensor):
            ref_images = image_refs.to(device=device, dtype=dtype)
        else:
            ref_images = torch.tensor(image_refs, dtype=dtype, device=device)
        if normalize_loss:
            target = torch_cache.get("target_norm")
            if target is None:
                target_mean = target_images.mean(dim=(1, 2), keepdim=True)
                target_std = target_images.std(dim=(1, 2), keepdim=True, unbiased=False)
                target = (target_images - target_mean) / target_std.clamp_min(1e-6)
        else:
            target = target_images
        row_stride = max(1, int(row_stride))
        rows = knots_init.shape[2]
        row_idx = np.arange(0, rows, row_stride)
        if row_idx[-1] != rows - 1:
            row_idx = np.hstack((row_idx, rows - 1))
        row_idx_t = torch.tensor(row_idx, device=device)
        row_centered = row_idx_t.to(dtype) - (rows - 1) / 2
        knots = torch.tensor(
            knots_init[:, :, row_idx, 0],
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        translation = None
        if learn_translation:
            translation = torch.zeros(
                (knots.shape[0], 2),
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
        affine = None
        if learn_affine:
            affine = torch.zeros(
                (knots.shape[0], 2),
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
        target = target.index_select(1, row_idx_t)
        params = [knots]
        if translation is not None:
            params.append(translation)
        if affine is not None:
            params.append(affine)
        optimizer = torch.optim.Adam(params, lr=lr)
        use_amp = use_amp and device.type == "cuda"
        amp_ctx = (
            torch.autocast(device_type=device.type, dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with amp_ctx:
            for _ in range(adam_steps):
                optimizer.zero_grad()
                row_offsets = knots
                if affine is not None:
                    row_offsets = row_offsets + affine[:, :, None] * row_centered[None, None, :]
                if translation is not None:
                    row_offsets = row_offsets + translation[:, :, None]
                xa = row_offsets[:, 0, :, None] + base_x[:, None, :]
                ya = row_offsets[:, 1, :, None] + base_y[:, None, :]
                xa = xa.clamp(0, out_h - 1.001)
                ya = ya.clamp(0, out_w - 1.001)
                if device.type != "mps":
                    grid_x = ya / (out_w - 1) * 2 - 1
                    grid_y = xa / (out_h - 1) * 2 - 1
                    grid = torch.stack((grid_x, grid_y), dim=-1)
                    warped = F.grid_sample(
                        ref_images[:, None, :, :],
                        grid,
                        mode="bilinear",
                        padding_mode="border",
                        align_corners=True,
                    )[:, 0]
                else:
                    xf = xa.floor().long().clamp(0, out_h - 2)
                    yf = ya.floor().long().clamp(0, out_w - 2)
                    dx = xa - xf.float()
                    dy = ya - yf.float()
                    b = torch.arange(ref_images.shape[0], device=device)[:, None, None]
                    warped = (
                        ref_images[b, xf, yf] * (1 - dx) * (1 - dy)
                        + ref_images[b, xf + 1, yf] * dx * (1 - dy)
                        + ref_images[b, xf, yf + 1] * (1 - dx) * dy
                        + ref_images[b, xf + 1, yf + 1] * dx * dy
                    )
                if normalize_loss:
                    warped_mean = warped.mean(dim=(1, 2), keepdim=True)
                    warped_std = warped.std(dim=(1, 2), keepdim=True, unbiased=False)
                    warped_norm = (warped - warped_mean) / warped_std.clamp_min(1e-6)
                    loss = ((warped_norm - target) ** 2).mean()
                else:
                    loss = ((warped - target) ** 2).mean()
                if translation is not None and translation_penalty is not None:
                    loss = loss + translation_penalty * (translation**2).mean()
                if affine is not None and affine_penalty is not None:
                    loss = loss + affine_penalty * (affine**2).mean()
                loss.backward()
                optimizer.step()
                if translation is not None and translation_center:
                    with torch.no_grad():
                        translation -= translation.mean(dim=0, keepdim=True)
                if affine is not None and affine_center:
                    with torch.no_grad():
                        affine -= affine.mean(dim=0, keepdim=True)

        row_offsets = knots
        if affine is not None:
            row_offsets = row_offsets + affine[:, :, None] * row_centered[None, None, :]
        if translation is not None:
            row_offsets = row_offsets + translation[:, :, None]
        knots_np = row_offsets.detach().cpu().numpy()
        if row_stride == 1 and len(row_idx) == rows:
            return knots_np[:, :, :, None]

        knots_full = np.zeros((knots_np.shape[0], knots_np.shape[1], rows), dtype=knots_np.dtype)
        full_rows = np.arange(rows)
        for img_idx in range(knots_np.shape[0]):
            for dim in range(knots_np.shape[1]):
                knots_full[img_idx, dim] = np.interp(full_rows, row_idx, knots_np[img_idx, dim])
        return knots_full[:, :, :, None]

    def _optimize_knots_scipy(
        self,
        idx: int,
        image_ref: np.ndarray,
        knots_init: np.ndarray,
        max_optimize_iterations: int = 10,
        solve_individual_rows: bool = True,
    ) -> np.ndarray:
        """SciPy L-BFGS optimization for one image."""
        shape_knots = knots_init.shape
        options = {"maxiter": max_optimize_iterations} if max_optimize_iterations else {}
        if solve_individual_rows:
            knots_updated = np.zeros_like(knots_init)
            for row_ind in range(knots_init.shape[1]):
                x0 = knots_init[:, row_ind, :].ravel()

                def cost_function(x):
                    knots_row = x.reshape(shape_knots[0], shape_knots[2])
                    xa, ya = self.interpolator[idx].transform_rows(knots_row)
                    xf = np.clip(np.floor(xa).astype(int), 0, self.shape[1] - 2)
                    yf = np.clip(np.floor(ya).astype(int), 0, self.shape[2] - 2)
                    dx, dy = xa - xf, ya - yf
                    warped = (
                        image_ref[xf, yf] * (1 - dx) * (1 - dy)
                        + image_ref[xf + 1, yf] * dx * (1 - dy)
                        + image_ref[xf, yf + 1] * (1 - dx) * dy
                        + image_ref[xf + 1, yf + 1] * dx * dy
                    )
                    return np.sum((warped - self.images[idx].array[row_ind, :]) ** 2)

                result = minimize(cost_function, x0, method="L-BFGS-B", options=options)
                knots_updated[:, row_ind, :] = result.x.reshape((2, -1))
        else:
            x0 = knots_init.ravel()

            def cost_function(x):
                knots = x.reshape(shape_knots)
                xa, ya = self.interpolator[idx].transform_coordinates(knots)
                xf = np.clip(np.floor(xa).astype(int), 0, self.shape[1] - 2)
                yf = np.clip(np.floor(ya).astype(int), 0, self.shape[2] - 2)
                dx, dy = xa - xf, ya - yf
                warped = (
                    image_ref[xf, yf] * (1 - dx) * (1 - dy)
                    + image_ref[xf + 1, yf] * dx * (1 - dy)
                    + image_ref[xf, yf + 1] * (1 - dx) * dy
                    + image_ref[xf + 1, yf + 1] * dx * dy
                )
                return np.sum((warped - self.images[idx].array) ** 2)

            result = minimize(cost_function, x0, method="L-BFGS-B", options=options)
            knots_updated = result.x.reshape(shape_knots)
        return knots_updated

    def generate_corrected_image(
        self,
        upsample_factor: int = 2,
        output_original_shape: bool = True,
        mask_output: bool = True,
        mask_edge_blend: float = 8.0,
        fourier_filter: bool = True,
        filter_midpoint: float = 0.5,
        kde_sigma: float = 0.5,
        weight_thresh=0.1,
        show_image: bool = True,
        **kwargs,
    ):
        """
        Generate the final drift-corrected image after aligning a stack of input images.

        Parameters
        ----------
        upsample_factor : int, default 2
            Factor to upsample the output image for enhanced interpolation accuracy.
        output_original_shape : bool, default True
            If True, crop the output image back to the original input dimensions after processing.
        mask_output : bool, default True
            If true, mask the output using the probe position weights
        mask_edge_blend : float, default 8.0
            Value in pixels to blend from the edge of the mask (where we have data)
        fourier_filter : bool, default True
            Whether to apply Fourier-based directional filtering to merge corrected images.
        filter_midpoint : float, default 0.5
            Midpoint for the sigmoid-based Fourier weighting filter, determining transition smoothness.
            Setting this to a low value close to 0 will include more signal but also more slow scan artifacts.
            If using 2 images at 0 and 90 degrees scan angles, any value >0.75 will be unstable.
            Only use larger values (close to 1.0) if multiple images covering many scan angles are used.
        kde_sigma : float, default 0.5
            Standard deviation for kernel density estimation used during image interpolation. Defaults
            to the object's stored kde_sigma if set to None.
        weight_thresh: float, default 0.1
            This value sets the threshold for masking the outputs.
            For very large jitter artifacts this value can be lowered.
        show_image : bool, default True
            Whether to display the final corrected image after processing.
        **kwargs : dict
            Additional keyword arguments passed to the plotting function when displaying the image.

        Returns
        -------
        image_corr : Dataset2d
            The final drift-corrected output image encapsulated in a Dataset2d object.

        Notes
        -----
        - The function applies per-frame warping using knot-based interpolation and optionally
          performs directional Fourier filtering to blend multiple warped images.
        - The Fourier filter suppresses directional artifacts by weighting image contributions based
          on their scan angles, utilizing a bounded sine sigmoid for smooth transition.
        - Upsampling enhances interpolation precision but may increase computational cost.
        """

        # init
        stack_corr = np.zeros(
            (
                self.shape[0],
                np.round(self.shape[1] * upsample_factor).astype("int"),
                np.round(self.shape[2] * upsample_factor).astype("int"),
            )
        )
        weight_corr = np.zeros(
            (
                self.shape[0],
                np.round(self.shape[1] * upsample_factor).astype("int"),
                np.round(self.shape[2] * upsample_factor).astype("int"),
            )
        )

        if kde_sigma is None:
            kde_sigma = self.kde_sigma

        # Update images
        for ind in range(self.shape[0]):
            stack_corr[ind], weight_corr[ind] = self.interpolator[ind].warp_image(
                self.images[ind].array,
                self.knots[ind],
                kde_sigma=kde_sigma,
                upsample_factor=upsample_factor,
            )

        if fourier_filter:
            # Apply fourier filtering
            kx = np.fft.fftfreq(stack_corr.shape[1])[:, None]
            ky = np.fft.fftfreq(stack_corr.shape[2])[None, :]
            kt = np.arctan2(ky, kx)

            stack_fft = np.fft.fft2(stack_corr)
            weights = np.zeros_like(stack_corr)

            for ind in range(stack_corr.shape[0]):
                # Calculate weights as a function of angle
                weights[ind] = np.abs(
                    np.mod((kt - self.scan_direction[ind]) / np.pi + 0.5, 1.0) - 0.5
                ) / (1 / 2)
                weights[ind][0, 0] = 1.0

                # Apply sigmoid to weighting function
                weights[ind] = bounded_sine_sigmoid(
                    weights[ind],
                    midpoint=filter_midpoint,
                )

                # Weight the fourier transformed images
                stack_fft[ind] *= weights[ind]

            weights_sum = np.sum(weights, axis=0)
            image_corr_fft = np.divide(
                np.sum(stack_fft, axis=0),
                weights_sum,
                where=weights_sum > 0.0,
            )

        else:
            image_corr_fft = np.fft.fft2(np.mean(stack_corr, axis=0))

        if mask_output:
            # Note that we compute 2 boolean masks to round off the corners of image blending

            # calculate mask from product of individual image masks
            # scale weights by upsample factor to normalize to mean value of 1.0
            mask_edge = np.prod(weight_corr >= (weight_thresh / upsample_factor**2), axis=0)

            # Set outermost pixels to False to define the boundary for edge blending
            mask_edge[:, 0] = False
            mask_edge[:, -1] = False
            mask_edge[0, :] = False
            mask_edge[-1, :] = False

            # Find inner boundary mask
            mask_inner = distance_transform_edt(mask_edge) <= mask_edge_blend

            # compute mask using edge blending value
            mask = (
                np.cos(
                    (np.pi / 2)
                    * np.clip(distance_transform_edt(mask_inner) / mask_edge_blend, 0.0, 1.0)
                )
                ** 2
            )

            # Mean pad value
            pad_value_mean = np.mean([ind.pad_value for ind in self.interpolator])

            # apply mask
            image_corr_fft = np.fft.fft2(
                np.fft.ifft2(image_corr_fft) * mask + pad_value_mean * (1 - mask)
            )

        if output_original_shape:
            image_corr_fft = fourier_cropping(image_corr_fft, self.shape[-2:]) / upsample_factor**2

        # TODO - adjust origin / sampling if output sampling is different from input
        # i.e. if output_original_shape is False, and upsample_factor > 1
        image_corr = Dataset2d.from_array(
            np.real(np.fft.ifft2(image_corr_fft)),
            name="drift corrected image",
            origin=self.images[0].origin,
            sampling=self.images[0].sampling,
            units=self.images[0].units,
        )

        if show_image:
            fig, ax = show_2d(image_corr.array, **kwargs)

            # Force a render whether we're drawing into a provided Axes or a fresh Figure
            ax_to_draw = kwargs.get("ax", ax)
            try:
                ax_to_draw.figure.canvas.draw_idle()
                # If we're not drawing into a caller-provided Axes, also pop the window
                if "ax" not in kwargs:
                    plt.show()
            except Exception:
                # Fallback: if backend is odd, try a blocking show
                plt.show()

        # if show_image:
        #     fig, ax = image_corr.show(**kwargs)

        return image_corr

    def calculate_error(
        self,
        mode,
    ):
        # Estimate current error
        images_mean = np.mean(self.images_warped.array, axis=0)
        sig_diff = np.mean(np.abs(self.images_warped.array - images_mean[None, :, :]), axis=(1, 2))

        # Error vector
        error_current = np.hstack((mode, np.mean(sig_diff), sig_diff))

        # Initialize or append to error tracking array
        if not hasattr(self, "error_track"):
            self.error_track = error_current[None, :]  # initialize with first row
        else:
            self.error_track = np.vstack((self.error_track, error_current))

    def plot_transformed_images(self, show_knots: bool = True, **kwargs):
        fig, ax = show_2d(
            list(self.images_warped.array),
            **kwargs,
        )
        if show_knots:
            for a0 in range(self.shape[0]):
                x = self.knots[a0][0]
                y = self.knots[a0][1]
                ax[a0].plot(
                    y,
                    x,
                    color="r",
                )

    def plot_convergence(
        self,
        figsize=(8, 3),
        **kwargs,
    ):
        """
        Plot the convergence of the drift correction.
        """
        sub = np.abs(self.error_track[:, 0] - 2) < 0.1
        error = self.error_track[:, 1]
        it = np.arange(error.shape[0])

        from matplotlib.ticker import FormatStrFormatter, MaxNLocator

        fig, ax = plt.subplots(1, 2, figsize=figsize)
        color = (1, 0, 0)  # red

        # Plot Affine
        if np.any(~sub):
            ax[0].plot(
                it[~sub],
                100 * error[~sub],
                marker="o",
                color=color,
                linestyle="-",
                label="Affine",
                **kwargs,
            )
            ax[0].set_xlabel("Affine Iterations")
            ax[0].set_ylabel("Mean Error [%]")
            ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        else:
            ax[0].axis("off")

        # Plot Non-Rigid
        if np.any(sub):
            first_true = np.argmax(sub)
            if first_true > 0:
                sub[first_true - 1] = True

            ax[1].plot(
                it[sub],
                100 * error[sub],
                marker="o",
                color=color,
                linestyle="-",
                label="Non-Rigid",
                **kwargs,
            )
            ax[1].set_xlabel("Non-Rigid Iterations")
            ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        else:
            ax[1].axis("off")

        plt.tight_layout()

        return self

    def plot_merged_images(self, show_knots: bool = True, **kwargs):
        """
        Plot the current transformed images, with knot overlays.
        """
        fig, ax = show_2d(
            self.images_warped.array.mean(0),
            **kwargs,
        )
        if show_knots:
            for a0 in range(self.shape[0]):
                x = self.knots[a0][0]
                y = self.knots[a0][1]
                ax.plot(
                    y,
                    x,
                )


class DriftInterpolator:
    def __init__(
        self,
        input_shape,
        output_shape,
        scan_fast,
        scan_slow,
        pad_value,
        kde_sigma,
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.scan_fast = scan_fast
        self.scan_slow = scan_slow
        self.pad_value = pad_value
        self.kde_sigma = kde_sigma

        self.rows_input = np.arange(input_shape[0])
        self.cols_input = np.arange(input_shape[1])
        self.u = np.linspace(0, 1, input_shape[1])

    def transform_rows(
        self,
        knots_row: NDArray,
    ):
        num_knots = knots_row.shape[-1]
        basis = np.linspace(0, 1, num_knots)

        if num_knots == 1:
            xa = knots_row[0] + self.u[None, :] * self.scan_fast[0] * (self.input_shape[0] - 1)
            ya = knots_row[1] + self.u[None, :] * self.scan_fast[1] * (self.input_shape[1] - 1)
        elif num_knots == 2:
            xa = interp1d(basis, knots_row[0], kind="linear", assume_sorted=True)(self.u)
            ya = interp1d(basis, knots_row[1], kind="linear", assume_sorted=True)(self.u)
        else:
            kind = "quadratic" if num_knots == 3 else "cubic"
            xa = interp1d(
                basis,
                knots_row[0],
                kind=kind,
                fill_value="extrapolate",
                assume_sorted=True,
            )(self.u)
            ya = interp1d(
                basis,
                knots_row[1],
                kind=kind,
                fill_value="extrapolate",
                assume_sorted=True,
            )(self.u)

        return xa, ya

    def transform_coordinates(
        self,
        knots: NDArray,
    ):
        num_knots = knots.shape[-1]

        if num_knots == 1:
            # vectorized version for speed
            xa, ya = self.transform_rows(knots)
        else:
            xa = np.zeros(self.input_shape)
            ya = np.zeros(self.input_shape)
            for i in range(self.input_shape[0]):
                xa[i], ya[i] = self.transform_rows(knots[:, i])

        return xa, ya

    def warp_image(
        self,
        image: NDArray,
        knots: NDArray,  # shape: (2, rows, num_knots)
        kde_sigma=None,
        output_shape=None,
        pad_value=None,
        upsample_factor=None,
    ) -> NDArray:
        xa, ya = self.transform_coordinates(
            knots,
        )

        if kde_sigma is None:
            kde_sigma = self.kde_sigma

        if output_shape is None:
            output_shape = self.output_shape

        if pad_value is None:
            pad_value = self.pad_value

        if upsample_factor is None:
            upsample_factor = 1.0

        image_interp, weight_interp = bilinear_kde(
            xa=xa * upsample_factor,  # rows
            ya=ya * upsample_factor,  # cols
            values=image,
            output_shape=np.round(np.array(output_shape) * upsample_factor).astype("int"),
            kde_sigma=kde_sigma * upsample_factor,
            pad_value=pad_value,
            return_pix_count=True,
        )

        return image_interp, weight_interp


def bounded_sine_sigmoid(x, midpoint=0.5, width=1.0):
    """
    Piecewise bounded sigmoid: zero, raised sine squared, one.

    Parameters
    ----------
    x : array-like, shape (...,)
        Input values in [0, 1].
    midpoint : float
        Center of the sigmoid transition.
    width : float
        Width of the sigmoid (range over which it ramps from 0 to 1).
    Returns
    -------
    y : array-like
        Output in [0, 1], same shape as x.
    """
    x = np.asarray(x)
    # Truncate width if midpoint too close to edge
    left_max = midpoint - width / 2
    right_min = midpoint + width / 2
    if left_max < 0:
        warnings.warn(
            f"width={width} is too large for midpoint={midpoint}, "
            f"clamping width to {2 * midpoint}.",
            RuntimeWarning,
        )
        width = 2 * midpoint

    if right_min > 1:
        warnings.warn(
            f"width={width} is too large for midpoint={midpoint}, "
            f"clamping width to {2 * (1 - midpoint)}.",
            RuntimeWarning,
        )
        width = 2 * (1 - midpoint)
    # Recalculate edges
    left = midpoint - width / 2
    right = midpoint + width / 2

    y = np.zeros_like(x, dtype=float)
    in_band = (x >= left) & (x <= right)
    # Map [left, right] to [0, pi/2]
    t = (x[in_band] - left) / width  # goes from 0 to 1
    y[in_band] = np.sin(t * np.pi / 2) ** 2
    y[x > right] = 1.0
    return y
