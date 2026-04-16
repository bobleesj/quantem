from collections.abc import Sequence
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.fft import fftfreq
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.optimize import minimize
from tqdm import tqdm

from quantem.core.config import validate_device

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
)
from quantem.imaging.drift_utils import (
    backward_warp,
    backward_warp_grid_search,
    bilinear_kde_batch,
    cross_corr_batch,
    gaussian_smooth_1d,
    initialize_scanline_knots,
    transform_coordinates_single_knot,
    translate_align,
)
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization import show_2d
import quantem.imaging.drift_viz as drift_viz


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
    - The class stores resampled images in `self.imgs_warped` and the control knots in `self.knots`.
    - Visualization is supported through `plot_merged_images()` and `plot_warped_images()`.

    Performance
    -----------
    ``align_affine`` uses PyTorch to run all heavy operations on GPU
    (works on CUDA, MPS, and CPU). The key optimizations are:

    - **Batched grid search**: all ~97 candidate drift vectors are warped
      and scored in parallel, instead of one-at-a-time in a Python loop.
    - **Batched bilinear KDE** (``drift_utils.bilinear_kde_batch``):
      scatter-based image warping via ``scatter_add_`` with int32 indices.
    - **Batched FFT cross-correlation** (``drift_utils.cross_corr_batch``):
      sub-pixel alignment using DFT upsampling across all candidates at once.
    - **Zero CPU round-trips**: coordinate transforms, Gaussian smoothing,
      translation alignment, and error computation all stay on GPU until
      the final sync.

    This gives ~300× speedup over the original NumPy implementation
    (e.g. 436 s → 1.5 s on 2048×2048 image pairs).

    Memory is automatically chunked when the full batch doesn't fit.
    Approximate memory per candidate at common sizes:

    ========== =========== ================
    Input size Canvas size Mem / candidate
    ========== =========== ================
    1024×1024  1280×1280     85 MB
    2048×2048  2560×2560    341 MB
    4096×4096  5120×5120   1.36 GB
    ========== =========== ================
    """

    _token = object()

    def __init__(
        self,
        imgs: list[Dataset2d],
        scan_direction_degrees: NDArray,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use DriftCorrection.from_data() or .from_file() to instantiate this class."
            )

        self._frames: list[Self] | None = None
        self.imgs = imgs
        self.scan_direction_degrees = ensure_valid_array(scan_direction_degrees, ndim=1)

        device, _ = validate_device(None)
        self._device = device
        self._dtype = torch.float32

    # -- series helpers ---------------------------------------------------

    @classmethod
    def _from_frames(cls, frames: list[Self]) -> Self:
        """Construct a series wrapper around pre-built single-pair frames."""
        obj = cls.__new__(cls)
        obj._frames = frames
        obj.scan_direction_degrees = frames[0].scan_direction_degrees
        obj._device = frames[0]._device
        obj._dtype = frames[0]._dtype
        return obj

    @property
    def is_series(self) -> bool:
        """True if this instance wraps a multi-frame series."""
        return self._frames is not None

    @property
    def n_frames(self) -> int:
        """Number of frames in a series (raises TypeError for single pair)."""
        if self._frames is None:
            raise TypeError(
                "n_frames is only available on series instances. "
                "Create one with from_data([stack_a, stack_b], ...)."
            )
        return len(self._frames)

    def __getitem__(self, idx: int) -> Self:
        """Access the *idx*-th frame of a series for per-frame inspection."""
        if self._frames is None:
            raise TypeError(
                "Single-pair DriftCorrection is not indexable. "
                "Use from_data with 3-D stacks for series mode."
            )
        return self._frames[idx]

    def __iter__(self):
        if self._frames is None:
            raise TypeError("Single-pair DriftCorrection is not iterable.")
        return iter(self._frames)

    def _ensure_single(self, name: str) -> None:
        """Raise TypeError if called on a series instance."""
        if self._frames is not None:
            raise TypeError(
                f"{name} is not supported on series instances. "
                f"Use drift[i].{name} for individual frames."
            )

    @classmethod
    def from_file(
        cls,
        file_paths: Sequence[str],
        scan_direction_degrees: Sequence[float] | NDArray,
        file_type: str | None = None,
    ) -> Self:
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
    ) -> Self:
        # Detect series: a list/tuple of 3-D stacks or Dataset3d objects
        if isinstance(images, (list, tuple)) and len(images) >= 2:
            first = images[0]
            is_3d = (isinstance(first, np.ndarray) and first.ndim == 3) or isinstance(first, Dataset3d)
            if is_3d:
                stacks = []
                for img in images:
                    if isinstance(img, Dataset3d):
                        stacks.append(img.array)
                    elif isinstance(img, np.ndarray) and img.ndim == 3:
                        stacks.append(img)
                    else:
                        raise TypeError(
                            "For series mode all images must be 3-D stacks "
                            f"(N, H, W), got ndim={getattr(img, 'ndim', '?')}"
                        )
                n = stacks[0].shape[0]
                if n == 0:
                    raise ValueError(
                        "Series stacks must contain at least 1 frame, "
                        f"got shape {stacks[0].shape}"
                    )
                for i, s in enumerate(stacks):
                    if s.shape[0] != n:
                        raise ValueError(
                            f"Frame count mismatch: images[0] has {n} frames "
                            f"but images[{i}] has {s.shape[0]}"
                        )
                frames = [
                    cls.from_data([s[j] for s in stacks], scan_direction_degrees)
                    for j in range(n)
                ]
                return cls._from_frames(frames)

        validated_images = validate_list_of_dataset2d(images)
        return cls(
            imgs=validated_images,
            scan_direction_degrees=scan_direction_degrees,
            _token=cls._token,
        )

    def preprocess(
        self,
        pad_fraction: float = 0.25,
        pad_value: float | str | list[float] = "median",
        kde_sigma: float = 0.5,
        number_knots: int = 1,
        normalize: bool = False,
        show_merged: bool = False,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        """Prepare images for drift correction by building the scanline model.

        Computes scan direction vectors, initializes Bezier knots that map
        each scanline onto a padded canvas, and generates the initial warped
        images. This must be called before any alignment step.

        Without preprocessing, there is no spatial model connecting the raw
        images to the shared canvas - alignment methods would have no
        coordinates to optimize.

        Parameters
        ----------
        pad_fraction : float
            Fraction of the image size to add as padding around the canvas.
            Larger values give more room for drift but use more memory.
            ``pad_fraction=0.25`` adds 25% on each side.
        pad_value : float, str, or list[float]
            Fill value for pixels outside the image footprint. Can be
            ``'median'``, ``'mean'``, ``'min'``, ``'max'``, a quantile
            (e.g. ``0.25``), or a per-image list of floats.
        kde_sigma : float
            Gaussian smoothing sigma (in pixels) applied after bilinear
            scatter. Smooths the warped images to reduce scatter noise.
        number_knots : int
            Number of Bezier knots per scanline. Use ``1`` (recommended)
            for linear drift correction. Higher values allow per-scanline
            curvature but are slower and rarely needed.
        normalize : bool
            If True, min-max normalize each image to ``[0, 1]`` before
            warping.  Recommended when aligning images with different
            intensity scales (e.g. HAADF reference vs 4D-STEM virtual
            dark-field) so the MAE cost function treats both equally.
            For same-detector pairs (e.g. 0°/90°) this is unnecessary.
        show_merged : bool
            Display the merged (averaged) warped images after preprocessing.
        show_images : bool
            Display each individual warped image after preprocessing.
        show_knots : bool
            Overlay knot positions on displayed images.
        **kwargs
            Additional keyword arguments passed to plotting functions.

        Returns
        -------
        Self
            For method chaining: ``drift.preprocess().align_affine()``.

        Examples
        --------
        >>> drift = DriftCorrection.from_data(
        ...     images=[im0, im1], scan_direction_degrees=[0, 90])
        >>> drift.preprocess(pad_fraction=0.25, kde_sigma=0.5, number_knots=1)

        For mixed-type images (HAADF + VDF), use normalize:

        >>> drift = DriftCorrection.from_data(
        ...     images=[haadf_ref, vdf], scan_direction_degrees=[0, 0])
        >>> drift.preprocess(normalize=True).align_affine(fixed_indices=[0])
        """
        if self._frames is not None:
            kw = {k: v for k, v in locals().items() if k != "self"}
            kw.update(kw.pop("kwargs"))
            kw["show_merged"] = False
            kw["show_images"] = False
            for f in tqdm(self._frames, desc="Preprocessing series"):
                f.preprocess(**kw)
            return self

        self._normalized = bool(normalize)
        if normalize:
            for img in self.imgs:
                arr = img.array.astype(np.float32)
                lo, hi = arr.min(), arr.max()
                img.array = (arr - lo) / (hi - lo + 1e-8)
        self.pad_fraction = float(pad_fraction)
        self.pad_value = validate_pad_value(pad_value, self.imgs)
        self.kde_sigma = float(kde_sigma)
        self.number_knots = int(number_knots)
        self.scan_direction = np.deg2rad(self.scan_direction_degrees)
        self.scan_fast = np.stack(
            [np.sin(-self.scan_direction), np.cos(-self.scan_direction)], axis=1)
        self.scan_slow = np.stack(
            [np.cos(-self.scan_direction), -np.sin(-self.scan_direction)], axis=1)
        self.shape = (
            len(self.imgs),
            int(np.round(self.imgs[0].shape[0] * (1 + self.pad_fraction) / 2) * 2),
            int(np.round(self.imgs[0].shape[1] * (1 + self.pad_fraction) / 2) * 2),
        )
        # Initialize knots - each image's scanlines mapped to the padded canvas
        self.knots = [
            torch.tensor(
                initialize_scanline_knots(
                    input_shape=self.imgs[img_idx].shape,
                    output_shape=self.shape[1:],
                    scan_fast=self.scan_fast[img_idx],
                    scan_slow=self.scan_slow[img_idx],
                    number_knots=self.number_knots,
                ),
                dtype=self._dtype,
                device=self._device,
            )
            for img_idx in range(self.shape[0])
        ]
        self.interpolator = [
            _DriftInterpolator(
                input_shape=self.imgs[i].shape,
                output_shape=self.shape[1:],
                scan_fast=self.scan_fast[i],
                pad_value=self.pad_value[i],
                kde_sigma=self.kde_sigma,
            )
            for i in range(self.shape[0])
        ]
        # Cache source data on GPU and generate initial warped images
        device = self._device
        dtype = self._dtype
        self.imgs_t = [
            torch.tensor(self.imgs[i].array, dtype=dtype, device=device)
            for i in range(self.shape[0])
        ]
        self.scan_fast_t = [
            torch.tensor(self.scan_fast[i], dtype=dtype, device=device)
            for i in range(self.shape[0])
        ]
        self.imgs_warped = Dataset3d.from_shape(self.shape)
        self.weights_warped = Dataset3d.from_shape(self.shape)
        canvas_shape = (self.shape[1], self.shape[2])
        warped_t = torch.zeros(self.shape[0], *canvas_shape, dtype=dtype, device=device)
        for img_idx in range(self.shape[0]):
            row_t, col_t = transform_coordinates_single_knot(
                self.knots[img_idx], self.scan_fast_t[img_idx], self.imgs[img_idx].shape)
            warped, weights = bilinear_kde_batch(
                row_t[None], col_t[None], self.imgs_t[img_idx], canvas_shape,
                self.kde_sigma, self.pad_value[img_idx])
            warped_t[img_idx] = warped[0]
            self.imgs_warped.array[img_idx] = warped[0].cpu().numpy()
            self.weights_warped.array[img_idx] = weights[0].cpu().numpy()
        self._initial_knots = [k.clone() for k in self.knots]
        self.calculate_error(0, _warped_t=warped_t)
        kwargs.pop("title", None)
        if show_merged:
            self.plot_merged_images(show_knots=show_knots, title="Merged: initial", **kwargs)
        if show_images:
            self.plot_warped_images(
                show_knots=show_knots,
                title=[f"Image {i}: initial" for i in range(self.shape[0])],
                **kwargs,
            )
        return self

    def align_translation(
        self,
        upsample_factor: int = 8,
        min_image_shift: float | None = None,
        max_image_shift: float = 32,
        show_merged: bool = True,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        """
        Solve for the translation between all images in DriftCorrection.imgs_warped
        """
        if self._frames is not None:
            kw = {k: v for k, v in locals().items() if k != "self"}
            kw.update(kw.pop("kwargs"))
            kw["show_merged"] = False
            kw["show_images"] = False
            for f in tqdm(self._frames, desc="Aligning translation"):
                f.align_translation(**kw)
            return self

        shifts = np.zeros((self.shape[0], 2))
        F_ref = np.fft.fft2(self.imgs_warped.array[0])
        for img_idx in range(1, self.shape[0]):
            shift, image_shift = cross_correlation_shift(
                F_ref,
                np.fft.fft2(self.imgs_warped.array[img_idx]),
                upsample_factor=upsample_factor,
                max_shift=max_image_shift,
                fft_input=True,
                fft_output=True,
                return_shifted_image=True,
            )
            shifts[img_idx, :] = shift
            F_ref = F_ref * img_idx / (img_idx + 1) + image_shift / (img_idx + 1)
        shifts -= np.mean(shifts, axis=0)
        if min_image_shift is not None:
            for img_idx in range(1, self.shape[0]):
                if np.linalg.norm(shifts[img_idx]) < min_image_shift:
                    shifts[img_idx] = 0.0
        for img_idx in range(self.shape[0]):
            self.knots[img_idx][0] += shifts[img_idx, 0]
            self.knots[img_idx][1] += shifts[img_idx, 1]
        for img_idx in range(self.shape[0]):
            self.imgs_warped.array[img_idx], self.weights_warped.array[img_idx] = self.interpolator[
                img_idx
            ].warp_image(
                self.imgs[img_idx].array,
                self.knots[img_idx].cpu().numpy(),
            )
        kwargs.pop("title", None)
        if show_merged:
            self.plot_merged_images(show_knots=show_knots, title="Merged: translation", **kwargs)
        if show_images:
            self.plot_warped_images(
                show_knots=show_knots,
                title=[f"Image {i}: translation" for i in range(self.shape[0])],
                **kwargs,
            )
        return self

    def align_affine(
        self,
        step: float = 0.01,
        num_tests: int = 9,
        refine: bool = True,
        upsample_factor: int = 8,
        max_image_shift: float | None = 32,
        chunk_size: int | None = None,
        fixed_indices: list[int] | None = None,
        show_merged: bool = True,
        show_images: bool = False,
        show_knots: bool = True,
        verbose: bool = False,
        **kwargs,
    ):
        """Correct affine drift between scan pairs using a batched grid search.

        Builds a grid of candidate linear-drift vectors, warps both images
        for each candidate, and picks the one with the lowest cross-correlation
        cost. An optional refinement pass subdivides the winning cell for
        sub-step accuracy. Without affine correction, per-scanline drift
        causes shear distortion that translation alignment alone cannot fix.

        Parameters
        ----------
        step : float
            Search resolution in pixels per scan line. The grid search
            tests drift rates from ``-step * num_tests/2`` to
            ``+step * num_tests/2`` px/line. For example, ``step=0.02``
            with ``num_tests=11`` searches drifts from -0.10 to +0.10
            px/line. Smaller values detect subtler drift but test more
            candidates.
        num_tests : int
            Number of drift rates to test along each axis. Must be odd
            so the grid is centered on zero drift. Total candidates
            ≈ ``π/4 * num_tests²``: ``num_tests=5`` → 21,
            ``num_tests=9`` → 61, ``num_tests=11`` → 97.
        refine : bool
            If True, run a second pass at ``step / (num_tests - 1)``
            resolution, centered on the coarse winner.
        upsample_factor : int
            Sub-pixel precision for measuring the translational shift
            between warped image pairs. 8 means 1/8-pixel precision.
            Higher values are more accurate but slower.
        max_image_shift : float or None
            Maximum allowed translational shift in pixels. Cross-correlation
            peaks beyond this radius are masked to reject spurious matches
            from noise or periodic artifacts. Set to None to allow any shift.
        chunk_size : int or None
            Number of candidates per pass. If None, all candidates at once.
            Set to a smaller value if you run out of memory.
        fixed_indices : list[int] or None
            Indices of images whose knots should never be modified.
            Use ``fixed_indices=[0]`` for single-sided alignment where
            image 0 is a fixed reference (e.g. a merged HAADF) and only
            the remaining images are optimized. When ``None`` (default),
            all images receive the affine drift correction — the standard
            behaviour for 0°/90° scan pairs.
        show_merged : bool
            Display the merged (averaged) image after alignment.
        show_images : bool
            Display each individual warped image after alignment.
        show_knots : bool
            Overlay knot positions on the displayed images.
        verbose : bool
            If True, print the top 5 candidate drift vectors with their
            cost and direction after the grid search. Useful for
            diagnosing ambiguous alignments or verifying the winning
            candidate has a clear margin over runner-ups.
        **kwargs
            Additional keyword arguments passed to the plotting functions.

        Returns
        -------
        DriftCorrection
            Self, for method chaining.

        Examples
        --------
        >>> drift = DriftCorrection.from_data(
        ...     images=[im0, im1], scan_direction_degrees=[0, 90])
        >>> drift.preprocess().align_affine(step=0.02, num_tests=11)

        Single-sided alignment (4D-STEM VDF against a fixed HAADF reference):

        >>> drift = DriftCorrection.from_data(
        ...     images=[haadf_ref, vdf], scan_direction_degrees=[0, 0])
        >>> drift.preprocess().align_affine(fixed_indices=[0])
        """
        if self._frames is not None:
            kw = {k: v for k, v in locals().items() if k != "self"}
            kw.update(kw.pop("kwargs"))
            kw["show_merged"] = False
            kw["show_images"] = False
            for f in tqdm(self._frames, desc="Aligning affine"):
                f.align_affine(**kw)
            return self

        if self.shape[0] < 2:
            raise ValueError(
                f"align_affine requires at least 2 images (got {self.shape[0]}). "
                f"Provide image pairs with different scan directions."
            )
        if num_tests % 2 == 0:
            raise ValueError(
                f"num_tests must be odd (got {num_tests}). Try {num_tests + 1}."
            )
        fixed_set = frozenset(fixed_indices) if fixed_indices is not None else frozenset()
        # Build candidate grid with circular mask (~21% fewer than square)
        grid_axis = np.arange(-(num_tests - 1) / 2, (num_tests + 1) / 2)
        row_grid, col_grid = np.meshgrid(grid_axis, grid_axis, indexing="ij")
        circular_mask = row_grid**2 + col_grid**2 <= (num_tests / 2) ** 2
        drift_vectors = np.vstack((row_grid[circular_mask], col_grid[circular_mask])).T * step

        def _print_top_candidates(label, candidates, costs_tensor):
            costs_np = costs_tensor.cpu().numpy()
            ranked = np.argsort(costs_np)
            best_cost = costs_np[ranked[0]]
            print(f"  {label} - top 5 candidates:")
            for rank in range(min(5, len(ranked))):
                idx = ranked[rank]
                drift_row, drift_col = candidates[idx]
                magnitude = np.sqrt(drift_row**2 + drift_col**2)
                gap = (costs_np[idx] - best_cost) / best_cost * 100 if rank > 0 else 0
                print(f"    drift=({drift_row:+.4f}, {drift_col:+.4f}) px/line "
                      f"({magnitude:.4f} magnitude), cost={costs_np[idx]:.4f}"
                      f"{f' (+{gap:.1f}%)' if rank > 0 else ' (best)'}")

        def _apply_drift(drift_vec):
            for img_idx in range(self.shape[0]):
                if img_idx in fixed_set:
                    continue
                num_rows = self.knots[img_idx].shape[1]
                scanline_offset = torch.arange(num_rows, dtype=self.knots[img_idx].dtype,
                                               device=self.knots[img_idx].device) - (num_rows - 1) / 2
                self.knots[img_idx][0] += drift_vec[0] * scanline_offset[:, None]
                self.knots[img_idx][1] += drift_vec[1] * scanline_offset[:, None]

        def _search_and_apply(candidates, label, accumulated_drift=None):
            # When fixed_indices is set, backward_warp_grid_search scores
            # absolute drift rates on the raw images (not canvas-warped).
            # After the coarse pass, _apply_drift bakes the coarse drift into
            # the knots, but the raw images are unchanged — so the refine
            # candidates (small deltas) must be offset by the accumulated
            # drift so that backward_warp_grid_search tests the correct total
            # drift rates.
            search_candidates = candidates
            if fixed_set and accumulated_drift is not None:
                search_candidates = candidates + accumulated_drift[None, :]
            best_idx, costs = self._affine_grid_search_batch(
                search_candidates, upsample_factor, max_image_shift, chunk_size,
                fixed_indices=fixed_set)
            _apply_drift(candidates[best_idx])
            if verbose:
                _print_top_candidates(label, candidates, costs)
            warped_t = self._warp_and_translate_torch(
                max_image_shift, upsample_factor, fixed_indices=fixed_set)
            self.calculate_error(1, _warped_t=warped_t)

            # Confidence: cost gap between best and runner-up (%)
            costs_np = costs.cpu().numpy()
            ranked = np.argsort(costs_np)
            best_cost = costs_np[ranked[0]]
            runner_up = costs_np[ranked[1]] if len(ranked) > 1 else best_cost
            margin = (runner_up - best_cost) / (best_cost + 1e-12) * 100
            return candidates[best_idx], margin

        drift_total, coarse_margin = _search_and_apply(
            drift_vectors, "Coarse search"
        )
        if refine:
            drift_fine = drift_vectors / (num_tests - 1)
            dt, refine_margin = _search_and_apply(
                drift_fine, "Refine search", accumulated_drift=drift_total)
            drift_total = drift_total + dt
            self.affine_confidence_margin = refine_margin
        else:
            self.affine_confidence_margin = coarse_margin
        if verbose:
            num_rows = self.imgs[0].shape[0]
            drift_rate = np.sqrt(drift_total[0] ** 2 + drift_total[1] ** 2)
            total_shift = drift_rate * num_rows
            angle_deg = np.degrees(np.arctan2(drift_total[1], drift_total[0]))
            print(f"align_affine: step={step}, num_tests={num_tests} "
                  f"({len(drift_vectors)} candidates), refine={refine}, "
                  f"max_image_shift={max_image_shift}")
            msg = (f"Drift: ({drift_total[0]:+.4f}, {drift_total[1]:+.4f}) px/line, "
                   f"{drift_rate:.4f} magnitude, {angle_deg:.1f} deg, "
                   f"{total_shift:.1f} px total over {num_rows} lines")
            if self.imgs[0].sampling is not None:
                px_size = self.imgs[0].sampling[0]
                unit = self.imgs[0].units[0] if self.imgs[0].units else "px"
                msg += f" = {total_shift * px_size:.2f} {unit}"
            print(msg)
            err = self.error_track
            print(f"Error: {err[0, 1]:.2f} -> {err[-1, 1]:.2f} "
                  f"({(err[0, 1] - err[-1, 1]) / err[0, 1] * 100:+.1f}%)")
            margin = self.affine_confidence_margin
            confidence = "high" if margin > 5 else "low" if margin < 2 else "moderate"
            print(f"Confidence: {margin:.1f}% cost margin to runner-up ({confidence})")

        # Plots
        kwargs.pop("title", None)
        if show_merged:
            self.plot_merged_images(
                show_knots=show_knots,
                title="Merged: affine",
                **kwargs,
            )
        if show_images:
            self.plot_warped_images(
                show_knots=show_knots,
                title=[f"Image {i}: affine" for i in range(self.shape[0])],
                **kwargs,
            )

        self._knots_after_affine = [k.clone() for k in self.knots]

        return self

    @torch.inference_mode()
    def _affine_grid_search_batch(self, drift_vectors, upsample_factor, max_image_shift,
                                  chunk_size=None, fixed_indices=None):
        """Evaluate all candidate drift vectors in parallel.

        Warps both images for each candidate using ``bilinear_kde_batch``
        and scores alignment quality via ``cross_corr_batch``. Without
        batching, each candidate would be a separate Python iteration - this
        is the key operation that enables the 300x speedup.

        When ``fixed_indices`` is provided, images at those indices are
        warped once with their current knots (no candidate drift) and
        reused across all chunks. Only non-fixed images receive the
        candidate drift offsets.

        Parameters
        ----------
        drift_vectors : ndarray, shape (N, 2)
            Candidate drift vectors to test, columns are (row, col).
        upsample_factor : int
            Subpixel cross-correlation upsampling factor.
        max_image_shift : float or None
            Maximum allowed shift for cross-correlation peak search.
        chunk_size : int or None
            Number of candidates per pass. If None, all at once.
        fixed_indices : frozenset[int] or None
            Indices of images whose knots should not receive candidate
            drift. These images are warped once and reused.

        Returns
        -------
        tuple[int, torch.Tensor]
            Index of the best candidate in ``drift_vectors``, and the full
            cost tensor of shape ``(N,)`` for all candidates (used by
            verbose mode to rank runner-ups).
        """
        device = self._device
        dtype = self._dtype
        fixed_set = fixed_indices if fixed_indices else frozenset()
        num_candidates = drift_vectors.shape[0]
        drift_vectors_t = torch.tensor(drift_vectors, dtype=dtype, device=device)

        # When fixed_indices is set, use backward-warp scoring to avoid
        # KDE forward-scatter bias.  The periodic wrapping in
        # bilinear_kde_batch creates geometry-dependent seam artifacts
        # that differ between the fixed reference and drift-shifted
        # moving image, making the MAE minimum diverge from the true
        # drift.  Backward-warp scoring works at original resolution
        # with grid_sample (no canvas, no KDE).
        if fixed_set:
            fixed_idx = sorted(fixed_set)[0]
            moving_indices = [i for i in range(len(self.imgs_t)) if i not in fixed_set]
            if not moving_indices:
                raise ValueError("All images are fixed — nothing to optimize.")
            total_costs = None
            for mov_idx in moving_indices:
                _, costs = backward_warp_grid_search(
                    self.imgs_t[fixed_idx], self.imgs_t[mov_idx],
                    drift_vectors_t, upsample_factor, max_image_shift,
                    chunk_size)
                total_costs = costs if total_costs is None else total_costs + costs
            return torch.argmin(total_costs).item(), total_costs

        canvas_shape = (self.shape[1], self.shape[2])
        n_images = len(self.imgs_t)
        # Base coordinates shared across all candidates
        base_data = []
        for img_idx in range(n_images):
            row_base, col_base = transform_coordinates_single_knot(
                self.knots[img_idx], self.scan_fast_t[img_idx], self.imgs[img_idx].shape)
            num_rows = self.knots[img_idx].shape[1]
            scanline_offset = (torch.arange(num_rows, dtype=dtype, device=device)
                               - (num_rows - 1) / 2)
            base_data.append((self.imgs_t[img_idx], row_base, col_base, scanline_offset))
        # Precompute shift mask and frequency grids (shared across chunks)
        shift_mask = None
        if max_image_shift is not None:
            canvas_rows, canvas_cols = canvas_shape
            freq_row = fftfreq(canvas_rows, 1.0 / canvas_rows, device=device, dtype=dtype)
            freq_col = fftfreq(canvas_cols, 1.0 / canvas_cols, device=device, dtype=dtype)
            shift_mask = freq_row[:, None] ** 2 + freq_col[None, :] ** 2 >= max_image_shift ** 2
        freq_grids = (
            fftfreq(canvas_shape[0], device=device, dtype=dtype)[:, None],
            fftfreq(canvas_shape[1], device=device, dtype=dtype)[None, :],
        )
        if chunk_size is None:
            chunk_size = self._auto_chunk_size(num_candidates, canvas_shape, dtype, device)
        on_cuda = torch.device(device).type == "cuda"
        chunked = on_cuda and chunk_size < num_candidates
        all_costs = []
        chunk_start = 0
        chunk_idx = 0
        while chunk_start < num_candidates:
            chunk_end = min(chunk_start + chunk_size, num_candidates)
            drift_chunk = drift_vectors_t[chunk_start:chunk_end]
            if chunk_idx == 0 and chunked:
                torch.cuda.reset_peak_memory_stats(device)
            # Warp each image (fixed → expand once, moving → drift-shifted)
            warped_images = []
            for img_idx in range(n_images):
                image_t, row_base, col_base, scanline_offset = base_data[img_idx]
                row_candidates = row_base[None] + drift_chunk[:, 0, None, None] * scanline_offset[None, :, None]
                col_candidates = col_base[None] + drift_chunk[:, 1, None, None] * scanline_offset[None, :, None]
                warped, _ = bilinear_kde_batch(
                    row_candidates, col_candidates, image_t,
                    canvas_shape, self.kde_sigma,
                    self.pad_value[img_idx])
                warped_images.append(warped)
            # Score all unique pairs and sum costs
            chunk_cost = torch.zeros(chunk_end - chunk_start, dtype=dtype, device=device)
            for i in range(n_images):
                for j in range(i + 1, n_images):
                    chunk_cost += cross_corr_batch(
                        warped_images[i], warped_images[j],
                        upsample_factor,
                        max_shift_mask=shift_mask,
                        freq_grids=freq_grids)
            all_costs.append(chunk_cost)
            # After chunk 0, replace the conservative static estimate with the
            # actual measured per-candidate cost and print one summary line so
            # the user can see how the chunking adapted to their GPU state.
            if chunk_idx == 0 and chunked:
                per_candidate_actual = torch.cuda.max_memory_allocated(device) / chunk_size
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                tuned_chunk_size = max(1, int(free_bytes * 0.5 / per_candidate_actual))
                tuned_chunk_size = min(tuned_chunk_size, num_candidates)
                if tuned_chunk_size > chunk_size:
                    chunk_size = tuned_chunk_size
                num_chunks_final = 1 + (num_candidates - chunk_end + chunk_size - 1) // chunk_size
                print(
                    f"  affine grid: {num_candidates} cand × {canvas_shape[0]}×{canvas_shape[1]}, "
                    f"{per_candidate_actual / 1e9:.2f} GB/cand → {chunk_size}/chunk × {num_chunks_final} passes "
                    f"({free_bytes / 1e9:.0f}/{total_bytes / 1e9:.0f} GB free)"
                )
            chunk_start = chunk_end
            chunk_idx += 1
        all_costs = torch.cat(all_costs)
        return torch.argmin(all_costs).item(), all_costs

    @staticmethod
    def _auto_chunk_size(num_candidates, canvas_shape, dtype, device):
        """Pick a candidate-batch size that fits in current free GPU memory.

        Empirical per-candidate peak (measured at 4096×4096): bilinear KDE
        scatter buffers, gaussian smoothing temporaries, then cross-correlation
        FFT pairs (complex64) - together about ``32 × canvas_pixels``
        ``× dtype_bytes`` at peak. We sample free memory at call time, divide
        by that estimate with a 0.4 safety factor, and cap the result at
        ``num_candidates`` (no point splitting if it all fits).
        On CPU we just process all candidates at once - no VRAM constraint.
        """
        device = torch.device(device)
        if device.type != "cuda":
            return num_candidates
        bytes_per_element = torch.finfo(dtype).bits // 8
        per_candidate_bytes = canvas_shape[0] * canvas_shape[1] * bytes_per_element * 32
        free_bytes, _ = torch.cuda.mem_get_info(device)
        chunk_size = max(1, int(free_bytes * 0.4 / per_candidate_bytes))
        return min(chunk_size, num_candidates)

    @torch.inference_mode()
    def _warp_and_translate_torch(
        self,
        max_image_shift: float | None,
        upsample_factor: int = 8,
        knots_batch: torch.Tensor | None = None,
        solve_translation: bool = True,
        fixed_indices: frozenset[int] | None = None,
    ) -> torch.Tensor:
        """Regenerate warped images and solve translation on GPU.

        Three phases: warp → solve translation → re-warp. When ``knots_batch``
        is provided, reads/writes a single batched torch tensor (zero numpy
        crossings). Without it, reads/writes ``self.knots`` (torch tensors)
        for compatibility with ``align_affine``.

        Set ``solve_translation=False`` to only warp and sync without
        re-solving translation - used after the nonrigid loop to populate
        ``self.imgs_warped`` from final knots.

        When ``fixed_indices`` is provided, translation shifts are anchored
        to those images (their shifts become zero) so that fixed images
        never move on the canvas.

        Parameters
        ----------
        max_image_shift : float or None
            Maximum allowed translational shift in pixels.
        upsample_factor : int
            Sub-pixel precision for cross-correlation (1/N pixel).
        knots_batch : torch.Tensor or None
            If provided, batched ``(N, 2, num_rows)`` torch tensor on GPU.
            Translation shifts are applied in-place. Skips numpy sync.
        solve_translation : bool
            If False, skip translation alignment (Phase 2+3). Only warp
            once using current knots and sync to CPU.
        fixed_indices : frozenset[int] or None
            Indices of images that should not receive translation shifts.
            When set, shifts are re-anchored so fixed images stay in place.

        Returns
        -------
        torch.Tensor
            Warped images on GPU, shape ``(num_images, H, W)``.
        """
        device = self._device
        dtype = self._dtype
        num_images = self.shape[0]
        canvas_shape = (self.shape[1], self.shape[2])
        fixed_set = fixed_indices if fixed_indices else frozenset()

        def _warp_all(warped_t, weights_t):
            """Warp all images onto the canvas using current knots."""
            for img_idx in range(num_images):
                if knots_batch is not None:
                    # transform_coordinates_single_knot expects (2, N, 1)
                    knots_img = knots_batch[img_idx].detach()[:, :, None]
                else:
                    knots_img = self.knots[img_idx]
                row_t, col_t = transform_coordinates_single_knot(
                    knots_img, self.scan_fast_t[img_idx], self.imgs[img_idx].shape)
                warped, weights = bilinear_kde_batch(
                    row_t[None], col_t[None], self.imgs_t[img_idx], canvas_shape,
                    self.kde_sigma, self.pad_value[img_idx])
                warped_t[img_idx] = warped[0]
                weights_t[img_idx] = weights[0]

        warped_t = torch.zeros(num_images, *canvas_shape, dtype=dtype, device=device)
        weights_t = torch.zeros_like(warped_t)
        _warp_all(warped_t, weights_t)
        if not solve_translation:
            self.imgs_warped.array[:] = warped_t.cpu().numpy()
            self.weights_warped.array[:] = weights_t.cpu().numpy()
            return warped_t
        # Solve translation shifts and apply to knots
        shifts_t = translate_align(warped_t, upsample_factor, max_image_shift)
        # When fixed images are present, anchor shifts to them instead of the mean
        if fixed_set:
            fixed_idx_list = sorted(fixed_set)
            anchor = shifts_t[fixed_idx_list].mean(0)
            shifts_t -= anchor
            for idx in fixed_set:
                shifts_t[idx] = 0.0
        if knots_batch is not None:
            knots_batch[:, 0] += shifts_t[:, 0:1]
            knots_batch[:, 1] += shifts_t[:, 1:2]
        else:
            for img_idx in range(num_images):
                self.knots[img_idx][0] += shifts_t[img_idx, 0]
                self.knots[img_idx][1] += shifts_t[img_idx, 1]
        # Re-warp with corrected knots
        _warp_all(warped_t, weights_t)
        if knots_batch is None:
            self.imgs_warped.array[:] = warped_t.cpu().numpy()
            self.weights_warped.array[:] = weights_t.cpu().numpy()
        return warped_t

    def align_nonrigid(
        self,
        backend: str = "pytorch",
        optimizer_name: str = "adam",
        num_iterations: int = 16,
        regularization_sigma_px: float = 8.0,
        regularization_update_step_size: float | None = 0.8,
        regularization_poly_order: int = 1,
        max_image_shift: float | None = 32.0,
        adam_steps: int = 30,
        lr: float | None = None,
        lbfgs_max_iter: int = 20,
        max_optimize_iterations: int = 10,
        regularization_max_image_shift_px: float | None = None,
        solve_individual_rows: bool = True,
        fixed_indices: list[int] | None = None,
        loss: str = "auto",
        loss_pre_smooth: float = 1.0,
        early_stop_patience: int = 3,
        early_stop_rtol: float = 1e-4,
        min_iterations: int = 4,
        show_merged: bool = True,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        """
        Non-rigid drift correction using PyTorch (default) or SciPy backend.

        Parameters
        ----------
        backend : str, default "pytorch"
            Optimization backend.
              - "pytorch": GPU-accelerated batched optimization. Single-knot only.
              - "scipy": CPU L-BFGS row-by-row. Use when you need multi-knot
                mode (``number_knots > 1``), which the pytorch path does not
                yet support.
        optimizer_name : str, default "adam"
            PyTorch optimizer (ignored if backend="scipy").

            **"adam"** - first-order momentum optimizer. Default. Best when:
              - You want the fastest possible runtime, especially at image
                sizes ≤512 px where the per-step grid_sample is small and
                Adam's tight inner loop wins on launch overhead.
              - You're confident ``max_image_shift`` reflects the true drift
                bound (Adam's auto-lr derives from it; if it's too small,
                Adam silently under-converges).
              - You want bit-reproducible results across runs (LBFGS line
                search has subtle non-determinism from Wolfe condition checks).

              **Provisional override guidance** (validated on one real-data
              pair - Bob's gold-nanoparticle HAADF on a spectra background -
              and the synthetic chevron test; needs broader testing on
              diverse datasets before being treated as authoritative). If
              you override ``lr`` manually, the rough formula is
              ``expected_drift_px / (num_iterations * adam_steps)``.
              Indicative starting values for the default ``num_iterations=16,
              adam_steps=30`` (480 total steps):
                * ~5 px drift (synthetic chevron, small drift): ``lr≈0.01``
                * ~50-100 px drift (gold-nanoparticle HAADF, real STEM): ``lr≈0.25``
                * larger / unknown drift: prefer ``optimizer_name="lbfgs"``
                  which auto-scales via line search and doesn't need this
                  per-dataset tuning.

            **"lbfgs"** - quasi-Newton optimizer with strong-Wolfe line search.
            Best when:
              - The image is ≥512 px and you don't mind paying Python closure
                overhead for fewer total steps (typically 2-3× faster than
                Adam at 2048+ px because it converges in ~30 steps not 240).
              - You're unsure about the drift magnitude or don't want to think
                about ``lr`` tuning - LBFGS line search auto-scales the step
                without any hand-tuning.
              - You want quality over speed.

            **Failure modes to avoid:**
              - **Don't normalize inputs to [0, 1] when using LBFGS** -
                strong-Wolfe's curvature condition needs absolute gradient
                magnitude above a threshold; with normalized intensities the
                gradient is ~1e-4 and the line search returns step=0,
                producing zero correction silently. Adam is unaffected.
              - **Don't set ``max_image_shift`` smaller than your actual drift
                if using Adam with default ``lr=None``** - Adam's auto-derived
                lr scales with max_image_shift, so a too-small bound silently
                clamps how much drift Adam can recover. LBFGS is unaffected.

            If unsure, start with Adam (the default) for ≤1024 px images and
            switch to LBFGS for ≥2048 px or for unknown-drift exploratory work.

        Shared Parameters
        -----------------
        num_iterations : int, default 16
            Number of outer iterations for alternating optimization.
        regularization_sigma_px : float, default 8.0
            Gaussian smoothing sigma for knot regularization. Smaller values
            allow finer per-row correction; larger values enforce smoother
            drift profiles. Values of 4-12 are typical for STEM data.
        regularization_update_step_size : float, default 0.8
            Step size for knot updates (0-1, lower = more conservative).
        regularization_poly_order : int, default 1
            Polynomial order for trend removal in knot regularization
            (used by both pytorch and scipy backends).
        max_image_shift : float, default 32.0
            Maximum shift for translation alignment between iterations.

        PyTorch Parameters (ignored if backend="scipy")
        -----------------------------------------------
        adam_steps : int, default 30
            Number of Adam optimization steps per outer iteration.
        lr : float or None, default None
            Learning rate for Adam. When None (default), auto-derived as
            ``max_image_shift / (num_iterations * adam_steps * 4)``.

            **Why auto-derive?** Adam's ``m/sqrt(v)`` update self-normalizes
            the gradient, so each step moves a knot by ~``lr`` pixels
            regardless of image intensity scale. The total movement budget
            is ``lr × num_iterations × adam_steps`` and is hard-bounded:
            Adam cannot find drift larger than that budget no matter how
            many iterations you give it. This means ``lr`` must be matched
            to the EXPECTED DRIFT MAGNITUDE IN PIXELS, not to gradient
            magnitude - the same default value that works on small synthetic
            drift will silently under-converge on real data with larger drift.

            The auto-derived formula uses a safety factor of 4 so the total
            movement budget covers ``max_image_shift / 4`` of drift - enough
            for refinement without overshooting at small image sizes where
            actual drift is well below ``max_image_shift``.

            Override with an explicit float when you know the actual drift
            magnitude - e.g. ``lr=2.0`` for very-large-drift in-situ data,
            or ``lr=0.005`` for atomic-resolution stable samples.
        lbfgs_max_iter : int, default 20
            Maximum LBFGS iterations per outer iteration (line search probes
            within each iter happen automatically). Only used when
            optimizer="lbfgs".

        SciPy Parameters (ignored if backend="pytorch")
        -----------------------------------------------
        max_optimize_iterations : int, default 10
            Maximum L-BFGS iterations per row.
        regularization_max_image_shift_px : float, optional
            Maximum allowed shift per iteration.
        solve_individual_rows : bool, default True
            If True, optimize each row independently.

        Fixed-Reference Parameters
        --------------------------
        fixed_indices : list[int] or None, default None
            Indices of images whose knots should NOT be optimized.
            Use ``fixed_indices=[0]`` for single-sided alignment where
            image 0 is the reference (e.g. merged HAADF) and only the
            remaining image(s) are corrected. When set:

            - **Reference**: The mean of the fixed images is used as
              the optimization target for every moving image, instead of
              the leave-one-out mean.
            - **Optimizer**: Only moving images' knots receive gradient
              updates; fixed knots are frozen.
            - **Translation**: Shifts are anchored to fixed images
              (passed through to ``_warp_and_translate_torch``).

        Loss Parameters
        ---------------
        loss : str, default "auto"
            Loss function for the nonrigid optimizer.

            **"auto"** (default) - resolves to ``"gradient_mse"`` when
            ``backend="pytorch"`` and ``"mse"`` when ``backend="scipy"``.
            This ensures cross-modality robustness by default on GPU
            while maintaining backward compatibility for CPU callers.

            **"mse"** - pixel-wise mean squared error on raw images.
            Works well when the reference and target have similar
            intensity and contrast (e.g. two HAADF scans of the same
            sample under the same conditions).

            **"gradient_mse"** - MSE on Sobel gradient magnitudes. Before
            computing the loss, both the warped reference and the target
            are edge-filtered (optional Gaussian pre-smooth → Sobel → L2
            gradient magnitude → per-image z-score normalization). This
            eliminates sensitivity to additive intensity offsets **and**
            multiplicative gain differences between images, focusing the
            optimizer purely on structural alignment.

            Use ``"gradient_mse"`` when the reference and target have
            different intensity profiles - e.g. a merged 4096² HAADF
            reference vs. a 512² VDF from a 4D-STEM acquisition. On real
            gold nanoparticle data, ``gradient_mse`` improved NCC from
            0.932 to 0.949 (+1.8%) and reduced residual shift from
            3.0 to 1.2 px compared to ``"mse"``.

            Only supported with ``backend="pytorch"``.

        loss_pre_smooth : float, default 1.0
            Gaussian sigma (in pixels) applied before Sobel edge
            detection when ``loss="gradient_mse"``. Suppresses
            high-frequency noise that would otherwise dominate the
            gradient magnitude. Set to 0 to disable. Ignored when
            ``loss="mse"``.

        Early Stopping Parameters
        -------------------------
        early_stop_patience : int, default 3
            Number of consecutive iterations without improvement before
            stopping early. Set to ``num_iterations`` to disable.
        early_stop_rtol : float, default 1e-4
            Minimum relative improvement in alignment error to count as
            progress. Iteration *i* is an improvement if
            ``error[i] < best_error * (1 - rtol)``.
        min_iterations : int, default 4
            Minimum number of iterations to run before early stopping
            can trigger. Ensures the optimizer explores enough before
            converging.

        Display Parameters
        ------------------
        show_merged : bool, default True
            Show merged image after alignment.
        show_images : bool, default False
            Show individual aligned images.
        show_knots : bool, default True
            Overlay knot positions on visualizations.

        Notes
        -----
        With backend="pytorch", ``self.imgs_warped`` is left STALE after
        the loop and refreshed lazily on first access via plot methods or
        ``calculate_error()``. Code that reads ``self.imgs_warped.array``
        directly should call ``self._ensure_warped_images()`` first, or
        use ``generate_corrected_image()`` which builds its own warps from
        ``self.knots``.
        """
        if self._frames is not None:
            kw = {k: v for k, v in locals().items() if k != "self"}
            kw.update(kw.pop("kwargs"))
            kw["show_merged"] = False
            kw["show_images"] = False
            for f in tqdm(self._frames, desc="Aligning nonrigid"):
                f.align_nonrigid(**kw)
            return self

        if not hasattr(self, "knots"):
            raise RuntimeError(
                "No knots found. Call .preprocess() before running alignment."
            )
        # Resolve "auto" loss: gradient_mse for pytorch, mse for scipy
        if loss == "auto":
            loss = "gradient_mse" if backend == "pytorch" else "mse"
        _valid_losses = ("mse", "gradient_mse")
        if loss not in _valid_losses:
            raise ValueError(
                f"loss must be one of {_valid_losses!r} or 'auto', got {loss!r}")
        if loss != "mse" and backend != "pytorch":
            raise ValueError(
                f"loss={loss!r} is only supported with backend='pytorch'. "
                f"Use backend='pytorch' or loss='mse'.")
        # Warn about normalize + LBFGS silent failure
        if (optimizer_name == "lbfgs" and backend == "pytorch"
                and getattr(self, "_normalized", False)):
            import warnings
            warnings.warn(
                "normalize=True + LBFGS can cause silent convergence failure. "
                "Wolfe line search may return step=0 on unit-variance images. "
                "Consider using optimizer_name='adam' or normalize=False.",
                UserWarning, stacklevel=2)
        fixed_set = frozenset(fixed_indices) if fixed_indices is not None else frozenset()
        moving_indices = [i for i in range(self.shape[0]) if i not in fixed_set]
        if fixed_set and not moving_indices:
            raise ValueError(
                "All images are fixed — nothing to optimize. "
                "fixed_indices must leave at least one moving image."
            )
        if backend == "pytorch":
            device = self._device
            dtype = self._dtype
            num_images = self.shape[0]
            canvas_shape = (self.shape[1], self.shape[2])
            if any(self.knots[i].shape[2] != 1 for i in range(num_images)):
                raise NotImplementedError(
                    "PyTorch backend only supports single knot. "
                    "Use backend='scipy' for multiple knots.")
            knots_batch = torch.stack(
                [self.knots[i][:, :, 0] for i in range(num_images)]
            ).detach().requires_grad_(True)
            num_rows_knot = knots_batch.shape[2]
            target_batch = torch.stack(self.imgs_t)
            # Build u tensors once and reuse - same scan-position vector projects
            # onto row and col offsets via the per-image scan_fast components.
            u_t = [
                torch.as_tensor(self.interpolator[i].u, dtype=dtype, device=device)
                for i in range(num_images)
            ]
            row_scan_offsets = torch.stack([
                u_t[i] * (self.interpolator[i].scan_fast[0] * (self.imgs[i].shape[0] - 1))
                for i in range(num_images)
            ])
            col_scan_offsets = torch.stack([
                u_t[i] * (self.interpolator[i].scan_fast[1] * (self.imgs[i].shape[1] - 1))
                for i in range(num_images)
            ])
            row_scale = 2.0 / (canvas_shape[0] - 1)
            col_scale = 2.0 / (canvas_shape[1] - 1)
            if optimizer_name == "adam":
                # Auto-derive lr so the total movement budget covers a quarter
                # of max_image_shift. The safety factor of 4 (not 2) prevents
                # over-shooting at small image sizes where actual drift is well
                # below max_image_shift; at large sizes the same factor still
                # converges because the loss surface is smoother. See the `lr`
                # parameter docstring for the full rationale.
                adam_lr = lr if lr is not None else max_image_shift / (num_iterations * adam_steps * 4)
                optimizer = torch.optim.Adam([knots_batch], lr=adam_lr, fused=True)
            elif optimizer_name == "lbfgs":
                optimizer = torch.optim.LBFGS(
                    [knots_batch], lr=1.0, max_iter=lbfgs_max_iter,
                    line_search_fn="strong_wolfe")
            else:
                raise ValueError(f"optimizer_name must be 'adam' or 'lbfgs', got {optimizer_name!r}")
            if regularization_sigma_px is not None and regularization_sigma_px > 0:
                x_knot = torch.arange(num_rows_knot, dtype=dtype, device=device)
                x_norm = (x_knot - x_knot.mean()) / x_knot.std()
                vander = torch.stack([x_norm ** p for p in range(regularization_poly_order + 1)], dim=1)
            else:
                vander = None
            # For gradient_mse, temporarily replace alignment images with
            # edge-filtered versions. The learned knots are spatial transforms
            # independent of image content, so optimizing in gradient space
            # yields the same drift field while being robust to intensity and
            # contrast differences between reference and target.
            _original_images_t = None
            if loss == "gradient_mse":
                _original_images_t = list(self.imgs_t)
                sobel_batch = self._sobel_gradient_magnitude(
                    target_batch, loss_pre_smooth, device, dtype)
                self.imgs_t = [sobel_batch[i] for i in range(num_images)]
                target_batch = sobel_batch
            warped_t = self._warp_and_translate_torch(
                max_image_shift, upsample_factor=8, knots_batch=knots_batch,
                fixed_indices=fixed_set)
            # Build a boolean mask on device to zero fixed gradients efficiently.
            # Shape: (num_images, 1, 1) — broadcasts over (2, num_rows_knot).
            if fixed_set:
                grad_mask = torch.ones(num_images, 1, 1, dtype=dtype, device=device)
                for idx in fixed_set:
                    grad_mask[idx] = 0.0
            error_buffer = []
            best_error = float('inf')
            patience_counter = 0
            pbar = tqdm(range(num_iterations), desc=f"Solving nonrigid drift ({optimizer_name})")
            for iter_idx in pbar:
                # Build the reference under no_grad: arithmetic on warped_t (an
                # inference tensor) would otherwise return an autograd-tracked
                # leaf, and the optimizer would build a graph through it.
                with torch.no_grad():
                    if fixed_set:
                        # Fixed images define the reference for all moving images.
                        fixed_mean = warped_t[sorted(fixed_set)].mean(0)
                        ref_batch = fixed_mean[None].expand(num_images, -1, -1)
                    else:
                        warped_sum = warped_t.sum(0)
                        ref_batch = (warped_sum[None] - warped_t) / (num_images - 1)
                    knots_prev = knots_batch.detach().clone()
                # Regularization alters the loss surface between outer iters, so
                # stale momentum / curvature history would push knots the wrong way.
                optimizer.state.clear()
                if optimizer_name == "adam":
                    self._optimize_knots_adam(
                        ref_batch, target_batch, knots_batch,
                        row_scan_offsets, col_scan_offsets, row_scale, col_scale,
                        optimizer, adam_steps,
                        grad_mask=grad_mask if fixed_set else None)
                else:
                    self._optimize_knots_lbfgs(
                        ref_batch, target_batch, knots_batch,
                        row_scan_offsets, col_scan_offsets, row_scale, col_scale,
                        optimizer,
                        grad_mask=grad_mask if fixed_set else None)
                self._regularize_knots(
                    knots_batch, knots_prev, vander,
                    regularization_max_image_shift_px,
                    regularization_sigma_px,
                    regularization_update_step_size)
                # Restore fixed knots — regularization is a global smooth that
                # would subtly shift them via polynomial detrend + Gaussian blur.
                if fixed_set:
                    with torch.no_grad():
                        for idx in fixed_set:
                            knots_batch[idx] = knots_prev[idx]
                warped_t = self._warp_and_translate_torch(
                    max_image_shift, upsample_factor=8, knots_batch=knots_batch,
                    fixed_indices=fixed_set)
                # Per-iter error stays on GPU; sync once after the loop
                images_mean = warped_t.mean(dim=0)
                iter_error = torch.mean(torch.abs(warped_t - images_mean[None]), dim=(1, 2))
                error_buffer.append(iter_error)
                # Early stopping: monitor post-iteration alignment quality
                current_error = float(iter_error.mean())
                if current_error < best_error * (1 - early_stop_rtol):
                    best_error = current_error
                    patience_counter = 0
                else:
                    patience_counter += 1
                if (iter_idx >= min_iterations - 1
                        and patience_counter >= early_stop_patience):
                    pbar.set_postfix_str(f"converged at iter {iter_idx + 1}")
                    break
            # Sync knots back; leave imgs_warped lazy so callers
            # that never plot avoid the GPU→CPU transfer of the warped stack.
            knots_final = knots_batch.detach()
            for img_idx in range(num_images):
                self.knots[img_idx][:, :, 0] = knots_final[img_idx]
            # Restore original images after gradient_mse alignment so that
            # apply_correction, visualization, and error metrics use the
            # original pixel intensities, not edge-filtered versions.
            if _original_images_t is not None:
                self.imgs_t = _original_images_t
            self._images_warped_stale = True
            self._max_image_shift_cached = max_image_shift
            if error_buffer:
                # Build all error rows in one DtoH transfer + one vstack, instead of
                # the quadratic vstack-per-iteration pattern used by calculate_error.
                errors_np = torch.stack(error_buffer).cpu().numpy()  # (num_iterations, num_images)
                mode_col = np.full((len(errors_np), 1), 2.0)
                mean_col = errors_np.mean(axis=1, keepdims=True)
                new_rows = np.hstack((mode_col, mean_col, errors_np))
                if not hasattr(self, "error_track"):
                    self.error_track = new_rows
                else:
                    self.error_track = np.vstack((self.error_track, new_rows))
        else:
            # Precompute fixed reference for scipy path when fixed_indices is set
            if fixed_set:
                fixed_idx_list = sorted(fixed_set)
            for _ in tqdm(range(num_iterations), desc="Solving nonrigid drift (scipy)"):
                for img_idx in range(self.shape[0]):
                    if img_idx in fixed_set:
                        continue
                    if fixed_set:
                        image_ref = self.imgs_warped.array[fixed_idx_list].mean(axis=0)
                    else:
                        image_ref = np.delete(self.imgs_warped.array, img_idx, axis=0).mean(axis=0)
                    knots_np = self.knots[img_idx].cpu().numpy()
                    knots_updated = self._optimize_knots_scipy(
                        img_idx, image_ref, knots_np,
                        max_optimize_iterations=max_optimize_iterations,
                        solve_individual_rows=solve_individual_rows)
                    if regularization_max_image_shift_px is not None:
                        knots_shift = knots_updated - knots_np
                        knots_dist = np.sqrt(np.sum(knots_shift**2, axis=0))
                        exceeds_max = knots_dist > regularization_max_image_shift_px
                        knots_updated[0][exceeds_max] = (knots_np[0][exceeds_max]
                            + knots_shift[0][exceeds_max] * regularization_max_image_shift_px / knots_dist[exceeds_max])
                        knots_updated[1][exceeds_max] = (knots_np[1][exceeds_max]
                            + knots_shift[1][exceeds_max] * regularization_max_image_shift_px / knots_dist[exceeds_max])
                    if regularization_sigma_px is not None and regularization_sigma_px > 0:
                        knots_smoothed = knots_updated.copy()
                        scanline_idx = np.arange(knots_updated.shape[1])
                        for axis in range(2):
                            for knot_ind in range(knots_updated.shape[2]):
                                knot_vals = knots_updated[axis, :, knot_ind]
                                coefs = np.polyfit(scanline_idx, knot_vals, deg=regularization_poly_order)
                                trend = np.polyval(coefs, scanline_idx)
                                residual = knot_vals - trend
                                knots_smoothed[axis, :, knot_ind] = gaussian_filter(residual, sigma=regularization_sigma_px) + trend
                        knots_updated = knots_smoothed
                    if regularization_update_step_size is not None:
                        knots_updated = (knots_np
                            + (knots_updated - knots_np) * regularization_update_step_size)
                    self.knots[img_idx] = torch.tensor(knots_updated, dtype=self._dtype, device=self._device)
                warped_t = self._warp_and_translate_torch(
                    max_image_shift, upsample_factor=8, fixed_indices=fixed_set)
                self.calculate_error(2, _warped_t=warped_t)

        if show_merged:
            self.plot_merged_images(
                show_knots=show_knots,
                title="Merged: non-rigid",
                **kwargs,
            )

        if show_images:
            self.plot_warped_images(
                show_knots=show_knots,
                title=[f"Image {i}: non-rigid" for i in range(self.shape[0])],
                **kwargs,
            )

        return self

    def _optimize_knots_adam(
        self, ref_batch, target_batch, knots_batch,
        row_scan_offsets, col_scan_offsets, row_scale, col_scale,
        optimizer, adam_steps, grad_mask=None,
    ):
        """Run ``adam_steps`` of Adam on a batched knot tensor against ``_compiled_loss_fn``."""
        ref_t = ref_batch[:, None]
        for _ in range(adam_steps):
            optimizer.zero_grad()
            loss = self._compiled_loss_fn(
                knots_batch, ref_t, target_batch,
                row_scan_offsets, col_scan_offsets, row_scale, col_scale)
            loss.backward()
            if grad_mask is not None:
                knots_batch.grad.mul_(grad_mask)
            optimizer.step()

    @staticmethod
    @torch.compile(mode="reduce-overhead", dynamic=False)
    def _compiled_loss_fn(
        knots_batch, ref_t, target_batch,
        row_scan_offsets, col_scan_offsets, row_scale, col_scale,
    ):
        """Fused forward pass: knot offsets → grid → grid_sample → MSE loss.

        The MSE is averaged over both the batch (N images) and the spatial
        dims, so each image's gradient is scaled by 1/N relative to a
        per-image solve. Adam's adaptive step size absorbs the constant
        rescale; LBFGS line search rescales itself.
        """
        grid_row = (knots_batch[:, 0, :, None] + row_scan_offsets[:, None, :]) * row_scale - 1.0
        grid_col = (knots_batch[:, 1, :, None] + col_scan_offsets[:, None, :]) * col_scale - 1.0
        grid = torch.stack([grid_col, grid_row], dim=-1)
        warped = torch.nn.functional.grid_sample(
            ref_t, grid, mode='bilinear', align_corners=True, padding_mode='border')[:, 0]
        return ((warped - target_batch) ** 2).mean()

    def _optimize_knots_lbfgs(
        self, ref_batch, target_batch, knots_batch,
        row_scan_offsets, col_scan_offsets, row_scale, col_scale,
        optimizer, grad_mask=None,
    ):
        """Run one LBFGS outer step (line search re-evaluates the closure several times)."""
        ref_t = ref_batch[:, None]
        def closure():
            optimizer.zero_grad()
            loss = self._compiled_loss_fn(
                knots_batch, ref_t, target_batch,
                row_scan_offsets, col_scan_offsets, row_scale, col_scale)
            loss.backward()
            if grad_mask is not None:
                knots_batch.grad.mul_(grad_mask)
            return loss
        optimizer.step(closure)

    @staticmethod
    def _sobel_gradient_magnitude(
        images: torch.Tensor,
        pre_smooth: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute per-image Sobel gradient magnitude.

        Parameters
        ----------
        images : Tensor, shape (N, H, W)
            Batch of images.
        pre_smooth : float
            Gaussian sigma applied before Sobel. 0 to disable.
        device, dtype : torch device and dtype for kernel creation.

        Returns
        -------
        Tensor, shape (N, H, W)
            Gradient magnitude images, per-image z-score normalized so
            each image has zero mean and unit variance.
        """
        img = images[:, None]  # (N, 1, H, W) for conv2d
        if pre_smooth > 0:
            ks = max(3, int(6 * pre_smooth) | 1)  # odd kernel size
            x = torch.arange(ks, dtype=dtype, device=device) - ks // 2
            g = torch.exp(-0.5 * (x / max(pre_smooth, 1e-6)) ** 2)
            g = g / g.sum()
            # Separable Gaussian: row then column (reflect padding)
            pad_h = ks // 2
            img = torch.nn.functional.pad(img, (pad_h, pad_h, 0, 0), mode='reflect')
            img = torch.nn.functional.conv2d(img, g.reshape(1, 1, 1, -1))
            img = torch.nn.functional.pad(img, (0, 0, pad_h, pad_h), mode='reflect')
            img = torch.nn.functional.conv2d(img, g.reshape(1, 1, -1, 1))
        # Sobel kernels
        sx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=dtype, device=device).reshape(1, 1, 3, 3)
        sy = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=dtype, device=device).reshape(1, 1, 3, 3)
        img_pad = torch.nn.functional.pad(img, (1, 1, 1, 1), mode='reflect')
        gx = torch.nn.functional.conv2d(img_pad, sx)
        gy = torch.nn.functional.conv2d(img_pad, sy)
        grad_mag = (gx ** 2 + gy ** 2).sqrt()[:, 0]  # (N, H, W)
        # Per-image z-score normalization: removes gain/offset sensitivity
        mean = grad_mag.mean(dim=(-2, -1), keepdim=True)
        std = grad_mag.std(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        return (grad_mag - mean) / std

    def _regularize_knots(
        self, knots_batch, knots_prev, vander,
        max_shift_px, sigma_px, step_size,
    ):
        """Apply per-iteration knot regularization (in-place on ``knots_batch``).

        Three independent stages, each gated by its parameter being non-None:
            1. Per-knot shift cap: clamp ``|new - prev|`` to ``max_shift_px``
               so the optimizer can't move any knot too far in one outer iter.
            2. Polynomial detrend + Gaussian smooth: keep low-order trends,
               smooth the residual along the scan-line dimension. Removes
               high-frequency optimizer wobble while preserving the drift signal.
            3. Step-size blend: ``new = prev + step_size · (new - prev)``,
               under-relaxes the update for stability across outer iterations.
        """
        num_images, _, num_rows_knot = knots_batch.shape
        with torch.no_grad():
            if max_shift_px is not None:
                shift = knots_batch - knots_prev
                dist = torch.norm(shift, dim=1, keepdim=True)
                scale_factor = torch.clamp(max_shift_px / dist.clamp(min=1e-8), max=1.0)
                knots_batch.copy_(knots_prev + shift * scale_factor)
            if sigma_px is not None and sigma_px > 0 and vander is not None:
                # Detrend + smooth all (N*2, num_rows) knots in one batched lstsq + smooth
                knots_flat = knots_batch.reshape(-1, num_rows_knot).T  # (num_rows, N*2)
                coefs, _, _, _ = torch.linalg.lstsq(vander, knots_flat)
                trend = (vander @ coefs).T  # (N*2, num_rows)
                residual = knots_batch.reshape(-1, num_rows_knot) - trend
                smoothed = gaussian_smooth_1d(residual, sigma_px)
                knots_batch.copy_((smoothed + trend).reshape(num_images, 2, num_rows_knot))
            if step_size is not None:
                knots_batch.copy_(knots_prev + (knots_batch - knots_prev) * step_size)

    def _optimize_knots_scipy(
        self, idx: int, image_ref: np.ndarray, knots_init: np.ndarray,
        max_optimize_iterations: int = 10, solve_individual_rows: bool = True,
    ) -> np.ndarray:
        """SciPy L-BFGS optimization for one image."""
        shape_knots = knots_init.shape
        options = {"maxiter": max_optimize_iterations} if max_optimize_iterations else {}

        def _bilinear_warp(row_coords, col_coords):
            rf = np.clip(np.floor(row_coords).astype(int), 0, self.shape[1] - 2)
            cf = np.clip(np.floor(col_coords).astype(int), 0, self.shape[2] - 2)
            dr, dc = row_coords - rf, col_coords - cf
            return (image_ref[rf, cf] * (1 - dr) * (1 - dc)
                    + image_ref[rf + 1, cf] * dr * (1 - dc)
                    + image_ref[rf, cf + 1] * (1 - dr) * dc
                    + image_ref[rf + 1, cf + 1] * dr * dc)

        if solve_individual_rows:
            knots_updated = np.zeros_like(knots_init)
            for row_ind in range(knots_init.shape[1]):
                x0 = knots_init[:, row_ind, :].ravel()
                def cost_function(x):
                    row_coords, col_coords = self.interpolator[idx].transform_rows(
                        x.reshape(shape_knots[0], shape_knots[2]))
                    return np.sum((_bilinear_warp(row_coords, col_coords)
                                   - self.imgs[idx].array[row_ind, :]) ** 2)
                result = minimize(cost_function, x0, method="L-BFGS-B", options=options)
                knots_updated[:, row_ind, :] = result.x.reshape((2, -1))
        else:
            x0 = knots_init.ravel()
            def cost_function(x):
                row_coords, col_coords = self.interpolator[idx].transform_coordinates(
                    x.reshape(shape_knots))
                return np.sum((_bilinear_warp(row_coords, col_coords)
                               - self.imgs[idx].array) ** 2)
            result = minimize(cost_function, x0, method="L-BFGS-B", options=options)
            knots_updated = result.x.reshape(shape_knots)
        return knots_updated

    def generate_corrected_image(
        self,
        upsample_factor: int = 2,
        output_original_shape: bool = True,
        strip_padding: bool = False,
        mask_output: bool = True,
        mask_edge_blend: float = 8.0,
        fourier_filter: bool = True,
        filter_midpoint: float = 0.5,
        kde_sigma: float = 0.5,
        weight_thresh: float = 0.1,
        show_image: bool = True,
        **kwargs,
    ):
        """
        Generate the final drift-corrected image after aligning a stack of input images.

        The entire pipeline (warping, Fourier filtering, masking, cropping) runs
        on GPU via PyTorch, transferring to CPU only for the final
        ``Dataset2d`` output and the ``distance_transform_edt`` mask step.

        Parameters
        ----------
        upsample_factor : int, default 2
            Factor to upsample the output image for enhanced interpolation accuracy.
        output_original_shape : bool, default True
            If True, crop the output image back to the original input dimensions after processing.
        strip_padding : bool, default False
            If True and ``output_original_shape`` is True, further crop the result
            to the original *scan* dimensions (removing the padding added by
            ``preprocess(pad_fraction=...)``).  This ensures the returned image
            covers exactly the same field-of-view as the raw input scans.
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
        if self._frames is not None:
            kw = {k: v for k, v in locals().items() if k != "self"}
            kw.update(kw.pop("kwargs"))
            kw["show_image"] = False
            results = []
            for f in tqdm(self._frames, desc="Generating corrected images"):
                results.append(f.generate_corrected_image(**kw))
            return np.stack([r.array for r in results], axis=0).astype(np.float32)

        device = self._device
        dtype = self._dtype

        up_h = round(self.shape[1] * upsample_factor)
        up_w = round(self.shape[2] * upsample_factor)
        canvas_up = (up_h, up_w)

        if kde_sigma is None:
            kde_sigma = self.kde_sigma

        # Warp all images onto upsampled canvas on GPU
        stack_corr = torch.zeros(self.shape[0], up_h, up_w, dtype=dtype, device=device)
        weight_corr = torch.zeros_like(stack_corr)

        for img_idx in range(self.shape[0]):
            row_t, col_t = transform_coordinates_single_knot(
                self.knots[img_idx], self.scan_fast_t[img_idx], self.imgs[img_idx].shape)
            warped, weights = bilinear_kde_batch(
                row_t[None] * upsample_factor,
                col_t[None] * upsample_factor,
                self.imgs_t[img_idx],
                canvas_up,
                kde_sigma * upsample_factor,
                self.pad_value[img_idx],
            )
            stack_corr[img_idx] = warped[0]
            weight_corr[img_idx] = weights[0]

        if fourier_filter:
            freq_row = torch.fft.fftfreq(up_h, dtype=dtype, device=device)[:, None]
            freq_col = torch.fft.fftfreq(up_w, dtype=dtype, device=device)[None, :]
            freq_angle = torch.atan2(freq_col, freq_row)

            stack_fft = torch.fft.fft2(stack_corr)
            weights = torch.zeros_like(stack_corr)

            for img_idx in range(self.shape[0]):
                weights[img_idx] = torch.abs(
                    torch.remainder((freq_angle - self.scan_direction[img_idx]) / np.pi + 0.5, 1.0) - 0.5
                ) / 0.5
                weights[img_idx, 0, 0] = 1.0
                weights[img_idx] = _bounded_sine_sigmoid_torch(
                    weights[img_idx], midpoint=filter_midpoint)
                stack_fft[img_idx] *= weights[img_idx]

            weights_sum = weights.sum(0)
            fft_sum = stack_fft.sum(0)
            image_corr_fft = torch.where(
                weights_sum > 0.0,
                fft_sum / weights_sum.clamp(min=1e-8),
                torch.zeros_like(fft_sum),
            )
        else:
            image_corr_fft = torch.fft.fft2(stack_corr.mean(0))

        if mask_output:
            # distance_transform_edt has no torch equivalent — compute on CPU
            weight_np = weight_corr.cpu().numpy()
            mask_edge = np.prod(weight_np >= (weight_thresh / upsample_factor**2), axis=0)
            mask_edge[:, 0] = False
            mask_edge[:, -1] = False
            mask_edge[0, :] = False
            mask_edge[-1, :] = False
            mask_inner = distance_transform_edt(mask_edge) <= mask_edge_blend
            mask_np = (
                np.cos(
                    (np.pi / 2)
                    * np.clip(distance_transform_edt(mask_inner) / mask_edge_blend, 0.0, 1.0)
                )
                ** 2
            )
            mask_t = torch.as_tensor(mask_np, dtype=dtype, device=device)
            pad_value_mean = np.mean([ind.pad_value for ind in self.interpolator])
            image_corr_fft = torch.fft.fft2(
                torch.fft.ifft2(image_corr_fft).real * mask_t + pad_value_mean * (1 - mask_t)
            )

        if output_original_shape:
            image_corr_fft = _fourier_crop_torch(
                image_corr_fft, self.shape[-2:]) / upsample_factor**2

        corr_np = torch.fft.ifft2(image_corr_fft).real.cpu().numpy()

        if strip_padding and output_original_shape:
            scan_h, scan_w = self.imgs[0].shape[:2]
            canvas_h, canvas_w = corr_np.shape[:2]
            pad_h = (canvas_h - scan_h) // 2
            pad_w = (canvas_w - scan_w) // 2
            corr_np = corr_np[pad_h:pad_h + scan_h, pad_w:pad_w + scan_w]

        image_corr = Dataset2d.from_array(
            corr_np,
            name="drift corrected image",
            origin=self.imgs[0].origin,
            sampling=self.imgs[0].sampling,
            units=self.imgs[0].units,
        )

        if show_image:
            show_2d(image_corr.array, **kwargs)
            plt.show()
        return image_corr

    def _canvas_to_raw_drift(
        self,
        drift_canvas: torch.Tensor,
        idx: int,
        scan_h: int,
        scan_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert canvas-space knot delta to raw-frame pixel drift.

        The knots live on a common padded canvas whose axes align with the
        physical specimen.  When the scan direction is non-zero the raw image
        axes are rotated relative to the canvas, so applying canvas deltas
        directly to raw pixels gives wrong results.

        The mapping from raw pixel ``(i, j)`` to canvas is::

            canvas_row = center + i·slow[0]  + (j/(W-1))·fast[0]·(H-1)
            canvas_col = center + i·slow[1]  + (j/(W-1))·fast[1]·(W-1)

        We invert this Jacobian to convert a canvas displacement back to
        raw-pixel displacement.  For square images (H == W) this reduces
        to a simple rotation by the scan angle.
        """
        device = drift_canvas.device
        dtype = drift_canvas.dtype
        slow = torch.tensor(self.scan_slow[idx], device=device, dtype=dtype)
        fast = torch.tensor(self.scan_fast[idx], device=device, dtype=dtype)
        alpha = float(scan_h - 1) / float(scan_w - 1) if scan_w > 1 else 1.0
        det = slow[0] * fast[1] - fast[0] * alpha * slow[1]
        drift_row = (
            fast[1] * drift_canvas[0] - fast[0] * alpha * drift_canvas[1]
        ) / det
        drift_col = (
            -slow[1] * drift_canvas[0] + slow[0] * drift_canvas[1]
        ) / det
        return drift_row, drift_col

    def apply_correction(
        self,
        images: torch.Tensor | np.ndarray | None = None,
        image_index: int = -1,
        mode: str = "bicubic",
    ) -> torch.Tensor:
        """Apply drift correction via backward interpolation.

        Uses the per-row drift estimated by :meth:`align_affine` and/or
        :meth:`align_nonrigid` to correct images with ``grid_sample``.
        This is the recommended way to apply corrections for single-sided
        optimization (e.g. 4D-STEM VDF against fixed HAADF reference).

        Parameters
        ----------
        images : torch.Tensor or np.ndarray, optional
            Images to correct, shape ``(H, W)`` or ``(N, H, W)``.
            If *None*, corrects the stored image at *image_index*.
            Pass external data (e.g. detector-pixel slices from a 4D-STEM
            cube) to apply the same drift correction to arbitrary images.
        image_index : int, default -1
            Which image's knot trajectory to use for the correction.
            Default ``-1`` selects the last image (typical for the
            ``[reference, target]`` two-image case).
        mode : str, default "bicubic"
            Interpolation kernel: ``"bicubic"`` or ``"bilinear"``.

        Returns
        -------
        torch.Tensor
            Corrected images on the same device, same shape as input.

        Examples
        --------
        >>> dc = DriftCorrection.from_data(
        ...     images=[haadf_ref, vdf], scan_direction_degrees=[0, 0])
        >>> dc.preprocess(normalize=True).align_affine(fixed_indices=[0])
        >>> dc.align_nonrigid(fixed_indices=[0])
        >>> corrected_vdf = dc.apply_correction(mode='bicubic')
        """
        self._ensure_single("apply_correction")
        if not hasattr(self, "_initial_knots"):
            msg = "Call preprocess() before apply_correction()"
            raise RuntimeError(msg)

        _valid_modes = {"bilinear", "bicubic"}
        if mode not in _valid_modes:
            raise ValueError(
                f"mode must be one of {_valid_modes}, got {mode!r}"
            )

        idx = image_index % len(self.knots)
        num_knots = self.knots[idx].shape[2]
        if num_knots != 1:
            raise NotImplementedError(
                f"apply_correction only supports number_knots=1 "
                f"(got {num_knots}). Use generate_corrected_image() "
                f"for multi-knot setups."
            )

        delta = self.knots[idx] - self._initial_knots[idx]  # (2, H, 1)
        drift_canvas = delta[:, :, 0]  # (2, H) — canvas-space drift
        knot_h = drift_canvas.shape[1]

        if images is None:
            images_t = self.imgs_t[idx]
        elif isinstance(images, np.ndarray):
            images_t = torch.tensor(
                images, dtype=self._dtype, device=self._device
            )
        else:
            images_t = images.to(device=self._device, dtype=self._dtype)

        img_h = images_t.shape[-2]
        img_w = images_t.shape[-1]
        if img_h != knot_h:
            raise ValueError(
                f"Image height ({img_h}) does not match knot grid "
                f"height ({knot_h}). The image must have the same "
                f"number of scan rows as the data used in preprocess()."
            )

        # Rotate canvas drift → raw-frame drift
        drift_row, drift_col = self._canvas_to_raw_drift(
            drift_canvas, idx, img_h, img_w,
        )
        drift_per_row = torch.stack([drift_row, drift_col])  # (2, H)
        return backward_warp(images_t, drift=drift_per_row, mode=mode)

    @staticmethod
    def compute_vdf(
        ds_4d: np.ndarray,
        chunk_rows: int | None = None,
    ) -> np.ndarray:
        """Compute a virtual dark-field image from a 4D-STEM dataset.

        Averages over the detector dimensions to produce a 2D scan image.
        Supports memory-mapped inputs — when *chunk_rows* is set, only a
        few scan rows are loaded at a time, keeping host RAM usage low.

        Parameters
        ----------
        ds_4d : np.ndarray, shape ``(H, W, det_h, det_w)``
            4D-STEM dataset.  Can be a ``np.memmap``.
        chunk_rows : int or None
            Number of scan rows to process at a time.  ``None`` loads
            everything at once (fastest for in-memory arrays).

        Returns
        -------
        np.ndarray, shape ``(H, W)``, dtype float32
        """
        H, W = ds_4d.shape[:2]
        det_pixels = 1
        for d in range(2, ds_4d.ndim):
            det_pixels *= ds_4d.shape[d]

        if chunk_rows is None:
            return ds_4d.reshape(H, W, det_pixels).mean(axis=2).astype(
                np.float32
            )

        vdf = np.empty((H, W), dtype=np.float32)
        for start in range(0, H, chunk_rows):
            end = min(start + chunk_rows, H)
            chunk = np.asarray(ds_4d[start:end])
            vdf[start:end] = chunk.reshape(end - start, W, det_pixels).mean(
                axis=2
            )
        return vdf

    @torch.inference_mode()
    def apply_correction_4dstem(
        self,
        ds_4d: torch.Tensor | np.ndarray,
        image_index: int = -1,
        mode: str = "bicubic",
        chunk_size: int | None = None,
        output_dtype: torch.dtype | np.dtype | str | None = None,
        output_device: str | torch.device | None = None,
        output: np.ndarray | None = None,
        progress: bool = False,
    ) -> torch.Tensor | np.ndarray:
        """Apply drift correction to a 3D or 4D data cube.

        Automatically selects a **single-shot GPU path** when the full
        cube fits in GPU memory (fastest), or falls back to chunked
        processing for truly enormous datasets.

        Parameters
        ----------
        ds_4d : torch.Tensor or np.ndarray
            3D array ``(H, W, C)`` for EDX/EELS spectral data, or
            4D array ``(H, W, det_h, det_w)`` for 4D-STEM.
            The first two axes must be scan rows and columns matching
            the shape used in :meth:`preprocess`.  Can be a
            ``np.memmap`` — only the channels being processed are
            paged in.
        image_index : int, default -1
            Which image's knot trajectory to use. Default selects the
            last image (target in a ``[reference, target]`` pair).
        mode : str, default "bicubic"
            Interpolation kernel: ``"bicubic"`` or ``"bilinear"``.
        chunk_size : int or None
            Number of channels to warp per GPU call.  When ``None``
            (default), the method auto-selects: single-shot if the cube
            fits in GPU memory with headroom, otherwise chunked with a
            sensible default.  Set explicitly to force chunking.
        output_dtype : dtype, optional
            Cast the output to this dtype. If ``None``, returns float32.
            Use ``"same"`` to match the input dtype.
        output_device : str or torch.device or None
            Device for the returned tensor.  ``None`` (default) returns
            a numpy array if input was numpy, CPU tensor if input was
            torch.  Set to ``"cuda"`` to keep the result on GPU
            (eliminates the GPU→CPU download, which is the main
            bottleneck for large cubes).
        output : np.ndarray or None
            Pre-allocated numpy array (or ``np.memmap``) with the same
            shape as *ds_4d*.  When provided, corrected data is written
            directly into this array chunk-by-chunk and returned —
            no extra host-memory copy is made.  This enables
            process-and-release workflows where you never hold both
            the input and a separate output copy in RAM.  When set,
            *output_device* is ignored.
        progress : bool, default False
            Show a tqdm progress bar (chunked path only).

        Returns
        -------
        torch.Tensor or np.ndarray
            Corrected cube with the same shape and axis layout as input.
            When *output* is provided, returns that same array.

        Examples
        --------
        >>> # EDX spectral cube - auto single-shot
        >>> corrected = dc.apply_correction_4dstem(cube_eds)
        >>> # 4D-STEM with pre-allocated output (memory-mapped)
        >>> out = np.memmap('corrected.dat', dtype='float32',
        ...                 mode='w+', shape=cube.shape)
        >>> dc.apply_correction_4dstem(cube, output=out)
        """
        self._ensure_single("apply_correction_4dstem")
        is_numpy = isinstance(ds_4d, np.ndarray)
        original_shape = ds_4d.shape if is_numpy else tuple(ds_4d.shape)
        input_np_dtype = ds_4d.dtype if is_numpy else None
        use_external_output = output is not None

        if use_external_output:
            if not isinstance(output, np.ndarray):
                raise TypeError(
                    "output must be a numpy ndarray (or np.memmap), "
                    f"got {type(output).__name__}"
                )
            if tuple(output.shape) != tuple(original_shape):
                raise ValueError(
                    f"output shape {output.shape} does not match "
                    f"ds_4d shape {original_shape}"
                )

        return_numpy = (
            use_external_output
            or (is_numpy and output_device is None)
        )

        ndim = len(original_shape)
        if ndim < 3:
            raise ValueError(
                f"ds_4d must be at least 3D, got shape {original_shape}"
            )

        scan_h, scan_w = original_shape[0], original_shape[1]
        n_channels = 1
        for d in range(2, ndim):
            n_channels *= original_shape[d]

        device = torch.device(self._device)

        # ── Per-row drift from knots (canvas → raw frame) ──
        idx = image_index % len(self.knots)
        num_knots = self.knots[idx].shape[2]
        if num_knots != 1:
            raise NotImplementedError(
                f"apply_correction_4dstem only supports number_knots=1 "
                f"(got {num_knots}). Use generate_corrected_image() "
                f"for multi-knot setups."
            )

        delta = self.knots[idx] - self._initial_knots[idx]   # (2, H_knot, 1)
        drift_canvas = delta[:, :, 0].to(device=device, dtype=torch.float32)

        if drift_canvas.shape[1] != scan_h:
            raise ValueError(
                f"Drift grid has {drift_canvas.shape[1]} rows but ds_4d has "
                f"{scan_h} scan rows. Ensure reference image and ds_4d "
                f"have matching scan dimensions (check padding / resize)."
            )

        drift_row, drift_col = self._canvas_to_raw_drift(
            drift_canvas, idx, scan_h, scan_w,
        )

        # ── Pre-compute warp grid ONCE (tiny: 1×H×W×2 f32) ──
        row_coords = torch.arange(scan_h, device=device, dtype=torch.float32)
        col_coords = torch.arange(scan_w, device=device, dtype=torch.float32)
        sample_row = row_coords[:, None].expand(scan_h, scan_w) - drift_row[:, None]
        sample_col = col_coords[None, :].expand(scan_h, scan_w) - drift_col[:, None]
        warp_grid = torch.stack([
            2.0 * sample_col / (scan_w - 1) - 1.0,
            2.0 * sample_row / (scan_h - 1) - 1.0,
        ], dim=-1)[None]                                       # (1, H, W, 2)

        # ── Flatten input to (H, W, C) view ──
        flat = (
            torch.from_numpy(ds_4d.reshape(scan_h, scan_w, n_channels))
            if is_numpy
            else ds_4d.reshape(scan_h, scan_w, n_channels)
        )

        # ── Output dtype (for GPU intermediates) ──
        out_dt = torch.float32
        if output_dtype == "same":
            if is_numpy and input_np_dtype is not None:
                out_dt = torch.from_numpy(
                    np.empty(0, dtype=input_np_dtype)
                ).dtype
            elif not is_numpy:
                out_dt = ds_4d.dtype
        elif isinstance(output_dtype, torch.dtype):
            out_dt = output_dtype

        # ── Target device ──
        if use_external_output:
            target = torch.device("cpu")
        elif output_device is not None:
            target = torch.device(output_device)
            if target.type == "cuda":
                target = device
        else:
            target = torch.device("cpu")

        # ── Chunk size (auto-fit within GPU memory) ──
        if chunk_size is None:
            bytes_per_ch = scan_h * scan_w * 4
            try:
                gpu_free, _ = torch.cuda.mem_get_info(device)
            except RuntimeError:
                gpu_free = 0
            if target.type == "cuda" and not use_external_output:
                out_elem = torch.tensor([], dtype=out_dt).element_size()
                gpu_free = max(
                    0,
                    gpu_free - n_channels * scan_h * scan_w * out_elem,
                )
            chunk_size = min(
                n_channels,
                max(1, int(gpu_free * 0.7 / (bytes_per_ch * 2))),
            )

        # ── Allocate output ──
        if use_external_output:
            out_flat = output.reshape(scan_h, scan_w, n_channels)
        else:
            internal_output = torch.empty(
                scan_h, scan_w, n_channels, dtype=out_dt, device=target,
            )

        # ── Numpy dtype for external output conversion ──
        if use_external_output:
            _out_np_dtype = output.dtype

        # ── Vectorized grid_sample with pre-computed grid ──
        chunks = range(0, n_channels, chunk_size)
        if progress:
            chunks = tqdm(
                chunks,
                total=(n_channels + chunk_size - 1) // chunk_size,
                desc="apply_correction_4dstem",
                unit="chunk",
            )
        for start in chunks:
            end = min(start + chunk_size, n_channels)
            warped = F.grid_sample(
                flat[:, :, start:end].permute(2, 0, 1).contiguous()
                .to(device=device, dtype=torch.float32)[None],
                warp_grid,
                mode=mode, align_corners=True, padding_mode="border",
            )[0].permute(1, 2, 0)

            if use_external_output:
                out_flat[:, :, start:end] = (
                    warped.cpu().numpy().astype(_out_np_dtype)
                )
            else:
                internal_output[:, :, start:end] = warped.to(
                    device=target, dtype=out_dt,
                )

        if use_external_output:
            return output

        result = internal_output.reshape(original_shape)
        if return_numpy:
            return result.cpu().numpy() if result.is_cuda else result.numpy()
        return result

    def calculate_error(
        self,
        mode: int,
        _warped_t: torch.Tensor | None = None,
    ):
        """Compute per-image MAE against the mean and append to error history.

        Measures how well the warped images agree by computing the mean
        absolute difference of each image from the stack mean. Without
        error tracking, there is no way to verify that alignment steps
        are actually improving the result.

        Parameters
        ----------
        mode : int
            Stage identifier (0=preprocess, 1=affine, 2=nonrigid).
        _warped_t : torch.Tensor or None
            If provided, compute error from this tensor directly,
            avoiding a GPU-to-CPU round-trip.
        """
        self._ensure_single("calculate_error")
        if _warped_t is not None:
            images_mean = _warped_t.mean(dim=0)
            sig_diff = torch.mean(
                torch.abs(_warped_t - images_mean[None]), dim=(1, 2)
            ).cpu().numpy()
        else:
            self._ensure_warped_images()
            images_mean = np.mean(self.imgs_warped.array, axis=0)
            sig_diff = np.mean(
                np.abs(self.imgs_warped.array - images_mean[None, :, :]), axis=(1, 2)
            )

        error_current = np.hstack((mode, np.mean(sig_diff), sig_diff))

        if not hasattr(self, "error_track"):
            self.error_track = error_current[None, :]
        else:
            self.error_track = np.vstack((self.error_track, error_current))

    def _ensure_warped_images(self):
        """Lazily populate images_warped from current knots if marked stale."""
        if getattr(self, "_images_warped_stale", False):
            self._warp_and_translate_torch(
                self._max_image_shift_cached, upsample_factor=8,
                solve_translation=False)
            self._images_warped_stale = False

    # ##################################################################### #
    #   Plotting methods (see drift_viz.py for implementations)  #
    # ##################################################################### #

    def plot_correction_summary(
        self,
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
    ):
        self._ensure_single("plot_correction_summary")
        return drift_viz.plot_correction_summary(
            self, corrected=corrected, reference_index=reference_index,
            target_index=target_index, crop=crop, mode=mode,
            show_fft=show_fft, show_diff=show_diff,
            fft_mask_radius=fft_mask_radius, axsize=axsize, **kwargs,
        )

    @property
    def drift_rate(self) -> tuple[float, float]:
        """Per-scanline drift rate ``(row, col)`` from the affine fit.

        Returns the linear slope of the knot displacement for the last
        image relative to the first.  Only meaningful after
        :meth:`align_affine`.
        """
        self._ensure_single("drift_rate")
        if not hasattr(self, "_initial_knots"):
            raise RuntimeError("Call preprocess() then align_affine() first.")
        idx = len(self.knots) - 1
        # Use affine-only knots if available, else current knots
        knots = (self._knots_after_affine[idx]
                 if hasattr(self, "_knots_after_affine")
                 else self.knots[idx])
        delta = knots - self._initial_knots[idx]  # (2, H, 1)
        n = delta.shape[1]
        row_rate = float((delta[0, -1, 0] - delta[0, 0, 0]) / max(n - 1, 1))
        col_rate = float((delta[1, -1, 0] - delta[1, 0, 0]) / max(n - 1, 1))
        return (row_rate, col_rate)

    def print_drift_stats(self, target_index: int = -1) -> None:
        """Print a concise summary of drift estimation results.

        Reports the linear drift rate, total displacement, and nonrigid
        correction magnitude (if :meth:`align_nonrigid` was run).
        """
        self._ensure_single("print_drift_stats")
        idx = target_index % len(self.knots)
        rate = self.drift_rate
        n = self.knots[idx].shape[1]
        print(f"Drift rate:  ({rate[0]:+.4f}, {rate[1]:+.4f}) px/line")
        print(f"Total drift: ({rate[0]*n:.1f}, {rate[1]*n:.1f}) px "
              f"over {n} lines")
        if hasattr(self, "_knots_after_affine"):
            delta_aff = (self._knots_after_affine[idx]
                         - self._initial_knots[idx])
            delta_nr = self.knots[idx] - self._initial_knots[idx]
            nr_max = float((delta_nr - delta_aff).abs().max())
            print(f"Nonrigid max correction: {nr_max:.2f} px")
        if hasattr(self, "affine_confidence_margin"):
            m = self.affine_confidence_margin
            tag = "high" if m > 5 else "low" if m < 2 else "moderate"
            print(f"Affine confidence: {m:.1f}% ({tag})")

    def plot_correction_comparison(
        self,
        crop: int | None = None,
        target_index: int = -1,
        axsize: tuple[float, float] = (3.5, 3.5),
        show_fft: bool = True,
        **kwargs,
    ):
        self._ensure_single("plot_correction_comparison")
        return drift_viz.plot_correction_comparison(
            self, crop=crop, target_index=target_index,
            axsize=axsize, show_fft=show_fft, **kwargs,
        )

    def plot_radial_power(
        self,
        methods: dict[str, np.ndarray] | None = None,
        crop: int | None = None,
        target_index: int = -1,
        figsize: tuple[float, float] = (10, 6),
    ):
        self._ensure_single("plot_radial_power")
        return drift_viz.plot_radial_power(
            self, methods=methods, crop=crop,
            target_index=target_index, figsize=figsize,
        )

    def plot_warped_images(self, show_knots: bool = True, **kwargs):
        self._ensure_single("plot_warped_images")
        return drift_viz.plot_warped_images(self, show_knots=show_knots, **kwargs)

    def plot_convergence(
        self,
        figsize=(8, 3),
        **kwargs,
    ):
        self._ensure_single("plot_convergence")
        return drift_viz.plot_convergence(self, figsize=figsize, **kwargs)

    def plot_merged_images(self, show_knots: bool = True, **kwargs):
        self._ensure_single("plot_merged_images")
        return drift_viz.plot_merged_images(self, show_knots=show_knots, **kwargs)

    def plot_knots(
        self, figsize: tuple[int, int] | None = None,
    ) -> tuple:
        self._ensure_single("plot_knots")
        return drift_viz.plot_knots(self, figsize=figsize)

    def plot_4dstem_correction(self, cube_raw, cube_corrected, **kwargs):
        """Visualize 4D-STEM correction: VDF, mean DP, CBED comparisons."""
        self._ensure_single("plot_4dstem_correction")
        return drift_viz.plot_4dstem_correction(
            self, cube_raw, cube_corrected, **kwargs,
        )


from dataclasses import dataclass


@dataclass
class PairedCorrectionResult:
    """Result returned by :func:`correct_4dstem_paired`.

    Attributes
    ----------
    merged : np.ndarray or None
        Average of ``corrected_a`` and the rotated ``corrected_b``.
        ``None`` when ``merge=False``.
    corrected_a : np.ndarray
        Drift-corrected first scan, same shape as ``cube_a``.
    corrected_b : np.ndarray
        Drift-corrected second scan, **rotated** into the first scan's
        coordinate frame.  Same shape as ``corrected_a`` for square scans.
    drift : DriftCorrection
        The fitted alignment object (for plots, knot inspection, etc.).
    vdf_a : np.ndarray
        Virtual dark-field image used for alignment (scan A).
    vdf_b : np.ndarray
        Virtual dark-field image used for alignment (scan B).
    """

    merged: np.ndarray | None
    corrected_a: np.ndarray
    corrected_b: np.ndarray
    drift: DriftCorrection
    vdf_a: np.ndarray
    vdf_b: np.ndarray


def correct_4dstem_paired(
    cube_a: np.ndarray,
    cube_b: np.ndarray,
    scan_direction_degrees: list[float],
    *,
    vdf_a: np.ndarray | None = None,
    vdf_b: np.ndarray | None = None,
    preprocess: dict | None = None,
    align_affine: dict | None = None,
    align_nonrigid: dict | bool = False,
    mode: str = "bicubic",
    chunk_size: int | None = None,
    merge: bool = True,
    progress: bool = False,
) -> PairedCorrectionResult:
    """Drift-correct paired orthogonal-scan data cubes end-to-end.

    Handles the full workflow: VDF extraction → alignment → correction
    → frame rotation → merge.  Works for 4D-STEM ``(H, W, det_h, det_w)``
    and 3D EDX/EELS ``(H, W, E)`` cubes alike.

    For memory-constrained workloads (cubes too large for RAM), use the
    lower-level API directly::

        dc = DriftCorrection.from_data([vdf_a, vdf_b], ...)
        dc.preprocess(...).align_affine(...)
        dc.apply_correction_4dstem(cube_a, image_index=0, output=mmap_a)
        del cube_a                          # release
        dc.apply_correction_4dstem(cube_b, image_index=1, output=mmap_b)
        del cube_b                          # release

    Parameters
    ----------
    cube_a, cube_b : np.ndarray
        Data cubes with shape ``(H, W, ...)`` where the first two axes are
        scan rows and columns.  Typically collected at orthogonal scan
        angles (e.g. 0° and 90°).
    scan_direction_degrees : list of float
        Scan direction for each cube, e.g. ``[0, -90]``.
    vdf_a, vdf_b : np.ndarray or None
        Pre-computed virtual dark-field images ``(H, W)``.  When ``None``
        (default), VDFs are computed automatically by averaging over
        the non-scan dimensions.
    preprocess : dict or None
        Kwargs forwarded to :meth:`DriftCorrection.preprocess`.
        ``number_knots`` defaults to 1 if not specified.
    align_affine : dict or None
        Kwargs forwarded to :meth:`DriftCorrection.align_affine`.
    align_nonrigid : dict, bool, or False
        ``False`` (default): skip.  ``True``: run with defaults.
        ``dict``: run with the given kwargs.
    mode : str
        Interpolation mode for ``apply_correction_4dstem``.
    chunk_size : int or None
        GPU channel chunk size (``None`` = auto).
    merge : bool
        If ``True`` (default), average the two corrected cubes.
    progress : bool
        Show tqdm progress bars.

    Returns
    -------
    PairedCorrectionResult
        Dataclass with ``merged``, ``corrected_a``, ``corrected_b``,
        ``drift``, ``vdf_a``, ``vdf_b``.

    Examples
    --------
    >>> result = correct_4dstem_paired(
    ...     cube_0deg, cube_90deg,
    ...     scan_direction_degrees=[0, -90],
    ...     preprocess=dict(pad_fraction=0.25, kde_sigma=0.5),
    ...     align_affine=dict(step=0.02, num_tests=11),
    ... )
    >>> result.merged.shape   # (H, W, det_h, det_w)
    >>> result.drift.plot_merged_images()
    """
    cube_a = np.asarray(cube_a)
    cube_b = np.asarray(cube_b)
    if cube_a.ndim < 3:
        raise ValueError(
            f"cube_a must be at least 3-D (H, W, ...), got {cube_a.ndim}-D"
        )
    if cube_b.ndim < 3:
        raise ValueError(
            f"cube_b must be at least 3-D (H, W, ...), got {cube_b.ndim}-D"
        )
    if len(scan_direction_degrees) != 2:
        raise ValueError(
            "scan_direction_degrees must have exactly 2 entries, "
            f"got {len(scan_direction_degrees)}"
        )

    # ── Rotation validation (only multiples of 90°) ──
    theta_a, theta_b = scan_direction_degrees
    delta = (theta_b - theta_a) % 360
    rot_k = round(delta / 90)
    if abs(delta - rot_k * 90) > 1.0:
        raise ValueError(
            f"Scan angle difference ({theta_b} - {theta_a} = "
            f"{theta_b - theta_a}°) must be a multiple of 90°. "
            f"Arbitrary rotations are not supported."
        )
    rot_k = rot_k % 4  # normalize to 0-3

    # ── Extract VDFs ──
    if vdf_a is None:
        vdf_a = DriftCorrection.compute_vdf(cube_a)
    if vdf_b is None:
        vdf_b = DriftCorrection.compute_vdf(cube_b)

    # ── Align VDFs ──
    pp_kw: dict = dict(show_merged=False, show_images=False, number_knots=1)
    if preprocess is not None:
        pp_kw.update(preprocess)

    dc = DriftCorrection.from_data(
        [vdf_a, vdf_b], scan_direction_degrees,
    )
    dc.preprocess(**pp_kw)

    aa_kw: dict = dict(show_merged=False, show_images=False)
    if align_affine is not None:
        aa_kw.update(align_affine)
    dc.align_affine(**aa_kw)

    if align_nonrigid is not False:
        nr_kw: dict = dict(show_merged=False, show_images=False)
        if isinstance(align_nonrigid, dict):
            nr_kw.update(align_nonrigid)
        dc.align_nonrigid(**nr_kw)

    # ── Apply correction to both cubes ──
    corr_a = dc.apply_correction_4dstem(
        cube_a, image_index=0, mode=mode,
        chunk_size=chunk_size, progress=progress,
    )
    corr_b = dc.apply_correction_4dstem(
        cube_b, image_index=1, mode=mode,
        chunk_size=chunk_size, progress=progress,
    )

    # ── Rotate cube_b to cube_a's coordinate frame ──
    if rot_k != 0:
        corr_b = np.rot90(corr_b, k=rot_k, axes=(0, 1)).copy()

    # ── Merge ──
    merged = None
    if merge:
        if corr_a.shape != corr_b.shape:
            raise ValueError(
                f"Cannot merge: corrected_a shape {corr_a.shape} != "
                f"rotated corrected_b shape {corr_b.shape}. "
                f"Paired scans must have compatible scan dimensions "
                f"after rotation."
            )
        merged = (corr_a.astype(np.float64) + corr_b.astype(np.float64))
        merged = (merged / 2).astype(np.float32)

    return PairedCorrectionResult(
        merged=merged,
        corrected_a=corr_a,
        corrected_b=corr_b,
        drift=dc,
        vdf_a=vdf_a,
        vdf_b=vdf_b,
    )


def correct_series(
    images_a: NDArray,
    images_b: NDArray,
    scan_direction_degrees: list[float],
    *,
    preprocess: dict | None = None,
    align_affine: dict | None = None,
    align_nonrigid: dict | bool = False,
    generate: dict | None = None,
) -> tuple[np.ndarray, list[DriftCorrection]]:
    """Drift-correct a paired image series frame by frame.

    Convenience wrapper around
    ``DriftCorrection.from_data([stack_a, stack_b], ...)``.  Each stage is
    configured through a keyword dictionary whose keys are the same as the
    corresponding ``DriftCorrection`` method parameters.

    Parameters
    ----------
    images_a : (N, H, W) ndarray
        First scan direction image stack (e.g. 0-deg scans).
    images_b : (N, H, W) ndarray
        Second scan direction image stack (e.g. 90-deg scans).
    scan_direction_degrees : list of float
        Scan direction angles, e.g. ``[0, -90]``.
    preprocess : dict or None
        Keyword arguments forwarded to
        :meth:`DriftCorrection.preprocess`.  ``None`` uses method defaults.
    align_affine : dict or None
        Keyword arguments forwarded to
        :meth:`DriftCorrection.align_affine`.  ``None`` uses method defaults.
    align_nonrigid : dict, bool, or False
        Controls nonrigid alignment after affine:

        - ``False`` (default): skip nonrigid alignment.
        - ``True``: run :meth:`DriftCorrection.align_nonrigid` with defaults.
        - ``dict``: run nonrigid with the given keyword arguments.
    generate : dict or None
        Keyword arguments forwarded to
        :meth:`DriftCorrection.generate_corrected_image`.  ``None`` uses
        method defaults.  ``strip_padding=True`` is set by default.

    Returns
    -------
    corrected : (N, H', W') float32 ndarray
        Stack of drift-corrected images.
    drift_objects : list of DriftCorrection
        One object per frame for post-hoc inspection (knots, plots, etc.).

    Examples
    --------
    >>> corrected, objs = correct_series(
    ...     images_0deg, images_90deg,
    ...     scan_direction_degrees=[0, -90],
    ...     preprocess=dict(pad_fraction=0.25, kde_sigma=0.5, number_knots=1),
    ...     align_affine=dict(step=0.02, num_tests=11),
    ...     generate=dict(upsample_factor=1, kde_sigma=0.5),
    ... )
    """
    images_a = np.asarray(images_a)
    images_b = np.asarray(images_b)
    if images_a.ndim != 3:
        raise ValueError(
            f"images_a must be 3-D (N, H, W), got shape {images_a.shape}"
        )
    if images_b.ndim != 3:
        raise ValueError(
            f"images_b must be 3-D (N, H, W), got shape {images_b.shape}"
        )
    if images_a.shape != images_b.shape:
        raise ValueError(
            f"Shape mismatch: images_a {images_a.shape} != "
            f"images_b {images_b.shape}"
        )

    dc = DriftCorrection.from_data(
        [images_a, images_b], scan_direction_degrees
    )

    dc.preprocess(**(preprocess or {}))
    dc.align_affine(**(align_affine or {}))

    if align_nonrigid is not False:
        nr_kw = align_nonrigid if isinstance(align_nonrigid, dict) else {}
        dc.align_nonrigid(**nr_kw)

    gen_kw = dict(strip_padding=True)
    if generate is not None:
        gen_kw.update(generate)
    corrected = dc.generate_corrected_image(**gen_kw)

    return corrected, list(dc)


class _DriftInterpolator:
    def __init__(
        self,
        input_shape,
        output_shape,
        scan_fast,
        pad_value,
        kde_sigma,
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.scan_fast = scan_fast
        self.pad_value = pad_value
        self.kde_sigma = kde_sigma
        self.u = np.linspace(0, 1, input_shape[1])

    def transform_rows(
        self,
        knots_row: NDArray,
    ):
        num_knots = knots_row.shape[-1]
        basis = np.linspace(0, 1, num_knots)

        if num_knots == 1:
            row_coords = knots_row[0] + self.u[None, :] * self.scan_fast[0] * (self.input_shape[0] - 1)
            col_coords = knots_row[1] + self.u[None, :] * self.scan_fast[1] * (self.input_shape[1] - 1)
        elif num_knots == 2:
            row_coords = interp1d(basis, knots_row[0], kind="linear", assume_sorted=True)(self.u)
            col_coords = interp1d(basis, knots_row[1], kind="linear", assume_sorted=True)(self.u)
        else:
            kind = "quadratic" if num_knots == 3 else "cubic"
            row_coords = interp1d(
                basis,
                knots_row[0],
                kind=kind,
                fill_value="extrapolate",
                assume_sorted=True,
            )(self.u)
            col_coords = interp1d(
                basis,
                knots_row[1],
                kind=kind,
                fill_value="extrapolate",
                assume_sorted=True,
            )(self.u)

        return row_coords, col_coords

    def transform_coordinates(
        self,
        knots: NDArray,
    ):
        num_knots = knots.shape[-1]

        if num_knots == 1:
            row_coords, col_coords = self.transform_rows(knots)
        else:
            row_coords = np.zeros(self.input_shape)
            col_coords = np.zeros(self.input_shape)
            for i in range(self.input_shape[0]):
                row_coords[i], col_coords[i] = self.transform_rows(knots[:, i])

        return row_coords, col_coords

    def warp_image(
        self,
        image: NDArray,
        knots: NDArray,  # shape: (2, rows, num_knots)
    ) -> NDArray:
        row_coords, col_coords = self.transform_coordinates(knots)
        image_interp, weight_interp = bilinear_kde(
            xa=row_coords,
            ya=col_coords,
            values=image,
            output_shape=self.output_shape,
            kde_sigma=self.kde_sigma,
            pad_value=self.pad_value,
            return_pix_count=True,
        )
        return image_interp, weight_interp


def _bounded_sine_sigmoid_torch(x: torch.Tensor, midpoint: float = 0.5,
                                 width: float = 1.0) -> torch.Tensor:
    """Bounded sine sigmoid (branchless, GPU-friendly).

    Maps values smoothly from 0 to 1 using a sine-squared transition
    between ``midpoint - width/2`` and ``midpoint + width/2``.
    """
    width = min(width, 2 * midpoint, 2 * (1 - midpoint))
    left = midpoint - width / 2
    right = midpoint + width / 2
    t = ((x - left) / width).clamp(0.0, 1.0)
    return torch.where(x > right, torch.ones_like(x), torch.sin(t * (np.pi / 2)) ** 2)


def _fourier_crop_torch(fft_array: torch.Tensor,
                         crop_shape: tuple[int, int]) -> torch.Tensor:
    """Crop a corner-centered FFT tensor to retain only lowest frequencies.

    Torch equivalent of :func:`fourier_cropping` — slices the four
    corner quadrants of the FFT to produce a smaller spectrum that,
    when inverse-transformed, gives a lower-resolution version of the
    original signal.
    """
    crop_h, crop_w = crop_shape
    h1 = crop_h // 2
    h2 = crop_h - h1
    w1 = crop_w // 2
    w2 = crop_w - w1
    result = torch.zeros(crop_shape, dtype=fft_array.dtype, device=fft_array.device)
    result[:h1, :w1] = fft_array[:h1, :w1]
    result[:h1, -w2:] = fft_array[:h1, -w2:]
    result[-h2:, :w1] = fft_array[-h2:, :w1]
    result[-h2:, -w2:] = fft_array[-h2:, -w2:]
    return result
