from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.fft import fftfreq
from numpy.typing import NDArray
from tqdm import tqdm

from quantem.core.config import validate_device
from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.compound_validators import (
    validate_list_of_dataset2d,
    validate_pad_value,
)
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization import show_2d
from quantem.imaging.drift_knot import (
    DriftKnot,
    bilinear_kde_batch,
    initialize_scanline_knots,
)
from quantem.imaging.drift_align import (
    backward_warp,
    backward_warp_grid_search,
    cross_corr_batch,
    translate_align,
)
import quantem.imaging.drift_4dstem as _4dstem
import quantem.imaging.drift_optimize as _optimize
import quantem.imaging.drift_visualization as drift_visualization


def _as_array(x):
    """Extract the array from an input dataset.

    Accepts a numpy ``ndarray`` (returned as-is), a ``torch.Tensor``
    (returned as-is, so device tensors stay in place through
    apply_correction), or any ``Dataset``-like wrapper exposing a
    ``.array`` attribute.  Raises ``TypeError`` for paths or other
    inputs: load files yourself with ``Dataset_X.from_file()`` and
    pass the resulting Dataset object.
    """
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return x
    arr = getattr(x, "array", None)
    if isinstance(arr, np.ndarray):
        return arr
    raise TypeError(
        f"DriftCorrection accepts ndarray, torch.Tensor, or Dataset "
        f"objects; got {type(x).__name__}. To load from disk, call "
        f"Dataset2d.from_file(path) (or Dataset4d.from_file) first.")


# CorrectionResult lives in drift_4dstem so the dataset-shaped
# concerns stay together.
from quantem.imaging.drift_4dstem import (  # noqa: E402, F401
    CorrectionResult,
)


def _distance_transform_edt_torch(mask: torch.Tensor) -> torch.Tensor:
    """Torchified euclidean distance transform — GPU drop-in for scipy.

    For each True pixel of ``mask``, returns the euclidean distance to the
    nearest False pixel; False pixels return 0. Bit-exact match to
    ``scipy.ndimage.distance_transform_edt`` on single-region masks (the shape
    that arises in drift-correction edge blending) — verified by parity test
    in ``test_drift.py``.

    Implementation: Jump Flooding Algorithm. Each pixel tracks the (row, col)
    of its nearest known False seed. Each pass examines 8 candidate seeds at
    a halving step distance and adopts the closest one. Runs in
    O(log max(H, W)) parallel passes — typically ~10 passes for a 1024² mask.

    Parameters
    ----------
    mask : torch.BoolTensor of shape (H, W) on GPU
        True where distance should be measured FROM, False where distance is 0.

    Returns
    -------
    distance : torch.FloatTensor of shape (H, W)
        Euclidean distance per pixel, on the same device as ``mask``.
    """
    H, W = mask.shape
    device = mask.device
    row_idx, col_idx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    # Sentinel = "no seed yet" (any value larger than the canvas works as a
    # never-wins distance for the comparisons below).
    NO_SEED = H + W + 1
    seed_row = torch.where(~mask, row_idx, torch.full_like(row_idx, NO_SEED))
    seed_col = torch.where(~mask, col_idx, torch.full_like(col_idx, NO_SEED))

    step = max(H, W) // 2
    while step >= 1:
        best_seed_row = seed_row.clone()
        best_seed_col = seed_col.clone()
        best_dist_sq = (row_idx - best_seed_row) ** 2 + (col_idx - best_seed_col) ** 2

        # 8 neighbors at offsets ±step / 0 (excluding center)
        for delta_row in (-step, 0, step):
            for delta_col in (-step, 0, step):
                if delta_row == 0 and delta_col == 0:
                    continue
                cand_seed_row = torch.roll(seed_row, shifts=(delta_row, delta_col), dims=(0, 1))
                cand_seed_col = torch.roll(seed_col, shifts=(delta_row, delta_col), dims=(0, 1))
                # Invalidate wrap-around regions so torus-rolled seeds don't
                # poison real candidates.
                if delta_row > 0:
                    cand_seed_row[:delta_row] = NO_SEED
                    cand_seed_col[:delta_row] = NO_SEED
                elif delta_row < 0:
                    cand_seed_row[delta_row:] = NO_SEED
                    cand_seed_col[delta_row:] = NO_SEED
                if delta_col > 0:
                    cand_seed_row[:, :delta_col] = NO_SEED
                    cand_seed_col[:, :delta_col] = NO_SEED
                elif delta_col < 0:
                    cand_seed_row[:, delta_col:] = NO_SEED
                    cand_seed_col[:, delta_col:] = NO_SEED

                cand_dist_sq = (
                    (row_idx - cand_seed_row) ** 2 + (col_idx - cand_seed_col) ** 2
                )
                closer = cand_dist_sq < best_dist_sq
                best_dist_sq = torch.where(closer, cand_dist_sq, best_dist_sq)
                best_seed_row = torch.where(closer, cand_seed_row, best_seed_row)
                best_seed_col = torch.where(closer, cand_seed_col, best_seed_col)

        seed_row = best_seed_row
        seed_col = best_seed_col
        step //= 2

    distance = torch.sqrt(((row_idx - seed_row) ** 2 + (col_idx - seed_col) ** 2).float())
    distance[~mask] = 0.0
    return distance


class DriftCorrection(AutoSerialize):
    """GPU-accelerated drift correction for scan-angle electron microscopy data.

    Aligns two (or more) images acquired at different scan directions to
    recover per-scanline drift, then produces a corrected output free of
    raster distortion. The same pipeline handles single 2-D image pairs,
    scan image series (tilt/time), and 4D-STEM collection / EDX datasets.

    Construction
    ------------
    Build a ``DriftCorrection`` from the constructor for scalar image pairs,
    or use the named constructors for dataset workflows where the intent
    should be explicit:

    ============================ =================== ==============================================
    Inputs                       ``scan_direction``  Mode + output of ``generate_corrected``
    ============================ =================== ==============================================
    2-D, 2-D                     different angles    scan collection alignment → :class:`Dataset2d`
    2-D, 2-D                     same angle          reference (a=ref, b=drifted) → :class:`Dataset2d`
    2-D, 3-D ``(H, W, E)``       any                 reference → :class:`Dataset3d` (corrected EDS/EELS)
    2-D, 4-D ``(H, W, det, det)``any                 reference → :class:`Dataset4d` (corrected 4D-STEM)
    4-D, 4-D                     orthogonal          4D-STEM collection merge → :class:`CorrectionResult`
    ============================ =================== ==============================================

    For tilt/time series of scan images, loop in user code:
    ``for i in range(N): DriftCorrection(stack_a[i], stack_b[i], ...)``.

    Inputs may be raw ``ndarray`` or any ``Dataset`` wrapper.  File loading
    is the user's responsibility — call ``Dataset2d.from_file(path)`` /
    ``Dataset4d.from_file(path)`` yourself and pass the result.

    The pipeline (``preprocess`` → ``align_affine`` → ``align_nonrigid`` →
    ``generate_corrected``) is identical across modes.

    Examples
    --------
    Scan collection 0°/90° HAADF:

    >>> dc = DriftCorrection(im0, im90, scan_direction_degrees=(0, 90))
    >>> dc.preprocess(pad_fraction=0.25, kde_sigma=0.5, number_knots=1)
    >>> dc.align_affine(step=0.02, num_tests=11)
    >>> dc.align_nonrigid()                       # optional
    >>> result = dc.generate_corrected()          # → Dataset2d

    Scan collection 0°/90° 4D-STEM:

    >>> dc = DriftCorrection.from_4dstem(
    ...     data_0, data_1, scan_direction_degrees=(0, 90))
    >>> result = dc.preprocess().align_affine().generate_corrected_4dstem()
    >>> result.corrected_4dstem, result.corrected_4dstem_0, result.corrected_4dstem_1

    HAADF reference + drifted EDS/EELS or 4D-STEM:

    >>> dc = DriftCorrection.from_reference(haadf, eds_data)
    >>> dc.preprocess(normalize=True).align_affine()
    >>> result = dc.generate_corrected()                  # → Dataset3d (corrected EDS)

    Loading from disk:

    >>> from quantem.core.datastructures.dataset2d import Dataset2d
    >>> ds_a = Dataset2d.from_file("haadf_0.tif")
    >>> ds_b = Dataset2d.from_file("haadf_90.tif")
    >>> dc = DriftCorrection(ds_a, ds_b, scan_direction_degrees=(0, 90))

    Applying learned drift to arbitrary data
    ----------------------------------------
    Use :meth:`apply_correction` to warp other 2-D, 3-D, or 4-D arrays
    with the same drift model. The method auto-detects scan-axis layout
    and chunks GPU memory for ≥3-D datasets.

    Performance
    -----------
    ``align_affine`` uses PyTorch on GPU (CUDA, MPS, or CPU) with batched
    grid search, scatter-based warping, and FFT cross-correlation. Roughly
    300× faster than the original NumPy implementation (~436 s → 1.5 s for
    2048×2048 image pairs). Memory auto-chunks when the full batch doesn't
    fit.

    Affine-search memory per candidate:

    ============ ============ ================
    Input size   Canvas size  Mem / candidate
    ============ ============ ================
    1024×1024    1280×1280    85 MB
    2048×2048    2560×2560    341 MB
    4096×4096    5120×5120    1.36 GB
    ============ ============ ================

    4D-STEM ``apply_correction`` single-shot fits: roughly 2× the
    dataset size in free GPU memory. A 19 GB dataset (512×512×192×192 float32)
    runs single-shot on 96 GB GPUs and chunks automatically below that.
    """

    def __init__(
        self,
        *datasets: Dataset2d | Dataset3d | NDArray,
        scan_direction_degrees: list[float] | NDArray | float = (0.0, 90.0),
        alignment_image: NDArray | None = None,
    ):
        """Initialize a DriftCorrection from two or more related datasets.

        Mode is selected by the shapes of the inputs and the angles supplied
        — see the class docstring for the full dispatch table.

        Parameters
        ----------
        *datasets : ndarray, Dataset2d, Dataset3d, or Dataset4d
            Two or more datasets to align.  The first is always the
            reference / first image; subsequent inputs may be 2-D, 3-D,
            or 4-D depending on the mode.  Pass raw ``ndarray`` or the
            ``Dataset`` wrapper (``.array`` is extracted automatically).
            To load from disk, call ``Dataset2d.from_file(...)`` /
            ``Dataset4d.from_file(...)`` yourself first.
        scan_direction_degrees : sequence of floats or single float
            Scan angle per dataset.  A single float broadcasts to all
            inputs (typical for HAADF + EDS reference workflows).  For
            image-series alignment, pass one angle per dataset.
        alignment_image : 2-D ndarray, optional
            Pre-computed VDF / summary of the second dataset, used as
            the alignment partner when the second dataset is ≥3-D in
            reference mode.  When ``None``, computed automatically via
            :meth:`compute_vdf`.

        Four supported use cases (the dispatch picks one by input shapes
        and angles)
        --------------------------------------------------------------------
        ===  ===================================  =============================  ============================  ===============================
        #    Use case                             Inputs                         Returns from                  Demo notebook
                                                                                  ``generate_corrected``
        ===  ===================================  =============================  ============================  ===============================
        1    2-D scan images                      2+× ``(H, W)``                :class:`Dataset2d`            ``api/01_from_images.ipynb``
        2    0/90 4-D STEM collection             2× ``(H, W, D_h, D_w)``        :class:`CorrectionResult`    ``api/03_from_4dstem.ipynb``
        3    Reference + drifted 4-D STEM         ``(H, W)`` + ``(H, W, D_h, D_w)``  :class:`Dataset4d`        ``api/04_from_reference_4dstem.ipynb``
        4    Reference + drifted 3-D EDS / EELS   ``(H, W)`` + ``(H, W, n_E)``   :class:`Dataset3d`            ``api/05_from_reference_eds.ipynb``
        ===  ===================================  =============================  ============================  ===============================

        Cases 1-2 are scan collections (two or more scans of the same area);
        cases 3-4 are *reference-mode* (one HAADF
        reference + one drifted dataset of the same scan, single-sided).
        Multi-angle HAADF (3+ inputs at different angles) follows the
        case-1 dispatch with ``len(datasets) > 2``.

        For tilt/time series of scan images, loop in user code:

        >>> for i in range(N):
        ...     dc = DriftCorrection.from_images(stack_a[i], stack_b[i], scan_direction_degrees=(0, 90))
        ...     out[i] = dc.preprocess().align_affine().generate_corrected().array

        Examples
        --------
        Case 1: 2-D HAADF scan images (0° / 90°):

        >>> dc = DriftCorrection.from_images(im0, im90, scan_direction_degrees=(0, 90))

        Case 2: 0/90 4D-STEM collection:

        >>> dc = DriftCorrection.from_4dstem(data_0, data_1, scan_direction_degrees=(0, 90))

        Case 3: HAADF reference + drifted 4-D STEM (single-sided):

        >>> dc = DriftCorrection.from_reference(haadf, data_drifted)

        Case 4: HAADF reference + drifted EDS spectral dataset:

        >>> dc = DriftCorrection.from_reference(haadf, eds_data)

        Multi-angle HAADF (e.g. 0° / 45° / 90°, follows case 1):

        >>> dc = DriftCorrection.from_images(im0, im45, im90, scan_direction_degrees=(0, 45, 90))
        """
        # Core state (always set so all code paths can rely on these).
        self._datasets: list[np.ndarray | None] | None = None
        self._datasets_consumed: bool = False
        self._normalized: bool = False
        self._reference_mode: bool = False
        device, _ = validate_device(None)
        self._device = device
        self._dtype = torch.float32

        if not datasets and alignment_image is None:
            return
        self._dispatch_and_setup(datasets, scan_direction_degrees, alignment_image)

    @classmethod
    def from_images(
        cls,
        *images,
        scan_direction_degrees: list[float] | NDArray | tuple[float, ...] = (0.0, 90.0),
    ) -> Self:
        """Create drift correction from two or more 2-D scan images."""
        return cls(*images, scan_direction_degrees=scan_direction_degrees)

    @classmethod
    def from_4dstem(
        cls,
        *datasets,
        scan_direction_degrees: list[float] | NDArray | tuple[float, ...] = (0.0, 90.0),
    ) -> Self:
        """Create a first-class 0/90 4D-STEM collection drift correction.

        The datasets are treated as independently drifted scans of the same
        specimen region. Alignment is estimated from their auto-extracted
        virtual images, then the learned scan-derived drift fields can be
        applied to the full diffraction-pattern datasets via
        :meth:`generate_corrected_4dstem`.

        Currently this path supports exactly two orthogonal 4D-STEM datasets.
        The constructor accepts a dataset collection so future scan-angle
        sets can use the same public API.

        This is distinct from reference mode:
        ``DriftCorrection.from_reference(reference_2d, drifted_dataset)``
        keeps image 0
        fixed and corrects one dataset toward that external reference.
        """
        return cls(
            *datasets,
            scan_direction_degrees=scan_direction_degrees,
        )

    @classmethod
    def from_reference(
        cls,
        reference_image,
        drifted_dataset,
        *,
        alignment_image: NDArray | None = None,
        scan_direction_degrees: list[float] | NDArray | float = 0.0,
    ) -> Self:
        """Create a reference-anchored drift correction.

        ``reference_image`` is a 2-D image that defines the fixed coordinate
        frame. ``drifted_dataset`` may be a 2-D image, a 3-D spectral dataset
        such as EDS/EELS, or a 4-D STEM dataset with scan axes leading. Alignment
        is estimated against either ``alignment_image`` or an automatically
        extracted virtual image from the drifted dataset, then
        :meth:`generate_corrected` warps the drifted dataset into the
        reference frame.

        This is distinct from 0/90 4D-STEM collection correction: reference mode
        anchors image 0 and corrects one target dataset; it does not merge two
        independently drifted scans.
        """
        result = cls(
            reference_image,
            drifted_dataset,
            scan_direction_degrees=scan_direction_degrees,
            alignment_image=alignment_image,
        )
        if not result._reference_mode:
            raise ValueError(
                "from_reference() requires a 2-D reference image and one "
                "drifted target dataset. For a 2-D target, pass a scalar "
                "scan_direction_degrees value or matching reference/target "
                "scan directions so the call is unambiguously single-sided. "
                "Use DriftCorrection.from_4dstem() for 0/90 4D-STEM "
                "collection correction."
            )
        return result

    def _dispatch_and_setup(self, datasets, scan_direction_degrees, alignment_image):
        """Validate inputs and populate mode-specific state."""
        if len(datasets) < 2:
            raise TypeError(
                f"DriftCorrection requires at least 2 datasets, got {len(datasets)}")

        # Normalize angles
        if np.isscalar(scan_direction_degrees):
            sd = [float(scan_direction_degrees)] * len(datasets)
        else:
            sd = [float(a) for a in scan_direction_degrees]
            if len(sd) != len(datasets):
                raise ValueError(
                    f"scan_direction_degrees length ({len(sd)}) must match "
                    f"number of datasets ({len(datasets)})")

        # Extract arrays
        arrays = [_as_array(d) for d in datasets]
        ndims = [a.ndim for a in arrays]
        for i, n in enumerate(ndims):
            if n < 2:
                raise TypeError(f"dataset {i} must be ≥2-D, got ndim={n}")

        # ── N == 2: dispatch by shape combo ──
        if len(arrays) == 2:
            a, b = arrays
            a_ndim, b_ndim = ndims

            # Reference mode: 2-D ref + ≥3-D drifted, or two 2-D images at the same angle.
            if a_ndim == 2 and (b_ndim >= 3 or sd[0] == sd[1]):
                if a.shape != b.shape[:2]:
                    raise ValueError(
                        f"reference shape {a.shape} must match the leading "
                        f"two axes of drifted (got {b.shape[:2]})")
                if alignment_image is not None:
                    vdf = np.asarray(alignment_image)
                elif b_ndim == 2:
                    vdf = b
                else:
                    vdf = self.compute_vdf(b)
                self._setup_image_collection([a, vdf], sd)
                self._reference_mode = True
                self._datasets = [None, b]
                return

            # ≥3-D ds_a + 2-D ds_b: confusing input order
            if a_ndim >= 3 and b_ndim == 2:
                raise TypeError(
                    f"first dataset is {a_ndim}-D but second is 2-D. For "
                    f"reference + drifted workflows pass the 2-D reference "
                    f"first: DriftCorrection.from_reference(reference_2d, drifted_dataset)")

            # 3-D image stacks are not a single correction object; loop in user code.
            if a_ndim == 3 and b_ndim == 3:
                raise TypeError(
                    "3-D image series support was removed. Loop in user code: "
                    "for i in range(N): DriftCorrection(stack_a[i], stack_b[i], ...)")

            # 4D-STEM collection merge
            if a_ndim >= 4 and b_ndim >= 4:
                if a_ndim != b_ndim:
                    raise TypeError(
                        f"4D-STEM collection expects matching ndim, got "
                        f"{a_ndim} and {b_ndim}.")
                if a.shape[:2] != b.shape[:2]:
                    raise ValueError(
                        f"4D-STEM collection scan dims {a.shape[:2]} must match "
                        f"second dataset scan dims {b.shape[:2]}")
                normalized = [int(round(angle)) % 360 for angle in sd[:2]]
                if any(abs(angle - round(angle)) > 1e-6 for angle in sd[:2]):
                    raise ValueError(
                        f"4D-STEM collection scan directions must be explicit "
                        f"0/90/-90 degree values, got {sd[:2]}.")
                if any(angle not in {0, 90, 270} for angle in normalized):
                    raise ValueError(
                        f"4D-STEM collection scan directions must be explicit "
                        f"0/90/-90 degree values, got {sd[:2]}.")
                delta = (normalized[1] - normalized[0]) % 360
                if delta not in {90, 270}:
                    raise ValueError(
                        f"4D-STEM collection scans need orthogonal 0/90 scan "
                        f"directions, got angle difference {sd[1] - sd[0]}°.")
                if alignment_image is not None:
                    raise TypeError(
                        "alignment_image= is only meaningful in reference mode; "
                        "for 4D-STEM collections the VDFs are auto-extracted.")
                vdf_a = self.compute_vdf(a)
                vdf_b = self.compute_vdf(b)
                self._setup_image_collection([vdf_a, vdf_b], sd)
                self._datasets = [a, b]
                return

            # 2-D scan image collection (different angles)
            if a.shape != b.shape:
                raise ValueError(
                    f"2-D scan image dims {a.shape} must match second "
                    f"image scan dims {b.shape}")
            self._setup_image_collection([a, b], sd)
            return

        # ── N ≥ 3: multi-angle alignment, all 2-D for now ──
        if all(n == 2 for n in ndims):
            self._setup_image_collection(arrays, sd)
            return

        raise TypeError(
            f"Unsupported input combination: {len(arrays)} datasets with "
            f"ndims {ndims}.  Multi-angle alignment (N≥3) currently supports "
            f"2-D inputs only.")

    def _setup_image_collection(self, arrays, scan_direction_degrees):
        """Populate state for standard 2-D image collection alignment."""
        self.imgs = validate_list_of_dataset2d(arrays)
        self.scan_direction_degrees = ensure_valid_array(
            scan_direction_degrees, ndim=1)

    @property
    def is_4dstem(self) -> bool:
        """True if this instance holds ≥3-D dataset(s) for correction.

        Covers both 4D-STEM collection mode (two datasets at orthogonal scan
        angles) *and* reference mode (one reference image + one drifted
        dataset).  Use :attr:`_is_4dstem_collection` to distinguish from
        reference mode.
        """
        return self._datasets is not None

    @property
    def _is_4dstem_collection(self) -> bool:
        """True only for 4D-STEM collection (two datasets, not reference mode)."""
        return self._datasets is not None and not self._reference_mode

    def _show_after_step(self, label: str, show_merged: bool, show_images: bool,
                          show_knots: bool, kwargs: dict):
        """Render the merged/per-image plots after an alignment step.

        ``show_knots`` here is purely the overlay flag forwarded to the
        merged/warped image plots. For the standalone 2-panel knot
        trajectory + per-row delta figure, call ``dc.plot_knots()``
        explicitly after alignment.
        """
        kwargs.pop("title", None)
        if show_merged:
            self.plot_merged_images(
                show_knots=show_knots, title=f"Merged: {label}", **kwargs)
        if show_images:
            self.plot_warped_images(
                show_knots=show_knots,
                title=[f"Image {i}: {label}" for i in range(self.shape[0])],
                **kwargs,
            )

    def _interpolator(self, img_idx: int, knots: torch.Tensor | None = None) -> DriftKnot:
        """Build the K-aware interpolator for image ``img_idx``.

        ``knots`` defaults to ``self.knots[img_idx]``; callers pass an
        override when they need to warp without mutating the stored knot
        grid (e.g. detached copies inside the optimizer).
        """
        if knots is None:
            knots = self.knots[img_idx]
        return DriftKnot(
            knots, self.scan_fast_t[img_idx], self.scan_slow_t[img_idx],
            self.imgs[img_idx].shape)

    def _knot_delta_canvas(self, idx: int) -> torch.Tensor:
        """Canvas-space knot delta for image ``idx``, shape ``(2, H, K)``.

        Subtracts ``self._initial_knots[idx]`` from ``self.knots[idx]``,
        leaving the trailing knot axis intact so K=1 and K>1 callers share
        the same shape.  Raises ``RuntimeError`` if ``preprocess`` /
        ``align_affine`` haven't run.
        """
        if not hasattr(self, "_initial_knots"):
            raise RuntimeError(
                "apply_correction() requires preprocess() and align_affine() "
                "first. Run dc.preprocess().align_affine() (and optionally "
                ".align_nonrigid()) before apply_correction().")
        return self.knots[idx] - self._initial_knots[idx]

    def drift_field(self, idx: int) -> torch.Tensor:
        """Per-scanline drift in raw-frame pixel coordinates for image ``idx``.

        Returns ``(2, H)`` for K=1 (single-knot mode, drift constant across cols)
        or ``(2, H, W)`` for K>=2. First axis is (row_drift, col_drift).
        """
        if not hasattr(self, "_initial_knots"):
            raise RuntimeError(
                "drift_field() requires preprocess() and align_affine() first. "
                "Run dc.preprocess().align_affine() (and optionally "
                ".align_nonrigid()) before drift_field().")
        return self._interpolator(idx).drift_raw(self._initial_knots[idx])

    def probe_positions(
        self,
        image_index: int = 0,
        *,
        corrected: bool = True,
        strip_padding: bool = True,
        plot: bool = True,
        stride: int = 16,
    ) -> np.ndarray:
        """Return nominal or drift-updated probe positions for one scan image.

        The returned array has shape ``(scan_h, scan_w, 2)`` in ``(row, col)``
        order. ``positions[r, c]`` belongs to the raw diffraction pattern
        acquired at ``dataset[r, c]`` for the same ``image_index``. For a
        0/90 4D-STEM collection, call this separately for image 0 and image 1;
        both outputs are expressed in the same corrected coordinate frame, so
        they can be used as initial coordinates for iterative ptychography
        without interpolating the diffraction patterns.

        Parameters
        ----------
        image_index : int, default 0
            Which scan image / 4D-STEM dataset to export positions for.
        corrected : bool, default True
            ``True`` returns the fitted drift-updated positions. ``False``
            returns the nominal positions before alignment.
        strip_padding : bool, default True
            Subtract the preprocessing canvas padding so coordinates are in
            the original image-0 pixel frame. ``False`` returns padded-canvas
            coordinates.
        plot : bool, default True
            Also draw a nominal-vs-corrected position plot for this image.
        stride : int, default 16
            Subsampling stride used by the plot only.
        """
        if not hasattr(self, "_initial_knots"):
            raise RuntimeError(
                "probe_positions() requires preprocess() first. Run "
                "dc.preprocess() before exporting nominal or corrected "
                "probe positions."
            )
        idx = image_index % len(self.imgs)
        knots = self.knots[idx] if corrected else self._initial_knots[idx]
        row_t, col_t = self._interpolator(idx, knots).to_canvas()
        positions = torch.stack([row_t, col_t], dim=-1)
        if strip_padding:
            scan_h, scan_w = self.imgs[0].shape[:2]
            canvas_h, canvas_w = self.shape[1], self.shape[2]
            pad_h = (canvas_h - scan_h) / 2.0
            pad_w = (canvas_w - scan_w) / 2.0
            offset = torch.tensor(
                [pad_h, pad_w],
                device=positions.device,
                dtype=positions.dtype,
            )
            positions = positions - offset
        positions_np = positions.detach().cpu().numpy().astype(np.float32)
        if plot:
            self.plot_probe_positions(
                image_index=idx,
                strip_padding=strip_padding,
                stride=stride,
            )
        return positions_np

    def preprocess(
        self,
        pad_fraction: float = 0.25,
        pad_value: float | str | list[float] = "median",
        kde_sigma: float = 0.5,
        number_knots: int = 1,
        normalize: bool = False,
        show_merged: bool = False,
        show_images: bool = False,
        overlay_knots: bool = True,
        show_knot_plot: bool = False,
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
        overlay_knots : bool, default True
            Overlay knot positions on top of the merged/warped image plots
            (cheap, useful diagnostic).
        show_knot_plot : bool, default False
            Render the standalone 2-panel knot trajectory + per-row delta
            chart via ``dc.plot_knots()`` after this step.
            Overlay knot positions on displayed images.
        **kwargs
            Additional keyword arguments passed to plotting functions.

        Returns
        -------
        Self
            For method chaining: ``drift.preprocess().align_affine()``.

        Examples
        --------
        >>> drift = DriftCorrection(
        ...     im0, im1, scan_direction_degrees=[0, 90])
        >>> drift.preprocess(pad_fraction=0.25, kde_sigma=0.5, number_knots=1)

        For mixed-type images (HAADF + VDF), use normalize:

        >>> drift = DriftCorrection(
        ...     haadf_ref, vdf, scan_direction_degrees=[0, 0])
        >>> drift.preprocess(normalize=True).align_affine(fixed_indices=[0])
        """
        self._normalized = bool(normalize)
        if normalize:
            for img in self.imgs:
                arr = img.array.astype(np.float32)
                lo, hi = arr.min(), arr.max()
                img.array = (arr - lo) / (hi - lo + 1e-8)
        self.pad_fraction = float(pad_fraction)
        self.pad_value = validate_pad_value(pad_value, self.imgs)
        self.kde_sigma = float(kde_sigma)
        K = int(number_knots)
        if K < 1:
            raise ValueError(f"number_knots must be >= 1 (got {number_knots}).")
        self.number_knots = K
        # Multi-direction scan collection (e.g. 0° / 90°) require square images.
        # The canvas geometry assumes a single scanline length, which only
        # holds when H == W; non-square scans yield inconsistent per-image
        # walks that the optimizer can't reconcile.
        unique_dirs = {round(d, 6) for d in self.scan_direction_degrees}
        if len(unique_dirs) > 1:
            for i, img in enumerate(self.imgs):
                if img.shape[0] != img.shape[1]:
                    raise ValueError(
                        f"Multi-direction scan collection require square images, "
                        f"but image {i} is {img.shape}. Either crop to square "
                        f"or use a single scan direction.")
        self.scan_direction = np.deg2rad(self.scan_direction_degrees)
        self.scan_fast = np.stack(
            [np.sin(self.scan_direction), np.cos(self.scan_direction)], axis=1)
        self.scan_slow = np.stack(
            [np.cos(self.scan_direction), -np.sin(self.scan_direction)], axis=1)
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
        # Per-image fast-axis parametrization u ∈ [0, 1] indexed by column.
        self.u_per_image = [
            np.linspace(0, 1, self.imgs[i].shape[1]) for i in range(self.shape[0])
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
        self.scan_slow_t = [
            torch.tensor(self.scan_slow[i], dtype=dtype, device=device)
            for i in range(self.shape[0])
        ]
        self.imgs_warped = Dataset3d.from_shape(self.shape)
        canvas_shape = (self.shape[1], self.shape[2])
        warped_t = torch.zeros(self.shape[0], *canvas_shape, dtype=dtype, device=device)
        for img_idx in range(self.shape[0]):
            warped, _ = self._interpolator(img_idx).warp_to_canvas(
                self.imgs_t[img_idx], canvas_shape,
                self.kde_sigma, self.pad_value[img_idx])
            warped_t[img_idx] = warped
            self.imgs_warped.array[img_idx] = warped.cpu().numpy()
        self._initial_knots = [k.clone() for k in self.knots]
        self.calculate_error(0, _warped_t=warped_t)
        self._show_after_step("initial", show_merged, show_images, overlay_knots, kwargs)
        if show_knot_plot:
            self.plot_knots()
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
        overlay_knots: bool = True,
        show_knot_plot: bool = False,
        verbose: bool = True,
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
        overlay_knots : bool, default True
            Overlay knot positions on top of the merged/warped image plots
            (cheap, useful diagnostic).
        show_knot_plot : bool, default False
            Render the standalone 2-panel knot trajectory + per-row delta
            chart via ``dc.plot_knots()`` after this step.
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
        >>> drift = DriftCorrection(
        ...     im0, im1, scan_direction_degrees=[0, 90])
        >>> drift.preprocess().align_affine(step=0.02, num_tests=11)

        Single-sided alignment (4D-STEM VDF against a fixed HAADF reference):

        >>> drift = DriftCorrection(
        ...     haadf_ref, vdf, scan_direction_degrees=[0, 0])
        >>> drift.preprocess().align_affine(fixed_indices=[0])
        """
        if self.shape[0] < 2:
            raise ValueError(
                f"align_affine requires at least 2 images (got {self.shape[0]}). "
                f"Provide image pairs with different scan directions."
            )
        if num_tests % 2 == 0:
            raise ValueError(
                f"num_tests must be odd (got {num_tests}). Try {num_tests + 1}."
            )
        # Reference-mode auto-anchors the reference image (index 0) so the
        # user doesn't repeat what they declared via the constructor reference mode.
        if fixed_indices is None and self._reference_mode:
            fixed_indices = [0]
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
            drift_t = torch.as_tensor(
                drift_vec, dtype=self.knots[0].dtype, device=self.knots[0].device)
            for img_idx in range(self.shape[0]):
                if img_idx in fixed_set:
                    continue
                self._interpolator(img_idx).apply_affine_shift(drift_t)

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

        self._show_after_step("affine", show_merged, show_images, overlay_knots, kwargs)
        if show_knot_plot:
            self.plot_knots()
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
            row_base, col_base, scanline_offset = (
                self._interpolator(img_idx).affine_candidate_base())
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
        On MPS (Apple unified memory) the same constraint applies - the candidate
        batch shares the system RAM budget, so we size it from ``torch.mps`` memory
        info exactly like CUDA. Skipping this (the old ``!= "cuda"`` early return)
        let MPS try all candidates at once and OOM a 24 GB Mac. On CPU we process all
        candidates at once - no separate device pool to overflow.
        """
        device = torch.device(device)
        bytes_per_element = torch.finfo(dtype).bits // 8
        per_candidate_bytes = canvas_shape[0] * canvas_shape[1] * bytes_per_element * 32
        if device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(device)
        elif device.type == "mps":
            # recommended_max is Metal's working-set ceiling; subtract what's already
            # live to get the headroom this batch can use.
            free_bytes = torch.mps.recommended_max_memory() - torch.mps.current_allocated_memory()
        else:
            return num_candidates
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
        imgs_t_override: list[torch.Tensor] | None = None,
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
        imgs_t = imgs_t_override if imgs_t_override is not None else self.imgs_t

        def _warp_all(warped_t, weights_t):
            """Warp all images onto the canvas using current knots."""
            for img_idx in range(num_images):
                # knots_batch is always 4D ``(N, 2, H, K)``; None means "use stored knots".
                knots_img = (knots_batch[img_idx].detach()
                             if knots_batch is not None else None)
                warped, weights = self._interpolator(img_idx, knots_img).warp_to_canvas(
                    imgs_t[img_idx], canvas_shape,
                    self.kde_sigma, self.pad_value[img_idx])
                warped_t[img_idx] = warped
                weights_t[img_idx] = weights

        warped_t = torch.zeros(num_images, *canvas_shape, dtype=dtype, device=device)
        weights_t = torch.zeros_like(warped_t)
        _warp_all(warped_t, weights_t)
        if not solve_translation:
            self.imgs_warped.array[:] = warped_t.cpu().numpy()
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
            # knots_batch is always 4D ``(N, 2, H, K)``; broadcast over (H, K).
            knots_batch[:, 0] += shifts_t[:, 0, None, None]
            knots_batch[:, 1] += shifts_t[:, 1, None, None]
        else:
            for img_idx in range(num_images):
                self.knots[img_idx][0] += shifts_t[img_idx, 0]
                self.knots[img_idx][1] += shifts_t[img_idx, 1]
        # Re-warp with corrected knots
        _warp_all(warped_t, weights_t)
        if knots_batch is None:
            self.imgs_warped.array[:] = warped_t.cpu().numpy()
        return warped_t

    def _setup_loss_kernel(self, K: int, canvas_shape: tuple[int, int]):
        """Pick the K-aware compiled loss kernel and precompute its constants.

        Returns ``(loss_fn, loss_args)`` ready to be passed to
        ``_optimize._optimize_knots_adam`` / ``_optimize_knots_lbfgs``.

        K=1 path: scan_fast walk offsets per image (``row_scan_offsets`` /
        ``col_scan_offsets``).  K>=2 path: per-output-column segment indices
        + local fractions for the linear knot interpolation.  Constants are
        lifted out of the inner Adam / LBFGS loop so the compiled kernel
        sees them as static.
        """
        device, dtype = self._device, self._dtype
        num_images = self.shape[0]
        row_scale = 2.0 / (canvas_shape[0] - 1)
        col_scale = 2.0 / (canvas_shape[1] - 1)
        if K > 1:
            # Identical to _transform_coordinates_multi_knot's geometry so
            # apply_correction downstream inverts it cleanly.
            num_cols = self.imgs[0].shape[1]
            t = torch.linspace(0, 1, num_cols, dtype=dtype, device=device) * (K - 1)
            seg_idx = torch.clamp(t.long(), max=K - 2)
            seg_frac = t - seg_idx.to(dtype)
            return (_optimize._compiled_loss_fn_multi,
                    (seg_idx, seg_frac, row_scale, col_scale))
        # K=1: same scan-position vector projects onto row/col via scan_fast.
        u_t = [
            torch.as_tensor(self.u_per_image[i], dtype=dtype, device=device)
            for i in range(num_images)
        ]
        row_scan_offsets = torch.stack([
            u_t[i] * (self.scan_fast[i][0] * (self.imgs[i].shape[0] - 1))
            for i in range(num_images)
        ])
        col_scan_offsets = torch.stack([
            u_t[i] * (self.scan_fast[i][1] * (self.imgs[i].shape[1] - 1))
            for i in range(num_images)
        ])
        return (_optimize._compiled_loss_fn_single,
                (row_scan_offsets, col_scan_offsets, row_scale, col_scale))

    def align_nonrigid(
        self,
        optimizer_name: str = "adam",
        num_iterations: int = 16,
        regularization_sigma_px: float = 8.0,
        regularization_update_step_size: float | None = 0.8,
        regularization_poly_order: int = 1,
        max_image_shift: float | None = 32.0,
        adam_steps: int = 30,
        lr: float | None = None,
        lbfgs_max_iter: int = 20,
        regularization_max_image_shift_px: float | None = None,
        fixed_indices: list[int] | None = None,
        loss: str = "auto",
        loss_pre_smooth: float = 1.0,
        early_stop_patience: int = 3,
        early_stop_rtol: float = 1e-4,
        min_iterations: int = 4,
        show_merged: bool = True,
        show_images: bool = False,
        overlay_knots: bool = True,
        show_knot_plot: bool = False,
        **kwargs,
    ):
        """Non-rigid drift correction via batched GPU optimization.

        Optimizes per-scanline knot positions to minimize misalignment
        between image collection scans. Runs entirely on GPU using
        PyTorch (Adam or LBFGS optimizer). Single-knot mode only.

        Parameters
        ----------
        optimizer_name : str, default "adam"
            ``"adam"`` (first-order momentum) or ``"lbfgs"`` (quasi-Newton
            with strong-Wolfe line search). Adam is the fastest default for
            ≤1024 px images. LBFGS auto-scales the step size and is preferred
            for ≥2048 px or when the drift magnitude is unknown. Don't normalize
            inputs to [0, 1] when using LBFGS — strong-Wolfe needs absolute
            gradient magnitude and silently returns step=0 on unit-variance images.
        num_iterations : int, default 16
            Outer iterations for alternating reference build + knot update.
        regularization_sigma_px : float, default 8.0
            Gaussian smoothing sigma for knot regularization. 4-12 typical
            for STEM data; smaller = finer per-row correction.
        regularization_update_step_size : float, default 0.8
            Step size for knot updates (0-1, lower = more conservative).
        regularization_poly_order : int, default 1
            Polynomial order for trend removal in knot regularization.
        max_image_shift : float, default 32.0
            Maximum shift for translation alignment between iterations.
            Adam's auto-``lr`` derives from this — set close to your expected
            drift bound, otherwise Adam silently under-converges.
        adam_steps : int, default 30
            Adam steps per outer iteration. Used only when ``optimizer_name="adam"``.
        lr : float or None, default None
            Adam learning rate. When ``None``, auto-derived as
            ``max_image_shift / (num_iterations * adam_steps * 4)``.
            Adam's ``m/sqrt(v)`` update self-normalizes the gradient, so each
            step moves a knot by ~``lr`` pixels regardless of intensity scale.
            Override only when you know the actual drift magnitude.
        lbfgs_max_iter : int, default 20
            Max LBFGS iterations per outer step. Used only when
            ``optimizer_name="lbfgs"``.
        fixed_indices : list[int] or None, default None
            Indices of images whose knots are frozen (their mean becomes the
            target for every moving image). Use ``[0]`` for single-sided
            alignment with a HAADF reference. Auto-set to ``[0]`` in
            reference mode (when the constructor was called with a 2-D ref +
            ≥3-D drifted dataset).
        loss : str, default "auto"
            ``"auto"`` resolves to ``"gradient_mse"``. Other choices:
            ``"mse"`` (raw-intensity MSE — fine when both images have the same
            detector / contrast) or ``"gradient_mse"`` (MSE on Sobel gradient
            magnitudes after Gaussian pre-smooth + per-image z-score, robust
            to intensity / contrast differences across detectors).
        loss_pre_smooth : float, default 1.0
            Gaussian sigma applied before Sobel when ``loss="gradient_mse"``.
            Set to 0 to disable. Ignored for ``"mse"``.
        early_stop_patience : int, default 3
            Stop when ``patience`` consecutive iterations show no improvement.
            Set to ``num_iterations`` to disable.
        early_stop_rtol : float, default 1e-4
            Minimum relative improvement to count as progress.
        min_iterations : int, default 4
            Floor before early stopping can trigger.
        show_merged, show_images : bool
            Display knobs forwarded to the plot helpers.

        Returns
        -------
        Self
            For method chaining.

        Examples
        --------
        >>> dc = DriftCorrection(im0, im1, scan_direction_degrees=[0, 90])
        >>> dc.preprocess().align_affine().align_nonrigid()

        Cross-detector alignment (HAADF + VDF):

        >>> dc.align_nonrigid(loss="gradient_mse", regularization_sigma_px=8.0)

        Notes
        -----
        ``self.imgs_warped`` is left STALE after the loop and refreshed lazily
        on first access via plot methods or ``calculate_error()``. Code reading
        ``self.imgs_warped.array`` directly should call
        ``self._ensure_warped_images()`` first, or use ``generate_corrected()``
        which builds its own warps from ``self.knots``.
        """
        if not hasattr(self, "knots"):
            raise RuntimeError(
                "No knots found. Call .preprocess() before running alignment.")
        if loss == "auto":
            loss = "gradient_mse"
        _valid_losses = ("mse", "gradient_mse")
        if loss not in _valid_losses:
            raise ValueError(
                f"loss must be one of {_valid_losses!r} or 'auto', got {loss!r}")
        if optimizer_name == "lbfgs" and self._normalized:
            import warnings
            warnings.warn(
                "normalize=True + LBFGS can cause silent convergence failure. "
                "Wolfe line search may return step=0 on unit-variance images. "
                "Consider using optimizer_name='adam' or normalize=False.",
                UserWarning, stacklevel=2)
        # Reference-mode auto-anchors the reference image (index 0).
        if fixed_indices is None and self._reference_mode:
            fixed_indices = [0]
        fixed_set = frozenset(fixed_indices) if fixed_indices is not None else frozenset()
        moving_indices = [i for i in range(self.shape[0]) if i not in fixed_set]
        if fixed_set and not moving_indices:
            raise ValueError(
                "All images are fixed — nothing to optimize. "
                "fixed_indices must leave at least one moving image."
            )
        device = self._device
        dtype = self._dtype
        num_images = self.shape[0]
        canvas_shape = (self.shape[1], self.shape[2])
        K_per_image = {self.knots[i].shape[2] for i in range(num_images)}
        if len(K_per_image) > 1:
            raise ValueError(
                f"All images must use the same number of knots, got {K_per_image}")
        K = K_per_image.pop()
        # Knots are 4D throughout the optimizer — K=1 just keeps the trailing 1
        # so downstream code (regularizer, warp, sync) doesn't branch on shape.
        knots_batch = torch.stack(
            [self.knots[i] for i in range(num_images)]
        ).detach().requires_grad_(True)
        num_rows_knot = knots_batch.shape[2]
        target_batch = torch.stack(self.imgs_t)
        loss_fn, loss_args = self._setup_loss_kernel(K, canvas_shape)
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
        # For gradient_mse, warp the edge-filtered images instead of the
        # raw ones. Knots are spatial transforms independent of image
        # content, so optimizing in gradient space yields the same drift
        # field while being robust to intensity/contrast differences.
        # imgs_t_override threads Sobel images through _warp_and_translate_torch
        # without mutating self.imgs_t (which would silently corrupt
        # apply_correction, visualization, and error metrics afterwards).
        if loss == "gradient_mse":
            sobel_batch = _optimize.sobel_gradient_magnitude(
                target_batch, loss_pre_smooth, device, dtype)
            imgs_t_override = [sobel_batch[i] for i in range(num_images)]
            target_batch = sobel_batch
        else:
            imgs_t_override = None
        warped_t = self._warp_and_translate_torch(
            max_image_shift, upsample_factor=8, knots_batch=knots_batch,
            fixed_indices=fixed_set, imgs_t_override=imgs_t_override)
        # Build a boolean mask on device to zero fixed gradients efficiently.
        # knots_batch is always 4D ``(N, 2, R, K)``; broadcast over (2, R, K).
        if fixed_set:
            grad_mask = torch.ones(num_images, 1, 1, 1, dtype=dtype, device=device)
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
                _optimize._optimize_knots_adam(
                    ref_batch, target_batch, knots_batch, loss_fn, loss_args,
                    optimizer, adam_steps,
                    grad_mask=grad_mask if fixed_set else None)
            else:
                _optimize._optimize_knots_lbfgs(
                    ref_batch, target_batch, knots_batch, loss_fn, loss_args,
                    optimizer,
                    grad_mask=grad_mask if fixed_set else None)
            _optimize._regularize_knots(
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
                fixed_indices=fixed_set, imgs_t_override=imgs_t_override)
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
            self.knots[img_idx][...] = knots_final[img_idx]
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

        self._show_after_step(
            "non-rigid", show_merged, show_images, overlay_knots, kwargs)
        if show_knot_plot:
            self.plot_knots()
        return self

    def generate_corrected(
        self,
        upsample_factor: int = 2,
        output_original_shape: bool = True,
        strip_padding: bool = False,
        mask_output: bool = False,
        mask_edge_blend: float = 8.0,
        fourier_filter: bool = False,
        filter_midpoint: float = 0.5,
        kde_sigma: float = 0.5,
        weight_thresh: float = 0.1,
        show_merged: bool = True,
        *,
        mode: str = "bilinear",
        chunk_size: int | None = None,
        merge: bool = True,
        verbose: bool = False,
        output_0: np.ndarray | None = None,
        output_1: np.ndarray | None = None,
        output_dtype: torch.dtype | np.dtype | str | None = None,
        output_device: str | torch.device | None = None,
        **kwargs,
    ):
        """Produce the canonical drift-corrected output for this instance.

        Same call works for all three modes; the return type follows the
        factory used:

        ===================================== ===================================
        Factory + inputs                      Return type
        ===================================== ===================================
        Constructor (2-D + 2-D)               :class:`Dataset2d`
        :meth:`from_reference` (2-D + 3-D)    :class:`Dataset3d` (corrected EDS/EELS)
        :meth:`from_reference` (2-D + 4-D)    :class:`Dataset4d` (corrected 4D-STEM)
        :meth:`from_4dstem` (4-D + 4-D)  :class:`CorrectionResult`
        ===================================== ===================================

        For 4D-STEM collection mode both datasets are corrected with their own
        learned knots into a shared scan-derived coordinate system, the
        corrected 4D-STEM dataset 1 is oriented into dataset 0's display
        frame, and the two are merged at the diffraction-pattern level.
        Image-mode parameters (``upsample_factor``, ``fourier_filter``,
        etc.) are ignored; dataset parameters (``mode``, ``chunk_size``,
        ``merge``, ``verbose``, ``output_0``, ``output_1``) take effect.

        For image collection mode, the entire pipeline (warping, Fourier
        filtering, masking, cropping) runs on GPU via PyTorch, transferring
        to CPU only for the final ``Dataset2d`` output and the
        edge-blend mask step (now torch).

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
        mask_output : bool, default False
            If true, blend the corrected image edge into ``pad_value_mean``
            with a cosine ramp of width ``mask_edge_blend`` px. Useful for
            FFT analysis or visualization where a hard data/pad step would
            cause spectral ringing or look bad. Skip for save-to-npy →
            downstream-pipeline workflows where the caller crops anyway.
        mask_edge_blend : float, default 8.0
            Width in pixels of the edge blend ramp (only used when
            ``mask_output=True``).
        fourier_filter : bool, default False
            Whether to apply Fourier-based directional filtering to merge
            corrected images. Only useful when blending ≥3 scan angles.
            For scan collection (0°, 90°) HAADF — the typical case — keep this off.
        filter_midpoint : float, default 0.5
            Midpoint for the sigmoid-based Fourier weighting filter, determining transition smoothness.
            Setting this to a low value close to 0 will include more signal but also more slow scan artifacts.
            If using 2 images at 0 and 90 degrees scan angles, any value >0.75 will be unstable.
            Only use larger values (close to 1.0) if multiple images covering many scan angles are used.
        kde_sigma : float, default 0.5
            Gaussian smoothing sigma applied during scatter warping for the
            corrected output. Independent of :meth:`preprocess`'s ``kde_sigma``;
            tune separately if needed.
        weight_thresh: float, default 0.1
            This value sets the threshold for masking the outputs.
            For very large jitter artifacts this value can be lowered.
        show_merged : bool, default True
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
        # Reference mode: warp the stored drifted dataset against the
        # anchored reference; image-mode merge / Fourier params are unused.
        if self._reference_mode:
            drifted = self._datasets[1]
            corrected = self.apply_correction(
                drifted, image_index=1, mode=mode, chunk_size=chunk_size,
                verbose=verbose,
            )
            if isinstance(corrected, torch.Tensor):
                corrected = corrected.cpu().numpy()
            corrected = corrected.astype(np.float32, copy=False)
            ndim = corrected.ndim
            if ndim == 2:
                return Dataset2d.from_array(corrected)
            if ndim == 3:
                return Dataset3d.from_array(corrected)
            from quantem.core.datastructures.dataset4d import Dataset4d
            return Dataset4d.from_array(corrected)
        # 4D-STEM collection has its own explicit API because it returns both
        # corrected inputs and the diffraction-pattern-level merge.
        if self._datasets is not None:
            raise RuntimeError(
                "4D-STEM collection correction uses the explicit "
                "generate_corrected_4dstem() API. Use "
                "DriftCorrection.from_4dstem(data_0, data_1, ...)"
                ".preprocess().align_affine().generate_corrected_4dstem()."
            )
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
            warped, weights = self._interpolator(img_idx).warp_to_canvas(
                self.imgs_t[img_idx], canvas_up,
                kde_sigma * upsample_factor, self.pad_value[img_idx],
                upsample_factor=upsample_factor,
            )
            stack_corr[img_idx] = warped
            weight_corr[img_idx] = weights

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
            # GPU edge-blend mask. Two-stage cosine ramp around the data area:
            #   1. Build the data mask (where every input image has weight
            #      above threshold). Force the outer border False so the
            #      distance transform knows where "outside" is.
            #   2. distance-transform once to get pixels within `blend` of the
            #      boundary — this is the "ramp band".
            #   3. distance-transform that ramp band to get a smooth distance
            #      from each band pixel to the deep interior.
            #   4. Apply cos² ramp so mask = 1 deep inside, 0 at the boundary.
            # Both distance transforms use the torch GPU implementation
            # (bit-exact to scipy on single-region masks).
            blend_px = float(mask_edge_blend)
            data_mask = (weight_corr >= (weight_thresh / upsample_factor**2)).all(dim=0)
            data_mask[:, 0] = False
            data_mask[:, -1] = False
            data_mask[0, :] = False
            data_mask[-1, :] = False

            distance_to_boundary = _distance_transform_edt_torch(data_mask)
            ramp_band = distance_to_boundary <= blend_px
            distance_in_band = _distance_transform_edt_torch(ramp_band)
            ramp_position = (distance_in_band / blend_px).clamp(0.0, 1.0)
            edge_blend_mask = (torch.cos((torch.pi / 2) * ramp_position) ** 2).to(dtype)

            pad_value_mean = float(np.mean(self.pad_value))
            corrected_image = torch.fft.ifft2(image_corr_fft).real
            blended_image = (
                corrected_image * edge_blend_mask
                + pad_value_mean * (1 - edge_blend_mask)
            )
            image_corr_fft = torch.fft.fft2(blended_image)

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

        if show_merged:
            show_2d(image_corr.array, **kwargs)
            plt.show()
        return image_corr

    def apply_correction(
        self,
        data: torch.Tensor | np.ndarray | None = None,
        image_index: int = -1,
        *,
        mode: str = "bilinear",
        chunk_size: int | None = None,
        output_dtype: torch.dtype | np.dtype | str | None = None,
        output_device: str | torch.device | None = None,
        output: np.ndarray | None = None,
        verbose: bool = False,
    ) -> torch.Tensor | np.ndarray:
        """Apply the learned drift correction to data.

        Works on 2-D images, image batches, 3-D EDX/EELS spectral datasets,
        and 4-D STEM datasets. Scan-axis convention depends on the
        construction mode:

        - 2-D image-collection mode: scan axes are the LAST two of input.
          Shapes ``(H, W)`` (single image) or ``(N, H, W)`` (batch).
        - 4D-STEM / reference mode (built with a ≥3-D drifted dataset):
          scan axes are the FIRST two of input. Shapes ``(H, W)`` (VDF),
          ``(H, W, C)`` (spectral), or ``(H, W, det_h, det_w)`` (4D-STEM).

        For ≥3-D dataset inputs (4D-STEM mode for 3-D, any 4-D), the warp is
        chunked along the trailing detector / spectral axes to fit GPU
        memory. Pre-allocated ``output=`` enables zero-copy writes
        (e.g. into ``np.memmap``) for datasets too large to hold in RAM.

        Parameters
        ----------
        data : ndarray or torch.Tensor, optional
            Data to correct. If ``None``, uses the stored alignment image
            (or stored 4D-STEM dataset if the constructor holds one).
        image_index : int, default -1
            Which image's knot trajectory to use. ``-1`` selects the last.
        mode : str, default "bicubic"
            Interpolation kernel: ``"bicubic"`` or ``"bilinear"``.
        chunk_size, output_dtype, output_device, output, verbose
            Dataset-mode (≥3-D in 4D-STEM mode, or 4-D) only; ignored for
            image / batch inputs. See dataset-streaming docs.

        Returns
        -------
        torch.Tensor or np.ndarray
            Corrected data with the same shape and axis layout as input.

        Examples
        --------
        >>> dc = DriftCorrection(haadf_ref, vdf, scan_direction_degrees=[0, 0])
        >>> dc.preprocess(normalize=True).align_affine(fixed_indices=[0])
        >>> corrected_vdf = dc.apply_correction()      # uses stored vdf

        4D-STEM with pre-allocated memmap output:

        >>> dc = DriftCorrection(data_0, data_1, scan_direction_degrees=(0, 90))
        >>> dc.preprocess().align_affine()
        >>> out = np.memmap('corrected.dat', dtype='float32', mode='w+', shape=data_1.shape)
        >>> dc.apply_correction(output=out)            # writes to memmap, returns it
        """
        if not hasattr(self, "knots") or not hasattr(self, "_initial_knots"):
            raise RuntimeError(
                "apply_correction() requires preprocess() and align_affine() "
                "first. Run dc.preprocess().align_affine() (and optionally "
                ".align_nonrigid()) before apply_correction().")
        _valid_modes = {"bilinear", "bicubic"}
        if mode not in _valid_modes:
            raise ValueError(f"mode must be one of {_valid_modes}, got {mode!r}")
        idx = image_index % len(self.knots)
        self._knot_delta_canvas(idx)  # validates preprocess+align ran

        # Dispatch: route to dataset path when the input lays out as
        # (scan_h, scan_w, *channels). Detected by matching the leading two
        # axes against the raw scan dims learned in preprocess(). Pure 4-D
        # input always goes to dataset path. Ambiguous shapes (square scans
        # with batch == scan_h) fall back on the factory mode.
        # Reference-mode is detected separately: _datasets[0] is None (only
        # the drifted side is stored), so the dataset path uses _datasets[1] directly.
        is_4dstem_mode = self._datasets is not None and not self._reference_mode
        scan_h = self.imgs[idx].shape[0]
        scan_w = self.imgs[idx].shape[1]
        if data is None:
            if self._reference_mode:
                data = self._datasets[1]
            elif is_4dstem_mode:
                return self._apply_correction_to_dataset(
                    None, image_index, mode, chunk_size,
                    output_dtype, output_device, output, verbose)
        if data is not None:
            ndim = data.ndim
            shape = tuple(data.shape)
            if ndim >= 4:
                return self._apply_correction_to_dataset(
                    data, image_index, mode, chunk_size,
                    output_dtype, output_device, output, verbose)
            if ndim == 3:
                cube_layout = (shape[0] == scan_h and shape[1] == scan_w)
                batch_layout = (shape[-2] == scan_h and shape[-1] == scan_w
                                and not (shape[0] == scan_h and shape[1] == scan_w))
                if cube_layout and not batch_layout:
                    return self._apply_correction_to_dataset(
                        data, image_index, mode, chunk_size,
                        output_dtype, output_device, output, verbose)
                if cube_layout and batch_layout and is_4dstem_mode:
                    return self._apply_correction_to_dataset(
                        data, image_index, mode, chunk_size,
                        output_dtype, output_device, output, verbose)

        # Image / image-batch path: scan axes are trailing.
        if data is None:
            data_t = self.imgs_t[idx]
        elif isinstance(data, np.ndarray):
            data_t = torch.tensor(data, dtype=self._dtype, device=self._device)
        else:
            data_t = data.to(device=self._device, dtype=self._dtype)

        img_h = data_t.shape[-2]
        img_w = data_t.shape[-1]
        knot_h = self.knots[idx].shape[1]
        if img_h != knot_h:
            raise ValueError(
                f"Input scan-row axis ({img_h}) does not match knot grid "
                f"height ({knot_h}). For 4D-STEM mode the leading axis is the "
                f"scan row; for image collection mode the trailing-2 axes are scan.")

        drift = self.drift_field(idx)
        return backward_warp(data_t, drift=drift, mode=mode)

    # 4D-STEM dataset path: implementations live in drift_4dstem.py so the
    # orchestrator stays focused on the image pipeline.

    def generate_corrected_4dstem(
        self,
        *,
        mode: str = "bilinear",
        chunk_size: int | None = None,
        merge: bool = True,
        verbose: bool = False,
        output_0: np.ndarray | None = None,
        output_1: np.ndarray | None = None,
        output_dtype: torch.dtype | np.dtype | str | None = None,
        output_device: str | torch.device | None = None,
    ) -> CorrectionResult:
        """Correct and optionally merge a 0/90 4D-STEM collection dataset pair.

        This is the explicit first-class API for 4D-STEM collection. It requires
        a ``DriftCorrection`` constructed from two 4D-STEM datasets, for example
        ``DriftCorrection.from_4dstem(data_0, data_1, ...)``. Both
        scans are corrected with their own learned knots into the shared
        scan-derived coordinate system; neither scan is treated as ground
        truth.

        New code should use this method when the intention is
        diffraction-pattern-level 0/90 correction.
        """
        if not self._is_4dstem_collection:
            raise RuntimeError(
                "generate_corrected_4dstem() requires 4D-STEM collection "
                "construction: DriftCorrection.from_4dstem(data_0, data_1, ...). "
                "For reference-mode EDS/EELS/4D-STEM, use generate_corrected()."
            )
        return self._generate_corrected_4dstem_collection(
            mode=mode, chunk_size=chunk_size, merge=merge,
            verbose=verbose, output_0=output_0, output_1=output_1,
            output_dtype=output_dtype, output_device=output_device,
        )

    def correct_virtual_images(
        self,
        image_0: np.ndarray,
        image_1: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Correct scalar virtual images like matching 4D-STEM channels.

        Use this for VDF/BF/DF diagnostics from 4D-STEM collection data. Each
        scalar image is corrected with the same operator used for diffraction
        pixels; image 1 is oriented into image 0's display frame before the
        average. ``result["corrected_image"]`` should match the same detector
        integration from ``generate_corrected_4dstem()`` output, up to output
        quantization.
        """
        return _4dstem.correct_virtual_images(
            self,
            image_0,
            image_1,
        )

    @staticmethod
    def integrate_virtual_detector(
        ds_4d: np.ndarray | torch.Tensor,
        detector_mask: np.ndarray | torch.Tensor | None = None,
        *,
        reduce: str = "mean",
        chunk_rows: int | None = None,
    ) -> np.ndarray:
        """Integrate a VDF/BF/DF-style virtual image from a 4D-STEM dataset."""
        return _4dstem.integrate_virtual_detector(
            ds_4d,
            detector_mask=detector_mask,
            reduce=reduce,
            chunk_rows=chunk_rows,
        )

    @staticmethod
    def compute_vdf(
        ds_4d: np.ndarray,
        chunk_rows: int | None = None,
    ) -> np.ndarray:
        """Virtual dark-field from a 4D-STEM dataset (delegates to drift_4dstem)."""
        return _4dstem.compute_vdf(ds_4d, chunk_rows)

    def _apply_correction_to_dataset(self, *args, **kwargs):
        """Delegates the ≥3-D dataset path to :mod:`drift_4dstem`."""
        return _4dstem.apply_correction_to_dataset(self, *args, **kwargs)

    def _generate_corrected_4dstem_collection(self, **kwargs) -> CorrectionResult:
        """Delegates the 4D-STEM collection merge to :mod:`drift_4dstem`."""
        return _4dstem.generate_corrected_4dstem_collection(self, **kwargs)

    # -- serialization -------------------------------------------------------

    def save(self, path, mode="w", store="auto", skip=(), compression_level=4):
        """Save alignment state (4D-STEM data excluded — too large)."""
        if isinstance(skip, (str, type)):
            skip = [skip]
        skip = list(skip) + ["_datasets"]
        super().save(
            path, mode=mode, store=store, skip=skip,
            compression_level=compression_level,
        )

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

    @property
    def drift_rate(self) -> tuple[float, float]:
        """Per-scanline drift rate ``(row, col)`` in pixels-per-line for the
        last image relative to the first.  Only meaningful after
        :meth:`align_affine`. Uses the affine-only knot snapshot when
        available so post-nonrigid wobble doesn't perturb the linear slope.
        """
        if not hasattr(self, "_initial_knots"):
            raise RuntimeError("Call preprocess() then align_affine() first.")
        idx = len(self.knots) - 1
        # Prefer the affine-only knot snapshot so post-nonrigid wobble doesn't
        # perturb the linear slope; fall back to the live knot delta otherwise.
        if hasattr(self, "_knots_after_affine"):
            delta = self._knots_after_affine[idx] - self._initial_knots[idx]
        else:
            delta = self._knot_delta_canvas(idx)
        n = delta.shape[1]
        row_rate = float((delta[0, -1, 0] - delta[0, 0, 0]) / max(n - 1, 1))
        col_rate = float((delta[1, -1, 0] - delta[1, 0, 0]) / max(n - 1, 1))
        return (row_rate, col_rate)

    # -- visualization methods bound directly from drift_visualization so hover
    #    shows the real signature + docstring (no `**kw` indirection).
    print_drift_stats = drift_visualization.print_drift_stats
    plot_correction_summary = drift_visualization.plot_correction_summary
    plot_correction_comparison = drift_visualization.plot_correction_comparison
    plot_radial_power = drift_visualization.plot_radial_power
    plot_warped_images = drift_visualization.plot_warped_images
    plot_convergence = drift_visualization.plot_convergence
    plot_merged_images = drift_visualization.plot_merged_images
    interactive_drift = drift_visualization.interactive_drift
    plot_knots = drift_visualization.plot_knots
    plot_probe_positions = drift_visualization.plot_probe_positions
    plot_diffraction = drift_visualization.plot_4dstem_correction
    view_corrected_dp = _4dstem.view_corrected_dp
    view_corrected_vdfs = _4dstem.view_corrected_vdfs


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
