"""Virtual-detector and paired-dataset products for 4D-STEM correction."""
from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm

from quantem.imaging.drift.apply import (
    apply_correction_to_dataset,
    crop_slices,
    padding_offset,
)
from quantem.imaging.drift.core import knots as drift_knots


@dataclass
class CorrectionResult:
    """Container returned by 0/90 4D-STEM collection correction.

    This result represents the scan-derived corrected coordinate system:
    both input 4D-STEM datasets are treated as drifted scans and corrected toward a
    shared consensus frame before optional diffraction-pattern-level merge.

    Attributes
    ----------
    corrected_4dstem_0, corrected_4dstem_1 : np.ndarray | torch.Tensor
        Per-side drift-corrected 4D-STEM datasets, scan-axis-leading layout.
        Dataset 1 has already been oriented into dataset 0's display frame.
    corrected_4dstem : np.ndarray | torch.Tensor | None
        Diffraction-pattern-level average of ``corrected_4dstem_0`` and the
        oriented ``corrected_4dstem_1``. ``None`` when ``merge=False``.
    scalar_corrected_vdf : np.ndarray | None
        Scan-derived corrected VDF computed by correcting the raw alignment
        VDFs with the same operator used for 4D-STEM channels. This is not an
        external ground truth; it is the scalar virtual-image result implied
        by the learned scan drift fields.
    """
    corrected_4dstem_0: np.ndarray | torch.Tensor
    corrected_4dstem_1: np.ndarray | torch.Tensor
    corrected_4dstem: np.ndarray | torch.Tensor | None = None
    scalar_corrected_vdf: np.ndarray | None = None


def _rot90_to_image0_frame(
    dc, image_index: int = 1, reference_index: int = 0
) -> int:
    """Return the scan-axis rot90 count to show ``image_index`` like the reference.

    Two 0/90 scans differ by a 90-degree scan-axis rotation, so displaying one in
    the other's frame is a ``rot90`` whose count is the signed angle difference in
    quarter turns. ``reference_index`` defaults to image 0 (the consensus frame);
    a caller comparing against a different reference passes its index.
    """
    delta = float(
        dc.scan_direction_degrees[image_index] - dc.scan_direction_degrees[reference_index]
    )
    return (-int(round(delta / 90.0))) % 4


def integrate_virtual_detector(
    dataset,
    detector_mask: np.ndarray | torch.Tensor | None = None,
    *,
    reduce: str = "mean",
) -> np.ndarray:
    """Integrate a virtual image from a scan-axis-leading dataset.

    Sums (or averages) the selected trailing detector/channel pixels for every
    scan position, producing a 2-D scan image. With ``detector_mask=None`` it
    integrates the *whole* detector (a full/total virtual image, bright-field
    dominated for thin samples); pass a disk mask for bright field or an annulus
    for dark field. This is the single canonical virtual-image reduction for the
    drift package; every other virtual-image helper delegates here.

    QuantEM's detector backend selects the resident NumPy, Torch, CuPy, CUDA,
    or MPS reduction path. Three-dimensional spectrum images are treated as a
    one-row detector so the same backend also integrates their channel axis.

    Parameters
    ----------
    dataset : ndarray or torch.Tensor, shape ``(H, W, ...channels)``
        3-D/4-D dataset with scan axes first. numpy may be a ``np.memmap``.
    detector_mask : ndarray or torch.Tensor, optional
        Boolean mask over the trailing detector / channel axes. ``None``
        selects every channel.
    reduce : {"mean", "sum"}
        Average or sum the selected detector pixels.
    Returns
    -------
    numpy.ndarray, shape ``(H, W)``, dtype float32

    Examples
    --------
    >>> adf = integrate_virtual_detector(data, detector_mask=annulus)
    """
    from quantem.gpu.detector import masked_sum

    if reduce not in {"mean", "sum"}:
        raise ValueError(f"reduce must be 'mean' or 'sum', got {reduce!r}")
    scan_shape = tuple(int(value) for value in dataset.shape[:2])
    detector_shape = tuple(int(value) for value in dataset.shape[2:])
    if len(detector_shape) == 1:
        detector_shape = (1, detector_shape[0])
        dataset = dataset.reshape(-1, *detector_shape)
    elif len(detector_shape) != 2:
        raise ValueError(
            "integrate_virtual_detector expects (row, col, channel) or "
            f"(row, col, detector_row, detector_col), got {tuple(dataset.shape)}"
        )
    mask = (
        np.ones(detector_shape, dtype=bool)
        if detector_mask is None
        else to_numpy(detector_mask, dtype=bool).reshape(detector_shape)
    )
    num_selected = int(mask.sum())
    if num_selected == 0:
        raise ValueError("detector_mask selects zero detector pixels")
    image = masked_sum(dataset, mask).reshape(scan_shape)
    return image / num_selected if reduce == "mean" else image


def drift_field(self, idx: int) -> torch.Tensor:
    """Return fitted raw-frame drift for one scan in ``(row, col)`` order.

    A one-knot model returns ``(2, scan_rows)`` because its displacement is
    constant along each scanline. Multi-knot models return
    ``(2, scan_rows, scan_columns)``.

    Parameters
    ----------
    idx : int
        Scan index in the correction pair.

    Returns
    -------
    torch.Tensor
        Row and column displacement in raw scan coordinates.

    Examples
    --------
    >>> field_90 = drift.drift_field(1)
    """
    if not hasattr(self, "_initial_knots"):
        raise RuntimeError(
            "drift_field() requires preprocess() and correct_affine() first. "
            "Run dc.preprocess().correct_affine() (and optionally "
            ".correct_nonrigid()) before drift_field()."
        )
    return drift_knots.interpolator(self, idx).drift_raw(
        self._initial_knots[idx]
    )


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

    Positions retain the raw diffraction-pattern indexing and use ``(row,
    col)`` coordinates in the shared corrected frame. This lets iterative
    ptychography consume corrected positions without interpolating detector
    data.

    Parameters
    ----------
    image_index : int, default 0
        Scan image or 4D-STEM acquisition to describe.
    corrected : bool, default True
        Return fitted positions instead of the nominal scan grid.
    strip_padding : bool, default True
        Express positions in the original image-0 frame instead of the padded
        solver canvas.
    plot : bool, default True
        Draw the nominal and corrected positions for inspection.
    stride : int, default 16
        Subsampling used only by the plot.

    Returns
    -------
    numpy.ndarray
        Position array with shape ``(scan_rows, scan_columns, 2)``.

    Examples
    --------
    >>> positions = drift.probe_positions(image_index=0, plot=False)
    """
    if not hasattr(self, "_initial_knots"):
        raise RuntimeError(
            "probe_positions() requires preprocess() first. Run "
            "dc.preprocess() before exporting nominal or corrected "
            "probe positions."
        )
    index = image_index % len(self.imgs)
    knots = (
        self.knots[index]
        if corrected
        else self._initial_knots[index]
    )
    row, column = drift_knots.interpolator(self, index, knots).to_canvas()
    positions = torch.stack([row, column], dim=-1)
    if strip_padding:
        pad_row, pad_column = padding_offset(
            (self.shape[1], self.shape[2]),
            self.imgs[0].shape[:2],
        )
        positions -= torch.tensor(
            [pad_row, pad_column],
            device=positions.device,
            dtype=positions.dtype,
        )
    result = to_numpy(positions, dtype=np.float32)
    if plot:
        self.plot_probe_positions(
            image_index=index,
            strip_padding=strip_padding,
            stride=stride,
        )
    return result


@torch.inference_mode()
def corrected_virtual_images(
    self,
    image_0,
    image_1,
) -> dict[str, np.ndarray]:
    """Correct two scalar virtual images like matching 4D-STEM channels.

    Each scalar image is treated as a one-channel dataset with scan axes first,
    corrected with the same ``grid_sample`` operator used for every diffraction
    pixel, and image 1 is oriented into image 0's display frame before the
    average. The returned ``corrected_image`` should therefore match integrating
    the same virtual detector from ``corrected_4dstem()`` output,
    up to output quantization.

    Parameters
    ----------
    image_0, image_1 : array-like
        Scalar virtual images from the two 4D-STEM acquisitions. Their scan
        shapes must match the images used to solve the correction.

    Returns
    -------
    dict[str, np.ndarray]
        The merged ``corrected_image`` and the separately corrected
        ``corrected_image_0`` and ``corrected_image_1``, all in image 0's scan
        frame.

    Examples
    --------
    >>> images = drift.corrected_virtual_images(vdf_0, vdf_90)
    >>> corrected_vdf = images["corrected_image"]
    """
    if not hasattr(self, "_initial_knots"):
        raise RuntimeError(
            "corrected_virtual_images() requires preprocess() and "
            "correct_affine() first."
        )
    if len(self.imgs) != 2:
        raise ValueError(
            "corrected_virtual_images() expects exactly two scan images"
        )
    images = [
        np.asarray(image_0, dtype=np.float32),
        np.asarray(image_1, dtype=np.float32),
    ]
    if images[0].shape != self.imgs[0].shape or images[1].shape != self.imgs[1].shape:
        raise ValueError(
            "virtual image shapes must match the raw scan images used for drift "
            f"alignment: got {images[0].shape}, {images[1].shape}; expected "
            f"{self.imgs[0].shape}, {self.imgs[1].shape}"
        )

    components = []
    for image_index, image in enumerate(images):
        image_t = torch.as_tensor(
            image[..., None],
            device=self._device,
            dtype=torch.float32,
        )
        corrected = apply_correction_to_dataset(
            self,
            image_t,
            image_index=image_index,
            mode="bilinear",
            chunk_size=1,
            output_dtype=torch.float32,
            output_device=self._device,
        )[..., 0]
        if image_index == 1:
            rot_k = _rot90_to_image0_frame(self, image_index=1)
            if rot_k:
                corrected = torch.rot90(corrected, k=rot_k, dims=(0, 1))
        components.append(corrected)

    merged = (components[0] + components[1]) * 0.5

    return {
        "corrected_image": to_numpy(merged, dtype=np.float32),
        "corrected_image_0": to_numpy(components[0], dtype=np.float32),
        "corrected_image_1": to_numpy(components[1], dtype=np.float32),
    }


def corrected_4dstem_views(correction, *, det_bin: int = 1) -> list[np.ndarray]:
    """Prepare a compact raw-to-corrected 4D-STEM comparison.

    Detector binning before correction preserves the correction field while
    avoiding work on detector detail that the interactive viewer will discard.
    The returned stages share image 0's scan frame and the solved crop.
    """
    def detector_bin(cube):
        if det_bin == 1:
            return cube
        detector_rows, detector_columns = cube.shape[-2:]
        shape = cube.shape
        reshaped = cube.reshape(
            shape[0],
            shape[1],
            detector_rows // det_bin,
            det_bin,
            detector_columns // det_bin,
            det_bin,
        )
        if isinstance(reshaped, torch.Tensor):
            dtype = reshaped.dtype if reshaped.is_floating_point() else torch.int32
        else:
            dtype = (
                reshaped.dtype
                if np.issubdtype(reshaped.dtype, np.floating)
                else np.int32
            )
        return reshaped.sum((3, 5), dtype=dtype)

    raw_0, raw_1 = (detector_bin(dataset) for dataset in correction._datasets)
    corrected_0 = correction.apply_correction(
        raw_0,
        image_index=0,
        output_dtype="same",
        output_device=correction.device,
        verbose=False,
    )
    corrected_1 = correction.apply_correction(
        raw_1,
        image_index=1,
        output_dtype="same",
        output_device=correction.device,
        verbose=False,
    )
    quarter_turns = _rot90_to_image0_frame(correction)
    if quarter_turns:
        corrected_1 = (
            torch.rot90(corrected_1, quarter_turns, dims=(0, 1))
            if isinstance(corrected_1, torch.Tensor)
            else np.rot90(corrected_1, quarter_turns, axes=(0, 1)).copy()
        )
    if isinstance(corrected_0, torch.Tensor):
        if corrected_0.is_floating_point():
            merged = (corrected_0 + corrected_1) * 0.5
        else:
            merged = (
                (corrected_0.to(torch.int64) + corrected_1.to(torch.int64)) >> 1
            ).to(corrected_0.dtype)
    elif np.issubdtype(corrected_0.dtype, np.floating):
        merged = (corrected_0 + corrected_1) * 0.5
    else:
        merged = (
            (corrected_0.astype(np.int64) + corrected_1.astype(np.int64)) >> 1
        ).astype(corrected_0.dtype)
    rows, columns = crop_slices(correction)
    return [to_numpy(cube[rows, columns]) for cube in (raw_0, corrected_0, merged)]


def corrected_4dstem(
    self,
    *,
    mode: str = "bilinear",
    chunk_size: int | None = None,
    merge: bool = True,
    verbose: bool = True,
    output_0: np.ndarray | None = None,
    output_1: np.ndarray | None = None,
    output_dtype: torch.dtype | np.dtype | str | None = None,
    output_device: str | torch.device | None = None,
) -> CorrectionResult:
    """Correct and optionally merge a 0/90 4D-STEM acquisition pair.

    Each acquisition receives its learned scan drift before the second scan is
    rotated into the first scan's frame. Preallocated NumPy or memmap outputs
    keep large detector datasets from requiring another full-size allocation.

    Parameters
    ----------
    mode : str, default "bilinear"
        Interpolation used along the scan axes.
    chunk_size : int or None, default None
        Detector channels corrected per batch. ``None`` selects automatically.
    merge : bool, default True
        Average the two corrected acquisitions in their shared frame.
    verbose : bool, default True
        Show progress for chunked correction and merging.
    output_0, output_1 : numpy.ndarray or None, default None
        Preallocated outputs for the two corrected acquisitions.
    output_dtype : torch.dtype, numpy dtype, str, or None, default None
        Output numeric type. Use ``"same"`` to preserve the input type.
    output_device : str, torch.device, or None, default None
        Device holding returned arrays when no preallocated output is supplied.

    Returns
    -------
    CorrectionResult
        Corrected acquisitions and their optional merged dataset.

    Examples
    --------
    >>> result = drift.corrected_4dstem(chunk_size=64)
    >>> merged = result.corrected_4dstem
    """
    if getattr(self, "_datasets", None) is None or self._reference_mode:
        raise RuntimeError(
            "corrected_4dstem() requires DriftCorrection.from_4dstem(data_0, "
            "data_1, ...). For reference-mode EDS/EELS/4D-STEM, use corrected()."
        )
    datasets = self._datasets
    if self._datasets_consumed:
        raise RuntimeError(
            "Raw datasets were already released to free device memory "
            "during a prior corrected call. Construct a new "
            "DriftCorrection to re-correct."
        )
    if len(datasets) < 2:
        raise ValueError(
            f"Need at least 2 datasets for scan collection correction, "
            f"got {len(datasets)}"
        )

    # When inputs are device-resident, release each raw dataset as soon as
    # its corrected output exists; otherwise we hold four full datasets
    # simultaneously, which exceeds device memory for multi-GB scan collections.
    inputs_on_device = (
        isinstance(datasets[0], torch.Tensor) and datasets[0].is_cuda
        and isinstance(datasets[1], torch.Tensor) and datasets[1].is_cuda
    )

    corrected_4dstem_0 = apply_correction_to_dataset(
        self, None, image_index=0, mode=mode, chunk_size=chunk_size,
        output_dtype=output_dtype, output_device=output_device,
        output=output_0, verbose=verbose,
        progress_desc="Correcting scan 1/2",
    )
    if inputs_on_device:
        self._datasets[0] = None
        torch.cuda.empty_cache()
    corrected_4dstem_1 = apply_correction_to_dataset(
        self, None, image_index=1, mode=mode, chunk_size=chunk_size,
        output_dtype=output_dtype, output_device=output_device,
        output=output_1, verbose=verbose,
        progress_desc="Correcting scan 2/2",
    )
    if inputs_on_device:
        self._datasets[1] = None
        self._datasets_consumed = True
        torch.cuda.empty_cache()

    rot_k = _rot90_to_image0_frame(self, image_index=1)
    if rot_k:
        if isinstance(corrected_4dstem_1, torch.Tensor):
            corrected_4dstem_1 = torch.rot90(
                corrected_4dstem_1, k=rot_k, dims=(0, 1),
            )
        else:
            corrected_4dstem_1 = np.rot90(
                corrected_4dstem_1, k=rot_k, axes=(0, 1),
            ).copy()

    corrected_4dstem = None
    if merge:
        if corrected_4dstem_0.shape != corrected_4dstem_1.shape:
            raise ValueError(
                f"Cannot merge: corrected_4dstem_0 shape {corrected_4dstem_0.shape} "
                f"!= corrected_4dstem_1 shape {corrected_4dstem_1.shape}. "
                f"Scan collection must have compatible scan dimensions "
                f"after correction and scan-angle rotation."
            )
        Hm = corrected_4dstem_0.shape[0]
        row_block = max(1, min(32, Hm))
        row_starts = range(0, Hm, row_block)
        merge_progress = tqdm(
            total=Hm,
            desc="Merging corrected scans",
            unit="row",
            disable=not verbose or len(row_starts) <= 1,
        )
        if isinstance(corrected_4dstem_0, torch.Tensor):
            corrected_4dstem = torch.empty_like(corrected_4dstem_0)
            for r0 in row_starts:
                r1 = min(r0 + row_block, Hm)
                if corrected_4dstem_0.is_floating_point():
                    corrected_4dstem[r0:r1] = (
                        corrected_4dstem_0[r0:r1]
                        + corrected_4dstem_1[r0:r1]
                    ) * 0.5
                else:
                    # Avoid promoting the full integer dataset to float32.
                    # int32 sum fits in 2× input bytes per row block.
                    a = corrected_4dstem_0[r0:r1].to(torch.int32)
                    a += corrected_4dstem_1[r0:r1].to(torch.int32)
                    a >>= 1  # divide by 2 (round-toward-zero for non-negative ints)
                    corrected_4dstem[r0:r1] = a.clamp_(
                        0, torch.iinfo(corrected_4dstem_0.dtype).max
                    ).to(corrected_4dstem_0.dtype)
                    del a
                merge_progress.update(r1 - r0)
        else:
            corrected_4dstem = np.empty_like(corrected_4dstem_0, dtype=np.float32)
            for r0 in row_starts:
                r1 = min(r0 + row_block, Hm)
                np.add(
                    corrected_4dstem_0[r0:r1],
                    corrected_4dstem_1[r0:r1],
                    out=corrected_4dstem[r0:r1],
                    dtype=np.float32,
                )
                corrected_4dstem[r0:r1] *= 0.5
                merge_progress.update(r1 - r0)
        merge_progress.close()

    # Extract raw VDFs from the stored alignment images. The scan collection
    # reference is the scalar channel correction implied by the learned scan
    # drift fields, not an external ground truth.
    alignment_vdf_0 = np.asarray(self.imgs[0].array)
    alignment_vdf_1 = np.asarray(self.imgs[1].array)
    scalar_corrected_vdf = corrected_virtual_images(
        self,
        alignment_vdf_0,
        alignment_vdf_1,
    )["corrected_image"]

    return CorrectionResult(
        corrected_4dstem=corrected_4dstem,
        corrected_4dstem_0=corrected_4dstem_0,
        corrected_4dstem_1=corrected_4dstem_1,
        scalar_corrected_vdf=scalar_corrected_vdf,
    )


def to_numpy(array, *, dtype=None):
    """Convert a device or host array to NumPy with optional dtype conversion."""
    if isinstance(array, torch.Tensor):
        result = (
            array.detach().cpu().numpy()
            if array.is_cuda or array.device.type == "mps"
            else array.detach().numpy()
        )
    elif hasattr(array, "get"):  # CuPy ndarray
        result = array.get()
    else:
        result = np.asarray(array)
    return result.astype(dtype) if dtype is not None else result
