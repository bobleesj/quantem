"""Torch implementation of bounded MAPED merging from resident acquisitions."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _summary_record(shape: tuple[int, int, int, int], sources: Sequence) -> dict:
    """Describe the MAPED reductions and corrected detector inputs."""
    scan_shape = [int(shape[0]), int(shape[1])]
    detector_shape = [int(shape[2]), int(shape[3])]
    corrections = [
        source.metadata.get(
            "hot_pixel_correction",
            {"method": "zero", "applied": False, "pixel_count": 0},
        )
        for source in sources
    ]
    methods = sorted({str(record.get("method", "zero")) for record in corrections})
    corrected = all(record.get("applied") is True for record in corrections)
    if corrected and methods == ["median"]:
        invalid_policy = "stored detector-mask pixels use their local 3x3 median"
    elif corrected and methods == ["zero"]:
        invalid_policy = "stored detector-mask pixels are replaced with zero"
    else:
        invalid_policy = (
            "stored detector-mask exclusions contribute zero; the divisor "
            "remains the complete detector pixel count"
        )
    return {
        "version": 1,
        "source_count": len(sources),
        "hot_pixel_correction": {
            "methods": methods,
            "applied_to_every_source": corrected,
            "pixel_counts": [
                int(record.get("pixel_count", 0)) for record in corrections
            ],
        },
        "mean_bright_field": {
            "operation": "arithmetic_mean",
            "reduction_axes": ["detector_row", "detector_column"],
            "divisor": math.prod(detector_shape),
            "output_shape": scan_shape,
            "detector_selection": "complete_detector",
            "invalid_pixel_policy": invalid_policy,
            "alignment_role": "real_space",
        },
        "mean_diffraction_pattern": {
            "operation": "arithmetic_mean",
            "reduction_axes": ["scan_row", "scan_column"],
            "divisor": math.prod(scan_shape),
            "output_shape": detector_shape,
            "invalid_pixel_policy": invalid_policy,
            "alignment_role": "diffraction_origin_and_shift",
        },
        "intensity_normalization": "none",
    }


def _weights(
    shape: tuple[int, int, int, int],
    real_space_shifts: torch.Tensor,
    diffraction_shifts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the established MAPED scan and detector weights in Torch."""
    rows, columns, detector_rows, detector_columns = shape
    device = real_space_shifts.device
    row = torch.arange(rows, dtype=torch.float32, device=device)
    column = torch.arange(columns, dtype=torch.float32, device=device)
    window = (
        ((row > 0) & (row < rows - 1))[:, None]
        * ((column > 0) & (column < columns - 1))[None]
    ).to(torch.float32)[None, None]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, detector_rows, device=device),
        torch.linspace(-1, 1, detector_columns, device=device),
        indexing="ij",
    )
    real_weights = []
    detector_weights = []
    detector_grids = []
    detector_ones = torch.ones(
        (1, 1, detector_rows, detector_columns),
        dtype=torch.float32,
        device=device,
    )
    for real_shift, diffraction_shift in zip(
        real_space_shifts, diffraction_shifts, strict=True
    ):
        source_row = row[:, None].expand(-1, columns) - real_shift[0]
        source_column = column[None].expand(rows, -1) - real_shift[1]
        grid = torch.stack(
            (
                2 * source_column / (columns - 1) - 1,
                2 * source_row / (rows - 1) - 1,
            ),
            dim=-1,
        )[None]
        real_weights.append(
            F.grid_sample(window, grid, align_corners=True)[0, 0]
        )
        grid = torch.stack(
            (
                xx - 2 * diffraction_shift[1] / detector_columns,
                yy - 2 * diffraction_shift[0] / detector_rows,
            ),
            dim=-1,
        )[None]
        detector_grids.append(grid[0])
        detector_weights.append(
            F.grid_sample(detector_ones, grid, align_corners=True)[0, 0].clamp(0, 1)
        )
    return (
        torch.stack(real_weights),
        torch.stack(detector_weights),
        torch.stack(detector_grids),
    )


def _automatic_region_frames(
    shape: tuple[int, int, int, int], device: torch.device
) -> int:
    """Choose a row-aligned Torch workspace from current accelerator memory."""
    pixels = math.prod(shape[2:])
    reserve = 512 * 1024**2
    if device.type == "cuda":
        available = max(0, int(torch.cuda.mem_get_info(device)[0]) - reserve)
    elif hasattr(torch.mps, "recommended_max_memory"):
        available = max(
            0,
            int(torch.mps.recommended_max_memory())
            - int(torch.mps.current_allocated_memory())
            - reserve,
        )
    else:
        available = 2 * 1024**3
    # Numerator, sampled and shifted values, denominator, decoded counts, and
    # encoded output may overlap briefly at a block boundary.
    frames = max(1, min(4096, available // max(1, pixels * 24)))
    columns = int(shape[1])
    if frames >= columns:
        frames = max(columns, frames // columns * columns)
    return int(frames)


def _sample_scan_rows(
    values: torch.Tensor,
    *,
    decoded_first_row: int,
    output_first_row: int,
    output_stop_row: int,
    shift: torch.Tensor | Sequence[float],
    out: torch.Tensor | None = None,
    decoded_first_column: int = 0,
    output_first_column: int = 0,
    output_stop_column: int | None = None,
) -> torch.Tensor:
    """Apply one rigid bilinear scan shift to a decoded row range."""
    decoded_rows, columns, detector_rows, detector_columns = values.shape
    # add_ converts native counts while accumulating into float32, avoiding
    # four materialized float copies of the overlapping interpolation taps.
    if values.dtype not in (torch.uint8, torch.uint16, torch.float32):
        values = values.to(torch.float32)
    output_rows = output_stop_row - output_first_row
    output_columns = (
        columns if output_stop_column is None
        else output_stop_column - output_first_column
    )
    output = (
        torch.empty(
            (output_rows, output_columns, detector_rows, detector_columns),
            dtype=torch.float32,
            device=values.device,
        )
        if out is None else out
    )
    output.zero_()
    row_offset = -float(shift[0])
    column_offset = -float(shift[1])
    row_floor = math.floor(row_offset)
    column_floor = math.floor(column_offset)
    row_fraction = row_offset - row_floor
    column_fraction = column_offset - column_floor
    row_taps = ((row_floor, 1 - row_fraction), (row_floor + 1, row_fraction))
    column_taps = (
        (column_floor, 1 - column_fraction),
        (column_floor + 1, column_fraction),
    )
    for row_delta, row_weight in row_taps:
        output_row0 = max(
            0, decoded_first_row - output_first_row - row_delta
        )
        output_row1 = min(
            output_rows,
            decoded_first_row + decoded_rows - output_first_row - row_delta,
        )
        if output_row0 >= output_row1 or row_weight == 0:
            continue
        source_row0 = output_first_row + output_row0 + row_delta - decoded_first_row
        source_row1 = output_first_row + output_row1 + row_delta - decoded_first_row
        for column_delta, column_weight in column_taps:
            output_column0 = max(
                0, decoded_first_column - output_first_column - column_delta
            )
            output_column1 = min(
                output_columns,
                decoded_first_column + columns - output_first_column - column_delta,
            )
            weight = row_weight * column_weight
            if output_column0 >= output_column1 or weight == 0:
                continue
            source_column0 = (
                output_first_column + output_column0 + column_delta - decoded_first_column
            )
            source_column1 = (
                output_first_column + output_column1 + column_delta - decoded_first_column
            )
            output[
                output_row0:output_row1,
                output_column0:output_column1,
            ].add_(
                values[
                    source_row0:source_row1,
                    source_column0:source_column1,
                ],
                alpha=weight,
            )
    return output


def _shift_detector(values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Apply MAPED's established constant bilinear detector shift."""
    frames, rows, columns = values.shape
    return F.grid_sample(
        values[:, None],
        grid[None].expand(frames, rows, columns, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0]


@torch.compile(fullgraph=True, dynamic=False)
def _sample_scan_interior(
    values: torch.Tensor, columns: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Fuse four ordered scan taps without specializing on shift values."""
    output = torch.zeros_like(values[:-1])
    output.addcmul_(values[:-1, columns[0]], weights[0, None, :, None, None])
    output.addcmul_(values[:-1, columns[1]], weights[1, None, :, None, None])
    output.addcmul_(values[1:, columns[0]], weights[2, None, :, None, None])
    output.addcmul_(values[1:, columns[1]], weights[3, None, :, None, None])
    return output


class ResidentMergeSource:
    """Re-readable Torch MAPED blocks consumed by ``quantem.gpu.io.save``."""

    dtype = np.dtype("float32")
    report_context = {
        "scope": "all merged values",
        "range_scope": "complete merged output",
        "measurement": "GPU comparison against merged float32 regions",
    }

    def __init__(
        self,
        sources: Sequence,
        real_space_shifts: torch.Tensor,
        diffraction_shifts: torch.Tensor,
        *,
        close_sources_before_reopen: bool,
        compile_merge: bool | None = None,
    ) -> None:
        self.sources = list(sources)
        if not self.sources:
            raise ValueError("Provide at least one resident MAPED acquisition.")
        self.shape = tuple(int(value) for value in self.sources[0].shape)
        if len(self.shape) != 4 or any(
            tuple(source.shape) != self.shape for source in self.sources
        ):
            raise ValueError("Resident MAPED acquisitions must share one 4D shape.")
        device = real_space_shifts.device
        for name, shifts in (
            ("real_space_shifts", real_space_shifts),
            ("diffraction_shifts", diffraction_shifts),
        ):
            if (
                not torch.is_tensor(shifts)
                or shifts.device != device
                or shifts.dtype != torch.float32
                or tuple(shifts.shape) != (len(self.sources), 2)
            ):
                raise ValueError(
                    f"{name} must be a float32 Torch tensor shaped "
                    f"({len(self.sources)}, 2) on one accelerator."
                )
        if device.type not in {"cuda", "mps"}:
            raise ValueError("Resident MAPED merging requires CUDA or MPS.")
        self._torch_device = device
        if device.type == "cuda":
            self._device_id = int(device.index or 0)
        self.real_space_shifts = real_space_shifts
        self.diffraction_shifts = diffraction_shifts
        # Scalar displacements only: reading each scalar in the region loop
        # would wait for previously queued accelerator work on every tilt.
        self._scan_shifts = real_space_shifts.detach().cpu().tolist()
        self._compile_scan = device.type == "mps" and compile_merge is not False
        self._scan_columns = []
        self._scan_weights = []
        if self._compile_scan:
            column = torch.arange(self.shape[1], device=device)
            for row_shift, column_shift in self._scan_shifts:
                row_fraction = -row_shift - math.floor(-row_shift)
                column_floor = math.floor(-column_shift)
                column_fraction = -column_shift - column_floor
                indices = torch.stack((column + column_floor, column + column_floor + 1))
                valid = ((indices >= 0) & (indices < self.shape[1])).to(torch.float32)
                self._scan_columns.append(indices.clamp(0, self.shape[1] - 1))
                self._scan_weights.append(torch.stack([
                    valid[index] * (row_weight * column_weight)
                    for row_weight in (1 - row_fraction, row_fraction)
                    for index, column_weight in enumerate((1 - column_fraction, column_fraction))
                ]))
        self.region_frames = _automatic_region_frames(self.shape, device)
        self.real_weights, self.detector_weights, self.detector_grids = _weights(
            self.shape, real_space_shifts, diffraction_shifts
        )
        self.detector_edge = 1 - self.detector_weights.sum(0).clamp(0, 1)
        self._pass_generation_seconds: list[float] = []
        self._close_sources_before_reopen = bool(close_sources_before_reopen)
        self.save_metadata = {
            "quantem_maped_summary_v1": json.dumps(
                _summary_record(self.shape, self.sources)
            )
        }
        self._update_merge_metadata()

    def _update_merge_metadata(self) -> None:
        representations = {
            str(source.metadata.get("representation", "unknown"))
            for source in self.sources
        }
        record = {
            "version": 1,
            "backend": self._torch_device.type,
            "source_representation": (
                representations.pop() if len(representations) == 1 else "mixed"
            ),
            "region_frames": self.region_frames,
            "merge_generation_pass_seconds": self._pass_generation_seconds,
            "released_sources_before_reopen": self._close_sources_before_reopen,
            "real_space_shifts_row_column": self.real_space_shifts.detach().cpu().tolist(),
            "diffraction_shifts_row_column": self.diffraction_shifts.detach().cpu().tolist(),
        }
        self.save_metadata["quantem_maped_merge_v1"] = json.dumps(record)

    def blocks(
        self, scan_region: tuple[int, int, int, int] | None = None
    ) -> Iterator[torch.Tensor]:
        """Yield complete row-aligned MAPED regions using only Torch math."""
        rows, columns, detector_rows, detector_columns = self.shape
        row_start, row_stop, column_start, column_stop = (
            (0, rows, 0, columns) if scan_region is None else scan_region
        )
        rows_per_region = max(1, self.region_frames // columns)
        generation_seconds = 0.0
        sampled_workspace = torch.empty(
            (
                min(row_stop - row_start, rows_per_region),
                column_stop - column_start,
                detector_rows,
                detector_columns,
            ),
            dtype=torch.float32,
            device=self._torch_device,
        )
        for output_row0 in range(row_start, row_stop, rows_per_region):
            output_row1 = min(row_stop, output_row0 + rows_per_region)
            started = time.perf_counter()
            numerator = None
            for index, source in enumerate(self.sources):
                shift = self._scan_shifts[index]
                row_offset = math.floor(-shift[0])
                column_offset = math.floor(-shift[1])
                decoded_row0 = max(0, output_row0 + row_offset)
                decoded_row1 = min(rows, output_row1 - 1 + row_offset + 2)
                decoded_column0 = max(0, column_start + column_offset)
                decoded_column1 = min(columns, column_stop + column_offset + 1)
                if column_start == 0 and column_stop == columns:
                    decoded_column0, decoded_column1 = 0, columns
                if decoded_row0 < decoded_row1 and decoded_column0 < decoded_column1:
                    decoded = source.read(
                        scan_region=(
                            decoded_row0, decoded_row1, decoded_column0, decoded_column1
                        )
                    )
                    if (
                        self._compile_scan
                        and scan_region is None
                        and (output_row1 - output_row0) * columns >= 2048
                        and decoded_row0 == output_row0 + row_offset
                        and decoded_row1 == output_row1 + row_offset + 1
                        and (shift[0] != int(shift[0]) or shift[1] != int(shift[1]))
                    ):
                        # The compiler does not accept native uint16. This single
                        # exact cast lets it fuse all four float32 tap updates.
                        sampled = _sample_scan_interior(
                            decoded.to(torch.float32),
                            self._scan_columns[index], self._scan_weights[index],
                        )
                    else:
                        sampled = _sample_scan_rows(
                            decoded,
                            decoded_first_row=decoded_row0,
                            output_first_row=output_row0,
                            output_stop_row=output_row1,
                            shift=shift,
                            out=sampled_workspace[: output_row1 - output_row0],
                            decoded_first_column=decoded_column0,
                            output_first_column=column_start,
                            output_stop_column=column_stop,
                        )
                    del decoded
                else:
                    sampled = sampled_workspace[: output_row1 - output_row0]
                    sampled.zero_()
                shifted = _shift_detector(
                    sampled.reshape(-1, detector_rows, detector_columns),
                    self.detector_grids[index],
                ).reshape_as(sampled)
                del sampled
                weight = self.real_weights[
                    index, output_row0:output_row1, column_start:column_stop, None, None
                ]
                if numerator is None:
                    numerator = shifted.mul_(weight)
                else:
                    numerator.addcmul_(weight, shifted)
                    del shifted
            assert numerator is not None
            denominator = torch.einsum(
                "nrc,nhw->rchw",
                self.real_weights[:, output_row0:output_row1, column_start:column_stop],
                self.detector_weights,
            )
            denominator += self.detector_edge[None, None]
            zero = denominator == 0
            numerator.div_(denominator)
            numerator.masked_fill_(zero, 0)
            del denominator, zero
            generation_seconds += time.perf_counter() - started
            yield numerator.reshape(-1, detector_rows, detector_columns)
        self._pass_generation_seconds.append(generation_seconds)
        self._update_merge_metadata()

    def close(self) -> None:
        """Release Torch planning tensors while leaving source ownership unchanged."""
        self.real_weights = None
        self.detector_weights = None
        self.detector_grids = None
        self.detector_edge = None
        self._scan_columns = []
        self._scan_weights = []
