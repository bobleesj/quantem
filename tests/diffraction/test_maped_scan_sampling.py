"""Resident inspection preserves ordered float32 scan interpolation."""

import math

import pytest
import torch

from quantem.diffraction._maped_resident import _sample_scan_rows


@pytest.mark.parametrize("device", ["cuda:0", "mps"])
def test_scan_regions_match_ordered_taps(device):
    """Border, integer, fractional, and empty overlaps preserve full-range counts."""
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    shape = (17, 13, 4, 8)
    indices_t = torch.arange(math.prod(shape), device=device).reshape(shape)
    values_t = ((indices_t * 37) % 65536).to(torch.uint16)
    for shift in [(0, 0), (-1.25, 0.6), (0.75, -1.4), (8.25, 4.5), (30, 0)]:
        row_floor, column_floor = math.floor(-shift[0]), math.floor(-shift[1])
        row_fraction = -shift[0] - row_floor
        column_fraction = -shift[1] - column_floor
        for row0, row1, column0, column1 in [(0, 8, 0, 13), (4, 12, 3, 10)]:
            expected_t = torch.zeros((row1 - row0, column1 - column0, *shape[2:]), device=device)
            for row_delta, row_weight in [
                (row_floor, 1 - row_fraction),
                (row_floor + 1, row_fraction),
            ]:
                for column_delta, column_weight in [
                    (column_floor, 1 - column_fraction),
                    (column_floor + 1, column_fraction),
                ]:
                    weight = row_weight * column_weight
                    if weight == 0:
                        continue
                    for row in range(row0, row1):
                        for column in range(column0, column1):
                            source_row, source_column = row + row_delta, column + column_delta
                            if 0 <= source_row < shape[0] and 0 <= source_column < shape[1]:
                                expected_t[row - row0, column - column0].add_(
                                    values_t[source_row, source_column], alpha=weight
                                )
            actual_t = _sample_scan_rows(
                values_t,
                decoded_first_row=0,
                output_first_row=row0,
                output_stop_row=row1,
                shift=shift,
                output_first_column=column0,
                output_stop_column=column1,
            )
            assert torch.equal(actual_t, expected_t)
