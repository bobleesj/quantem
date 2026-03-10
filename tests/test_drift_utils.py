import pytest
import torch
from quantem.imaging.drift_utils import _roll_image, _normalize_grid, _build_base_grid


@pytest.mark.parametrize(
    "shift_y, shift_x, expected",
    [
        (1, 0, [[4, 5, 6],
                [1, 2, 3]]),       # shift y only
        (0, 1, [[3, 1, 2],
                [6, 4, 5]]),       # shift x only
        (1, 1, [[6, 4, 5],
                [3, 1, 2]]),       # both axes
    ],
    ids=["shift_y", "shift_x", "both"],
)
def test_roll_image(shift_y, shift_x, expected):
    """Roll 2x3 image and verify wrap-around.

    Input:
    [[1, 2, 3],
     [4, 5, 6]]
    """
    im = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
    result = _roll_image(im, shift_y=shift_y, shift_x=shift_x)
    assert torch.equal(result, torch.tensor(expected, dtype=torch.float32))


def test_normalize_grid_5x5():
    """Rescale pixel coordinates to [-1, 1] for PyTorch grid_sample.

    PyTorch grid_sample only accepts coordinates in [-1, 1], not pixels.
    For a 5x5 image, pixel coords are 0..4. _normalize_grid converts:

    Pixel space:          Normalized space (PyTorch convention):
      col: 0  1  2  3  4    x: -1  -0.5  0  0.5  1
      row: 0  1  2  3  4    y: -1  -0.5  0  0.5  1

    Note: PyTorch uses (x, y) = (col, row), which is opposite to the
    microscopy convention where x = row (scan line) and y = column.

    grid[r, c] = (row, col)  -->  norm[r, c] = (x=col, y=row)
    """
    grid = torch.zeros(5, 5, 2)
    grid[0, 0] = torch.tensor([0.0, 0.0])  # row=0, col=0
    grid[1, 1] = torch.tensor([1.0, 1.0])  # row=1, col=1
    grid[2, 2] = torch.tensor([2.0, 2.0])  # row=2, col=2 (center)
    grid[3, 3] = torch.tensor([3.0, 3.0])  # row=3, col=3
    grid[4, 4] = torch.tensor([4.0, 4.0])  # row=4, col=4
    result = _normalize_grid(grid, input_shape=(5, 5))
    # norm[r,c,0] = x (from col), norm[r,c,1] = y (from row)
    assert result[0, 0, 0].item() == -1.0   # col=0 -> x=-1
    assert result[0, 0, 1].item() == -1.0   # row=0 -> y=-1
    assert result[1, 1, 0].item() == -0.5   # col=1 -> x=-0.5
    assert result[1, 1, 1].item() == -0.5   # row=1 -> y=-0.5
    assert result[2, 2, 0].item() == 0.0    # col=2 -> x=0 (center)
    assert result[2, 2, 1].item() == 0.0    # row=2 -> y=0 (center)
    assert result[3, 3, 0].item() == 0.5    # col=3 -> x=0.5
    assert result[3, 3, 1].item() == 0.5    # row=3 -> y=0.5
    assert result[4, 4, 0].item() == 1.0    # col=4 -> x=1
    assert result[4, 4, 1].item() == 1.0    # row=4 -> y=1


def test_build_base_grid_0deg_identity():
    """At 0 degrees with same input/output shape, grid is identity.

    For a 3x3 image, each output pixel (r, c) should map to
    the same input pixel (r, c):

    grid[r, c] = (r, c)

    row map:      col map:
    [[0, 0, 0],   [[0, 1, 2],
     [1, 1, 1],    [0, 1, 2],
     [2, 2, 2]]    [0, 1, 2]]
    """
    grid = _build_base_grid(
        input_shape=(3, 3),
        output_shape=(3, 3),
        scan_angle_deg=0.0,
        device=torch.device("cpu"),
    )
    assert grid.shape == (3, 3, 2)
    # Check corners and center
    assert grid[0, 0, 0].item() == 0.0  # row
    assert grid[0, 0, 1].item() == 0.0  # col
    assert grid[1, 1, 0].item() == 1.0  # center row
    assert grid[1, 1, 1].item() == 1.0  # center col
    assert grid[2, 2, 0].item() == 2.0  # bottom-right row
    assert grid[2, 2, 1].item() == 2.0  # bottom-right col


def test_build_base_grid_90deg():
    """At 90 degrees, the grid rotates sampling coordinates.

    For a 3x3 image at 90 degrees, output pixel (0, 0) should
    sample from a rotated position in the input. The center pixel
    (1, 1) stays at (1, 1) since rotation is around the center.
    """
    grid = _build_base_grid(
        input_shape=(3, 3),
        output_shape=(3, 3),
        scan_angle_deg=90.0,
        device=torch.device("cpu"),
    )
    assert grid.shape == (3, 3, 2)
    # Center should map to itself
    assert grid[1, 1, 0].item() == 1.0  # center row
    assert grid[1, 1, 1].item() == 1.0  # center col
    # Top-left output (0,0) should sample from a rotated input position
    # At 90deg: r = oy - out_center_c + in_center_r = 0 - 1 + 1 = 0
    #           c = out_center_r + in_center_c - ox = 1 + 1 - 0 = 2
    assert grid[0, 0, 0].item() == 0.0  # row
    assert grid[0, 0, 1].item() == 2.0  # col
