"""Independent float32 normalized-grid oracle for zero-padded translation."""

import json
from pathlib import Path

import numpy as np


shape = (192, 192)
shift = np.array([-0.2942330539226532, 0.21367645263671875], np.float32)


def coordinates(length, displacement):
    """Preserve separately rounded float32 operations in the scientific grid."""
    step = np.float32(2) / np.float32(length - 1)
    base = np.arange(length, dtype=np.float32) * step - np.float32(1)
    normalized = base - np.float32(2) * displacement / np.float32(length)
    return (normalized + np.float32(1)) * np.float32(0.5) * np.float32(length - 1)


rows, columns = np.meshgrid(
    coordinates(shape[0], shift[0]),
    coordinates(shape[1], shift[1]),
    indexing="ij",
)
first_row, first_column = np.floor(rows), np.floor(columns)
row_fraction, column_fraction = rows - first_row, columns - first_column
result = np.zeros(shape, dtype=np.float32)
for dr, row_weight in [(0, 1 - row_fraction), (1, row_fraction)]:
    for dc, column_weight in [(0, 1 - column_fraction), (1, column_fraction)]:
        valid = (
            (first_row + dr >= 0) & (first_row + dr < shape[0])
            & (first_column + dc >= 0) & (first_column + dc < shape[1])
        )
        result += valid.astype(np.float32) * (row_weight * column_weight)

# The zero-padding boundary exposes an extra ULP in coordinate construction.
indices = np.unique(np.concatenate([
    np.arange(shape[1]),
    np.arange((shape[0] - 1) * shape[1], shape[0] * shape[1]),
    np.arange(0, shape[0] * shape[1], shape[1]),
    np.arange(shape[1] - 1, shape[0] * shape[1], shape[1]),
]))
fixture = {
    "shape": shape,
    "shift": shift.tolist(),
    "indices": indices.tolist(),
    "values": result.ravel()[indices].tolist(),
}
path = Path(__file__).parent / "QuantEMMAPEDTests/Fixtures/normalized_grid.json"
path.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
