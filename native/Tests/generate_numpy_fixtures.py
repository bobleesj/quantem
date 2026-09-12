"""Generate small independent NumPy oracles; never used in production workflows."""

import json
from pathlib import Path

import numpy as np


folder = Path(__file__).parent / "QuantEMMAPEDTests/Fixtures"
rng = np.random.default_rng(314159)
shape = (1, 17, 5, 7)
raw = rng.integers(0, 65536, shape, dtype=np.uint16)
bad = [0, 1, 10, 18, 34]
corrected = raw.copy()
for scan in range(17):
    for pixel in bad:
        row, column = divmod(pixel, 7)
        neighbors = [
            int(raw[0, scan, rr, cc])
            for rr in range(max(0, row - 1), min(5, row + 2))
            for cc in range(max(0, column - 1), min(7, column + 2))
            if rr * 7 + cc not in bad
        ]
        corrected[0, scan, row, column] = int(np.median(neighbors)) if neighbors else 0

image = rng.normal(size=(17, 19)).astype(np.float32)
sigma = 1.25
radius = int(2 * sigma)
kernel = np.exp(-np.arange(-radius, radius + 1, dtype=np.float64) ** 2 / (2 * sigma**2))
kernel /= kernel.sum()


def gaussian(values):
    result = values.astype(np.float64)
    for axis in (1, 0):
        result = np.apply_along_axis(
            lambda row: np.convolve(np.pad(row, radius, mode="reflect"), kernel, mode="valid"), axis, result
        )
    return result.astype(np.float32)


fixture = {
    "generator": "independent NumPy integer median, uint64 means, reflect Gaussian, and FFT",
    "shape": list(shape), "bad": bad, "raw": raw.ravel().tolist(),
    "corrected": corrected.ravel().tolist(),
    "dp_mean": (corrected.astype(np.uint64).sum(axis=(0, 1)).astype(np.float32) / 17).ravel().tolist(),
    "im_bf": (corrected.astype(np.uint64).sum(axis=(2, 3)).astype(np.float32) / 35).ravel().tolist(),
    "image_shape": [17, 19], "image": image.ravel().tolist(), "sigma": sigma,
    "gaussian": gaussian(image).ravel().tolist(),
    "translated": np.roll(image, (3, -4), axis=(0, 1)).ravel().tolist(),
    "alignment_shift": [-3.0, 4.0],
}
(folder / "numpy.json").write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
