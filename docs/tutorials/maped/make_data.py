"""Create portable synthetic MAPED inputs inspired by PR #256's validation.

Run ``python make_data.py synthetic-tilts`` once before opening a notebook.
All images are analytic; no experimental pixels or microscope metadata are used.
"""

import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin
import torch

SCAN_SHIFTS = [[0, 0], [2.3, -1.4], [-1.7, 3.1], [3.4, 2.2], [-2.6, -3.3]]
DETECTOR_SHIFTS = [[0, 0], [-1.2, 5.5], [2.1, -11.8], [-3, 5.1], [3.6, 2.9]]


def gaussian(row_t, col_t, center_row, center_col, sigma, amplitude=1):
    """Evaluate an analytic Gaussian at the supplied coordinates."""
    return amplitude * torch.exp(
        -((row_t - center_row) ** 2 + (col_t - center_col) ** 2) / (2 * sigma**2)
    )


def generate(folder: Path) -> None:
    """Write five 64 x 64 scans with 64 x 64 diffraction patterns and known shifts."""
    folder.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(0)
    row_t = torch.arange(64, dtype=torch.float32)[:, None]
    col_t = torch.arange(64, dtype=torch.float32)[None, :]
    for index, (scan, detector) in enumerate(zip(SCAN_SHIFTS, DETECTOR_SHIFTS, strict=True)):
        scan_row_t, scan_col_t = row_t - scan[0], col_t - scan[1]
        image_t = (
            3
            - gaussian(scan_row_t, scan_col_t, 24, 20, 15)
            + gaussian(scan_row_t, scan_col_t, 40, 44, 5, 0.7)
            + gaussian(scan_row_t, scan_col_t, 30, 38, 3, 0.5)
        )
        detector_row_t, detector_col_t = row_t - detector[0], col_t - detector[1]
        pattern_t = (
            gaussian(detector_row_t, detector_col_t, 32, 32, 2, 3)
            + gaussian(detector_row_t, detector_col_t, 18, 42, 2, 0.35)
            + gaussian(detector_row_t, detector_col_t, 42, 46, 2, 0.25)
            + gaussian(detector_row_t, detector_col_t, 46, 22, 2, 0.25)
            + gaussian(detector_row_t, detector_col_t, 22, 18, 2, 0.25)
        )
        expected_t = 200 * image_t[:, :, None, None] * pattern_t
        counts_t = torch.poisson(expected_t, generator=generator).to(torch.uint16)
        with h5py.File(folder / f"tilt_{index}_master.h5", "w") as file:
            file.attrs["scan_shape"] = (64, 64)
            file.attrs["name"] = f"Synthetic tilt {index}"
            file.attrs["synthetic"] = True
            file.create_dataset(
                "entry/data/data_000001",
                data=counts_t.numpy().reshape(-1, 64, 64),
                chunks=(1, 64, 64),
                **hdf5plugin.Bitshuffle(cname="lz4"),
            )
    truth = {
        "source": "https://github.com/electronmicroscopy/quantem/pull/256",
        "source_notebook": "https://github.com/user-attachments/files/29561645/maped_torch_validation.ipynb",
        "description": "Analytic Gaussians and shifts from PR256, evaluated at displaced coordinates; Poisson integer counts replace additive Gaussian noise.",
        "seed": 0,
        "shape": [64, 64, 64, 64],
        "tilts": 5,
        "scan_shifts_px": SCAN_SHIFTS,
        "detector_shifts_px": DETECTOR_SHIFTS,
        "shift_tolerance_px": 0.2,
        "minimum_bright_field_ncc": 0.95,
        "units": ["pixels", "pixels", "pixels", "pixels"],
    }
    (folder / "ground_truth.json").write_text(json.dumps(truth, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    generate(parser.parse_args().folder)
