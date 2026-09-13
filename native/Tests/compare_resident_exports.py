"""Validate native exports with GPU loaders and frozen Torch merge arithmetic."""
import argparse
import json
from pathlib import Path

import torch

from quantem.gpu import io
from quantem.diffraction import MAPEDTorch
from quantem.diffraction._maped_resident import ResidentMergeSource


def main() -> None:
    """Compare five complete scan rows from a seven-tilt qualification run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("run", type=Path, help="Contains display.json and exports/")
    args = parser.parse_args()
    report = json.loads((args.run / "display.json").read_text())
    files = sorted(args.inputs.glob("*_master.h5"))
    model = MAPEDTorch.from_files(files, device="mps")
    try:
        source = ResidentMergeSource(
            model.datasets.sources,
            torch.tensor(report["real_space_shifts"], device="mps").reshape(7, 2),
            torch.tensor(report["diffraction_shifts"], device="mps").reshape(7, 2),
            close_sources_before_reopen=False,
            compile_merge=False,
        )
        records = []
        for row in (0, 4, 252, 508, 511):
            region = (row, row + 1, 0, 512)
            reference = torch.cat(list(source.blocks(scan_region=region)))
            with io.load(
                args.run / "exports/float_master.h5", backend="mps",
                scan_region=region, representation="dense", output="torch",
                verbose=False,
            ) as loaded:
                actual = loaded.data.reshape(reference.shape)
                assert actual.device.type == "mps"
                difference = actual - reference
                assert bool(torch.all(
                    difference.abs() <= 2e-5 + 3e-6 * reference.abs()
                )), row
                item = dict(
                    row=row, values=reference.numel(),
                    float32_rmse=float(difference.square().mean().sqrt()),
                    float32_max=float(difference.abs().max()),
                )
            with io.load(
                args.run / "exports/scaled_master.h5", backend="mps",
                scan_region=region, verbose=False,
            ) as loaded:
                scaled = loaded.read().reshape(reference.shape)
                assert scaled.device.type == "mps"
                difference = scaled - actual
                calibration = report["precision"]["regions"][row // 8]
                assert bool(torch.all(
                    difference.abs() <= calibration["scale"] / 2 + 3e-5
                )), row
                item.update(
                    storage_rmse=float(difference.square().mean().sqrt()),
                    storage_max=float(difference.abs().max()),
                )
            records.append(item)
            print(item, flush=True)
        (args.run / "export-parity.json").write_text(
            json.dumps(dict(passed=True, records=records), indent=2) + "\n"
        )
    finally:
        model.close()


if __name__ == "__main__":
    main()
