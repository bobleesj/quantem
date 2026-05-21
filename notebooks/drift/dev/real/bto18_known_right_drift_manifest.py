#!/usr/bin/env python
"""Write a manifest for BTO_18 known-right-drift exports and SSB baselines."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path("/home/owner/repos/quantem")
QUANTEM_SRC = REPO / "src"
if QUANTEM_SRC.exists() and str(QUANTEM_SRC) not in sys.path:
    sys.path.insert(0, str(QUANTEM_SRC))

from quantem.imaging import read_known_drift_metadata
EXPORT_BASE = Path("/home/owner/ssd/data/dasol/20260415_BTOSTO/quantem/drift/real")
OUTPUT_BASE = REPO / "notebooks" / "drift" / "dev" / "outputs"
MANIFEST_CSV = OUTPUT_BASE / "bto18_known_right_drift_manifest.csv"
MANIFEST_JSON = OUTPUT_BASE / "bto18_known_right_drift_manifest.json"

DRIFT_OUTPUTS = {
    30: OUTPUT_BASE / "real_14_bto18_known_right_drift_live_ssb_baseline",
    60: OUTPUT_BASE / "real_bto18_known_right60_live_ssb_locked",
    90: OUTPUT_BASE / "real_bto18_known_right90_live_ssb_locked",
}


def master_path(right_px: int, acquisition: str) -> Path:
    export_dir = EXPORT_BASE / f"BTO_18_known_right{right_px}_crop400_detbin2_u16"
    if acquisition == "clean0":
        stem = "BTO_18_ground_truth_crop400_detbin2"
    elif acquisition == "drift0":
        stem = f"BTO_18_right{right_px}_image_0_crop400_detbin2"
    elif acquisition == "drift90":
        stem = f"BTO_18_right{right_px}_image_1_crop400_detbin2"
    else:
        raise ValueError(f"unknown acquisition {acquisition!r}")
    return export_dir / f"{stem}_master.h5"


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_h5_row(right_px: int, acquisition: str) -> dict[str, Any]:
    metadata = read_known_drift_metadata(master_path(right_px, acquisition))
    row = metadata.as_manifest_row()
    row.update({
        "right_px": right_px,
        "acquisition": acquisition,
    })
    return row


def read_locked_summary(right_px: int) -> dict[str, Any]:
    output_dir = DRIFT_OUTPUTS[right_px]
    summary_path = output_dir / "locked_from_clean0" / "locked_from_clean0_summary.json"
    with summary_path.open() as f:
        summary = json.load(f)
    figure_name = (
        f"bto18_known_right{right_px}_live_ssb_locked_from_clean0.png"
        if right_px != 30
        else "bto18_known_right30_live_ssb_locked_from_clean0.png"
    )
    return {
        "output_dir": str(output_dir),
        "locked_summary_path": str(summary_path),
        "locked_figure_path": str(output_dir / figure_name),
        "clean0_rotation_angle_deg": float(summary["clean0_rotation_angle_deg"]),
        "clean0_C10_nm": float(summary["clean0_aberrations"]["C10"]),
        "clean0_C12_nm": float(summary["clean0_aberrations"]["C12"]),
        "clean0_phi12_rad": float(summary["clean0_aberrations"]["phi12"]),
        "locked": summary["locked"],
    }


def build_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for right_px in (30, 60, 90):
        locked = read_locked_summary(right_px)
        for acquisition in ("clean0", "drift0", "drift90"):
            row = read_h5_row(right_px, acquisition)
            row.update({
                "output_dir": locked["output_dir"],
                "locked_summary_path": locked["locked_summary_path"],
                "locked_figure_path": locked["locked_figure_path"],
                "clean0_rotation_angle_deg": locked["clean0_rotation_angle_deg"],
                "clean0_C10_nm": locked["clean0_C10_nm"],
                "clean0_C12_nm": locked["clean0_C12_nm"],
                "clean0_phi12_rad": locked["clean0_phi12_rad"],
            })
            if acquisition == "clean0":
                row.update({
                    "locked_branch": "reference",
                    "locked_rotation_angle_deg": locked["clean0_rotation_angle_deg"],
                    "locked_loss": "",
                })
            else:
                selected = locked["locked"][acquisition]["selected"]
                row.update({
                    "locked_branch": locked["locked"][acquisition]["selected_branch"],
                    "locked_rotation_angle_deg": float(selected["rotation_angle_deg"]),
                    "locked_loss": float(selected["loss"]),
                })
            rows.append(row)
    return rows


def write_manifest(rows: list[dict[str, Any]]) -> None:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    with MANIFEST_JSON.open("w") as f:
        json.dump(rows, f, indent=2, default=jsonable)

    fieldnames = [
        "right_px",
        "acquisition",
        "master_path",
        "label",
        "known_drift_total_px_down",
        "known_drift_total_px_right",
        "scan_crop_row_start",
        "scan_crop_row_stop",
        "scan_crop_col_start",
        "scan_crop_col_stop",
        "probe_positions_shape",
        "positions_offset_shape",
        "det_bin",
        "detector_shape_px",
        "output_dir",
        "locked_summary_path",
        "locked_figure_path",
        "clean0_rotation_angle_deg",
        "clean0_C10_nm",
        "clean0_C12_nm",
        "clean0_phi12_rad",
        "locked_branch",
        "locked_rotation_angle_deg",
        "locked_loss",
    ]
    with MANIFEST_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key, "")) for key in fieldnames})


def main() -> None:
    rows = build_manifest()
    write_manifest(rows)
    print(f"wrote {MANIFEST_CSV}")
    print(f"wrote {MANIFEST_JSON}")
    for row in rows:
        print(
            f"right{row['right_px']:>2} {row['acquisition']:<7} "
            f"crop=({row['scan_crop_row_start']}:{row['scan_crop_row_stop']},"
            f"{row['scan_crop_col_start']}:{row['scan_crop_col_stop']}) "
            f"branch={row['locked_branch']} loss={row['locked_loss']}"
        )


if __name__ == "__main__":
    main()
