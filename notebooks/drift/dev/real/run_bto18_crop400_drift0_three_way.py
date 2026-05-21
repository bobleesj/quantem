"""Run the BTO_18 crop400 image-0 drift three-way ptychography comparison.

This active-development runner keeps the experiment intentionally narrow:

1. clean 0 degree, no added drift, regular raster positions
2. 0 degree with known right drift, regular raster positions
3. the same 0 degree drifted diffraction patterns, but known drift-corrected
   probe positions injected into the quantem.live ptychography path

The point is to isolate drift correction before adding the extra 90 degree
rotation/phase ambiguity. Case 2 and case 3 use the exact same drifted DP
source; only the position source differs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np

import run_bto18_crop400_ptycho_suite as suite


EXPERIMENT_NAME = "bto18_crop400_drift0_three_way"
DEFAULT_TRIAL_ID_BASE = 2030


def case_note(case_key: str, *, slices: int, thickness: int, probes: int, obj_lr: float, probe_lr: float, batch: int, iters: int) -> str:
    shared = (
        "BTO_18 regular crop400/bin2 image-0 drift three-way comparison. "
        "Physical scan is 400 x 400 from crop rows/cols 56:456; detector is 96 x 96. "
        "Calibration is locked to clean0: rotation 158.9 deg, defocus 781 A. "
        f"Recipe: S={slices}, slice_thickness={thickness} A, total_thickness={slices * thickness} A, "
        f"probe_modes={probes}, obj_lr={obj_lr}, probe_lr={probe_lr}, batch_size={batch}, iters={iters}. "
    )
    if case_key == "clean0":
        return (
            shared
            + "Case 1/3. DP source is the no-added-drift clean0 crop export. "
            + "Position source is the regular image-0 raster. This is the origin/reference."
        )
    if case_key == "drift0_raster":
        return (
            shared
            + "Case 2/3. DP source is image 0 with known right30 scan drift. "
            + "Position source is still the regular raster, intentionally uncorrected."
        )
    if case_key == "drift0_corrected":
        return (
            shared
            + "Case 3/3. DP source is exactly the same image-0 right30 drift cube as case 2. "
            + "Position source is the known drift-corrected image-0 probe-position array. "
            + "The detector pixels/DPs are not changed between cases 2 and 3."
        )
    raise KeyError(case_key)


def install_case_position_hooks() -> None:
    def positions_for_case(case: suite.Case, positions: dict[str, np.ndarray]) -> np.ndarray | None:
        if case.position_mode == "image0_corrected":
            return positions["image0_corrected"]
        return None

    def reference_positions_for_case(case: suite.Case, positions: dict[str, np.ndarray]) -> np.ndarray:
        if case.position_mode == "image0_corrected":
            return positions["image0_corrected"]
        return positions["image0_raster"]

    suite.positions_for_case = positions_for_case
    suite.reference_positions_for_case = reference_positions_for_case


def install_extra_config_hook(notes_by_case: dict[str, str]) -> None:
    original_extra_config = suite.extra_config

    def extra_config(case: suite.Case, cfg: suite.TrialConfig, data: cp.ndarray, elapsed_s: float) -> dict[str, Any]:
        out = original_extra_config(case, cfg, data, elapsed_s)
        out["drift_three_way"] = {
            "schema_version": 1,
            "experiment": EXPERIMENT_NAME,
            "case": case.name,
            "note": notes_by_case[case.name],
            "dp_source": [str(path) for path in case.h5_paths],
            "position_source": case.position_mode,
            "same_dp_as": (
                "drift0_raster"
                if case.position_mode == "image0_corrected"
                else None
            ),
            "interpretation": (
                "Case 2 and case 3 use the same drifted diffraction patterns; "
                "only the probe positions differ."
            ),
        }
        out["drift_suite"]["notes"] = out["drift_three_way"]["interpretation"]
        return out

    suite.extra_config = extra_config


def build_cases(args: argparse.Namespace) -> tuple[suite.Case, ...]:
    suffix = (
        f"s{args.slices}_t{args.slice_thickness}_p{args.probes}_"
        f"olr{str(args.obj_lr).replace('.', 'p')}_"
        f"plr{str(args.probe_lr).replace('.', 'p')}_"
        f"b{args.batch_size}_it{args.iters}_locked"
    )
    b = int(args.trial_id_base)
    return (
        suite.Case(
            b,
            f"{b}_det96_crop400_clean0_original_{suffix}",
            "case 1/3: clean 0 degree, no added drift, regular raster positions",
            suite.BASE_ROTATION_DEG,
            (suite.GROUND_TRUTH_H5,),
            "clean0_raster",
        ),
        suite.Case(
            b + 1,
            f"{b + 1}_det96_crop400_right30_img0_drift_raster_{suffix}",
            "case 2/3: 0 degree with known right30 drift, regular raster positions",
            suite.BASE_ROTATION_DEG,
            (suite.IMAGE0_H5,),
            "image0_raster",
        ),
        suite.Case(
            b + 2,
            f"{b + 2}_det96_crop400_right30_img0_drift_corrected_positions_{suffix}",
            "case 3/3: same drifted image-0 DPs, known drift-corrected probe positions",
            suite.BASE_ROTATION_DEG,
            (suite.IMAGE0_H5,),
            "image0_corrected",
        ),
    )


def write_case_note(case: suite.Case, note: str) -> None:
    case.trial_dir.mkdir(parents=True, exist_ok=True)
    (case.trial_dir / "NOTE.md").write_text(
        f"# {case.name}\n\n"
        f"{note}\n\n"
        f"DP source: `{case.h5_paths[0]}`\n\n"
        f"Position source: `{case.position_mode}`\n"
    )


def validate_cases(cases: tuple[suite.Case, ...], notes_by_case: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        if not (case.trial_dir / "config.json").exists():
            continue
        cfg = json.loads((case.trial_dir / "config.json").read_text())
        loss = np.load(case.trial_dir / "loss.npy")
        obj = np.load(case.trial_dir / "obj_phase.npy", mmap_mode="r")
        probe = np.load(case.trial_dir / "probe.npy", mmap_mode="r")
        row = {
            "case": case.name,
            "label": case.label,
            "trial_dir": str(case.trial_dir),
            "dp_source": [str(path) for path in case.h5_paths],
            "position_source": case.position_mode,
            "positions_injected": case.position_mode == "image0_corrected",
            "note": notes_by_case[case.name],
            "status": cfg.get("status"),
            "loss0": float(loss[0]),
            "loss_final": float(loss[-1]),
            "iters": int(len(loss)),
            "scan_shape": cfg.get("data", {}).get("scan_shape"),
            "num_positions": cfg.get("data", {}).get("num_positions"),
            "det_size_px": cfg.get("data", {}).get("det_size_px"),
            "rotation_deg": cfg.get("reconstruction", {}).get("forced_rotation_deg"),
            "defocus_A": cfg.get("reconstruction", {}).get("probe_defocus_A"),
            "obj_shape": [int(x) for x in obj.shape],
            "probe_shape": [int(x) for x in probe.shape],
        }
        rows.append(row)
    summary_path = suite.TRIALS_DIR / f"{EXPERIMENT_NAME}_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"summary: {summary_path}")
    for row in rows:
        print(json.dumps(row, indent=2))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", "clean0", "drift0_raster", "drift0_corrected"], default="all")
    parser.add_argument("--trial-id-base", type=int, default=DEFAULT_TRIAL_ID_BASE)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--slices", type=int, default=6)
    parser.add_argument("--slice-thickness", type=int, default=15)
    parser.add_argument("--probes", type=int, default=12)
    parser.add_argument("--obj-lr", type=float, default=0.2)
    parser.add_argument("--probe-lr", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--force", action="store_true", help="Reserved; existing finished trial dirs are still skipped.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite.ITERS = int(args.iters)
    suite.SLICES = int(args.slices)
    suite.SLICE_THICKNESS_A = int(args.slice_thickness)
    suite.PROBES = int(args.probes)
    suite.OBJ_LR = float(args.obj_lr)
    suite.PROBE_LR = float(args.probe_lr)
    suite.BATCH_SIZE = int(args.batch_size)

    install_case_position_hooks()
    suite.install_explicit_position_patch()

    cases = build_cases(args)
    notes_by_case = {
        case.name: case_note(
            key,
            slices=args.slices,
            thickness=args.slice_thickness,
            probes=args.probes,
            obj_lr=args.obj_lr,
            probe_lr=args.probe_lr,
            batch=args.batch_size,
            iters=args.iters,
        )
        for key, case in zip(("clean0", "drift0_raster", "drift0_corrected"), cases)
    }
    install_extra_config_hook(notes_by_case)
    for case in cases:
        write_case_note(case, notes_by_case[case.name])

    selected = cases if args.case == "all" else tuple(case for case in cases if args.case in case.name)
    if args.case == "clean0":
        selected = (cases[0],)
    elif args.case == "drift0_raster":
        selected = (cases[1],)
    elif args.case == "drift0_corrected":
        selected = (cases[2],)

    positions = suite.load_position_sets()
    det_mask = suite.detector_mask()
    needed_paths = sorted({path for case in selected for path in case.h5_paths})
    cubes = {path: suite.load_cube(path) for path in needed_paths}
    try:
        for case in selected:
            suite.run_case(case, cubes, positions, det_mask)
        validate_cases(cases, notes_by_case)
    finally:
        cubes.clear()
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
