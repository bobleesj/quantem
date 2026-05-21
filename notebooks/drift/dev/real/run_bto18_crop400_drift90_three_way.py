"""Run the BTO_18 crop400 image-1/90-degree drift three-way comparison.

This active-development runner mirrors the image-0 control, but keeps the
90-degree geometry explicit:

1. clean global-frame reference, no added drift
2. 90 degree with known right drift saved in the global/image-0 frame,
   uncorrected raster positions
3. the same 90 degree drifted diffraction patterns, but known drift-corrected
   probe positions

All three cases lock the reconstruction to the common clean0/base rotation.
The final H5 export has already reindexed the 90-degree scan axes into the
global frame, so no +90 rotation branch is used here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np

import run_bto18_crop400_ptycho_suite as suite


EXPERIMENT_NAME = "bto18_crop400_drift90_three_way"
DEFAULT_TRIAL_ID_BASE = 2040


def case_note(
    case_key: str,
    *,
    slices: int,
    thickness: int,
    probes: int,
    obj_lr: float,
    probe_lr: float,
    batch: int,
    iters: int,
) -> str:
    shared = (
        "BTO_18 crop400/bin2 image-1/90-degree drift three-way comparison. "
        "Final H5 scan axes are in the global/image-0 specimen frame; probe positions are explicit "
        "crop-local coordinates in that same frame. "
        "Calibration is locked to clean0/base rotation 158.9 deg and defocus 781 A; "
        "no +90 branch is used after canonical export. "
        f"Recipe: S={slices}, slice_thickness={thickness} A, total_thickness={slices * thickness} A, "
        f"probe_modes={probes}, obj_lr={obj_lr}, probe_lr={probe_lr}, batch_size={batch}, iters={iters}. "
        "Ptychography position correction does not bilinear-interpolate DPs; "
        "the fused kernel uses rounded object patch centers plus fractional Fourier phase ramps. "
    )
    if case_key == "clean90":
        return (
            shared
            + "Case 1/3. DP source is the clean0 crop export in the global frame. "
            + "Position source is the explicit image-1/global no-drift raster."
        )
    if case_key == "drift90_raster":
        return (
            shared
            + "Case 2/3. DP source is image 1 with known right30 scan drift, already saved in the global frame. "
            + "Position source is the global no-drift raster, intentionally uncorrected for drift."
        )
    if case_key == "drift90_corrected":
        return (
            shared
            + "Case 3/3. DP source is exactly the same image-1 right30 drift cube as case 2. "
            + "Position source is the known drift-corrected image-1/90 probe-position array. "
            + "The detector pixels/DPs are not changed between cases 2 and 3."
        )
    raise KeyError(case_key)


def install_case_hooks(notes_by_case: dict[str, str]) -> None:
    original_make_data = suite.make_data
    original_extra_config = suite.extra_config

    def positions_for_case(case: suite.Case, positions: dict[str, np.ndarray]) -> np.ndarray | None:
        if case.position_mode == "image1_raster_global":
            return positions["image1_raster"]
        if case.position_mode == "image1_corrected_global":
            return positions["image1_corrected"]
        return None

    def reference_positions_for_case(case: suite.Case, positions: dict[str, np.ndarray]) -> np.ndarray:
        explicit = positions_for_case(case, positions)
        if explicit is None:
            raise ValueError(f"no reference positions for {case.name}")
        return explicit

    def make_data(case: suite.Case, cubes: dict[Path, cp.ndarray]) -> cp.ndarray:
        return original_make_data(case, cubes)

    def positions_for_case_with_clean(case: suite.Case, positions: dict[str, np.ndarray]) -> np.ndarray | None:
        if case.position_mode == "image1_clean_raster_global":
            return positions["image1_raster"]
        return positions_for_case(case, positions)

    def reference_positions_for_case_with_clean(case: suite.Case, positions: dict[str, np.ndarray]) -> np.ndarray:
        explicit = positions_for_case_with_clean(case, positions)
        if explicit is None:
            raise ValueError(f"no reference positions for {case.name}")
        return explicit

    def extra_config(case: suite.Case, cfg: suite.TrialConfig, data: cp.ndarray, elapsed_s: float) -> dict[str, Any]:
        out = original_extra_config(case, cfg, data, elapsed_s)
        same_dp_as = None
        if case.position_mode == "image1_corrected_global":
            same_dp_as = "drift90_raster"
        out["drift90_three_way"] = {
            "schema_version": 1,
            "experiment": EXPERIMENT_NAME,
            "case": case.name,
            "note": notes_by_case[case.name],
            "dp_source": [str(path) for path in case.h5_paths],
            "dp_source_transform": None,
            "position_source": case.position_mode,
            "same_dp_as": same_dp_as,
            "geometry_rule": (
                "Image-1 final H5 scan axes and probe positions are in the canonical global frame; "
                "therefore this run locks the common clean0/base rotation and does not add +90 deg."
            ),
        }
        out["drift_suite"]["notes"] = out["drift90_three_way"]["geometry_rule"]
        return out

    suite.positions_for_case = positions_for_case_with_clean
    suite.reference_positions_for_case = reference_positions_for_case_with_clean
    suite.make_data = make_data
    suite.extra_config = extra_config


def build_cases(args: argparse.Namespace) -> tuple[suite.Case, ...]:
    suffix = (
        f"s{args.slices}_t{args.slice_thickness}_p{args.probes}_"
        f"olr{str(args.obj_lr).replace('.', 'p')}_"
        f"plr{str(args.probe_lr).replace('.', 'p')}_"
        f"b{args.batch_size}_it{args.iters}_locked_globalframe"
    )
    b = int(args.trial_id_base)
    return (
        suite.Case(
            b,
            f"{b}_det96_crop400_clean90_original_{suffix}",
            "case 1/3: clean 90 degree, no added drift, explicit image-1 raster positions",
            suite.BASE_ROTATION_DEG,
            (suite.GROUND_TRUTH_H5,),
            "image1_clean_raster_global",
        ),
        suite.Case(
            b + 1,
            f"{b + 1}_det96_crop400_right30_img1_drift_raster_{suffix}",
            "case 2/3: 90 degree with known right30 drift, explicit image-1 raster positions",
            suite.BASE_ROTATION_DEG,
            (suite.IMAGE1_H5,),
            "image1_raster_global",
        ),
        suite.Case(
            b + 2,
            f"{b + 2}_det96_crop400_right30_img1_drift_corrected_positions_{suffix}",
            "case 3/3: same drifted image-1 DPs, explicit corrected image-1 positions",
            suite.BASE_ROTATION_DEG,
            (suite.IMAGE1_H5,),
            "image1_corrected_global",
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
            "positions_injected": True,
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
    parser.add_argument("--case", choices=["all", "clean90", "drift90_raster", "drift90_corrected"], default="all")
    parser.add_argument("--trial-id-base", type=int, default=DEFAULT_TRIAL_ID_BASE)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--slices", type=int, default=6)
    parser.add_argument("--slice-thickness", type=int, default=18)
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--obj-lr", type=float, default=0.2)
    parser.add_argument("--probe-lr", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4096)
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
        for key, case in zip(("clean90", "drift90_raster", "drift90_corrected"), cases)
    }
    install_case_hooks(notes_by_case)
    suite.install_explicit_position_patch()
    for case in cases:
        write_case_note(case, notes_by_case[case.name])

    if args.case == "all":
        selected = cases
    elif args.case == "clean90":
        selected = (cases[0],)
    elif args.case == "drift90_raster":
        selected = (cases[1],)
    else:
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
