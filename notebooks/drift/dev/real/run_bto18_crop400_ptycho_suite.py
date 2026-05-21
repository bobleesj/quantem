"""Run the BTO_18 crop400 known-drift ptychography ablation.

This is an active-development runner for the four-case comparison:

1. drifted image 0 alone, regular raster positions
2. drifted image 1 alone, regular raster positions with +90 deg locked rotation
3. image 0 + image 1 stacked, regular raster positions in the image-0 frame
4. image 0 + image 1 stacked, known drift-adjusted probe positions

The combined trials intentionally use the same stacked diffraction patterns.
Only the probe-position array changes between cases 3 and 4.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cupy as cp
import h5py
import numpy as np

from quantem.imaging.drift_simulation import (
    rotated_scan_positions,
    scan_time_drift_field,
)
from quantem.live.engine.ptycho.dataset import Dataset4dstemGPU
from quantem.live.engine.ptycho.pipeline import run_reconstruction
from quantem.live.engine.ptycho.save import save_trial
from quantem.live.engine.ptycho.trial_config import TrialConfig
from quantem.live.io import load as live_load


SESSION_DIR = Path("/home/owner/data/dasol/20260415_BTOSTO")
SOURCE_TRIAL = (
    SESSION_DIR
    / "quantem/ptycho/BTO_18/trials/"
    / "118_det96_scan0_s6_t15_p8_it50_pure_phase_decay"
)
DRIFT_DIR = (
    SESSION_DIR
    / "quantem/drift/real/BTO_18_known_right30_crop400_detbin2_u16"
)
GROUND_TRUTH_H5 = DRIFT_DIR / "BTO_18_ground_truth_crop400_detbin2_master.h5"
IMAGE0_H5 = DRIFT_DIR / "BTO_18_right30_image_0_crop400_detbin2_master.h5"
IMAGE1_H5 = DRIFT_DIR / "BTO_18_right30_image_1_crop400_detbin2_master.h5"
TRIALS_DIR = SESSION_DIR / "quantem/ptycho/BTO_18/trials"

ITERS = 10
BASE_ROTATION_DEG = 158.9
ROTATION_90_DEG = BASE_ROTATION_DEG + 90.0
SCAN_SHAPE = (400, 400)
DET_SHAPE = (96, 96)
SCAN_SAMPLING_A = 0.264
DET_SAMPLING_MRAD = 1.1108
VOLTAGE_KV = 300.0
SEMIANGLE_MRAD = 30.0
DEFOCUS_A = 781.0
SLICES = 6
SLICE_THICKNESS_A = 15
PROBES = 8
OBJ_LR = 0.2
PROBE_LR = 0.2
LR_SCHEDULE = "decay"
OBJ_TYPE = "pure_phase"
PADDING_PX = 64
BATCH_SIZE = 8192
SEED = 42


@dataclass(frozen=True)
class Case:
    trial_id: int
    name: str
    label: str
    rotation_deg: float
    h5_paths: tuple[Path, ...]
    position_mode: str

    @property
    def trial_dir(self) -> Path:
        return TRIALS_DIR / self.name


CASES = (
    Case(
        1986,
        "1986_det96_crop400_clean0_raster_s6_t15_p8_it10_locked",
        "0 deg clean no-added-drift reference, raster positions",
        BASE_ROTATION_DEG,
        (GROUND_TRUTH_H5,),
        "clean_raster",
    ),
    Case(
        1982,
        "1982_det96_crop400_right30_img0_drift_raster_s6_t15_p8_it10_locked",
        "0 deg drifted image alone, raster positions",
        BASE_ROTATION_DEG,
        (IMAGE0_H5,),
        "raster",
    ),
    Case(
        1983,
        "1983_det96_crop400_right30_img1_drift_raster_plus90_s6_t15_p8_it10_locked",
        "90 deg drifted image alone, raster positions, +90 deg locked rotation",
        ROTATION_90_DEG,
        (IMAGE1_H5,),
        "raster",
    ),
    Case(
        1984,
        "1984_det96_crop400_right30_0p90_combined_raster_s6_t15_p8_it10_locked",
        "0/90 combined, raster positions in image-0 frame",
        BASE_ROTATION_DEG,
        (IMAGE0_H5, IMAGE1_H5),
        "combined_raster",
    ),
    Case(
        1985,
        "1985_det96_crop400_right30_0p90_combined_corrected_s6_t15_p8_it10_locked",
        "0/90 combined, known drift-adjusted probe positions",
        BASE_ROTATION_DEG,
        (IMAGE0_H5, IMAGE1_H5),
        "combined_corrected",
    ),
)


def install_explicit_position_patch() -> None:
    """Let quantem.live's pipeline consume dset.probe_positions_px.

    The public live pipeline currently builds a PtychographyDatasetRaster from
    the dataset without passing custom positions. For this active-dev runner we
    patch that classmethod in-process so the rest of the CUDA path is unchanged.
    """

    from quantem.diffractive_imaging import PtychographyDatasetRaster

    if getattr(PtychographyDatasetRaster, "_quantem_live_probe_pos_patch", False):
        return

    original = PtychographyDatasetRaster.from_dataset4dstem.__func__

    @classmethod
    def from_dataset4dstem_with_positions(cls, dset, *args, **kwargs):
        if kwargs.get("probe_positions_px") is None:
            positions = getattr(dset, "probe_positions_px", None)
            if positions is not None:
                kwargs["probe_positions_px"] = positions
        return original(cls, dset, *args, **kwargs)

    PtychographyDatasetRaster.from_dataset4dstem = from_dataset4dstem_with_positions
    PtychographyDatasetRaster._quantem_live_probe_pos_patch = True


def load_source_config() -> dict[str, Any]:
    return json.loads((SOURCE_TRIAL / "config.json").read_text())


def load_cube(path: Path) -> cp.ndarray:
    result = live_load(str(path), verbose=True, det_bin=1)
    data = result.data
    if data.ndim == 3:
        data = data.reshape(*SCAN_SHAPE, *DET_SHAPE)
    if tuple(data.shape[:2]) != SCAN_SHAPE or tuple(data.shape[-2:]) != DET_SHAPE:
        raise ValueError(f"{path} loaded as {data.shape}, expected {(*SCAN_SHAPE, *DET_SHAPE)}")
    return cp.ascontiguousarray(data)


def h5_drift_metadata(path: Path) -> tuple[np.ndarray, tuple[slice, slice], tuple[float, float]]:
    with h5py.File(path, "r") as f:
        group = f["entry/quantem/drift"]
        positions = group["probe_positions_px"][...].astype(np.float32)
        row0, row1 = (int(x) for x in group.attrs["scan_crop_rows"])
        col0, col1 = (int(x) for x in group.attrs["scan_crop_cols"])
        drift_total = tuple(float(x) for x in group.attrs["known_drift_total_px_down_right"])
    return positions, (slice(row0, row1), slice(col0, col1)), drift_total


def crop_local_positions(path: Path, scan_direction_deg: float) -> tuple[np.ndarray, np.ndarray]:
    corrected_global, scan_crop, drift_total = h5_drift_metadata(path)
    source_shape = (512, 512)
    nominal_global = rotated_scan_positions(source_shape, scan_direction_deg)[scan_crop]
    drift_field = scan_time_drift_field(source_shape, total_drift_px=drift_total)[scan_crop]
    expected_corrected = nominal_global - drift_field
    np.testing.assert_allclose(corrected_global, expected_corrected, rtol=0, atol=1e-5)

    crop_origin = np.array([scan_crop[0].start, scan_crop[1].start], dtype=np.float32)
    raster_local = nominal_global - crop_origin
    corrected_local = corrected_global - crop_origin
    return raster_local.astype(np.float32), corrected_local.astype(np.float32)


def load_position_sets() -> dict[str, np.ndarray]:
    raster0, corrected0 = crop_local_positions(IMAGE0_H5, 0.0)
    raster1, corrected1 = crop_local_positions(IMAGE1_H5, 90.0)
    return {
        "image0_raster": raster0,
        "image1_raster": raster1,
        "image0_corrected": corrected0,
        "image1_corrected": corrected1,
        "combined_raster": np.concatenate([raster0, raster1], axis=0).astype(np.float32),
        "combined_corrected": np.concatenate([corrected0, corrected1], axis=0).astype(np.float32),
    }


def detector_mask() -> np.ndarray | None:
    path = SOURCE_TRIAL / "detector_mask.npy"
    if not path.exists():
        return None
    mask = np.load(path)
    if mask.shape != DET_SHAPE:
        raise ValueError(f"detector mask shape {mask.shape} does not match {DET_SHAPE}")
    return mask.astype(np.float32)


def bright_field(data: cp.ndarray, mask: cp.ndarray | None, chunk_size: int = 4096) -> np.ndarray:
    flat = data.reshape(-1, data.shape[-2], data.shape[-1])
    if mask is None:
        mask_cp = cp.ones(data.shape[-2:], dtype=cp.float32)
    else:
        mask_cp = cp.asarray(mask, dtype=cp.float32)
    out = cp.zeros(flat.shape[0], dtype=cp.float32)
    for start in range(0, flat.shape[0], chunk_size):
        stop = min(start + chunk_size, flat.shape[0])
        out[start:stop] = (flat[start:stop].astype(cp.float32) * mask_cp[None]).sum(axis=(-2, -1))
    return cp.asnumpy(out.reshape(data.shape[:2]))


def mean_dp(data: cp.ndarray) -> np.ndarray:
    return cp.asnumpy(data.astype(cp.float32).reshape(-1, *data.shape[-2:]).mean(axis=0))


def make_data(case: Case, cubes: dict[Path, cp.ndarray]) -> cp.ndarray:
    if len(case.h5_paths) == 1:
        return cp.ascontiguousarray(cubes[case.h5_paths[0]])
    return cp.ascontiguousarray(cp.concatenate([cubes[path] for path in case.h5_paths], axis=0))


def positions_for_case(case: Case, positions: dict[str, np.ndarray]) -> np.ndarray | None:
    if case.position_mode == "combined_raster":
        return positions["combined_raster"]
    if case.position_mode == "combined_corrected":
        return positions["combined_corrected"]
    return None


def reference_positions_for_case(case: Case, positions: dict[str, np.ndarray]) -> np.ndarray:
    if case.trial_id in (1982, 1986):
        return positions["image0_raster"]
    if case.trial_id == 1983:
        return positions["image1_raster"]
    explicit = positions_for_case(case, positions)
    if explicit is None:
        raise ValueError(f"no reference positions for {case.name}")
    return explicit


def summarize_positions(positions_px: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(positions_px, dtype=np.float32).reshape(-1, 2)
    return {
        "shape": [int(x) for x in positions_px.shape],
        "row_min_px": float(np.min(flat[:, 0])),
        "row_max_px": float(np.max(flat[:, 0])),
        "col_min_px": float(np.min(flat[:, 1])),
        "col_max_px": float(np.max(flat[:, 1])),
        "row_mean_px": float(np.mean(flat[:, 0])),
        "col_mean_px": float(np.mean(flat[:, 1])),
        "row_ptp_px": float(np.ptp(flat[:, 0])),
        "col_ptp_px": float(np.ptp(flat[:, 1])),
    }


def write_position_artifacts(
    case: Case,
    positions: dict[str, np.ndarray],
    *,
    explicit_positions: np.ndarray | None,
) -> None:
    """Persist the position source used by a trial."""

    case.trial_dir.mkdir(parents=True, exist_ok=True)
    reference = reference_positions_for_case(case, positions)
    if explicit_positions is not None:
        np.save(case.trial_dir / "probe_positions_px.npy", explicit_positions.astype(np.float32))
    np.save(case.trial_dir / "reference_probe_positions_px.npy", reference.astype(np.float32))
    summary = {
        "schema_version": 1,
        "case": case.name,
        "label": case.label,
        "position_mode": case.position_mode,
        "positions_injected_into_live": explicit_positions is not None,
        "locked_base_rotation_deg": BASE_ROTATION_DEG,
        "locked_image1_rotation_deg": ROTATION_90_DEG,
        "forced_rotation_deg": float(case.rotation_deg),
        "source_h5_paths": [str(path) for path in case.h5_paths],
        "reference_positions_px": summarize_positions(reference),
    }
    if explicit_positions is not None:
        summary["injected_positions_px"] = summarize_positions(explicit_positions)
    (case.trial_dir / "position_summary.json").write_text(json.dumps(summary, indent=2))


def trial_config(case: Case, data: cp.ndarray) -> TrialConfig:
    return TrialConfig(
        path=" + ".join(str(path) for path in case.h5_paths),
        scan=int(data.shape[0]),
        scan_sampling=SCAN_SAMPLING_A,
        voltage_kV=VOLTAGE_KV,
        semiangle=SEMIANGLE_MRAD,
        raw_det=DET_SHAPE[0],
        det=DET_SHAPE[0],
        slices=SLICES,
        slice_thickness=SLICE_THICKNESS_A,
        probes=PROBES,
        iters=ITERS,
        obj_lr=OBJ_LR,
        probe_lr=PROBE_LR,
        lr_schedule=LR_SCHEDULE,
        mode="per_trial",
        obj_type=OBJ_TYPE,
        padding=PADDING_PX,
        defocus=DEFOCUS_A,
        batch_size=BATCH_SIZE,
        rotation=case.rotation_deg,
        seed=SEED,
        denoise="tv",
        command_as_typed="python notebooks/drift/dev/real/run_bto18_crop400_ptycho_suite.py",
        parent_acq={
            "stem": "BTO_18",
            "session": "dasol/20260415_BTOSTO",
            "source_trial": SOURCE_TRIAL.name,
            "rotation_deg": BASE_ROTATION_DEG,
        },
    )


def extra_config(case: Case, cfg: TrialConfig, data: cp.ndarray, elapsed_s: float) -> dict[str, Any]:
    return {
        "sample": load_source_config().get("sample", {}),
        "microscope": load_source_config().get("microscope", {}),
        "data": {
            "path": cfg.path,
            "source_h5_paths": [str(path) for path in case.h5_paths],
            "scan_shape": [int(data.shape[0]), int(data.shape[1])],
            "num_positions": int(data.shape[0] * data.shape[1]),
            "scan_sampling_A_per_px": SCAN_SAMPLING_A,
            "det_sampling_mrad_per_px": DET_SAMPLING_MRAD,
            "rotation_deg": float(case.rotation_deg),
            "det_size_raw_px": DET_SHAPE[0],
            "det_size_px": DET_SHAPE[0],
            "bin_factor": 1,
            "bin_pad": False,
            "known_drift_total_px_down_right": [0.0, 30.0],
            "position_mode": case.position_mode,
            "position_label": case.label,
        },
        "reconstruction": {
            "mode": "per_trial",
            "slice_thickness_A": SLICE_THICKNESS_A,
            "batch_size": BATCH_SIZE,
            "obj_padding_px": PADDING_PX,
            "obj_type": OBJ_TYPE,
            "obj_lr_step": OBJ_LR,
            "probe_lr_step": PROBE_LR,
            "probe_defocus_A": DEFOCUS_A,
            "lr_schedule": LR_SCHEDULE,
            "requested_iters": ITERS,
            "warmup_iters": ITERS,
            "total_thickness_A": SLICES * SLICE_THICKNESS_A,
            "forced_rotation_deg": float(case.rotation_deg),
            "locked_base_rotation_deg": BASE_ROTATION_DEG,
            "locked_image1_rotation_deg": ROTATION_90_DEG,
            "seed": SEED,
            "kernel": f"fused_{DET_SHAPE[0]}_pfa",
        },
        "drift_suite": {
            "schema_version": 1,
            "case": case.name,
            "label": case.label,
            "position_mode": case.position_mode,
            "same_dp_stack_as": (
                "1984_det96_crop400_right30_0p90_combined_raster_s6_t15_p8_it10_locked"
                if case.position_mode == "combined_corrected"
                else None
            ),
            "notes": (
                "Single 90 degree run locks rotation to base+90. "
                "Combined runs use explicit image-0-frame positions and lock "
                "the global rotation to the base angle."
            ),
        },
        "results": {
            "elapsed_s": round(elapsed_s, 1),
            "elapsed_s_per_iter": round(elapsed_s / ITERS, 2),
        },
        "provenance": {
            "custom_runner": str(Path(__file__).resolve()),
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }


def write_runspec(case: Case, cfg: TrialConfig) -> None:
    case.trial_dir.mkdir(parents=True, exist_ok=True)
    spec = cfg.model_dump()
    spec["case_label"] = case.label
    spec["position_mode"] = case.position_mode
    (case.trial_dir / "_runspec.json").write_text(json.dumps(spec, indent=2))


def run_case(case: Case, cubes: dict[Path, cp.ndarray], positions: dict[str, np.ndarray], det_mask: np.ndarray | None) -> None:
    probe_positions_px = positions_for_case(case, positions)
    write_position_artifacts(case, positions, explicit_positions=probe_positions_px)
    if (case.trial_dir / "obj_phase.npy").exists() and (case.trial_dir / "config.json").exists():
        print(f"SKIP existing {case.trial_dir}")
        return

    data = make_data(case, cubes)
    if probe_positions_px is not None and tuple(probe_positions_px.shape[:2]) != tuple(data.shape[:2]):
        raise ValueError(
            f"{case.name}: positions shape {probe_positions_px.shape} does not match data {data.shape}"
        )

    cfg = trial_config(case, data)
    write_runspec(case, cfg)
    dset = Dataset4dstemGPU(
        data,
        scan_sampling=SCAN_SAMPLING_A,
        det_sampling=DET_SAMPLING_MRAD,
        name=case.name,
        file_path=str(case.h5_paths[0]),
        detector_mask=det_mask,
    )
    if probe_positions_px is not None:
        dset.probe_positions_px = probe_positions_px

    print(f"\n=== {case.name} ===")
    print(f"label: {case.label}")
    print(f"data: {data.shape}, positions: {None if probe_positions_px is None else probe_positions_px.shape}")
    print(f"locked rotation: {case.rotation_deg:.2f} deg")
    start = time.perf_counter()
    ptycho, elapsed = run_reconstruction(cfg, dset, trial_dir=case.trial_dir)
    elapsed = time.perf_counter() - start if elapsed is None else elapsed

    cfg_out = save_trial(
        case.trial_dir,
        ptycho,
        bf_image=bright_field(data, det_mask),
        bf_full=bright_field(data, det_mask),
        mean_dp=mean_dp(data),
        extra_config=extra_config(case, cfg, data, elapsed),
        seed=SEED,
    )
    if cfg.denoise and cfg.denoise != "none":
        try:
            from quantem.live.engine.denoise import run_and_persist_trial_denoise

            run_and_persist_trial_denoise(case.trial_dir, cfg.denoise)
        except (ImportError, RuntimeError, OSError, ValueError) as exc:
            print(f"denoise skipped for {case.name}: {exc}")

    loss = np.load(case.trial_dir / "loss.npy")
    print(
        f"saved {case.trial_dir} loss[0]={loss[0]:.6g} "
        f"loss[-1]={loss[-1]:.6g} num_iters={len(loss)}"
    )
    print(f"config status: {cfg_out.get('status')}")

    del ptycho, dset, data
    cp.get_default_memory_pool().free_all_blocks()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


def validate_outputs() -> None:
    rows = []
    for case in CASES:
        config_path = case.trial_dir / "config.json"
        loss_path = case.trial_dir / "loss.npy"
        obj_path = case.trial_dir / "obj_phase.npy"
        probe_path = case.trial_dir / "probe.npy"
        missing = [p.name for p in (config_path, loss_path, obj_path, probe_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{case.name} missing outputs: {missing}")
        cfg = json.loads(config_path.read_text())
        loss = np.load(loss_path)
        obj = np.load(obj_path, mmap_mode="r")
        probe = np.load(probe_path, mmap_mode="r")
        if len(loss) != ITERS:
            raise AssertionError(f"{case.name} loss length {len(loss)} != {ITERS}")
        if cfg.get("status") != "finished":
            raise AssertionError(f"{case.name} config status is {cfg.get('status')!r}")
        rows.append(
            {
                "trial": case.name,
                "loss0": float(loss[0]),
                "loss_final": float(loss[-1]),
                "obj_shape": tuple(int(x) for x in obj.shape),
                "probe_shape": tuple(int(x) for x in probe.shape),
                "positions": cfg.get("data", {}).get("num_positions"),
                "rotation": cfg.get("reconstruction", {}).get("forced_rotation_deg"),
                "position_mode": cfg.get("data", {}).get("position_mode"),
            }
        )
    summary_path = TRIALS_DIR / "bto18_right30_crop400_ptycho_suite_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"\nvalidated outputs; summary: {summary_path}")
    for row in rows:
        print(row)


def main() -> None:
    install_explicit_position_patch()
    positions = load_position_sets()
    det_mask = detector_mask()
    cubes = {GROUND_TRUTH_H5: load_cube(GROUND_TRUTH_H5), IMAGE0_H5: load_cube(IMAGE0_H5), IMAGE1_H5: load_cube(IMAGE1_H5)}
    try:
        for case in CASES:
            run_case(case, cubes, positions, det_mask)
        validate_outputs()
    finally:
        cubes.clear()
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
