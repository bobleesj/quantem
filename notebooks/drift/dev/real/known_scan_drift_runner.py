"""Backend helpers for the active known 4D-STEM scan-drift workflows.

The workflow notebooks should stay short. This module handles notebook
orchestration while QuantEM modules own the drift math, 4D-STEM export
metadata, crop selection, and plotting helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO = Path("/home/owner/repos/quantem")
QUANTEM_SRC = REPO / "src"
QUANTEM_LIVE_SRC = Path("/home/owner/repos/quantem.live/src")
for _src in (QUANTEM_SRC, QUANTEM_LIVE_SRC):
    if _src.exists() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from quantem.imaging import read_known_4dstem_drift_metadata


def px_token(value: float) -> str:
    value = float(value)
    sign = "m" if value < 0 else ""
    mag = abs(value)
    if np.isclose(mag, round(mag)):
        body = str(int(round(mag)))
    else:
        body = f"{mag:.2f}".replace(".", "p")
    return sign + body


def drift_label(drift_total_px_down_right: tuple[float, float]) -> str:
    down_px, right_px = drift_total_px_down_right
    return f"down{px_token(down_px)}_right{px_token(right_px)}"


@dataclass
class KnownScanDriftConfig:
    """Configuration shared by the forward-model and locked-SSB notebooks."""

    source_h5: Path
    drift_total_px_down_right: tuple[float, float] = (0.0, 30.0)
    dataset_label: str | None = None
    det_bin: int = 2
    save_crop: int = 400
    save_dtype: str = "u16"
    channel_chunk: int = 512
    voltage_kv: float = 300.0
    semiangle_mrad: float = 30.0
    scan_sampling_a: float = 0.264
    ssb_n_trials: int = 200
    ssb_refine: str = "nmead"
    gpu: int = 0
    export_base: Path | None = None
    output_base: Path = REPO / "notebooks" / "drift" / "dev" / "outputs"

    def __post_init__(self) -> None:
        self.source_h5 = Path(self.source_h5)
        if self.dataset_label is None:
            label = self.source_h5.stem
            if label.endswith("_master"):
                label = label[:-7]
            self.dataset_label = label
        if self.export_base is None:
            self.export_base = self.source_h5.parent / "quantem" / "drift" / "real"
        self.export_base = Path(self.export_base)
        self.output_base = Path(self.output_base)

    @property
    def drift_label(self) -> str:
        return drift_label(self.drift_total_px_down_right)

    @property
    def export_dir(self) -> Path:
        return self.export_base / (
            f"{self.dataset_label}_known_{self.drift_label}"
            f"_crop{self.save_crop}_detbin{self.det_bin}_{self.save_dtype}"
        )

    @property
    def output_dir(self) -> Path:
        return self.output_base / f"{self.dataset_label}_known_{self.drift_label}"

    @property
    def names(self) -> dict[str, str]:
        return {
            "clean0": f"{self.dataset_label}_ground_truth_crop{self.save_crop}_detbin{self.det_bin}",
            "drift0": f"{self.dataset_label}_{self.drift_label}_image_0_crop{self.save_crop}_detbin{self.det_bin}",
            "drift90": f"{self.dataset_label}_{self.drift_label}_image_1_crop{self.save_crop}_detbin{self.det_bin}",
        }

    @property
    def masters(self) -> dict[str, Path]:
        return {key: self.export_dir / f"{stem}_master.h5" for key, stem in self.names.items()}

    @property
    def forward_summary_npz(self) -> Path:
        return self.output_dir / f"{self.dataset_label}_known_{self.drift_label}_forward_model_summary.npz"

    @property
    def forward_vectors_png(self) -> Path:
        return self.output_dir / f"{self.dataset_label}_known_{self.drift_label}_probe_positions.png"

    @property
    def locked_summary_json(self) -> Path:
        return self.output_dir / "locked_ssb" / "locked_from_clean0_summary.json"

    @property
    def locked_figure_path(self) -> Path:
        return self.output_dir / f"{self.dataset_label}_known_{self.drift_label}_locked_ssb.png"

    def summary(self) -> dict[str, Any]:
        return {
            "source_h5": str(self.source_h5),
            "dataset_label": self.dataset_label,
            "drift_total_px_down_right": tuple(float(x) for x in self.drift_total_px_down_right),
            "export_dir": str(self.export_dir),
            "output_dir": str(self.output_dir),
        }


def export_status(config: KnownScanDriftConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, path in config.masters.items():
        row: dict[str, Any] = {"name": name, "exists": path.exists(), "path": str(path)}
        if path.exists():
            row.update(read_known_4dstem_drift_metadata(path).as_manifest_row())
        rows.append(row)
    return rows


def print_export_status(config: KnownScanDriftConfig) -> None:
    for row in export_status(config):
        if not row["exists"]:
            print(f"{row['name']:<7} missing  {row['path']}")
            continue
        print(
            f"{row['name']:<7} drift=({row['known_drift_total_px_down']}, "
            f"{row['known_drift_total_px_right']}) "
            f"crop=({row['scan_crop_row_start']}:{row['scan_crop_row_stop']},"
            f"{row['scan_crop_col_start']}:{row['scan_crop_col_stop']})"
        )


def _as_numpy(x: Any) -> np.ndarray:
    if x.__class__.__module__.startswith("torch") and hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float32, copy=False)
    if hasattr(x, "get"):
        return x.get().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def generate_forward_model_exports(config: KnownScanDriftConfig, *, overwrite: bool = False) -> dict[str, Any]:
    """Generate clean0/drift0/drift90 H5 exports and forward-model diagnostics."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)

    import cupy as cp
    import torch

    from quantem.core import config as quantem_config
    from quantem.imaging import (
        find_valid_square_scan_crop,
        integrate_virtual_detector_image,
        plot_known_4dstem_forward_model_vectors,
        rotated_scan_positions,
        save_known_4dstem_drift_export,
        scan_time_drift_field,
        simulate_drifted_4dstem,
        valid_scan_position_mask,
    )
    from quantem.live import detect_bf_radius, dp_mean
    from quantem.live import load as live_load
    from quantem.live.io import save as live_save

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for export generation.")
    quantem_config.set_device(0)
    device = torch.device(quantem_config.get("device"))
    config.export_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    def virtual_image(data_t, mask_np):
        return _as_numpy(integrate_virtual_detector_image(data_t, mask_np))

    def save_one(key: str, data_t: Any, scan_crop: tuple[slice, slice], positions, offsets, label: str):
        result = save_known_4dstem_drift_export(
            config.masters[key],
            data_t,
            scan_crop=scan_crop,
            positions_px=positions,
            positions_offset_px=offsets,
            label=label,
            source_master=config.source_h5,
            det_bin=config.det_bin,
            known_drift_total_px_down_right=config.drift_total_px_down_right,
            save_func=live_save,
            scan_shape=(config.save_crop, config.save_crop),
            dtype=config.save_dtype,
            overwrite=overwrite,
            save_kwargs={
                "batch_size": 4096,
                "frames_per_file": 32768,
                "compression": "lz4",
                "verbose": True,
            },
        )
        if result.skipped:
            print(f"{key}: exists, skipping export ({result.path})")
        elif result.stats is not None:
            stats = result.stats
            print(
                f"{key}: crop rows={stats.scan_crop_rows[0]}:{stats.scan_crop_rows[1]}, "
                f"cols={stats.scan_crop_cols[0]}:{stats.scan_crop_cols[1]}"
            )
            print(
                f"{key}: detector={stats.detector_shape_px}, "
                f"pre-quant range=[{stats.min_value:.3f}, {stats.max_value:.3f}], "
                f"clipped below/above uint16={stats.clipped_below}/{stats.clipped_above}"
            )
        return result

    t0 = time.perf_counter()
    load_result = live_load(
        str(config.source_h5),
        det_bin=config.det_bin,
        output_dtype=np.float32,
        verbose=True,
    )
    clean_cp = load_result.data
    print(f"loaded shape={clean_cp.shape}, dtype={clean_cp.dtype}")

    mean_dp_cp = dp_mean(clean_cp)
    (center_row, center_col), bf_radius = detect_bf_radius(mean_dp_cp)
    det_h, det_w = clean_cp.shape[-2:]
    rr, cc = np.meshgrid(
        np.arange(det_h, dtype=np.float32),
        np.arange(det_w, dtype=np.float32),
        indexing="ij",
    )
    radius = np.sqrt((rr - center_row) ** 2 + (cc - center_col) ** 2)
    bf_mask = radius <= bf_radius
    df_mask = (radius >= bf_radius * 1.5) & (radius <= min(det_h, det_w) * 0.46)

    clean_4dstem = torch.utils.dlpack.from_dlpack(clean_cp).contiguous()
    if clean_4dstem.device != device:
        clean_4dstem = clean_4dstem.to(device)
    del clean_cp, mean_dp_cp
    cp.get_default_memory_pool().free_all_blocks()

    drift_field = scan_time_drift_field(
        clean_4dstem.shape[:2],
        total_drift_px=config.drift_total_px_down_right,
    )
    sim0 = simulate_drifted_4dstem(
        clean_4dstem,
        scan_direction_degrees=0,
        drift_field_px=drift_field,
        channel_chunk=config.channel_chunk,
        device=device,
    )
    sim90 = simulate_drifted_4dstem(
        clean_4dstem,
        scan_direction_degrees=90,
        drift_field_px=drift_field,
        channel_chunk=config.channel_chunk,
        device=device,
    )
    torch.cuda.synchronize()

    clean_bf = virtual_image(clean_4dstem, bf_mask)
    clean_df = virtual_image(clean_4dstem, df_mask)
    bf0 = virtual_image(sim0["data"], bf_mask)
    df0 = virtual_image(sim0["data"], df_mask)
    bf90 = virtual_image(sim90["data"], bf_mask)
    df90 = virtual_image(sim90["data"], df_mask)
    print(f"generated forward model in {time.perf_counter() - t0:.2f} s")
    print(
        "DF std clean/drift0/drift90_display = "
        f"{np.std(clean_df):.5g}/{np.std(df0):.5g}/{np.std(np.rot90(df90, k=-1)):.5g}"
    )

    source_shape = clean_4dstem.shape[:2]
    nominal0 = rotated_scan_positions(source_shape, 0)
    nominal90 = rotated_scan_positions(source_shape, 90)
    zero_offset = np.zeros_like(nominal0, dtype=np.float32)
    clean_crop = find_valid_square_scan_crop(np.ones(source_shape, dtype=bool), config.save_crop)
    crop0 = find_valid_square_scan_crop(valid_scan_position_mask(sim0["positions"], source_shape), config.save_crop)
    crop90 = find_valid_square_scan_crop(valid_scan_position_mask(sim90["positions"], source_shape), config.save_crop)

    results = {
        "clean0": save_one("clean0", clean_4dstem, clean_crop, nominal0, zero_offset, "ground_truth_no_added_drift"),
        "drift0": save_one(
            "drift0",
            sim0["data"],
            crop0,
            sim0["positions"],
            sim0["positions_offset_px"],
            f"image_0_known_{config.drift_label}_drift",
        ),
        "drift90": save_one(
            "drift90",
            sim90["data"],
            crop90,
            sim90["positions"],
            sim90["positions_offset_px"],
            f"image_1_known_{config.drift_label}_drift",
        ),
    }

    fig, _ = plot_known_4dstem_forward_model_vectors(
        clean_df,
        drift_field,
        nominal0,
        nominal90,
        sim0["positions"],
        sim90["positions"],
    )
    fig.savefig(config.forward_vectors_png, bbox_inches="tight", dpi=220)
    import matplotlib.pyplot as plt
    plt.close(fig)

    np.savez_compressed(
        config.forward_summary_npz,
        clean_bf=clean_bf,
        clean_df=clean_df,
        image_0_bf=bf0,
        image_0_df=df0,
        image_90_bf=bf90,
        image_90_df=df90,
        drift_field_px=drift_field,
        positions_0=sim0["positions"],
        positions_90=sim90["positions"],
        positions_offset_px_0=sim0["positions_offset_px"],
        positions_offset_px_90=sim90["positions_offset_px"],
        bf_mask=bf_mask,
        df_mask=df_mask,
        source=str(config.source_h5),
        det_bin=config.det_bin,
        drift_total_px=np.asarray(config.drift_total_px_down_right, dtype=np.float32),
    )
    print(f"saved {config.forward_vectors_png}")
    print(f"saved {config.forward_summary_npz}")

    del clean_4dstem, sim0, sim90
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    return {"exports": results, "vectors_png": str(config.forward_vectors_png), "summary_npz": str(config.forward_summary_npz)}


def _phase_to_numpy(result: Any) -> np.ndarray:
    import cupy as cp

    phase = result.phase
    return (cp.asnumpy(phase) if hasattr(phase, "get") else np.asarray(phase)).astype(np.float32)


def _wrap_deg(angle: float) -> float:
    return ((float(angle) + 180.0) % 360.0) - 180.0


def _zero_mean(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return arr - np.nanmean(arr)


def _align_to_clean_frame(key: str, phase: np.ndarray) -> np.ndarray:
    return np.rot90(phase, k=-1) if key == "drift90" else phase


def run_locked_ssb(config: KnownScanDriftConfig) -> dict[str, Any]:
    """Fit clean0 SSB calibration, then lock it for drift0 and drift90."""
    import cupy as cp
    import matplotlib.pyplot as plt

    from quantem.live import load as live_load
    from quantem.live import ssb as live_ssb

    config.output_dir.mkdir(parents=True, exist_ok=True)
    locked_dir = config.output_dir / "locked_ssb"
    locked_dir.mkdir(parents=True, exist_ok=True)

    def load_numpy(key: str) -> np.ndarray:
        data, _ = live_load(str(config.masters[key]), verbose=False)
        arr = cp.asnumpy(data)
        del data
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        return arr

    print("Fitting clean0 calibration reference")
    clean_input = load_numpy("clean0")
    clean_result = live_ssb(
        clean_input,
        voltage_kV=config.voltage_kv,
        semiangle_mrad=config.semiangle_mrad,
        scan_sampling_A=config.scan_sampling_a,
        n_trials=config.ssb_n_trials,
        refine=config.ssb_refine,
        source_path=str(config.masters["clean0"]),
        verbose=True,
    )
    clean_phase = _phase_to_numpy(clean_result)
    clean_rotation = float(clean_result.rotation_angle_deg)
    clean_aberrations = {key: float(value) for key, value in clean_result.aberrations.items()}
    clean_loss = float(clean_result.loss) if clean_result.loss is not None else None
    del clean_result, clean_input
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    def run_candidate_set(key: str, branch_offsets: dict[str, float]):
        ssb_input = load_numpy(key)
        candidates: list[dict[str, Any]] = []
        for branch, offset in branch_offsets.items():
            rotation = _wrap_deg(clean_rotation + float(offset))
            print(f"locked {key} candidate {branch}: rot={rotation:.3f}, C10={clean_aberrations.get('C10', 0):.3f}")
            result = live_ssb(
                ssb_input,
                voltage_kV=config.voltage_kv,
                semiangle_mrad=config.semiangle_mrad,
                scan_sampling_A=config.scan_sampling_a,
                rotation_angle_deg=rotation,
                aberrations=clean_aberrations,
                n_trials=0,
                refine=None,
                source_path=str(config.masters[key]),
                verbose=True,
            )
            phase_raw = _phase_to_numpy(result)
            candidates.append({
                "name": key,
                "branch": branch,
                "offset_deg": float(offset),
                "rotation_angle_deg": float(result.rotation_angle_deg),
                "aberrations": {k: float(v) for k, v in result.aberrations.items()},
                "loss": float(result.loss) if result.loss is not None else None,
                "phase_raw": phase_raw,
                "phase_aligned": _align_to_clean_frame(key, phase_raw),
            })
            del result, phase_raw
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        del ssb_input
        return min(candidates, key=lambda item: np.inf if item["loss"] is None else item["loss"]), candidates

    best0, candidates0 = run_candidate_set("drift0", {"same": 0.0})
    best90, candidates90 = run_candidate_set("drift90", {"minus90": -90.0, "plus90": 90.0})

    locked_phases = {
        "clean0": clean_phase,
        "drift0": best0["phase_aligned"],
        "drift90": best90["phase_aligned"],
    }
    np.save(locked_dir / "ssb_phase_clean0_reference.npy", locked_phases["clean0"])
    np.save(locked_dir / "ssb_phase_drift0_locked_from_clean0_aligned.npy", locked_phases["drift0"])
    np.save(locked_dir / "ssb_phase_drift90_locked_from_clean0_aligned.npy", locked_phases["drift90"])

    def strip_arrays(candidate: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in candidate.items() if key not in {"phase_raw", "phase_aligned"}}

    summary = {
        "source": "clean0_calibration_locked_for_drift_acquisitions",
        "dataset_label": config.dataset_label,
        "drift_total_px_down_right": list(map(float, config.drift_total_px_down_right)),
        "clean0_rotation_angle_deg": clean_rotation,
        "clean0_aberrations": clean_aberrations,
        "clean0_loss": clean_loss,
        "locked": {
            "drift0": {
                "selected_branch": best0["branch"],
                "selected": strip_arrays(best0),
                "candidates": [strip_arrays(c) for c in candidates0],
            },
            "drift90": {
                "selected_branch": best90["branch"],
                "selected": strip_arrays(best90),
                "candidates": [strip_arrays(c) for c in candidates90],
            },
        },
    }
    config.locked_summary_json.write_text(json.dumps(summary, indent=2))

    stack = np.stack([_zero_mean(locked_phases[key]) for key in ("clean0", "drift0", "drift90")])
    vmin, vmax = np.nanpercentile(stack, [1, 99])
    clean = _zero_mean(locked_phases["clean0"])
    diffs = {
        "clean0": clean * 0,
        "drift0": _zero_mean(locked_phases["drift0"]) - clean,
        "drift90": _zero_mean(locked_phases["drift90"]) - clean,
    }
    dlim = float(np.nanpercentile(np.abs(np.stack([diffs["drift0"], diffs["drift90"]])), 99))
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), constrained_layout=True)
    labels = {"clean0": "clean 0", "drift0": "0 + drift", "drift90": "90 + drift"}
    for col, key in enumerate(("clean0", "drift0", "drift90")):
        ax = axes[0, col]
        im = ax.imshow(_zero_mean(locked_phases[key]), cmap="magma", vmin=vmin, vmax=vmax)
        if key == "clean0":
            title = f"clean 0 calibration\nrot={clean_rotation:.2f} deg, C10={clean_aberrations.get('C10', np.nan):.2f} nm"
        else:
            selected = summary["locked"][key]["selected"]
            title = (
                f"{labels[key]}\nlocked branch={summary['locked'][key]['selected_branch']}, "
                f"rot={selected['rotation_angle_deg']:.2f} deg\nloss={selected['loss']:.4g}"
            )
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        ax = axes[1, col]
        im = ax.imshow(diffs[key], cmap="coolwarm", vmin=-dlim, vmax=dlim)
        ax.set_title("phase minus clean0", fontsize=11)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"{config.dataset_label} known {config.drift_label} SSB: clean0 calibration locked",
        fontsize=14,
    )
    fig.savefig(config.locked_figure_path, bbox_inches="tight", dpi=240)
    plt.close(fig)
    print(config.locked_figure_path)
    print(config.locked_summary_json)
    print(json.dumps(summary["locked"], indent=2))
    return summary
