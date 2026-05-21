#!/usr/bin/env python
"""Generate BTO_18 known-right-drift exports and locked SSB figures.

This is the reusable version of notebooks 13/14 for BTO_18. It creates the
three acquisition files used by the downstream ptychography comparisons:

* clean0: clean source crop
* drift0: 0-degree scan with known right drift
* drift90: 90-degree scan with the same known right drift

The locked SSB section keeps the clean0 aberrations fixed. Final H5 scan axes
are canonicalized into the global/image-0 frame, so drift90 uses the same
locked rotation as clean0 and drift0.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--right-px", type=float, required=True, help="Total right drift in scan pixels.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to expose through CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--force-screen", action="store_true", help="Force live screen recomputation.")
    parser.add_argument("--force-generate", action="store_true", help="Overwrite generated H5 exports if present.")
    parser.add_argument("--skip-generate", action="store_true", help="Use existing generated H5 exports.")
    parser.add_argument("--skip-screen", action="store_true", help="Skip live screen.")
    parser.add_argument("--skip-locked", action="store_true", help="Skip locked SSB figure generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    warnings.filterwarnings("ignore", message=r".*pynvml package is deprecated.*", category=FutureWarning)

    import cupy as cp
    import matplotlib
    import numpy as np
    import torch

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    repo = Path("/home/owner/repos/quantem")
    quantem_src = repo / "src"
    quantem_live_src = Path("/home/owner/repos/quantem.live/src")
    for src in (quantem_src, quantem_live_src):
        if src.exists() and str(src) not in sys.path:
            sys.path.insert(0, str(src))

    from quantem.core import config
    from quantem.imaging import (
        find_valid_square_scan_crop,
        integrate_virtual_detector_image,
        rotated_scan_positions,
        scan_time_drift_field,
        save_known_4dstem_drift_export,
        simulate_drifted_4dstem,
        valid_scan_position_mask,
    )
    from quantem.live import detect_bf_radius
    from quantem.live import dp_mean
    from quantem.live import load as live_load
    from quantem.live import ssb as live_ssb
    from quantem.live.control.pipelines.screen import screen
    from quantem.live.io import save as live_save

    if not torch.cuda.is_available():
        raise RuntimeError("This script is intended to run on GPU.")
    config.set_device(0)
    device = torch.device(config.get("device"))
    print(f"visible GPU device {device}: {torch.cuda.get_device_name(device)}")

    right_px = float(args.right_px)
    right_label = int(round(right_px))
    drift_total_px = (0.0, right_px)

    source = Path("/home/owner/ssd/data/dasol/20260415_BTOSTO/BTO_18_master.h5")
    export_dir = (
        source.parent
        / "quantem"
        / "drift"
        / "real"
        / f"BTO_18_known_right{right_label}_crop400_detbin2_u16"
    )
    out = repo / "notebooks" / "drift" / "dev" / "outputs" / f"real_bto18_known_right{right_label}_live_ssb_locked"
    sidecar_root = out / "screen_sidecar"
    screen_dir = sidecar_root / "quantem" / "screen"
    out.mkdir(parents=True, exist_ok=True)

    det_bin = 2
    save_crop = 400
    save_dtype = "u16"
    channel_chunk = 512
    voltage_kv = 300.0
    semiangle_mrad = 30.0
    scan_sampling_a = 0.264
    ssb_n_trials = 200
    ssb_refine = "nmead"

    names = {
        "clean0": "BTO_18_ground_truth_crop400_detbin2",
        "drift0": f"BTO_18_right{right_label}_image_0_crop400_detbin2",
        "drift90": f"BTO_18_right{right_label}_image_1_crop400_detbin2",
    }
    masters = {key: export_dir / f"{stem}_master.h5" for key, stem in names.items()}

    def as_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32, copy=False)
        return np.asarray(x, dtype=np.float32)

    def virtual_image(data_t, mask_np):
        return as_numpy(integrate_virtual_detector_image(data_t, mask_np))

    def save_one_export(name, data_t, scan_crop, positions, positions_offset, label, *, scan_direction_degrees, nominal_positions):
        path = export_dir / f"{name}_master.h5"
        result = save_known_4dstem_drift_export(
            path,
            data_t,
            scan_crop=scan_crop,
            positions_px=positions,
            positions_offset_px=positions_offset,
            label=label,
            source_master=source,
            det_bin=det_bin,
            known_drift_total_px_down_right=drift_total_px,
            save_func=live_save,
            scan_shape=(save_crop, save_crop),
            dtype=save_dtype,
            overwrite=args.force_generate,
            scan_direction_degrees=scan_direction_degrees,
            scan_axes_frame="global",
            nominal_positions_px=nominal_positions,
            save_kwargs={
                "batch_size": 4096,
                "frames_per_file": 32768,
                "compression": "lz4",
                "verbose": True,
            },
        )
        if result.skipped:
            print(f"{name}: exists, skipping export ({path})")
        elif result.stats is not None:
            stats = result.stats
            print(f"{name}: global crop rows={stats.scan_crop_rows[0]}:{stats.scan_crop_rows[1]}, cols={stats.scan_crop_cols[0]}:{stats.scan_crop_cols[1]}")
            print(f"{name}: saved detector={stats.detector_shape_px}, no detector padding")
            print(f"{name}: pre-quant range=[{stats.min_value:.3f}, {stats.max_value:.3f}], clipped below/above uint16 = {stats.clipped_below}/{stats.clipped_above}")
        cp.get_default_memory_pool().free_all_blocks()
        return path

    def generate_exports():
        export_dir.mkdir(parents=True, exist_ok=True)
        print(f"generating right{right_label}: source={source}")
        print(f"export directory: {export_dir}")
        t0 = time.perf_counter()
        load_result = live_load(str(source), det_bin=det_bin, output_dtype=np.float32, verbose=True)
        clean_cp = load_result.data
        print(f"loaded shape={clean_cp.shape}, dtype={clean_cp.dtype}, type={type(clean_cp).__name__}")

        mean_dp_cp = dp_mean(clean_cp)
        (center_row, center_col), bf_radius = detect_bf_radius(mean_dp_cp)
        det_h, det_w = clean_cp.shape[-2:]
        rr, cc = np.meshgrid(np.arange(det_h, dtype=np.float32), np.arange(det_w, dtype=np.float32), indexing="ij")
        radius = np.sqrt((rr - center_row) ** 2 + (cc - center_col) ** 2)
        bf_mask = radius <= bf_radius
        df_mask = (radius >= bf_radius * 1.5) & (radius <= min(det_h, det_w) * 0.46)

        clean_4dstem = torch.utils.dlpack.from_dlpack(clean_cp).contiguous()
        if clean_4dstem.device != device:
            clean_4dstem = clean_4dstem.to(device)
        del clean_cp, mean_dp_cp
        cp.get_default_memory_pool().free_all_blocks()
        clean_df = virtual_image(clean_4dstem, df_mask)

        drift_field = scan_time_drift_field(clean_4dstem.shape[:2], total_drift_px=drift_total_px)
        sim_0 = simulate_drifted_4dstem(
            clean_4dstem,
            scan_direction_degrees=0,
            drift_field_px=drift_field,
            channel_chunk=channel_chunk,
            device=device,
        )
        sim_90 = simulate_drifted_4dstem(
            clean_4dstem,
            scan_direction_degrees=90,
            drift_field_px=drift_field,
            channel_chunk=channel_chunk,
            device=device,
        )
        torch.cuda.synchronize()
        df_0 = virtual_image(sim_0["data"], df_mask)
        df_90 = virtual_image(sim_90["data"], df_mask)
        print(f"generated simulations in {time.perf_counter() - t0:.2f} s")
        print(f"known lab-frame drift: down 0.00->{drift_total_px[0]:.2f} px, right 0.00->{drift_total_px[1]:.2f} px")
        print(f"DF std clean/drift0/drift90_rot = {np.std(clean_df):.5g}/{np.std(df_0):.5g}/{np.std(np.rot90(df_90, k=-1)):.5g}")

        source_shape = clean_4dstem.shape[:2]
        nominal_0 = rotated_scan_positions(source_shape, 0)
        nominal_90 = rotated_scan_positions(source_shape, 90)
        positions_0 = sim_0["positions"]
        positions_90 = sim_90["positions"]
        clean_crop = find_valid_square_scan_crop(np.ones(source_shape, dtype=bool), save_crop)
        crop_0 = find_valid_square_scan_crop(valid_scan_position_mask(positions_0, source_shape), save_crop)
        crop_90 = find_valid_square_scan_crop(valid_scan_position_mask(positions_90, source_shape), save_crop)
        zero_offset = np.zeros_like(nominal_0, dtype=np.float32)
        print(f"theoretical largest clean square = {clean_4dstem.shape[0] - int(np.ceil(np.max(np.abs(drift_field[..., 1]))))} px")

        paths = [
            save_one_export(
                names["clean0"],
                clean_4dstem,
                clean_crop,
                nominal_0,
                zero_offset,
                "ground_truth_no_added_drift",
                scan_direction_degrees=0,
                nominal_positions=nominal_0,
            ),
            save_one_export(
                names["drift0"],
                sim_0["data"],
                crop_0,
                positions_0,
                sim_0["positions_offset_px"],
                f"image_0_known_right{right_label}_drift",
                scan_direction_degrees=0,
                nominal_positions=nominal_0,
            ),
            save_one_export(
                names["drift90"],
                sim_90["data"],
                crop_90,
                positions_90,
                sim_90["positions_offset_px"],
                f"image_1_known_right{right_label}_drift",
                scan_direction_degrees=90,
                nominal_positions=nominal_90,
            ),
        ]
        for path in paths:
            print(path)
        del clean_4dstem, sim_0, sim_90
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()

    def run_screen():
        print(f"running live screen for right{right_label}: {export_dir}")
        written = screen(
            export_dir,
            force=args.force_screen,
            verbose=True,
            run_ssb=True,
            voltage_kV=voltage_kv,
            semiangle_mrad=semiangle_mrad,
            scan_sampling_A=scan_sampling_a,
            ssb_n_trials=ssb_n_trials,
            ssb_refine=ssb_refine,
            output=sidecar_root,
            forced_rotation_deg=None,
            raise_on_failure=True,
        )
        print(f"screen output dirs written or updated: {len(written)}")
        print(f"screen cache: {screen_dir}")

    def load_screen_result(key):
        folder = screen_dir / names[key]
        phase_path = folder / "ssb_phase.npy"
        config_path = folder / "config.json"
        if not phase_path.exists() or not config_path.exists():
            raise FileNotFoundError(f"missing screen result for {key}: {folder}")
        phase = np.load(phase_path)
        config_json = json.loads(config_path.read_text())
        return phase, config_json

    def wrap_deg(angle):
        return ((float(angle) + 180.0) % 360.0) - 180.0

    def phase_to_numpy(result):
        phase = result.phase
        return (cp.asnumpy(phase) if hasattr(phase, "get") else np.asarray(phase)).astype(np.float32)

    def align_to_clean_frame(key, phase):
        return phase

    def run_locked_candidate_set(key, *, base_rotation_deg, aberrations, branch_offsets_deg):
        data, _ = live_load(str(masters[key]), verbose=False)
        ssb_input = cp.asnumpy(data)
        del data
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        candidates = []
        for branch, offset in branch_offsets_deg.items():
            rotation_deg = wrap_deg(float(base_rotation_deg) + float(offset))
            print(f"locked {key} candidate {branch}: rot={rotation_deg:.3f}, C10={aberrations.get('C10', 0):.3f}")
            result = live_ssb(
                ssb_input,
                voltage_kV=voltage_kv,
                semiangle_mrad=semiangle_mrad,
                scan_sampling_A=scan_sampling_a,
                rotation_angle_deg=rotation_deg,
                aberrations=aberrations,
                n_trials=0,
                refine=None,
                source_path=str(masters[key]),
                verbose=True,
            )
            phase_raw = phase_to_numpy(result)
            candidates.append({
                "name": key,
                "branch": branch,
                "offset_deg": float(offset),
                "rotation_angle_deg": float(result.rotation_angle_deg),
                "aberrations": {k: float(v) for k, v in result.aberrations.items()},
                "loss": float(result.loss) if result.loss is not None else None,
                "n_trials": result.n_trials,
                "phase_raw": phase_raw,
                "phase_aligned": align_to_clean_frame(key, phase_raw),
            })
            del result, phase_raw
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        best = min(candidates, key=lambda item: np.inf if item["loss"] is None else item["loss"])
        return best, candidates

    def strip_candidate_arrays(candidate):
        return {key: value for key, value in candidate.items() if key not in {"phase_raw", "phase_aligned"}}

    def zero_mean(arr):
        arr = np.asarray(arr, dtype=np.float32)
        return arr - np.nanmean(arr)

    def branch_rotation_title(label, selected, *, reference_rotation_deg):
        branch = selected["branch"]
        wrapped = float(selected["rotation_angle_deg"])
        c10 = float(selected["aberrations"].get("C10", np.nan))
        if branch == "same":
            return f"{label}\nsame clean0 calibration\nrot={wrapped:.2f} deg, C10={c10:.2f} nm"
        if branch == "plus90":
            unwrapped = float(reference_rotation_deg) + 90.0
            return f"{label}\nclean0 + 90 branch: rot={unwrapped:.2f} deg\nwrapped={wrapped:.2f} deg, C10 fixed"
        if branch == "minus90":
            unwrapped = float(reference_rotation_deg) - 90.0
            return f"{label}\nclean0 - 90 branch: rot={unwrapped:.2f} deg\nwrapped={wrapped:.2f} deg, C10 fixed"
        return f"{label}\n{branch}, rot={wrapped:.2f} deg, C10 fixed"

    def save_locked_figure(clean0_rotation_deg, clean0_aberrations, summary, phases_aligned):
        labels = {
            "clean0": "clean 0",
            "drift0": f"0 + right{right_label} drift",
            "drift90": f"90 + right{right_label} drift",
        }
        stack = np.stack([zero_mean(phases_aligned[key]) for key in ("clean0", "drift0", "drift90")])
        vmin, vmax = np.nanpercentile(stack, [1, 99])
        clean = zero_mean(phases_aligned["clean0"])
        diffs = {
            "clean0": clean * 0,
            "drift0": zero_mean(phases_aligned["drift0"]) - clean,
            "drift90": zero_mean(phases_aligned["drift90"]) - clean,
        }
        dlim = float(np.nanpercentile(np.abs(np.stack([diffs["drift0"], diffs["drift90"]])), 99))
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), constrained_layout=True)
        for col, key in enumerate(("clean0", "drift0", "drift90")):
            ax = axes[0, col]
            im = ax.imshow(zero_mean(phases_aligned[key]), cmap="magma", vmin=vmin, vmax=vmax)
            if key == "clean0":
                c10 = float(clean0_aberrations.get("C10", np.nan))
                title = f"clean 0\ncalibration reference\nrot={clean0_rotation_deg:.2f} deg, C10={c10:.2f} nm"
            else:
                title = branch_rotation_title(labels[key], summary["locked"][key]["selected"], reference_rotation_deg=clean0_rotation_deg)
            ax.set_title(title, fontsize=12)
            ax.set_axis_off()
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

            ax = axes[1, col]
            im = ax.imshow(diffs[key], cmap="coolwarm", vmin=-dlim, vmax=dlim)
            ax.set_title("phase minus clean0", fontsize=12)
            ax.set_axis_off()
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.suptitle(
            f"BTO_18 known-right{right_label}-drift SSB: clean0 calibration locked (same C10/C12/phi12)",
            fontsize=15,
        )
        plot_path = out / f"bto18_known_right{right_label}_live_ssb_locked_from_clean0.png"
        fig.savefig(plot_path, bbox_inches="tight", dpi=240)
        plt.close(fig)
        print(f"saved {plot_path}")

    def run_locked_ssb():
        print(f"running locked clean0-calibration SSB for right{right_label}")
        phases = {}
        configs = {}
        for key in ("clean0", "drift0", "drift90"):
            phases[key], configs[key] = load_screen_result(key)

        clean0_ssb = configs["clean0"]["computed"]["ssb"]
        clean0_aberrations = {key: float(value) for key, value in clean0_ssb["aberrations"].items()}
        clean0_rotation_deg = float(clean0_ssb["rotation_angle_deg"])
        locked_dir = out / "locked_from_clean0"
        locked_dir.mkdir(parents=True, exist_ok=True)
        locked_results = {}
        locked_candidates = {}
        locked_phases = {"clean0": phases["clean0"].astype(np.float32)}
        locked_phases_aligned = {"clean0": phases["clean0"].astype(np.float32)}

        best0, candidates0 = run_locked_candidate_set(
            "drift0",
            base_rotation_deg=clean0_rotation_deg,
            aberrations=clean0_aberrations,
            branch_offsets_deg={"same": 0.0},
        )
        best90, candidates90 = run_locked_candidate_set(
            "drift90",
            base_rotation_deg=clean0_rotation_deg,
            aberrations=clean0_aberrations,
            branch_offsets_deg={"same": 0.0},
        )
        locked_results["drift0"] = best0
        locked_results["drift90"] = best90
        locked_candidates["drift0"] = candidates0
        locked_candidates["drift90"] = candidates90
        for key in ("drift0", "drift90"):
            locked_phases[key] = locked_results[key]["phase_raw"]
            locked_phases_aligned[key] = locked_results[key]["phase_aligned"]

        np.save(locked_dir / "ssb_phase_clean0_reference.npy", locked_phases["clean0"])
        for key in ("drift0", "drift90"):
            for cand in locked_candidates[key]:
                np.save(locked_dir / f"ssb_phase_{key}_locked_from_clean0_{cand['branch']}_raw.npy", cand["phase_raw"])
                np.save(locked_dir / f"ssb_phase_{key}_locked_from_clean0_{cand['branch']}_aligned.npy", cand["phase_aligned"])
            np.save(locked_dir / f"ssb_phase_{key}_locked_from_clean0_raw.npy", locked_phases[key])
            np.save(locked_dir / f"ssb_phase_{key}_locked_from_clean0_aligned.npy", locked_phases_aligned[key])

        summary = {
            "source": "clean0_locked_microscope_calibration",
            "drift_total_px_down_right": [0.0, right_px],
            "clean0_rotation_angle_deg": clean0_rotation_deg,
            "clean0_aberrations": clean0_aberrations,
            "locked": {
                key: {
                    "selected_branch": locked_results[key]["branch"],
                    "selected": strip_candidate_arrays(locked_results[key]),
                    "candidates": [strip_candidate_arrays(cand) for cand in locked_candidates[key]],
                }
                for key in ("drift0", "drift90")
            },
        }
        (locked_dir / "locked_from_clean0_summary.json").write_text(json.dumps(summary, indent=2))
        np.savez_compressed(
            out / f"bto18_known_right{right_label}_live_ssb_locked_phases.npz",
            clean0=locked_phases_aligned["clean0"],
            drift0=locked_phases_aligned["drift0"],
            drift90=locked_phases_aligned["drift90"],
            summary_json=json.dumps(summary),
        )
        save_locked_figure(clean0_rotation_deg, clean0_aberrations, summary, locked_phases_aligned)
        print(json.dumps(summary["locked"], indent=2))

    if args.skip_generate:
        print("skipping generation")
    elif all(path.exists() for path in masters.values()) and not args.force_generate:
        print(f"all exports already exist for right{right_label}, skipping generation")
    else:
        generate_exports()

    if args.skip_screen:
        print("skipping live screen")
    else:
        run_screen()

    if args.skip_locked:
        print("skipping locked SSB")
    else:
        run_locked_ssb()

    print(f"done right{right_label}")
    print(f"exports: {export_dir}")
    print(f"outputs: {out}")


if __name__ == "__main__":
    main()
