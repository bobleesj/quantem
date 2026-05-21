"""Build QA artifacts for the BTO_18 crop400 image-1/90-degree drift run."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


TRIALS_DIR = Path("/home/owner/data/dasol/20260415_BTOSTO/quantem/ptycho/BTO_18/trials")
SUMMARY_PATH = TRIALS_DIR / "bto18_crop400_drift90_three_way_summary.json"
QA_PATH = TRIALS_DIR / "bto18_crop400_drift90_three_way_QA.png"
README_PATH = TRIALS_DIR / "bto18_crop400_drift90_three_way_README.md"
REFERENCE_TRIAL = TRIALS_DIR / "2014_det96_scan400_s6_t18_p8_it50_pure_phase_decay"

CASE_ORDER = (
    "image1_clean_raster_global",
    "image1_raster_global",
    "image1_corrected_global",
)
DISPLAY_LABELS = {
    "image1_clean_raster_global": "1. clean 90, image1 raster",
    "image1_raster_global": "2. drift 90, image1 raster",
    "image1_corrected_global": "3. drift 90, corrected positions",
}


def load_rows() -> list[dict]:
    rows = json.loads(SUMMARY_PATH.read_text())
    by_mode = {row["position_source"]: row for row in rows}
    missing = [mode for mode in CASE_ORDER if mode not in by_mode]
    if missing:
        raise RuntimeError(f"missing cases in {SUMMARY_PATH}: {missing}")
    ordered = [by_mode[mode] for mode in CASE_ORDER]
    for row in ordered:
        trial_dir = Path(row["trial_dir"])
        required = ("config.json", "loss.npy", "obj_phase.npy", "NOTE.md", "position_summary.json")
        missing_files = [name for name in required if not (trial_dir / name).exists()]
        if missing_files:
            raise FileNotFoundError(f"{trial_dir} missing {missing_files}")
        if row["status"] != "finished":
            raise RuntimeError(f"{trial_dir} status is {row['status']!r}")
        if int(row["iters"]) != 10:
            raise RuntimeError(f"{trial_dir} has {row['iters']} iterations, expected 10")
        if row["scan_shape"] != [400, 400] or row["det_size_px"] != 96:
            raise RuntimeError(f"{trial_dir} has unexpected scan/det shape")
        if abs(float(row["rotation_deg"]) - 158.9) > 1e-6:
            raise RuntimeError(f"{trial_dir} rotation is not locked to common-frame 158.9 deg")
        if abs(float(row["defocus_A"]) - 781.0) > 1e-6:
            raise RuntimeError(f"{trial_dir} defocus is not locked to 781 A")

    if ordered[1]["dp_source"] != ordered[2]["dp_source"]:
        raise RuntimeError("case 2 and case 3 do not use the same drifted image1 DP source")
    return ordered


def projected_phase(trial_dir: Path) -> np.ndarray:
    phase = np.load(trial_dir / "obj_phase.npy", mmap_mode="r")
    projection = np.asarray(phase, dtype=np.float32).sum(axis=0)
    projection = projection - np.nanmean(projection)
    return projection


def center_crop(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    row0 = (arr.shape[0] - shape[0]) // 2
    col0 = (arr.shape[1] - shape[1]) // 2
    return arr[row0 : row0 + shape[0], col0 : col0 + shape[1]]


def phase_sign(reference: np.ndarray, candidate: np.ndarray, *, sigma: float = 4.0) -> tuple[int, float]:
    shape = (min(reference.shape[0], candidate.shape[0]), min(reference.shape[1], candidate.shape[1]))
    ref = center_crop(reference, shape)
    cand = center_crop(candidate, shape)
    ref = gaussian_filter(ref, sigma=sigma)
    cand = gaussian_filter(cand, sigma=sigma)
    ref = (ref - np.nanmean(ref)) / (np.nanstd(ref) + 1e-12)
    cand = (cand - np.nanmean(cand)) / (np.nanstd(cand) + 1e-12)
    corr = float(np.nanmean(ref * cand))
    return (1 if corr >= 0 else -1), corr


def robust_limits(images: list[np.ndarray], percentiles: tuple[float, float]) -> tuple[float, float]:
    stack = np.stack([image[np.isfinite(image)] for image in images])
    lo, hi = np.nanpercentile(stack, percentiles)
    return float(lo), float(hi)


def make_qa(rows: list[dict]) -> None:
    phases = [projected_phase(Path(row["trial_dir"])) for row in rows]
    sign_reference = projected_phase(REFERENCE_TRIAL) if (REFERENCE_TRIAL / "obj_phase.npy").exists() else phases[0]
    signs_and_corrs = [phase_sign(sign_reference, phase) for phase in phases]
    signs = [sign for sign, _corr in signs_and_corrs]
    phases = [sign * phase for sign, phase in zip(signs, phases)]
    common_shape = (min(phase.shape[0] for phase in phases), min(phase.shape[1] for phase in phases))
    crops = [center_crop(phase, common_shape) for phase in phases]
    clean = crops[0]
    diffs = [crop - clean for crop in crops]

    vmin, vmax = robust_limits(crops, (1, 99))
    diff_limit = float(np.nanpercentile(np.abs(np.stack(diffs[1:])), 99))
    fig, axes = plt.subplots(3, 3, figsize=(14, 13), constrained_layout=True)

    for col, (row, crop) in enumerate(zip(rows, crops)):
        loss = np.load(Path(row["trial_dir"]) / "loss.npy")
        mode = row["position_source"]

        ax = axes[0, col]
        im = ax.imshow(crop, cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(
            f"{DISPLAY_LABELS[mode]}\n"
            f"loss {loss[0] / 1e9:.3f} -> {loss[-1] / 1e9:.3f}e9, sign {signs[col]:+d}"
        )
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        ax = axes[1, col]
        im = ax.imshow(diffs[col], cmap="coolwarm", vmin=-diff_limit, vmax=diff_limit)
        ax.set_title("projected phase minus clean90")
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        ax = axes[2, col]
        ax.plot(np.arange(1, len(loss) + 1), loss / 1e9, marker="o", lw=1.8)
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss (1e9)")
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"positions: {row['position_source']}\n"
            f"injected: {row['positions_injected']}"
        )

    fig.suptitle(
        "BTO_18 crop400 image-1/90-degree drift three-way ptychography QA\n"
        "all runs use explicit sample-frame positions and the common base rotation\n"
        f"phase signs aligned to {REFERENCE_TRIAL.name if REFERENCE_TRIAL.exists() else 'case 1'}",
        fontsize=15,
    )
    fig.savefig(QA_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    for row, (sign, corr) in zip(rows, signs_and_corrs):
        row["phase_display_sign"] = int(sign)
        row["phase_sign_reference"] = str(REFERENCE_TRIAL if REFERENCE_TRIAL.exists() else Path(rows[0]["trial_dir"]))
        row["phase_sign_correlation_smoothed"] = float(corr)


def write_readme(rows: list[dict]) -> None:
    lines = [
        "# BTO_18 crop400 image-1/90-degree drift three-way ptychography",
        "",
        "All three runs use explicit probe positions in the shared physical sample frame and lock the common clean0/base rotation to 158.9 deg.",
        "The 90-degree scan geometry is carried by the image1 probe-position arrays, not by adding +90 deg to the reconstruction rotation.",
        "The recipe follows the 2014 condition but with 10 iterations: S=6, slice thickness 18 A, P=8, object/probe LR 0.2, batch 4096.",
        "",
        "| Case | Trial | DP source | Position source | Phase display sign | Final loss |",
        "|---|---|---|---|---:|---:|",
    ]
    for index, row in enumerate(rows, start=1):
        trial = Path(row["trial_dir"]).name
        dp_name = " + ".join(Path(path).name for path in row["dp_source"])
        lines.append(
            f"| {index} | `{trial}` | `{dp_name}` | `{row['position_source']}` | "
            f"{row.get('phase_display_sign', 1):+d} | {row['loss_final']:.6g} |"
        )
    lines.extend(
        [
            "",
            "Critical control:",
            "",
            "- Case 1 reindexes the clean0 crop into raw 90-degree scan order in memory.",
            "- Case 2 and case 3 use the exact same drifted image1 diffraction patterns.",
            "- Case 2 uses the image1 no-drift raster positions, intentionally uncorrected for drift.",
            "- Case 3 injects the known image1 drift-corrected probe positions.",
            "- Detector pixels are not rolled, warped, or otherwise changed between case 2 and case 3.",
            "- Ptychography position correction does not bilinear-interpolate DPs. The fused kernel uses `scan_positions_px` as rounded object-patch centers plus fractional Fourier phase ramps on the probe.",
            "- Do not combine these global/sample-frame positions with `clean0 + 90`; that would double-count the 90-degree geometry.",
            "",
            f"Summary JSON: `{SUMMARY_PATH}`",
            f"QA figure: `{QA_PATH}`",
        ]
    )
    README_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = load_rows()
    make_qa(rows)
    write_readme(rows)
    print(f"validated {len(rows)} trials")
    print(f"qa: {QA_PATH}")
    print(f"readme: {README_PATH}")


if __name__ == "__main__":
    main()
