#!/usr/bin/env python3
import argparse
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/.mplconfig")

import numpy as np
from scipy.ndimage import gaussian_filter

import quantem as em
from quantem.core import config as quantem_config

SIZES = {
    1: "128x128",
    2: "256x256",
    4: "512x512",
    8: "1024x1024",
}

OPTIMIZED_ADAM_BASE_CONFIG = {
    "num_iterations": 2,
    "adam_steps": 15,
    "lr": 0.02,
    "pytorch_reference_mode": "leave_one_out",
    "pytorch_normalize_loss": False,
    "pytorch_row_stride": 1,
    "pytorch_multiscale": False,
    "pytorch_fast_schedule": False,
    "pytorch_learn_translation": False,
    "translation_interval": 0,
    "translation_upsample_factor": 4,
    "translation_downsample_factor": 1,
}

JOINT_SWEEP_BASE_CONFIG = {
    "num_iterations": 8,
    "adam_steps": 120,
    "lr": 0.01,
    "regularization_sigma_px": 8.0,
    "regularization_update_step_size": 1.0,
    "pytorch_reference_mode": "leave_one_out",
    "pytorch_normalize_loss": False,
    "pytorch_row_stride": 1,
    "pytorch_fast_schedule": False,
    "pytorch_multiscale": True,
    "pytorch_refine_steps": 40,
    "pytorch_refine_lr_scale": 0.5,
    "pytorch_refine_normalize_loss": False,
}

JOINT_SWEEP_VARIANTS = [
    ("rmse_base", {}),
    (
        "rmse_deep",
        {
            "num_iterations": 10,
            "adam_steps": 200,
            "lr": 0.008,
            "pytorch_refine_steps": 60,
        },
    ),
    (
        "rmse_low_reg",
        {
            "regularization_sigma_px": 0.0,
            "regularization_update_step_size": None,
        },
    ),
    (
        "rmse_high_reg",
        {
            "regularization_sigma_px": 16.0,
            "regularization_update_step_size": 0.6,
        },
    ),
    (
        "rmse_no_multiscale",
        {
            "pytorch_multiscale": False,
            "pytorch_refine_steps": 80,
        },
    ),
    (
        "rmse_mean_ref",
        {
            "pytorch_reference_mode": "mean",
        },
    ),
    (
        "rmse_penalized",
        {
            "pytorch_translation_penalty": 1e-4,
            "pytorch_affine_penalty": 1e-4,
        },
    ),
]


def parse_scales(text: str) -> list[int]:
    scales = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        scale = int(token)
        if scale <= 0:
            raise ValueError("Scale factors must be positive integers.")
        scales.append(scale)
    if not scales:
        raise ValueError("At least one scale must be provided.")
    return scales


def parse_backends(text: str) -> list[str]:
    mapping = {
        "scipy": "scipy",
        "pytorch": "pytorch",
        "torch": "pytorch",
        "optimized": "optimized",
        "optimized_adam": "optimized_adam",
        "opt_adam": "optimized_adam",
        "adam_opt": "optimized_adam",
        "opt": "optimized",
        "joint": "joint",
        "pytorch_joint": "joint",
        "optimized_joint": "joint",
        "opt_joint": "joint",
        "joint_opt": "joint",
    }
    backends = []
    for token in text.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token not in mapping:
            raise ValueError(f"Unknown backend '{token}'.")
        backend = mapping[token]
        if backend not in backends:
            backends.append(backend)
    if not backends:
        raise ValueError("At least one backend must be provided.")
    return backends


def set_quantem_device(device: str, strict: bool = False) -> str:
    if device.isdigit():
        device_value = int(device)
    else:
        device_value = device
    try:
        quantem_config.set_device(device_value)
    except Exception as exc:
        if strict or str(device).lower() == "cpu":
            raise
        print(f"Warning: unable to set device to '{device}' ({exc}). Falling back to CPU.")
        quantem_config.set_device("cpu")
    return quantem_config.get_device()


def get_optimized_adam_config(min_dim: int) -> dict[str, object]:
    config = dict(OPTIMIZED_ADAM_BASE_CONFIG)
    if min_dim <= 160:
        config.update(
            {
                "num_iterations": 8,
                "regularization_sigma_px": 4.0,
                "adam_steps": 10,
                "lr": 0.04,
                "translation_interval": 0,
                "translation_upsample_factor": 4,
                "translation_downsample_factor": 1,
            }
        )
    elif min_dim <= 768 and min_dim > 320:
        config.update(
            {
                "translation_upsample_factor": 4,
                "translation_downsample_factor": 2,
            }
        )
    return config


def get_joint_sweep_configs() -> list[tuple[str, dict[str, object]]]:
    configs = []
    for name, overrides in JOINT_SWEEP_VARIANTS:
        cfg = dict(JOINT_SWEEP_BASE_CONFIG)
        cfg.update(overrides)
        if cfg.get("pytorch_multiscale") and "pytorch_multiscale_scales" not in cfg:
            cfg["pytorch_multiscale_scales"] = None
        configs.append((name, cfg))
    return configs


def generate_synthetic_data(scale: int = 1, seed: int = 42, constant_jitter: bool = True):
    np.random.seed(seed)
    shape = (200 * scale, 200 * scale)
    xa, ya = np.meshgrid(
        np.arange(-shape[0] / 2, shape[0] / 2),
        np.arange(-shape[0] / 2, shape[0] / 2),
        indexing="ij",
    )
    im = (np.mod(np.abs(xa) + np.abs(ya), 16 * scale) < 8 * scale).astype("float")
    im[np.logical_and(xa > 0, ya > 0)] += 0.5
    im[np.maximum(np.abs(xa), np.abs(ya)) < 20 * scale] = 2
    im = gaussian_filter(im, sigma=0.667 * scale)
    scan_size = 128 * scale
    u = np.arange(scan_size)
    x_drift = u * 0.001 if constant_jitter else u * 0.001 * scale
    y_drift = u * 0.1 if constant_jitter else u * 0.1 * scale
    jitter_mag = 0.5 if constant_jitter else 0.5 * scale
    jitter0 = np.random.randn(2, scan_size) * jitter_mag
    jitter1 = np.random.randn(2, scan_size) * jitter_mag
    im0, im1 = np.zeros((scan_size, scan_size)), np.zeros((scan_size, scan_size))
    for a0 in range(scan_size):
        x0, y0 = (
            40 * scale + a0 + x_drift[a0] + jitter0[0, a0],
            30 * scale + y_drift[a0] + jitter0[1, a0],
        )
        x, y = np.clip(x0 + u * 0, 0, shape[0] - 2), np.clip(y0 + u * 1, 0, shape[1] - 2)
        xf, yf = np.floor(x).astype(int), np.floor(y).astype(int)
        dx, dy = x - np.floor(x), y - np.floor(y)
        im0[a0, :] = (
            im[xf, yf] * (1 - dx) * (1 - dy)
            + im[xf + 1, yf] * dx * (1 - dy)
            + im[xf, yf + 1] * (1 - dx) * dy
            + im[xf + 1, yf + 1] * dx * dy
        )
        x0, y0 = (
            170 * scale + x_drift[a0] + jitter1[0, a0],
            30 * scale + a0 + y_drift[a0] + jitter1[1, a0],
        )
        x, y = np.clip(x0 - u * 1, 0, shape[0] - 2), np.clip(y0 + u * 0, 0, shape[1] - 2)
        xf, yf = np.floor(x).astype(int), np.floor(y).astype(int)
        dx, dy = x - np.floor(x), y - np.floor(y)
        im1[a0, :] = (
            im[xf, yf] * (1 - dx) * (1 - dy)
            + im[xf + 1, yf] * dx * (1 - dy)
            + im[xf, yf + 1] * (1 - dx) * dy
            + im[xf + 1, yf + 1] * dx * dy
        )
    gt_crop = im[40 * scale : 40 * scale + scan_size, 30 * scale : 30 * scale + scan_size]
    return {
        "im0": im0,
        "im1": im1,
        "ground_truth": gt_crop,
        "scan_size": scan_size,
    }


def compute_rmse(img1: np.ndarray, img2: np.ndarray) -> float:
    h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
    diff = img1[:h, :w] - img2[:h, :w]
    valid = ~np.isnan(diff)
    return np.sqrt(np.mean(diff[valid] ** 2)) if valid.any() else float("inf")


def align_and_crop(corrected: np.ndarray, gt: np.ndarray):
    h, w = gt.shape
    ch, cw = corrected.shape
    margin = min(h, w) // 4
    g_center = gt[margin:-margin, margin:-margin]
    best_score, best_pos = -np.inf, (0, 0)
    pad = (ch - h) // 2
    search_range = max(20, int(0.08 * min(h, w)))
    step = max(2, search_range // 20)
    for dy in range(-search_range, search_range + 1, step):
        for dx in range(-search_range, search_range + 1, step):
            y, x = pad + dy, pad + dx
            if y < 0 or x < 0 or y + h > ch or x + w > cw:
                continue
            crop = corrected[y : y + h, x : x + w]
            c_center = crop[margin:-margin, margin:-margin]
            score = np.sum((c_center - c_center.mean()) * (g_center - g_center.mean()))
            if score > best_score:
                best_score, best_pos = score, (y, x)
    y, x = best_pos
    return corrected[y : y + h, x : x + w], (y, x)


def prepare_state(data, align_affine: bool = True):
    drift = em.imaging.DriftCorrection.from_data(
        images=[data["im0"], data["im1"]],
        scan_direction_degrees=[0, 90],
    )
    drift.preprocess(
        pad_fraction=0.25,
        pad_value="median",
        kde_sigma=0.5,
        number_knots=1,
        show_merged=False,
    )
    if align_affine:
        drift.align_affine(step=0.02, num_tests=11, show_merged=False)
    return {
        "knots": [k.copy() for k in drift.knots],
        "images_warped": drift.images_warped.array.copy(),
        "weights_warped": drift.weights_warped.array.copy(),
    }


def run_backend(
    data,
    state,
    backend: str,
    schedule: str,
    config_override: dict[str, object] | None = None,
):
    drift = em.imaging.DriftCorrection.from_data(
        images=[data["im0"], data["im1"]],
        scan_direction_degrees=[0, 90],
    )
    drift.preprocess(
        pad_fraction=0.25,
        pad_value="median",
        kde_sigma=0.5,
        number_knots=1,
        show_merged=False,
    )
    drift.knots = [k.copy() for k in state["knots"]]
    drift.images_warped.array[:] = state["images_warped"]
    drift.weights_warped.array[:] = state["weights_warped"]

    t0 = time.perf_counter()
    if backend == "scipy":
        drift.align_nonrigid(
            backend="scipy",
            num_iterations=8,
            regularization_sigma_px=16.0,
            show_merged=False,
        )
    elif backend == "pytorch":
        drift.align_nonrigid(
            backend="pytorch",
            num_iterations=8,
            regularization_sigma_px=16.0,
            show_merged=False,
        )
    elif backend == "optimized":
        drift.align_nonrigid(
            backend="pytorch_optimized",
            pytorch_schedule=schedule,
            show_merged=False,
        )
    elif backend == "optimized_adam":
        min_dim = min(data["im0"].shape)
        drift.align_nonrigid(
            backend="pytorch_optimized",
            show_merged=False,
            **get_optimized_adam_config(min_dim),
        )
    elif backend == "joint":
        kwargs = {
            "backend": "pytorch_joint",
            "show_merged": False,
        }
        if config_override:
            kwargs.update(config_override)
        else:
            kwargs["pytorch_schedule"] = schedule
        drift.align_nonrigid(**kwargs)
    else:
        raise ValueError(f"Unknown backend '{backend}'.")
    elapsed = time.perf_counter() - t0

    image = drift.generate_corrected_image(
        upsample_factor=1,
        kde_sigma=0.5,
        show_image=False,
    )
    aligned, _ = align_and_crop(image.array, data["ground_truth"])
    rmse = compute_rmse(aligned, data["ground_truth"])
    return elapsed, rmse


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    lines = []
    header_line = " ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append("-" * len(header_line))
    for row in rows:
        line = " ".join(row[i].ljust(widths[i]) for i in range(len(headers)))
        lines.append(line)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark nonrigid drift correction backends on synthetic data."
    )
    parser.add_argument(
        "--scales",
        default="1,2,4,8",
        help="Comma-separated scale factors (1=128x128).",
    )
    parser.add_argument(
        "--backends",
        default="scipy,pytorch,optimized",
        help="Comma-separated list: scipy,pytorch,optimized,optimized_adam,joint.",
    )
    parser.add_argument(
        "--joint-sweep",
        action="store_true",
        help="Run an RMSE-focused joint Adam sweep (ignores --backends).",
    )
    parser.add_argument(
        "--sweep-top",
        type=int,
        default=3,
        help="Number of joint sweep configs to display per size.",
    )
    parser.add_argument(
        "--no-init-affine",
        action="store_true",
        help="Skip affine alignment before nonrigid optimization.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Torch device for PyTorch backends (mps, cpu, cuda:0, gpu).",
    )
    parser.add_argument(
        "--strict-device",
        action="store_true",
        help="Fail if the requested device is unavailable (defaults to strict when device=mps).",
    )
    parser.add_argument(
        "--schedule",
        default="pytroch_optmized",
        help=("Schedule name for optimized backend (pytroch_optmized, pytroch_joint_rmse)."),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data.",
    )
    args = parser.parse_args()

    strict = args.strict_device or args.device.lower() == "mps"
    device = set_quantem_device(args.device, strict=strict)
    print(f"PyTorch device: {device}")

    scales = parse_scales(args.scales)
    if args.joint_sweep:
        backends = ["scipy"]
    else:
        backends = parse_backends(args.backends)
        if "scipy" not in backends:
            backends.insert(0, "scipy")
            print("Note: adding SciPy baseline for comparison.")

    results = []
    for scale in scales:
        name = SIZES.get(scale, f"{128 * scale}x{128 * scale}")
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        data = generate_synthetic_data(scale=scale, seed=args.seed, constant_jitter=True)
        state = prepare_state(data, align_affine=not args.no_init_affine)
        row = {"size": name}
        for backend in backends:
            print(f"Running {backend}...", end=" ", flush=True)
            elapsed, rmse = run_backend(data, state, backend, args.schedule)
            row[f"{backend}_time"] = elapsed
            row[f"{backend}_rmse"] = rmse
            print(f"Done! ({elapsed:.2f}s, RMSE={rmse:.4f})")
        if args.joint_sweep:
            sweep_results = []
            configs = get_joint_sweep_configs()
            for name_cfg, cfg in configs:
                print(f"Running joint sweep {name_cfg}...", end=" ", flush=True)
                elapsed, rmse = run_backend(
                    data,
                    state,
                    "joint",
                    args.schedule,
                    config_override=cfg,
                )
                sweep_results.append({"name": name_cfg, "time": elapsed, "rmse": rmse})
                print(f"Done! ({elapsed:.2f}s, RMSE={rmse:.4f})")
            sweep_results.sort(key=lambda item: item["rmse"])
            top_n = max(1, min(args.sweep_top, len(sweep_results)))
            best = sweep_results[0]
            row["joint_time"] = best["time"]
            row["joint_rmse"] = best["rmse"]
            row["joint_best_name"] = best["name"]
            print("\nTop joint configs by RMSE:")
            for item in sweep_results[:top_n]:
                print(f"  {item['name']}: {item['rmse']:.4f} RMSE, {item['time']:.2f}s")
        results.append(row)

    headers = ["Size"]
    if "scipy" in backends:
        headers += ["SciPy Time", "SciPy RMSE"]
    if "pytorch" in backends:
        headers += ["PyTorch Time", "PyTorch RMSE"]
    if "optimized" in backends:
        headers += ["Opt Time", "Opt RMSE"]
    if "optimized_adam" in backends:
        headers += ["Opt Adam Time", "Opt Adam RMSE"]
    if "joint" in backends or args.joint_sweep:
        headers += ["Joint Time", "Joint RMSE"]
    if "scipy" in backends and "pytorch" in backends:
        headers += ["Speedup Py"]
    if "scipy" in backends and "optimized" in backends:
        headers += ["Speedup Opt"]
    if "scipy" in backends and ("joint" in backends or args.joint_sweep):
        headers += ["Speedup Joint"]
    if "pytorch" in backends and "optimized" in backends:
        headers += ["Opt vs Py"]
    if "pytorch" in backends and "joint" in backends:
        headers += ["Joint vs Py"]

    rows = []
    for row in results:
        values = [row["size"]]
        t_scipy = row.get("scipy_time")
        t_pytorch = row.get("pytorch_time")
        t_opt = row.get("optimized_time")
        t_opt_adam = row.get("optimized_adam_time")
        t_joint = row.get("joint_time")
        if "scipy" in backends:
            values += [f"{t_scipy:.2f}s", f"{row['scipy_rmse']:.4f}"]
        if "pytorch" in backends:
            values += [f"{t_pytorch:.2f}s", f"{row['pytorch_rmse']:.4f}"]
        if "optimized" in backends:
            values += [f"{t_opt:.2f}s", f"{row['optimized_rmse']:.4f}"]
        if "optimized_adam" in backends:
            values += [f"{t_opt_adam:.2f}s", f"{row['optimized_adam_rmse']:.4f}"]
        if "joint" in backends or args.joint_sweep:
            values += [f"{t_joint:.2f}s", f"{row['joint_rmse']:.4f}"]
        if "scipy" in backends and "pytorch" in backends:
            values += [f"{t_scipy / t_pytorch:.1f}x"]
        if "scipy" in backends and "optimized" in backends:
            values += [f"{t_scipy / t_opt:.1f}x"]
        if "scipy" in backends and ("joint" in backends or args.joint_sweep):
            values += [f"{t_scipy / t_joint:.1f}x"]
        if "pytorch" in backends and "optimized" in backends:
            values += [f"{t_pytorch / t_opt:.1f}x"]
        if "pytorch" in backends and "joint" in backends:
            values += [f"{t_pytorch / t_joint:.1f}x"]
        rows.append(values)

    print("\n" + format_table(headers, rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
