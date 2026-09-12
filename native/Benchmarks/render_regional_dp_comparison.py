"""Render native-exported DPs; scientific differences and metrics use Torch MPS.

Usage: python render_regional_dp_comparison.py INPUT_DIRECTORY OUTPUT_DIRECTORY
CPU work is limited to reading exported bytes and rendering figures.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
if not torch.backends.mps.is_available():
    raise RuntimeError("This comparison requires MPS; run it on the qualification Mac.")
args.output.mkdir(parents=True, exist_ok=True)
rows = [4, 252, 508]
names = ["float32", "global", "regional"]
data = {}
metrics = []
for row in rows:
    for name in names:
        raw = bytearray((args.source / f"row-{row}-col-256-{name}.f32").read_bytes())
        data[row, name] = torch.frombuffer(raw, dtype=torch.float32).reshape(192, 192).to("mps")
    record = {"scan_row": row, "scan_column": 256}
    for name in names[1:]:
        error = data[row, name] - data[row, "float32"]
        data[row, name + "_error"] = error
        record[name] = {"rmse": error.square().mean().sqrt().item(),
                        "max_abs_error": error.abs().amax().item()}
    metrics.append(record)
vmax = torch.stack([data[row, name].amax() for row in rows for name in names]).amax().item()
error_limit = 0.014
fig, axes = plt.subplots(3, 5, figsize=(14, 8.6), layout="constrained")
intensity_cmap = plt.get_cmap("magma").copy()
intensity_cmap.set_bad(intensity_cmap(0))
intensity_cmap.set_under(intensity_cmap(0))
labels = ["Float32 reference", "Global scaled uint16", "Regional scaled uint16",
          "Global − reference", "Regional − reference"]
for i, row in enumerate(rows):
    for j, name in enumerate(names + ["global_error", "regional_error"]):
        ax = axes[i, j]
        error = j >= 3
        plot = ax.imshow(data[row, name].cpu().numpy(), origin="upper",
            cmap="RdBu_r" if error else intensity_cmap,
            norm=Normalize(-error_limit, error_limit) if error else LogNorm(vmin=0.01, vmax=vmax),
            interpolation="nearest")
        ax.set_xticks([0, 96, 191])
        ax.set_yticks([0, 96, 191])
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.set_title(labels[j], fontsize=11)
        if j == 0:
            ax.set_ylabel(f"Scan ({row}, 256)\nDetector row", fontsize=10)
        if i == 2:
            ax.set_xlabel("Detector column", fontsize=9)
        if error:
            key = "global" if j == 3 else "regional"
            ax.text(0.03, 0.03, f"RMSE {metrics[i][key]['rmse']:.5f}", transform=ax.transAxes,
                    fontsize=9, bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"))
        if j == 2:
            intensity_plot = plot
        if j == 4:
            error_plot = plot
fig.colorbar(intensity_plot, ax=axes[:, :3], location="bottom", shrink=0.7,
             label="Restored intensity • shared logarithmic scale (zeros below display floor)")
fig.colorbar(error_plot, ax=axes[:, 3:], location="bottom", shrink=0.85,
             label="Signed storage error • shared linear scale")
fig.suptitle("Same MAPED float32 merge, two uint16 storage choices", fontsize=16)
fig.savefig(args.output / "dp-comparison.png", dpi=160)
plt.close(fig)
(args.output / "dp-metrics.json").write_text(json.dumps({"backend": "torch-mps",
    "coordinates": "zero-based (scan row, scan column)", "display_min": 0.01,
    "display_max": vmax, "error_display_limit": error_limit, "patterns": metrics}, indent=2) + "\n")
print(json.dumps(metrics))
