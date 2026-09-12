"""Capture a new small Torch MPS oracle; never overwrite an existing fixture."""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur

from quantem.diffraction.maped import tukey_torch

path = Path(sys.argv[1])
if path.exists():
    raise FileExistsError("Frozen fixtures must not be overwritten.")
if not torch.backends.mps.is_available():
    raise RuntimeError("This oracle requires a physical MPS device.")


def image(rows, columns):
    indices = torch.arange(rows * columns, device="mps").reshape(rows, columns)
    return ((indices * 37) % 251 - 125).float() * 0.03125


def observe(values):
    indices = sorted(
        {0, 1, values.numel() - 1, values.numel() - 2}
        | {i * (values.numel() - 1) // 63 for i in range(64)}
    )
    selected = values.flatten()[indices]
    if values.is_complex():
        selected = torch.view_as_real(selected)
    return {"indices": indices, "values": selected.cpu().tolist()}


fixture = {"torch": torch.__version__, "torch_revision": torch.version.git_version}
raw = image(512, 512)
kernel = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device="mps")
padded = F.pad(raw[None, None], (1, 1, 1, 1), mode="reflect")
row = F.conv2d(padded, kernel[None, None])[0]
column = F.conv2d(padded, kernel.T[None, None])[0]
fixture["gradient"] = {}
for sigma in (0.25, 1.0, 2.0, 4.0):
    blur = GaussianBlur([2 * int(2 * sigma) + 1] * 2, [sigma] * 2)
    fixture["gradient"][str(sigma)] = observe((blur(row) ** 2 + blur(column) ** 2).sqrt())
fixture["fft"] = observe(torch.fft.fft2(image(520, 520)))
fixture["windows"] = {}
for edge in (0.0, 2.0, 16.0, 96.0):
    window = tukey_torch(192, 2 * edge / 192, device="mps")
    fixture["windows"][str(edge)] = observe(window[:, None] * window[None, :])
fixture["centered"] = observe(raw - torch.stack([raw] * 7).sum((1, 2))[0] / raw.numel())
path.write_text(json.dumps(fixture, indent=2) + "\n")
