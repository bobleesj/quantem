"""Inspect real-data first-iteration rounding without changing frozen gates."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur

from quantem.diffraction.maped import (
    cross_correlation_shift_torch,
    shift_images_torch,
    tukey_torch,
)

prefix, directory = sys.argv[1], Path(sys.argv[2])


def read(path, shape, complex=False):
    values = np.fromfile(path, dtype=np.complex64 if complex else np.float32).reshape(shape)
    return torch.from_numpy(values).to("mps")


def metrics(a, b):
    error = (a - b).abs()
    return {
        "max": error.max().item(),
        "rmse": error.square().mean().sqrt().item(),
        "equal": (a == b).float().mean().item(),
    }


settings = json.loads((directory / "settings.json").read_text())
pad = settings["padding"]
sigma = settings["sigma"]
size = 512 + 2 * pad
images = torch.stack([read(prefix + f".im_bf-{i}.f32", (512, 512)) for i in range(7)])
kernel = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device="mps")
window = torch.ones((512, 512), device="mps")
if settings["hann"]:
    window = (
        torch.hann_window(512, device="mps")[:, None]
        * torch.hann_window(512, device="mps")[None, :]
    )
window = F.pad(window, (pad, pad, pad, pad))
print("window", metrics(read(directory / "window.f32", (size, size)), window))
spectra = []
for i in range(7):
    padded = F.pad(images[i][None, None], (1, 1, 1, 1), mode="reflect")
    row = F.conv2d(padded, kernel[None, None])[0, 0]
    col = F.conv2d(padded, kernel.T[None, None])[0, 0]
    blur = GaussianBlur([2 * int(2 * sigma) + 1] * 2, [sigma, sigma])
    gradient = (blur(row[None]) ** 2 + blur(col[None]) ** 2).sqrt()[0]
    shifted = shift_images_torch(
        F.pad(gradient, (pad, pad, pad, pad)), torch.zeros(2, device="mps")
    )
    centered = (shifted - (shifted * window).sum() / window.sum()) * window
    spectrum = torch.fft.fft2(centered)
    spectra.append(spectrum)
    for name, item in [
        ("gradient", gradient),
        ("shifted", shifted),
        ("centered", centered),
        ("spectrum", spectrum),
    ]:
        actual = read(directory / f"{name}-{i}.f32", item.shape, item.is_complex())
        print(i, name, metrics(actual, item))
stack = torch.stack(spectra)
reference = stack.mean(0)
sequential = torch.zeros_like(reference)
for item in stack:
    sequential += item
sequential /= 7
incremental = stack[0]
for i in range(1, 7):
    incremental = incremental * (i / (i + 1)) + stack[i] / (i + 1)
actual = read(directory / "reference.f32", reference.shape, True)
print("reference native", metrics(actual, reference))
print("reference sequential", metrics(sequential, reference))
print("reference incremental", metrics(incremental, reference))
for i in range(1, 7):
    shift = cross_correlation_shift_torch(reference, stack[i], 100, fft_input=True)
    actual = read(directory / f"shift-{i}.f32", (2,))
    print("shift", i, actual.cpu().tolist(), shift.cpu().tolist(), metrics(actual, shift))
print("Correlation using identical native spectra:")
native_stack = torch.stack(
    [read(directory / f"spectrum-{i}.f32", (size, size), True) for i in range(7)]
)
native_ref = read(directory / "reference.f32", (size, size), True)
for i in range(1, 7):
    calculated = cross_correlation_shift_torch(native_ref, native_stack[i], 100, fft_input=True)
    stored = read(directory / f"shift-{i}.f32", (2,))
    print(i, calculated.cpu().tolist(), metrics(stored, calculated))
print("Stage comparison using identical inputs:")
for i in range(7):
    g = read(directory / f"gradient-{i}.f32", (512, 512))
    shifted = shift_images_torch(F.pad(g, (pad, pad, pad, pad)), torch.zeros(2, device="mps"))
    nshift = read(directory / f"shifted-{i}.f32", (size, size))
    center = (nshift - (nshift * window).sum() / window.sum()) * window
    ncenter = read(directory / f"centered-{i}.f32", (size, size))
    print(
        i,
        "shift",
        metrics(nshift, shifted),
        "center",
        metrics(ncenter, center),
        "FFT",
        metrics(native_stack[i], torch.fft.fft2(ncenter)),
    )
print("mean native spectra", metrics(native_ref, native_stack.mean(0)))
expected = native_stack.mean(0)
print(
    "mean real view",
    metrics(torch.view_as_complex(torch.view_as_real(native_stack).mean(0)), expected),
)
print(
    "mean separate components",
    metrics(torch.complex(native_stack.real.mean(0), native_stack.imag.mean(0)), expected),
)
for kind in ["tree", "adjacent", "sequential"]:
    lanes = [native_stack[i] for i in range(7)] + [torch.zeros_like(expected)]
    if kind == "tree":
        for stride in [4, 2, 1]:
            lanes = [lanes[i] + lanes[i + stride] for i in range(stride)]
        total = lanes[0]
    elif kind == "adjacent":
        while len(lanes) > 1:
            lanes = [lanes[i] + lanes[i + 1] for i in range(0, len(lanes), 2)]
        total = lanes[0]
    else:
        total = lanes[0]
        for item in lanes[1:7]:
            total = total + item
    print(
        "mean",
        kind,
        "divide",
        metrics(total / 7, expected),
        "multiply",
        metrics(total * (1 / 7), expected),
    )
shifted_stack = torch.stack([read(directory / f"shifted-{i}.f32", (size, size)) for i in range(7)])
batch_means = (shifted_stack * window).sum((1, 2)) / window.sum()
batch_center = (shifted_stack - batch_means[:, None, None]) * window
for i in range(7):
    actual = read(directory / f"centered-{i}.f32", (size, size))
    print("batch-center", i, metrics(actual, batch_center[i]))
x = read(directory / "centered-0.f32", (size, size))
expected = read(directory / "spectrum-0.f32", (size, size), True)
for axes in ([0, 1], [1, 0]):
    print("FFT axes", axes, metrics(expected, torch.ops.aten._fft_r2c.default(x, axes, 0, False)))
    print(
        "FFT complex axes",
        axes,
        metrics(expected, torch.ops.aten._fft_c2c.default(x.to(torch.complex64), axes, 0, True)),
    )


w = tukey_torch(192, 2 * 16 / 192, device="mps")
w = w[:, None] * w[None, :]
ref = None
fr = torch.fft.fftfreq(192, device="mps")[:, None]
fc = torch.fft.fftfreq(192, device="mps")[None, :]
for i in range(7):
    dp = read(prefix + f".dp_mean-{i}.f32", (192, 192))
    weighted = dp * w
    spectrum = torch.fft.fft2(weighted)
    nw = read(directory / f"dwindow-{i}.f32", (192, 192))
    ns = read(directory / f"dspectrum-{i}.f32", (192, 192), True)
    if i:
        shift = cross_correlation_shift_torch(ref, spectrum, 200, fft_input=True)
        nshift = read(directory / f"dshift-{i}.f32", (2,))
        print("diffshift", i, metrics(nshift, shift))
        prior = read(directory / f"dreference-{i - 1}.f32", (192, 192), True)
        shared = prior * (i / (i + 1)) + ns * torch.exp(
            -2j * torch.pi * (fr * nshift[0] + fc * nshift[1])
        ) / (i + 1)
        ref = ref * (i / (i + 1)) + spectrum * torch.exp(
            -2j * torch.pi * (fr * shift[0] + fc * shift[1])
        ) / (i + 1)
    else:
        ref = spectrum
        shared = ref
    nr = read(directory / f"dreference-{i}.f32", (192, 192), True)
    print(
        "diffraction",
        i,
        "window",
        metrics(nw, weighted),
        "FFT",
        metrics(ns, spectrum),
        "blend_same_inputs",
        metrics(nr, shared),
    )
