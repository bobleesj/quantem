"""Multi-GPU 5D-STEM series for MAPED / time-series.

A tilt or time series of N 4D-STEM frames is often larger than one GPU's VRAM
(7 tilts at 512x512x192x192 uint16 = 135 GB). ``Dataset5dstem`` keeps the frames
*resident* but **distributed across GPUs**, so the whole series stays available
for viewing (flip through each frame) and feeds MAPED one frame at a time.

Design:
- Frames keep their **native integer dtype** (uint16 = half of float32); compute
  casts to float per-frame on demand, so VRAM holds raw data, not float copies.
- Each frame lives on an assigned GPU. ``MAPEDTorch`` iterates frames and moves
  the active one to the compute device, so the compute GPU only ever holds one
  frame's float cast + the merge working set, never all N.
- Pure torch; CUDA is just the backend. No cupy, no widget dependency (this lives
  in core ``quantem.diffraction``; the widget re-exports it for viewing).

This is the multi-GPU successor to the single-device ``Dataset5dstem`` (PR 231):
prototyped here against real MAPED data, to be promoted into core once proven.
"""

from __future__ import annotations

from typing import Sequence

import torch

_SERIES_TYPES = ("generic", "tilt", "time")


def _resolve_devices(devices, n_frames: int) -> list[str]:
    """Turn a ``devices`` spec into a per-frame device list of length n_frames.

    'auto'  -> round-robin over all visible CUDA devices (CPU if none).
    [0, 1]  -> round-robin over the given indices / device strings.
    'cuda:0'/0 -> all frames on that one device.
    """
    if devices == "auto" or devices is None:
        count = torch.cuda.device_count()
        pool = [f"cuda:{i}" for i in range(count)] if count else ["cpu"]
    elif isinstance(devices, (str, int, torch.device)):
        pool = [_as_device_str(devices)]
    else:
        pool = [_as_device_str(d) for d in devices]
    return [pool[i % len(pool)] for i in range(n_frames)]


def _as_device_str(d) -> str:
    if isinstance(d, int):
        return f"cuda:{d}"
    return str(d)


class Dataset5dstem:
    """A 5D-STEM series (N frames of 4D data) distributed across GPUs.

    Frames are a Python list of 4D torch tensors (scan, scan, k, k), each on its
    assigned device, kept in native dtype. The series axis (frame index) is
    described by ``series_type`` ('tilt' / 'time' / 'generic') + ``series``.
    """

    def __init__(
        self,
        frames: list[torch.Tensor],
        series_type: str = "generic",
        series: Sequence | None = None,
        name: str | None = None,
    ):
        if not frames:
            raise ValueError("Dataset5dstem needs at least one frame.")
        if series_type not in _SERIES_TYPES:
            raise ValueError(f"series_type must be one of {_SERIES_TYPES}, got {series_type!r}.")
        base_shape = tuple(frames[0].shape)
        for i, f in enumerate(frames):
            if f.ndim != 4:
                raise ValueError(f"frame {i} must be 4D (scan, scan, k, k), got {tuple(f.shape)}.")
            if tuple(f.shape) != base_shape:
                raise ValueError(
                    f"all frames must share shape; frame 0 is {base_shape}, "
                    f"frame {i} is {tuple(f.shape)}."
                )
        self.frames = list(frames)
        self.series_type = series_type
        self.series = list(series) if series is not None else list(range(len(frames)))
        self.name = name or "5D-STEM series (multi-GPU)"
        # Tell Show4DSTEM to take its no-gather GPU-frames path: it scrubs the
        # series by integer-indexing one frame at a time (each frame stays on its
        # home GPU), instead of stacking a 135 GB sharded series onto one card.
        self._is_gpu_frames = True

    # --- Show4DSTEM duck-type: a 5D GPU-frame stack ---
    @property
    def shape(self) -> tuple[int, ...]:
        """5D series shape ``(n_frames, scan_r, scan_c, k_r, k_c)``."""
        return (len(self.frames),) + tuple(self.frames[0].shape)

    @property
    def ndim(self) -> int:
        return 5

    @property
    def device(self):
        """Representative device (frame 0's). Sharded frames each keep their own;
        Show4DSTEM moves the indexed frame to the view device per scrub."""
        return self.frames[0].device

    def element_size(self) -> int:
        return self.frames[0].element_size()

    def numel(self) -> int:
        n = len(self.frames)
        for d in self.frames[0].shape:
            n *= d
        return n

    @classmethod
    def from_tensors(
        cls,
        tensors: Sequence[torch.Tensor],
        devices="auto",
        series_type: str = "generic",
        series: Sequence | None = None,
        name: str | None = None,
    ) -> "Dataset5dstem":
        """Distribute a sequence of 4D frame tensors across ``devices``.

        Each frame is moved to its assigned GPU (round-robin for 'auto' / a list).
        Frames keep their dtype - pass uint16 to halve VRAM vs float32.
        """
        tensors = list(tensors)
        placement = _resolve_devices(devices, len(tensors))
        frames = [t.to(dev) for t, dev in zip(tensors, placement)]
        return cls(frames, series_type=series_type, series=series, name=name)

    # --- series / placement introspection ---
    @property
    def devices(self) -> list[str]:
        """Device of each frame, in series order."""
        return [str(f.device) for f in self.frames]

    @property
    def is_sharded(self) -> bool:
        """True when frames span more than one device."""
        return len({str(f.device) for f in self.frames}) > 1

    @property
    def dtype(self) -> torch.dtype:
        return self.frames[0].dtype

    @property
    def frame_shape(self) -> tuple[int, ...]:
        return tuple(self.frames[0].shape)

    @property
    def nbytes(self) -> int:
        return sum(f.element_size() * f.nelement() for f in self.frames)

    def bytes_per_device(self) -> dict[str, int]:
        """Resident bytes on each device - the sharding's memory footprint."""
        out: dict[str, int] = {}
        for f in self.frames:
            out[str(f.device)] = out.get(str(f.device), 0) + f.element_size() * f.nelement()
        return out

    # --- iteration: MAPED reads frames one at a time, moving to compute device ---
    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.frames[i]

    def __iter__(self):
        return iter(self.frames)

    def to_list(self) -> list[torch.Tensor]:
        """The frame list, for ``MAPEDTorch.from_datasets`` (which moves each frame
        to its compute device on demand)."""
        return self.frames

    def summary(self) -> str:
        per = ", ".join(f"{d}: {b / 1e9:.1f} GB" for d, b in self.bytes_per_device().items())
        return (
            f"Dataset5dstem: {len(self)} {self.series_type} frames {self.frame_shape} "
            f"{self.dtype}, {self.nbytes / 1e9:.1f} GB total | sharded={self.is_sharded} | {per}"
        )
