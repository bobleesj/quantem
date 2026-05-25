from typing import Iterator, Self

import numpy as np
import torch
from numpy.typing import NDArray

from quantem.core.datastructures.dataset import Dataset
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.utils.validators import validate_ndinfo, validate_units


_SERIES_TYPES = ("time", "tilt", "energy", "dose", "focus", "generic")
_GiB = 1 << 30


class Dataset5dstem(Dataset):
    """**EXPERIMENTAL.** Torch-backed 5D-STEM series ``(N, scan_row, scan_col, k_row, k_col)``.

    Stack of 4D-STEM acquisitions sharing identical scan + k calibration. Axis 0
    represents ONE monotonically varying experimental parameter (time, tilt,
    focus, dose, energy, generic).

    ``sampling`` / ``units`` / ``origin`` are 4-length (scan + k only) - the
    series axis is described separately by ``series_type`` + ``series``. This
    diverges from base Dataset's ``len(sampling) == ndim`` convention but keeps
    the user-facing API clean (no axis-0 placeholders).

    Two backings, one logical view:

    - **single tensor** (one device) - the common case, axis 0 is the series.
    - **series of frames** (multi-device) - each frame is its own 4D torch
      tensor that knows its device, so a series larger than one card fits across
      several GPUs (e.g. 6x 512²x192² no-bin = 108 GiB across two 96 GB cards)
      while still presenting one ``(N, scan, scan, k, k)`` dataset. Each frame is
      an independent acquisition, so placement is a per-frame property and freeing
      VRAM is per-frame. Build via ``from_4dstem`` with frames that live on
      different devices; inspect with ``.devices`` / ``.summary()``; release with
      ``.free()``. API is experimental.
    """

    def __init__(
        self,
        tensor: torch.Tensor | None,
        name: str = "",
        sampling: NDArray | tuple | list | None = None,
        units: list[str] | tuple | list | None = None,
        origin: NDArray | tuple | list | None = None,
        signal_units: str = "arb. units",
        metadata: dict | None = None,
        series_type: str = "generic",
        series: NDArray | list | tuple | None = None,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use Dataset5dstem.from_tensor() or Dataset5dstem.from_4dstem() to instantiate."
            )
        if series_type not in _SERIES_TYPES:
            raise ValueError(f"series_type must be one of {_SERIES_TYPES}, got {series_type!r}.")
        # Multi-device backing (a list of per-frame 4D tensors, each on its own
        # device) is attached by _from_frames AFTER construction; default to the
        # single-tensor backing so base init + the series validator see a normal
        # length. When _frames is set, the anchor self._tensor (frame 0) is used
        # only for dtype/device metadata; the logical 5D view comes from the
        # __len__ / shape / __getitem__ overrides below.
        self._frames: list[torch.Tensor] | None = None
        super().__init__(
            tensor=tensor, name=name,
            sampling=sampling, units=units, origin=origin,
            signal_units=signal_units, metadata=metadata, _token=_token,
        )
        self.series_type = series_type
        self.series = series

    @property
    def is_sharded(self) -> bool:
        """True if the series is a list of per-frame tensors across >1 device."""
        return self._frames is not None and len({str(t.device) for t in self._frames}) > 1

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        name: str | None = None,
        sampling: NDArray | tuple | list | None = None,
        units: list[str] | tuple | list | None = None,
        origin: NDArray | tuple | list | None = None,
        signal_units: str = "arb. units",
        metadata: dict | None = None,
        series_type: str = "generic",
        series: NDArray | list | tuple | None = None,
    ) -> Self:
        """Wrap a 5D torch tensor. ``sampling`` / ``units`` / ``origin`` are 4-length
        (scan_row, scan_col, k_row, k_col); axis 0 lives in ``series_type`` + ``series``.
        """
        if tensor.ndim != 5:
            raise ValueError(
                f"from_tensor requires 5D tensor (N, scan_row, scan_col, k_row, k_col), "
                f"got shape {tuple(tensor.shape)}."
            )
        return cls(
            tensor=tensor,
            name=name if name is not None else "5D-STEM dataset (torch)",
            sampling=sampling if sampling is not None else np.ones(4),
            units=units if units is not None else ["pixels"] * 4,
            origin=origin if origin is not None else np.zeros(4),
            signal_units=signal_units, metadata=metadata,
            series_type=series_type, series=series,
            _token=cls._token,
        )

    @classmethod
    def _from_frames(
        cls,
        frames: list[torch.Tensor],
        name: str,
        sampling, units, origin,
        signal_units: str = "arb. units",
        metadata: dict | None = None,
        series_type: str = "generic",
        series=None,
    ) -> Self:
        """Build a series from per-frame 4D tensors (each may be on its own device).

        This is the single place that decides the backing: if all frames share
        ONE device they are stacked into a compact 5D tensor (the common path);
        only a genuinely MULTI-device set is kept as a frame list. That keeps the
        invariant ``_frames is not None`` ⟺ multi-device ⟺ ``is_sharded``, so
        slices and inherited code never see a one-device frame list. Multi-device
        frames keep their device (no gather) - that is what lets a series exceed
        one card. The anchor tensor (frame 0) carries dtype/device metadata; the
        logical 5D view comes from the overrides.
        """
        if not frames:
            raise ValueError("from_4dstem needs at least one frame; got an empty list.")
        base_shape = tuple(frames[0].shape)
        base_dtype = frames[0].dtype
        for i, f in enumerate(frames):
            if f.ndim != 4:
                raise ValueError(f"frame {i} must be 4D (scan, scan, k, k), got {tuple(f.shape)}.")
            if tuple(f.shape) != base_shape:
                raise ValueError(
                    f"all frames must share shape; frame 0 is {base_shape}, frame {i} is {tuple(f.shape)}."
                )
            if f.dtype != base_dtype:
                raise ValueError(
                    f"all frames must share dtype; frame 0 is {base_dtype}, frame {i} is {f.dtype}."
                )
        # One device → stack into a single 5D tensor (compact, and the inherited
        # single-tensor methods stay correct). Only keep a frame list when the
        # frames genuinely span devices.
        if len({str(f.device) for f in frames}) == 1:
            return cls.from_tensor(
                tensor=torch.stack(list(frames), dim=0), name=name,
                sampling=sampling, units=units, origin=origin,
                signal_units=signal_units, metadata=metadata,
                series_type=series_type, series=series,
            )
        obj = cls(
            tensor=frames[0], name=name,
            sampling=sampling, units=units, origin=origin,
            signal_units=signal_units, metadata=metadata,
            series_type=series_type, series=None, _token=cls._token,
        )
        obj._frames = list(frames)
        obj.series = series  # validated against len(_frames), which is now set
        return obj

    @classmethod
    def from_4dstem(
        cls,
        datasets: list[Dataset4dstem],
        name: str | None = None,
        series_type: str = "generic",
        series: NDArray | list | tuple | None = None,
    ) -> Self:
        """Stack tensor-backed ``Dataset4dstem`` into a series.

        Same-device frames stack into one compact 5D tensor; frames spread across
        DIFFERENT devices stay a per-frame list (each on its own card), so a
        series larger than one GPU just works. Spatial calibration inherits from
        the first frame.
        """
        if not datasets:
            raise ValueError("from_4dstem needs at least one Dataset4dstem.")
        first = datasets[0]
        name = name if name is not None else f"{len(datasets)}x {first.name}"
        return cls._from_frames(
            [d.tensor for d in datasets], name=name,
            sampling=first.sampling, units=first.units, origin=first.origin,
            series_type=series_type, series=series,
        )

    # --- Override base sampling/units/origin: 4-length (scan + k), not ndim-length ---
    @property
    def sampling(self) -> NDArray: return self._sampling

    @sampling.setter
    def sampling(self, value) -> None:
        self._sampling = validate_ndinfo(value, 4, "sampling")

    @property
    def origin(self) -> NDArray: return self._origin

    @origin.setter
    def origin(self, value) -> None:
        self._origin = validate_ndinfo(value, 4, "origin")

    @property
    def units(self) -> list[str]: return self._units

    @units.setter
    def units(self, value) -> None:
        self._units = validate_units(value, 4)

    # --- Logical 5D view (single-tensor OR series-of-frames) ---
    @property
    def shape(self) -> tuple[int, ...]:
        if self._frames is not None:
            return (len(self._frames), *tuple(self._frames[0].shape))
        if self._tensor is None:
            raise RuntimeError("Dataset5dstem has been freed; re-load to use it again.")
        return tuple(self._tensor.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)  # always 5 (base would report the anchor frame's 4)

    @property
    def devices(self) -> list[str]:
        """Device of each frame, in series order."""
        if self._frames is not None:
            return [str(t.device) for t in self._frames]
        return [str(self._tensor.device)] * len(self)

    @property
    def frames(self) -> list[torch.Tensor]:
        """The per-frame 4D torch tensors, in series order, each on its device.

        This is the plain-torch view a viewer (e.g. ``Show4DSTEM``) consumes -
        no dataset class needed downstream: ``Show4DSTEM(dset.frames)``.
        """
        if self._frames is not None:
            return list(self._frames)
        return [self._tensor[i] for i in range(len(self))]

    def summary(self) -> dict[str, float]:
        """Print a frame | device | GiB | dtype table; return per-device GiB totals."""
        if self._frames is not None:
            frames = self._frames
        else:
            frames = [self._tensor[i] for i in range(len(self))]
        per_device: dict[str, float] = {}
        print(f"{self.name}  ({self.series_type} series, {len(self)} frames)")
        print(f"{'frame':>5}  {'device':>8}  {'GiB':>6}  dtype")
        for i, f in enumerate(frames):
            gib = f.element_size() * f.nelement() / _GiB
            dev = str(f.device)
            per_device[dev] = per_device.get(dev, 0.0) + gib
            print(f"{i:>5}  {dev:>8}  {gib:>6.2f}  {f.dtype}")
        for dev, gib in sorted(per_device.items()):
            print(f"  total {dev}: {gib:.2f} GiB")
        return per_device

    def free(self) -> None:
        """Release all frame VRAM. The dataset is spent afterward (accessing it
        raises a clear error; re-load to use again). The CUDA caching allocator
        is emptied per freed device so the memory returns to the OS view."""
        if self._frames is not None:
            devs = {t.device for t in self._frames}
        elif self._tensor is not None:
            devs = {self._tensor.device}
        else:
            devs = set()
        self._frames = None
        self._tensor = None
        for d in devs:
            if d.type == "cuda":
                with torch.cuda.device(d):
                    torch.cuda.empty_cache()

    # --- Series metadata ---
    @property
    def series(self) -> NDArray | None:
        return self._series

    @series.setter
    def series(self, value) -> None:
        if value is None:
            self._series = None
            return
        arr = np.asarray(value, dtype=float)
        n = len(self)
        if arr.ndim != 1 or len(arr) != n:
            raise ValueError(f"series must be 1D length {n}, got shape {arr.shape}.")
        self._series = arr

    # --- Frame access ---
    def __len__(self) -> int:
        if self._frames is not None:
            return len(self._frames)
        if self._tensor is None:
            raise RuntimeError("Dataset5dstem has been freed; re-load to use it again.")
        return int(self._tensor.shape[0])

    def _frame_tensor(self, index: int) -> torch.Tensor:
        """The 4D tensor for series step ``index``, on its own device."""
        if self._frames is not None:
            return self._frames[index]
        return self._tensor[index]

    def __getitem__(self, index: int | slice) -> Dataset4dstem | Self:
        if isinstance(index, int):
            return Dataset4dstem.from_tensor(
                self._frame_tensor(index),
                name=f"{self.name}[{index}]",
                sampling=self.sampling, units=self.units,
            )
        sub_series = None if self._series is None else self._series[index]
        if self._frames is not None:
            return Dataset5dstem._from_frames(
                self._frames[index], name=self.name,
                sampling=self.sampling, units=self.units, origin=self.origin,
                signal_units=self.signal_units, metadata=self._metadata,
                series_type=self.series_type, series=sub_series,
            )
        return Dataset5dstem.from_tensor(
            tensor=self._tensor[index],
            name=self.name,
            sampling=self.sampling, units=self.units, origin=self.origin,
            signal_units=self.signal_units, metadata=self._metadata,
            series_type=self.series_type, series=sub_series,
        )

    def __iter__(self) -> Iterator[Dataset4dstem]:
        for i in range(len(self)):
            yield self[i]
