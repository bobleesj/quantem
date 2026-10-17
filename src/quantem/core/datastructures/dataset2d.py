from typing import Any, Self, Union

import numpy as np
import torch
from numpy.typing import NDArray

from quantem.core.datastructures.dataset import Dataset
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization.visualization import show_2d
from quantem.core.visualization.visualization_utils import ScalebarConfig


@Dataset.register_dimension(2)
class Dataset2d(Dataset):
    """2D dataset class that inherits from Dataset.

    This class represents all 2D datasets

    Attributes
    ----------
    None beyond base Dataset.
    """

    def __init__(
        self,
        array: NDArray | Any,
        name: str,
        origin: NDArray | tuple | list | float | int,
        sampling: NDArray | tuple | list | float | int,
        units: list[str] | tuple | list,
        signal_units: str = "arb. units",
        _token: object | None = None,
        tensor: torch.Tensor | None = None,
    ):
        """Initialize a 2D dataset.

        Parameters
        ----------
        array : NDArray | Any
            The underlying 2D array data
        name : str
            A descriptive name for the dataset
        origin : NDArray | tuple | list | float | int
            The origin coordinates for each dimension in calibrated units
        sampling : NDArray | tuple | list | float | int
            The sampling rate/spacing for each dimension
        units : list[str] | tuple | list
            Units for each dimension
        signal_units : str, optional
            Units for the array values, by default "arb. units"
        _token : object | None, optional
            Token to prevent direct instantiation, by default None
        """
        super().__init__(
            array=array,
            tensor=tensor,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            _token=_token,
        )

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        name: str | None = None,
        origin: NDArray | tuple | list | None = None,
        sampling: NDArray | tuple | list | None = None,
        units: list[str] | tuple | None = None,
        signal_units: str = "arb. units",
    ) -> Self:
        """Wrap a two-dimensional Torch tensor without copying its storage.

        Parameters
        ----------
        tensor : torch.Tensor
            Values in (row, col) order, retained on their current device.
        name : str, optional
            Dataset label.
        origin, sampling : array-like, optional
            Coordinate origin and sampling for each axis.
        units : sequence of str, optional
            Coordinate units for each axis.
        signal_units : str, optional
            Units of the stored values.

        Returns
        -------
        Dataset2d
            Tensor-backed dataset. Use ``tensor`` for device access or ``numpy()``
            to explicitly copy values to the host.

        Examples
        --------
        >>> values_t = torch.ones((16, 16))
        >>> dataset = Dataset2d.from_tensor(values_t)
        >>> dataset.tensor is values_t
        True
        """
        if tensor.ndim != 2:
            raise ValueError(f"Expected (row, col) tensor, got shape {tuple(tensor.shape)}.")
        return cls(
            array=None,
            tensor=tensor,
            name=name or "2D dataset (torch)",
            origin=np.zeros(2) if origin is None else origin,
            sampling=np.ones(2) if sampling is None else sampling,
            units=["pixels"] * 2 if units is None else units,
            signal_units=signal_units,
            _token=cls._token,
        )

    @classmethod
    def from_file(cls, file_path: str, file_type: str | None = None) -> "Dataset2d":
        """
        Load a Dataset2d from a file using the appropriate file reader.

        Parameters
        ----------
        file_path : str
            Path to the data file.
        file_type : str | None, optional
            File type hint. If None, the format will be inferred automatically.

        Returns
        -------
        Dataset2d
            The loaded dataset.
        """
        # Import here to avoid circular imports
        from quantem.core.io.file_readers import read_2d

        return read_2d(file_path, file_type)

    @classmethod
    def from_array(
        cls,
        array: NDArray | Any,
        name: str | None = None,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
    ) -> Self:
        """
        Create a new Dataset2d from an array.

        Parameters
        ----------
        array : NDArray | Any
            The underlying 4D array data
        name : str | None, optional
            A descriptive name for the dataset. If None, defaults to "2D dataset"
        origin : NDArray | tuple | list | float | int | None, optional
            The origin coordinates for each dimension in calibrated units. If None, defaults to zeros
        sampling : NDArray | tuple | list | float | int | None, optional
            The sampling rate/spacing for each dimension in calibrated units. If None, defaults to ones
        units : list[str] | tuple | list | None, optional
            Units for each dimension. If None, defaults to ["pixels"] * 4
        signal_units : str, optional
            Units for the array values, by default "arb. units"

        Returns
        -------
        Dataset2d
            A new Dataset2d instance
        """
        array = ensure_valid_array(array, ndim=2)
        return cls(
            array=array,
            name=name if name is not None else "2D dataset",
            origin=origin if origin is not None else np.zeros(2),
            sampling=sampling if sampling is not None else np.ones(2),
            units=units if units is not None else ["pixels"] * 2,
            signal_units=signal_units,
            _token=cls._token,
        )

    @classmethod
    def from_shape(
        cls,
        shape: tuple[int, int],
        name: str = "constant 2D dataset",
        fill_value: float = 0.0,
        origin: Union[NDArray, tuple, list, float, int] | None = None,
        sampling: Union[NDArray, tuple, list, float, int] | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
    ) -> Self:
        """Create a new Dataset2d filled with a constant value."""
        array = np.full(shape, fill_value, dtype=np.float32)
        return cls.from_array(
            array=array,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
        )

    def show(
        self,
        scalebar: ScalebarConfig | bool = True,
        title: str | None = None,
        **kwargs,
    ):
        """
        Displays Dataset as a 2D image

        Parameters
        ----------
        scalebar: ScalebarConfig or bool
            If True, displays scalebar
        title: str
            Title of Dataset
        **kwargs: dict
            Keyword arguments for show_2d
        """

        if scalebar is True:
            scalebar = ScalebarConfig(
                sampling=self.sampling[-1],
                units=self.units[-1],
            )

        if title is None:
            title = self.name

        return show_2d(self.numpy(), scalebar=scalebar, title=title, **kwargs)
