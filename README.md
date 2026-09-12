# quantem

This is the home repository for the quantitative electron microscopy (quantem) data analysis toolkit.

## Installation Instructions

The package is available on the Python Package Index (PyPi), as [quantem](https://pypi.org/project/quantem/).

You can install it using `pip install quantem`.

For a developer install, please refer to [CONTRIBUTORS.md](CONTRIBUTORS.md).

## MAPED tutorials

The [MAPED tutorial index](docs/tutorials/README.md) collects the interactive
seven-tilt explainer, the executable CUDA notebook, and the 24 GiB workflow in
one place. Start with the interactive tutorial to understand the alignment,
bounded merge, encoded residency, and globally scaled uint16 output before
running the full dataset.

For Python-free execution on Apple GPUs, see the
[native MAPED code and tests](native/README.md). It retains the scientific
workflow and parameter names while using QuantEM.GPU for native infrastructure.

## License

quantem is free and open source software, distributed under the [MIT License](LICENSE).
