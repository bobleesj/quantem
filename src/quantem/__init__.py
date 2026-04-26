from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

# Suppress torch's pynvml-deprecated FutureWarning (from torch/cuda/__init__.py)
# before any submodule triggers a torch import. The warning targets
# torch's own maintainers, not anyone running quantem code, so showing
# it on every CLI invocation is just noise.
import warnings as _warnings
_warnings.filterwarnings(
    "ignore",
    message=r".*pynvml package is deprecated.*",
    category=FutureWarning,
)

from importlib.metadata import version

from quantem.core import io as io
from quantem.core import datastructures as datastructures
from quantem.core import visualization as visualization

from quantem import imaging as imaging
from quantem import diffractive_imaging as diffractive_imaging

__version__ = version("quantem")
