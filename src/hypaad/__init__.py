import logging
from importlib.metadata import PackageNotFoundError, version

from hypaad.log import setup_logging  # pragma: no cover

from .algorithms import *
from .cluster import *
from .cluster_config import *
from .config import *
from .early_stopping import *
from .hypaad_executor import *
from .optimizer import *
from .registry import *
from .trial_result import *
from .utils import *

try:
    # Change here if project is renamed and does not equal the package name
    DIST_NAME = __name__
    __version__ = version(DIST_NAME)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

setup_logging(logging.INFO)

# pylint: disable=undefined-variable
__all__ = (
    algorithms.__all__
    + early_stopping.__all__
    + config.__all__
    + cluster.__all__
    + cluster_config.__all__
    + hypaad_executor.__all__
    + optimizer.__all__
    + registry.__all__
    + trial_result.__all__
    + utils.__all__
)
