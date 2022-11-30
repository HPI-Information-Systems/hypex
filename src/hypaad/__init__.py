import logging
from importlib.metadata import PackageNotFoundError, version

from hypaad.log import setup_logging  # pragma: no cover

from .algorithms import *
from .cluster import *
from .cluster_config import *
from .config import *
from .high_level_execution import *
from .modules import *
from .optuna_storage import *
from .registry import *
from .seed import *
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
    + cluster.__all__
    + cluster_config.__all__
    + config.__all__
    + high_level_execution.__all__
    + modules.__all__
    + optuna_storage.__all__
    + registry.__all__
    + seed.__all__  # pylint: disable=no-member
    + trial_result.__all__
    + utils.__all__
)
