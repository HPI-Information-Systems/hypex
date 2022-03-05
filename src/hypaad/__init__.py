import logging
from importlib.metadata import PackageNotFoundError, version

from hypaad.log import setup_logging  # pragma: no cover

from .anomaly_detectors import *
from .config import *
from .optimizer import *

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
__all__ = anomaly_detectors.__all__ + config.__all__ + optimizer.__all__
