import logging
import sys

from nextrec.__version__ import __version__


def init_default_logging() -> None:
    root = logging.getLogger()
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)


init_default_logging()

__all__ = [
    "__version__",
]

# Package metadata
__author__ = "zerolovesea"
__email__ = "zyaztec@gmail.com"
__license__ = "Apache 2.0"
__url__ = "https://github.com/zerolovesea/NextRec"
