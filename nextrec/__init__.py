from nextrec.__version__ import __version__

# Default thread limits to avoid oversubscription in multi-process inference.
# Users can override by setting environment variables before importing NextRec.
import os

_THREAD_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
}
for _key, _value in _THREAD_DEFAULTS.items():
    os.environ.setdefault(_key, _value)

__all__ = [
    "__version__",
]

# Package metadata
__author__ = "zerolovesea"
__email__ = "zyaztec@gmail.com"
__license__ = "Apache 2.0"
__url__ = "https://github.com/zerolovesea/NextRec"
