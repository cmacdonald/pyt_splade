"""Backward-compatible alias for :mod:`pyterrier_splade`.

This package was renamed to ``pyterrier_splade``. Importing ``pyt_splade``
continues to work by re-exporting everything from the new package and
aliasing the (private) submodules in ``sys.modules`` so that existing
imports such as ``from pyt_splade._model import Splade`` keep functioning.
"""
import sys
import warnings

import pyterrier_splade
from pyterrier_splade import *  # noqa: F401,F403
from pyterrier_splade import (  # noqa: F401
    _model,
    _encoder,
    _scorer,
    _utils,
)

warnings.warn(
    "The 'pyt_splade' package has been renamed to 'pyterrier_splade'. "
    "Please update your imports; 'pyt_splade' is kept only for backward "
    "compatibility and may be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__version__ = pyterrier_splade.__version__
__all__ = pyterrier_splade.__all__

# Alias submodules so `import pyt_splade._model` etc. resolve to the same
# module objects as their `pyterrier_splade` counterparts.
for _name in ('_model', '_encoder', '_scorer', '_utils'):
    sys.modules['pyt_splade.' + _name] = getattr(pyterrier_splade, _name)
del _name, sys, warnings
