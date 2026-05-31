import sys
from ._mohu import Tensor
from . import array_api

__all__ = ["Tensor", "array_api", "abs", "add"]

# ── Array API Namespace standard hooks ──
def __array_namespace__(self, api_version=None):
    return array_api

Tensor.__array_namespace__ = __array_namespace__

# ── NumPy interoperability protocols (__array_ufunc__) ──
def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    if method == '__call__':
        import mohu as mh
        if hasattr(mh, ufunc.__name__):
            return getattr(mh, ufunc.__name__)(*inputs, **kwargs)
    return NotImplemented

Tensor.__array_ufunc__ = __array_ufunc__

# ── Core operations routed directly through mohu-ops ──
def abs(x: Tensor) -> Tensor:
    return array_api.abs(x)

def add(x1: Tensor, x2: Tensor) -> Tensor:
    return array_api.add(x1, x2)
