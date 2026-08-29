"""Device backends the solver can dispatch compute through.

See base.py for the interface and why it exists. `metal` is the only
implementation today; a CUDA sibling is the intended next one.
"""

from .base import BACKENDS, Caps, DeviceError, available, select

__all__ = ["BACKENDS", "Caps", "DeviceError", "available", "select"]
