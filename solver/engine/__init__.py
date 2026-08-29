"""Simulation engines: the thing that actually advances the fluid.

An engine owns the particle state and knows how to step it. `metal_engine`
dispatches compute kernels through solver/backend/; a CPU engine using numpy is
the fallback for machines with no usable device, and the reference the GPU
engine is checked against.

This is a level above solver/backend/, and the split matters: the backend is a
generic device (buffers, kernels, a queue) that knows nothing about SPH, while
an engine is all SPH and may not involve a device at all. That is what lets a
CUDA engine reuse the pass bodies while a numpy engine reuses none of them.

solver/sph.py keeps the timeline, the cache and the operators, and asks an
engine to do the physics.
"""

_reason = None


def create(preferred=None):
    """Bring up an engine, or return None if none can run.

    `preferred` names one explicitly ("metal", "cpu"); the default tries the
    GPU and falls back. None rather than raising, because "this machine has no
    usable device" is an ordinary state the caller reports in the panel.
    unavailable_reason() says why.
    """
    global _reason
    _reason = None

    if preferred in (None, "metal"):
        from . import metal_engine

        engine = metal_engine.create()
        if engine is not None:
            return engine
        from ..backend import metal as metal_backend

        _reason = metal_backend.load_error() or "no Metal device"
        if preferred == "metal":
            return None

    if preferred in (None, "cpu"):
        from . import cpu_engine

        try:
            return cpu_engine.create()
        except ImportError as exc:
            # numpy ships with Blender, so this should not happen - but a CPU
            # engine that cannot import is still better reported than crashed.
            _reason = f"CPU engine unavailable ({exc})"

    return None


def describe(engine):
    """A short name for the panel: the engine, and the device where there is one."""
    if engine is None:
        return "none"
    device = getattr(getattr(engine, "backend", None), "caps", None)
    return f"{engine.name} ({device.device})" if device is not None else engine.name


def unavailable_reason():
    """Why create() returned None, for the panel and the operator report."""
    return _reason or "no compute engine available"
