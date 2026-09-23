"""Simulation engines: the thing that actually advances the fluid.

An engine owns the particle state and knows how to step it. `metal_engine`
dispatches compute kernels through solver/backend/; a CPU engine using numpy is
the fallback for machines with no usable device, and the reference the GPU
engine is checked against.

This is a level above solver/backend/, and the split matters: the backend is a
generic device (buffers, kernels, a queue) that knows nothing about fluids, while
an engine owns one fluid method and may not involve a device at all. That is what lets a
CUDA engine reuse the pass bodies while a numpy engine reuses none of them.

solver/sph.py keeps the timeline, the cache and the operators, and asks an
engine to do the physics.

Two orthogonal axes pick an engine: the *method* (the algorithm - "pbf",
"apic") and the *device* it runs on ("metal", "cpu"). The method decides which
module to import; the device decides which module inside the method.
"""

_reason = None

# method -> device -> engine module. A method with no entry here has no
# implementation at all; a module whose import fails is one this machine cannot
# run, which the caller reports rather than crashes on.
_METHOD_MODULES = {
    "pbf": {"metal": "metal_engine", "cpu": "cpu_engine"},
    "apic": {"metal": "apic_metal", "cpu": "apic_cpu"},
}


def create(preferred=None, method="pbf"):
    """Bring up an engine for `method`, or return None if none can run.

    `preferred` names a device explicitly ("metal", "cpu"); the default tries
    the GPU and falls back to the *same method* on the CPU. Methods are never
    substituted: selecting APIC must not silently run PBF. None rather than
    raising, because "this machine has no usable device" is an ordinary state
    the caller reports in the panel. unavailable_reason() says why.
    """
    global _reason
    _reason = None

    modules = _METHOD_MODULES.get(method)
    if modules is None:
        _reason = f"unknown solver method {method!r}"
        return None

    return _from_modules(modules, preferred)


def _from_modules(modules, preferred):
    global _reason

    if preferred in (None, "metal"):
        engine, err = _try_create(modules["metal"])
        if engine is not None:
            return engine
        from ..backend import metal as metal_backend

        # metal_engine.create() declines quietly; the reason lives in the
        # backend (no dylib, no device, failed compile). An ImportError here
        # means the method has no Metal engine to ask.
        _reason = err or metal_backend.load_error() or "no Metal device"
        if preferred == "metal":
            return None

    if preferred in (None, "cpu"):
        engine, err = _try_create(modules["cpu"])
        if engine is not None:
            return engine
        if err:
            # numpy ships with Blender, so this should not happen - but a CPU
            # engine that cannot import is still better reported than crashed.
            _reason = f"CPU engine unavailable ({err})"

    return None


def _try_create(module_name):
    """Import an engine module and call its create(); never raise."""
    try:
        from importlib import import_module

        module = import_module(f".{module_name}", __package__)
        return module.create(), None
    except ImportError as exc:
        return None, str(exc)
    except Exception as exc:
        from ..backend import DeviceError

        if isinstance(exc, DeviceError):
            return None, str(exc)
        raise


def describe(engine):
    """A short name for the panel: method and engine, and the device where there is one."""
    if engine is None:
        return "none"
    method = getattr(engine, "method", "pbf").upper()
    label = f"{method} {engine.name}"
    device = getattr(getattr(engine, "backend", None), "caps", None)
    return f"{label} ({device.device})" if device is not None else label


def unavailable_reason():
    """Why create() returned None, for the panel and the operator report."""
    return _reason or "no compute engine available"


def fallback_note():
    """Compatibility shim: methods are no longer substituted."""
    return None
