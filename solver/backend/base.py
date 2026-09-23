"""The device abstraction the solver dispatches through.

Flow-X used to run every compute pass on Blender's `gpu` module. Its Metal
backend has no image atomics, caps read-write images at 8 per shader, offers no
read-write storage buffers, and cannot read a 1D texture back - limits that had
stopped being annoyances and started choosing the solver's algorithms. This
package is the replacement seam: a small, generic device interface with a Metal
implementation today and room for a CUDA one later.

Deliberately *generic*. Nothing here knows what SPH is; that lives in
solver/engine/. Keeping the split honest is what makes the CUDA port a matter of
adding a sibling module rather than touching solver logic, and it is why this
package imports neither `bpy` nor `gpu` - so scripts/test_backend.py can
exercise it in plain CPython.

The shape:

    backend = select()                     # or None, if no device is usable
    program = backend.program(source)      # compiles kernel source
    kernel  = program.kernel("sph_predict")
    buf     = backend.buffer(nbytes, data)
    queue   = backend.queue()
    queue.dispatch(kernel, threads, group, [buf, ...], constants=bytes)
    queue.commit()                         # everything recorded runs here
    view    = buf.map()                    # a memoryview, not a copy

`dispatch` only *records*. That is the whole reason the interface looks like
this: the grid build issues on the order of a hundred compare-exchange
dispatches per substep, and paying a Python-to-driver crossing for each one was
the dominant cost. One `commit()` hands the entire list over at once.
"""


class DeviceError(RuntimeError):
    """A device, compile, allocation or dispatch failure.

    One exception type on purpose: every caller's recovery is the same - report
    it and fall back to another engine - so distinguishing them would only add
    except-clauses nobody branches on.
    """


class Caps:
    """What a backend can do, for the parts of the solver that care.

    `atomics` is the one worth branching on: it is what decides between an
    atomic counting sort and the bitonic sort the old Blender path was forced
    into.
    """

    __slots__ = ("name", "device", "atomics", "max_threads_per_group", "unified_memory")

    def __init__(self, name, device, atomics, max_threads_per_group, unified_memory):
        self.name = name
        self.device = device
        self.atomics = atomics
        self.max_threads_per_group = max_threads_per_group
        self.unified_memory = unified_memory

    def __repr__(self):
        return f"<Caps {self.name} on {self.device!r} atomics={self.atomics}>"


# Names in the order select() tries them.
BACKENDS = ("metal",)


def available():
    """The backend names that can actually be brought up on this machine."""
    return tuple(name for name in BACKENDS if _load(name) is not None)


def select(name=None):
    """Bring up a backend, or return None if none is usable.

    None rather than raising: no usable device is an expected state (no Metal,
    a dylib that was never built, a quarantined download), and the caller's
    answer is to run the CPU engine, not to fail. A *named* backend that cannot
    load is different - that is the user asking for something specific - so
    that raises.
    """
    if name is not None:
        module = _load(name)
        if module is None:
            raise DeviceError(f"backend {name!r} is not available on this machine")
        return module.create()
    for candidate in BACKENDS:
        module = _load(candidate)
        if module is not None:
            try:
                return module.create()
            except DeviceError:
                # A loadable helper can still find no physical device (remote
                # sessions and sandboxed macOS processes are common cases).
                # Auto selection treats that exactly like an absent backend.
                continue
    return None


def _load(name):
    """The backend module for `name`, or None if it cannot be used here.

    A backend module reports its own usability through `is_available()` rather
    than by failing to import, so that a genuine bug in one does not silently
    read as "this machine doesn't support it".
    """
    if name == "metal":
        from . import metal

        return metal if metal.is_available() else None
    return None
