"""Metal backend: ctypes bindings to bin/libflowx_metal.dylib.

The dylib (built from native/flowx_metal.mm by scripts/build_native.py) does
the Objective-C; this module is the Python half of that ABI and nothing more.
Keep the two in step - native/flowx_metal.h is the contract, and FLOWX_ABI
below must match the FLOWX_ABI_VERSION it declares.

Everything here is plain ctypes and stdlib. No `bpy`, no `gpu`, no third-party
package: the add-on ships with zero dependencies, and this module has to stay
importable from a bare CPython so scripts/test_backend.py can run without
Blender.
"""

import ctypes
import sys
from pathlib import Path

from .base import Caps, DeviceError

# Must match FLOWX_ABI_VERSION in native/flowx_metal.h. A mismatch means a stale
# bin/ from before an ABI change, which would otherwise be read as garbage.
FLOWX_ABI = 1

DYLIB = Path(__file__).resolve().parent.parent.parent / "bin" / "libflowx_metal.dylib"

# Metal's own limit on setBytes; past it the constants need a real buffer. The
# solver's parameter block is far below this - it exists as a guard so that
# overrunning it fails here with a clear message rather than inside the driver.
MAX_CONSTANTS = 4096

_lib = None
_load_error = None


class _Dispatch(ctypes.Structure):
    """Mirror of FlowxDispatch in native/flowx_metal.h. Field order matters."""

    _fields_ = [
        ("kernel", ctypes.c_void_p),
        ("buffers", ctypes.POINTER(ctypes.c_void_p)),
        ("nbuffers", ctypes.c_uint32),
        ("constants", ctypes.c_void_p),
        ("constants_size", ctypes.c_uint32),
        ("constants_index", ctypes.c_uint32),
        ("threads", ctypes.c_uint32),
        ("threads_per_group", ctypes.c_uint32),
    ]


_P = ctypes.c_void_p
_ERR = ctypes.POINTER(ctypes.c_char_p)

_SIGNATURES = {
    "flowx_abi_version": (ctypes.c_int32, []),
    "flowx_string_free": (None, [ctypes.c_char_p]),
    "flowx_device_create": (_P, [_ERR]),
    "flowx_device_destroy": (None, [_P]),
    "flowx_device_name": (ctypes.c_char_p, [_P]),
    "flowx_device_max_buffer_length": (ctypes.c_uint64, [_P]),
    "flowx_device_has_unified_memory": (ctypes.c_int32, [_P]),
    "flowx_library_new": (_P, [_P, ctypes.c_char_p, _ERR]),
    "flowx_library_destroy": (None, [_P]),
    "flowx_kernel_new": (_P, [_P, ctypes.c_char_p, _ERR]),
    "flowx_kernel_destroy": (None, [_P]),
    "flowx_kernel_max_threads_per_group": (ctypes.c_uint32, [_P]),
    "flowx_kernel_execution_width": (ctypes.c_uint32, [_P]),
    "flowx_buffer_new": (_P, [_P, ctypes.c_uint64, ctypes.c_void_p, _ERR]),
    "flowx_buffer_destroy": (None, [_P]),
    "flowx_buffer_contents": (_P, [_P]),
    "flowx_buffer_length": (ctypes.c_uint64, [_P]),
    "flowx_queue_new": (_P, [_P, _ERR]),
    "flowx_queue_destroy": (None, [_P]),
    "flowx_submit": (
        ctypes.c_int32,
        [_P, ctypes.POINTER(_Dispatch), ctypes.c_uint32, ctypes.c_int32, _ERR],
    ),
}


def _load():
    """Load and bind the dylib once, remembering why if it cannot be loaded.

    Failure is recorded rather than raised: a missing or unloadable dylib is an
    ordinary state (never built, wrong platform, quarantined after download)
    whose answer is to run the CPU engine. load_error() carries the reason so
    the UI can say which of those it was.
    """
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib

    if sys.platform != "darwin":
        _load_error = "Metal is macOS-only"
        return None
    if not DYLIB.exists():
        _load_error = f"{DYLIB.name} not built - run scripts/build_native.py"
        return None
    try:
        lib = ctypes.CDLL(str(DYLIB))
    except OSError as exc:
        # The likely cause on someone else's machine: the dylib arrived inside a
        # downloaded zip, so it is quarantined and unsigned. Say so, because
        # "cannot open" on its own sends people looking in the wrong place.
        _load_error = f"could not load {DYLIB.name} ({exc}); it may be unsigned or quarantined"
        return None

    try:
        for name, (restype, argtypes) in _SIGNATURES.items():
            fn = getattr(lib, name)
            fn.restype = restype
            fn.argtypes = argtypes
    except AttributeError as exc:
        _load_error = f"{DYLIB.name} is missing an entry point ({exc}); rebuild it"
        return None

    version = lib.flowx_abi_version()
    if version != FLOWX_ABI:
        _load_error = f"{DYLIB.name} is ABI {version}, expected {FLOWX_ABI}; rebuild it"
        return None

    _lib = lib
    return _lib


def is_available():
    return _load() is not None


def load_error():
    """Why the dylib could not be loaded, or None if it loaded fine."""
    _load()
    return _load_error


def _check(err, what):
    """Raise with the dylib's own message, and release it.

    The message is malloc'd on the C side, so it has to go back through
    flowx_string_free rather than being left to Python.
    """
    if not err.value:
        raise DeviceError(f"{what} failed with no error detail")
    message = err.value.decode("utf-8", "replace")
    _lib.flowx_string_free(err)
    err.value = None
    raise DeviceError(f"{what}: {message}")


class Buffer:
    """A device buffer. Shared storage, so map() is the live memory, not a copy."""

    __slots__ = ("_handle", "nbytes", "__weakref__")

    def __init__(self, backend, nbytes, data=None):
        err = ctypes.c_char_p()
        view = None
        if data is not None:
            view = memoryview(data).cast("B")
            if view.nbytes > nbytes:
                raise DeviceError(f"initial data is {view.nbytes} bytes, buffer is {nbytes}")

        # Hand the data to Metal only when it fills the buffer. A partial upload
        # through newBufferWithBytes: would leave the tail undefined, whereas
        # newBufferWithLength: is documented to zero - and callers rely on the
        # untouched tail of a partly-seeded buffer being zero.
        source = None
        if view is not None and view.nbytes == nbytes:
            source = (ctypes.c_char * view.nbytes).from_buffer_copy(view)

        handle = _lib.flowx_buffer_new(backend._device, nbytes, source, ctypes.byref(err))
        if not handle:
            _check(err, f"allocating {nbytes} bytes")
        self._handle = handle
        self.nbytes = nbytes

        if source is None and view is not None:
            self.write(view)

    def map(self):
        """A writable memoryview over the buffer's memory.

        Valid until the buffer is released. On unified memory this is the same
        pages the GPU sees, so reading state back after a commit() is free -
        no blit, no per-element Python loop.
        """
        if self._handle is None:
            raise DeviceError("buffer has been released")
        pointer = _lib.flowx_buffer_contents(self._handle)
        if not pointer:
            raise DeviceError("buffer has no host-visible contents")
        # Cast to unsigned bytes: a c_char array's memoryview has format "<c",
        # which does not support slice assignment.
        return memoryview((ctypes.c_char * self.nbytes).from_address(pointer)).cast("B")

    def write(self, data, offset=0):
        view = memoryview(data).cast("B")
        if offset + view.nbytes > self.nbytes:
            raise DeviceError(f"write of {view.nbytes} at {offset} overruns {self.nbytes} bytes")
        self.map()[offset : offset + view.nbytes] = view

    def zero(self):
        # ctypes.memset over the mapping: a Python slice assignment of a bytes
        # object that large would materialise the zeros first.
        ctypes.memset(_lib.flowx_buffer_contents(self._handle), 0, self.nbytes)

    def release(self):
        if self._handle is not None:
            _lib.flowx_buffer_destroy(self._handle)
            self._handle = None

    def __del__(self):
        # Guarded: interpreter shutdown can clear module globals before the last
        # buffer is collected, and a NameError in __del__ is only printed, never
        # raised - so check rather than let it happen every run.
        if getattr(self, "_handle", None) is not None and _lib is not None:
            self.release()


class Bindings:
    """A prepared buffer-binding set, reusable across dispatches.

    Every pass in a frame binds the same table (see the BUF_* indices in
    kernels/flowx_prelude.h), and a frame records hundreds of dispatches, so
    rebuilding the pointer array each time was pure overhead. Built once when
    the buffers change and handed to dispatch() as-is.

    Holds the Buffer objects, not just their handles: the array is raw
    pointers, and nothing else would keep the buffers alive.
    """

    __slots__ = ("_handles", "_buffers", "count")

    def __init__(self, buffers):
        self._buffers = list(buffers)
        self.count = len(self._buffers)
        self._handles = (ctypes.c_void_p * self.count)()
        for i, buf in enumerate(self._buffers):
            self._handles[i] = buf._handle if buf is not None else None

    @property
    def pointer(self):
        return ctypes.cast(self._handles, ctypes.POINTER(ctypes.c_void_p))


class Kernel:
    """A compiled entry point, ready to dispatch.

    Holds its Program. Program.release() frees the kernels it handed out, so
    without this back-reference the perfectly reasonable

        kernel = backend.program(src).kernel("step")

    would leave `kernel` pointing at a freed pipeline as soon as the temporary
    Program was collected - and the failure surfaces later, inside submit, as
    a null kernel rather than at the line that caused it.
    """

    __slots__ = ("_handle", "_program", "name", "max_threads_per_group", "execution_width")

    def __init__(self, program, name):
        err = ctypes.c_char_p()
        handle = _lib.flowx_kernel_new(program._handle, name.encode(), ctypes.byref(err))
        if not handle:
            _check(err, f"compiling kernel {name!r}")
        self._handle = handle
        self._program = program
        self.name = name
        self.max_threads_per_group = _lib.flowx_kernel_max_threads_per_group(handle)
        self.execution_width = _lib.flowx_kernel_execution_width(handle)

    def release(self):
        if self._handle is not None:
            _lib.flowx_kernel_destroy(self._handle)
            self._handle = None

    def __del__(self):
        if getattr(self, "_handle", None) is not None and _lib is not None:
            self.release()


class Program:
    """Kernel source compiled into a library, from which entry points are drawn."""

    __slots__ = ("_handle", "_kernels")

    def __init__(self, backend, source):
        err = ctypes.c_char_p()
        handle = _lib.flowx_library_new(backend._device, source.encode(), ctypes.byref(err))
        if not handle:
            _check(err, "compiling kernel source")
        self._handle = handle
        self._kernels = {}

    def kernel(self, name):
        """The named entry point, compiled once and cached.

        Cached because a pipeline state is expensive to build and the solver
        asks for the same handful every frame.
        """
        if name not in self._kernels:
            self._kernels[name] = Kernel(self, name)
        return self._kernels[name]

    def release(self):
        for kernel in self._kernels.values():
            kernel.release()
        self._kernels.clear()
        if self._handle is not None:
            _lib.flowx_library_destroy(self._handle)
            self._handle = None

    def __del__(self):
        if getattr(self, "_handle", None) is not None and _lib is not None:
            self.release()


class Queue:
    """Records dispatches, then runs them all in one crossing into the dylib."""

    __slots__ = ("_handle", "_records", "_keepalive")

    def __init__(self, backend):
        err = ctypes.c_char_p()
        handle = _lib.flowx_queue_new(backend._device, ctypes.byref(err))
        if not handle:
            _check(err, "creating command queue")
        self._handle = handle
        self._records = []
        # Holds the ctypes arrays backing the recorded dispatches. They must
        # outlive the submit call, and nothing else refers to them.
        self._keepalive = []

    def dispatch(self, kernel, threads, group, buffers, constants=None, constants_index=None):
        """Record one dispatch. Nothing runs until commit().

        `buffers` is a Bindings, or a sequence bound at indices 0..n-1 with a
        None entry leaving that index unbound - a pass that ignores a slot costs
        nothing for it. `constants` is a bytes-like parameter block, bound after
        the buffers by default.
        """
        if threads <= 0:
            return
        if not isinstance(buffers, Bindings):
            buffers = Bindings(buffers)
        self._keepalive.append(buffers)

        blob = None
        size = 0
        if constants is not None:
            view = memoryview(constants).cast("B")
            size = view.nbytes
            if size > MAX_CONSTANTS:
                raise DeviceError(f"constants are {size} bytes, limit is {MAX_CONSTANTS}")
            blob = (ctypes.c_char * size).from_buffer_copy(view)
            self._keepalive.append(blob)

        self._records.append(
            _Dispatch(
                kernel=kernel._handle,
                buffers=buffers.pointer,
                nbuffers=buffers.count,
                constants=ctypes.cast(blob, ctypes.c_void_p) if blob is not None else None,
                constants_size=size,
                constants_index=buffers.count if constants_index is None else constants_index,
                threads=threads,
                threads_per_group=min(group, kernel.max_threads_per_group),
            )
        )

    def commit(self, wait=True):
        """Run everything recorded, then clear the recording.

        `wait` blocks until the GPU is finished, which is what makes a
        subsequent Buffer.map() see this submission's results.
        """
        if not self._records:
            return
        count = len(self._records)
        array = (_Dispatch * count)(*self._records)
        err = ctypes.c_char_p()
        # Cleared before the call, not after: if submit raises, the recording
        # must not be replayed by the next commit.
        self._records = []
        keepalive = self._keepalive
        self._keepalive = []
        status = _lib.flowx_submit(self._handle, array, count, 1 if wait else 0, ctypes.byref(err))
        del keepalive
        if status != 0:
            _check(err, f"submitting {count} dispatches")

    @property
    def pending(self):
        return len(self._records)

    def release(self):
        if self._handle is not None:
            _lib.flowx_queue_destroy(self._handle)
            self._handle = None

    def __del__(self):
        if getattr(self, "_handle", None) is not None and _lib is not None:
            self.release()


class MetalBackend:
    """A Metal device, its programs, buffers and queues."""

    __slots__ = ("_device", "caps", "max_buffer_length")

    def __init__(self):
        err = ctypes.c_char_p()
        device = _lib.flowx_device_create(ctypes.byref(err))
        if not device:
            _check(err, "creating Metal device")
        self._device = device
        self.max_buffer_length = _lib.flowx_device_max_buffer_length(device)
        self.caps = Caps(
            name="metal",
            device=_lib.flowx_device_name(device).decode("utf-8", "replace"),
            # Unlike Blender's Metal backend, which could not compile an image
            # atomic at all, buffer atomics are ordinary MSL here.
            atomics=True,
            max_threads_per_group=1024,
            unified_memory=bool(_lib.flowx_device_has_unified_memory(device)),
        )

    def buffer(self, nbytes, data=None):
        return Buffer(self, nbytes, data)

    def bindings(self, buffers):
        """Prepare a reusable binding set. See Bindings."""
        return Bindings(buffers)

    def program(self, sources):
        """Compile kernel source. A list is concatenated, prelude first."""
        if not isinstance(sources, str):
            sources = "\n".join(sources)
        return Program(self, sources)

    def queue(self):
        return Queue(self)

    def release(self):
        if self._device is not None:
            _lib.flowx_device_destroy(self._device)
            self._device = None

    def __repr__(self):
        return f"<MetalBackend {self.caps.device}>"


_backend = None


def create():
    """The process's Metal backend, created once.

    A singleton because buffers, pipelines and queues all belong to the device
    that made them and cannot be mixed across devices. The collider grid is
    allocated from collision/, the particle state from the engine, and the two
    have to end up on the same device for a kernel to see both.
    """
    global _backend
    if _load() is None:
        raise DeviceError(load_error() or "Metal backend unavailable")
    if _backend is None:
        _backend = MetalBackend()
    return _backend
