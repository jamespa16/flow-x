"""Tests for the Metal backend, runnable without Blender.

    python3 scripts/test_backend.py

Deliberately Blender-free, the same way scripts/test_marching_cubes.py is: this
answers "can we drive the GPU on this machine at all", which is the first
question to ask when the solver misbehaves, and mixing Blender into it would
make the answer ambiguous. solver/backend/ imports neither bpy nor gpu, so the
package is loaded here by path to avoid solver/__init__.py, which does.

Skips cleanly rather than failing when there is no usable device - a Linux
runner or a checkout with no bin/ built is not a broken backend.
"""

import ctypes
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flowx_standalone import load  # noqa: E402

backend_pkg = load("solver.backend")
engine_kernels = load("solver.engine.kernels")
engine_params = load("solver.engine.params")

# A single library holding every kernel the tests need. Kept in one string so a
# compile failure is reported once, with line numbers that match what is here.
SOURCE = """
#include <metal_stdlib>
using namespace metal;

struct Params { uint count; float scale; float bias; };

kernel void scale_bias(device float *values [[buffer(0)]],
                       constant Params &p [[buffer(1)]],
                       uint i [[thread_position_in_grid]])
{
  if (i >= p.count) { return; }
  values[i] = values[i] * p.scale + p.bias;
}

kernel void count_above(device const float *values [[buffer(0)]],
                        device atomic_uint *counter [[buffer(1)]],
                        constant Params &p [[buffer(2)]],
                        uint i [[thread_position_in_grid]])
{
  if (i >= p.count) { return; }
  if (values[i] > p.bias) {
    atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
  }
}

/* Reads every element written by the previous dispatch, not just its own, so
 * it can only produce the right answer if dispatches are ordered and the
 * previous one's writes are visible. That is what the solver's pass chain
 * assumes, and what flowx_submit's serial encoder is there to provide. */
kernel void sum_then_store(device const float *src [[buffer(0)]],
                           device float *dst [[buffer(1)]],
                           constant Params &p [[buffer(2)]],
                           uint i [[thread_position_in_grid]])
{
  if (i >= p.count) { return; }
  float total = 0.0f;
  for (uint k = 0; k < p.count; ++k) { total += src[k]; }
  dst[i] = total;
}

kernel void increment(device float *values [[buffer(0)]],
                      constant Params &p [[buffer(1)]],
                      uint i [[thread_position_in_grid]])
{
  if (i >= p.count) { return; }
  values[i] = values[i] + 1.0f;
}
"""


def params(count, scale=1.0, bias=0.0):
    return struct.pack("<Iff", count, scale, bias)


def floats(buf, count):
    return list((ctypes.c_float * count).from_buffer(buf.map()))


class Failure(AssertionError):
    pass


def check(condition, message):
    if not condition:
        raise Failure(message)


def test_device_reports_itself(backend):
    caps = backend.caps
    check(caps.name == "metal", f"backend name is {caps.name!r}")
    check(bool(caps.device), "device reported no name")
    check(caps.atomics, "Metal backend must report atomics support")
    check(caps.max_threads_per_group >= 64, f"threadgroup cap {caps.max_threads_per_group} too low")
    print(f"  device: {caps.device} (unified memory: {caps.unified_memory})")


def test_buffer_roundtrip(backend):
    data = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    buf = backend.buffer(16, data)
    check(floats(buf, 4) == [1.0, 2.0, 3.0, 4.0], "buffer did not round-trip its initial data")

    buf.write(struct.pack("<f", 9.0), offset=4)
    check(floats(buf, 4) == [1.0, 9.0, 3.0, 4.0], "offset write landed in the wrong place")

    buf.zero()
    check(floats(buf, 4) == [0.0] * 4, "zero() left non-zero contents")

    empty = backend.buffer(16)
    check(floats(empty, 4) == [0.0] * 4, "a fresh buffer must be zero-filled")


def test_dispatch_with_constants(backend, program):
    n = 1000
    buf = backend.buffer(n * 4, struct.pack(f"<{n}f", *(float(i) for i in range(n))))
    queue = backend.queue()
    queue.dispatch(program.kernel("scale_bias"), n, 64, [buf], constants=params(n, 2.0, 1.5))
    queue.commit()
    got = floats(buf, n)
    check(got[:4] == [1.5, 3.5, 5.5, 7.5], f"scale_bias produced {got[:4]}")
    check(got[-1] == 2.0 * (n - 1) + 1.5, f"last element is {got[-1]}")


def test_atomics(backend, program):
    """The capability the whole migration turns on.

    Blender's Metal backend could not compile an image atomic at all, which is
    why the solver's grid build is a bitonic sort. If this fails, the counting
    sort that replaces it is off the table.
    """
    n = 1024
    values = backend.buffer(n * 4, struct.pack(f"<{n}f", *(float(i) for i in range(n))))
    counter = backend.buffer(4)
    queue = backend.queue()
    queue.dispatch(
        program.kernel("count_above"),
        n,
        64,
        [values, counter],
        constants=params(n, 1.0, 511.5),
        constants_index=2,
    )
    queue.commit()
    got = struct.unpack("<I", bytes(counter.map()))[0]
    check(got == 512, f"atomic counter is {got}, expected 512")


def test_dispatches_are_ordered(backend, program):
    """Each recorded dispatch must observe the previous one's writes.

    The solver's pass chain (predict -> grid -> lambda -> delta -> ...) is built
    on this. Metal only guarantees it for a serial-dispatch encoder, so this is
    really a test of flowx_submit's encoder type.
    """
    n = 256
    buf = backend.buffer(n * 4)
    queue = backend.queue()
    rounds = 120
    for _ in range(rounds):
        queue.dispatch(program.kernel("increment"), n, 64, [buf], constants=params(n))
    check(queue.pending == rounds, f"recorded {queue.pending} dispatches, expected {rounds}")
    queue.commit()
    check(queue.pending == 0, "commit() did not clear the recording")
    got = floats(buf, n)
    check(
        all(v == float(rounds) for v in got),
        f"after {rounds} chained increments the buffer holds {got[0]}..{max(got)}",
    )

    # A read-all-of-the-previous-output pass, which cannot be satisfied by luck.
    src = backend.buffer(n * 4, struct.pack(f"<{n}f", *([1.0] * n)))
    dst = backend.buffer(n * 4)
    queue.dispatch(program.kernel("increment"), n, 64, [src], constants=params(n))
    queue.dispatch(
        program.kernel("sum_then_store"), n, 64, [src, dst], constants=params(n), constants_index=2
    )
    queue.commit()
    total = floats(dst, 1)[0]
    check(total == 2.0 * n, f"cross-dispatch sum is {total}, expected {2.0 * n}")


def test_errors_are_reported(backend, program):
    errors = backend_pkg.DeviceError

    try:
        backend.program("kernel void broken(")
        raise Failure("a malformed kernel compiled without error")
    except errors as exc:
        check("compil" in str(exc).lower(), f"compile error was unhelpful: {exc}")

    try:
        program.kernel("no_such_kernel")
        raise Failure("a missing entry point resolved")
    except errors as exc:
        check("no_such_kernel" in str(exc), f"missing-kernel error was unhelpful: {exc}")

    queue = backend.queue()
    buf = backend.buffer(64)
    try:
        queue.dispatch(program.kernel("increment"), 16, 64, [buf], constants=b"\0" * 8192)
        raise Failure("oversized constants were accepted")
    except errors as exc:
        check("limit" in str(exc), f"constants-limit error was unhelpful: {exc}")

    try:
        buf.write(b"\0" * 128)
        raise Failure("an overrunning write was accepted")
    except errors as exc:
        check("overrun" in str(exc), f"overrun error was unhelpful: {exc}")


def test_kernel_outlives_a_temporary_program(backend):
    """A kernel drawn from a temporary Program must stay usable.

    Regression: Program.release() frees the kernels it handed out, so without a
    back-reference from Kernel the idiomatic one-liner below left a dangling
    pipeline - and it failed inside submit, far from the cause.
    """
    import gc

    kernel = backend.program(SOURCE).kernel("increment")
    gc.collect()
    buf = backend.buffer(16, struct.pack("<4f", 1.0, 1.0, 1.0, 1.0))
    queue = backend.queue()
    queue.dispatch(kernel, 4, 64, [buf], constants=params(4))
    queue.commit()
    check(floats(buf, 4) == [2.0] * 4, "kernel from a temporary program did not run")


def test_zero_threads_is_a_no_op(backend, program):
    """The solver records dispatches whose count can legitimately be zero."""
    buf = backend.buffer(16, struct.pack("<4f", 5.0, 5.0, 5.0, 5.0))
    queue = backend.queue()
    queue.dispatch(program.kernel("increment"), 0, 64, [buf], constants=params(4))
    check(queue.pending == 0, "a zero-thread dispatch should not be recorded")
    queue.commit()
    check(floats(buf, 4) == [5.0] * 4, "a zero-thread dispatch changed the buffer")


def test_solver_kernel_libraries_compile(backend):
    """PBF and APIC compile independently and expose every declared pass."""
    for label, passes in (
        ("PBF", engine_kernels.PBF_ALL_PASSES),
        ("APIC", engine_kernels.APIC_ALL_PASSES),
    ):
        program = backend.program(engine_kernels.library_source(passes))
        for name in engine_kernels.entry_points(passes):
            program.kernel(name)
        print(f"  {label}: {len(passes)} kernels")


def test_params_layout(backend):
    """The Python parameter block matches Metal through every APIC field."""
    source = "\n".join(
        (
            engine_kernels.read("flowx_prelude.h", suffix=""),
            engine_kernels.read("flowx_params_probe"),
        )
    )
    program = backend.program(source)
    count = 15
    out = backend.buffer(count * 4)
    queue = backend.queue()
    queue.dispatch(
        program.kernel("flowx_params_probe"),
        1,
        1,
        [out],
        constants=engine_params.ParamBlock().pack(),
        constants_index=20,
    )
    queue.commit()
    got = struct.unpack(f"<{count}I", bytes(out.map()))
    fields = (
        "particle_count",
        "lo_x",
        "smoothing_radius",
        "collider_voxel",
        "surface_kernel_radius",
        "ww_capacity",
        "frame_dt",
        "nodes_x",
        "nodes_y",
        "nodes_z",
        "grid_spacing",
        "vorticity_epsilon",
        "grid_max_speed",
        "pressure_ping",
    )
    expected = (engine_params.PARAMS_SIZE,) + tuple(
        engine_params.offset_of(name) for name in fields
    )
    check(got == expected, f"Params layout is {got}, expected {expected}")


def main():
    metal = backend_pkg.select()
    if metal is None:
        reason = load("solver.backend.metal").load_error()
        print(f"skipped: no usable GPU backend ({reason})")
        return 0

    program = metal.program(SOURCE)
    tests = [
        ("device reports itself", lambda: test_device_reports_itself(metal)),
        ("buffer round-trip", lambda: test_buffer_roundtrip(metal)),
        ("dispatch with constants", lambda: test_dispatch_with_constants(metal, program)),
        ("atomics", lambda: test_atomics(metal, program)),
        ("dispatches are ordered", lambda: test_dispatches_are_ordered(metal, program)),
        ("errors are reported", lambda: test_errors_are_reported(metal, program)),
        (
            "kernel outlives a temporary program",
            lambda: test_kernel_outlives_a_temporary_program(metal),
        ),
        ("zero threads is a no-op", lambda: test_zero_threads_is_a_no_op(metal, program)),
        ("solver kernel libraries compile", lambda: test_solver_kernel_libraries_compile(metal)),
        ("parameter layout", lambda: test_params_layout(metal)),
    ]

    failures = 0
    for name, run in tests:
        try:
            run()
        except Failure as exc:
            print(f"FAIL {name}: {exc}")
            failures += 1
        else:
            print(f"ok   {name}")

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
