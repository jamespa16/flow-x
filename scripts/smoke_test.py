"""Headless smoke test for the Flow-X extension.

Verifies the extension installs and enables cleanly in Blender 4.2+, with
zero errors, on a clean checkout - the Phase 0 exit criterion. Later phases
should append their new operators to OPERATORS_TO_SMOKE_TEST below and this
script will exercise each one after enabling the extension.

Run from the repo root (after `python3 scripts/dev_link.py`):
    blender --background --python scripts/smoke_test.py
"""

import math
import sys

import bpy

ADDON_MODULE = "bl_ext.user_default.flow_x"

# Set by _init_gpu(): whether this run has a GPU context, and so whether the
# solver's compute passes are expected to actually run.
_gpu_available = False

# (operator idname, kwargs) pairs to run once the extension is enabled.
OPERATORS_TO_SMOKE_TEST = [
    ("flowx.domain_add", {}),
    # A collider inside the domain, so the surface has something to wrap
    # around - Phase 6's exit criterion covers the two together.
    ("mesh.primitive_cube_add", {"size": 0.5, "location": (0, 0, 0)}),
    ("flowx.toggle_collider", {}),
    ("flowx.solver_gpu_test_toggle", {}),
    ("flowx.sph_toggle", {}),
]

# Operators that only make sense while the solver is running, so they run after
# the toggle above rather than being folded into that list (which also runs
# with no GPU context, where the solver never starts).
SOLVER_OPERATORS_TO_SMOKE_TEST = [
    ("flowx.sph_reset", {}),
]


def _get_operator(idname):
    module_name, func_name = idname.split(".")
    return getattr(getattr(bpy.ops, module_name), func_name)


def _init_gpu():
    """Bring up a GPU context in background mode, where Blender offers one.

    Without this the solver operators report a warning and no-op, so nothing
    would ever compile a compute shader in CI. gpu.init() only exists in
    Blender 5.0+; on older builds this stays a compile-free smoke test.
    """
    global _gpu_available
    import gpu

    if not bpy.app.background:
        _gpu_available = True
        return
    if not hasattr(gpu, "init"):
        print("[smoke_test] gpu.init() unavailable; GPU passes will be skipped")
        return
    try:
        gpu.init()
    except Exception as exc:
        print(f"[smoke_test] gpu.init() failed ({exc}); GPU passes will be skipped")
    else:
        _gpu_available = True
        print(f"[smoke_test] GPU backend: {gpu.platform.backend_type_get()}")


def _check_solver():
    """Confirm the SPH solver really started, and step it a few frames.

    The solver operators deliberately report a warning instead of raising when
    there's no GPU context, so a green operator run alone would not catch a
    broken compute shader. Where a GPU is available, insist that the last
    toggle in OPERATORS_TO_SMOKE_TEST actually left the solver running.
    """
    if not _gpu_available:
        print("[smoke_test] no GPU context; skipping solver step check")
        return

    sph = sys.modules[ADDON_MODULE].solver.sph
    if not sph.is_running():
        raise RuntimeError("flowx.sph_toggle did not start the solver (see warnings above)")

    scene = bpy.context.scene
    for frame in range(1, 4):
        scene.frame_set(frame)
    stats = sph.stats()
    print(f"[smoke_test] stepped SPH solver 3 frames: {stats}")

    _check_particles(sph, stats)
    _check_surface(sph, stats)
    _check_playback(sph)


def _check_playback(sph):
    """Phase 7: the handler's three timeline cases, and the ms/step readout.

    The solver has no cache, so the contract is: re-seed at or before the
    scene's start frame, step forward, and hold with a warning on a backward
    scrub rather than showing a frame that was never simulated.
    """
    for idname, kwargs in SOLVER_OPERATORS_TO_SMOKE_TEST:
        result = _get_operator(idname)(**kwargs)
        if "FINISHED" not in result:
            raise RuntimeError(f"{idname} returned {result}, expected FINISHED")
        print(f"[smoke_test] ran {idname} -> {result}")

    scene = bpy.context.scene
    scene.frame_set(scene.frame_start + 4)
    stats = sph.stats()
    if stats["frame"] != scene.frame_current:
        raise RuntimeError(
            f"solver is at frame {stats['frame']}, expected {scene.frame_current} "
            "after stepping forward"
        )
    if stats["warning"]:
        raise RuntimeError(f"unexpected playback warning stepping forward: {stats['warning']}")
    if stats["step_ms"] is None:
        raise RuntimeError("no ms/step timing was recorded while stepping")

    # Backward scrub: hold the simulated frame, and say so.
    simulated = stats["frame"]
    scene.frame_set(scene.frame_start + 2)
    stats = sph.stats()
    if stats["frame"] != simulated:
        raise RuntimeError(
            f"solver stepped to {stats['frame']} on a backward scrub; it should have "
            f"held at {simulated}"
        )
    if not stats["warning"]:
        raise RuntimeError("a backward scrub should warn rather than silently hold")

    # The start frame is the run's origin: re-seed and clear the warning.
    scene.frame_set(scene.frame_start)
    stats = sph.stats()
    if stats["frame"] != scene.frame_start:
        raise RuntimeError(f"scrubbing to the start frame did not re-seed (at {stats['frame']})")
    if stats["warning"]:
        raise RuntimeError(f"re-seeding left a stale warning: {stats['warning']}")

    print(f"[smoke_test] playback: re-seeded at frame {scene.frame_start}, backward scrub warned")


def _check_particles(sph, stats):
    """The debug overlay, which the domain has to opt into since Phase 6."""
    domain = sys.modules[ADDON_MODULE].domain.find_domain(bpy.context.scene)
    if not domain.flowx_domain.show_particles:
        if sph.viz.points():
            raise RuntimeError("particle overlay is off but points were still read back")
        return

    points = sph.viz.points()
    if len(points) != stats["particles"]:
        raise RuntimeError(f"expected {stats['particles']} particles, read back {len(points)}")
    if not all(all(math.isfinite(c) for c in p) for p in points):
        raise RuntimeError("solver produced non-finite particle positions")


def _check_surface(sph, stats):
    """Phase 6: the extracted surface mesh is the add-on's actual output.

    The exit criterion is a continuous surface rather than points, so a green
    dispatch isn't enough - insist on a real mesh with faces, sitting inside
    the domain bounds.
    """
    surface_stats = stats["surface"]
    if surface_stats is None:
        raise RuntimeError("surface extraction did not start (see warnings above)")

    domain = sys.modules[ADDON_MODULE].domain.find_domain(bpy.context.scene)
    name = domain.name + sys.modules[ADDON_MODULE].solver.surface.SURFACE_SUFFIX
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise RuntimeError(f"expected a surface object named {name!r}")
    if obj.parent is not domain:
        raise RuntimeError(f"{name} should be parented to the domain")

    mesh = obj.data
    if len(mesh.polygons) == 0 or len(mesh.vertices) == 0:
        raise RuntimeError(
            f"{name} has no geometry ({len(mesh.vertices)} verts, "
            f"{len(mesh.polygons)} faces) - the fluid should have a surface by now"
        )
    if not mesh.materials:
        raise RuntimeError(f"{name} was built without a material")

    lo, hi = sys.modules[ADDON_MODULE].domain.world_bounds(domain)
    matrix = obj.matrix_world
    tolerance = surface_stats["spacing"] * 2.0
    for vertex in mesh.vertices:
        point = matrix @ vertex.co
        for axis in range(3):
            if not (lo[axis] - tolerance <= point[axis] <= hi[axis] + tolerance):
                raise RuntimeError(f"surface vertex {tuple(point)} lies outside the domain")

    print(
        f"[smoke_test] surface mesh: {len(mesh.vertices)} verts, "
        f"{len(mesh.polygons)} faces from {surface_stats['samples']} samples"
    )


def main():
    _init_gpu()
    bpy.ops.extensions.repo_refresh_all()

    bpy.ops.preferences.addon_enable(module=ADDON_MODULE)
    if ADDON_MODULE not in bpy.context.preferences.addons:
        raise RuntimeError(f"{ADDON_MODULE} did not appear in enabled addons after enable")
    print(f"[smoke_test] enabled {ADDON_MODULE}")

    for idname, kwargs in OPERATORS_TO_SMOKE_TEST:
        result = _get_operator(idname)(**kwargs)
        if "FINISHED" not in result:
            raise RuntimeError(f"{idname} returned {result}, expected FINISHED")
        print(f"[smoke_test] ran {idname} -> {result}")

    _check_solver()

    bpy.ops.preferences.addon_disable(module=ADDON_MODULE)
    if ADDON_MODULE in bpy.context.preferences.addons:
        raise RuntimeError(f"{ADDON_MODULE} still enabled after disable")
    print(f"[smoke_test] disabled {ADDON_MODULE}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        print("SMOKE TEST FAILED")
        sys.exit(1)
    else:
        print("SMOKE TEST PASSED")
        sys.exit(0)
