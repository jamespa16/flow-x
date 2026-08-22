"""Headless smoke test for the Flow-X extension.

Verifies the extension installs and enables cleanly in Blender 5.2+, with
zero errors, on a clean checkout - the Phase 0 exit criterion. Later phases
should append their new operators to OPERATORS_TO_SMOKE_TEST below and this
script will exercise each one after enabling the extension.

Run from the repo root (after `python3 scripts/dev_link.py`):
    blender --background --python scripts/smoke_test.py
"""

import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

ADDON_MODULE = "bl_ext.user_default.flow_x"
REPO_ROOT = Path(__file__).resolve().parent.parent

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
    _check_pbf_convergence(sph, stats)
    _check_surface(sph, stats)
    _check_collider_grids()
    _check_playback(sph)
    _check_cache(sph)


def _check_pbf_convergence(sph, stats):
    """PBF's actual promise: the density constraint should hold near rest.

    WCSPH never had this property to check - its pressure was a spring
    reacting to density error, not a projection onto it. lambda_img.x holds
    each particle's density as of its last constraint-solve iteration (see
    sph_lambda.glsl); after a few frames of settling under gravity alone
    (no collider/wall contact forcing local compression), the mean should sit
    close to rest_density if the constraint loop is doing its job.
    """
    mod = sys.modules[ADDON_MODULE]
    gpu_util = mod.solver.gpu_util
    config = sph._state["config"]
    densities = gpu_util.read_texture(
        sph._state["textures"]["lambda_img"], config.particle_count, channels=1
    )
    mean_density = sum(densities) / len(densities)
    rest_density = config.rest_density
    error = abs(mean_density - rest_density) / rest_density
    print(
        f"[smoke_test] PBF mean density {mean_density:.1f} vs rest "
        f"{rest_density:.1f} ({error * 100:.1f}% error)"
    )
    if error > 0.15:
        raise RuntimeError(
            f"PBF density constraint not converging: {error * 100:.1f}% mean error "
            f"after {stats['frame']} frames"
        )


def _check_playback(sph):
    """Phase 7: the handler's timeline cases without a cache, and the ms/step readout.

    The disk cache is off by default, so the contract here is the uncached
    one: re-seed at or before the scene's start frame, step forward, and hold
    with a warning on a backward scrub rather than showing a frame that was
    never simulated. _check_cache covers the enabled path.
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


def _check_cache(sph):
    """The disk cache: write-through, scrub-back loads, invalidation, clear.

    Caching is off by default, so first confirm the uncached contract still
    holds, then enable it and check: the file grows one frame per simulated
    frame, a backward scrub loads the frame without a warning, loading the
    same frame twice round-trips exactly, a settings change discards the old
    file, and Clear Cache deletes it and restores the warn-and-hold behavior.
    """
    mod = sys.modules[ADDON_MODULE]
    cache = mod.solver.cache
    gpu_util = mod.solver.gpu_util
    scene = bpy.context.scene
    domain = mod.domain.find_domain(scene)
    settings = domain.flowx_domain
    start = scene.frame_start

    def positions():
        return gpu_util.read_texture(
            sph._state["textures"]["positions_img"], sph._state["config"].particle_count
        )

    # Cache off: a backward scrub holds and warns, as before.
    scene.frame_set(start + 3)
    scene.frame_set(start + 1)
    stats = sph.stats()
    if stats["frame"] != start + 3:
        raise RuntimeError(
            f"backward scrub with no cache should hold at {start + 3}, solver at {stats['frame']}"
        )
    if not stats["warning"]:
        raise RuntimeError("backward scrub with no cache should warn")

    # Enable and re-seed: a fresh file, no frames yet.
    settings.cache_enabled = True
    result = _get_operator("flowx.sph_reset")()
    if "FINISHED" not in result:
        raise RuntimeError(f"flowx.sph_reset with caching on returned {result}")
    info = cache.info()
    if not info["open"]:
        raise RuntimeError("caching is enabled but no cache file is open")
    if info["frames"] is not None:
        raise RuntimeError(f"fresh cache should hold no frames, holds {info['frames']}")
    path = cache.cache_path(domain, scene)

    # Every simulated frame lands on disk.
    for frame in (start + 1, start + 2, start + 3):
        scene.frame_set(frame)
    # The reset above re-seeded at frame start + 1 (the scene was parked
    # there), so the seeded frame itself is not stored: the file holds the
    # frames simulated after it.
    info = cache.info()
    if info["frames"] != (start + 2, start + 3):
        raise RuntimeError(
            f"cache should hold frames {start + 2}-{start + 3}, holds {info['frames']}"
        )
    if not path.exists():
        raise RuntimeError(f"cache file is missing at {path}")

    # Backward scrub: the frame loads, with no warning.
    scene.frame_set(start + 2)
    stats = sph.stats()
    if stats["frame"] != start + 2:
        raise RuntimeError(
            f"cached backward scrub should be at frame {start + 2}, solver at {stats['frame']}"
        )
    if stats["warning"]:
        raise RuntimeError(f"cached backward scrub should not warn: {stats['warning']}")
    loaded = positions()
    if not all(math.isfinite(c) for p in loaded for c in p):
        raise RuntimeError("the loaded cache frame has non-finite particle positions")

    # A forward jump into the cached range loads rather than simulates.
    scene.frame_set(start + 3)
    stats = sph.stats()
    if stats["frame"] != start + 3 or stats["warning"]:
        raise RuntimeError(
            f"forward jump into the cached range should load cleanly "
            f"(at {stats['frame']}, warning: {stats['warning']})"
        )

    # Loading the same frame again round-trips exactly. (The seeded frame
    # itself is never stored, so the round trip goes through a second,
    # different cached frame on the way back.)
    scene.frame_set(start + 3)
    scene.frame_set(start + 2)
    if positions() != loaded:
        raise RuntimeError("loading the same cached frame twice did not round-trip exactly")

    # A settings change moves the config hash. Until the next Reset the old
    # file is stale: the first load attempt must close it - and frames
    # simulated under the new state must not be appended under the old
    # header.
    old_hash = cache.config_hash(domain, scene)
    settings.fluid_level = 75.0
    if cache.config_hash(domain, scene) == old_hash:
        raise RuntimeError("changing the fluid level did not change the config hash")
    # The solver sits at start + 2 and the file holds start + 2..start + 3,
    # so this lands inside the covered range and forces a (rejected) load.
    scene.frame_set(start + 3)
    info = cache.info()
    if info["open"]:
        raise RuntimeError("a stale cache must close when its hash no longer matches")
    if not info["warning"]:
        raise RuntimeError("a stale cache should leave its warning in the panel")
    stale = cache._read_existing_header(path)
    if stale is None or stale["last_frame"] != start + 3:
        got = stale["last_frame"] if stale is not None else "no file"
        raise RuntimeError(
            f"the stale file must not grow under the new settings "
            f"(last frame {got}, expected {start + 3})"
        )

    # The next Reset starts a fresh file.
    result = _get_operator("flowx.sph_reset")()
    if "FINISHED" not in result:
        raise RuntimeError(f"flowx.sph_reset after a settings change returned {result}")
    info = cache.info()
    if info["frames"] is not None:
        raise RuntimeError(
            f"changed settings should have discarded the old cache, holds {info['frames']}"
        )

    # The run continues from the fresh file; the stale tail is not loaded.
    for frame in (start + 1, start + 2):
        scene.frame_set(frame)
    scene.frame_set(start + 5)
    stats = sph.stats()
    if stats["frame"] != start + 5:
        raise RuntimeError(
            f"forward jump after invalidation should have simulated to {start + 5}, "
            f"solver at {stats['frame']}"
        )
    if stats["warning"]:
        raise RuntimeError(
            f"forward simulation after invalidation should not warn: {stats['warning']}"
        )

    # Clear Cache deletes the file; the uncached contract returns.
    result = _get_operator("flowx.cache_clear")()
    if "FINISHED" not in result:
        raise RuntimeError(f"flowx.cache_clear returned {result}")
    if path.exists():
        raise RuntimeError(f"cache file still exists after clear: {path}")
    scene.frame_set(start + 1)
    stats = sph.stats()
    if stats["frame"] != start + 5:
        raise RuntimeError("with the cache cleared, a backward scrub should hold")
    if not stats["warning"]:
        raise RuntimeError("with the cache cleared, a backward scrub should warn")

    print(f"[smoke_test] disk cache: write, load, invalidation, clear ({path})")


def _check_collider_grids():
    """Colliders overlapping the domain must have non-empty voxel grids.

    Without this, a collider that voxelizes to nothing (e.g. a parity query
    in the wrong coordinate space) passes every other check: the surface
    still extracts, the particles still fall - the fluid just passes straight
    through the obstacle.
    """
    mod = sys.modules[ADDON_MODULE]
    collision = mod.collision
    domain = mod.domain.find_domain(bpy.context.scene)
    dom_lo, dom_hi = mod.domain.world_bounds(domain)
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or not obj.flowx_collider.is_collider:
            continue
        obj_lo, obj_hi = mod.domain.world_bounds(obj)
        overlaps = not (
            obj_lo.x >= dom_hi.x
            or obj_hi.x <= dom_lo.x
            or obj_lo.y >= dom_hi.y
            or obj_hi.y <= dom_lo.y
            or obj_lo.z >= dom_hi.z
            or obj_hi.z <= dom_lo.z
        )
        if not overlaps:
            continue
        voxels = collision.occupied_count(obj.name)
        if voxels == 0:
            raise RuntimeError(
                f"collider '{obj.name}' overlaps the domain but voxelized to nothing"
            )
        print(f"[smoke_test] collider '{obj.name}': {voxels} voxels")


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

    mod = sys.modules[ADDON_MODULE]
    lo, hi = mod.domain.world_bounds(domain)
    matrix = obj.matrix_world
    tolerance = surface_stats["spacing"] * 2.0
    points = [matrix @ vertex.co for vertex in mesh.vertices]
    for point in points:
        for axis in range(3):
            if not (lo[axis] - tolerance <= point[axis] <= hi[axis] + tolerance):
                raise RuntimeError(f"surface vertex {tuple(point)} lies outside the domain")

    _check_surface_collider_penetration(points, surface_stats["spacing"])

    print(
        f"[smoke_test] surface mesh: {len(mesh.vertices)} verts, "
        f"{len(mesh.polygons)} faces from {surface_stats['samples']} samples"
    )


_AXIS_DIRECTIONS = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


def _collider_bvh(obj, depsgraph):
    """A world-space BVH of the collider's evaluated mesh.

    BVHTree.FromObject builds in the object's *local* space (which is why the
    voxelizer in collision transforms its query rays), and local distances
    are not world distances under non-uniform scale. This test runs in world
    space, so it builds the tree from the world-space polygons instead.
    """
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        verts = [eval_obj.matrix_world @ vert.co for vert in mesh.vertices]
        # Materialized, not the lazy RNA proxies: to_mesh_clear() frees the
        # mesh before FromPolygons would iterate them.
        faces = [tuple(poly.vertices) for poly in mesh.polygons]
    finally:
        eval_obj.to_mesh_clear()
    return BVHTree.FromPolygons(verts, faces)


def _first_hit_parity(bvh, point, direction, max_bounces=8):
    """(first hit distance or None, hit-count parity) for a ray from `point`.

    Walks the hit chain like the voxelizer's inside/outside test
    (collision._point_inside), so a closed collider reads as odd from inside
    and even from outside.
    """
    origin = point
    first = None
    count = 0
    for _ in range(max_bounces):
        hit, _normal, _index, dist = bvh.ray_cast(origin, direction)
        if hit is None:
            break
        if first is None:
            first = dist
        count += 1
        origin = hit + direction * 1e-4
    return first, count % 2


def _check_surface_collider_penetration(points, spacing):
    """No surface vertex may sit deep inside a collider's mesh.

    The splat carves the collider's occupied voxels out of the field, so the
    extracted surface pools against a collider rather than bleeding into it.
    A vertex deeper inside than the carve's own discretization - the surface
    sample spacing and the collider voxel size - means the carve sampled the
    occupancy grid from the wrong origin (e.g. the surface grid's corner
    instead of the domain's), shifting the carve by the kernel margin.

    The test runs against the collider's actual mesh, not its world AABB: a
    tilted collider's AABB is full of space where fluid legitimately rests on
    its faces. A vertex is exempt while it is within one grid step of the
    surface, because marching cubes interpolates the iso-crossing between a
    zeroed and a live sample (which can land it up to a spacing inside the
    face) and the voxel carve is stair-stepped by a voxel.
    """
    mod = sys.modules[ADDON_MODULE]
    scene = bpy.context.scene
    shrink = max(spacing, mod.collision.get_solver_grid()[1])
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in scene.objects:
        if obj.type != "MESH" or not obj.flowx_collider.is_collider:
            continue
        bvh = _collider_bvh(obj, depsgraph)
        c_lo, c_hi = mod.domain.world_bounds(obj)
        for point in points:
            if not (
                c_lo.x < point.x < c_hi.x
                and c_lo.y < point.y < c_hi.y
                and c_lo.z < point.z < c_hi.z
            ):
                continue
            results = [_first_hit_parity(bvh, point, Vector(d)) for d in _AXIS_DIRECTIONS]
            if not any(parity for _first, parity in results):
                continue
            firsts = [first for first, _parity in results if first is not None]
            if firsts and min(firsts) >= shrink:
                raise RuntimeError(
                    f"surface vertex {tuple(point)} lies deeper than {shrink:.3f} m "
                    f"inside collider '{obj.name}' - the collider carve sampled the wrong origin"
                )


def _install_packaged_zip():
    """Install the packaged extension from FLOWX_ZIP, when set.

    Release CI sets this to the zip built from the tagged commit, together
    with a fresh BLENDER_USER_RESOURCES and no dev symlink - so the zip
    itself is what gets installed and smoke tested, exactly what a third
    party would do.
    """
    zip_path = os.environ.get("FLOWX_ZIP")
    if not zip_path:
        return
    if bpy.app.version >= (5, 0):
        # Blender 5.x renamed the op: the directory and the files to install
        # from it are separate properties (files is a collection of
        # {"name": ...} items). Dispatch on version, not hasattr - bpy.ops
        # hands out a proxy for any attribute and only fails at call time.
        result = bpy.ops.extensions.package_install_files(
            directory=os.path.dirname(zip_path),
            files=[{"name": os.path.basename(zip_path)}],
            repo="user_default",
            enable_on_install=True,
        )
    else:
        result = bpy.ops.extensions.repo_install_local(directory=zip_path)
    if "FINISHED" not in result:
        raise RuntimeError(f"installing {zip_path} returned {result}")
    print(f"[smoke_test] installed packaged extension from {zip_path}")


def _run_demo(demo):
    """Start the solver on one demo scene and apply the standard checks."""
    mod = sys.modules[ADDON_MODULE]
    scene = bpy.context.scene
    domain = mod.domain.find_domain(scene)
    if domain is None:
        raise RuntimeError(f"{demo.name}: no tagged fluid domain found")
    colliders = [obj.name for obj in scene.objects if obj.flowx_collider.is_collider]
    if not colliders:
        raise RuntimeError(f"{demo.name}: no tagged colliders found")
    print(f"[smoke_test] {demo.name}: domain='{domain.name}', colliders={colliders}")

    result = _get_operator("flowx.sph_toggle")()
    if "FINISHED" not in result:
        raise RuntimeError(f"{demo.name}: flowx.sph_toggle returned {result}")
    sph = mod.solver.sph
    if not sph.is_running():
        raise RuntimeError(f"{demo.name}: solver did not start (see warnings above)")

    for frame in range(scene.frame_start, scene.frame_start + 6):
        scene.frame_set(frame)
    stats = sph.stats()
    _check_particles(sph, stats)
    _check_surface(sph, stats)
    _check_collider_grids()


def _check_demo_scenes():
    """Phase 8: replay the shipped demo scenes the way a user would.

    The exit criterion is that a third party opens a demo, hits play, and
    gets the surface mesh - so open every file in demos/ and apply the same
    solver and surface checks the empty-scene pass uses.
    """
    if not _gpu_available:
        print("[smoke_test] no GPU context; skipping demo scene replay")
        return

    demos_dir = REPO_ROOT / "demos"
    demo_files = sorted(demos_dir.glob("*.blend"))
    if not demo_files:
        raise RuntimeError(f"no demo .blend files in {demos_dir}; run scripts/make_demo.py")

    for demo in demo_files:
        print(f"[smoke_test] replaying demo {demo.name}")
        # Stop the previous run (the empty-scene pass, or the earlier demo)
        # before its scene is replaced: a running toggle would stop rather
        # than start, and the old scene's domain dies with the file.
        sys.modules[ADDON_MODULE].solver.sph.stop()
        bpy.ops.wm.open_mainfile(filepath=str(demo))
        _run_demo(demo)


def main():
    _init_gpu()
    _install_packaged_zip()
    bpy.ops.extensions.repo_refresh_all()

    if ADDON_MODULE not in bpy.context.preferences.addons:
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
    _check_demo_scenes()

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
