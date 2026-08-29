"""Whitewater: secondary spray/foam/bubble particles, driven off the PBF state.

Once per frame, after the substeps have finished (mirroring Phase 6's
surface reconstruction, which this runs alongside):

    score every fluid particle (whitewater_potential.metal) -> sort descending
    by score, reusing the fluid's own bitonic-sort trick (whitewater_sort.metal,
    kept from when image atomics could not compile at all - see
    solver/engine/metal_engine.py's bitonic()) -> spawn a fixed K particles
    from the top of that order into a fixed-capacity ring buffer
    (whitewater_spawn.metal) -> advect every
    live slot (whitewater_advect.metal) -> read back and rebuild a point-cloud
    child object, '<Domain>.Whitewater', the same way solver/surface.py
    rebuilds '<Domain>.FluidSurface'.

The ring buffer is the whole story for a dynamic particle count without
atomics: the spawn budget K is a domain setting rather than something read
back from the GPU, so "birth the top K scorers" is a fixed-size dispatch with
no allocation counter needed. A particle's ring slot gets silently
overwritten once the cursor wraps back around to it, whether or not its
lifetime had actually expired - that overwrite *is* the garbage collection,
by design (see whitewater_spawn.metal's header).

'<Domain>.Whitewater' currently holds raw point positions plus 'life'/'kind'
generic attributes and nothing else - it renders as vertices, not sprites.
Turning that into an actual spray/foam/bubble look is a Geometry Nodes
modifier reading those attributes, deliberately left for a follow-up rather
than bundled in here.
"""

import bmesh
import bpy

from ..domain import is_alive

WHITEWATER_SUFFIX = ".Whitewater"

_state = {
    "running": False,
    # The engine this run is bound to, kept so extract() can read the pool back
    # without sph.py handing it over a second time.
    "engine": None,
    "config": None,
    "object": None,
    "cursor": 0,
    "live": 0,
    # The last installed (x, y, z, life, kind) points, kept so the disk cache
    # can store a frame's whitewater and the CPU render path can replay it.
    "last_points": None,
}


class WhitewaterConfig:
    """Resolved whitewater parameters for one run, mirroring sph.SolverConfig."""

    __slots__ = (
        "capacity",
        "spawn_rate",
        "trapped_air_weight",
        "wave_crest_weight",
        "kinetic_weight",
        "kinetic_reference_speed",
        "spray_speed_threshold",
        "bubble_trapped_threshold",
        "jitter_strength",
        "normal_offset",
        "spray_life",
        "foam_life",
        "bubble_life",
        "drag",
        "buoyancy",
    )


def resolve(domain):
    settings = domain.flowx_domain
    config = WhitewaterConfig()
    config.capacity = max(1, settings.whitewater_capacity)
    config.spawn_rate = settings.whitewater_spawn_rate
    config.trapped_air_weight = settings.whitewater_trapped_air_weight
    config.wave_crest_weight = settings.whitewater_wave_crest_weight
    config.kinetic_weight = settings.whitewater_kinetic_weight
    config.kinetic_reference_speed = settings.whitewater_kinetic_reference_speed
    config.spray_speed_threshold = settings.whitewater_spray_speed_threshold
    config.bubble_trapped_threshold = settings.whitewater_bubble_trapped_threshold
    config.jitter_strength = settings.whitewater_jitter_strength
    config.normal_offset = settings.whitewater_normal_offset
    config.spray_life = (settings.whitewater_spray_life_min, settings.whitewater_spray_life_max)
    config.foam_life = (settings.whitewater_foam_life_min, settings.whitewater_foam_life_max)
    config.bubble_life = (settings.whitewater_bubble_life_min, settings.whitewater_bubble_life_max)
    config.drag = settings.whitewater_drag
    config.buoyancy = settings.whitewater_buoyancy
    return config


def start(engine, domain, fluid_config):
    """Allocate the pool. Safe to call repeatedly.

    There is no per-pass compile step any more: the engine compiles one library
    holding every kernel, these four included, when the solver starts.
    """
    _state["running"] = True
    reseed(engine, domain, fluid_config)


def reseed(engine, domain, fluid_config):
    """Re-resolve the pool for a re-seeded solver.

    Cheap enough to run on every playback loop. The pool is zeroed rather than
    carried forward: a re-seed restarts the fluid from scratch, and stale
    whitewater particles hanging around from the previous run would read as a
    bug, not a feature.
    """
    if not _state["running"]:
        return
    config = resolve(domain)
    _state["config"] = config
    _state["engine"] = engine

    engine.alloc_whitewater(config.capacity, fluid_config.sorted_count)

    _state["cursor"] = 0
    _state["live"] = 0
    _state["object"] = _whitewater_object(domain)
    # The pool is zeroed, so the honest display is no points - clear whatever a
    # previous run left in the object and record an empty live set (which is
    # also what the seed frame's cache record stores).
    _set_points(_state["object"], [])


def stop():
    """Drop the device state. The point-cloud object is left as-is, like the surface."""
    _state.update({"running": False, "engine": None, "config": None, "object": None})


def is_running():
    return _state["running"]


def stats():
    config = _state["config"]
    if config is None:
        return None
    return {
        "capacity": config.capacity,
        "live": _state["live"],
        "spawn_rate": config.spawn_rate,
    }


def record(engine, fluid_config, frame_dt, frame):
    """Ask the engine to score, sort, spawn and advect the pool.

    The spawn budget is computed here rather than in the engine because it is
    the ring buffer's own bookkeeping: a fixed K per frame is what lets the
    spawn step be a fixed size with no allocation counter (see
    kernels/whitewater_spawn.metal).
    """
    config = _state["config"]
    if config is None:
        return

    n = fluid_config.sorted_count
    spawn_count = max(0, min(config.capacity, n, round(config.spawn_rate * max(frame_dt, 0.0))))
    cursor = _state["cursor"]

    engine.step_whitewater(config, cursor, spawn_count, frame, frame_dt)

    # Advanced here rather than after the flush: the cursor is CPU-side
    # bookkeeping that has to move exactly once per spawn, and the engine has
    # already captured this frame's value.
    _state["cursor"] = (cursor + spawn_count) % config.capacity


def extract():
    """Read the pool back and rebuild the point cloud.

    Must follow the engine's flush: it reads what the recorded passes wrote.
    """
    config = _state["config"]
    engine = _state["engine"]
    if config is None or engine is None:
        return
    positions, vel_kind = engine.read_whitewater(config.capacity)
    obj = _state["object"]
    if is_alive(obj):
        _rebuild_points(obj, positions, vel_kind)


def _whitewater_object(domain):
    """Find or create the domain's child whitewater point-cloud object."""
    name = domain.name + WHITEWATER_SUFFIX
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != "MESH":
        obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))
        obj.parent = domain

    # Same reasoning as surface._surface_object(): extraction/rebuild emits
    # world-space points, so the parent's transform must be cancelled out.
    obj.matrix_parent_inverse = domain.matrix_world.inverted()

    collection = domain.users_collection[0] if domain.users_collection else None
    if collection is None:
        collection = bpy.context.scene.collection
    if obj.name not in collection.objects:
        collection.objects.link(obj)
    return obj


def _set_points(obj, alive):
    """Replace the object's geometry from a list of (x, y, z, life, kind) points.

    Shared by the GPU read-back path (_rebuild_points) and the CPU render path
    (install_mesh): both end with a list of live points and need the same
    bmesh rebuild plus life/kind attributes.
    """
    mesh = obj.data
    _state["live"] = len(alive)
    _state["last_points"] = alive

    if not alive:
        mesh.clear_geometry()
        mesh.update()
        return

    bm = bmesh.new()
    for x, y, z, _life, _kind in alive:
        bm.verts.new((x, y, z))
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    for name in ("life", "kind"):
        existing = mesh.attributes.get(name)
        if existing is not None:
            mesh.attributes.remove(existing)
    life_attr = mesh.attributes.new(name="life", type="FLOAT", domain="POINT")
    kind_attr = mesh.attributes.new(name="kind", type="INT", domain="POINT")
    for i, (_x, _y, _z, life, kind) in enumerate(alive):
        life_attr.data[i].value = life
        kind_attr.data[i].value = int(kind)


def _rebuild_points(obj, positions, vel_kind):
    """Replace the object's geometry with the pool's currently-live points."""
    alive = [
        (p[0], p[1], p[2], p[3], vk[3])
        for p, vk in zip(positions, vel_kind, strict=True)
        if p[3] > 0.0
    ]
    _set_points(obj, alive)


def install_points(domain, points):
    """Rebuild the whitewater object from cached points, with no GPU.

    The CPU render path: `points` are the (x, y, z, life, kind) tuples replayed
    from the disk cache, and `domain` only locates the child object.
    """
    obj = _whitewater_object(domain)
    _set_points(obj, points)


def last_points():
    """The (x, y, z, life, kind) points last installed, or None if never built."""
    return _state["last_points"]
