"""Whitewater: secondary spray/foam/bubble particles, driven off the PBF state.

Once per frame, after the substeps have finished (mirroring Phase 6's
surface reconstruction, which this runs alongside):

    score every fluid particle (whitewater_potential.glsl) -> sort descending
    by score, reusing the fluid's own bitonic-sort trick (whitewater_sort.glsl,
    since image atomics still don't compile on Metal - see solver/sph.py's
    module docstring) -> spawn a fixed K particles from the top of that order
    into a fixed-capacity ring buffer (whitewater_spawn.glsl) -> advect every
    live slot (whitewater_advect.glsl) -> read back and rebuild a point-cloud
    child object, '<Domain>.Whitewater', the same way solver/surface.py
    rebuilds '<Domain>.FluidSurface'.

The ring buffer is the whole story for a dynamic particle count without
atomics: the spawn budget K is a domain setting rather than something read
back from the GPU, so "birth the top K scorers" is a fixed-size dispatch with
no allocation counter needed. A particle's ring slot gets silently
overwritten once the cursor wraps back around to it, whether or not its
lifetime had actually expired - that overwrite *is* the garbage collection,
by design (see whitewater_spawn.glsl's header).

'<Domain>.Whitewater' currently holds raw point positions plus 'life'/'kind'
generic attributes and nothing else - it renders as vertices, not sprites.
Turning that into an actual spray/foam/bubble look is a Geometry Nodes
modifier reading those attributes, deliberately left for a follow-up rather
than bundled in here.
"""

from array import array

import bmesh
import bpy

from ..domain import is_alive
from .gpu_util import (
    bind_image,
    bind_push_constants,
    build_compute_shader,
    dispatch_1d,
    make_texture,
    read_texture,
    shader_source,
    texture_width,
)

LOCAL_GROUP_SIZE = 64

# Kept in sync with solver/sph.py's GRAVITY by hand (not imported - sph.py
# imports this module, so importing back would be circular). Gravity is a
# fixed constant there too, not a domain setting, so this is safe.
GRAVITY = -9.81

WHITEWATER_SUFFIX = ".Whitewater"

_POTENTIAL_IMAGES = (
    ("RGBA32F", "FLOAT_2D", "positions_img"),
    ("RGBA32F", "FLOAT_2D", "velocities_img"),
    ("RGBA32F", "FLOAT_2D", "keys_img"),
    ("R32F", "FLOAT_2D", "cell_start_img"),
    ("R32F", "FLOAT_2D", "cell_end_img"),
    ("RGBA32F", "FLOAT_2D", "ww_keys_img"),
    ("R32F", "FLOAT_3D", "collider_img"),
)
_POTENTIAL_PUSH = (
    ("IVEC4", "i_layout"),
    ("IVEC4", "i_grid"),
    ("VEC4", "f_lo"),
    ("IVEC4", "i_collider"),
    ("VEC4", "f_sph"),
    ("VEC4", "f_potential"),
    ("VEC4", "f_seed"),
)

_SORT_IMAGES = (("RGBA32F", "FLOAT_2D", "ww_keys_img"),)
_SORT_PUSH = (("IVEC4", "i_sort"),)

_SPAWN_IMAGES = (
    ("RGBA32F", "FLOAT_2D", "positions_img"),
    ("RGBA32F", "FLOAT_2D", "velocities_img"),
    ("RGBA32F", "FLOAT_2D", "ww_keys_img"),
    ("RGBA32F", "FLOAT_2D", "ww_positions_img"),
    ("RGBA32F", "FLOAT_2D", "ww_velocity_kind_img"),
    ("R32F", "FLOAT_3D", "collider_img"),
)
_SPAWN_PUSH = (
    ("IVEC4", "i_layout"),
    # i_grid: not read by this pass's own body, but required anyway - see
    # whitewater_spawn.glsl's header.
    ("IVEC4", "i_grid"),
    ("IVEC4", "i_ww"),
    ("VEC4", "f_lo"),
    ("IVEC4", "i_collider"),
    ("VEC4", "f_kind"),
    ("VEC4", "f_life"),
    ("VEC4", "f_life2"),
)

_ADVECT_IMAGES = (
    ("RGBA32F", "FLOAT_2D", "ww_positions_img"),
    ("RGBA32F", "FLOAT_2D", "ww_velocity_kind_img"),
    ("R32F", "FLOAT_3D", "collider_img"),
)
_ADVECT_PUSH = (
    ("IVEC4", "i_ww"),
    # i_layout/i_grid: not read by this pass's own body, but required anyway
    # - see whitewater_advect.glsl's header.
    ("IVEC4", "i_layout"),
    ("IVEC4", "i_grid"),
    ("VEC4", "f_lo"),
    ("VEC4", "f_hi"),
    ("IVEC4", "i_collider"),
    ("VEC4", "f_sim"),
)

_state = {
    "shaders": None,
    "missing": {},
    "textures": {},
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
        "tex_width",
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
    config.tex_width = texture_width(config.capacity)
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


def _compile():
    prelude = shader_source("sph_common")
    return {
        "potential": build_compute_shader(
            [prelude, shader_source("whitewater_potential")],
            _POTENTIAL_IMAGES,
            _POTENTIAL_PUSH,
            LOCAL_GROUP_SIZE,
        ),
        # No prelude: this pass is a standalone bitonic sort with no need for
        # the fluid's kernel/collider helpers (see whitewater_sort.glsl).
        "sort": build_compute_shader(
            [shader_source("whitewater_sort")], _SORT_IMAGES, _SORT_PUSH, LOCAL_GROUP_SIZE
        ),
        "spawn": build_compute_shader(
            [prelude, shader_source("whitewater_spawn")],
            _SPAWN_IMAGES,
            _SPAWN_PUSH,
            LOCAL_GROUP_SIZE,
        ),
        "advect": build_compute_shader(
            [prelude, shader_source("whitewater_advect")],
            _ADVECT_IMAGES,
            _ADVECT_PUSH,
            LOCAL_GROUP_SIZE,
        ),
    }


def start(domain, fluid_config):
    """Compile the four passes and allocate the pool. Safe to call repeatedly."""
    _state["shaders"] = _compile()
    _state["missing"] = {name: set() for name in _state["shaders"]}
    reseed(domain, fluid_config)


def reseed(domain, fluid_config):
    """Re-resolve the pool for a re-seeded solver, keeping the compiled shaders.

    Like surface.reseed(), this only rebuilds sized state, not the shaders
    (which depend only on the binding layout) - cheap enough to run on every
    playback loop. The pool is zeroed rather than carried forward: a re-seed
    restarts the fluid from scratch, and stale whitewater particles hanging
    around from the previous run would read as a bug, not a feature.
    """
    if _state["shaders"] is None:
        return
    config = resolve(domain)
    _state["config"] = config
    zeros = array("f", [0.0]) * (config.capacity * 4)
    _state["textures"] = {
        "ww_positions_img": make_texture(config.capacity, values=zeros, width=config.tex_width),
        "ww_velocity_kind_img": make_texture(config.capacity, values=zeros, width=config.tex_width),
        # Sized identically to the fluid's own keys_img so whitewater_potential.glsl
        # can address it with the fluid's particle_texel()/i_layout unchanged.
        "ww_keys_img": make_texture(fluid_config.sorted_count, width=fluid_config.tex_width),
    }
    _state["cursor"] = 0
    _state["live"] = 0
    _state["object"] = _whitewater_object(domain)
    # The pool is zeroed, so the honest display is no points - clear whatever
    # a previous run left in the object and record an empty live set (which is
    # also what the seed frame's cache record stores).
    _set_points(_state["object"], [])


def stop():
    """Drop the GPU state. The point-cloud object is left as-is, like FluidSurface."""
    _state.update({"shaders": None, "missing": {}, "textures": {}, "config": None, "object": None})


def is_running():
    return _state["shaders"] is not None


def stats():
    config = _state["config"]
    if config is None:
        return None
    return {
        "capacity": config.capacity,
        "live": _state["live"],
        "spawn_rate": config.spawn_rate,
    }


def update(fluid_config, fluid_textures, fluid_constants, collider, frame_dt, frame):
    """Score, sort, spawn and advect the pool, then rebuild the point cloud.

    `fluid_constants` is the SPH push-constant block sph.py already built for
    this frame (see sph.push_constant_values) - only the shared fluid-layout
    slots (i_layout/i_grid/f_lo/f_hi/f_sph) are reused, the same way
    surface.update() borrows it.
    """
    config = _state["config"]
    shaders = _state["shaders"]
    if config is None or shaders is None:
        return

    collider_texture, i_collider = collider
    n = fluid_config.sorted_count
    spawn_count = max(0, min(config.capacity, n, round(config.spawn_rate * max(frame_dt, 0.0))))

    _dispatch_potential(fluid_textures, fluid_constants, i_collider, collider_texture, frame, n)
    _dispatch_sort(n, fluid_config.tex_width)
    if spawn_count > 0:
        _dispatch_spawn(
            fluid_textures, fluid_constants, i_collider, collider_texture, frame, spawn_count
        )
    _dispatch_advect(fluid_textures, fluid_constants, i_collider, collider_texture, frame_dt)

    positions = read_texture(_state["textures"]["ww_positions_img"], config.capacity)
    vel_kind = read_texture(_state["textures"]["ww_velocity_kind_img"], config.capacity)
    obj = _state["object"]
    if is_alive(obj):
        _rebuild_points(obj, positions, vel_kind)


def _dispatch_potential(fluid_textures, fluid_constants, i_collider, collider_texture, frame, n):
    config = _state["config"]
    shader = _state["shaders"]["potential"]
    missing = _state["missing"]["potential"]
    values = {
        "i_layout": fluid_constants["i_layout"],
        "i_grid": fluid_constants["i_grid"],
        "f_lo": fluid_constants["f_lo"],
        "i_collider": i_collider,
        "f_sph": fluid_constants["f_sph"],
        "f_potential": (
            config.trapped_air_weight,
            config.wave_crest_weight,
            config.kinetic_weight,
            config.kinetic_reference_speed,
        ),
        "f_seed": (float(frame), 0.0, 0.0, 0.0),
    }
    shader.bind()
    bind_push_constants(shader, values, missing)
    for name in ("positions_img", "velocities_img", "keys_img", "cell_start_img", "cell_end_img"):
        bind_image(shader, name, fluid_textures[name], missing)
    bind_image(shader, "ww_keys_img", _state["textures"]["ww_keys_img"], missing)
    bind_image(shader, "collider_img", collider_texture, missing)
    dispatch_1d(shader, n, LOCAL_GROUP_SIZE)


def _dispatch_sort(n, width):
    shader = _state["shaders"]["sort"]
    missing = _state["missing"]["sort"]
    k = 2
    while k <= n:
        j = k >> 1
        while j > 0:
            shader.bind()
            bind_push_constants(shader, {"i_sort": (k, j, n, width)}, missing)
            bind_image(shader, "ww_keys_img", _state["textures"]["ww_keys_img"], missing)
            dispatch_1d(shader, n, LOCAL_GROUP_SIZE)
            j >>= 1
        k <<= 1


def _dispatch_spawn(
    fluid_textures, fluid_constants, i_collider, collider_texture, frame, spawn_count
):
    config = _state["config"]
    shader = _state["shaders"]["spawn"]
    missing = _state["missing"]["spawn"]
    cursor = _state["cursor"]
    values = {
        "i_layout": fluid_constants["i_layout"],
        "i_grid": fluid_constants["i_grid"],
        "i_ww": (config.tex_width, config.capacity, cursor, spawn_count),
        "f_lo": fluid_constants["f_lo"],
        "i_collider": i_collider,
        "f_kind": (
            config.spray_speed_threshold,
            config.bubble_trapped_threshold,
            config.jitter_strength,
            config.normal_offset,
        ),
        "f_life": (*config.spray_life, *config.foam_life),
        "f_life2": (*config.bubble_life, 0.0, float(frame)),
    }
    shader.bind()
    bind_push_constants(shader, values, missing)
    bind_image(shader, "positions_img", fluid_textures["positions_img"], missing)
    bind_image(shader, "velocities_img", fluid_textures["velocities_img"], missing)
    bind_image(shader, "ww_keys_img", _state["textures"]["ww_keys_img"], missing)
    bind_image(shader, "ww_positions_img", _state["textures"]["ww_positions_img"], missing)
    bind_image(shader, "ww_velocity_kind_img", _state["textures"]["ww_velocity_kind_img"], missing)
    bind_image(shader, "collider_img", collider_texture, missing)
    dispatch_1d(shader, spawn_count, LOCAL_GROUP_SIZE)
    _state["cursor"] = (cursor + spawn_count) % config.capacity


def _dispatch_advect(fluid_textures, fluid_constants, i_collider, collider_texture, frame_dt):
    config = _state["config"]
    shader = _state["shaders"]["advect"]
    missing = _state["missing"]["advect"]
    values = {
        "i_ww": (config.tex_width, config.capacity, 0, 0),
        "i_layout": fluid_constants["i_layout"],
        "i_grid": fluid_constants["i_grid"],
        "f_lo": fluid_constants["f_lo"],
        "f_hi": fluid_constants["f_hi"],
        "i_collider": i_collider,
        "f_sim": (GRAVITY, frame_dt, config.drag, config.buoyancy),
    }
    shader.bind()
    bind_push_constants(shader, values, missing)
    bind_image(shader, "ww_positions_img", _state["textures"]["ww_positions_img"], missing)
    bind_image(shader, "ww_velocity_kind_img", _state["textures"]["ww_velocity_kind_img"], missing)
    bind_image(shader, "collider_img", collider_texture, missing)
    dispatch_1d(shader, config.capacity, LOCAL_GROUP_SIZE)


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
