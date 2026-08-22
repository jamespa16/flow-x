"""Phase 4: the PBF solver core.

Each substep runs as a chain of compute dispatches over GPU-resident particle
state, with no CPU round-trip except the position read-back that feeds the
debug point-cloud viz:

    predict (gravity + surface tension) -> grid keys -> bitonic sort ->
    cell clear -> cell ranges -> [lambda -> delta -> apply delta] x iterations
    -> velocity -> XSPH -> finalize (collision + domain clamp)

This is Position Based Fluids (Macklin & Muller 2013), not WCSPH: there is no
pressure force and no equation-of-state stiffness. A particle is predicted
forward under gravity alone, then a fixed number of Jacobi constraint-solve
iterations project its predicted position so the local density matches rest
density, and its velocity is derived from how far that projection actually
moved it. The density constraint is unconditionally stable on its own, which
is what lets this run larger, fixed substeps than WCSPH's CFL-limited ones
ever could - the substep budget here is bounded only by how far gravity can
carry a particle before the neighbor grid it predicted into goes stale, and by
XSPH's own explicit diffusion limit (see ``SolverConfig.substep_dt``).

The grid build is where this departs from the textbook GPU build. A counting
sort needs ``imageAtomicAdd``, and image atomics do not compile on Blender's
Metal backend, so the (cell key, particle index) pairs go through a bitonic
sort instead - pure compare-exchange, no atomics, O(log^2 n) host-driven
passes. Dispatch overhead measured at ~13 us each, and with ~105
compare-exchange dispatches it is the dominant per-substep cost, which is why
it runs once per substep rather than once per constraint-solve iteration: the
neighbor list only drifts by one iteration's small position correction, not
enough to be worth rebuilding 3-4x over.

Finalize also does Phase 5's collision response: it samples the collider
occupancy grid built in ``collision`` and pushes a particle out to the nearest
free voxel, with a velocity reflection/damping term along the push direction
(see ``shaders/sph_finalize.glsl``), the same as the WCSPH integrator did.

Phase 6 hangs one more dispatch off the end of each *frame* (not each
substep): ``surface`` splats the particles onto a scalar grid and extracts a
mesh from it, which is the add-on's actual output. The point-cloud viz is now
a debug overlay behind a checkbox rather than the thing the user watches.

Phase 7 wires that to the timeline. ``_state["last_frame"]`` is the frame the
GPU state actually represents, which is not the same thing as the scene's
current frame: the handler re-seeds at or before the scene's start frame,
steps forward frame by frame from there, and on a backward scrub loads the
frame from the disk cache when one holds it - otherwise it *holds*, because
with no cache there is no honest way to go back, and it says so in the panel
rather than quietly showing a frame it never simulated. Seeding is from a
fixed RNG seed and the substep size comes only from the scene's frame rate, so
the same timeline replays identically every time; that determinism is what
makes the cache (``cache``) able to trust a stored frame: same settings and
same collider state hash to the same file, and a per-frame collider-matrix
fingerprint walk catches animation edits the config hash cannot see.
"""

import math
import random
import struct
import time
from collections import deque

import bpy
import gpu
from bpy.app.handlers import persistent
from bpy.types import Operator
from mathutils import Vector

from ..collision import ensure_grids, get_solver_grid, rebuild_animated_grids
from ..domain import find_domain, is_alive, is_degenerate, world_bounds
from . import cache, surface, viz
from .gpu_util import (
    TEXTURE_WIDTH,
    bind_image,
    bind_push_constants,
    build_compute_shader,
    dispatch_1d,
    make_texture,
    read_texture,
    shader_source,
)

LOCAL_GROUP_SIZE = 64
GRAVITY = -9.81

# Fraction of incoming normal velocity kept when a particle hits a domain wall.
BOUNDARY_DAMPING = 0.35

# Particle budget. Past this the seeder coarsens its spacing (and scales the
# smoothing radius to match) rather than allocating unbounded GPU state.
MAX_PARTICLES = 16384

# Neighbor cells are searched 3x3x3, so cells must be at least one smoothing
# radius across. These cap the other end: too fine a grid costs memory and
# cell-clear time for no benefit.
MAX_CELLS = 262144
MAX_CELLS_PER_AXIS = 128

# Frames to simulate in one go when the timeline jumps forward. Playback only
# ever asks for one, so this bounds how long a forward scrub can lock the UI up
# before the handler gives up and admits the frame is approximate.
MAX_CATCHUP_FRAMES = 30

# Frames of wall-clock timing averaged for the panel's ms/frame readout.
TIMING_WINDOW = 30

# Limit on the substep, in the same Courant-ish spirit WCSPH used: a particle
# must not free-fall or diffuse more than a meaningful fraction of a
# smoothing radius before the constraint loop or XSPH can respond. PBF's
# density constraint itself needs no such limit - it is a projection, not an
# explicit spring - so unlike WCSPH there is no speed-of-sound term here.
CFL_FACTOR = 0.25

# Every SPH pass declares this same push-constant block so shaders/sph_common.glsl
# can provide shared helpers. 128 bytes total, right at the budget - see that
# file before adding to it.
_PUSH_CONSTANTS = (
    ("IVEC4", "i_layout"),
    ("IVEC4", "i_grid"),
    ("IVEC4", "i_sort"),
    ("VEC4", "f_lo"),
    ("VEC4", "f_hi"),
    ("VEC4", "f_sph"),
    ("VEC4", "f_sim"),
    ("IVEC4", "i_collider"),
)

_IMAGES = (
    ("RGBA32F", "FLOAT_2D", "positions_img"),
    ("RGBA32F", "FLOAT_2D", "velocities_img"),
    # Holds (density, lambda) during the constraint loop - the same slot
    # WCSPH used for (density, pressure); renaming it would touch every pass
    # for no behavioral change, so only its packing's meaning has moved.
    ("RGBA32F", "FLOAT_2D", "lambda_img"),
    # This substep's predicted (pre-constraint, then constraint-corrected)
    # position. WCSPH used this texture for acceleration; PBF has no force
    # accumulation step, so it now holds position throughout the substep.
    ("RGBA32F", "FLOAT_2D", "predicted_img"),
    # Scratch accumulator for whichever pass currently needs a place to write
    # a per-particle correction without racing readers of predicted_img or
    # velocities_img: sph_delta's position correction, then later in the same
    # substep sph_xsph's velocity correction.
    ("RGBA32F", "FLOAT_2D", "delta_img"),
    # xyz = surface-tension color-field gradient (normal), w = curvature. Only
    # meaningful when surface_tension > 0, in which case sph_normal.glsl
    # writes it fresh every substep before sph_predict.glsl reads it; the
    # array is always allocated even when the pass is skipped.
    ("RGBA32F", "FLOAT_2D", "normal_img"),
    ("RGBA32F", "FLOAT_2D", "keys_img"),
    ("R32F", "FLOAT_2D", "cell_start_img"),
    ("R32F", "FLOAT_2D", "cell_end_img"),
    ("R32F", "FLOAT_3D", "collider_img"),
)

# PBF constraint-solve iterations per substep. A fixed count rather than a
# convergence check: the density constraint is a projection, not a stiff
# spring, so a handful of Jacobi passes gets close enough without needing to
# detect convergence on the GPU.
_PASSES = (
    "sph_normal",
    "sph_predict",
    "sph_grid_key",
    "sph_sort",
    "sph_cell_clear",
    "sph_cell_range",
    "sph_lambda",
    "sph_delta",
    "sph_apply_delta",
    "sph_velocity",
    "sph_xsph",
    "sph_finalize",
)

_state = {
    "running": False,
    "config": None,
    "domain": None,
    "shaders": {},
    # Per-pass sets of push-constant/image names this compiled shader has no
    # slot for (the OpenGL backend eliminates slots a pass never reads - see
    # gpu_util.bind_push_constants). Rebuilt alongside the shaders.
    "missing": {},
    "textures": {},
    "substeps": 0,
    # Playback bookkeeping (Phase 7). `last_frame` is the frame the GPU state
    # represents; `seed_frame` is the frame the run is seeded at and re-seeds
    # at. `warning` is user-facing text for the panel, set when the timeline
    # asks for something the solver cannot honestly deliver.
    "seed_frame": 0,
    "last_frame": 0,
    "warning": None,
    "timings": deque(maxlen=TIMING_WINDOW),
}


class SolverConfig:
    """Resolved simulation parameters for one run, derived from the domain.

    The domain's properties are targets, not the last word: seeding coarsens
    the particle spacing (and the smoothing radius with it, so density stays
    consistent) if the requested resolution would blow the particle budget.
    """

    __slots__ = (
        "lo",
        "hi",
        "fill_fraction",
        "particle_count",
        "sorted_count",
        "cell_dims",
        "cell_count",
        "cell_size",
        "smoothing_radius",
        "spacing",
        "mass",
        "rest_density",
        "relaxation",
        "scorr_strength",
        "surface_tension",
        "viscosity",
        "max_substeps",
        "iterations",
    )

    @property
    def particle_radius(self):
        return self.spacing * 0.5

    def substep_dt(self, frame_dt):
        """(substeps, dt) covering `frame_dt` without exceeding the diffusion/
        free-fall limit.

        PBF's density constraint is a projection, unconditionally stable on
        its own - unlike WCSPH there is no speed-of-sound/stiffness term
        bounding it. What remains is bounding how far a particle can predict
        forward under gravity alone before the constraint loop's neighbor
        grid (built from that prediction) goes stale, and XSPH's own explicit
        diffusion limit, both ported from WCSPH unchanged. When the budget
        needs more substeps than allowed, the solver advances less than a
        full frame of simulated time rather than going unstable - i.e. it
        degrades to slow motion, visibly.
        """
        gravity_dt = CFL_FACTOR * math.sqrt(self.smoothing_radius / abs(GRAVITY))
        # Diffusion limit: an explicit viscosity term goes unstable once it can
        # transport momentum more than a smoothing radius in one substep.
        kinematic = self.viscosity / max(self.rest_density, 1e-6)
        viscous_dt = 0.125 * self.smoothing_radius**2 / max(kinematic, 1e-12)
        limit = min(gravity_dt, viscous_dt)
        steps = min(max(1, math.ceil(frame_dt / limit)), self.max_substeps)
        return steps, min(frame_dt / steps, limit)


def _resolve_config(domain):
    settings = domain.flowx_domain
    lo, hi = world_bounds(domain)
    size = hi - lo

    config = SolverConfig()
    config.lo = lo
    config.hi = hi
    config.fill_fraction = settings.fluid_level / 100.0
    config.rest_density = settings.rest_density
    config.relaxation = settings.pbf_relaxation
    config.iterations = settings.pbf_iterations
    config.scorr_strength = settings.pbf_scorr_k
    config.surface_tension = settings.surface_tension
    config.viscosity = settings.viscosity
    config.max_substeps = settings.max_substeps

    # Particles sit half a smoothing radius apart, which puts a comfortable
    # number of neighbors inside the kernel's support without over-sampling.
    config.smoothing_radius = settings.smoothing_radius
    config.spacing = config.smoothing_radius * 0.5

    fill_height = size.z * config.fill_fraction
    for _ in range(8):
        nx, ny, nz = _seed_counts(size, fill_height, config.spacing)
        if nx * ny * nz <= MAX_PARTICLES:
            break
        # Scale spacing and smoothing radius together so particle mass, kernel
        # support and rest density stay mutually consistent after coarsening.
        config.spacing *= (nx * ny * nz / MAX_PARTICLES) ** (1 / 3)
        config.smoothing_radius = config.spacing * 2.0

    config.mass = config.rest_density * config.spacing**3

    # Cells must span at least one smoothing radius for the 3x3x3 neighbor
    # search to be exhaustive; beyond that, grow them to stay within budget.
    config.cell_size = max(
        config.smoothing_radius,
        max(size.x, size.y, size.z) / MAX_CELLS_PER_AXIS,
    )
    for _ in range(8):
        dims = tuple(
            min(MAX_CELLS_PER_AXIS, max(1, math.ceil(axis / config.cell_size)))
            for axis in (size.x, size.y, size.z)
        )
        if dims[0] * dims[1] * dims[2] <= MAX_CELLS:
            break
        config.cell_size *= 1.5
    config.cell_dims = dims
    config.cell_count = dims[0] * dims[1] * dims[2]

    return config


def _seed_counts(size, fill_height, spacing):
    """Particles per axis for a `spacing` lattice inset by half a spacing."""
    margin = spacing
    return (
        max(1, int((size.x - margin) / spacing)),
        max(1, int((size.y - margin) / spacing)),
        max(1, int(max(fill_height - margin, 0.0) / spacing)),
    )


def _seed_positions(config):
    """Jittered lattice filling the domain up to its fluid level.

    This is where the MVP's "set the fluid level to X% of domain height"
    requirement lands: the lattice stops at `fill_fraction` of the domain's
    height, so everything above it starts as empty air.

    The jitter breaks the lattice's symmetry - a perfectly regular seed makes
    SPH forces cancel in a way that reads as a solid block rather than a
    liquid for the first few frames.
    """
    size = config.hi - config.lo
    nx, ny, nz = _seed_counts(size, size.z * config.fill_fraction, config.spacing)
    rng = random.Random(0)
    jitter = config.spacing * 0.1
    base = config.lo + Vector((config.spacing, config.spacing, config.spacing)) * 0.5

    values = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if len(values) // 4 >= MAX_PARTICLES:
                    return values
                values.extend(
                    (
                        base.x + i * config.spacing + rng.uniform(-jitter, jitter),
                        base.y + j * config.spacing + rng.uniform(-jitter, jitter),
                        base.z + k * config.spacing + rng.uniform(-jitter, jitter),
                        1.0,
                    )
                )
    return values


def _allocate(config):
    positions = _seed_positions(config)
    config.particle_count = len(positions) // 4
    config.sorted_count = max(2, 1 << (max(1, config.particle_count) - 1).bit_length())

    zeros = [0.0] * (config.particle_count * 4)
    return {
        "positions_img": make_texture(config.particle_count, values=positions),
        "velocities_img": make_texture(config.particle_count, values=zeros),
        "lambda_img": make_texture(config.particle_count, values=zeros),
        # Seeded from the same initial positions as positions_img: sph_normal
        # (once surface tension lands) and the first substep's grid build both
        # read predicted_img before sph_predict has ever run.
        "predicted_img": make_texture(config.particle_count, values=positions),
        "delta_img": make_texture(config.particle_count, values=zeros),
        "normal_img": make_texture(config.particle_count, values=zeros),
        "keys_img": make_texture(config.sorted_count),
        "cell_start_img": make_texture(config.cell_count, channels=1, fmt="R32F"),
        "cell_end_img": make_texture(config.cell_count, channels=1, fmt="R32F"),
    }


def _images_for_pass(body):
    """The subset of _IMAGES a pass's own source actually reads or writes.

    OpenGL dead-code-eliminates image slots a compiled pass never touches,
    but Metal keeps every slot GPUShaderCreateInfo.image() declares - and
    caps read-write textures at 8 per shader. Declaring the full 10-image
    _IMAGES set (positions/velocities/lambda/predicted/delta/normal/keys/
    cell_start/cell_end/collider) on every pass compiles fine on OpenGL and
    fails on Metal.

    collider_img always stays in: sph_common.glsl's collider_occupied() /
    collider_coord() / collider_in_bounds() reference it by name, and that
    prelude is concatenated onto every pass whether or not the pass itself
    calls those functions - an image identifier a shader's source mentions
    at all must be declared, dead-code elimination happens after linking,
    not before. No pass reads more than 6 of the rest, so 6 + collider_img
    stays under the 8-texture cap everywhere.
    """
    used = {name for _fmt, _type, name in _IMAGES if name != "collider_img" and name in body}
    used.add("collider_img")
    return tuple(image for image in _IMAGES if image[2] in used)


def _compile_passes():
    prelude = shader_source("sph_common")
    shaders = {}
    for name in _PASSES:
        body = shader_source(name)
        shaders[name] = build_compute_shader(
            [prelude, body], _images_for_pass(body), _PUSH_CONSTANTS, LOCAL_GROUP_SIZE
        )
    # The absent-slot sets only describe this particular compile.
    _state["missing"] = {name: set() for name in _PASSES}
    return shaders


_EMPTY_COLLIDER_TEXTURE = None


def _empty_collider_texture():
    """1x1x1 "nothing occupied" texture, bound when no collider is tagged."""
    global _EMPTY_COLLIDER_TEXTURE
    if _EMPTY_COLLIDER_TEXTURE is None:
        buf = gpu.types.Buffer("FLOAT", [1], [0.0])
        _EMPTY_COLLIDER_TEXTURE = gpu.types.GPUTexture((1, 1, 1), format="R32F", data=buf)
    return _EMPTY_COLLIDER_TEXTURE


def _collider_binding():
    """(texture, i_collider) fetched live each bind, so toggling a collider's
    tag mid-run takes effect on the very next substep rather than needing the
    solver restarted.
    """
    texture, voxel_size, dims = get_solver_grid()
    if texture is None:
        # A voxel size of 0 tells the shader there's nothing to sample.
        return _empty_collider_texture(), (1, 1, 1, 0)
    voxel_size_bits = struct.unpack("<i", struct.pack("<f", voxel_size))[0]
    return texture, (*dims, voxel_size_bits)


def push_constant_values(config, dt, i_collider, sort_k=0, sort_j=0):
    """The shared push-constant block as {name: 4-tuple}.

    Phase 6's splat pass binds this same block against the same
    shaders/sph_common.glsl prelude, overriding only the slots it repurposes,
    so it is built here rather than inline in _bind().
    """
    scorr_k_bits = struct.unpack("<i", struct.pack("<f", config.scorr_strength))[0]
    tension_bits = struct.unpack("<i", struct.pack("<f", config.surface_tension))[0]
    return {
        "i_layout": (config.particle_count, TEXTURE_WIDTH, TEXTURE_WIDTH, config.sorted_count),
        "i_grid": (*config.cell_dims, config.cell_count),
        # z/w carry sph_delta.glsl's s_corr strength and sph_predict.glsl's
        # surface-tension coefficient (both bit-packed as floats), not sort
        # parameters - sph_sort is the only pass that reads x/y, and it never
        # reads z/w, so these ride for free in otherwise-unused lanes rather
        # than needing a second push-constant block.
        "i_sort": (sort_k, sort_j, scorr_k_bits, tension_bits),
        "f_lo": (*config.lo, config.cell_size),
        "f_hi": (*config.hi, config.particle_radius),
        "f_sph": (
            config.smoothing_radius,
            config.mass,
            config.rest_density,
            config.relaxation,
        ),
        "f_sim": (config.viscosity, dt, GRAVITY, BOUNDARY_DAMPING),
        "i_collider": i_collider,
    }


def _bind(name, dt, sort_k=0, sort_j=0):
    """Bind a pass's shader with the shared push-constant block and images."""
    config = _state["config"]
    shader = _state["shaders"][name]
    missing = _state["missing"][name]
    collider_texture, i_collider = _collider_binding()
    shader.bind()
    bind_push_constants(
        shader, push_constant_values(config, dt, i_collider, sort_k, sort_j), missing
    )
    for image_name in (image[2] for image in _IMAGES):
        if image_name == "collider_img":
            bind_image(shader, image_name, collider_texture, missing)
        else:
            bind_image(shader, image_name, _state["textures"][image_name], missing)
    return shader


def _build_grid(dt):
    config = _state["config"]
    n = config.sorted_count

    dispatch_1d(_bind("sph_grid_key", dt), n, LOCAL_GROUP_SIZE)

    # Bitonic sort: k is the current merge width, j the compare distance.
    k = 2
    while k <= n:
        j = k >> 1
        while j > 0:
            dispatch_1d(_bind("sph_sort", dt, k, j), n, LOCAL_GROUP_SIZE)
            j >>= 1
        k <<= 1

    dispatch_1d(_bind("sph_cell_clear", dt), config.cell_count, LOCAL_GROUP_SIZE)
    dispatch_1d(_bind("sph_cell_range", dt), n, LOCAL_GROUP_SIZE)


def _substep(dt):
    config = _state["config"]
    # sph_predict.glsl treats sigma <= 0 as a no-op, so skip the normal/
    # curvature pass's neighbor loop entirely when surface tension is off
    # (the shipped default) rather than paying for a dispatch nothing reads.
    if config.surface_tension > 0.0:
        dispatch_1d(_bind("sph_normal", dt), config.particle_count, LOCAL_GROUP_SIZE)
    dispatch_1d(_bind("sph_predict", dt), config.particle_count, LOCAL_GROUP_SIZE)
    _build_grid(dt)
    for _ in range(config.iterations):
        for name in ("sph_lambda", "sph_delta", "sph_apply_delta"):
            dispatch_1d(_bind(name, dt), config.particle_count, LOCAL_GROUP_SIZE)
    for name in ("sph_velocity", "sph_xsph", "sph_finalize"):
        dispatch_1d(_bind(name, dt), config.particle_count, LOCAL_GROUP_SIZE)


def _step(frame_dt, frame):
    """Advance one frame of simulated time and refresh the frame's output.

    Timed end to end rather than per stage: compute dispatches return before
    the GPU has run them, so it is the read-back at the end that the whole
    frame's GPU work actually lands in. Only the total means anything.
    """
    started = time.perf_counter()

    config = _state["config"]
    substeps, dt = config.substep_dt(frame_dt)
    _state["substeps"] = substeps
    for _ in range(substeps):
        _substep(dt)

    # The cache is the only other consumer of the particle state after a step,
    # and the debug overlay wants the same read-back - share it when both do.
    positions = None
    if cache.is_open() or _state["domain"].flowx_domain.show_particles:
        positions = read_texture(_state["textures"]["positions_img"], config.particle_count)
    if cache.is_open():
        velocities = read_texture(_state["textures"]["velocities_img"], config.particle_count)
        cache.write_frame(frame, positions, velocities, _state["domain"])

    _update_surface(dt)
    _update_viz(positions)

    _state["last_frame"] = frame
    _state["timings"].append((time.perf_counter() - started) * 1000.0)


def _update_surface(dt):
    """Re-extract the fluid surface, once per frame rather than per substep.

    It's the only CPU round-trip in the pipeline, and nothing between substeps
    looks at its output.
    """
    if not surface.is_running():
        return
    collider = _collider_binding()
    surface.update(
        _state["config"],
        _state["textures"],
        push_constant_values(_state["config"], dt, collider[1]),
        collider,
    )


def _update_viz(positions=None):
    """Refresh the debug point cloud, if the domain still asks for one.

    Phase 6's surface mesh is the real output now, so the particle read-back -
    the largest single transfer in a step - is skipped unless the user turns
    the overlay back on to check what the solver is doing underneath.
    A step that already read the positions for the cache passes them in.
    """
    config = _state["config"]
    domain = _state["domain"]
    if not is_alive(domain):
        return
    if not domain.flowx_domain.show_particles:
        viz.set_points([])
        return
    if positions is None:
        positions = read_texture(_state["textures"]["positions_img"], config.particle_count)
    viz.set_points([p[:3] for p in positions])


def _seed(domain):
    """Re-resolve the run's parameters and refill particle state from scratch.

    Everything except the compiled shaders is rebuilt, so a re-seed picks up
    edits to fluid level, resolution and the solver parameters. The shaders
    depend only on the binding layout, never on the config, so they survive -
    which is what makes re-seeding cheap enough to do on every playback loop.
    """
    config = _resolve_config(domain)
    _state["config"] = config
    _state["textures"] = _allocate(config)

    if domain.flowx_domain.show_surface:
        if surface.is_running():
            surface.reseed(domain, config)
        else:
            surface.start(domain, config)
        # Extract once up front so the seeded fluid is visible as a surface
        # straight away instead of as an empty object until playback starts.
        # The splat gathers through the spatial hash, so that has to exist -
        # a zero-length step builds it without advancing the simulation.
        _build_grid(0.0)
        _update_surface(0.0)
    elif surface.is_running():
        surface.stop()

    _update_viz()


def _reset_clock(scene):
    """Pin the run's timeline to `scene` and drop the previous run's stats."""
    _state["seed_frame"] = scene.frame_start
    _state["last_frame"] = scene.frame_current
    _state["warning"] = None
    _state["substeps"] = 0
    _state["timings"].clear()


def _start(domain):
    # Collider grids are in-memory only, so after a restart or an extension
    # reload a tagged collider may have its tag but no grid; build the missing
    # ones so the first substep collides correctly.
    ensure_grids(bpy.context.scene)
    _state["domain"] = domain
    _state["shaders"] = _compile_passes()
    _state["running"] = True
    _seed(domain)
    _reset_clock(bpy.context.scene)
    cache.open(bpy.context.scene, domain, _state["config"].particle_count)

    viz.enable()
    if _on_frame_change not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(_on_frame_change)
    viz.tag_viewports_redraw()


def reseed(scene=None):
    """Re-seed the running solver at the domain's current fluid level.

    Returns False if there is nothing to re-seed - the solver isn't running,
    its domain has been deleted out from under it, or the domain has been
    zeroed to no volume, in which case the caller should stop the run rather
    than keep simulating a stale or degenerate config.
    """
    domain = _state["domain"]
    if not _state["running"] or not is_alive(domain):
        return False
    if is_degenerate(domain):
        return False
    scene = scene or bpy.context.scene
    _seed(domain)
    _reset_clock(scene)
    # Re-open the cache so a mid-run edit to any hashed setting - or to the
    # toggle itself - decides the file's fate on this re-seed: reuse while the
    # hash still matches, start fresh when it has moved.
    cache.open(scene, domain, _state["config"].particle_count)
    viz.tag_viewports_redraw()
    return True


def stop():
    if _on_frame_change in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(_on_frame_change)
    cache.close()
    surface.stop()
    _state.update({"running": False, "config": None, "domain": None, "shaders": {}, "textures": {}})
    _state["timings"].clear()
    _state["warning"] = None
    viz.disable()


def _deferred_stop():
    stop()
    return None


def stop_deferred():
    """Stop the run from inside a frame handler.

    Removing the frame or draw handlers mid-dispatch skips whatever handlers
    follow this one for that frame, so only the flag flips now and the
    teardown - handler removal, GPU state, draw handler - lands a tick later.
    """
    _state["running"] = False
    viz.disable()
    if not bpy.app.timers.is_registered(_deferred_stop):
        bpy.app.timers.register(_deferred_stop, first_interval=0.1)


def _frame_dt(scene):
    """Simulated seconds per frame, from the scene's frame rate."""
    fps = scene.render.fps / scene.render.fps_base if scene.render.fps_base else 24.0
    return 1.0 / fps if fps > 0 else 1.0 / 24.0


def _apply_cached(positions, velocities, frame):
    """Install a cached frame's particle state and refresh its outputs.

    Only positions and velocities are stored; the spatial hash is rebuilt for
    the loaded particles (a zero-length step, exactly as a re-seed does) so
    the surface splat can gather, and the frame's surface is extracted from
    the scene's current collider grid - which the depsgraph path has already
    brought to this frame before the handler ran.
    """
    config = _state["config"]
    _state["textures"]["positions_img"] = make_texture(config.particle_count, values=positions)
    _state["textures"]["velocities_img"] = make_texture(config.particle_count, values=velocities)
    # The grid build below reads predicted_img, not positions_img (see
    # sph_grid_key.glsl) - a loaded frame has no "prediction" of its own, so
    # this just mirrors positions_img the way a fresh seed does in _allocate().
    _state["textures"]["predicted_img"] = make_texture(config.particle_count, values=positions)
    # Also reset lambda_img rather than leaving whatever density the solver
    # last computed for a *different* particle configuration: if surface
    # tension is on, sph_normal reads lambda_img's density channel as a
    # per-neighbor weight before sph_lambda ever runs against these loaded
    # positions (see _substep's pass order). Zeroing it here mirrors
    # _allocate()'s fresh seed, where every neighbor's density is the same
    # clamped constant - that uniform weight cancels out of both the normal's
    # direction and the curvature ratio, unlike a stale, non-uniform density
    # left over from wherever the timeline was before this jump.
    _state["textures"]["lambda_img"] = make_texture(
        config.particle_count, values=[0.0] * (config.particle_count * 4)
    )
    _state["last_frame"] = frame
    _state["warning"] = None
    _build_grid(0.0)
    _update_surface(0.0)
    _update_viz()
    viz.tag_viewports_redraw()


def _load_cached(frame, scene, domain):
    """Try to load a cached frame, closing the cache when it goes stale.

    A file whose config hash, collider set or fingerprint walk no longer
    matches the scene is stale for this run: it must stop serving loads and,
    just as important, stop growing - frames simulated under the new state
    must not be appended under the old header. The next Reset re-opens a
    file that matches.
    """
    loaded = cache.try_load(frame, scene, domain)
    if loaded is not None:
        return loaded
    if cache.warning() is not None:
        cache.close()
    return None


def _pick_up_cache_end(scene, domain, frame):
    """Load the cache's furthest frame when it lies between here and `frame`.

    A forward jump that outruns the catch-up budget can often start much
    closer to its target than the live state: the cache may already hold the
    ground in between, from this run or an earlier one with the same hash.
    """
    header = cache.header()
    if header is None:
        return False
    end = header["last_frame"]
    if not _state["last_frame"] < end < frame:
        return False
    loaded = _load_cached(end, scene, domain)
    if loaded is None:
        return False
    _apply_cached(*loaded, end)
    return True


@persistent
def _on_frame_change(scene, _depsgraph):
    """Step the solver to `scene.frame_current`, or say why it can't.

    In the order the timeline hits them during playback: the scene's start
    frame (or anything before it, including negative frames) is the run's
    origin and re-seeds; a frame the disk cache still holds - verified
    against the scene's current settings and collider state - loads with no
    simulation at all, which is the honest way back; a forward jump steps
    frame by frame, picking the cache's end up first when the jump outruns
    the catch-up budget; and a backward jump with no usable cache holds
    rather than showing a frame that was never simulated.
    """
    if not _state["running"]:
        return

    # The domain is the run's config source and the surface mesh's parent. If
    # it was deleted, or zeroed to no volume, out from under the run, stop
    # rather than keep simulating a stale or degenerate config with no panel
    # left to stop it from.
    domain = _state["domain"]
    if not is_alive(domain) or is_degenerate(domain):
        stop_deferred()
        return

    frame = scene.frame_current
    if frame == _state["last_frame"]:
        # Re-entering the frame the state already represents (a frame_set to
        # the current frame): there is nothing to advance and nothing wrong.
        return
    # An animated collider's transform changes with the frame, not with an
    # object edit, so its grid is not rebuilt by the depsgraph path; refresh
    # it now, at this frame's transform, before re-seeding, a cache load, or
    # the frame's physics reads it.
    rebuild_animated_grids(scene)
    if frame <= _state["seed_frame"]:
        if not reseed(scene):
            stop_deferred()
            return
        viz.tag_viewports_redraw()
        return

    loaded = _load_cached(frame, scene, domain)
    if loaded is not None:
        _apply_cached(*loaded, frame)
        return

    pending = frame - _state["last_frame"]
    if pending < 0:
        _state["warning"] = cache.warning() or (
            f"Frame {frame} is behind the simulation, which is at frame "
            f"{_state['last_frame']}. Flow-X has no cache to scrub back through - "
            "enable the cache and play forward, or return to the seed frame to re-run."
        )
        return

    if pending > MAX_CATCHUP_FRAMES:
        if _pick_up_cache_end(scene, domain, frame):
            pending = frame - _state["last_frame"]
        if pending > MAX_CATCHUP_FRAMES:
            _state["warning"] = (
                f"Jumped {frame - _state['last_frame']} frames; only "
                f"{MAX_CATCHUP_FRAMES} were simulated, so this frame is approximate. "
                f"Return to frame {_state['seed_frame']} and play forward for a correct "
                "result."
            )
            pending = MAX_CATCHUP_FRAMES
    else:
        # Per-frame playback warnings clear when a frame is simulated, but a
        # stale cache's explanation outlives the step until a Reset re-opens
        # a matching file - a closed cache keeps it, an open or disabled one
        # holds none.
        _state["warning"] = None if cache.is_open() else cache.warning()

    frame_dt = _frame_dt(scene)
    target = _state["last_frame"] + pending
    while _state["last_frame"] < target:
        _step(frame_dt, _state["last_frame"] + 1)
    # A clamped jump only simulated as far as `target`, but the run claims
    # the scene's frame anyway - the warning above says why it's approximate.
    if frame != target:
        _state["last_frame"] = frame
    viz.tag_viewports_redraw()


def is_running():
    return _state["running"]


def stats():
    """Resolved run parameters for the panel, or None when not running.

    The flag is checked, not just the config: a run stopped mid-frame (its
    domain deleted) drops the flag before the deferred teardown clears the
    config, and the panel would otherwise draw stats for a dead run.
    """
    config = _state["config"]
    if not _state["running"] or config is None:
        return None
    timings = _state["timings"]
    return {
        "particles": config.particle_count,
        "cells": config.cell_count,
        "cell_dims": config.cell_dims,
        "smoothing_radius": config.smoothing_radius,
        "spacing": config.spacing,
        "substeps": _state["substeps"],
        "surface": surface.stats(),
        "seed_frame": _state["seed_frame"],
        "frame": _state["last_frame"],
        "warning": _state["warning"],
        "step_ms": sum(timings) / len(timings) if timings else None,
        "cache": cache.info(),
    }


class FLOWX_OT_sph_toggle(Operator):
    """Seed the domain at its fluid level and run the GPU SPH solver on playback"""

    bl_idname = "flowx.sph_toggle"
    bl_label = "Toggle SPH Simulation"
    # Deliberately not UNDO: the run's state is GPU-side and not part of
    # Blender's undo stack, so undoing only the scene delta (the surface
    # object) would leave a simulation that is still running but no longer
    # visible. Stopping the run is its own undo.
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return find_domain(context.scene) is not None

    def execute(self, context):
        if _state["running"]:
            stop()
            self.report({"INFO"}, "Flow-X SPH solver stopped")
            return {"FINISHED"}

        from . import gpu_test

        gpu_test.stop()

        domain = find_domain(context.scene)
        if domain.flowx_domain.fluid_level <= 0.0:
            self.report({"WARNING"}, "Fluid level is 0% - nothing to seed.")
            return {"CANCELLED"}
        if is_degenerate(domain):
            self.report(
                {"WARNING"},
                "The domain has no volume (an axis is scaled to zero) - "
                "scale it back up before running the simulation.",
            )
            return {"CANCELLED"}

        try:
            _start(domain)
        except Exception as exc:
            # No GPU context (some --background runs) or a shader compile
            # failure shouldn't hard-crash the operator.
            stop()
            self.report({"WARNING"}, f"Could not start the SPH solver: {exc}")
            return {"FINISHED"}

        self.report(
            {"INFO"},
            f"Flow-X SPH solver running ({_state['config'].particle_count} particles)",
        )
        return {"FINISHED"}


class FLOWX_OT_sph_reset(Operator):
    """Re-seed the fluid at the domain's fluid level and restart the simulation clock"""

    bl_idname = "flowx.sph_reset"
    bl_label = "Reset Simulation"
    # Not UNDO either: a re-seed discards the previous GPU state, which no
    # undo of the scene-side mesh rebuild could bring back.
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return is_running()

    def execute(self, context):
        if not reseed(context.scene):
            domain = _state["domain"]
            if is_alive(domain) and is_degenerate(domain):
                self.report(
                    {"WARNING"},
                    "The domain has no volume (an axis is scaled to zero) - "
                    "scale it back up before resetting.",
                )
            else:
                self.report({"WARNING"}, "Nothing to reset - the solver is not running.")
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Flow-X simulation re-seeded ({_state['config'].particle_count} particles)",
        )
        return {"FINISHED"}
