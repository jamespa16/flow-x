"""Fluid-solver configuration, timeline, cache, and Blender operators.

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

The opt-in APIC method uses the same particles, spatial hash, colliders and
outputs, but transfers momentum to staggered MAC faces, projects the grid to a
constant-density divergence-free velocity, then transfers velocity and three
affine rows back to each particle. Its implementation lives beside the PBF
engines, in ``apic_cpu.py`` and ``apic_metal.py``.

The grid build is where this departs from the textbook GPU build. A counting
sort needs an atomic add, and under Blender's ``gpu`` module - which this
solver used to dispatch through - image atomics did not compile on the Metal
backend at all. So the (cell key, particle index) pairs go through a bitonic
sort instead: pure compare-exchange, no atomics, O(log^2 n) host-driven passes,
about 105 of them per substep. Flow-X owns its own Metal device now and atomics
work, so the counting sort is available and simply not written yet; the bitonic
sort is kept until something re-bakes the reference runs deliberately.

The device work itself lives in ``solver/engine``: this module resolves the
config, drives the timeline and owns the cache, and asks an engine to advance
the fluid. A whole frame is recorded into one command queue and submitted once,
which is why the read-backs below all follow a single ``flush()``.

Finalize also does Phase 5's collision response: it samples the collider
occupancy grid built in ``collision`` and pushes a particle out to the nearest
free voxel, with a velocity reflection/damping term along the push direction
(see ``kernels/sph_finalize.metal``), the same as the WCSPH integrator did.

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
import time
from array import array
from collections import deque

import bpy
from bpy.app.handlers import persistent
from bpy.types import Operator
from mathutils import Vector

from ..collision import (
    ensure_grids,
    get_solver_grid,
    get_solver_occupancy,
    rebuild_animated_grids,
)
from ..domain import find_domain, is_alive, is_degenerate, world_bounds
from . import cache, surface, viz, whitewater
from . import engine as engines

GRAVITY = -9.81

# Fraction of incoming normal velocity kept when a particle hits a domain wall.
BOUNDARY_DAMPING = 0.35

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

# The per-substep pass chain lives in solver/engine/metal_engine.py now,
# alongside the buffer table it dispatches against: it is device bookkeeping,
# not timeline logic, and this module is the timeline.
#
# PBF constraint-solve iterations per substep are a fixed count rather than a
# convergence check: the density constraint is a projection, not a stiff
# spring, so a handful of Jacobi passes gets close enough without needing to
# detect convergence on the GPU.


_state = {
    "running": False,
    "config": None,
    "domain": None,
    # The compute engine: device, compiled kernels, buffers and the frame's
    # command queue. None when the solver is not running, or when no device
    # could be brought up at all.
    "engine": None,
    "substeps": 0,
    # Playback bookkeeping (Phase 7). `last_frame` is the frame the GPU state
    # represents; `seed_frame` is the frame the run is seeded at and re-seeds
    # at. `warning` is user-facing text for the panel, set when the timeline
    # asks for something the solver cannot honestly deliver.
    "seed_frame": 0,
    "last_frame": 0,
    # The frame the GPU textures actually represent. Normally equal to
    # last_frame, but the CPU render path advances last_frame without touching
    # the GPU, so after a render the two can diverge and a forward step must
    # re-sync the GPU state before simulating from it.
    "gpu_frame": 0,
    "warning": None,
    # Reserved for a persistent engine-selection note. Methods are never
    # substituted; Auto may only fall back to the same method's CPU engine.
    "fallback": None,
    "timings": deque(maxlen=TIMING_WINDOW),
    # Bake Cache bookkeeping: `baking` gates the timer that steps the timeline
    # forward one frame per tick, and `bake_target` is the frame it stops at
    # (the scene's end frame).
    "baking": False,
    "bake_target": None,
    # True while a render is running, set by the render_pre/render_post
    # handlers. render_pre fires before the first frame's frame_change_pre, so
    # - unlike the compute context, which Blender only drops partway into that
    # first frame - every rendered frame sees this and takes the CPU path.
    "rendering": False,
}


class SolverConfig:
    """Resolved simulation parameters for one run, derived from the domain.

    The domain's properties are targets, not the last word: seeding coarsens
    the particle spacing (and the smoothing radius with it, so density stays
    consistent) if the requested resolution would blow the particle budget.
    """

    __slots__ = (
        "solver_method",
        "lo",
        "hi",
        "fill_fraction",
        "max_particles",
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
        "pressure_iterations",
        "vorticity_strength",
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
        length_scale = self.cell_size if self.solver_method == "apic" else self.smoothing_radius
        gravity_dt = CFL_FACTOR * math.sqrt(length_scale / abs(GRAVITY))
        if self.solver_method == "apic":
            steps = min(max(1, math.ceil(frame_dt / gravity_dt)), self.max_substeps)
            return steps, min(frame_dt / steps, gravity_dt)
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
    config.solver_method = settings.solver_method.lower()
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
    config.max_particles = settings.max_particles
    config.pressure_iterations = settings.apic_pressure_iterations
    config.vorticity_strength = settings.apic_vorticity_strength

    # One knob sizes the simulation: particles seed one per voxel of the
    # domain's `resolution` lattice, and the SPH kernel follows the spacing -
    # twice it, the classic radius/spacing ratio that puts a comfortable
    # number of neighbors inside the kernel's support without over-sampling.
    longest = max(size.x, size.y, size.z)
    config.spacing = longest / max(settings.resolution, 1)
    config.smoothing_radius = config.spacing * 2.0

    fill_height = size.z * config.fill_fraction
    for _ in range(8):
        nx, ny, nz = _seed_counts(size, fill_height, config.spacing)
        if nx * ny * nz <= config.max_particles:
            break
        # Scale spacing and smoothing radius together so particle mass, kernel
        # support and rest density stay mutually consistent after coarsening.
        config.spacing *= (nx * ny * nz / config.max_particles) ** (1 / 3)
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

    values = array("f")
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if len(values) // 4 >= config.max_particles:
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
    """Seed the particle lattice and size every device buffer for it.

    `sorted_count` is the particle count rounded up to a power of two, which is
    what the bitonic sort needs; the padding slots carry a sentinel key that
    sorts to the tail. The buffers are all flat arrays indexed by particle or
    by cell - the 2D texture wrap the old GLSL path needed (because Blender
    could not read a 1D texture back) has no successor.
    """
    positions = _seed_positions(config)
    config.particle_count = len(positions) // 4
    config.sorted_count = max(2, 1 << (max(1, config.particle_count) - 1).bit_length())
    _state["engine"].allocate(config, positions)


def _sync_params(config, dt):
    """Push the run's resolved settings into the engine's parameter block.

    Called once per frame rather than per dispatch: only `dt` and the bitonic
    step vary inside a frame, and the engine overrides those per record.

    Everything here used to be squeezed into a 128-byte push-constant block
    that was exactly full, which is why the s_corr strength and the surface
    tension coefficient arrived bit-packed into spare lanes of the sort slot.
    They are ordinary fields now.
    """
    engine = _state["engine"]
    engine.params.update(
        particle_count=config.particle_count,
        sorted_count=config.sorted_count,
        cell_count=config.cell_count,
        cells_x=config.cell_dims[0],
        cells_y=config.cell_dims[1],
        cells_z=config.cell_dims[2],
        lo_x=config.lo.x,
        lo_y=config.lo.y,
        lo_z=config.lo.z,
        cell_size=config.cell_size,
        hi_x=config.hi.x,
        hi_y=config.hi.y,
        hi_z=config.hi.z,
        particle_radius=config.particle_radius,
        smoothing_radius=config.smoothing_radius,
        mass=config.mass,
        rest_density=config.rest_density,
        relaxation=config.relaxation,
        viscosity=config.viscosity,
        dt=dt,
        gravity=GRAVITY,
        boundary_damping=BOUNDARY_DAMPING,
        scorr_k=config.scorr_strength,
        surface_tension=config.surface_tension,
        nodes_x=config.cell_dims[0] + 1,
        nodes_y=config.cell_dims[1] + 1,
        nodes_z=config.cell_dims[2] + 1,
        grid_spacing=config.cell_size,
        vorticity_epsilon=config.vorticity_strength,
        grid_max_speed=CFL_FACTOR * config.cell_size / max(dt, 1e-6),
        pressure_ping=0,
    )
    _sync_collider()


def _sync_collider():
    """Point the engine at the current collider grid.

    Fetched live rather than cached, so toggling a collider's tag mid-run takes
    effect on the very next substep rather than needing the solver restarted.
    """
    buffer, voxel_size, dims = get_solver_grid()
    if buffer is None and get_solver_occupancy() is None:
        # A voxel size of 0 tells the engine there is nothing to sample.
        _state["engine"].set_collider(None, (1, 1, 1), 0.0)
    else:
        # Both forms: a GPU engine binds the buffer, the CPU engine reads the
        # occupancy. Whichever it does not need, it ignores.
        _state["engine"].set_collider(buffer, dims, voxel_size, occupancy=get_solver_occupancy())


def _step(frame_dt, frame):
    """Advance one frame of simulated time and refresh the frame's output.

    The whole frame - every substep, then the surface splat, then whitewater -
    is recorded into one command queue and submitted once, at the flush below.
    Recording is cheap and does no GPU work, so timing anything finer than the
    whole step would just measure how fast Python can append to a list.
    """
    started = time.perf_counter()

    engine = _state["engine"]
    config = _state["config"]
    substeps, dt = config.substep_dt(frame_dt)
    _state["substeps"] = substeps

    _sync_params(config, dt)
    for _ in range(substeps):
        engine.substep(dt)

    # Recorded into the same submission as the substeps: both read the grid the
    # last substep built, and neither needs a read-back before the other runs.
    _record_surface(dt)
    _record_whitewater(frame_dt, frame)
    engine.flush()

    # The cache is the only other consumer of the particle state after a step,
    # and the debug overlay wants the same read-back - share it when both do.
    positions = None
    if cache.is_open() or _state["domain"].flowx_domain.show_particles:
        positions = engine.read_vec4("positions", config.particle_count)
    # Extraction is CPU work on what the flush produced, so it has to follow it.
    surface.extract()
    whitewater.extract()
    # The cache write comes after the surface and whitewater are extracted so
    # it stores this frame's own mesh, not the previous frame's - the render
    # path replays exactly what was written here.
    if cache.is_open():
        snapshot = engine.snapshot_state(include_whitewater=whitewater.is_running())
        if whitewater.is_running():
            snapshot["whitewater_cursor"] = whitewater.cursor()
        cache.write_frame(
            frame,
            snapshot,
            _state["domain"],
            surface.last_mesh(),
            whitewater.last_points(),
        )
    _update_viz(positions)

    _state["last_frame"] = frame
    _state["gpu_frame"] = frame
    _state["timings"].append((time.perf_counter() - started) * 1000.0)


def _record_whitewater(frame_dt, frame):
    """Record the whitewater passes, once per frame.

    Uses the full frame_dt rather than the substep dt the surface splat gets:
    spawn rate and advection are both real-time rates, and whitewater has no
    stake in the substep loop's internal stability limit the way the SPH
    passes do.
    """
    if not whitewater.is_running():
        return
    whitewater.record(_state["engine"], _state["config"], frame_dt, frame)


def _record_surface(dt):
    """Record the surface splat, once per frame rather than per substep.

    Its read-back and marching-cubes extraction are the only CPU round-trip in
    the pipeline, and nothing between substeps looks at the result.
    """
    if not surface.is_running():
        return
    surface.record(_state["engine"], _state["config"], dt)


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
        positions = _state["engine"].read_vec4("positions", config.particle_count)
    viz.set_points([p[:3] for p in positions])


def _seed(domain):
    """Re-resolve the run's parameters and refill particle state from scratch.

    Everything except the compiled kernels is rebuilt, so a re-seed picks up
    edits to fluid level, resolution and the solver parameters. The kernels
    depend only on the binding layout, never on the config, so they survive -
    which is what makes re-seeding cheap enough to do on every playback loop.
    """
    config = _resolve_config(domain)
    _state["config"] = config
    _allocate(config)
    _sync_params(config, 0.0)

    if domain.flowx_domain.show_surface:
        if surface.is_running():
            surface.reseed(_state["engine"], domain, config)
        else:
            surface.start(_state["engine"], domain, config)
        # Extract once up front so the seeded fluid is visible as a surface
        # straight away instead of as an empty object until playback starts.
        # The splat gathers through the spatial hash, so that has to exist -
        # building it here costs one dispatch chain and advances nothing.
        _state["engine"].build_grid()
        _record_surface(0.0)
        _state["engine"].flush()
        surface.extract()
    elif surface.is_running():
        surface.stop()

    if domain.flowx_domain.show_whitewater:
        if whitewater.is_running():
            whitewater.reseed(_state["engine"], domain, config)
        else:
            whitewater.start(_state["engine"], domain, config)
    elif whitewater.is_running():
        whitewater.stop()

    _update_viz()


def _reset_clock(scene):
    """Pin the run's timeline to `scene` and drop the previous run's stats."""
    _state["seed_frame"] = scene.frame_start
    _state["last_frame"] = scene.frame_current
    # Just re-seeded, so the GPU state is the seed at the current frame.
    _state["gpu_frame"] = scene.frame_current
    _state["warning"] = None
    _state["substeps"] = 0
    _state["timings"].clear()


def _start(domain):
    # Collider grids are in-memory only, so after a restart or an extension
    # reload a tagged collider may have its tag but no grid; build the missing
    # ones so the first substep collides correctly.
    ensure_grids(bpy.context.scene)
    preferred = domain.flowx_domain.engine
    method = domain.flowx_domain.solver_method.lower()
    engine = engines.create(None if preferred == "AUTO" else preferred.lower(), method)
    if engine is None:
        raise RuntimeError(engines.unavailable_reason())
    _state["engine"] = engine
    _state["fallback"] = engines.fallback_note()
    _state["domain"] = domain
    _state["running"] = True
    _seed(domain)
    _reset_clock(bpy.context.scene)
    ww_stats = whitewater.stats()
    cache.open(
        bpy.context.scene,
        domain,
        _state["config"].particle_count,
        method=engine.method,
        device=engine.name,
        whitewater_capacity=ww_stats["capacity"] if ww_stats else 0,
    )
    # The seed surface was extracted in _seed; store it in the mesh cache so a
    # render has the first frame's surface to replay (the particle file skips
    # the seed frame, but the mesh file keeps it).
    cache.write_seed_mesh(domain, bpy.context.scene, surface.last_mesh(), whitewater.last_points())

    viz.enable()
    if _on_frame_change not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(_on_frame_change)
    _register_render_handlers()
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
    # hash still matches, start fresh when it has moved. The re-extracted seed
    # surface is re-stored in the mesh cache so a render has the first frame.
    ww_stats = whitewater.stats()
    cache.open(
        scene,
        domain,
        _state["config"].particle_count,
        method=_state["engine"].method,
        device=_state["engine"].name,
        whitewater_capacity=ww_stats["capacity"] if ww_stats else 0,
    )
    cache.write_seed_mesh(domain, scene, surface.last_mesh(), whitewater.last_points())
    viz.tag_viewports_redraw()
    return True


def stop():
    cancel_bake()
    if _on_frame_change in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(_on_frame_change)
    _unregister_render_handlers()
    _state["rendering"] = False
    cache.close()
    surface.stop()
    whitewater.stop()
    engine = _state["engine"]
    if engine is not None:
        engine.release()
    _state.update(
        {"running": False, "config": None, "domain": None, "engine": None, "fallback": None}
    )
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


def start_bake(scene=None, domain=None):
    """Bake the disk cache from the scene's start frame to its end frame.

    Enables caching, starts the solver if it isn't running, re-seeds at the
    start frame, and then steps the timeline forward one frame per timer tick
    so every frame is simulated on the GPU and written to the cache -
    particles and the extracted surface. The surface is what a later render
    replays on the CPU. Returns True if the bake started.
    """
    scene = scene or bpy.context.scene
    domain = domain or find_domain(scene)
    if domain is None or is_degenerate(domain):
        return False
    # Caching must be on for the bake to accumulate a file; force it so the
    # user doesn't have to remember, and the render path has something to read.
    domain.flowx_domain.cache_enabled = True
    if not _state["running"]:
        _start(domain)
    # Force a clean re-seed at the start frame so the bake covers the whole
    # range from the seed, opens the cache, and records the seed frame's
    # surface - explicitly, rather than hoping a frame_set trips the re-seed
    # path (it short-circuits when the clock already sits on the start frame).
    scene.frame_set(scene.frame_start)
    reseed(scene)
    _state["baking"] = True
    _state["bake_target"] = scene.frame_end
    if not bpy.app.timers.is_registered(_bake_tick):
        bpy.app.timers.register(_bake_tick, first_interval=0.05)
    return True


def cancel_bake():
    """Stop a running bake. The simulation is left running, ready to render."""
    _state["baking"] = False
    if bpy.app.timers.is_registered(_bake_tick):
        bpy.app.timers.unregister(_bake_tick)


def is_baking():
    return _state["baking"]


def _bake_tick():
    """Step one frame of the bake per timer tick, until the scene's end frame."""
    if not _state["baking"] or not _state["running"]:
        _state["baking"] = False
        return None
    scene = bpy.context.scene
    frame = scene.frame_current
    if frame >= _state["bake_target"]:
        _state["baking"] = False
        return None
    # frame_set triggers _on_frame_change, which simulates this frame (the GPU
    # is available in the viewport) and writes it - particles and surface - to
    # the cache. Returning 0.0 reschedules the next tick as soon as possible;
    # the tick boundary is what keeps the UI responsive and the bake cancellable.
    scene.frame_set(frame + 1)
    return 0.0


def _frame_dt(scene):
    """Simulated seconds per frame, from the scene's frame rate."""
    fps = scene.render.fps / scene.render.fps_base if scene.render.fps_base else 24.0
    return 1.0 / fps if fps > 0 else 1.0 / 24.0


def _apply_cached(state, frame):
    """Install a cached frame's particle state and refresh its outputs.

    The complete method-specific state is stored; the spatial hash is rebuilt
    for the loaded particles (a zero-length step, exactly as a re-seed does) so
    the surface splat can gather, and the frame's surface is extracted from
    the scene's current collider grid - which the depsgraph path has already
    brought to this frame before the handler ran.
    """
    config = _state["config"]
    engine = _state["engine"]
    engine.restore_state(state)
    if "whitewater_cursor" in state:
        whitewater.restore_cursor(state["whitewater_cursor"])

    _state["last_frame"] = frame
    _state["gpu_frame"] = frame
    _state["warning"] = None

    _sync_params(config, 0.0)
    engine.build_grid()
    _record_surface(0.0)
    engine.flush()
    surface.extract()
    if "ww_positions" in state:
        whitewater.extract()
    _update_viz()
    viz.tag_viewports_redraw()


def _install_cached_mesh(frame, scene, domain):
    """CPU render path: install a frame's baked surface without the GPU.

    During a render the compute context is owned by the render, so the sim
    cannot step and the usual cache-load path (which re-derives the surface on
    the GPU) cannot run either. If the disk cache holds this frame's extracted
    surface (from a prior bake), rebuild the surface and whitewater child
    objects from it with plain bmesh and skip every GPU pass. A frame the bake
    did not cover warns rather than silently freezing the surface.
    """
    if not cache.is_open():
        _state["warning"] = (
            "Rendering needs a baked cache: the simulation cannot run on the GPU "
            "while a render owns it. Bake the cache first (Playback > Bake "
            "Cache), then render."
        )
        return
    mesh = cache.load_mesh(frame, scene, domain)
    if mesh is None:
        _state["warning"] = cache.warning()
        return
    vertices, triangles, ww_points = mesh
    surface.install_mesh(domain, vertices, triangles)
    whitewater.install_points(domain, ww_points)
    _state["last_frame"] = frame
    _state["warning"] = None
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
    _apply_cached(loaded, end)
    return True


def _in_render():
    """Whether a render is running and owns the GPU.

    A backup signal for the rendering flag: true for most of a render, though
    like the compute context it lags on the very first frame.
    """
    try:
        return bpy.app.is_job_running("render", "render")
    except Exception:
        return False


@persistent
def _on_render_pre(scene, _depsgraph):
    """Mark the run as rendering. Fires before the first frame's handler."""
    _state["rendering"] = True


@persistent
def _on_render_post(scene, _depsgraph):
    """Clear the rendering mark after a frame (or a still) has rendered."""
    _state["rendering"] = False


def _register_render_handlers():
    if _on_render_pre not in bpy.app.handlers.render_pre:
        bpy.app.handlers.render_pre.append(_on_render_pre)
    if _on_render_post not in bpy.app.handlers.render_post:
        bpy.app.handlers.render_post.append(_on_render_post)


def _unregister_render_handlers():
    if _on_render_pre in bpy.app.handlers.render_pre:
        bpy.app.handlers.render_pre.remove(_on_render_pre)
    if _on_render_post in bpy.app.handlers.render_post:
        bpy.app.handlers.render_post.remove(_on_render_post)


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
    # While a render owns the GPU the sim cannot step: the compute context is
    # dropped (and the cache-load path below is GPU-bound too), so replay the
    # frame's baked surface on the CPU instead - no GPU work at all. The
    # rendering flag (set in render_pre, which fires before this handler even
    # on the first frame) is the reliable signal; _in_render() and the compute
    # context probe are backups, since Blender only drops the context partway
    # into the first frame. Checked before the frame==last_frame short-circuit:
    # every rendered frame must install its cached surface even if the clock
    # already sits on it.
    if _state["rendering"] or _in_render():
        _install_cached_mesh(frame, scene, domain)
        return
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
        _apply_cached(loaded, frame)
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

    # If the GPU state is behind the logical clock - a CPU render advanced
    # last_frame without touching the GPU - re-sync it from the cache before
    # stepping, so we don't simulate forward from a stale frame.
    if _state["gpu_frame"] != _state["last_frame"]:
        reloaded = _load_cached(_state["last_frame"], scene, domain)
        if reloaded is not None:
            _apply_cached(reloaded, _state["last_frame"])
        elif not reseed(scene):
            stop_deferred()
            return
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


def read_state():
    """The run's current (positions, velocities) as flat lists of 4-tuples.

    A read-back on demand, for tools that need the particle state outside the
    normal step path - notably scripts/golden.py, which compares a run against
    a reference dump. _step() does its own read-back and shares it between the
    cache and the overlay; this is deliberately not wired into that, so asking
    for the state never changes what a frame costs.

    Kept as a public accessor rather than reaching into the engine's buffers
    directly, because it has to keep working across a backend change - which
    is exactly what it was written for.
    """
    config = _state["config"]
    engine = _state["engine"]
    if not _state["running"] or config is None or engine is None:
        return None
    return (
        engine.read_vec4("positions", config.particle_count),
        engine.read_vec4("velocities", config.particle_count),
    )


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
        "method": getattr(_state["engine"], "method", None),
        "device": getattr(_state["engine"], "name", None),
        "method_note": _state["fallback"],
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
        "engine": engines.describe(_state["engine"]),
    }


class FLOWX_OT_sph_toggle(Operator):
    """Seed the domain and run its selected fluid solver on playback"""

    bl_idname = "flowx.sph_toggle"
    bl_label = "Toggle Fluid Simulation"
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
            self.report({"INFO"}, "Flow-X fluid solver stopped")
            return {"FINISHED"}

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
            self.report({"WARNING"}, f"Could not start the fluid solver: {exc}")
            return {"FINISHED"}

        self.report(
            {"INFO"},
            f"Flow-X {_state['engine'].method.upper()} solver running "
            f"({_state['config'].particle_count} particles)",
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


class FLOWX_OT_sph_bake(Operator):
    """Bake the disk cache (particles and surface) from the start frame to the end"""

    bl_idname = "flowx.sph_bake"
    bl_label = "Bake Cache"
    # Not UNDO: the bake writes files outside Blender's undo stack and advances
    # the timeline over many ticks - there is no single undo to capture it.
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return find_domain(context.scene) is not None

    def execute(self, context):
        if is_baking():
            cancel_bake()
            self.report({"INFO"}, "Flow-X bake stopped - the cache is ready to render.")
            return {"FINISHED"}
        if not start_bake(context.scene, find_domain(context.scene)):
            self.report({"WARNING"}, "Could not start the bake - no valid domain.")
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Flow-X baking the cache to frame {context.scene.frame_end} " "(Stop Baking cancels).",
        )
        return {"FINISHED"}
