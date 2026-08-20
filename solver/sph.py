"""Phase 4: the WCSPH solver core.

Each substep runs as a chain of compute dispatches over GPU-resident particle
state, with no CPU round-trip except the position read-back that feeds the
debug point-cloud viz:

    grid keys -> bitonic sort -> cell clear -> cell ranges
              -> density/pressure -> forces -> integrate

The sort is where this departs from the textbook GPU build. A counting sort
needs ``imageAtomicAdd``, and image atomics do not compile on Blender's Metal
backend, so the (cell key, particle index) pairs go through a bitonic sort
instead - pure compare-exchange, no atomics, O(log^2 n) host-driven passes.
Dispatch overhead measured at ~13 us, so the extra passes cost single-digit
milliseconds per frame rather than anything structural.

The integrate pass also does Phase 5's collision response: it samples the
collider occupancy grid built in ``collision`` and pushes a particle out to
the nearest free voxel, with a velocity reflection/damping term along the
push direction (see ``shaders/sph_integrate.glsl``).

Phase 6 hangs one more dispatch off the end of each *frame* (not each
substep): ``surface`` splats the particles onto a scalar grid and extracts a
mesh from it, which is the add-on's actual output. The point-cloud viz is now
a debug overlay behind a checkbox rather than the thing the user watches.

Phase 7 wires that to the timeline. ``_state["last_frame"]`` is the frame the
GPU state actually represents, which is not the same thing as the scene's
current frame: the handler re-seeds at or before the scene's start frame,
steps forward frame by frame from there, and *holds* on a backward scrub -
without a cache there is no honest way to go back, so it says so in the panel
rather than quietly showing a frame it never simulated. Seeding is from a
fixed RNG seed and the substep size comes only from the scene's frame rate, so
the same timeline replays identically every time.
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

from ..collision import ensure_grids, get_solver_grid
from ..domain import find_domain, is_alive, is_degenerate, world_bounds
from . import surface, viz
from .gpu_util import (
    TEXTURE_WIDTH,
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

# Courant-style limit on the substep: a particle must not cross a meaningful
# fraction of a smoothing radius before the pressure field can respond.
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
    ("RGBA32F", "FLOAT_2D", "density_img"),
    ("RGBA32F", "FLOAT_2D", "forces_img"),
    ("RGBA32F", "FLOAT_2D", "keys_img"),
    ("R32F", "FLOAT_2D", "cell_start_img"),
    ("R32F", "FLOAT_2D", "cell_end_img"),
    ("R32F", "FLOAT_3D", "collider_img"),
)

_PASSES = (
    "sph_grid_key",
    "sph_sort",
    "sph_cell_clear",
    "sph_cell_range",
    "sph_density",
    "sph_force",
    "sph_integrate",
)

_state = {
    "running": False,
    "config": None,
    "domain": None,
    "shaders": {},
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
        "stiffness",
        "viscosity",
        "max_substeps",
    )

    @property
    def particle_radius(self):
        return self.spacing * 0.5

    @property
    def sound_speed(self):
        """Tait EOS speed of sound: c^2 = dp/drho at rest density, gamma = 7."""
        return math.sqrt(7.0 * self.stiffness / max(self.rest_density, 1e-6))

    def substep_dt(self, frame_dt):
        """(substeps, dt) covering `frame_dt` without exceeding the CFL limit.

        When the CFL limit needs more substeps than the budget allows, the
        solver advances less than a full frame of simulated time rather than
        going unstable - i.e. it degrades to slow motion, visibly.
        """
        cfl_dt = CFL_FACTOR * self.smoothing_radius / max(self.sound_speed, 1e-6)
        gravity_dt = CFL_FACTOR * math.sqrt(self.smoothing_radius / abs(GRAVITY))
        # Diffusion limit: an explicit viscosity term goes unstable once it can
        # transport momentum more than a smoothing radius in one substep.
        kinematic = self.viscosity / max(self.rest_density, 1e-6)
        viscous_dt = 0.125 * self.smoothing_radius**2 / max(kinematic, 1e-12)
        limit = min(cfl_dt, gravity_dt, viscous_dt)
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
    config.stiffness = settings.stiffness
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
        "density_img": make_texture(config.particle_count, values=zeros),
        "forces_img": make_texture(config.particle_count, values=zeros),
        "keys_img": make_texture(config.sorted_count),
        "cell_start_img": make_texture(config.cell_count, channels=1, fmt="R32F"),
        "cell_end_img": make_texture(config.cell_count, channels=1, fmt="R32F"),
    }


def _compile_passes():
    prelude = shader_source("sph_common")
    return {
        name: build_compute_shader(
            [prelude, shader_source(name)], _IMAGES, _PUSH_CONSTANTS, LOCAL_GROUP_SIZE
        )
        for name in _PASSES
    }


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
    return {
        "i_layout": (config.particle_count, TEXTURE_WIDTH, TEXTURE_WIDTH, config.sorted_count),
        "i_grid": (*config.cell_dims, config.cell_count),
        "i_sort": (sort_k, sort_j, 0, 0),
        "f_lo": (*config.lo, config.cell_size),
        "f_hi": (*config.hi, config.particle_radius),
        "f_sph": (
            config.smoothing_radius,
            config.mass,
            config.rest_density,
            config.stiffness,
        ),
        "f_sim": (config.viscosity, dt, GRAVITY, BOUNDARY_DAMPING),
        "i_collider": i_collider,
    }


def _bind(name, dt, sort_k=0, sort_j=0):
    """Bind a pass's shader with the shared push-constant block and images."""
    config = _state["config"]
    shader = _state["shaders"][name]
    collider_texture, i_collider = _collider_binding()
    shader.bind()
    bind_push_constants(shader, push_constant_values(config, dt, i_collider, sort_k, sort_j))
    for image_name in (image[2] for image in _IMAGES):
        if image_name == "collider_img":
            shader.image(image_name, collider_texture)
        else:
            shader.image(image_name, _state["textures"][image_name])
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
    _build_grid(dt)
    for name in ("sph_density", "sph_force", "sph_integrate"):
        dispatch_1d(_bind(name, dt), config.particle_count, LOCAL_GROUP_SIZE)


def _step(frame_dt):
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

    _update_surface(dt)
    _update_viz()

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


def _update_viz():
    """Refresh the debug point cloud, if the domain still asks for one.

    Phase 6's surface mesh is the real output now, so the particle read-back -
    the largest single transfer in a step - is skipped unless the user turns
    the overlay back on to check what the solver is doing underneath.
    """
    config = _state["config"]
    domain = _state["domain"]
    if not is_alive(domain):
        return
    if not domain.flowx_domain.show_particles:
        viz.set_points([])
        return
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
    _seed(domain)
    _reset_clock(scene or bpy.context.scene)
    viz.tag_viewports_redraw()
    return True


def stop():
    if _on_frame_change in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(_on_frame_change)
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


@persistent
def _on_frame_change(scene, _depsgraph):
    """Step the solver to `scene.frame_current`, or say why it can't.

    Three cases, in the order the timeline hits them during playback: the
    scene's start frame (or anything before it, including negative frames) is
    the run's origin and re-seeds; a forward jump steps that many frames; a
    backward one holds, because replaying it would mean re-simulating from the
    seed frame and there is no cache to do it from.
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
    if frame <= _state["seed_frame"]:
        if not reseed(scene):
            stop_deferred()
            return
        viz.tag_viewports_redraw()
        return

    pending = frame - _state["last_frame"]
    if pending == 0:
        # Re-entering the frame the state already represents (a frame_set to
        # the current frame): there is nothing to advance and nothing wrong.
        return
    if pending < 0:
        _state["warning"] = (
            f"Frame {frame} is behind the simulation, which is at frame "
            f"{_state['last_frame']}. Flow-X has no cache to scrub back through - "
            f"return to frame {_state['seed_frame']} to re-seed and re-run."
        )
        return

    if pending > MAX_CATCHUP_FRAMES:
        _state["warning"] = (
            f"Jumped {pending} frames; only {MAX_CATCHUP_FRAMES} were simulated, so "
            f"this frame is approximate. Return to frame {_state['seed_frame']} and "
            "play forward for a correct result."
        )
        pending = MAX_CATCHUP_FRAMES
    else:
        _state["warning"] = None

    frame_dt = _frame_dt(scene)
    for _ in range(pending):
        _step(frame_dt)
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
    }


class FLOWX_OT_sph_toggle(Operator):
    """Seed the domain at its fluid level and run the GPU SPH solver on playback"""

    bl_idname = "flowx.sph_toggle"
    bl_label = "Toggle SPH Simulation"
    bl_options = {"REGISTER", "UNDO"}

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
    bl_options = {"REGISTER", "UNDO"}

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
