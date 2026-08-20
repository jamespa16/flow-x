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
"""

import math
import random
import struct

import bpy
import gpu
from bpy.app.handlers import persistent
from bpy.types import Operator
from mathutils import Vector

from ..collision import get_solver_grid
from ..domain import find_domain, world_bounds
from . import viz
from .gpu_util import (
    TEXTURE_WIDTH,
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
    "shaders": {},
    "textures": {},
    "substeps": 0,
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


def _bind(name, dt, sort_k=0, sort_j=0):
    """Bind a pass's shader with the shared push-constant block and images."""
    config = _state["config"]
    shader = _state["shaders"][name]
    collider_texture, i_collider = _collider_binding()
    shader.bind()
    shader.uniform_int(
        "i_layout",
        (config.particle_count, TEXTURE_WIDTH, TEXTURE_WIDTH, config.sorted_count),
    )
    shader.uniform_int("i_grid", (*config.cell_dims, config.cell_count))
    shader.uniform_int("i_sort", (sort_k, sort_j, 0, 0))
    shader.uniform_float("f_lo", (*config.lo, config.cell_size))
    shader.uniform_float("f_hi", (*config.hi, config.particle_radius))
    shader.uniform_float(
        "f_sph",
        (config.smoothing_radius, config.mass, config.rest_density, config.stiffness),
    )
    shader.uniform_float("f_sim", (config.viscosity, dt, GRAVITY, BOUNDARY_DAMPING))
    shader.uniform_int("i_collider", i_collider)
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
    config = _state["config"]
    substeps, dt = config.substep_dt(frame_dt)
    _state["substeps"] = substeps
    for _ in range(substeps):
        _substep(dt)

    positions = read_texture(_state["textures"]["positions_img"], config.particle_count)
    viz.set_points([p[:3] for p in positions])


def _start(domain):
    config = _resolve_config(domain)
    _state["config"] = config
    _state["textures"] = _allocate(config)
    _state["shaders"] = _compile_passes()
    _state["running"] = True

    positions = read_texture(_state["textures"]["positions_img"], config.particle_count)
    viz.set_points([p[:3] for p in positions])
    viz.enable()
    if _on_frame_change not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(_on_frame_change)
    viz.tag_viewports_redraw()


def stop():
    if _on_frame_change in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(_on_frame_change)
    _state.update({"running": False, "config": None, "shaders": {}, "textures": {}})
    viz.disable()


@persistent
def _on_frame_change(scene, _depsgraph):
    if not _state["running"]:
        return
    fps = scene.render.fps / scene.render.fps_base if scene.render.fps_base else 24.0
    _step(1.0 / fps if fps > 0 else 1.0 / 24.0)
    viz.tag_viewports_redraw()


def is_running():
    return _state["running"]


def stats():
    """Resolved run parameters for the panel, or None when not running."""
    config = _state["config"]
    if config is None:
        return None
    return {
        "particles": config.particle_count,
        "cells": config.cell_count,
        "cell_dims": config.cell_dims,
        "smoothing_radius": config.smoothing_radius,
        "spacing": config.spacing,
        "substeps": _state["substeps"],
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
