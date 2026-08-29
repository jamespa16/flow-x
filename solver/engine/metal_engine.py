"""The GPU engine: device state and dispatch for every solver pass.

This is what replaced solver/gpu_util.py. The solver used to reach into
Blender's `gpu` module for textures and shader binding; it now owns a device
through solver/backend/ and dispatches into buffers.

What the modules above keep and what moved:

* solver/sph.py keeps the timeline, the cache, the operators and the resolved
  SolverConfig. Its texture allocation, shader compilation, per-pass binding
  and substep chain moved here.
* solver/surface.py and solver/whitewater.py keep their own sizing, read-back
  and mesh building, and borrow this engine's queue, parameter block and
  particle buffers for the passes they own.

Three things the old plumbing needed are simply gone:

* Per-pass image lists. Metal caps read-write *images* at 8 per shader and
  keeps every declared slot, so each pass had to be scanned to work out which
  images to declare. Buffers have no such cap; the binding table in
  kernels/flowx_prelude.h is shared by every pass.
* The `missing` sets. OpenGL eliminated push-constant and image slots a pass
  never read, so binding had to remember which lookups had failed. The
  parameter block is one struct passed by value now - there is nothing to look
  up and nothing to be missing.
* The 2D texture wrap. State lived in 2D textures at a shared width because
  Blender could not read a 1D texture back. Buffers index directly, so
  `tex_width`, `particle_texel()` and `cell_texel()` have no successors.

A whole frame - every substep, then the surface splat, then whitewater - is
recorded into one queue and committed once. Recording is cheap; the commit is
where the GPU work happens and where a read-back becomes valid.
"""

import ctypes

from ..backend import DeviceError, select
from . import kernels
from .params import ParamBlock

# Threads per group. 64 matches the old GLSL local_group_size and is a whole
# number of the M-series' 32-wide execution width.
GROUP_SIZE = 64

# The SPH state buffers, and how wide each element is in floats. Every one is a
# flat array indexed by particle (or by cell), which is the whole simplification
# over the texture model these replaced.
_PARTICLE_BUFFERS = (
    ("positions", 4),
    ("velocities", 4),
    ("lambda", 4),
    ("predicted", 4),
    ("delta", 4),
    ("normal", 4),
)

_SORTED_BUFFERS = (("keys", 4),)
_CELL_BUFFERS = (("cell_start", 1), ("cell_end", 1))

# Binding order, matching the BUF_* indices in kernels/flowx_prelude.h. One
# table for every pass: a kernel declares the slots it wants at their fixed
# index and ignores the rest.
BINDING_ORDER = (
    "positions",
    "velocities",
    "lambda",
    "predicted",
    "delta",
    "normal",
    "keys",
    "cell_start",
    "cell_end",
    "collider",
    "surface",
    "ww_keys",
    "ww_positions",
    "ww_velkind",
)
PARAMS_INDEX = 14


class MetalEngine:
    """Device, compiled kernels, buffers and the frame's command queue."""

    name = "metal"

    def __init__(self, backend):
        self.backend = backend
        self.program = backend.program(kernels.library_source())
        # Compile every pipeline up front: a bad kernel should be reported when
        # the solver starts, not three frames into playback.
        self.kernels = {name: self.program.kernel(name) for name in kernels.entry_points()}
        self.queue = backend.queue()
        self.params = ParamBlock()
        self.buffers = {}
        # The prepared binding table, rebuilt whenever a buffer is swapped.
        # Every pass binds the same one, so it is built once, not per dispatch.
        self._bindings_cache = None
        self.config = None
        # A one-element stand-in bound whenever no collider is tagged. The
        # kernels guard on collider_voxel == 0 and never read it, but a bound
        # slot still has to exist.
        self.empty_collider = backend.buffer(4)

    # --- lifecycle --------------------------------------------------------

    def allocate(self, config, positions):
        """Size every buffer for `config` and seed the particle state.

        `positions` is a flat sequence of particle_count * 4 floats. Everything
        else starts zeroed, which is what a fresh buffer already is.
        """
        self.config = config
        n = config.sorted_count
        make = self.backend.buffer

        seed = (ctypes.c_float * (n * 4))()
        seed[: len(positions)] = positions

        self.buffers = {}
        for name, width in _PARTICLE_BUFFERS:
            data = seed if name in ("positions", "predicted") else None
            self.buffers[name] = make(n * width * 4, data)
        for name, width in _SORTED_BUFFERS:
            self.buffers[name] = make(n * width * 4)
        for name, width in _CELL_BUFFERS:
            self.buffers[name] = make(max(1, config.cell_count) * width * 4)
        self._bindings_cache = None
        self.buffers["collider"] = self.empty_collider
        # Owned by solver/surface.py and solver/whitewater.py, which install
        # them here so the shared binding table stays complete.
        for name in ("surface", "ww_keys", "ww_positions", "ww_velkind"):
            self.buffers.setdefault(name, None)

    def set_collider(self, buffer, dims, voxel_size, occupancy=None):
        """Point the collider slot at a grid, or back at the empty stand-in.

        `occupancy` is the same grid in raw host bytes, which only the CPU
        engine needs; this one binds the device buffer and ignores it.
        """
        del occupancy
        self.buffers["collider"] = buffer if buffer is not None else self.empty_collider
        self._bindings_cache = None
        self.params.update(
            collider_x=dims[0],
            collider_y=dims[1],
            collider_z=dims[2],
            collider_voxel=voxel_size,
        )

    def install(self, name, buffer):
        """Let surface/whitewater put their own buffers in the binding table."""
        self.buffers[name] = buffer
        self._bindings_cache = None

    def release(self):
        self.buffers = {}
        self._bindings_cache = None
        self.config = None

    # --- dispatch ---------------------------------------------------------

    def _bindings(self):
        if self._bindings_cache is None:
            self._bindings_cache = self.backend.bindings(
                [self.buffers.get(name) for name in BINDING_ORDER]
            )
        return self._bindings_cache

    def record(self, pass_name, threads, **overrides):
        """Record one dispatch. Nothing runs until flush().

        Every pass is bound the same list of buffers and the same parameter
        block; `overrides` are the handful of fields that change per dispatch
        (the bitonic step, mostly).
        """
        if threads <= 0:
            return
        self.queue.dispatch(
            self.kernels[pass_name],
            threads,
            GROUP_SIZE,
            self._bindings(),
            constants=self.params.pack(**overrides),
            constants_index=PARAMS_INDEX,
        )

    def flush(self):
        """Run everything recorded so far and wait for it.

        Waiting is what makes a following read() see this frame's results. It
        is also the only place the frame's GPU time is actually spent, which is
        why sph.py times a whole step rather than individual passes.
        """
        self.queue.commit(wait=True)

    def bitonic(self, pass_name, count):
        """Record a full bitonic sort of `count` (a power of two) elements.

        O(log^2 n) compare-exchange passes - about 105 of them at the shipped
        particle counts, which is why they are recorded rather than submitted
        one at a time. An atomic counting sort would take three dispatches
        instead; it was impossible under Blender's Metal backend and is the
        intended replacement now that it is not.
        """
        k = 2
        while k <= count:
            j = k >> 1
            while j > 0:
                self.record(pass_name, count, bitonic_k=k, bitonic_j=j)
                j >>= 1
            k <<= 1

    def build_grid(self):
        """Record the spatial-hash build: keys, sort, clear, ranges."""
        config = self.config
        self.record("sph_grid_key", config.sorted_count)
        self.bitonic("sph_sort", config.sorted_count)
        self.record("sph_cell_clear", config.cell_count)
        self.record("sph_cell_range", config.sorted_count)

    def substep(self, dt):
        """Record one PBF substep."""
        config = self.config
        n = config.particle_count
        self.params.update(dt=dt)

        if config.surface_tension > 0.0:
            self.record("sph_normal", n)
        self.record("sph_predict", n)
        self.build_grid()
        for _ in range(config.iterations):
            self.record("sph_lambda", n)
            self.record("sph_delta", n)
            self.record("sph_apply_delta", n)
        self.record("sph_velocity", n)
        self.record("sph_xsph", n)
        self.record("sph_finalize", n)

    # --- surface ----------------------------------------------------------

    def alloc_surface(self, sample_count):
        """Allocate the surface field: one float per lattice point."""
        buffer = self.backend.buffer(sample_count * 4)
        self.install("surface", buffer)

    def splat_surface(self, surface):
        """Record the surface splat for `surface` (a solver.surface.SurfaceConfig).

        The grid's geometry and its own kernel radius are fields of the shared
        parameter block. They used to be smuggled through slots borrowed from
        the SPH block - the grid corner rode in the domain-max lanes and the
        kernel radius overwrote the smoothing radius - because that block was
        full at exactly 128 bytes.
        """
        self.params.update(
            surface_x=surface.dims[0],
            surface_y=surface.dims[1],
            surface_z=surface.dims[2],
            surface_spacing=surface.spacing,
            surface_lo_x=surface.lo.x,
            surface_lo_y=surface.lo.y,
            surface_lo_z=surface.lo.z,
            surface_kernel_radius=surface.kernel_radius,
        )
        self.record("surface_splat", surface.sample_count)

    def read_surface(self, sample_count):
        return self.read_floats("surface", sample_count)

    # --- whitewater ---------------------------------------------------------

    def alloc_whitewater(self, capacity, sorted_count):
        """Allocate the whitewater pool and its scoring array."""
        self.install("ww_positions", self.backend.buffer(capacity * 4 * 4))
        self.install("ww_velkind", self.backend.buffer(capacity * 4 * 4))
        # Sized to the fluid's own key array: whitewater_potential scores one
        # fluid particle per slot and the sort runs over the same length.
        self.install("ww_keys", self.backend.buffer(sorted_count * 4 * 4))

    def step_whitewater(self, ww, cursor, spawn_count, frame, frame_dt):
        """Record score, sort, spawn and advect.

        All four read the parameter block the engine already holds for this
        frame. They used to need three separate 128-byte push-constant blocks,
        each re-declaring the fluid's own layout slots because the shared
        prelude referenced them whether or not the pass body did.
        """
        n = self.config.sorted_count
        self.params.update(
            ww_capacity=ww.capacity,
            ww_cursor=cursor,
            ww_spawn_count=spawn_count,
            frame=frame,
            frame_dt=frame_dt,
            trapped_air_weight=ww.trapped_air_weight,
            wave_crest_weight=ww.wave_crest_weight,
            kinetic_weight=ww.kinetic_weight,
            kinetic_reference_speed=ww.kinetic_reference_speed,
            spray_speed_threshold=ww.spray_speed_threshold,
            bubble_trapped_threshold=ww.bubble_trapped_threshold,
            jitter_strength=ww.jitter_strength,
            normal_offset=ww.normal_offset,
            spray_life_min=ww.spray_life[0],
            spray_life_max=ww.spray_life[1],
            foam_life_min=ww.foam_life[0],
            foam_life_max=ww.foam_life[1],
            bubble_life_min=ww.bubble_life[0],
            bubble_life_max=ww.bubble_life[1],
            ww_drag=ww.drag,
            ww_buoyancy=ww.buoyancy,
        )
        self.record("whitewater_potential", n)
        self.bitonic("whitewater_sort", n)
        if spawn_count > 0:
            self.record("whitewater_spawn", spawn_count)
        self.record("whitewater_advect", ww.capacity)

    def read_whitewater(self, capacity):
        """The pool as (positions+life, velocity+kind) lists of 4-tuples."""
        return (self.read_vec4("ww_positions", capacity), self.read_vec4("ww_velkind", capacity))

    # --- read-back --------------------------------------------------------

    def read_vec4(self, name, count):
        """A buffer's first `count` elements as a list of 4-float tuples.

        On unified memory the mapping *is* the buffer, so nothing is
        transferred here - the whole cost is building Python tuples. The
        zip-of-one-iterator trick groups the flat floats into fours in C rather
        than with a slice per particle, which at these counts is worth the
        moment it takes to read.
        """
        flat = self.buffers[name].map().cast("f")[: count * 4].tolist()
        stream = iter(flat)
        return list(zip(stream, stream, stream, stream, strict=True))

    def read_floats(self, name, count):
        """A single-channel buffer's first `count` values as a list of floats.

        memoryview.tolist() over the mapping, which is a single C-level
        conversion - this is the surface field, and it is the largest read-back
        in the pipeline.
        """
        return self.buffers[name].map().cast("f")[:count].tolist()

    def zero(self, name):
        """Zero a state buffer in place."""
        self.buffers[name].zero()

    def upload_vec4(self, name, values, count):
        """Overwrite a buffer's first `count` elements from flat float data.

        Used by the cache-load path, which restores a frame's state without
        simulating it.
        """
        view = self.buffers[name].map()
        flat = (ctypes.c_float * (count * 4)).from_buffer(view)
        flat[: len(values)] = values


def create():
    """Bring up the Metal engine, or return None if there is no usable device."""
    backend = select()
    if backend is None:
        return None
    try:
        return MetalEngine(backend)
    except DeviceError as exc:
        print(f"Flow-X: Metal engine unavailable ({exc})")
        return None
