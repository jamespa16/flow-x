"""Metal implementation of the staggered-grid APIC engine."""

import ctypes

from ..backend import select
from . import kernels
from .metal_engine import GROUP_SIZE, MetalEngine

# Slots of the PCG scalar buffer; PCG_* in kernels/apic_common.h.
_PCG_SCALAR_COUNT = 8
_PCG_ITERATIONS = 5
_PCG_RR = 7

# apic_pcg_reduce stages; PCG_STAGE_* in kernels/apic_common.h.
_STAGE_INIT = 0
_STAGE_CURVATURE = 1
_STAGE_RESIDUAL = 2


def _whole_groups(count):
    """`count` rounded up to whole threadgroups.

    The reduction passes need every group full: their fixed tree has a barrier
    at each level that all 64 lanes must reach, and a lane Metal never launched
    would leave its slot of the tree unwritten. Lanes past the end contribute
    zero instead.
    """
    return -(-count // GROUP_SIZE) * GROUP_SIZE


class ApicMetalEngine(MetalEngine):
    method = "apic"

    def __init__(self, backend):
        super().__init__(backend, passes=kernels.APIC_ALL_PASSES)

    def allocate(self, config, positions):
        super().allocate(config, positions)
        make = self.backend.buffer
        particle_count = config.particle_count
        nx, ny, nz = config.cell_dims
        node_count = (nx + 1) * (ny + 1) * (nz + 1)
        cell_count = config.cell_count
        self.install("affine", make(particle_count * 12 * 4))
        self.install("grid_mass", make(node_count * 4 * 4))
        self.install("grid_momentum", make(node_count * 4 * 4))
        self.install("grid_velocity", make(node_count * 4 * 4))
        self.install("grid_vort", make(cell_count * 4 * 4))
        self.install("grid_scratch", make(cell_count * 4 * 4))
        self.install("grid_velocity_old", make(node_count * 4 * 4))
        self.install("pcg", make(cell_count * 4 * 4))
        self.install("pcg_partials", make(_whole_groups(cell_count) // GROUP_SIZE * 4 * 4))
        self.install("pcg_scalars", make(_PCG_SCALAR_COUNT * 4))

    def substep(self, dt):
        config = self.config
        n = config.particle_count
        cells = config.cell_count
        nx, ny, nz = config.cell_dims
        nodes = (nx + 1) * (ny + 1) * (nz + 1)
        self.params.update(dt=dt)
        # Under FLIP blending, advection moves to the end of the substep: see
        # ApicCpuEngine.substep. At blend 0 the historical order is kept.
        flip = self.params["flip_blend"] > 0.0

        if not flip:
            self.record("apic_advect", n)
        self.build_grid()
        self.record("apic_grid_clear", max(nodes, cells))
        self.record("apic_classify", cells)
        self.record("apic_p2g", nodes)
        self.record("apic_grid_update", nodes)
        if config.vorticity_strength > 0.0:
            self.record("apic_vorticity", cells)
            self.record("apic_confinement", nodes)
        self.record("apic_divergence", cells)

        ping = 0
        if config.pressure_solver == "pcg":
            self.record_pcg()
        else:
            for _ in range(config.pressure_iterations):
                self.record("apic_pressure", cells, pressure_ping=ping)
                ping = 1 - ping
        self.record("apic_project", nodes, pressure_ping=ping)
        self.record("apic_g2p", n)
        if flip:
            self.record("apic_advect", n)

    def record_pcg(self):
        """Record a PCG pressure solve into cell_data.y; see apic_pcg_reduce.

        A fixed `pressure_iterations` passes of five dispatches each, with no
        read-back: the reduce pass decides on the device when to stop, and the
        vector passes after that return at once. A frame stays one submit, and
        record() stays free of GPU work - at the price of dispatching the
        iterations a converged solve no longer needs, which costs one uniform
        branch per thread each.
        """
        config = self.config
        cells = _whole_groups(config.cell_count)
        self.record("apic_pcg_init", cells)
        self.record("apic_pcg_reduce", GROUP_SIZE, pcg_stage=_STAGE_INIT)
        for _ in range(config.pressure_iterations):
            self.record("apic_pcg_matvec", cells)
            self.record("apic_pcg_reduce", GROUP_SIZE, pcg_stage=_STAGE_CURVATURE)
            self.record("apic_pcg_update", cells)
            self.record("apic_pcg_reduce", GROUP_SIZE, pcg_stage=_STAGE_RESIDUAL)
            self.record("apic_pcg_direction", config.cell_count)

    def pcg_stats(self):
        """(iterations, final r.r) of the last flushed PCG solve, for tests."""
        values = self.buffers["pcg_scalars"].map().cast("f")
        return int(values[_PCG_ITERATIONS]), float(values[_PCG_RR])

    def snapshot_state(self, include_whitewater=False):
        state = super().snapshot_state(include_whitewater)
        count = self.config.particle_count
        flat = self.buffers["affine"].map().cast("f")[: count * 12].tolist()
        stream = iter(flat)
        state["affine"] = list(zip(*([stream] * 12), strict=True))
        return state

    def restore_state(self, state):
        super().restore_state(state)
        count = self.config.particle_count
        values = [component for row in state["affine"] for component in row]
        view = self.buffers["affine"].map()
        flat = (ctypes.c_float * (count * 12)).from_buffer(view)
        flat[: len(values)] = values
        for name in (
            "grid_mass",
            "grid_momentum",
            "grid_velocity",
            "grid_vort",
            "grid_scratch",
            "grid_velocity_old",
            "pcg",
            "pcg_partials",
            "pcg_scalars",
        ):
            self.buffers[name].zero()


def create():
    backend = select()
    if backend is None:
        return None
    return ApicMetalEngine(backend)
