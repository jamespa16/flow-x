"""Metal implementation of the staggered-grid APIC engine."""

import ctypes

from ..backend import select
from . import kernels
from .metal_engine import MetalEngine


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

    def substep(self, dt):
        config = self.config
        n = config.particle_count
        cells = config.cell_count
        nx, ny, nz = config.cell_dims
        nodes = (nx + 1) * (ny + 1) * (nz + 1)
        self.params.update(dt=dt)

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
        for _ in range(config.pressure_iterations):
            self.record("apic_pressure", cells, pressure_ping=ping)
            ping = 1 - ping
        self.record("apic_project", nodes, pressure_ping=ping)
        self.record("apic_g2p", n)

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
        ):
            self.buffers[name].zero()


def create():
    backend = select()
    if backend is None:
        return None
    return ApicMetalEngine(backend)
