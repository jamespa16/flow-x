"""NumPy reference and fallback for the MAC-grid APIC solver.

Particles carry velocity plus a locally affine velocity field.  A substep
advects them, transfers momentum to staggered grid faces, applies forces and a
constant-density pressure projection, then transfers the divergence-reduced
velocity back.  Surface extraction and whitewater are inherited from the PBF
CPU engine because both consume only particle positions, velocities and the
shared spatial hash.

No ``bpy`` import belongs here: this module is exercised directly in CI.
"""

import itertools
import math

import numpy as np

from .cpu_engine import CpuEngine

_CORNERS = tuple(itertools.product((0, 1), repeat=3))
_FACE_OFFSETS = (
    np.array((0.0, 0.5, 0.5), dtype=np.float64),
    np.array((0.5, 0.0, 0.5), dtype=np.float64),
    np.array((0.5, 0.5, 0.0), dtype=np.float64),
)
_JACOBI_OMEGA = 2.0 / 3.0
_INVERSE_EPSILON = 1e-8


class ApicCpuEngine(CpuEngine):
    """APIC persistent state plus a vectorized staggered-grid solve."""

    method = "apic"

    def __init__(self):
        super().__init__()
        self.grid_mass = None
        self.grid_momentum = None
        self.grid_velocity = None
        self.grid_vorticity = None
        self.grid_cell = None
        self.last_divergence_before = 0.0
        self.last_divergence_after = 0.0

    def allocate(self, config, positions):
        super().allocate(config, positions)
        n = config.particle_count
        self.state["affine"] = np.zeros((n, 3, 4), dtype=np.float32)
        nx, ny, nz = config.cell_dims
        node_shape = (nz + 1, ny + 1, nx + 1, 4)
        cell_shape = (nz, ny, nx, 4)
        self.grid_mass = np.zeros(node_shape, dtype=np.float32)
        self.grid_momentum = np.zeros(node_shape, dtype=np.float32)
        self.grid_velocity = np.zeros(node_shape, dtype=np.float32)
        self.grid_vorticity = np.zeros(cell_shape, dtype=np.float32)
        self.grid_cell = np.zeros(cell_shape, dtype=np.float32)

    def release(self):
        super().release()
        self.grid_mass = None
        self.grid_momentum = None
        self.grid_velocity = None
        self.grid_vorticity = None
        self.grid_cell = None

    # --- particle and grid geometry -------------------------------------

    def _advect_particles(self, dt):
        P = self.params
        positions = self.state["positions"][:, :3]
        velocities = self.state["velocities"][:, :3]
        positions += velocities * dt

        voxel = P["collider_voxel"]
        if self.collider is not None and voxel > 0.0:
            stuck = np.where(self._occupied(positions))[0]
            radius = P["particle_radius"]
            damping = P["boundary_damping"]
            for index in stuck:
                nearest, dist2 = self._nearest_free_voxel(positions[index], voxel)
                if nearest is None:
                    continue
                dist = math.sqrt(dist2)
                push = (
                    (nearest - positions[index]) / dist
                    if dist > 1e-6
                    else np.array((0.0, 0.0, 1.0), dtype=np.float32)
                )
                positions[index] += push * (dist + radius)
                vn = float(np.dot(velocities[index], push))
                if vn < 0.0:
                    velocities[index] -= vn * (1.0 + damping) * push

        lo = self._lo() + P["particle_radius"]
        hi = self._hi() - P["particle_radius"]
        damping = P["boundary_damping"]
        for axis in range(3):
            below = positions[:, axis] < lo[axis]
            above = positions[:, axis] > hi[axis]
            positions[below, axis] = lo[axis]
            positions[above, axis] = hi[axis]
            velocities[below | above, axis] *= -damping

        self.state["positions"][:, 3] = 1.0
        self.state["velocities"][:, 3] = 0.0
        self.state["predicted"][:] = self.state["positions"]

    def _solid_cells(self):
        nx, ny, nz = self.config.cell_dims
        dx = self.params["grid_spacing"]
        z, y, x = np.indices((nz, ny, nx), dtype=np.float64)
        points = self._lo() + np.stack((x + 0.5, y + 0.5, z + 0.5), axis=-1) * dx
        return self._occupied(points.reshape(-1, 3)).reshape(nz, ny, nx)

    def _classify_cells(self):
        nx, ny, nz = self.config.cell_dims
        cell_type = np.zeros((nz, ny, nx), dtype=np.float32)
        coords = self._cell_coords(self.state["positions"][:, :3])
        cell_type[coords[:, 2], coords[:, 1], coords[:, 0]] = 1.0
        cell_type[self._solid_cells()] = -1.0
        self.grid_cell[..., 3] = cell_type
        return cell_type

    @staticmethod
    def _face_dims(axis, dims):
        nx, ny, nz = dims
        if axis == 0:
            return nx + 1, ny, nz
        if axis == 1:
            return nx, ny + 1, nz
        return nx, ny, nz + 1

    def _face_samples(self, axis):
        """Yield particle indices, face indices, weights and offsets per corner."""
        points = self.state["positions"][:, :3].astype(np.float64)
        dx = float(self.params["grid_spacing"])
        local = (points - self._lo()) / dx - _FACE_OFFSETS[axis]
        base = np.floor(local).astype(np.int64)
        frac = local - base
        dims = np.array(self._face_dims(axis, self.config.cell_dims), dtype=np.int64)

        for corner in _CORNERS:
            c = np.asarray(corner, dtype=np.int64)
            index = base + c
            valid = np.all((index >= 0) & (index < dims), axis=1)
            if not valid.any():
                continue
            weights = np.where(c, frac, 1.0 - frac).prod(axis=1)
            valid &= weights > 0.0
            if not valid.any():
                continue
            particle = np.flatnonzero(valid)
            face = index[valid]
            face_position = self._lo() + (face + _FACE_OFFSETS[axis]) * dx
            offset = face_position - points[valid]
            yield particle, face, weights[valid], offset

    # --- APIC transfers --------------------------------------------------

    def _particle_to_grid(self):
        self.grid_mass.fill(0.0)
        self.grid_momentum.fill(0.0)
        self.grid_velocity.fill(0.0)
        self.grid_vorticity.fill(0.0)
        self.grid_cell.fill(0.0)

        mass = float(self.params["mass"])
        velocity = self.state["velocities"][:, :3].astype(np.float64)
        affine = self.state["affine"][:, :, :3].astype(np.float64)
        for axis in range(3):
            for particle, face, weight, offset in self._face_samples(axis):
                value = velocity[particle, axis] + np.einsum(
                    "ij,ij->i", affine[particle, axis], offset
                )
                target = (face[:, 2], face[:, 1], face[:, 0], axis)
                np.add.at(self.grid_mass, target, (weight * mass).astype(np.float32))
                np.add.at(
                    self.grid_momentum,
                    target,
                    (weight * mass * value).astype(np.float32),
                )

    def _normalise_and_force(self, dt):
        active = self.grid_mass[..., :3] > 1e-12
        np.divide(
            self.grid_momentum[..., :3],
            self.grid_mass[..., :3],
            out=self.grid_velocity[..., :3],
            where=active,
        )
        self.grid_velocity[..., 2][active[..., 2]] += self.params["gravity"] * dt
        vmax = max(float(self.params["grid_max_speed"]), 1e-6)
        np.clip(self.grid_velocity[..., :3], -vmax, vmax, out=self.grid_velocity[..., :3])

    def _apply_solid_faces(self, cell_type):
        u = self.grid_velocity[..., 0]
        v = self.grid_velocity[..., 1]
        w = self.grid_velocity[..., 2]
        nx, ny, nz = self.config.cell_dims

        u[:, :, 0] = 0.0
        u[:, :, nx] = 0.0
        if nx > 1:
            u[:nz, :ny, 1:nx][(cell_type[:, :, :-1] < 0.0) | (cell_type[:, :, 1:] < 0.0)] = 0.0
        v[:, 0, :] = 0.0
        v[:, ny, :] = 0.0
        if ny > 1:
            v[:nz, 1:ny, :nx][(cell_type[:, :-1, :] < 0.0) | (cell_type[:, 1:, :] < 0.0)] = 0.0
        w[0, :, :] = 0.0
        w[nz, :, :] = 0.0
        if nz > 1:
            w[1:nz, :ny, :nx][(cell_type[:-1, :, :] < 0.0) | (cell_type[1:, :, :] < 0.0)] = 0.0

    @staticmethod
    def _derivative(values, axis, spacing):
        if values.shape[axis] <= 1:
            return np.zeros_like(values)
        return np.gradient(values, spacing, axis=axis, edge_order=1)

    def _vorticity_confinement(self, dt, cell_type):
        epsilon = float(self.params["vorticity_epsilon"])
        if epsilon <= 0.0:
            return
        nx, ny, nz = self.config.cell_dims
        dx = float(self.params["grid_spacing"])
        u = self.grid_velocity[:nz, :ny, : nx + 1, 0]
        v = self.grid_velocity[:nz, : ny + 1, :nx, 1]
        w = self.grid_velocity[: nz + 1, :ny, :nx, 2]
        cell_velocity = np.stack(
            (
                0.5 * (u[:, :, :-1] + u[:, :, 1:]),
                0.5 * (v[:, :-1, :] + v[:, 1:, :]),
                0.5 * (w[:-1, :, :] + w[1:, :, :]),
            ),
            axis=-1,
        )
        curl = np.empty_like(cell_velocity)
        curl[..., 0] = self._derivative(cell_velocity[..., 2], 1, dx) - self._derivative(
            cell_velocity[..., 1], 0, dx
        )
        curl[..., 1] = self._derivative(cell_velocity[..., 0], 0, dx) - self._derivative(
            cell_velocity[..., 2], 2, dx
        )
        curl[..., 2] = self._derivative(cell_velocity[..., 1], 2, dx) - self._derivative(
            cell_velocity[..., 0], 1, dx
        )
        magnitude = np.linalg.norm(curl, axis=-1)
        self.grid_vorticity[..., :3] = curl
        self.grid_vorticity[..., 3] = magnitude

        grad = np.stack(
            (
                self._derivative(magnitude, 2, dx),
                self._derivative(magnitude, 1, dx),
                self._derivative(magnitude, 0, dx),
            ),
            axis=-1,
        )
        length = np.linalg.norm(grad, axis=-1)
        normal = np.divide(
            grad, length[..., None], out=np.zeros_like(grad), where=length[..., None] > 1e-8
        )
        force = epsilon * dx * np.cross(normal, curl)
        force[cell_type < 0.0] = 0.0

        u[:, :, 1:nx] += 0.5 * dt * (force[:, :, :-1, 0] + force[:, :, 1:, 0])
        v[:, 1:ny, :] += 0.5 * dt * (force[:, :-1, :, 1] + force[:, 1:, :, 1])
        w[1:nz, :, :] += 0.5 * dt * (force[:-1, :, :, 2] + force[1:, :, :, 2])
        vmax = float(self.params["grid_max_speed"])
        np.clip(u[:, :, 1:nx], -vmax, vmax, out=u[:, :, 1:nx])
        np.clip(v[:, 1:ny, :], -vmax, vmax, out=v[:, 1:ny, :])
        np.clip(w[1:nz, :, :], -vmax, vmax, out=w[1:nz, :, :])

    # --- pressure projection --------------------------------------------

    def _divergence(self):
        nx, ny, nz = self.config.cell_dims
        dx = float(self.params["grid_spacing"])
        u = self.grid_velocity[:nz, :ny, : nx + 1, 0]
        v = self.grid_velocity[:nz, : ny + 1, :nx, 1]
        w = self.grid_velocity[: nz + 1, :ny, :nx, 2]
        return (
            u[:, :, 1:] - u[:, :, :-1] + v[:, 1:, :] - v[:, :-1, :] + w[1:, :, :] - w[:-1, :, :]
        ) / dx

    @staticmethod
    def _neighbour_sum_and_diag(pressure, cell_type):
        total = np.zeros_like(pressure)
        diagonal = np.zeros_like(pressure)
        for axis in range(3):
            for shift in (-1, 1):
                src = [slice(None)] * 3
                dst = [slice(None)] * 3
                if shift < 0:
                    src[axis] = slice(0, -1)
                    dst[axis] = slice(1, None)
                else:
                    src[axis] = slice(1, None)
                    dst[axis] = slice(0, -1)
                src, dst = tuple(src), tuple(dst)
                neighbour_type = cell_type[src]
                non_solid = neighbour_type >= 0.0
                diagonal[dst] += non_solid
                total[dst] += np.where(neighbour_type > 0.0, pressure[src], 0.0)
        return total, diagonal

    def _project(self, dt, cell_type):
        divergence = self._divergence()
        fluid = cell_type > 0.0
        self.grid_cell[..., 0] = divergence
        self.last_divergence_before = (
            float(np.sqrt(np.mean(divergence[fluid] ** 2))) if fluid.any() else 0.0
        )

        dx = float(self.params["grid_spacing"])
        density = max(float(self.params["rest_density"]), 1e-6)
        rhs = density * dx * dx * divergence / max(dt, 1e-8)
        pressure = np.zeros_like(divergence)
        other = np.zeros_like(divergence)
        iterations = int(getattr(self.config, "pressure_iterations", 40))
        for _ in range(iterations):
            total, diagonal = self._neighbour_sum_and_diag(pressure, cell_type)
            candidate = np.divide(
                total - rhs,
                diagonal,
                out=np.zeros_like(total),
                where=diagonal > 0.0,
            )
            other[:] = 0.0
            other[fluid] = (1.0 - _JACOBI_OMEGA) * pressure[fluid] + _JACOBI_OMEGA * candidate[
                fluid
            ]
            pressure, other = other, pressure
        self.grid_cell[..., 1] = pressure

        scale = dt / (density * dx)
        nx, ny, nz = self.config.cell_dims
        u = self.grid_velocity[:nz, :ny, : nx + 1, 0]
        v = self.grid_velocity[:nz, : ny + 1, :nx, 1]
        w = self.grid_velocity[: nz + 1, :ny, :nx, 2]

        if nx > 1:
            left, right = cell_type[:, :, :-1], cell_type[:, :, 1:]
            active = (left > 0.0) | (right > 0.0)
            solid = (left < 0.0) | (right < 0.0)
            gradient = np.where(right > 0.0, pressure[:, :, 1:], 0.0) - np.where(
                left > 0.0, pressure[:, :, :-1], 0.0
            )
            target = u[:, :, 1:nx]
            target[active & ~solid] -= scale * gradient[active & ~solid]
            target[solid] = 0.0
        if ny > 1:
            low, high = cell_type[:, :-1, :], cell_type[:, 1:, :]
            active = (low > 0.0) | (high > 0.0)
            solid = (low < 0.0) | (high < 0.0)
            gradient = np.where(high > 0.0, pressure[:, 1:, :], 0.0) - np.where(
                low > 0.0, pressure[:, :-1, :], 0.0
            )
            target = v[:, 1:ny, :]
            target[active & ~solid] -= scale * gradient[active & ~solid]
            target[solid] = 0.0
        if nz > 1:
            low, high = cell_type[:-1, :, :], cell_type[1:, :, :]
            active = (low > 0.0) | (high > 0.0)
            solid = (low < 0.0) | (high < 0.0)
            gradient = np.where(high > 0.0, pressure[1:, :, :], 0.0) - np.where(
                low > 0.0, pressure[:-1, :, :], 0.0
            )
            target = w[1:nz, :, :]
            target[active & ~solid] -= scale * gradient[active & ~solid]
            target[solid] = 0.0

        self._apply_solid_faces(cell_type)
        after = self._divergence()
        self.last_divergence_after = (
            float(np.sqrt(np.mean(after[fluid] ** 2))) if fluid.any() else 0.0
        )

    def _grid_to_particle(self):
        n = self.config.particle_count
        velocity = np.zeros((n, 3), dtype=np.float64)
        affine = np.zeros((n, 3, 3), dtype=np.float64)
        dx = float(self.params["grid_spacing"])

        for axis in range(3):
            moment = np.zeros((n, 3, 3), dtype=np.float64)
            covariance = np.zeros((n, 3), dtype=np.float64)
            for particle, face, weight, offset in self._face_samples(axis):
                values = self.grid_velocity[face[:, 2], face[:, 1], face[:, 0], axis]
                np.add.at(velocity[:, axis], particle, weight * values)
                local = offset / dx
                for a in range(3):
                    np.add.at(covariance[:, a], particle, weight * values * local[:, a])
                    for b in range(3):
                        np.add.at(
                            moment[:, a, b],
                            particle,
                            weight * local[:, a] * local[:, b],
                        )

            determinant = np.linalg.det(moment)
            good = np.abs(determinant) >= _INVERSE_EPSILON
            if good.any():
                inverse = np.linalg.inv(moment[good])
                affine[good, axis] = np.einsum("ij,ijk->ik", covariance[good], inverse) / dx

        self.state["velocities"][:, :3] = velocity.astype(np.float32)
        self.state["velocities"][:, 3] = 0.0
        self.state["affine"][:, :, :3] = affine.astype(np.float32)
        self.state["affine"][:, :, 3] = 0.0

    # --- engine protocol -------------------------------------------------

    def substep(self, dt):
        self.params.update(dt=dt)
        self._advect_particles(dt)
        self.build_grid()
        self._particle_to_grid()
        cell_type = self._classify_cells()
        self._normalise_and_force(dt)
        self._apply_solid_faces(cell_type)
        self._vorticity_confinement(dt, cell_type)
        self._apply_solid_faces(cell_type)
        self._project(dt, cell_type)
        self._grid_to_particle()

    def snapshot_state(self, include_whitewater=False):
        state = super().snapshot_state(include_whitewater)
        count = self.config.particle_count
        state["affine"] = [
            tuple(row) for row in self.state["affine"][:count].reshape(count, 12).tolist()
        ]
        return state

    def restore_state(self, state):
        super().restore_state(state)
        count = self.config.particle_count
        affine = np.asarray(state["affine"], dtype=np.float32).reshape(-1, 3, 4)[:count]
        self.state["affine"][: len(affine)] = affine
        self.grid_mass.fill(0.0)
        self.grid_momentum.fill(0.0)
        self.grid_velocity.fill(0.0)
        self.grid_vorticity.fill(0.0)
        self.grid_cell.fill(0.0)


def create():
    return ApicCpuEngine()
