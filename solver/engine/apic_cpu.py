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
# PCG breakdown guard, relative to r.z at the start of the solve. p.Ap can only
# approach zero once the residual has nothing left in the matrix's range - a
# converged solve, or a sealed fluid pocket whose right-hand side has a
# roundoff-sized component along the constant null vector. Either way the right
# move is to stop, not to divide by it. kernels/apic_pcg_reduce.metal uses the
# same value.
_PCG_BREAKDOWN = 1e-12
_NEIGHBOUR_AXES = ((2, -1), (2, 1), (1, -1), (1, 1), (0, -1), (0, 1))


class ApicCpuEngine(CpuEngine):
    """APIC persistent state plus a vectorized staggered-grid solve."""

    method = "apic"

    def __init__(self):
        super().__init__()
        self.grid_mass = None
        self.grid_momentum = None
        self.grid_velocity = None
        # The transferred grid velocity before forces and projection, which
        # FLIP blending measures the grid's change against.
        self.grid_velocity_old = None
        self.grid_vorticity = None
        self.grid_cell = None
        self.last_divergence_before = 0.0
        self.last_divergence_after = 0.0
        # Max-norm counterparts of the RMS pair above, and how many iterations
        # the last solve actually ran (PCG can stop early; Jacobi never does).
        self.last_divergence_max_before = 0.0
        self.last_divergence_max_after = 0.0
        self.last_pressure_iterations = 0

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
        self.grid_velocity_old = np.zeros(node_shape, dtype=np.float32)
        self.grid_vorticity = np.zeros(cell_shape, dtype=np.float32)
        self.grid_cell = np.zeros(cell_shape, dtype=np.float32)

    def release(self):
        super().release()
        self.grid_mass = None
        self.grid_momentum = None
        self.grid_velocity = None
        self.grid_velocity_old = None
        self.grid_vorticity = None
        self.grid_cell = None

    # --- particle and grid geometry -------------------------------------

    def _advect_particles(self, dt, motion=None):
        """Move particles, then resolve collider and domain penetration.

        `motion` is the velocity to move by when it is not the stored one: under
        FLIP blending, the grid's interpolated field from this substep's G2P
        (see substep). The stored velocity still takes the collision response,
        since it is what the next P2G transfers.
        """
        if motion is not None:
            self._advect_flip(dt, motion)
            return
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

    def _advect_flip(self, dt, motion):
        P = self.params
        positions = self.state["positions"][:, :3]
        velocities = self.state["velocities"][:, :3]
        positions += (motion * dt).astype(np.float32)

        voxel = P["collider_voxel"]
        radius = P["particle_radius"]
        damping = P["boundary_damping"]
        if self.collider is not None and voxel > 0.0:
            for index in np.where(self._occupied(positions))[0]:
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

        # The carried velocity can point anywhere relative to the motion that
        # hit the wall, so only its into-wall component is reflected - unlike
        # the APIC path, where the two are the same vector.
        lo = self._lo() + radius
        hi = self._hi() - radius
        for axis in range(3):
            below = positions[:, axis] < lo[axis]
            above = positions[:, axis] > hi[axis]
            positions[below, axis] = lo[axis]
            positions[above, axis] = hi[axis]
            into = (below & (velocities[:, axis] < 0.0)) | (above & (velocities[:, axis] > 0.0))
            velocities[into, axis] *= -damping

        self.state["positions"][:, 3] = 1.0
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
        vmax = max(float(self.params["grid_max_speed"]), 1e-6)
        # Capped, but before gravity and solid faces - see apic_grid_update.metal.
        np.clip(self.grid_velocity, -vmax, vmax, out=self.grid_velocity_old)
        self.grid_velocity[..., 2][active[..., 2]] += self.params["gravity"] * dt
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

    @staticmethod
    def _divergence_norms(divergence, fluid):
        """(RMS, max) of |divergence| over fluid cells - the residual metric."""
        if not fluid.any():
            return 0.0, 0.0
        values = divergence[fluid]
        return float(np.sqrt(np.mean(values**2))), float(np.abs(values).max())

    def _solve_jacobi(self, rhs, cell_type):
        fluid = cell_type > 0.0
        pressure = np.zeros_like(rhs)
        other = np.zeros_like(rhs)
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
        self.last_pressure_iterations = iterations
        return pressure

    def _solve_pcg(self, rhs, cell_type):
        """Diagonally preconditioned conjugate gradient on the Jacobi system.

        The same matrix the Jacobi loop relaxes - for each fluid cell, the count
        of non-solid neighbours on the diagonal and -1 per fluid neighbour, air
        pinned at zero pressure - but solved as A p = -rhs over a compact vector
        of fluid cells only. The Metal chain (apic_pcg_*.metal) runs the same
        algorithm with the same stopping rules, in float32.

        Always starts from zero pressure. Warm-starting from the previous
        substep would converge faster, but that pressure is not part of the
        cached state, so a restored frame would continue differently from an
        uninterrupted one.

        Dot products are np.sum over a product array: numpy's pairwise sum, in
        an order fixed by the vector length, so a run is reproducible on one
        machine. np.dot is avoided on purpose - it goes to BLAS, which may
        split the sum across a varying number of threads.
        """
        fluid = cell_type > 0.0
        pressure = np.zeros(cell_type.shape, dtype=np.float32)
        self.last_pressure_iterations = 0
        count = int(fluid.sum())
        if count == 0:
            return pressure

        # Index every fluid cell into the compact vector; everything else maps
        # to slot `count`, an appended zero, so the stencil needs no masking.
        index = np.full(cell_type.shape, count, dtype=np.int64)
        index[fluid] = np.arange(count)
        coords = np.nonzero(fluid)
        neighbours = np.full((6, count), count, dtype=np.int64)
        diagonal = np.zeros(count, dtype=np.float64)
        for slot, (axis, shift) in enumerate(_NEIGHBOUR_AXES):
            moved = list(coords)
            moved[axis] = coords[axis] + shift
            inside = (moved[axis] >= 0) & (moved[axis] < cell_type.shape[axis])
            moved[axis] = np.clip(moved[axis], 0, cell_type.shape[axis] - 1)
            moved = tuple(moved)
            # Outside the domain is a wall, exactly as in the Jacobi stencil.
            neighbour_type = np.where(inside, cell_type[moved], -1.0)
            diagonal += neighbour_type >= 0.0
            neighbours[slot] = np.where(inside & (neighbour_type > 0.0), index[moved], count)
        inverse = np.divide(1.0, diagonal, out=np.zeros_like(diagonal), where=diagonal > 0.0)

        def apply(vector):
            padded = np.append(vector, 0.0)
            return diagonal * vector - padded[neighbours].sum(axis=0)

        def dot(a, b):
            return float(np.sum(a * b))

        residual = -rhs[fluid].astype(np.float64)
        solution = np.zeros(count, dtype=np.float64)
        direction = inverse * residual
        rz = dot(residual, direction)
        tolerance = float(self.params["pressure_tolerance"])
        threshold = tolerance * tolerance * dot(residual, residual)
        guard = _PCG_BREAKDOWN * rz
        iterations = 0
        if rz > 0.0:
            for _ in range(int(getattr(self.config, "pressure_iterations", 40))):
                product = apply(direction)
                curvature = dot(direction, product)
                if not curvature > guard:
                    break
                alpha = rz / curvature
                solution += alpha * direction
                residual -= alpha * product
                iterations += 1
                if dot(residual, residual) <= threshold:
                    break
                preconditioned = inverse * residual
                rz_next = dot(residual, preconditioned)
                direction = preconditioned + (rz_next / rz) * direction
                rz = rz_next
        self.last_pressure_iterations = iterations
        pressure[fluid] = solution.astype(np.float32)
        return pressure

    def _project(self, dt, cell_type):
        divergence = self._divergence()
        fluid = cell_type > 0.0
        self.grid_cell[..., 0] = divergence
        (
            self.last_divergence_before,
            self.last_divergence_max_before,
        ) = self._divergence_norms(divergence, fluid)

        dx = float(self.params["grid_spacing"])
        density = max(float(self.params["rest_density"]), 1e-6)
        rhs = density * dx * dx * divergence / max(dt, 1e-8)
        if getattr(self.config, "pressure_solver", "jacobi") == "pcg":
            pressure = self._solve_pcg(rhs, cell_type)
        else:
            pressure = self._solve_jacobi(rhs, cell_type)
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
        (
            self.last_divergence_after,
            self.last_divergence_max_after,
        ) = self._divergence_norms(self._divergence(), fluid)

    def _grid_to_particle(self):
        n = self.config.particle_count
        velocity = np.zeros((n, 3), dtype=np.float64)
        affine = np.zeros((n, 3, 3), dtype=np.float64)
        dx = float(self.params["grid_spacing"])
        blend = float(self.params["flip_blend"])
        old_velocity = np.zeros((n, 3), dtype=np.float64) if blend > 0.0 else None

        for axis in range(3):
            moment = np.zeros((n, 3, 3), dtype=np.float64)
            covariance = np.zeros((n, 3), dtype=np.float64)
            for particle, face, weight, offset in self._face_samples(axis):
                values = self.grid_velocity[face[:, 2], face[:, 1], face[:, 0], axis]
                np.add.at(velocity[:, axis], particle, weight * values)
                if old_velocity is not None:
                    old = self.grid_velocity_old[face[:, 2], face[:, 1], face[:, 0], axis]
                    np.add.at(old_velocity[:, axis], particle, weight * old)
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

        motion = None
        if old_velocity is not None:
            # FLIP: the particle's own velocity plus the grid's change, blended
            # with the APIC value; the affine rows stay APIC's. Skipped outright
            # at blend 0, so that setting is exactly APIC. See apic_g2p.metal.
            motion = velocity.copy()
            flip = self.state["velocities"][:, :3] + velocity - old_velocity
            velocity += blend * (flip - velocity)

        self.state["velocities"][:, :3] = velocity.astype(np.float32)
        self.state["velocities"][:, 3] = 0.0
        self.state["affine"][:, :, :3] = affine.astype(np.float32)
        self.state["affine"][:, :, 3] = 0.0
        return motion

    # --- engine protocol -------------------------------------------------

    def substep(self, dt):
        """One substep. Under FLIP blending, advection moves to the end.

        FLIP particles carry their own velocity, which accumulates noise that is
        not divergence-free; moving them by it lets them drift together inside
        cells the pressure solve cannot resolve, and a dam break loses half its
        volume in a second. So positions move through the grid's own
        interpolated field instead (Zhu & Bridson 2005), and that field only
        exists between G2P and the next P2G - hence advection after G2P rather
        than before P2G. Nothing extra is persisted, so cache continuation is
        unaffected. At blend 0 the two fields are the same, and the historical
        order is kept so blend 0 stays APIC bit for bit.
        """
        self.params.update(dt=dt)
        flip = float(self.params["flip_blend"]) > 0.0
        if not flip:
            self._advect_particles(dt)
        self.build_grid()
        self._particle_to_grid()
        cell_type = self._classify_cells()
        self._normalise_and_force(dt)
        self._apply_solid_faces(cell_type)
        self._vorticity_confinement(dt, cell_type)
        self._apply_solid_faces(cell_type)
        self._project(dt, cell_type)
        motion = self._grid_to_particle()
        if flip:
            self._advect_particles(dt, motion)

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
        self.grid_velocity_old.fill(0.0)
        self.grid_vorticity.fill(0.0)
        self.grid_cell.fill(0.0)


def create():
    return ApicCpuEngine()
