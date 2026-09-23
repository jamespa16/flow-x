"""A CPU engine: the same PBF solve, vectorized with numpy.

Three jobs, in order of how much they matter:

* **A fallback.** A machine with no usable Metal device - or a checkout where
  bin/libflowx_metal.dylib was never built - still simulates, just slowly.
* **A reference.** When the GPU engine produces something odd, this answers
  "is the maths wrong, or the kernels?" without a debugger that can step a
  compute shader.
* **A test that runs anywhere.** scripts/test_cpu_engine.py exercises the whole
  PBF pipeline in CI, on a Linux runner, with no Blender and no GPU. Before
  this, CI could only test marching cubes.

Deliberately *not* a kernel-by-kernel port. Each pass here is written the way
the maths is written - one array expression over every particle, or one
scatter-add over every neighbour pair - rather than as a transcription of a
kernel's per-invocation body. A pass-by-pass mirror would be slower, longer,
and no more correct.

The one structural difference from the GPU engine worth knowing: neighbour
pairs are enumerated once per substep, from the same spatial hash, and reused
across the constraint-solve iterations. The kernels re-derive each particle's
base cell every pass, so a particle that drifts across a cell boundary
*during* the constraint loop can pick up a slightly different neighbourhood
there. That is the textbook PBF formulation either way, and it is why results
here match the GPU engine closely without matching it bit for bit - which is
also why the engine's identity goes into the cache's config hash.

No `bpy` import, and none may be added: that is what lets the test run outside
Blender. Same discipline as solver/marching_cubes.py.
"""

import math

import numpy as np

from .params import ParamBlock

# Neighbour cell offsets, in the order the kernels scan them. The order only
# affects floating-point summation order, but keeping it aligned costs nothing.
_OFFSETS = np.array(
    [(dx, dy, dz) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
    dtype=np.int64,
)

# Matches MAX_COLLIDER_SEARCH in kernels/sph_finalize.metal.
MAX_COLLIDER_SEARCH = 2

# Matches SURFACE_CELL_RADIUS's intent in kernels/surface_splat.metal: there the
# gather is bounded in spatial-hash cells, here the scatter is bounded in
# surface lattice points, which is the natural form for a scatter.

# s_corr constants, from kernels/sph_delta.metal.
SCORR_N = 4.0
SCORR_DELTA_Q = 0.2


def _poly6_coef(h):
    return 315.0 / (64.0 * math.pi * h**9)


def _spiky_grad_coef(h):
    return -45.0 / (math.pi * h**6)


def _visc_lap_coef(h):
    return 45.0 / (math.pi * h**6)


def _w_poly6(r2, h2, coef):
    d = np.maximum(h2 - r2, 0.0)
    return coef * d * d * d


def _w_spiky_grad(r, h, coef):
    d = np.maximum(h - r, 0.0)
    return coef * d * d


def _w_visc_lap(r, h, coef):
    return coef * np.maximum(h - r, 0.0)


class _Pairs:
    """Neighbour (i, j) pairs within the smoothing radius, plus their geometry.

    Built once per substep from the spatial hash and carried through every pass
    that needs a neighbour sum. Self-pairs (i == j) are kept, because the
    density sum includes a particle's own kernel contribution; the gradient
    sums mask them out.
    """

    __slots__ = ("i", "j", "d", "r2", "r", "self_mask", "pair_mask")

    def __init__(self, i, j, d, r2):
        self.i = i
        self.j = j
        self.d = d
        self.r2 = r2
        self.r = np.sqrt(r2)
        self.self_mask = i == j
        # Pairs a gradient can be taken over: distinct particles, non-coincident.
        self.pair_mask = (~self.self_mask) & (r2 > 1e-12)


class CpuEngine:
    """Particle state as numpy arrays, and the passes that advance it."""

    name = "cpu"
    method = "pbf"

    def __init__(self):
        self.params = ParamBlock()
        self.config = None
        self.state = {}
        self.surface_field = None
        self.ww = {}
        self.collider = None
        self.collider_dims = (1, 1, 1)
        self._pairs = None

    # --- lifecycle --------------------------------------------------------

    def allocate(self, config, positions):
        self.config = config
        n = config.particle_count
        seed = np.asarray(positions, dtype=np.float32).reshape(-1, 4)[:n]
        self.state = {
            "positions": seed.copy(),
            "velocities": np.zeros((n, 4), dtype=np.float32),
            # (density, lambda, 1/denominator, unused), as the GPU buffer holds.
            "lambda": np.zeros((n, 4), dtype=np.float32),
            "predicted": seed.copy(),
            "delta": np.zeros((n, 4), dtype=np.float32),
            "normal": np.zeros((n, 4), dtype=np.float32),
        }
        self._pairs = None

    def set_collider(self, buffer, dims, voxel_size, occupancy=None):
        """Take the collider occupancy. The device buffer is not usable here."""
        del buffer
        self.collider_dims = tuple(dims)
        if occupancy is None or voxel_size <= 0.0:
            self.collider = None
            voxel_size = 0.0
        else:
            self.collider = np.frombuffer(bytes(occupancy), dtype=np.uint8).reshape(
                dims[2], dims[1], dims[0]
            )
        self.params.update(
            collider_x=dims[0], collider_y=dims[1], collider_z=dims[2], collider_voxel=voxel_size
        )

    def release(self):
        self.state = {}
        self.ww = {}
        self.surface_field = None
        self.config = None
        self._pairs = None

    def flush(self):
        """No-op: every pass here has already run. Kept for one engine protocol."""

    def zero(self, name):
        self.state[name][:] = 0.0

    # --- helpers ----------------------------------------------------------

    def _lo(self):
        P = self.params
        return np.array([P["lo_x"], P["lo_y"], P["lo_z"]], dtype=np.float32)

    def _hi(self):
        P = self.params
        return np.array([P["hi_x"], P["hi_y"], P["hi_z"]], dtype=np.float32)

    def _cell_dims(self):
        P = self.params
        return np.array([P["cells_x"], P["cells_y"], P["cells_z"]], dtype=np.int64)

    def _cell_coords(self, points):
        cells = self._cell_dims()
        g = np.floor((points - self._lo()) / self.params["cell_size"]).astype(np.int64)
        return np.clip(g, 0, cells - 1)

    def _occupied(self, points):
        """Boolean mask: which points sit inside a collider voxel."""
        voxel = self.params["collider_voxel"]
        if self.collider is None or voxel <= 0.0:
            return np.zeros(len(points), dtype=bool)
        c = np.floor((points - self._lo()) / voxel).astype(np.int64)
        dims = self.collider_dims
        inside = np.all((c >= 0) & (c < np.array(dims, dtype=np.int64)), axis=1)
        out = np.zeros(len(points), dtype=bool)
        if not inside.any():
            return out
        ci = c[inside]
        out[inside] = self.collider[ci[:, 2], ci[:, 1], ci[:, 0]] > 0
        return out

    def build_grid(self):
        """Enumerate neighbour pairs from the spatial hash, once per substep.

        The hash itself is the same one the kernels build - particles bucketed
        into cells at least one smoothing radius across, scanned 3x3x3 - but
        the result is materialised as a pair list rather than left as cell
        ranges, because every pass downstream wants a scatter-add over pairs.
        """
        points = self.state["predicted"][:, :3]
        n = len(points)
        cells = self._cell_dims()
        cell_count = int(cells[0] * cells[1] * cells[2])

        g = self._cell_coords(points)
        keys = (g[:, 2] * cells[1] + g[:, 1]) * cells[0] + g[:, 0]
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]

        all_cells = np.arange(cell_count)
        starts = np.searchsorted(sorted_keys, all_cells, side="left")
        ends = np.searchsorted(sorted_keys, all_cells, side="right")

        neighbour = g[:, None, :] + _OFFSETS[None, :, :]
        valid = np.all((neighbour >= 0) & (neighbour < cells), axis=2)
        flat = (neighbour[:, :, 2] * cells[1] + neighbour[:, :, 1]) * cells[0] + neighbour[:, :, 0]
        flat = np.where(valid, flat, 0)

        counts = np.where(valid, ends[flat] - starts[flat], 0).ravel()
        total = int(counts.sum())
        if total == 0:
            self._pairs = _Pairs(
                np.zeros(0, np.int64),
                np.zeros(0, np.int64),
                np.zeros((0, 3), np.float32),
                np.zeros(0, np.float32),
            )
            return

        # Expand each (particle, neighbour cell) run into its members: `slot`
        # names the run, `within` the position inside it.
        slot = np.repeat(np.arange(counts.size), counts)
        run_offsets = np.zeros(counts.size, dtype=np.int64)
        np.cumsum(counts[:-1], out=run_offsets[1:])
        within = np.arange(total) - run_offsets[slot]

        i = slot // _OFFSETS.shape[0]
        j = order[starts[flat.ravel()[slot]] + within]

        d = points[i] - points[j]
        r2 = np.einsum("ij,ij->i", d, d)
        # Trim to the kernel's support immediately: beyond h every weight is
        # zero anyway, and the cell scan overshoots it by a wide margin.
        h2 = self.params["smoothing_radius"] ** 2
        keep = r2 < h2
        self._pairs = _Pairs(i[keep], j[keep], d[keep], r2[keep])
        del n

    # --- passes -----------------------------------------------------------

    def _sum_scalar(self, index, values, n):
        return np.bincount(index, weights=values, minlength=n)

    def _sum_vector(self, index, values, n):
        out = np.empty((n, 3), dtype=np.float64)
        for axis in range(3):
            out[:, axis] = np.bincount(index, weights=values[:, axis], minlength=n)
        return out

    def _normal_pass(self):
        """Color-field gradient and curvature (Morris 2000), as sph_normal."""
        P = self.params
        pairs = self._pairs
        n = self.config.particle_count
        h, h2 = P["smoothing_radius"], P["smoothing_radius"] ** 2
        poly6, visc = _poly6_coef(h), _visc_lap_coef(h)

        mask = ~pairs.self_mask
        i, j, d, r2 = pairs.i[mask], pairs.j[mask], pairs.d[mask], pairs.r2[mask]
        density_j = np.maximum(self.state["lambda"][j, 0], 1e-6)
        weight = P["mass"] / density_j

        t = h2 - r2
        grad = (poly6 * -6.0 * t * t)[:, None] * d
        normal = self._sum_vector(i, weight[:, None] * grad, n)
        laplacian = self._sum_scalar(i, weight * _w_visc_lap(np.sqrt(r2), h, visc), n)

        mag = np.linalg.norm(normal, axis=1)
        curvature = np.where(mag > 1e-4, -laplacian / np.maximum(mag, 1e-30), 0.0)
        self.state["normal"][:, :3] = normal
        self.state["normal"][:, 3] = curvature

    def _predict_pass(self, dt):
        P = self.params
        v = self.state["velocities"][:, :3]
        v[:, 2] += P["gravity"] * dt

        sigma = P["surface_tension"]
        if sigma > 0.0:
            normal = self.state["normal"][:, :3]
            curvature = self.state["normal"][:, 3]
            mag = np.linalg.norm(normal, axis=1)
            active = mag > 1e-4
            if active.any():
                direction = normal[active] / mag[active, None]
                v[active] += (-sigma * curvature[active, None] * direction / P["mass"]) * dt

        v_max = 0.4 * P["smoothing_radius"] / max(dt, 1e-6)
        speed = np.linalg.norm(v, axis=1)
        fast = speed > v_max
        if fast.any():
            v[fast] *= (v_max / speed[fast])[:, None]

        self.state["predicted"][:, :3] = self.state["positions"][:, :3] + v * dt
        self.state["predicted"][:, 3] = 1.0

    def _lambda_pass(self):
        """Density and the constraint multiplier, as sph_lambda."""
        P = self.params
        pairs = self._pairs
        n = self.config.particle_count
        h, h2 = P["smoothing_radius"], P["smoothing_radius"] ** 2
        mass, rest = P["mass"], P["rest_density"]
        poly6, spiky = _poly6_coef(h), _spiky_grad_coef(h)

        points = self.state["predicted"][:, :3]
        d = points[pairs.i] - points[pairs.j]
        r2 = np.einsum("ij,ij->i", d, d)

        density = self._sum_scalar(pairs.i, mass * _w_poly6(r2, h2, poly6), n)

        mask = pairs.pair_mask & (r2 < h2) & (r2 > 1e-12)
        i, dm, r2m = pairs.i[mask], d[mask], r2[mask]
        r = np.sqrt(r2m)
        grad = (mass * _w_spiky_grad(r, h, spiky) / r)[:, None] * dm
        grad_self = self._sum_vector(i, grad, n)
        grad_norm2 = self._sum_scalar(i, np.einsum("ij,ij->i", grad, grad), n)

        density = np.maximum(density, 1e-6)
        constraint = density / rest - 1.0
        denom = (np.einsum("ij,ij->i", grad_self, grad_self) + grad_norm2) / (rest * rest)
        denom += P["relaxation"]
        inv_denom = 1.0 / np.maximum(denom, 1e-9)

        self.state["lambda"][:, 0] = density
        self.state["lambda"][:, 1] = -constraint * inv_denom
        self.state["lambda"][:, 2] = inv_denom

    def _delta_pass(self):
        """Position correction plus the s_corr tensile fix, as sph_delta."""
        P = self.params
        pairs = self._pairs
        n = self.config.particle_count
        h, h2 = P["smoothing_radius"], P["smoothing_radius"] ** 2
        mass, rest = P["mass"], P["rest_density"]
        poly6, spiky = _poly6_coef(h), _spiky_grad_coef(h)
        scorr_k = P["scorr_k"]

        points = self.state["predicted"][:, :3]
        d = points[pairs.i] - points[pairs.j]
        r2 = np.einsum("ij,ij->i", d, d)
        mask = pairs.pair_mask & (r2 < h2) & (r2 > 1e-12)
        i, j, dm, r2m = pairs.i[mask], pairs.j[mask], d[mask], r2[mask]
        r = np.sqrt(r2m)

        lam = self.state["lambda"]
        total = lam[i, 1] + lam[j, 1]

        w_q = _w_poly6(np.float64((SCORR_DELTA_Q * h) ** 2), h2, poly6)
        if scorr_k > 0.0 and w_q > 1e-9:
            ratio = np.maximum(_w_poly6(r2m, h2, poly6) / w_q, 0.0)
            total = total - scorr_k * ratio**SCORR_N * (lam[i, 2] + lam[j, 2])

        contribution = (total * mass * _w_spiky_grad(r, h, spiky) / r)[:, None] * dm
        self.state["delta"][:, :3] = self._sum_vector(i, contribution, n) / rest

    def _xsph_pass(self):
        """XSPH velocity smoothing, as sph_xsph."""
        P = self.params
        pairs = self._pairs
        n = self.config.particle_count
        h, h2 = P["smoothing_radius"], P["smoothing_radius"] ** 2
        poly6 = _poly6_coef(h)

        points = self.state["predicted"][:, :3]
        velocities = self.state["velocities"][:, :3]
        mask = ~pairs.self_mask
        i, j = pairs.i[mask], pairs.j[mask]
        d = points[i] - points[j]
        r2 = np.einsum("ij,ij->i", d, d)
        inside = r2 < h2
        i, j, r2 = i[inside], j[inside], r2[inside]

        density_j = np.maximum(self.state["lambda"][j, 0], 1e-6)
        weight = (P["mass"] / density_j) * _w_poly6(r2, h2, poly6)
        relative = velocities[j] - velocities[i]
        self.state["delta"][:, :3] = P["viscosity"] * self._sum_vector(
            i, weight[:, None] * relative, n
        )

    def _finalize_pass(self):
        """XSPH correction, collider push-out, domain clamp, as sph_finalize."""
        P = self.params
        v = self.state["velocities"][:, :3] + self.state["delta"][:, :3]
        p = self.state["predicted"][:, :3].copy()
        radius = P["particle_radius"]
        damping = P["boundary_damping"]

        voxel = P["collider_voxel"]
        if self.collider is not None and voxel > 0.0:
            stuck = np.where(self._occupied(p))[0]
            for index in stuck:
                nearest, dist2 = self._nearest_free_voxel(p[index], voxel)
                if nearest is None:
                    continue
                dist = math.sqrt(dist2)
                push = (nearest - p[index]) / dist if dist > 1e-6 else np.array([0.0, 0.0, 1.0])
                p[index] += push * (dist + radius)
                vn = float(np.dot(v[index], push))
                if vn < 0.0:
                    v[index] -= vn * (1.0 + damping) * push

        lo = self._lo() + radius
        hi = self._hi() - radius
        for axis in range(3):
            below = p[:, axis] < lo[axis]
            above = (~below) & (p[:, axis] > hi[axis])
            p[below, axis] = lo[axis]
            v[below, axis] *= -damping
            p[above, axis] = hi[axis]
            v[above, axis] *= -damping

        self.state["positions"][:, :3] = p
        self.state["positions"][:, 3] = 1.0
        # Keep the next surface-normal grid derived from the finalized state.
        self.state["predicted"][:, :3] = p
        self.state["predicted"][:, 3] = 1.0
        self.state["velocities"][:, :3] = v

    def _nearest_free_voxel(self, point, voxel):
        """Centre of the closest unoccupied voxel within the search window.

        A per-particle Python loop, but it only runs for particles actually
        found inside a collider, which is a handful per frame at most - the
        constraint solve keeps the fluid out in the first place.
        """
        base = np.floor((point - self._lo()) / voxel).astype(np.int64)
        dims = np.array(self.collider_dims, dtype=np.int64)
        span = np.arange(-MAX_COLLIDER_SEARCH, MAX_COLLIDER_SEARCH + 1)
        grid = np.stack(np.meshgrid(span, span, span, indexing="ij"), axis=-1).reshape(-1, 3)
        candidates = base + grid[:, [2, 1, 0]]
        inside = np.all((candidates >= 0) & (candidates < dims), axis=1)
        candidates = candidates[inside]
        if not len(candidates):
            return None, 0.0
        free = self.collider[candidates[:, 2], candidates[:, 1], candidates[:, 0]] == 0
        candidates = candidates[free]
        if not len(candidates):
            return None, 0.0
        centres = self._lo() + (candidates + 0.5) * voxel
        offsets = centres - point
        dist2 = np.einsum("ij,ij->i", offsets, offsets)
        best = int(np.argmin(dist2))
        return centres[best], float(dist2[best])

    def substep(self, dt):
        """One PBF substep, in the kernels' pass order."""
        P = self.params
        P.update(dt=dt)
        self.build_grid()
        if P["surface_tension"] > 0.0:
            self._normal_pass()
        self._predict_pass(dt)
        # Rebuilt against the predicted positions, which is what the kernels'
        # grid build runs on.
        self.build_grid()
        for _ in range(self.config.iterations):
            self._lambda_pass()
            self._delta_pass()
            self.state["predicted"][:, :3] += self.state["delta"][:, :3]

        self.state["velocities"][:, :3] = (
            self.state["predicted"][:, :3] - self.state["positions"][:, :3]
        ) / max(dt, 1e-6)
        self._xsph_pass()
        self._finalize_pass()

    # --- surface ----------------------------------------------------------

    def alloc_surface(self, sample_count):
        self.surface_field = np.zeros(sample_count, dtype=np.float32)

    def splat_surface(self, surface):
        """Scatter particle mass onto the surface lattice.

        The kernel gathers - one thread per lattice point, walking the spatial
        hash - because a scatter needs an atomic add, which Blender's Metal
        backend could not compile. numpy has np.add.at, so this scatters, which
        touches only the lattice points a particle can actually reach instead
        of visiting all of them.
        """
        P = self.params
        field = self.surface_field
        field[:] = 0.0
        dims = surface.dims
        spacing = surface.spacing
        origin = np.array([surface.lo.x, surface.lo.y, surface.lo.z], dtype=np.float32)
        h = surface.kernel_radius
        h2 = h * h
        coef = _poly6_coef(h)

        points = self.state["positions"][:, :3]
        base = np.floor((points - origin) / spacing).astype(np.int64)
        reach = int(math.ceil(h / spacing))
        span = np.arange(-reach, reach + 2)
        stencil = np.stack(np.meshgrid(span, span, span, indexing="ij"), axis=-1).reshape(-1, 3)

        # One chunk of particles at a time: the full outer product of particles
        # and stencil points is large enough to matter at high resolutions.
        chunk = max(1, 2_000_000 // max(len(stencil), 1))
        for begin in range(0, len(points), chunk):
            block = slice(begin, begin + chunk)
            lattice = base[block][:, None, :] + stencil[None, :, :]
            valid = np.all((lattice >= 0) & (lattice < np.array(dims, dtype=np.int64)), axis=2)
            centres = origin + lattice * spacing
            offsets = centres - points[block][:, None, :]
            r2 = np.einsum("ijk,ijk->ij", offsets, offsets)
            valid &= r2 < h2
            if not valid.any():
                continue
            flat = (lattice[:, :, 2] * dims[1] + lattice[:, :, 1]) * dims[0] + lattice[:, :, 0]
            np.add.at(field, flat[valid], P["mass"] * _w_poly6(r2[valid], h2, coef))

        field /= max(P["rest_density"], 1e-6)

        voxel = P["collider_voxel"]
        if self.collider is not None and voxel > 0.0:
            # Carve the colliders back out: the kernel has a radius of support,
            # so the field otherwise bleeds into a collider and the extracted
            # surface buries its geometry instead of pooling against it.
            index = np.arange(surface.sample_count)
            gx = index % dims[0]
            gy = (index // dims[0]) % dims[1]
            gz = index // (dims[0] * dims[1])
            centres = origin + np.stack([gx, gy, gz], axis=1) * spacing
            field[self._occupied(centres)] = 0.0

    def read_surface(self, sample_count):
        return self.surface_field[:sample_count].tolist()

    # --- whitewater ---------------------------------------------------------

    def alloc_whitewater(self, capacity, sorted_count):
        del sorted_count
        self.params.update(ww_capacity=capacity)
        self.ww = {
            "positions": np.zeros((capacity, 4), dtype=np.float32),
            "velkind": np.zeros((capacity, 4), dtype=np.float32),
        }

    def step_whitewater(self, ww, cursor, spawn_count, frame, frame_dt):
        """Score, rank, spawn and advect, as the four whitewater kernels do."""
        self._whitewater_spawn(ww, cursor, spawn_count, frame)
        self._whitewater_advect(ww, frame_dt)

    def _whitewater_scores(self, ww):
        """Trapped-air / wave-crest / kinetic potentials per fluid particle."""
        P = self.params
        pairs = self._pairs
        n = self.config.particle_count
        h, h2 = P["smoothing_radius"], P["smoothing_radius"] ** 2
        poly6 = _poly6_coef(h)

        points = self.state["positions"][:, :3]
        velocities = self.state["velocities"][:, :3]
        mask = ~pairs.self_mask
        i, j = pairs.i[mask], pairs.j[mask]
        d = points[i] - points[j]
        r2 = np.einsum("ij,ij->i", d, d)
        inside = r2 < h2
        i, j, d, r2 = i[inside], j[inside], d[inside], r2[inside]

        t = h2 - r2
        gradient = self._sum_vector(i, (poly6 * -6.0 * t * t)[:, None] * d, n)

        relative = velocities[i] - velocities[j]
        rel_len = np.linalg.norm(relative, axis=1)
        r = np.sqrt(np.maximum(r2, 1e-30))
        usable = (rel_len > 1e-5) & (r2 > 1e-10)
        closing = np.zeros(len(i))
        closing[usable] = (
            np.maximum(
                0.0,
                -np.einsum(
                    "ij,ij->i",
                    relative[usable] / rel_len[usable, None],
                    d[usable] / r[usable, None],
                ),
            )
            * rel_len[usable]
        )
        divergence = self._sum_scalar(i, closing, n)
        neighbours = self._sum_scalar(i, np.ones(len(i)), n)

        grad_mag = np.linalg.norm(gradient, axis=1)
        outward = np.zeros_like(gradient)
        near_surface = grad_mag > 1e-4
        outward[near_surface] = gradient[near_surface] / grad_mag[near_surface, None]

        reference = max(ww.kinetic_reference_speed, 1e-3)
        speed = np.linalg.norm(velocities, axis=1)
        wave_crest = (
            grad_mag * np.maximum(0.0, -np.einsum("ij,ij->i", velocities, outward)) / reference
        )
        trapped_air = divergence / np.maximum(neighbours, 1.0)
        kinetic = np.clip(speed / reference, 0.0, 1.0)

        score = (
            ww.trapped_air_weight * trapped_air
            + ww.wave_crest_weight * wave_crest
            + ww.kinetic_weight * kinetic
        )
        return score, trapped_air, speed

    def _whitewater_spawn(self, ww, cursor, spawn_count, frame):
        if spawn_count <= 0:
            return
        score, trapped_air, speed = self._whitewater_scores(ww)
        # Descending by score, which the bitonic sort in the kernels produces.
        ranked = np.argsort(-score, kind="stable")[:spawn_count]
        ranked = ranked[score[ranked] > 0.0]
        if not len(ranked):
            return

        seeds = float(frame) + np.arange(len(ranked)) * 0.6180339887
        jitter = np.stack(
            [
                np.modf(np.sin(seeds * m) * 43758.5453)[0] * 2.0 - 1.0
                for m in (12.9898, 78.233, 37.719)
            ],
            axis=1,
        )

        positions = self.state["positions"][ranked, :3] + jitter * ww.normal_offset
        velocities = self.state["velocities"][ranked, :3] + jitter * ww.jitter_strength

        kind = np.where(
            trapped_air[ranked] > ww.bubble_trapped_threshold,
            2,
            np.where(speed[ranked] > ww.spray_speed_threshold, 0, 1),
        )
        life = np.select(
            [kind == 0, kind == 1],
            [
                ww.spray_life[0]
                + (ww.spray_life[1] - ww.spray_life[0]) * np.modf(seeds * 1.618034)[0],
                ww.foam_life[0]
                + (ww.foam_life[1] - ww.foam_life[0]) * np.modf(seeds * 2.718282)[0],
            ],
            ww.bubble_life[0]
            + (ww.bubble_life[1] - ww.bubble_life[0]) * np.modf(seeds * 3.141593)[0],
        )
        # Don't spawn inside geometry; the slot just stays retired.
        life = np.where(self._occupied(positions), 0.0, life)

        slots = (cursor + np.arange(len(ranked))) % ww.capacity
        self.ww["positions"][slots, :3] = positions
        self.ww["positions"][slots, 3] = life
        self.ww["velkind"][slots, :3] = velocities
        self.ww["velkind"][slots, 3] = kind

    def _whitewater_advect(self, ww, dt):
        pool = self.ww["positions"]
        velkind = self.ww["velkind"]
        alive = pool[:, 3] > 0.0
        if not alive.any():
            return

        p = pool[alive, :3]
        v = velkind[alive, :3]
        kind = velkind[alive, 3].astype(np.int64)
        life = pool[alive, 3]
        gravity = self.params["gravity"]
        damping = np.clip(1.0 - ww.drag * dt, 0.0, 1.0)

        spray, bubble, foam = kind == 0, kind == 2, kind == 1
        v[spray, 2] += gravity * dt
        v[bubble, 2] += gravity * (1.0 - ww.buoyancy) * dt
        v[bubble, :2] *= damping
        v[foam] *= damping
        v[foam, 2] += gravity * 0.1 * dt

        p = p + v * dt
        life = life - dt

        # Leaving the domain retires the particle outright rather than clamping
        # it to a wall it was never simulated against - unlike the fluid,
        # whitewater has no obligation to stay inside the box.
        margin = 0.2
        outside = np.any((p < self._lo() - margin) | (p > self._hi() + margin), axis=1)
        life = np.where(outside, 0.0, life)

        blocked = self._occupied(p)
        if blocked.any():
            p[blocked] -= v[blocked] * dt
            v[blocked] *= 0.2

        pool[alive, :3] = p
        pool[alive, 3] = life
        velkind[alive, :3] = v

    def read_whitewater(self, capacity):
        return (
            [tuple(row) for row in self.ww["positions"][:capacity].tolist()],
            [tuple(row) for row in self.ww["velkind"][:capacity].tolist()],
        )

    # --- read-back --------------------------------------------------------

    def read_vec4(self, name, count):
        return [tuple(row) for row in self.state[name][:count].tolist()]

    def read_floats(self, name, count):
        return self.state[name][:count].tolist()

    def upload_vec4(self, name, values, count):
        data = np.asarray(values, dtype=np.float32).reshape(-1, 4)[:count]
        self.state[name][: len(data)] = data

    def snapshot_state(self, include_whitewater=False):
        """Return the persistent state needed to continue this PBF run."""
        count = self.config.particle_count
        state = {
            "positions": self.read_vec4("positions", count),
            "velocities": self.read_vec4("velocities", count),
        }
        if self.method == "pbf":
            # Surface tension runs before the next lambda pass and consumes
            # this prior density channel through sph_normal.
            state["densities"] = self.state["lambda"][:count, 0].tolist()
        if include_whitewater and self.ww:
            capacity = len(self.ww["positions"])
            state["ww_positions"], state["ww_velkind"] = self.read_whitewater(capacity)
        return state

    def restore_state(self, state):
        """Restore persistent state; derived PBF scratch is rebuilt by sph.py."""
        count = self.config.particle_count
        self.upload_vec4("positions", state["positions"], count)
        self.upload_vec4("velocities", state["velocities"], count)
        self.upload_vec4("predicted", state["positions"], count)
        self.zero("lambda")
        if "densities" in state:
            self.state["lambda"][:count, 0] = np.asarray(state["densities"], dtype=np.float32)[
                :count
            ]
        if "ww_positions" in state and self.ww:
            capacity = len(self.ww["positions"])
            self.ww["positions"][:] = np.asarray(state["ww_positions"], dtype=np.float32).reshape(
                -1, 4
            )[:capacity]
            self.ww["velkind"][:] = np.asarray(state["ww_velkind"], dtype=np.float32).reshape(
                -1, 4
            )[:capacity]


def create():
    return CpuEngine()
