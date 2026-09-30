"""Tests for the numpy CPU engine, runnable without Blender.

    python3 scripts/test_cpu_engine.py

The CPU engine is the fallback for machines with no usable device, and the
reference the GPU engine is checked against - so it needs a test that runs
where neither Blender nor a GPU is available. solver/engine/cpu_engine.py
imports only numpy, and solver/engine/params.py only stdlib, so both are loaded
here by path to avoid solver/__init__.py, which imports bpy.

Two kinds of check:

* **Invariants.** A dam break has properties that hold whatever the numbers do
  in detail: nothing escapes the box, nothing becomes NaN, the fluid settles
  towards rest density, colliders are not penetrated. These run everywhere.
* **Agreement with the GPU engine.** Where a Metal device exists, both engines
  run the same configuration and their results are compared as aggregates.
  They are not expected to match bit for bit - the CPU engine enumerates
  neighbours once per substep where the kernels re-derive them per pass - but
  they must describe the same fluid. This is the check that would catch a
  kernel and its numpy counterpart drifting apart.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flowx_standalone import load  # noqa: E402

cpu_engine = load("solver.engine.cpu_engine")
apic_cpu = load("solver.engine.apic_cpu")
params_mod = load("solver.engine.params")
marching_cubes = load("solver.marching_cubes")


class Failure(AssertionError):
    pass


def check(condition, message):
    if not condition:
        raise Failure(message)


class Vec:
    """Minimal stand-in for mathutils.Vector - the engines only read .x/.y/.z."""

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class Config:
    """Stand-in for solver.sph.SolverConfig, with only the fields engines read."""

    def __init__(self, particle_count, cell_dims, cell_count, iterations=3, surface_tension=0.0):
        self.particle_count = particle_count
        self.sorted_count = max(2, 1 << (max(1, particle_count) - 1).bit_length())
        self.cell_dims = cell_dims
        self.cell_count = cell_count
        self.iterations = iterations
        self.surface_tension = surface_tension
        self.pressure_iterations = 40
        self.pressure_solver = "pcg"
        self.vorticity_strength = 0.0


class SurfaceCfg:
    def __init__(self, dims, spacing, lo, kernel_radius):
        self.dims = dims
        self.spacing = spacing
        self.lo = lo
        self.kernel_radius = kernel_radius
        self.sample_count = dims[0] * dims[1] * dims[2]


class WhitewaterCfg:
    capacity = 512
    trapped_air_weight = 1.0
    wave_crest_weight = 1.0
    kinetic_weight = 1.0
    kinetic_reference_speed = 2.0
    spray_speed_threshold = 1.0
    bubble_trapped_threshold = 0.5
    jitter_strength = 0.1
    normal_offset = 0.01
    spray_life = (0.5, 1.5)
    foam_life = (1.0, 3.0)
    bubble_life = (0.5, 2.0)
    drag = 1.0
    buoyancy = 1.0


# A small dam break: a block of fluid in the corner of a 1m box, released.
DOMAIN_LO = (0.0, 0.0, 0.0)
DOMAIN_HI = (1.0, 1.0, 1.0)
SPACING = 0.05
REST_DENSITY = 1000.0


def _scene():
    """(config, params, seed positions) for the dam break, shared by the tests."""
    smoothing = SPACING * 2.0
    cell_size = smoothing
    dims = tuple(max(1, int(math.ceil(1.0 / cell_size))) for _ in range(3))
    cell_count = dims[0] * dims[1] * dims[2]

    coords = np.arange(SPACING, 0.5, SPACING)
    zs = np.arange(SPACING, 0.6, SPACING)
    grid = np.stack(np.meshgrid(coords, coords, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    seed = np.zeros((len(grid), 4), dtype=np.float32)
    seed[:, :3] = grid
    seed[:, 3] = 1.0

    config = Config(len(grid), dims, cell_count)
    block = params_mod.ParamBlock(
        particle_count=config.particle_count,
        sorted_count=config.sorted_count,
        cell_count=cell_count,
        cells_x=dims[0],
        cells_y=dims[1],
        cells_z=dims[2],
        lo_x=DOMAIN_LO[0],
        lo_y=DOMAIN_LO[1],
        lo_z=DOMAIN_LO[2],
        cell_size=cell_size,
        hi_x=DOMAIN_HI[0],
        hi_y=DOMAIN_HI[1],
        hi_z=DOMAIN_HI[2],
        particle_radius=SPACING * 0.5,
        smoothing_radius=smoothing,
        mass=REST_DENSITY * SPACING**3,
        rest_density=REST_DENSITY,
        relaxation=1e-4,
        viscosity=0.05,
        gravity=-9.81,
        boundary_damping=0.35,
        scorr_k=0.1,
        nodes_x=dims[0] + 1,
        nodes_y=dims[1] + 1,
        nodes_z=dims[2] + 1,
        grid_spacing=cell_size,
        vorticity_epsilon=0.0,
        grid_max_speed=0.25 * cell_size / (1.0 / 48.0),
        pressure_tolerance=1e-3,
    )
    return config, block, seed.ravel()


def _run(engine, frames=8, substeps=2, dt=1.0 / 48.0):
    for _ in range(frames):
        for _ in range(substeps):
            engine.substep(dt)


def _positions(engine):
    return np.asarray(engine.read_vec4("positions", engine.config.particle_count))[:, :3]


def _make_cpu(surface_tension=0.0):
    config, block, seed = _scene()
    config.surface_tension = surface_tension
    block.update(surface_tension=surface_tension)
    engine = cpu_engine.create()
    engine.params = block
    engine.allocate(config, seed)
    return engine


def _make_apic():
    config, block, seed = _scene()
    engine = apic_cpu.create()
    engine.params = block
    engine.allocate(config, seed)
    return engine


def test_apic_stays_finite_and_bounded():
    engine = _make_apic()
    _run(engine, frames=8)
    p = _positions(engine)
    v = np.asarray(engine.read_vec4("velocities", engine.config.particle_count))[:, :3]
    radius = engine.params["particle_radius"]
    check(np.isfinite(p).all() and np.isfinite(v).all(), "APIC produced NaN or inf")
    check((p >= radius - 1e-5).all(), "APIC particles escaped the low domain bound")
    check((p <= 1.0 - radius + 1e-5).all(), "APIC particles escaped the high domain bound")
    check(p[:, 2].mean() < 0.30, "APIC fluid did not fall")


def _make_apic_solver(solver, iterations=40, tolerance=1e-3):
    engine = _make_apic()
    engine.config.pressure_solver = solver
    engine.config.pressure_iterations = iterations
    engine.params.update(pressure_tolerance=tolerance)
    return engine


def _projection_residual(solver, iterations=40, tolerance=1e-3):
    """(rms, max) post/pre divergence ratio over fluid cells after 4 substeps."""
    engine = _make_apic_solver(solver, iterations, tolerance)
    # The first free-surface projection begins from an empty grid and is the
    # hardest. A few normal substeps are a more useful steady-state check.
    for _ in range(4):
        engine.substep(1.0 / 48.0)
    check(engine.last_divergence_before > 0.0, "APIC test field had no divergence to project")
    rms = engine.last_divergence_after / engine.last_divergence_before
    peak = engine.last_divergence_max_after / engine.last_divergence_max_before
    return rms, peak, engine.last_pressure_iterations


def test_apic_projection_residual():
    """The residual-divergence metric, for Jacobi and PCG on the same run.

    Recorded baseline (this dam break, fourth substep, 40 iterations):
    Jacobi leaves 19% RMS / 17% max of the pre-projection divergence (it was
    3.0% / 1.2% before G2P's affine rows stopped blowing up on face planes;
    the projection now sees a real velocity field instead of a noisy one). PCG
    reaches the 1e-3 tolerance in about 20 iterations, and with the tolerance
    at zero runs until its breakdown guard stops it at the float64 floor,
    around 1e-6. The bounds below are loose on both sides of those numbers so
    that a change in either direction is noticed rather than absorbed.

    Equal wall time favours PCG further: on the CPU engine one PCG iteration
    over the compact fluid vector costs less than one full-grid Jacobi sweep
    (about 0.02 ms against 0.09 ms here), so the comparison at equal iteration
    count is the conservative one.
    """
    jacobi_rms, jacobi_max, _ = _projection_residual("jacobi")
    print(f"  Jacobi x40 residual: rms {jacobi_rms:.2e}, max {jacobi_max:.2e}")
    check(0.05 < jacobi_rms < 0.40, f"Jacobi residual {jacobi_rms:.2e} left its baseline")
    check(0.05 < jacobi_max < 0.40, f"Jacobi max residual {jacobi_max:.2e} left its baseline")

    pcg_rms, pcg_max, ran = _projection_residual("pcg")
    print(f"  PCG <=40 (tol 1e-3) residual: rms {pcg_rms:.2e}, max {pcg_max:.2e}, {ran} its")
    check(pcg_rms <= 1e-3, f"PCG stopped at rms {pcg_rms:.2e}, above its 1e-3 tolerance")
    check(ran < 40, f"PCG needed all {ran} iterations to reach 1e-3")

    full_rms, full_max, ran = _projection_residual("pcg", tolerance=0.0)
    print(f"  PCG x40 (tol 0) residual: rms {full_rms:.2e}, max {full_max:.2e}, {ran} its")
    # The Phase 2 exit margin: two orders of magnitude below Jacobi at the
    # same iteration cap, on both norms. Measured margin is about 10^4.
    check(full_rms < jacobi_rms * 1e-2, "PCG is not clearly better than Jacobi (rms)")
    check(full_max < jacobi_max * 1e-2, "PCG is not clearly better than Jacobi (max)")


def _make_pool(solver, fill=0.4):
    """A still pool over the whole floor: nothing should move, nothing be lost."""
    config, block, _ = _scene()
    xs = np.arange(SPACING * 0.5, 1.0, SPACING)
    zs = np.arange(SPACING * 0.5, fill, SPACING)
    grid = np.stack(np.meshgrid(xs, xs, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    seed = np.zeros((len(grid), 4), dtype=np.float32)
    seed[:, :3] = grid
    seed[:, 3] = 1.0
    config = Config(len(grid), config.cell_dims, config.cell_count)
    config.pressure_solver = solver
    block.update(particle_count=config.particle_count, sorted_count=config.sorted_count)
    engine = apic_cpu.create()
    engine.params = block
    engine.allocate(config, seed.ravel())
    return engine


def _pool_volume_drift(solver, frames=24):
    """Fractional fluid-cell loss of a settled pool over `frames` frames.

    Particle count is fixed, so the cells the solver sees as fluid are its own
    measure of volume. A weak solve lets the pool compact slowly under gravity,
    which is the visible symptom: the surface sinks and the top layer of cells
    empties.
    """
    engine = _make_pool(solver)
    start = None
    for _ in range(frames):
        for _ in range(2):
            engine.substep(1.0 / 48.0)
        count = int((engine.grid_cell[..., 3] > 0.0).sum())
        start = count if start is None else start
    return (start - count) / start


def test_apic_pool_volume_drift():
    """Baseline: Jacobi x40 loses 27% of the pool's fluid cells in one second.

    PCG at the default tolerance loses none. The Jacobi bounds bracket the
    recorded 27% (14% before the affine-row fix) so a regression - or an
    accidental improvement to the path kept for comparison - shows up.
    """
    jacobi = _pool_volume_drift("jacobi")
    pcg = _pool_volume_drift("pcg")
    print(f"  pool volume drift over 24 frames: Jacobi {jacobi:.1%}, PCG {pcg:.1%}")
    check(0.10 < jacobi < 0.45, f"Jacobi pool drift {jacobi:.1%} left its baseline")
    check(pcg <= 0.01, f"PCG pool lost {pcg:.1%} of its volume")


def test_apic_affine_rows_stay_bounded_at_rest():
    """A pool at rest must stay at rest: its affine rows used to blow up.

    Seeded on a lattice, every particle sits on face planes, where the old
    B.D^-1 reconstruction was a 0/0 cancellation. The float64 engine kept those
    inverses: rows reached ~1e4 1/s and the pool reached ~1.4 m/s within two
    seconds (sane rows are ~10). The gradient form is finite on the plane.
    """
    engine = _make_pool("pcg")
    for _ in range(96):
        engine.substep(1.0 / 96.0)
    rows = np.abs(engine.state["affine"][:, :, :3]).max()
    speed = np.abs(engine.state["velocities"][:, :3]).max()
    print(f"  resting pool after 1 s: max |C| {rows:.2e} 1/s, max speed {speed:.2e} m/s")
    check(np.isfinite(rows) and rows < 10.0, f"affine rows blew up: max |C| {rows:.1f}")
    check(speed < 0.05, f"resting pool accelerated to {speed:.3f} m/s")


def _pressure_diagonal(cell_type):
    """Per-cell count of non-solid neighbours: the pressure matrix's diagonal."""
    padded = np.pad(cell_type, 1, constant_values=-1.0)
    diagonal = np.zeros(cell_type.shape)
    for axis in range(3):
        for shift in (-1, 1):
            diagonal += np.roll(padded, shift, axis=axis)[1:-1, 1:-1, 1:-1] >= 0.0
    return np.where(cell_type > 0.0, diagonal, 0.0)


def _sealed_projection(cell_type, seed=7):
    """Project a random face field through `cell_type` with PCG; return the engine."""
    engine = _make_transfer_apic()
    engine.config.pressure_solver = "pcg"
    engine.config.pressure_iterations = 200
    engine.params.update(pressure_tolerance=1e-5)
    rng = np.random.default_rng(seed)
    engine.grid_velocity[..., :3] = rng.uniform(-1.0, 1.0, engine.grid_velocity[..., :3].shape)
    engine.grid_cell[..., 3] = cell_type
    engine._apply_solid_faces(cell_type)
    engine._project(1.0 / 48.0, cell_type)
    return engine


def test_apic_pcg_sealed_regions():
    """A fluid region with no air cell makes the pressure matrix singular.

    Its null vector is a constant pressure over the region, and the right-hand
    side is consistent with it only up to roundoff (the divergence of a field
    with every boundary face zeroed sums to zero exactly, in exact arithmetic).
    Starting from zero keeps the Krylov iterates in the preconditioned range,
    and the breakdown guard stops the solve instead of dividing by a vanishing
    p.Ap once the range part is exhausted. Both cases here - the whole domain
    full, and a sealed tank beside an open pool - must converge, stay finite,
    and not pick up a drifting constant.

    "No constant" is measured with the diagonal as weight: with the diagonal
    preconditioner every iterate lies in D^-1 range(A), and range(A) is the
    zero-sum vectors, so sum(D p) over a sealed region stays zero while the
    plain mean need not.
    """
    dims = (8, 8, 8)
    full = np.ones(dims, dtype=np.float32)
    engine = _sealed_projection(full)
    pressure = engine.grid_cell[..., 1]
    weights = _pressure_diagonal(full)
    offset = abs((weights * pressure).sum()) / (weights * np.abs(pressure)).sum()
    ratio = engine.last_divergence_after / engine.last_divergence_before
    print(
        f"  sealed box: residual {ratio:.2e} in {engine.last_pressure_iterations} its, "
        f"weighted mean p {offset:.2e} of mean |p|"
    )
    check(np.isfinite(pressure).all(), "sealed box pressure is not finite")
    check(ratio < 1e-3, f"sealed box kept {ratio:.2e} of its divergence")
    check(offset < 1e-4, "sealed box pressure picked up a constant offset")

    # A tank: a solid shell around a fluid-filled interior, with an open pool
    # (air above it) in the rest of the domain.
    tank = np.zeros(dims, dtype=np.float32)
    tank[:4] = 1.0
    tank[0:6, 0:6, 0:6] = -1.0
    tank[1:5, 1:5, 1:5] = 1.0
    engine = _sealed_projection(tank, seed=11)
    ratio = engine.last_divergence_after / engine.last_divergence_before
    pressure = engine.grid_cell[..., 1]
    weights = _pressure_diagonal(tank)
    inside = weights[1:5, 1:5, 1:5] * pressure[1:5, 1:5, 1:5]
    offset = abs(inside.sum()) / np.abs(inside).sum()
    print(
        f"  sealed tank + open pool: residual {ratio:.2e} in "
        f"{engine.last_pressure_iterations} its, tank weighted mean p {offset:.2e}"
    )
    check(np.isfinite(pressure).all(), "sealed tank pressure is not finite")
    check(ratio < 1e-3, f"sealed tank kept {ratio:.2e} of its divergence")
    check(offset < 1e-4, "sealed tank pressure picked up a constant offset")


def test_apic_pcg_is_deterministic():
    """Two identical runs agree bit for bit: cached scrubbing depends on it."""
    runs = []
    for _ in range(2):
        engine = _make_apic_solver("pcg")
        _run(engine, frames=3)
        runs.append(engine.snapshot_state())
    for key in ("positions", "velocities", "affine"):
        check(runs[0][key] == runs[1][key], f"PCG run changed {key} between identical runs")


def _make_transfer_apic(points_per_axis=4):
    """A compact interior block for transfer and rotation invariants."""
    dx = 0.125
    dims = (8, 8, 8)
    coords = np.linspace(0.3, 0.7, points_per_axis)
    points = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1).reshape(-1, 3)
    seed = np.zeros((len(points), 4), dtype=np.float32)
    seed[:, :3] = points
    seed[:, 3] = 1.0
    config = Config(len(points), dims, math.prod(dims))
    block = params_mod.ParamBlock(
        particle_count=len(points),
        sorted_count=config.sorted_count,
        cell_count=config.cell_count,
        cells_x=dims[0],
        cells_y=dims[1],
        cells_z=dims[2],
        nodes_x=dims[0] + 1,
        nodes_y=dims[1] + 1,
        nodes_z=dims[2] + 1,
        lo_x=0.0,
        lo_y=0.0,
        lo_z=0.0,
        hi_x=1.0,
        hi_y=1.0,
        hi_z=1.0,
        cell_size=dx,
        grid_spacing=dx,
        particle_radius=0.01,
        smoothing_radius=dx,
        mass=1.0,
        rest_density=1.0,
        gravity=0.0,
        boundary_damping=0.0,
        grid_max_speed=100.0,
        vorticity_epsilon=0.0,
        pressure_tolerance=1e-3,
    )
    engine = apic_cpu.create()
    engine.params = block
    engine.allocate(config, seed.ravel())
    return engine


def _transfer_roundtrip(engine):
    engine._particle_to_grid()
    engine._normalise_and_force(0.0)
    engine._grid_to_particle()


def test_apic_transfer_preserves_translation_and_rotation():
    engine = _make_transfer_apic()
    points = _positions(engine)
    translation = np.array((0.17, -0.11, 0.08), dtype=np.float32)
    engine.state["velocities"][:, :3] = translation
    _transfer_roundtrip(engine)
    got = engine.state["velocities"][:, :3]
    check(np.max(np.abs(got - translation)) < 1e-5, "uniform APIC translation drifted")

    center = points.mean(axis=0)
    omega = np.array((0.2, -0.3, 0.7), dtype=np.float32)
    matrix = np.array(
        ((0.0, -omega[2], omega[1]), (omega[2], 0.0, -omega[0]), (-omega[1], omega[0], 0.0)),
        dtype=np.float32,
    )
    initial_velocity = (points - center) @ matrix.T
    engine.state["velocities"][:, :3] = initial_velocity
    engine.state["affine"][:, :, :3] = matrix
    before_linear = initial_velocity.sum(axis=0)
    before_angular = np.cross(points - center, initial_velocity).sum(axis=0)
    _transfer_roundtrip(engine)
    after_velocity = engine.state["velocities"][:, :3]
    after_linear = after_velocity.sum(axis=0)
    after_angular = np.cross(points - center, after_velocity).sum(axis=0)
    check(
        np.max(np.abs(after_linear - before_linear)) < 1e-5,
        "affine transfer changed linear momentum",
    )
    check(
        np.max(np.abs(after_angular - before_angular)) < 1e-5,
        "affine transfer changed angular momentum",
    )
    check(
        np.max(np.abs(engine.state["affine"][:, :, :3] - matrix)) < 1e-5,
        "affine rows were not reconstructed",
    )


def test_apic_rotating_block_holds_angular_momentum(blend=0.0):
    engine = _make_transfer_apic(points_per_axis=4)
    engine.params.update(flip_blend=blend)
    points = _positions(engine)
    center = points.mean(axis=0)
    omega = 0.4
    matrix = np.array(((0.0, -omega, 0.0), (omega, 0.0, 0.0), (0.0, 0.0, 0.0)))
    engine.state["velocities"][:, :3] = (points - center) @ matrix.T
    engine.state["affine"][:, :, :3] = matrix

    def momentum():
        p = _positions(engine)
        v = engine.state["velocities"][:, :3]
        com = p.mean(axis=0)
        return np.linalg.norm(np.cross(p - com, v).sum(axis=0))

    before = momentum()
    for _ in range(300):
        engine.substep(1.0 / 240.0)
    after = momentum()
    drift = abs(after - before) / before
    print(f"  rotating-block angular momentum drift {drift:.2%} at FLIP blend {blend}")
    check(drift < 0.05, f"APIC angular momentum drifted {drift:.2%} over 300 steps")


def test_apic_confinement_clamps_each_face_component():
    """Confinement is an acceleration, but it must retain the grid CFL cap."""
    engine = _make_transfer_apic()
    cap = 0.1
    engine.params.update(vorticity_epsilon=100.0, grid_max_speed=cap)
    rng = np.random.default_rng(42)
    engine.grid_velocity[..., :3] = rng.uniform(-0.09, 0.09, engine.grid_velocity[..., :3].shape)
    cell_type = np.ones(engine.grid_cell.shape[:3], dtype=np.float32)

    engine._vorticity_confinement(0.1, cell_type)

    nx, ny, nz = engine.config.cell_dims
    faces = (
        engine.grid_velocity[:nz, :ny, 1:nx, 0],
        engine.grid_velocity[:nz, 1:ny, :nx, 1],
        engine.grid_velocity[1:nz, :ny, :nx, 2],
    )
    for axis, values in enumerate(faces):
        peak = float(np.abs(values).max())
        check(peak <= cap + 1e-6, f"confinement exceeded the speed cap on axis {axis}: {peak}")
        check(
            np.isclose(np.abs(values), cap, atol=1e-6).any(),
            f"test field did not exercise confinement on axis {axis}",
        )


def test_apic_collider_surface_and_whitewater():
    engine = _make_apic()
    voxel = SPACING
    dims = (20, 20, 20)
    occupancy = np.zeros((20, 20, 20), dtype=np.uint8)
    occupancy[4:7, :, :] = 1
    engine.set_collider(None, dims, voxel, occupancy=occupancy.tobytes())
    _run(engine, frames=10)
    check(not engine._occupied(_positions(engine)).any(), "APIC penetrated the collider")

    margin = engine.params["smoothing_radius"]
    spacing = SPACING
    surface_dims = tuple(int((1.0 + 2 * margin) / spacing) + 1 for _ in range(3))
    surface = SurfaceCfg(surface_dims, spacing, Vec(-margin, -margin, -margin), margin)
    engine.alloc_surface(surface.sample_count)
    engine.splat_surface(surface)
    field = np.asarray(engine.read_surface(surface.sample_count))
    vertices, triangles = marching_cubes.extract(
        field, surface_dims, (-margin, -margin, -margin), spacing, 0.5
    )
    check(len(vertices) > 0 and len(triangles) > 0, "APIC surface extraction was empty")
    vertices = np.asarray(vertices)
    check(np.isfinite(vertices).all(), "APIC surface contained non-finite vertices")
    check(
        (vertices >= -margin - spacing).all() and (vertices <= 1.0 + margin + spacing).all(),
        "APIC surface escaped its extraction bounds",
    )

    ww = WhitewaterCfg()
    engine.alloc_whitewater(ww.capacity, engine.config.sorted_count)
    engine.build_grid()
    cursor = 0
    for frame in range(6):
        engine.step_whitewater(ww, cursor, 20, frame, 1.0 / 24.0)
        cursor = (cursor + 20) % ww.capacity
    pool, _velkind = engine.read_whitewater(ww.capacity)
    check(any(p[3] > 0.0 for p in pool), "APIC did not spawn whitewater")
    for _ in range(200):
        engine.step_whitewater(ww, cursor, 0, 99, 1.0 / 24.0)
    pool, _velkind = engine.read_whitewater(ww.capacity)
    check(all(p[3] <= 0.0 for p in pool), "APIC whitewater did not expire")


def test_apic_preserves_moving_solid_face_velocity():
    engine = _make_apic()
    voxel = SPACING
    dims = (int(1.0 / voxel),) * 3
    occupancy = np.zeros((dims[2], dims[1], dims[0]), dtype=np.uint8)
    velocity = np.zeros((dims[2], dims[1], dims[0], 3), dtype=np.float32)
    occupancy[:, :, 10:12] = 1
    velocity[:, :, 10:12, 0] = 1.0
    engine.set_collider(
        None,
        dims,
        voxel,
        occupancy=occupancy.tobytes(),
        velocities=velocity.tobytes(),
    )
    engine.build_grid()
    cell_type = engine._classify_cells()
    engine.grid_velocity.fill(0.0)
    engine._apply_solid_faces(cell_type)
    nx, ny, nz = engine.config.cell_dims
    interface = engine.grid_velocity[:nz, :ny, 5, 0]
    check(np.allclose(interface, 1.0), "APIC did not impose moving-wall face velocity")
    engine._project(1.0 / 48.0, cell_type)
    interface = engine.grid_velocity[:nz, :ny, 5, 0]
    check(np.allclose(interface, 1.0), "APIC projection erased moving-wall velocity")


def test_apic_snapshot_continuation_is_exact():
    """Restoring all persistent state must reproduce an uninterrupted step."""
    ww = WhitewaterCfg()
    ww.capacity = 64
    original = _make_apic()
    original.alloc_whitewater(ww.capacity, original.config.sorted_count)
    _run(original, frames=2, substeps=1)
    original.build_grid()
    original.step_whitewater(ww, 0, 16, 1, 1.0 / 24.0)
    checkpoint = original.snapshot_state(include_whitewater=True)

    original.substep(1.0 / 48.0)
    original.build_grid()
    original.step_whitewater(ww, 16, 12, 2, 1.0 / 24.0)
    uninterrupted = original.snapshot_state(include_whitewater=True)

    resumed = _make_apic()
    resumed.alloc_whitewater(ww.capacity, resumed.config.sorted_count)
    resumed.restore_state(checkpoint)
    resumed.build_grid()
    resumed.substep(1.0 / 48.0)
    resumed.build_grid()
    resumed.step_whitewater(ww, 16, 12, 2, 1.0 / 24.0)
    continued = resumed.snapshot_state(include_whitewater=True)

    for key in ("positions", "velocities", "affine", "ww_positions", "ww_velkind"):
        check(
            np.array_equal(np.asarray(uninterrupted[key]), np.asarray(continued[key])),
            f"restored APIC continuation changed {key}",
        )


def test_apic_flip_snapshot_continuation_is_exact():
    """FLIP moves advection to the end of the substep and persists nothing new.

    The carried velocity is the stored velocity and the grid field it moves
    by is rebuilt inside the substep, so a restored frame must still continue
    exactly as the uninterrupted run did.
    """
    runs = []
    original = _make_apic()
    original.params.update(flip_blend=0.5)
    _run(original, frames=2, substeps=1)
    checkpoint = original.snapshot_state()
    original.substep(1.0 / 48.0)
    runs.append(original.snapshot_state())

    resumed = _make_apic()
    resumed.params.update(flip_blend=0.5)
    resumed.restore_state(checkpoint)
    resumed.substep(1.0 / 48.0)
    runs.append(resumed.snapshot_state())
    for key in ("positions", "velocities", "affine"):
        check(
            np.array_equal(np.asarray(runs[0][key]), np.asarray(runs[1][key])),
            f"restored FLIP continuation changed {key}",
        )


def test_apic_flip_blend_zero_is_apic():
    """Blend 0 must not read anything FLIP added: it is APIC, bit for bit.

    Poisons the saved pre-force grid velocity with NaN every substep; a blend-0
    run that so much as multiplied it by zero would come out NaN.
    """
    clean = _make_apic()
    poisoned = _make_apic()
    normalise = poisoned._normalise_and_force

    def poisoned_normalise(dt):
        normalise(dt)
        poisoned.grid_velocity_old.fill(np.nan)

    poisoned._normalise_and_force = poisoned_normalise
    _run(clean, frames=3)
    _run(poisoned, frames=3)
    a, b = clean.snapshot_state(), poisoned.snapshot_state()
    for key in ("positions", "velocities", "affine"):
        check(a[key] == b[key], f"blend 0 depends on the FLIP grid copy ({key})")


def _dam_break_summary(blend, frames=24):
    engine = _make_apic()
    engine.params.update(flip_blend=blend)
    _run(engine, frames=frames)
    velocity = engine.state["velocities"][:, :3]
    fluid = int((engine.grid_cell[..., 3] > 0.0).sum())
    return engine, fluid, velocity


def test_apic_flip_dam_break_holds_volume():
    """FLIP at the recommended blends stays finite and keeps the fluid's volume.

    Before advection moved to the grid field, a pure-FLIP dam break lost half
    its fluid cells in a second as particles clumped against the floor; this
    is the regression check for that. The comparison is against APIC's own
    count at the same frame, since a settled dam break legitimately has fewer
    fluid cells than the block it started as.
    """
    _engine, apic_cells, _velocity = _dam_break_summary(0.0)
    for blend in (0.25, 0.5, 1.0):
        _engine, cells, velocity = _dam_break_summary(blend)
        speed = float(np.linalg.norm(velocity, axis=1).max())
        print(f"  blend {blend:.2f}: {cells} fluid cells (APIC {apic_cells}), peak {speed:.2f} m/s")
        check(np.isfinite(velocity).all(), f"FLIP blend {blend} produced NaN or inf")
        check(cells >= 0.85 * apic_cells, f"FLIP blend {blend} lost volume: {cells} cells")
        check(speed < 10.0, f"FLIP blend {blend} blew up to {speed:.1f} m/s")


def test_pbf_snapshot_continuation_is_exact():
    """PBF continuation preserves prior density for surface tension."""
    ww = WhitewaterCfg()
    ww.capacity = 64
    original = _make_cpu(surface_tension=0.05)
    original.alloc_whitewater(ww.capacity, original.config.sorted_count)
    _run(original, frames=2, substeps=1)
    # Force the next finalization through the x-domain clamp. Before predicted
    # was synchronized there, an uninterrupted surface-tension step read the
    # stale pre-clamp position while a restored run read the finalized one.
    radius = original.params["particle_radius"]
    original.state["predicted"][0, 0] = -1.0
    original._finalize_pass()
    check(
        np.isclose(original.state["positions"][0, 0], radius),
        "PBF continuation setup did not exercise the domain clamp",
    )
    check(
        np.array_equal(original.state["positions"], original.state["predicted"]),
        "PBF finalization did not synchronize predicted positions",
    )
    original.build_grid()
    original.step_whitewater(ww, 0, 16, 1, 1.0 / 24.0)
    checkpoint = original.snapshot_state(include_whitewater=True)

    original.substep(1.0 / 48.0)
    original.build_grid()
    original.step_whitewater(ww, 16, 12, 2, 1.0 / 24.0)
    uninterrupted = original.snapshot_state(include_whitewater=True)

    resumed = _make_cpu(surface_tension=0.05)
    resumed.alloc_whitewater(ww.capacity, resumed.config.sorted_count)
    resumed.restore_state(checkpoint)
    resumed.substep(1.0 / 48.0)
    resumed.build_grid()
    resumed.step_whitewater(ww, 16, 12, 2, 1.0 / 24.0)
    continued = resumed.snapshot_state(include_whitewater=True)

    for key in ("positions", "velocities", "densities", "ww_positions", "ww_velkind"):
        check(
            np.array_equal(np.asarray(uninterrupted[key]), np.asarray(continued[key])),
            f"restored PBF continuation changed {key}",
        )


def test_dam_break_stays_finite():
    engine = _make_cpu()
    print(f"  {engine.config.particle_count} particles, {engine.config.cell_count} cells")
    _run(engine)
    p = _positions(engine)
    check(np.isfinite(p).all(), "positions contain NaN or inf")
    v = np.asarray(engine.read_vec4("velocities", engine.config.particle_count))[:, :3]
    check(np.isfinite(v).all(), "velocities contain NaN or inf")
    check(np.abs(v).max() < 100.0, f"velocities exploded to {np.abs(v).max():.1f} m/s")


def test_fluid_stays_in_the_box():
    engine = _make_cpu()
    _run(engine)
    p = _positions(engine)
    radius = engine.params["particle_radius"]
    lo = np.array(DOMAIN_LO) + radius
    hi = np.array(DOMAIN_HI) - radius
    # A tolerance of one float epsilon's worth of slack: the clamp is exact.
    check(
        (p >= lo - 1e-5).all() and (p <= hi + 1e-5).all(),
        f"fluid escaped the domain: {p.min(axis=0)} .. {p.max(axis=0)}",
    )


def test_it_actually_falls_and_spreads():
    """Guards against a solver that runs, stays finite, and does nothing."""
    engine = _make_cpu()
    before = _positions(engine).copy()
    _run(engine, frames=12)
    after = _positions(engine)
    check(after[:, 2].mean() < before[:, 2].mean() - 0.01, "fluid did not fall under gravity")
    spread_before = before[:, 0].max() - before[:, 0].min()
    spread_after = after[:, 0].max() - after[:, 0].min()
    check(spread_after > spread_before + 0.05, "the dam break did not spread")


def test_density_approaches_rest():
    """The whole point of PBF: the constraint solve should hold density near rest."""
    engine = _make_cpu()
    _run(engine, frames=12)
    density = np.asarray(engine.read_vec4("lambda", engine.config.particle_count))[:, 0]
    # Interior particles only: a surface particle has half a neighbourhood and
    # is *supposed* to read low, so a mean over everything measures the
    # surface-to-volume ratio rather than the constraint solve.
    interior = density > 0.5 * REST_DENSITY
    mean = density[interior].mean()
    error = abs(mean - REST_DENSITY) / REST_DENSITY
    print(f"  interior mean density {mean:.1f} vs rest {REST_DENSITY:.0f} ({error * 100:.1f}%)")
    check(error < 0.15, f"interior density is {mean:.1f}, {error * 100:.1f}% off rest")


def test_collider_is_not_penetrated():
    engine = _make_cpu()
    # A slab across the lower half of the box, on the collider grid.
    voxel = SPACING
    dims = (int(1.0 / voxel), int(1.0 / voxel), int(1.0 / voxel))
    occupancy = np.zeros((dims[2], dims[1], dims[0]), dtype=np.uint8)
    occupancy[4:7, :, :] = 1
    engine.set_collider(None, dims, voxel, occupancy=occupancy.tobytes())
    _run(engine, frames=10)
    p = _positions(engine)
    inside = engine._occupied(p)
    check(not inside.any(), f"{int(inside.sum())} particles ended up inside the collider")


def _moving_voxel_fixture(engine, wall_velocity):
    """Install one occupied voxel and a matching packed wall-velocity field."""
    voxel = SPACING
    dims = (int(1.0 / voxel),) * 3
    occupancy = np.zeros((dims[2], dims[1], dims[0]), dtype=np.uint8)
    velocity = np.zeros((dims[2], dims[1], dims[0], 3), dtype=np.float32)
    cell = (10, 10, 4)
    occupancy[cell[2], cell[1], cell[0]] = 1
    velocity[cell[2], cell[1], cell[0]] = wall_velocity
    engine.set_collider(
        None,
        dims,
        voxel,
        occupancy=occupancy.tobytes(),
        velocities=velocity.tobytes(),
    )
    return cell


def test_moving_collider_relative_normal_response():
    engine = _make_cpu()
    _moving_voxel_fixture(engine, (0.0, 0.0, 1.0))
    index = 0
    # Near the occupied voxel's upper face, so recovery has an unambiguous +Z
    # normal. X is tangential and must remain untouched by a frictionless wall.
    point = np.array((0.525, 0.525, 0.249), dtype=np.float32)
    engine.state["predicted"][index, :3] = point
    engine.state["velocities"][index, :3] = (0.7, 0.0, -0.2)
    engine.state["delta"][index, :3] = 0.0
    engine._finalize_pass()
    velocity = engine.state["velocities"][index, :3]
    expected_normal = -0.2 - (-1.2) * (1.0 + engine.params["boundary_damping"])
    check(abs(float(velocity[0]) - 0.7) < 1e-5, "moving collider changed tangential speed")
    check(
        abs(float(velocity[2]) - expected_normal) < 1e-5,
        f"moving-wall normal response was {velocity[2]:.5f}, expected {expected_normal:.5f}",
    )
    check(
        not engine._occupied(engine.state["positions"][[index], :3])[0],
        "particle was not pushed out",
    )


def test_static_collider_response_is_unchanged():
    engine = _make_cpu()
    voxel = SPACING
    dims = (int(1.0 / voxel),) * 3
    occupancy = np.zeros((dims[2], dims[1], dims[0]), dtype=np.uint8)
    occupancy[4, 10, 10] = 1
    engine.set_collider(None, dims, voxel, occupancy=occupancy.tobytes())
    index = 0
    engine.state["predicted"][index, :3] = (0.525, 0.525, 0.249)
    engine.state["velocities"][index, :3] = (0.7, 0.0, -0.2)
    engine.state["delta"][index, :3] = 0.0
    engine._finalize_pass()
    velocity = engine.state["velocities"][index, :3]
    expected_normal = -0.2 - (-0.2) * (1.0 + engine.params["boundary_damping"])
    check(abs(float(velocity[0]) - 0.7) < 1e-6, "static collider changed tangential speed")
    check(abs(float(velocity[2]) - expected_normal) < 1e-5, "static response regressed")


def test_rotating_collider_velocity_sampling():
    engine = _make_cpu()
    voxel = SPACING
    dims = (int(1.0 / voxel),) * 3
    occupancy = np.ones((dims[2], dims[1], dims[0]), dtype=np.uint8)
    z, y, x = np.indices((dims[2], dims[1], dims[0]), dtype=np.float32)
    centers = np.stack((x + 0.5, y + 0.5, z + 0.5), axis=-1) * voxel
    relative = centers - np.array((0.5, 0.5, 0.5), dtype=np.float32)
    omega = np.array((0.0, 0.0, 2.0), dtype=np.float32)
    velocity = np.cross(np.broadcast_to(omega, relative.shape), relative)
    engine.set_collider(
        None,
        dims,
        voxel,
        occupancy=occupancy.tobytes(),
        velocities=velocity.astype(np.float32).tobytes(),
    )
    points = np.array(((0.225, 0.525, 0.525), (0.775, 0.525, 0.525)), dtype=np.float32)
    sampled = engine._collider_velocity(points)
    check(sampled[0, 1] < 0.0 < sampled[1, 1], "rotating wall directions were not preserved")
    check(
        abs(float(sampled[0, 0] - sampled[1, 0])) < 1e-6,
        "rotation sampling was not symmetric",
    )


def test_surface_field_brackets_the_iso():
    engine = _make_cpu()
    _run(engine, frames=4)
    spacing = SPACING
    margin = engine.params["smoothing_radius"]
    dims = tuple(int((1.0 + 2 * margin) / spacing) + 1 for _ in range(3))
    surface = SurfaceCfg(dims, spacing, Vec(-margin, -margin, -margin), margin)
    engine.alloc_surface(surface.sample_count)
    engine.splat_surface(surface)
    field = np.asarray(engine.read_surface(surface.sample_count))
    check(np.isfinite(field).all(), "surface field has non-finite samples")
    check(field.max() > 0.5, f"surface field never reaches the iso value (max {field.max():.3f})")
    check(field.min() <= 0.0, "surface field has no empty samples, so nothing would close")
    print(f"  field {field.min():.2f}..{field.max():.2f}, {int((field > 0.5).sum())} above iso")


def test_whitewater_spawns_and_expires():
    engine = _make_cpu()
    ww = WhitewaterCfg()
    engine.alloc_whitewater(ww.capacity, engine.config.sorted_count)
    _run(engine, frames=4)
    engine.build_grid()

    cursor, dt = 0, 1.0 / 24.0
    for frame in range(6):
        engine.step_whitewater(ww, cursor, 20, frame, dt)
        cursor = (cursor + 20) % ww.capacity
    pool, velkind = engine.read_whitewater(ww.capacity)
    alive = [p for p in pool if p[3] > 0.0]
    check(len(alive) > 0, "no whitewater particles were spawned")
    kinds = {int(v[3]) for p, v in zip(pool, velkind, strict=True) if p[3] > 0.0}
    check(kinds <= {0, 1, 2}, f"whitewater produced unknown kinds {kinds}")
    print(f"  {len(alive)} live whitewater particles, kinds {sorted(kinds)}")

    # Long enough for every lifetime to run out, with no new spawns.
    for _ in range(200):
        engine.step_whitewater(ww, cursor, 0, 99, dt)
    pool, _velkind = engine.read_whitewater(ww.capacity)
    check(all(p[3] <= 0.0 for p in pool), "whitewater particles never expired")


def test_whitewater_uses_moving_collider_normal_velocity():
    engine = _make_cpu()
    _moving_voxel_fixture(engine, (0.0, 0.0, 2.0))
    engine.params.update(gravity=0.0)
    ww = WhitewaterCfg()
    engine.alloc_whitewater(ww.capacity, engine.config.sorted_count)
    # Tangential X motion lands at the occupied voxel's X centre, leaving the
    # nearest-free recovery normal unambiguously along +Z.
    engine.ww["positions"][0] = (0.509, 0.525, 0.190, 1.0)
    engine.ww["velkind"][0] = (0.4, 0.0, 1.0, 0.0)
    engine._whitewater_advect(ww, 0.04)
    position = engine.ww["positions"][0, :3]
    velocity = engine.ww["velkind"][0, :3]
    expected_normal = 1.0 - (1.0 - 2.0) * (1.0 + engine.params["boundary_damping"])
    check(abs(float(velocity[0]) - 0.4) < 1e-6, "whitewater tangential speed changed")
    check(
        abs(float(velocity[2]) - expected_normal) < 1e-5,
        "whitewater did not use collider-relative normal speed",
    )
    check(not engine._occupied(position[None, :])[0], "whitewater remained inside collider")


def test_moving_collider_agrees_with_metal():
    """Packed wall motion drives the same PBF contact and APIC face on Metal."""
    backend_pkg = load("solver.backend")
    backend = backend_pkg.select()
    if backend is None:
        print("  skipped: no GPU device on this machine")
        return

    # PBF relative-normal collision response.
    cpu = _make_cpu()
    _moving_voxel_fixture(cpu, (0.0, 0.0, 1.0))
    index = 0
    point = np.array((0.525, 0.525, 0.249), dtype=np.float32)
    cpu.state["predicted"][index, :3] = point
    cpu.state["velocities"][index, :3] = (0.7, 0.0, -0.2)
    cpu.state["delta"][index, :3] = 0.0

    packed = np.concatenate(
        (cpu.collider.astype(np.float32).ravel(), cpu.collider_velocity.ravel())
    ).astype(np.float32)
    collider_buffer = backend.buffer(packed.nbytes, packed)
    gpu = load("solver.engine.metal_engine").MetalEngine(backend)
    gpu.params = cpu.params
    gpu.allocate(cpu.config, np.asarray(cpu.state["positions"], dtype=np.float32).ravel())
    gpu.set_collider(
        collider_buffer,
        cpu.collider_dims,
        cpu.params["collider_voxel"],
        velocities=cpu.collider_velocity.tobytes(),
    )
    predicted = np.frombuffer(gpu.buffers["predicted"].map(), dtype=np.float32).reshape(-1, 4)
    velocity = np.frombuffer(gpu.buffers["velocities"].map(), dtype=np.float32).reshape(-1, 4)
    delta = np.frombuffer(gpu.buffers["delta"].map(), dtype=np.float32).reshape(-1, 4)
    predicted[index, :3] = point
    velocity[index, :3] = (0.7, 0.0, -0.2)
    delta[index, :3] = 0.0
    cpu._finalize_pass()
    gpu.record("sph_finalize", cpu.config.particle_count)
    gpu.flush()
    check(
        np.allclose(velocity[index, :3], cpu.state["velocities"][index, :3], atol=1e-5),
        "Metal PBF moving-wall response disagreed with CPU",
    )

    # APIC moving no-penetration face, before and after projection.
    apic = _make_apic()
    voxel = SPACING
    dims = (int(1.0 / voxel),) * 3
    occupancy = np.zeros((dims[2], dims[1], dims[0]), dtype=np.uint8)
    wall_velocity = np.zeros((dims[2], dims[1], dims[0], 3), dtype=np.float32)
    occupancy[:, :, 10:12] = 1
    wall_velocity[:, :, 10:12, 0] = 1.0
    apic.set_collider(
        None,
        dims,
        voxel,
        occupancy=occupancy.tobytes(),
        velocities=wall_velocity.tobytes(),
    )
    packed = np.concatenate((occupancy.astype(np.float32).ravel(), wall_velocity.ravel())).astype(
        np.float32
    )
    collider_buffer = backend.buffer(packed.nbytes, packed)
    apic_gpu = load("solver.engine.apic_metal").ApicMetalEngine(backend)
    apic_gpu.params = apic.params
    apic_gpu.allocate(apic.config, np.asarray(apic.state["positions"], dtype=np.float32).ravel())
    apic_gpu.set_collider(collider_buffer, dims, voxel, velocities=wall_velocity.tobytes())
    nodes = math.prod(n + 1 for n in apic.config.cell_dims)
    apic_gpu.build_grid()
    apic_gpu.record("apic_grid_clear", max(nodes, apic.config.cell_count))
    apic_gpu.record("apic_classify", apic.config.cell_count)
    apic_gpu.record("apic_p2g", nodes)
    apic_gpu.record("apic_grid_update", nodes)
    apic_gpu.flush()
    grid = np.frombuffer(apic_gpu.buffers["grid_velocity"].map(), dtype=np.float32).reshape(
        nodes, 4
    )
    nx, ny, _nz = apic.config.cell_dims
    face_index = (0 * (ny + 1) + 0) * (nx + 1) + 5
    check(abs(float(grid[face_index, 0]) - 1.0) < 1e-6, "Metal APIC missed wall velocity")
    apic_gpu.record("apic_divergence", apic.config.cell_count)
    ping = 0
    for _ in range(apic.config.pressure_iterations):
        apic_gpu.record("apic_pressure", apic.config.cell_count, pressure_ping=ping)
        ping = 1 - ping
    apic_gpu.record("apic_project", nodes, pressure_ping=ping)
    apic_gpu.flush()
    check(
        abs(float(grid[face_index, 0]) - 1.0) < 1e-6,
        "Metal APIC projection erased wall velocity",
    )


def test_agrees_with_the_gpu_engine():
    """Both engines, same configuration, compared as aggregates.

    Skipped where there is no device. Not a bit-exact comparison: the CPU
    engine enumerates neighbours once per substep and the kernels re-derive
    them per pass, so the two diverge in the last bits and then chaotically.
    They must still describe the same fluid.
    """
    backend_pkg = load("solver.backend")
    backend = backend_pkg.select()
    if backend is None:
        print("  skipped: no GPU device on this machine")
        return

    gpu = load("solver.engine.metal_engine").MetalEngine(backend)
    config, block, seed = _scene()
    gpu.params = block
    gpu.allocate(config, seed)
    gpu.set_collider(None, (1, 1, 1), 0.0)

    cpu = _make_cpu()
    for _ in range(8):
        for _ in range(2):
            gpu.substep(1.0 / 48.0)
        gpu.flush()
    _run(cpu)

    a = np.asarray(gpu.read_vec4("positions", config.particle_count))[:, :3]
    b = _positions(cpu)
    centroid = np.abs(a.mean(axis=0) - b.mean(axis=0)).max()
    bounds = max(
        np.abs(a.min(axis=0) - b.min(axis=0)).max(), np.abs(a.max(axis=0) - b.max(axis=0)).max()
    )
    print(f"  centroid delta {centroid:.4f} m, bounds delta {bounds:.4f} m")
    check(centroid < 0.02, f"engines disagree on the fluid's centroid by {centroid:.4f} m")
    check(bounds < 0.10, f"engines disagree on the fluid's extent by {bounds:.4f} m")


def test_apic_agrees_with_metal_and_projects():
    """Short APIC CPU/Metal agreement plus a grid-level projection check."""
    backend_pkg = load("solver.backend")
    backend = backend_pkg.select()
    if backend is None:
        print("  skipped: no GPU device on this machine")
        return

    metal_type = load("solver.engine.apic_metal").ApicMetalEngine
    for solver, blend in (("jacobi", 0.0), ("pcg", 0.0), ("pcg", 0.5)):
        gpu = metal_type(backend)
        config, block, seed = _scene()
        config.pressure_solver = solver
        block.update(flip_blend=blend)
        gpu.params = block
        gpu.allocate(config, seed)
        gpu.set_collider(None, (1, 1, 1), 0.0)
        cpu = _make_apic_solver(solver)
        cpu.params.update(flip_blend=blend)

        for _ in range(2):
            for _ in range(2):
                gpu.substep(1.0 / 48.0)
                cpu.substep(1.0 / 48.0)
            gpu.flush()

        a = np.asarray(gpu.read_vec4("positions", config.particle_count))[:, :3]
        b = _positions(cpu)
        centroid = np.abs(a.mean(axis=0) - b.mean(axis=0)).max()
        bounds = max(
            np.abs(a.min(axis=0) - b.min(axis=0)).max(),
            np.abs(a.max(axis=0) - b.max(axis=0)).max(),
        )
        label = f"{solver}, FLIP {blend}"
        print(f"  APIC {label}: centroid delta {centroid:.4f} m, bounds delta {bounds:.4f} m")
        check(centroid < 0.02, f"APIC {label} CPU/Metal centroid drift is {centroid:.4f} m")
        check(bounds < 0.10, f"APIC {label} CPU/Metal bounds drift is {bounds:.4f} m")

    # A single fluid cell with a divergent face field has an exact pressure
    # solution and isolates the Metal Jacobi/projection chain from transfers.
    nx, ny, nz = config.cell_dims
    nodes = (nx + 1) * (ny + 1) * (nz + 1)
    velocity = np.frombuffer(gpu.buffers["grid_velocity"].map(), dtype=np.float32).reshape(nodes, 4)
    cell = np.frombuffer(gpu.buffers["grid_scratch"].map(), dtype=np.float32).reshape(
        config.cell_count, 4
    )
    velocity.fill(0.0)
    cell.fill(0.0)
    center = (nx // 2, ny // 2, nz // 2)

    def cell_index(c):
        return (c[2] * ny + c[1]) * nx + c[0]

    def node_index(c):
        return (c[2] * (ny + 1) + c[1]) * (nx + 1) + c[0]

    center_index = cell_index(center)
    cell[center_index, 3] = 1.0
    velocity[node_index((center[0] + 1, center[1], center[2])), 0] = 1.0
    gpu.record("apic_divergence", config.cell_count)
    gpu.flush()
    before = abs(float(cell[center_index, 0]))
    ping = 0
    for _ in range(config.pressure_iterations):
        gpu.record("apic_pressure", config.cell_count, pressure_ping=ping)
        ping = 1 - ping
    gpu.record("apic_project", nodes, pressure_ping=ping)
    gpu.flush()
    base = node_index(center)
    after = abs(
        (
            velocity[node_index((center[0] + 1, center[1], center[2])), 0]
            - velocity[base, 0]
            + velocity[node_index((center[0], center[1] + 1, center[2])), 1]
            - velocity[base, 1]
            + velocity[node_index((center[0], center[1], center[2] + 1)), 2]
            - velocity[base, 2]
        )
        / gpu.params["grid_spacing"]
    )
    print(f"  Metal APIC fixture divergence {before:.4f} -> {after:.4f}")
    check(after <= before * 0.10, "Metal APIC pressure left too much divergence")

    # Exercise Metal's affine P2G/G2P reconstruction on an exact rigid field.
    transfer = _make_transfer_apic()
    points = _positions(transfer)
    center_point = points.mean(axis=0)
    omega = 0.4
    matrix = np.array(((0.0, -omega, 0.0), (omega, 0.0, 0.0), (0.0, 0.0, 0.0)), dtype=np.float32)
    expected_velocity = (points - center_point) @ matrix.T
    transfer.state["velocities"][:, :3] = expected_velocity
    transfer.state["affine"][:, :, :3] = matrix
    transfer_state = transfer.snapshot_state()
    gpu.params = transfer.params
    gpu.allocate(transfer.config, np.asarray(transfer_state["positions"]).ravel())
    gpu.set_collider(None, (1, 1, 1), 0.0)
    gpu.restore_state(transfer_state)
    transfer_nodes = math.prod(n + 1 for n in transfer.config.cell_dims)
    gpu.build_grid()
    gpu.record("apic_grid_clear", max(transfer_nodes, transfer.config.cell_count))
    gpu.record("apic_classify", transfer.config.cell_count)
    gpu.record("apic_p2g", transfer_nodes)
    gpu.record("apic_grid_update", transfer_nodes)
    gpu.record("apic_g2p", transfer.config.particle_count)
    gpu.flush()
    snapshot = gpu.snapshot_state()
    got_velocity = np.asarray(snapshot["velocities"])[:, :3]
    got_affine = np.asarray(snapshot["affine"]).reshape(-1, 3, 4)[:, :, :3]
    check(
        np.max(np.abs(got_velocity - expected_velocity)) < 1e-5,
        "Metal APIC rigid velocity did not round-trip",
    )
    check(
        np.max(np.abs(got_affine - matrix)) < 1e-5,
        "Metal APIC affine rows were not reconstructed",
    )
    gpu.buffers["affine"].zero()
    gpu.restore_state(snapshot)
    restored = gpu.snapshot_state()
    check(snapshot["affine"] == restored["affine"], "Metal APIC affine state did not round-trip")


def test_apic_pcg_agrees_with_metal():
    """The Metal PCG chain against the numpy one, on the same grid problem.

    Both solve the pressure system for one divergence field and one cell
    classification - taken from a real dam-break substep, with a solid block
    added so walls, solids and air all appear in the stencil - and the
    pressure fields are compared directly, which isolates the solve from the
    transfers around it. The CPU runs in float64 and Metal in float32, so they
    are not expected to agree to the bit, and near the tolerance they could
    stop an iteration apart. Measured: the same iteration, and pressures within
    4e-7 of their peak. The bounds leave a decade of room over that.

    Also: repeated Metal solves are bit-identical (the reduction order is
    fixed), and a Metal projection with PCG pressure removes the divergence of
    the single-cell fixture the Jacobi check uses, far past Jacobi's 90%.
    """
    backend_pkg = load("solver.backend")
    backend = backend_pkg.select()
    if backend is None:
        print("  skipped: no GPU device on this machine")
        return

    cpu = _make_apic_solver("pcg")
    for _ in range(4):
        cpu.substep(1.0 / 48.0)
    cell_type = cpu.grid_cell[..., 3].copy()
    cell_type[2:4, 3:6, 3:6] = -1.0
    divergence = cpu.grid_cell[..., 0].copy()
    divergence[cell_type <= 0.0] = 0.0
    dt = 1.0 / 48.0

    gpu = load("solver.engine.apic_metal").ApicMetalEngine(backend)
    config, block, seed = _scene()
    gpu.params = block
    gpu.allocate(config, seed)
    gpu.set_collider(None, (1, 1, 1), 0.0)
    gpu.params.update(dt=dt)
    cells = np.frombuffer(gpu.buffers["grid_scratch"].map(), dtype=np.float32).reshape(
        config.cell_count, 4
    )

    rhs = REST_DENSITY * SPACING * 2.0 * SPACING * 2.0 * divergence / dt
    want = cpu._solve_pcg(rhs.astype(np.float32), cell_type).ravel()
    want_iterations = cpu.last_pressure_iterations

    solutions = []
    for _ in range(3):
        cells[:] = 0.0
        cells[:, 0] = divergence.ravel()
        cells[:, 3] = cell_type.ravel()
        gpu.record_pcg()
        gpu.flush()
        solutions.append(cells[:, 1].copy())
    iterations, _rr = gpu.pcg_stats()
    got = solutions[0]
    error = float(np.abs(got - want).max() / np.abs(want).max())
    print(
        f"  PCG pressure CPU/Metal relative error {error:.1e}; "
        f"iterations CPU {want_iterations}, Metal {iterations}"
    )
    check(
        all(np.array_equal(got, other) for other in solutions[1:]),
        "repeated Metal PCG solves differ",
    )
    check(abs(iterations - want_iterations) <= 1, "Metal PCG stopped far from the CPU solve")
    check(error < 1e-5, f"Metal PCG pressure differs from the CPU by {error:.1e} of peak")

    # The single-cell fixture from the Jacobi check, through PCG.
    nx, ny, nz = config.cell_dims
    nodes = (nx + 1) * (ny + 1) * (nz + 1)
    velocity = np.frombuffer(gpu.buffers["grid_velocity"].map(), dtype=np.float32).reshape(nodes, 4)
    velocity.fill(0.0)
    cells.fill(0.0)
    center = (nx // 2, ny // 2, nz // 2)
    center_cell = (center[2] * ny + center[1]) * nx + center[0]

    def node_index(c):
        return (c[2] * (ny + 1) + c[1]) * (nx + 1) + c[0]

    cells[center_cell, 3] = 1.0
    velocity[node_index((center[0] + 1, center[1], center[2])), 0] = 1.0
    gpu.record("apic_divergence", config.cell_count)
    gpu.flush()
    before = abs(float(cells[center_cell, 0]))
    gpu.record_pcg()
    gpu.record("apic_project", nodes, pressure_ping=0)
    gpu.record("apic_divergence", config.cell_count)
    gpu.flush()
    after = abs(float(cells[center_cell, 0]))
    print(f"  Metal PCG fixture divergence {before:.4f} -> {after:.2e}")
    check(after <= before * 1e-3, "Metal PCG projection left too much divergence")


def main():
    tests = [
        ("dam break stays finite", test_dam_break_stays_finite),
        ("fluid stays in the box", test_fluid_stays_in_the_box),
        ("it falls and spreads", test_it_actually_falls_and_spreads),
        ("density approaches rest", test_density_approaches_rest),
        ("collider is not penetrated", test_collider_is_not_penetrated),
        (
            "moving collider uses relative normal velocity",
            test_moving_collider_relative_normal_response,
        ),
        ("static collider response is unchanged", test_static_collider_response_is_unchanged),
        ("rotating collider velocity samples", test_rotating_collider_velocity_sampling),
        ("surface field brackets the iso", test_surface_field_brackets_the_iso),
        ("whitewater spawns and expires", test_whitewater_spawns_and_expires),
        (
            "whitewater uses moving collider normal velocity",
            test_whitewater_uses_moving_collider_normal_velocity,
        ),
        ("moving collider agrees with Metal", test_moving_collider_agrees_with_metal),
        ("PBF snapshot continuation is exact", test_pbf_snapshot_continuation_is_exact),
        ("APIC stays finite and bounded", test_apic_stays_finite_and_bounded),
        ("APIC projection residual", test_apic_projection_residual),
        ("APIC affine rows stay bounded at rest", test_apic_affine_rows_stay_bounded_at_rest),
        ("APIC pool volume drift", test_apic_pool_volume_drift),
        ("APIC PCG sealed regions", test_apic_pcg_sealed_regions),
        ("APIC PCG is deterministic", test_apic_pcg_is_deterministic),
        (
            "APIC transfer preserves translation and rotation",
            test_apic_transfer_preserves_translation_and_rotation,
        ),
        (
            "APIC rotating block holds angular momentum",
            test_apic_rotating_block_holds_angular_momentum,
        ),
        (
            "APIC rotating block at the top recommended FLIP blend",
            lambda: test_apic_rotating_block_holds_angular_momentum(blend=0.5),
        ),
        ("APIC FLIP blend 0 is APIC", test_apic_flip_blend_zero_is_apic),
        ("APIC FLIP dam break holds volume", test_apic_flip_dam_break_holds_volume),
        ("APIC FLIP continuation is exact", test_apic_flip_snapshot_continuation_is_exact),
        (
            "APIC confinement clamps every face component",
            test_apic_confinement_clamps_each_face_component,
        ),
        ("APIC collider, surface and whitewater", test_apic_collider_surface_and_whitewater),
        (
            "APIC preserves moving solid-face velocity",
            test_apic_preserves_moving_solid_face_velocity,
        ),
        ("APIC snapshot continuation is exact", test_apic_snapshot_continuation_is_exact),
        ("agrees with the GPU engine", test_agrees_with_the_gpu_engine),
        ("APIC agrees with Metal and projects", test_apic_agrees_with_metal_and_projects),
        ("APIC PCG agrees with Metal", test_apic_pcg_agrees_with_metal),
    ]
    failures = 0
    for name, run in tests:
        try:
            run()
        except Failure as exc:
            print(f"FAIL {name}: {exc}")
            failures += 1
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
