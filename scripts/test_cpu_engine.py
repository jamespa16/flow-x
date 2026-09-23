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
    )
    return config, block, seed.ravel()


def _run(engine, frames=8, substeps=2, dt=1.0 / 48.0):
    for _ in range(frames):
        for _ in range(substeps):
            engine.substep(dt)


def _positions(engine):
    return np.asarray(engine.read_vec4("positions", engine.config.particle_count))[:, :3]


def _make_cpu():
    config, block, seed = _scene()
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


def test_apic_projection_reduces_divergence():
    engine = _make_apic()
    # The first free-surface projection begins from an empty grid and is the
    # hardest. A few normal substeps are a more useful steady-state check.
    for _ in range(4):
        engine.substep(1.0 / 48.0)
    before = engine.last_divergence_before
    after = engine.last_divergence_after
    print(f"  APIC divergence {before:.4f} -> {after:.4f}")
    check(before > 0.0, "APIC test field had no divergence to project")
    check(after <= before * 0.10, f"APIC pressure left {after / before:.1%} divergence")


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


def test_apic_rotating_block_holds_angular_momentum():
    engine = _make_transfer_apic(points_per_axis=4)
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
    print(f"  rotating-block angular momentum drift {drift:.2%}")
    check(drift < 0.05, f"APIC angular momentum drifted {drift:.2%} over 300 steps")


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


def test_pbf_snapshot_continuation_is_exact():
    """PBF position/velocity plus the whitewater pool are sufficient state."""
    ww = WhitewaterCfg()
    ww.capacity = 64
    original = _make_cpu()
    original.alloc_whitewater(ww.capacity, original.config.sorted_count)
    _run(original, frames=2, substeps=1)
    original.build_grid()
    original.step_whitewater(ww, 0, 16, 1, 1.0 / 24.0)
    checkpoint = original.snapshot_state(include_whitewater=True)

    original.substep(1.0 / 48.0)
    original.build_grid()
    original.step_whitewater(ww, 16, 12, 2, 1.0 / 24.0)
    uninterrupted = original.snapshot_state(include_whitewater=True)

    resumed = _make_cpu()
    resumed.alloc_whitewater(ww.capacity, resumed.config.sorted_count)
    resumed.restore_state(checkpoint)
    resumed.substep(1.0 / 48.0)
    resumed.build_grid()
    resumed.step_whitewater(ww, 16, 12, 2, 1.0 / 24.0)
    continued = resumed.snapshot_state(include_whitewater=True)

    for key in ("positions", "velocities", "ww_positions", "ww_velkind"):
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
    gpu = metal_type(backend)
    config, block, seed = _scene()
    gpu.params = block
    gpu.allocate(config, seed)
    gpu.set_collider(None, (1, 1, 1), 0.0)
    cpu = _make_apic()

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
    print(f"  APIC centroid delta {centroid:.4f} m, bounds delta {bounds:.4f} m")
    check(centroid < 0.02, f"APIC CPU/Metal centroid drift is {centroid:.4f} m")
    check(bounds < 0.10, f"APIC CPU/Metal bounds drift is {bounds:.4f} m")

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


def main():
    tests = [
        ("dam break stays finite", test_dam_break_stays_finite),
        ("fluid stays in the box", test_fluid_stays_in_the_box),
        ("it falls and spreads", test_it_actually_falls_and_spreads),
        ("density approaches rest", test_density_approaches_rest),
        ("collider is not penetrated", test_collider_is_not_penetrated),
        ("surface field brackets the iso", test_surface_field_brackets_the_iso),
        ("whitewater spawns and expires", test_whitewater_spawns_and_expires),
        ("PBF snapshot continuation is exact", test_pbf_snapshot_continuation_is_exact),
        ("APIC stays finite and bounded", test_apic_stays_finite_and_bounded),
        ("APIC projection reduces divergence", test_apic_projection_reduces_divergence),
        (
            "APIC transfer preserves translation and rotation",
            test_apic_transfer_preserves_translation_and_rotation,
        ),
        (
            "APIC rotating block holds angular momentum",
            test_apic_rotating_block_holds_angular_momentum,
        ),
        ("APIC collider, surface and whitewater", test_apic_collider_surface_and_whitewater),
        ("APIC snapshot continuation is exact", test_apic_snapshot_continuation_is_exact),
        ("agrees with the GPU engine", test_agrees_with_the_gpu_engine),
        ("APIC agrees with Metal and projects", test_apic_agrees_with_metal_and_projects),
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
