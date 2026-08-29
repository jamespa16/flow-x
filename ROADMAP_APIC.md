# flow-x Roadmap — APIC solver

Affine Particle-In-Cell (Jiang et al., *The Affine Particle-In-Cell Method*, SIGA 2015)
as a second simulation method alongside PBF. This is the `eventually, APIC` item in
`notes.md`, expanded into a build plan grounded in the current architecture.

## Why APIC, and what it buys Flow-X specifically

- **Angular momentum.** PBF (and PIC before it) bleeds rotation; vortices in a drained
  bathtub are the visible symptom. APIC stores an affine velocity field per particle,
  which provably conserves linear *and* angular momentum through the transfer. This is
  the feature PBF cannot fake.
- **Detail for free.** APIC's grid velocity pairs with vorticity confinement (Fedkiw),
  re-injecting the swirl numerical dissipation eats. Flow-X's whitewater already fakes
  "energetic fluid" with spawn scoring; confinement makes the *fluid itself* energetic.
- **Bigger steps.** No constraint loop, no XSPH. Per-substep cost drops from
  `3 × pbf_iterations + ~105 bitonic passes` to a fixed handful of O(nodes) grid
  passes. The frame budget moves from particle count to grid resolution, which is a
  knob the user already has.
- **Collisions get better.** The collider occupancy grid is already the same shape as
  the APIC grid; zeroing/projecting grid velocity inside occupied voxels beats the
  current per-particle "push to nearest free voxel" for thin and animated colliders —
  which is exactly note #1's weakness.

What it costs: APIC's grid transfer is more diffuse than PBF near boundaries (fluid
clings/leaks slightly at walls), and it needs *two* engines (Metal + CPU twin) because
the engine protocol is pass-group level and the cache demands engine identity.

## Architecture decisions

| Decision | Choice | Why |
|---|---|---|
| Method vs. device | New `solver_method` enum (`pbf`, `apic`) on the domain, orthogonal to the existing `engine` enum (`auto`, `metal`, `cpu`) | The engine protocol in `solver/engine/__init__.py` is per pass-*group* (`substep`, `splat_surface`, `step_whitewater`), so an APIC engine is a new protocol implementor, not a protocol change. `sph.py`'s timeline, cache and operators stay untouched |
| Engines | `apic_metal.py` + `apic_cpu.py`, registered in `solver/engine/create()` | Mirrors the existing metal/cpu pair exactly; the CPU twin is required anyway (fallback + the reference the GPU is checked against) |
| Particle state | Add an `affine` buffer: 12 floats/particle (three `float4` rows, stride 48 B — a multiple of 16 so MSL and `struct.pack` agree without padding games) | One new slot appended to `BINDING_ORDER` and a `BUF_AFFINE 15` in `kernels/flowx_prelude.h`; PBF kernels ignore it like they ignore the whitewater slots |
| Grid | Node grid at the existing cell lattice (dims `cells_x/y/z + 1`), reusing `cell_size`/`lo` Params | The collider occupancy grid is already voxelized at the domain-grid resolution; same lattice means collisions are sampled at exactly the grid's resolution, for free |
| Transfer weights | Trilinear (linear B-spline, 8-node stencil) for v1 | 2×2×2 node access per particle vs. quadratic's 5×5×5 cell gather; momentum conservation holds for any partition-of-unity weights with the MLS correction; quadratic becomes a later quality knob |
| Transfer direction | **Gather, never scatter.** Each grid node iterates the particle ranges of its surrounding cells; each particle gathers from its 8 nodes | Atomics-free and deterministic by construction, which the disk cache's determinism guarantee requires. Float atomic-add scatter is order-dependent and would silently break cached scrubbing — do not "simplify" it that way |
| Spatial hash | Reuse `build_grid()` (keys + bitonic sort + cell ranges) unchanged | The gather transfer needs exactly the sorted-by-cell order it already produces; the `(cell_key, particle_index)` key keeps it deterministic |
| Collisions | Grid-level (zero/reflect node velocity in occupied voxels) *plus* the existing particle-level `finalize` clamp for the domain walls | Grid collision handles thin/animated geometry; the clamp keeps particles honest when the grid projection is degenerate |
| Whitewater | Keep writing the `normal` buffer (and a density proxy) from an APIC pass so `whitewater_potential/sort/spawn/advect` are **unchanged** | Whitewater scores off the neighbor-loop density/normal field; keeping its inputs identical means zero changes downstream and no whitewater re-validation |
| Surface | `surface_splat` unchanged (cubic kernel over particle positions) | APIC's quadratic transfer kernel has negative lobes — splatting it produces holes in the iso-surface. The splat kernel and the transfer kernel are deliberately different kernels |
| Surface tension | Out of scope for APIC v1 | Akinci cohesion is SPH-shaped; the grid analog (CSF curvature force) is its own project. UI hides the slider when `solver_method == apic` |
| Vorticity confinement | On the grid, 2 passes (curl → confinement force), `epsilon` exposed, default ~0.3 | This is APIC's payoff; it is also the one pass that *adds* energy, so it must be user-visible, never silently on |
| FLIP/MIP hybrid | Deferred | The classic 95/5 FLIP weight is a big quality win on top, but it needs a PIC baseline to blend against — land APIC first and measure |

## Phase A — Method plumbing (no physics yet)

- `solver_method` EnumProperty on `obj.flowx_domain`; UI: a Method dropdown in the SPH
  Solver panel; hide PBF-only properties (iterations, XSPH viscosity, surface tension)
  when APIC is selected.
- `SolverConfig` gains the method; `resolve()` branches. `create(preferred)` in
  `solver/engine/__init__.py` becomes method-aware: `(method, engine)` → one of
  `pbf_metal` / `pbf_cpu` / `apic_metal` / `apic_cpu`.
- **Cache invalidation:** `cache.py` hashes `settings.engine` already (its comment:
  *"the engine is part of the physics, not just a performance choice"*). The method is
  the same case one level up — hash `solver_method` into the config digest and bump the
  frame-record format version, because an APIC frame also stores the per-particle affine
  matrix and a PBF-restore must never see it.
- `engine.describe()` / the panel readout / `smoke_test.py` report the method.

**Exit criteria:** selecting APIC and hitting Run falls back to PBF with a panel warning
("APIC engine not implemented"); PBF caches from before the change still load; the
`cache` digest changes the moment `solver_method` does.

## Phase B — Metal engine core (the physics)

New files: `solver/engine/apic_metal.py`, `kernels/apic_*.metal`. Per-substep chain:

```
build_grid()                       # unchanged: keys, bitonic sort, cell ranges
apic_grid_clear                    # mass/momentum/velocity nodes → 0
apic_transfer                      # gather: particles → nodes (mass, MLS affine momentum)
apic_grid_update                   # gravity; collider: zero/reflect occupied nodes
apic_vorticity                     # curl of grid velocity → |ω| (skip when epsilon == 0)
apic_confinement                   # N × ω stencil force onto grid velocity
apic_transfer_back                 # gather nodes → v_new; recompute affine C per particle
apic_finalize                      # domain clamp + wall damping (mirror sph_finalize)
```

- New buffers: `affine` (12 f), `grid_mass` (1 f), `grid_momentum` (4 f),
  `grid_velocity` (4 f), `grid_vort` (4 f), `grid_scratch` (4 f, curl/|ω| double-duty).
- New `Params` fields: `nodes_x/y/z`, `grid_spacing`, `vorticity_epsilon`,
  `grid_max_speed` clamp. **Append-only**, in lockstep across
  `kernels/flowx_prelude.h` and `solver/engine/params.py`, and extend
  `kernels/flowx_params_probe.metal` with the new offsets — `test_backend.py` compares
  them, and a drifted field silently corrupts everything after it.
- `C` update is the paper's formula, `v_i = (Σ_p w_ip v̂_p) / (Σ_p w_ip)` with
  `C_i = (Σ_p w_ip (v̂_p − v_i) ⊗ (x_p − x_i)) (Σ_p w_ip (x_p − x_i) ⊗ (x_p − x_i))⁻¹`;
  the 3×3 inverse needs a guard (analytic cofactor inverse + fallback `C = 0` on
  singular — a lone particle in a cell is a real case at the free surface).
- Seed path (`sph.py` seeding) is shared: APIC seeds identically, `C` starts zero.

**Exit criteria:** a block of APIC fluid drops and settles in `pool_with_ramp.blend`;
`test_backend`'s params probe passes; density readout in `smoke_test.py` stays sane
(≤ ~2% on the settled pool); frame time ≤ the PBF frame at the same resolution.

## Phase C — The CPU twin

`solver/engine/apic_cpu.py`, numpy, same protocol. The gather transfers map to
`np.bincount`-style segmented sums over the cell-sorted order; the grid stencils are
plain array shifts. This is the bigger half of Phase B's work, and it is not optional —
the no-Metal fallback and the reference implementation both live here.

- Extend `scripts/test_cpu_engine.py` to cross-check APIC metal↔cpu the way it already
  does PBF (same seed, N frames, bounded divergence).
- Add the APIC-signature regression: a **rotating block** — a cuboid of fluid with
  initial angular velocity, no gravity. APIC must hold the rotation for hundreds of
  frames; a momentum-losing method visibly stops spinning. This test is *why* APIC is
  being added, so it fails loudly if the affine transfer is wrong.

**Exit criteria:** APIC cpu↔metal agreement test passes; rotating-block angular
momentum drift < ~5% over 300 frames; `--engine=cpu` APIC runs the demos headless.

## Phase D — Collisions & timeline integration

- Grid-level collider response in `apic_grid_update`: occupied voxel → node velocity
  projected against the occupancy-gradient normal (the gradient is the same BVH ray
  parity field `collision/` already builds; add a cheap central-difference normal at
  upload time, CPU-side).
- Animated colliders (`rebuild_animated_grids`) now matter more, not less — grid
  collision is exactly where a stale animated grid shows up. Re-test note #1's case.
- `SolverConfig.substep_dt`: replace the PBF/XSPH limit with a grid CFL
  (`grid_max_speed · dt ≤ CFL_FACTOR · grid_spacing`) — larger steps than PBF takes;
  substep count stays derived from scene frame rate only, never wall-clock.
- Wire whitewater's density/normal proxy pass so spray/foam/bubble spawn off APIC
  surfaces with the same scoring semantics as PBF.

**Exit criteria:** ball-drops-into-pool demo with APIC reads the same as PBF; fluid
pools against the ramp and the animated ball without tunneling; backward-scrubbed cached
frames replay bit-exactly on both methods.

## Phase E — Validation, references & polish

- Re-bake `scripts/golden.py` references for APIC (`--method=apic` flag alongside the
  existing `--engine`). PBF references stay as they are — the two methods must never
  share a golden file.
- Sweep vorticity epsilon (0 / 0.15 / 0.3 / 0.6) on a dam-break-into-ramp; document the
  visual range in README ("swirl detail") rather than picking a heroic default.
- Panel: vorticity strength, grid resolution (default: follow the resolution knob,
  shown as node count), method + engine line in the stats readout.
- README: new section on method choice; Troubleshooting row for "APIC fluid clings to
  walls" (known PIC-family boundary stickiness — lower epsilon / raise resolution).

**Exit criteria:** `smoke_test.py` covers both methods on both engines; golden dumps
exist for `pool_with_ramp` under APIC; README documents the method dropdown, epsilon,
and APIC's known limits.

## Out of scope (explicitly deferred)

- FLIP/MIP blending, and any PIC/APIC/FLIP weight slider — needs v1 measured first.
- Newton-crash-style implicit grid solve / plasticine / viscoplastic materials.
- Mesh-based ELLS remeshing to fight APIC's long-run diffusion.
- Surface tension on APIC; two-way coupling; multi-domain.
- CUDA backend for APIC (same engine split as PBF: the CUDA engine, when it exists,
  implements both methods' protocol groups).

## Sequencing note

Phases A→B→D ship a usable PBF-quality APIC; Phase C can land in parallel but gates
release (no-Metal users and the CPU↔GPU agreement test both need it). Expect B and C to
be about equal in effort — the numpy twin is the honest test of the Metal engine, in the
same way `cpu_engine.py` is for PBF today.
