# Flow-X APIC pressure solve and FLIP blending

Status: implemented. Follows `ROADMAP_APIC.md`, which delivered the MAC-grid APIC solver with a
weighted-Jacobi pressure solve and lists "FLIP blending and PCG pressure solving" as deferred.
The plan below is kept as written; [Results](#results) records what was measured and where the
implementation departs from it.

## Goal

Make APIC's incompressibility measurably better and its motion less damped, without breaking
determinism (the cache depends on it) or the CPU/Metal agreement checks.

The order matters: measure first, improve the CPU reference, port to Metal, then add the
blend. Each phase leaves the tree green and is independently shippable.

## Phase 1 - Measure: divergence metric and baselines

Nothing below can be judged without a number for "how incompressible is this?".

- Add a residual-divergence metric to `scripts/test_cpu_engine.py`: max and RMS of the
  post-projection divergence over fluid cells, as a fraction of the pre-projection value.
  The current test only asserts a 90% reduction; record what Jacobi actually achieves at 40
  iterations so later phases have a baseline to beat.
- Add a volume-drift check: particle count is fixed, so use the fluid-cell count (or the
  density-splat volume) of a settled pool over N frames. Slow volume loss is the visible
  symptom of a weak solve.
- Record `scripts/golden.py` references for `--method apic` on both `--engine=cpu` and
  `--engine=metal` (skip Metal if `bin/` is unbuilt) before touching the solver. Commit the
  numbers, not the `.npz`, in the test's expected-value comments.

**Exit criteria:** `test_cpu_engine.py` prints the residual and drift for the Jacobi baseline
and asserts loose bounds around it; golden references exist for the pre-change solver.

## Phase 2 - PCG on the CPU engine

In `solver/engine/apic_cpu.py`, the numpy reference.

- Implement preconditioned conjugate gradient over the same pressure-cell system the Jacobi
  loop solves (solid / fluid / air classification is unchanged). Start with Jacobi
  (diagonal) preconditioning; consider a modified incomplete Cholesky or a multigrid
  V-cycle only if diagonal PCG is not clearly better.
- Keep Jacobi selectable. The `pressure_iterations` setting keeps its meaning (an iteration
  cap); add a solver choice and a convergence tolerance for early exit.
- Compare Jacobi and PCG at equal iteration count and at equal wall time using the Phase 1
  metric. Decide from data whether PCG should become the default.
- Determinism: a numpy reduction is deterministic on one machine, but confirm the summation
  order is fixed so cached frames stay valid.

**Exit criteria:** at 40 iterations PCG reaches a lower residual than Jacobi by a margin
recorded in the test, volume drift improves, and existing APIC tests still pass.

## Phase 3 - Port PCG to Metal

The Metal engine needs dot products, which the Jacobi path never did.

- Add a reduction kernel. Reductions must be deterministic: use a fixed two-level tree
  (per-threadgroup partial sums in a fixed order, then a second pass), not float atomics.
  Float-atomic accumulation would make the result order-dependent and silently break cached
  scrubbing, which is the same reason P2G is a face-owned gather.
- Add the PCG vector kernels (`apic_pcg_*.metal`): matrix-vector product, axpy, and the
  preconditioner apply. Register them in `APIC_ALL_PASSES` in `solver/engine/kernels.py` and
  record them from `ApicMetalEngine.substep` in place of the Jacobi loop.
- The scalars (alpha, beta) either round-trip through a small buffer that kernels read
  directly, or the loop runs a fixed iteration count with no host read-back. Prefer the
  latter: `record()` must stay free of GPU work and a whole frame is one submit.
- Extend the Metal-vs-CPU agreement test in `scripts/test_cpu_engine.py` to cover PCG.
- New buffers are installed in `ApicMetalEngine.allocate` and zeroed in `restore_state`.

**Exit criteria:** Metal PCG agrees with the CPU PCG within the tolerance already used for
the Jacobi comparison; `test_backend.py` covers the reduction kernel including an
order-independence check; a golden `compare` against the Phase 1 baseline shows the
deliberate, explained change.

## Phase 4 - FLIP blending

- Add a blend parameter (0 = pure APIC, 1 = pure FLIP; default stays 0 so existing scenes
  and caches are unchanged). In G2P, particle velocity becomes
  `(1 - a) * v_grid_new + a * (v_particle + (v_grid_new - v_grid_old))`. This requires
  keeping the pre-projection grid velocity: an extra `grid_velocity_old` buffer, or a
  second half of `grid_velocity`.
- Expose it on the domain property group and the N-panel next to the other APIC settings.
- `Params` change: add the field to `kernels/flowx_prelude.h` and
  `solver/engine/params.py` together, at the same position with the same type. The
  `flowx_params_probe.metal` comparison in `test_backend.py` is the guard; run it.
- Cache: the blend value goes into the config hash so different blends never share frames.
  If the per-frame state layout changes, bump the cache version and recreate old caches
  (as v3 to v4 did).
- FLIP is noisier than APIC. Check angular-momentum drift (the existing rotating-block test)
  and dam-break stability at a few blend values, and pick a documented recommended range.

**Exit criteria:** blend 0 reproduces Phase 3 results bit for bit on the CPU engine; blend
above 0 visibly reduces damping in a dam-break; the rotating-block drift test still passes
at the recommended blend.

## Housekeeping

- CLAUDE.md and the README say CI runs three Blender-free test scripts; `ci.yml` also runs
  `scripts/test_cache.py`. Correct the docs.
- Update `ROADMAP_APIC.md`'s "Deferred work" as items land.
- The smoke test should exercise PCG and a non-zero blend in Blender once they exist.

## Results

### Phase 1 - baselines

`scripts/test_cpu_engine.py` records the metric and asserts bounds on both sides of it.

| Weighted Jacobi, 40 iterations | RMS residual | Max residual |
|---|---|---|
| Dam break, fourth substep | 3.0% | 1.2% |
| Settled pool | 49% | 36% |

A still pool (fill 40%, 1 m box, 0.1 m cells) loses **14%** of its fluid cells in 24 frames under
Jacobi. On `demos/pool_with_ramp.blend` (Metal, 24 frames) the pool's centroid sinks from
z = 0.311 to 0.177, the same volume loss at demo resolution.

Golden references were recorded before any solver change (`--method=apic`, 24 frames,
9610 particles):

| Engine | Final centroid | Mean speed | Surface verts |
|---|---|---|---|
| CPU | (0.127, 0.061, 0.210) | 1.041 | 8642 |
| Metal | (−0.005, −0.030, 0.177) | 0.126 | 7322 |

After the change, `compare --pressure=jacobi` against these matches the CPU engine bit for bit.
On Metal the worst per-particle difference is 1.8e-7 m: the kernels were recompiled, which
changes the compiler's choices, and a Metal run is still bit-identical to itself.

### Phase 2 - PCG on the CPU engine

The diagonally preconditioned PCG over a compact fluid-cell vector:

| 40-iteration cap | Jacobi | PCG |
|---|---|---|
| Dam break RMS residual | 6.0e-2 | 1.0e-6 (converged at 26) |
| Settled pool RMS residual | 4.9e-1 | 1.5e-6 |
| CPU time per solve | 3.6 ms | 0.6 ms |
| Pool fluid cells lost in 24 frames | 14% | 0% |

Jacobi at 200 iterations is still worse than PCG at 20. Equal wall time favours PCG further,
since a PCG iteration over the compact vector is cheaper than a full-grid Jacobi sweep. PCG is
the default, with a relative-residual tolerance of 1e-3 (about 16 iterations on the dam break).
`pressure_iterations` keeps its meaning as the cap.

Determinism: dot products are `np.sum(a * b)` (pairwise, in an order fixed by the vector
length), not `np.dot`, which goes to BLAS and may split across a varying number of threads.

Singular systems: the solve starts from zero, so its iterates stay in D⁻¹·range(A), and a
breakdown guard stops it when p·Ap is not safely positive (1e-12 of the initial r·z). A domain
entirely full of fluid, and a sealed tank next to an open pool, both converge to 1e-5 with a
diagonal-weighted mean pressure of 1e-8 and 1e-6 of its magnitude, so no constant drifts in.
Explicit mean projection was not needed.

### Phase 3 - PCG on Metal

Five passes per iteration (`apic_pcg_matvec`, `_reduce`, `_update`, `_reduce`, `_direction`)
after `apic_pcg_init`, recorded at the fixed iteration cap with no read-back. The reduce pass
keeps alpha, beta, the threshold and a converged flag in a scalar buffer. Once the flag is set,
the remaining passes return immediately, so Metal early exit has the same semantics as the CPU
engine. Reductions are a 64-lane threadgroup tree plus one single-group second pass.

- CPU and Metal PCG on the same grid problem stop on the same iteration (12) and agree to
  3.5e-7 of peak pressure. Repeated Metal solves are bit-identical.
- `test_backend.py` checks the two-level sum bit for bit against a float32 emulation of the
  same order, over eight runs.
- Cost at 15.6k cells: 47 µs per PCG iteration against 32 µs per Jacobi iteration, so about
  8 µs per deterministic dot product. The smoke test's APIC frame is 1.09× PBF (limit 1.5×).
- `compare` against the Phase 1 baseline shows the deliberate change. On the ramp demo the
  Metal pool now holds its height: centroid z stays at 0.313 through 24 frames, against 0.177
  under Jacobi, and it stays at rest (mean speed 0.018 m/s against 0.126).

### Phase 4 - FLIP blending

Two departures from the plan, both forced by measurement:

1. **Positions move through the grid field, not the carried velocity.** As written (FLIP
   velocity used for advection too), a dam break at blend 1 lost half its fluid cells in one
   second (150 → 75) as particle-level noise, which is not divergence-free, packed particles
   against the floor. Following Zhu & Bridson, particles now move by the grid's interpolated
   velocity, and the FLIP velocity is carried only for the next P2G. That field exists only
   between G2P and the next P2G, so under a blend above 0 advection runs at the end of the
   substep. It uses the PBF-only `delta` buffer as scratch on Metal, and nothing new is
   persisted, so the cache layout, and its version, are unchanged. At blend 0 the historical
   order is kept, and the code paths are separate, so blend 0 is APIC bit for bit. A test
   poisons the FLIP grid copy with NaN at blend 0 to prove it is never read.
2. **FLIP's reference grid velocity is taken after the speed cap** (0.25 cell per substep),
   before gravity and solid faces. Taken before the cap, every particle faster than the cap
   lost its excess to FLIP's change each substep, and the dam break lost energy faster than
   under APIC.

The exit criterion "blend above 0 visibly reduces damping in a dam-break" is **not met by the
test dam break**, and the reason matters. At that resolution and time step, energy is set by
the speed cap rather than by transfer dissipation: lifting the cap makes APIC itself explode,
and with it in place APIC and FLIP 0.25–0.75 retain about the same energy. In a sealed
Taylor–Green vortex (no gravity, speeds far below the cap), pure FLIP keeps 1.03 of its
kinetic energy over 120 steps where plain PIC keeps 1e-4. APIC could not be compared there;
see the pre-existing issue below.

Recommended range, from the rotating-block test (64 particles, 300 steps):

| Blend | 0 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|
| Angular-momentum drift | 1.1% | 2.2% | 4.2% | 13% |

**0 to 0.5** is documented as recommended, and the test asserts the drift bound at 0.5. The dam
break stays finite and keeps its volume at 0.25, 0.5 and 1.0. CPU and Metal agree at blend 0.5
(centroid within 3 mm after two frames), and the smoke test scrubs a cached FLIP 0.5 run back
and restores it exactly.

### Pre-existing issue found: APIC affine reconstruction

G2P rebuilds each affine row as B·D⁻¹ with a determinant cutoff of 1e-8. With trilinear
weights D = diag(f(1−f)), which is singular whenever a particle sits on a face plane, and B·D⁻¹
is then a near-0/0 cancellation. The float64 CPU engine keeps those inverses; float32 Metal
mostly zeroes them. The result:

- On `pool_with_ramp.blend` the CPU engine's affine rows reach |C| ≈ 27,000 s⁻¹ (sane values are
  about 10), and a resting pool accelerates to 0.9 m/s. Metal stays at rest, but a Metal step
  fed the CPU state accelerates the same way. This, not PCG, is why the two engines disagree on
  that demo, and it predates this work: the Jacobi baseline above shows it too.
- In the sealed Taylor–Green box APIC's kinetic energy grows 34,000× in 40 steps, under Jacobi
  as well as PCG.

The gradient form C = Σ vᵢ∇wᵢ equals B·D⁻¹ wherever D is invertible (rigid round trips stay
exact to 3e-7) and is well defined where it is not. Prototyped on the CPU engine, it holds the
Taylor–Green vortex at the normal APIC retention (0.62 → 0.37 over 120 steps). It changes APIC
results, so it is left for its own change with a re-baked golden reference.

## Risks

- Mixed-precision drift between the engines grows with iteration count; PCG makes the two
  engines' rounding differences more visible than Jacobi did. Set tolerances from data.
- PCG breaks down on singular systems (a fully enclosed fluid region with no air cell). The
  usual fix is to pin one pressure or project out the mean; handle it explicitly and test a
  sealed box.
- A deterministic reduction is slower than an atomic one. Measure it; a few microseconds per
  dot product at Flow-X grid sizes is expected to be acceptable.

## Out of scope

Reseeding and remeshing, collider velocity transfer and two-way coupling, APIC viscosity and
surface tension, CUDA, and multi-domain interaction remain in the deferred list.
