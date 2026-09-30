# Flow-X APIC pressure solve and FLIP blending

Status: proposed. Follows `ROADMAP_APIC.md`, which delivered the MAC-grid APIC solver with a
weighted-Jacobi pressure solve and lists "FLIP blending and PCG pressure solving" as deferred.

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
