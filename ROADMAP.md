# flow-x Roadmap — MVP: GPU SPH Fluids for Blender

## MVP definition

A Blender add-on that lets a user, entirely from the UI:

1. Add a **Fluid Domain** object to the scene.
2. Set the **initial fluid level** as a percentage of the domain's height.
3. Tag arbitrary mesh objects as **Colliders** so the fluid interacts with them.
4. Press play and watch a GPU-simulated SPH fluid fall, settle, and respond to those colliders — rendered as an actual liquid **surface mesh**, not a particle cloud — in the viewport, in real time or near-real time.

Explicitly **out of scope for MVP**: photoreal surface shading (foam, whitecaps, refraction), viscosity/surface-tension tuning UI, baking to disk cache, multi-domain interaction, CPU fallback solver, GPU-side marching cubes. These become the post-MVP backlog.

## Architecture decisions (locked in for MVP)

| Decision | Choice | Why |
|---|---|---|
| Target Blender version | 5.2 LTS+ (raised from 4.2 LTS+ in v0.2.0, when the disk cache needed the string-form `permissions` manifest schema) | New Extensions platform (`blender_manifest.toml`), stable `gpu` compute API |
| Distribution format | Blender Extension (not legacy `bl_info` add-on) | Extensions is the only supported path going forward |
| GPU compute path | Blender's `gpu` module + GLSL compute shaders (`gpu.compute.dispatch`) | Runs inside Blender's existing GL context — no native build/toolchain per OS/GPU. A native CUDA/Vulkan backend is a post-MVP perf stretch goal, not required for MVP |
| Solver algorithm | WCSPH (weakly-compressible SPH) | Simplest SPH variant that's numerically stable enough to demo; PCISPH/PBF are later perf/quality upgrades |
| Neighbor search | Uniform grid spatial hash, built on GPU each step; cells sorted with a bitonic sort rather than a counting sort | Standard approach for GPU SPH; avoids CPU/GPU round-trips. The usual counting-sort build needs `imageAtomicAdd`, and image atomics don't compile on Blender's Metal backend — bitonic sort is pure compare-exchange, so it needs no atomics (see `solver/sph.py`) |
| Collision representation | Colliders voxelized into a signed-distance/occupancy grid, rebuilt when the collider mesh's world matrix or geometry changes | Point-vs-mesh raycasting per particle per frame is too slow; a grid lookup is O(1) per particle in the compute shader |
| Playback integration | `frame_change_pre` handler steps the solver; no disk caching yet | Gets a visible, scrubbable-ish result fastest; baking is a real feature to design properly later, not to rush into MVP |
| Surface extraction | GPU compute splats particle density onto a regular scalar grid (resolution decoupled from particle count) → grid read back to CPU → marching cubes each frame → live-updated `bmesh` "FluidSurface" mesh | A liquid-looking mesh is core to MVP, but full GPU marching cubes needs histopyramid/atomic-counter triangle compaction — a project of its own. CPU-side extraction on a modestly-sized grid is a known, bounded cost and ships an actual surface now; porting extraction to GPU is a targeted post-MVP perf upgrade, not a rewrite |
| Marching cubes implementation | Small self-contained implementation (standard 256-case lookup table) in the add-on, not `scikit-image` | Blender extensions shouldn't bundle heavy third-party deps just for one algorithm; the classic MC table is compact and well-understood |
| Visualization (debug only) | Particles as instanced point sprites during early phases (3–5), replaced by the surface mesh once Phase 6 lands | Point-cloud viz isolates "is the physics right" from "is the surface right" while bringing the solver up |

## Phase 0 — Project scaffolding

- Set up the extension package: `blender_manifest.toml`, `__init__.py`, module layout (`domain/`, `collision/`, `solver/`, `ui/`, `shaders/`).
- Register a minimal add-on that installs cleanly in Blender 4.2+ (no functionality yet).
- Decide on and set up a dev loop: symlinking the repo into Blender's extensions directory, a reload-on-save script, and a way to run Blender headless (`blender --background --python`) for smoke tests.
- Basic CI: lint (ruff/black), and a headless smoke test that installs the extension and calls `bpy.ops` for each operator added so far.

**Exit criteria:** `blender --background --python smoke_test.py` installs and enables the extension with zero errors, on a clean checkout.

## Phase 1 — Domain object

- `flowx.domain_add` operator: creates a cuboid mesh (or reuses selection bounds) tagged as a Domain via a custom `PropertyGroup` (`obj.flowx_domain`), analogous to how Mantaflow tags its domain.
- Domain properties: world-space bounds, voxel/particle resolution, and a placeholder `fluid_level` (0–100%, default e.g. 50%).
- Sidebar (N-panel) tab "Flow-X" with a domain settings panel that only shows when the active object is a Domain.
- Guard against multiple overlapping domains in MVP (single active domain is fine — simplifies solver bring-up).

**Exit criteria:** Add Domain from a menu/operator, see it in the viewport as a bounding cube, edit its resolution/level in the N-panel, values persist through save/reload.

## Phase 2 — Collision tagging

- `flowx.toggle_collider` operator + `obj.flowx_collider` PropertyGroup (boolean `is_collider`, plus later: friction/bounciness stubs, but keep MVP to just "on/off").
- UI: a toggle in the Object Properties tab (and/or N-panel) for any mesh object.
- CPU-side voxelization: for each tagged collider, rasterize its mesh (via `BVHTree` + the domain's voxel grid) into an occupancy or SDF buffer sized to the domain resolution. Rebuild only on transform/geometry change (dependency graph update handler), not every frame, unless the object is animated.
- Upload the collider grid to a GPU buffer the solver can read.

**Exit criteria:** Tagging/untagging an object updates a visible debug overlay of the voxelized collider grid inside the domain bounds.

## Phase 3 — GPU compute foundations

- Stand up the `gpu` compute shader plumbing: create/compile a trivial compute shader, allocate `GPUStorageBuf`s for particle position/velocity, dispatch it, and read results back — prove the round-trip works before any SPH math.
- Smoke-test shader: e.g. apply gravity and integrate positions for N particles, no neighbors, no collisions. This isolates "can we drive GPU compute from Python" from "is the SPH math correct."

**Exit criteria:** N particles (as simple point-cloud viz) visibly fall under gravity inside the domain bounds, driven entirely by a GPU compute dispatch per frame.

## Phase 4 — SPH solver core

- Particle seeding: fill particles on a jittered grid, respecting the domain's `fluid_level` percentage (seed particles only up to `level% * domain_height`). This is where the "set fluid level to X% of domain height" requirement actually lands.
- Uniform-grid spatial hash build (compute pass): bucket particles into cells sized ~2×smoothing radius.
- SPH passes as separate compute dispatches: density → pressure (Tait equation of state) → pressure+viscosity force → integrate (semi-implicit Euler) → boundary clamp to domain bounds.
- Expose smoothing radius, rest density, stiffness, viscosity as domain properties with sane defaults — don't build a full tuning UI yet, just enough to not be hardcoded.

**Exit criteria:** A block of fluid seeded at `fluid_level`% of the domain drops, splashes against the domain floor/walls, and settles into a roughly flat pool — visually recognizable as a liquid, not a gas of points.

## Phase 5 — Collision response

- Extend the integrate/boundary compute pass to sample the collider SDF/occupancy grid from Phase 2 and push particles out along the gradient (or nearest free cell), with a velocity reflection/damping term.
- Handle the common case first: a single static or simply-animated mesh (e.g. a ramp or a sphere dropped into the pool) — don't try to generalize to deforming/skinned colliders for MVP.

**Exit criteria:** Fluid visibly flows around/pools against a tagged collider object placed inside the domain; toggling the collider tag off makes particles pass through it.

## Phase 6 — Surface reconstruction (marching cubes)

- Density-splat compute pass: for each cell of a surface grid (its own resolution setting, independent of the simulation's particle/spatial-hash resolution), accumulate SPH-kernel-weighted particle mass to get a scalar field over the domain.
- Read the density grid back from GPU to CPU once per step and run a self-contained marching cubes implementation (standard 256-case edge table) to extract a triangle soup at a configurable iso-value.
- Rebuild a child mesh object (e.g. `<Domain>.FluidSurface`) from the extracted triangles each frame via `bmesh`, and assign it a simple default translucent/water-ish material so it reads as liquid without further tuning.
- Make sure the surface correctly wraps around tagged colliders (validates Phase 5's collision grid and this phase's splat/extraction together, not just particles-in-isolation).
- Expose surface grid resolution and iso-value as domain properties, defaulted sensibly; document the resolution/perf tradeoff.

**Exit criteria:** With the same "ball drops into a pool with a ramp collider" setup from Phase 5, the viewport shows a continuous surface mesh — not points — that deforms plausibly frame to frame and correctly envelopes the collider.

## Phase 7 — Playback & UX polish

- `frame_change_pre` handler steps the solver deterministically from frame 0 (re-seed on frame 0/negative scrub, step forward otherwise) — accept that arbitrary scrubbing backward won't be physically correct without caching, and say so in the UI (disable/warn rather than silently produce garbage).
- Play/pause/reset controls in the panel; a "reset simulation" operator that re-seeds particles at `fluid_level`.
- Basic performance readout (particle count, ms/step) to help users size their domain resolution sensibly.

**Exit criteria:** A user can open a fresh scene, add a Domain, set fluid level, tag two colliders, hit play, and get a stable-looking real-time simulation, rendered as a surface mesh, without touching Python.

## Phase 8 — MVP hardening & packaging

- Handle the obvious failure modes: no domain in scene, zero-volume domain, collider with zero faces, extension re-enable after Blender restart.
- Write install docs (`README.md`) and a couple of example `.blend` demo files.
- Tag `v0.1.0-mvp`.

**Exit criteria:** A third party can install the packaged extension from the repo's release, follow the README, and reproduce the "ball drops into a pool with a ramp collider" demo — surface mesh and all — unassisted.

## Post-MVP backlog (explicitly deferred)

- GPU-side marching cubes (histopyramid / atomic-counter triangle compaction), replacing the CPU readback path once it becomes the bottleneck.
- Surface shading beyond a flat default material: foam/whitecaps, refraction, spray particles.
- ~~Bake-to-disk caching so scrubbing/rendering doesn't require re-simulating from frame 0.~~ Shipped in v0.2.0: a per-frame write-through cache (`solver/cache.py`, Playback > Cache in the N-panel) that validates each loaded frame against a settings hash and per-frame collider-matrix fingerprints.
- PCISPH or Position-Based Fluids for stiffer, more incompressible behavior at lower iteration counts.
- Multi-domain and domain-to-domain interaction.
- Moving/deforming/skinned colliders (currently limited to rigid transform updates).
- Native compute backend (Vulkan/CUDA) if `gpu`-module compute proves to be a performance ceiling.
- Per-collider material properties (friction, stickiness), viscosity/surface tension UI, forces (wind, vortex).
