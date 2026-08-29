# Flow-X

GPU-simulated PBF fluids for Blender, rendered as a live liquid surface mesh.

Add a fluid domain, set a fill level, tag a few colliders, and hit play: a
Position Based Fluids (PBF) simulation, with surface tension, falls,
splashes, and settles on the GPU, with a marching-cubes surface extracted
every frame - no Mantaflow, no baking, no Python required.

## Requirements

- **Blender 5.2 LTS or newer** (Flow-X is an [extension](https://docs.blender.org/manual/en/latest/advanced/extensions/addons.html), not a legacy add-on)
- **A GPU, for the fast path.** On Apple silicon the simulation runs on the
  GPU through Flow-X's own Metal helper, independent of Blender's `gpu`
  module. Without a usable GPU it falls back to a numpy CPU solver: the same
  physics, roughly ten times slower, which is fine for small sims and for
  checking a setup before committing to it. The `Engine` dropdown in the SPH
  Solver panel picks between them, and the panel says which one is running.
  The surface mesh is extracted on the CPU either way - a bounded cost, but
  the fastest thing to lower when things get slow is `Surface Resolution`, not
  the physics.
- A mesh collider should be a closed (manifold) mesh; the inside/outside
  test is a ray-parity count, so open meshes can voxelize to the wrong
  side.

## Installation

### From a release (recommended)

1. Download `flow_x-<version>.zip` from the
   [releases](https://github.com/jamespa16/flow-x/releases) page.
2. In Blender: **Edit > Preferences > Extensions** (the Extensions tab).
3. Click **Install from Disk** (top right) and select the zip, or drag the
   zip straight into the Extensions window.
4. In the **Local** repository, click **Enable** next to *Flow-X*.

### From source (development)

```sh
python3 scripts/dev_link.py          # symlink this repo into Blender's extensions dir
# restart Blender (or Preferences > Get Extensions > Refresh Local), then Enable
```

`scripts/reload_on_save.py` (run from Blender's Scripting tab) re-enables the
extension on every file save.

## Quick start

1. **Add a domain.** In the 3D viewport press **Shift+A > Fluid Domain**.
   With objects selected it sizes itself to their bounds; otherwise it's a
   2 m cube at the cursor.
2. **Set the fluid level.** Select the domain, open the **N-panel > Flow-X**
   tab, and set *Fluid Level* (percentage of the domain's height the fluid
   starts filled to).
3. **Tag colliders.** Select any mesh, open **Object Properties > Flow-X
   Collider** (the tab with the checkbox), and toggle it on. The tab shows
   the collider's voxel count; zero means it will not collide (no faces, or
   no overlap with the domain).
4. **Play.** N-panel > Flow-X > Playback > **Play**.

The solver steps one frame per timeline frame. Scrubbing **forward** steps
the simulation. Scrubbing **back** loads the frame from the disk cache when
[one is enabled](#caching); otherwise the solver holds its last frame and
says so in the panel - press **Reset** (or return to the start frame) to
re-seed and run again.

## Example scenes

Both are in `demos/` and are included in the release zip. Open one, hit
play, done.

| Scene | What it shows |
|---|---|
| `demos/pool_with_ramp.blend` | A ball drops into a pool with a ramp collider. The ball is itself a keyframed collider, so the collider grid is rebuilt while it falls. |
| `demos/pool_with_obstacle.blend` | A pool settling around a pillar and a half-wall; no animation, the simplest possible first run. |

## How it works (short version)

- **Solver.** Position Based Fluids: a particle predicts forward under
  gravity and a surface-tension cohesion force, then a fixed number of
  Jacobi constraint-solve iterations project it onto a density constraint,
  and its velocity is derived from how far that projection actually moved
  it - unconditionally stable for the constraint itself, so substeps are
  sized by a free-fall/diffusion limit rather than a stiffness-driven CFL
  limit.
- **Neighbors.** A uniform grid rebuilt once per substep (not per constraint
  iteration); particles are bucketed with a bitonic sort. That was originally
  forced - Blender's Metal backend could not compile an image atomic, so a
  counting sort was impossible - and is now simply what is there, since Flow-X
  runs its own Metal kernels where atomics work.
- **Colliders.** CPU-voxelized into the domain's grid (BVH ray parity) and
  uploaded as an occupancy buffer the finalize pass samples; rebuilt when a
  collider's transform or geometry changes. A keyframed collider is tagged
  with the *Animated Collider* option (Object Properties > Flow-X Collider),
  which rebuilds its grid every frame from its animation so the fluid tracks
  its motion.
- **Surface.** Particles are splatted onto a scalar grid on the GPU, read
  back once per frame, and extracted with a self-contained marching-cubes
  implementation into the `<Domain>.FluidSurface` child mesh.
- **Whitewater.** Each frame, every fluid particle is scored for how likely
  it is to throw off secondary spray/foam/bubble (a real-time-budget version
  of Ihmsen et al.'s trapped-air / wave-crest / kinetic-energy
  classification), the top scorers are spawned into a fixed pool that is
  advected with coarse per-kind motion, and the live pool is read back into
  the `<Domain>.Whitewater` point-cloud child - off by default.
- **Deterministic.** Seeding uses a fixed RNG seed and the substep size comes
  only from the scene's frame rate, so the same timeline replays to the same
  particle state (bit-for-bit, which is what the cache relies on).
- **Cache.** With *Cache to Disk* enabled, each frame's positions and
  velocities are appended to a binary file as the run goes, and a backward
  scrub loads the frame from it instead of re-simulating. A paired surface
  file stores the extracted mesh for the render path. A settings hash and a
  per-frame fingerprint of every collider's transform validate the file before
  anything is loaded from it.
- **Render.** Each rendered frame replays its cached surface and whitewater
  rather than re-extracting it, so a render matches the frames that were baked
  (see [Rendering](#rendering)).

## Performance

The panel shows particles, substeps, and ms/step (averaged). Two knobs do
most of the work:

- **Resolution** (domain panel) - particles are seeded one per voxel of this
  lattice, so doubling it roughly octuples the particle count; the solver
  coarsens the spacing automatically if you exceed the particle budget.
- **Max Particles** (solver panel) - the budget itself, with no hard ceiling
  (GPU memory is the limit). Raise it to hold a finer resolution at the cost
  of frame time, which grows super-linearly - slow, but for an offline render
  that trade is usually worth it.
- **Surface Multiplier** (surface panel) - sizes the extraction grid as a
  multiple of Resolution, so it tracks the sim automatically. Extraction cost
  grows with the *cube* of the result; pull it below 1.0 for a cheap surface
  under a high-res sim, raise it for a final look.
- **Whitewater** (whitewater panel) - off by default; when on it adds a
  per-frame score/sort/spawn/advect pass plus a point-cloud read-back on top
  of the core solve, so it's an additive cost. *Capacity* bounds the pool and
  *Spawn Rate* bounds how fast it fills.

If ms/step is above the scene's frame budget, the panel says so.

## Caching

By default Flow-X simulates forward only: scrubbing back through the
timeline holds the last simulated frame and says so in the panel. The
Playback panel's **Cache to Disk** option changes that. When enabled, every
simulated frame's particle state is written to a file as it runs, and
scrubbing back loads the frame from disk instead of re-simulating. The cache
also survives Blender restarts, and a forward jump lands directly on a frame
the cache already holds.

- **Location.** `<scene>.flowx_cache` next to the saved .blend file (the
  system temp dir until the scene is saved), or any file at *Cache Path*. A
  paired `<scene>.flowx_cache.mesh` holds the extracted surface and whitewater
  for the [render path](#rendering).
- **Size.** About 0.5 MB per frame for the particles at the default 16k
  budget; the surface file is larger (it stores the extracted mesh). The panel
  shows the frames covered and the running size of each.
- **Validity.** The file is keyed by a hash of everything that changes the
  simulation *or the extracted surface* - solver settings, domain bounds and
  resolution, collider geometry, frame rate, the surface and whitewater
  settings, this extension's version - plus a per-frame fingerprint of each
  collider's world transform. Change any of those and the next Reset starts a
  fresh file; until then the stale file warns instead of showing old state and
  stops growing, so frames simulated under the new settings never mix with the
  old run.
- **Clear Cache** deletes both files. A running simulation stops writing until
  the next Reset.

## Rendering

Blender's render owns the GPU, so the simulation cannot step while a render
runs (the compute context is dropped). Flow-X handles this by replaying the
baked surface instead of re-simulating it:

1. **Bake.** In the Playback panel, click **Bake Cache**. It re-seeds at the
   start frame and steps the whole frame range forward, writing every frame's
   particle state *and* extracted surface to the cache. The button becomes
   **Stop Baking** while it runs, so a long bake is cancellable.
2. **Render.** Leave the simulation running, then render the animation (or any
   range the bake covered). Each rendered frame replays its cached surface and
   whitewater on the CPU - no GPU compute - so the fluid animates correctly in
   the render rather than freezing on the seed frame.

A frame the bake did not cover warns in the panel instead of silently freezing
the surface. Because the surface and whitewater settings are part of the cache
hash, changing them invalidates the bake - re-bake before rendering the new
look.

## Limitations (MVP)

- One domain per scene.
- The disk cache is off by default; without it, playback is forward-only
  from the seed frame.
- The domain is static during a run: moving it mid-run leaves the surface
  mesh offset from the fluid until the next re-seed (Reset, or the playback
  loop). Colliders may move; the domain may not.
- Colliders are static or simply-animated rigid meshes; no deforming/skinned
  colliders. Zero-face or out-of-domain colliders are tagged but warn.
- A domain scaled to zero volume is refused, not simulated.
- The surface uses a flat water-ish material; there is no refraction.
  Whitewater spray/foam/bubble renders as a raw point cloud carrying `life`
  and `kind` attributes - a real spray/foam look is a Geometry Nodes
  modifier on top of those attributes, deliberately left for a follow-up.
- Rendering needs a [baked cache](#rendering). Re-extracting a surface is not
  bit-for-bit deterministic, so a freshly extracted frame can differ slightly
  from its baked twin - the render path avoids that by replaying the baked
  mesh.
- The GPU and CPU engines do not produce identical results, so switching
  engines invalidates a baked cache and it re-bakes.
- The Metal helper is a compiled dylib. A release downloaded from the internet
  carries a quarantine flag, and macOS will refuse to load it until it is
  signed and notarized; when that happens Flow-X falls back to the CPU engine
  and says so in the panel rather than failing to start.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Could not start the SPH solver" warning on Run | No usable engine at all. Run `python3 scripts/test_backend.py` to check GPU compute independently of the SPH math, and `python3 scripts/test_cpu_engine.py` for the fallback. Neither needs Blender. |
| Panel says `Engine: cpu` when you expected the GPU | The Metal helper is missing or would not load. Build it with `python3 scripts/build_native.py`; if it is a downloaded release, see the note under Limitations. |
| Scrubbing back holds and warns | Enable *Cache to Disk* in the Playback panel and play forward, or press *Reset*. |
| Fluid passes through a collider | Check its voxel count in the Object Properties tab: 0 means no faces or no overlap with the domain. Move it in and/or check the mesh is closed. |
| Surface looks blocky | Raise *Surface Resolution* (cubic cost). |
| ms/step too high | Raise *Smoothing Radius* (fewer particles) and/or lower *Surface Resolution*. |
| Simpler-looking result after a Blender restart | Re-enable the extension, then Run - collider grids are rebuilt from the scene's tags on start. |

## Development

```
domain/       domain object, properties, add operator
collision/    collider tagging + CPU voxelization
solver/       PBF solver, marching cubes, surface, whitewater, timeline
solver/backend/  device abstraction (buffers, kernels, a queue)
solver/engine/   the Metal and numpy simulation engines
ui/           N-panel and Object Properties panels
kernels/      Metal compute kernels + the shared portable prelude
native/       the Metal helper dylib's Objective-C++ source
bin/          the built helper (gitignored; scripts/build_native.py)
scripts/      build, dev link, reload, tests, demo builder, packager
demos/        shipped example scenes
```

- `scripts/smoke_test.py` - headless end-to-end test (run
  `blender --background --python scripts/smoke_test.py` after
  `scripts/dev_link.py`); replays every file in `demos/` and insists on a
  real surface mesh.
- `scripts/make_demo.py` - rebuilds the demo scenes.
- `scripts/package.py` - builds the release zip (also run by CI on tags).
- Lint/format: `ruff check .` and `black --check .` (see CI).

## License

MIT - see [LICENSE](LICENSE).
