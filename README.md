# Flow-X

GPU-simulated SPH fluids for Blender, rendered as a live liquid surface mesh.

Add a fluid domain, set a fill level, tag a few colliders, and hit play: a
weakly-compressible SPH (WCSPH) fluid falls, splashes, and settles on the
GPU, with a marching-cubes surface extracted every frame - no Mantaflow, no
baking, no Python required.

## Requirements

- **Blender 4.2 LTS or newer** (Flow-X is an [extension](https://docs.blender.org/manual/en/latest/advanced/extensions/addons.html), not a legacy add-on)
- **A GPU** the simulation runs entirely on the GPU through Blender's `gpu`
  compute API (OpenGL / Metal / Vulkan). The surface mesh is extracted on the
  CPU each frame - a bounded cost, but the fastest thing to lower when
  things get slow is `Surface Resolution`, not the physics.
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
the simulation; scrubbing **back** is impossible without a cache, so the
solver holds its last frame and says so in the panel - press **Reset** (or
return to the start frame) to re-seed and run again.

## Example scenes

Both are in `demos/` and are included in the release zip. Open one, hit
play, done.

| Scene | What it shows |
|---|---|
| `demos/pool_with_ramp.blend` | A ball drops into a pool with a ramp collider. The ball is itself a keyframed collider, so the collider grid is rebuilt while it falls. |
| `demos/pool_with_obstacle.blend` | A pool settling around a pillar and a half-wall; no animation, the simplest possible first run. |

## How it works (short version)

- **Solver.** WCSPH: density, Tait pressure, pressure+viscosity forces, and
  semi-implicit integration run as GPU compute dispatches, with substeps
  sized by a CFL limit (so a stiff setup degrades to slow motion instead of
  exploding).
- **Neighbors.** A uniform grid rebuilt every substep; particles are bucketed
  with a bitonic sort rather than a counting sort because image atomics don't
  compile on Blender's Metal backend.
- **Colliders.** CPU-voxelized into the domain's grid (BVH ray parity) and
  uploaded as a 3D texture the integrate pass samples; rebuilt when a
  collider's transform or geometry changes.
- **Surface.** Particles are splatted onto a scalar grid on the GPU, read
  back once per frame, and extracted with a self-contained marching-cubes
  implementation into the `<Domain>.FluidSurface` child mesh.
- **Deterministic.** Seeding uses a fixed RNG seed and the substep size comes
  only from the scene's frame rate, so the same timeline replays identically.

## Performance

The panel shows particles, substeps, and ms/step (averaged). Two knobs do
most of the work:

- **Smoothing Radius** (domain panel) - particles are seeded half a radius
  apart, so halving it roughly octuples the particle count; the solver
  coarsens it automatically if you exceed its budget.
- **Surface Resolution** (surface panel) - extraction cost grows with the
  *cube* of this. Raise it for a final look, not while setting up the shot.

If ms/step is above the scene's frame budget, the panel says so.

## Limitations (MVP)

- One domain per scene.
- No disk cache: playback is forward-only from the seed frame.
- Colliders are static or simply-animated rigid meshes; no deforming/skinned
  colliders. Zero-face or out-of-domain colliders are tagged but warn.
- A domain scaled to zero volume is refused, not simulated.
- The surface uses a flat water-ish material; no foam, spray, or refraction.
- A GPU context is required; there is no CPU fallback.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Could not start the SPH solver" warning on Run | No GPU context or a shader compile failure. Run the *GPU Compute* smoke test from the domain panel to isolate GPU compute from the SPH math. |
| Fluid passes through a collider | Check its voxel count in the Object Properties tab: 0 means no faces or no overlap with the domain. Move it in and/or check the mesh is closed. |
| Surface looks blocky | Raise *Surface Resolution* (cubic cost). |
| ms/step too high | Raise *Smoothing Radius* (fewer particles) and/or lower *Surface Resolution*. |
| Simpler-looking result after a Blender restart | Re-enable the extension, then Run - collider grids are rebuilt from the scene's tags on start. |

## Development

```
domain/       domain object, properties, add operator
collision/    collider tagging + CPU voxelization
solver/       gpu plumbing, WCSPH solver, marching cubes, surface
ui/           N-panel and Object Properties panels
shaders/      GLSL compute passes
scripts/      dev link, reload, smoke test, demo builder, packager
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
