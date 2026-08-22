# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Flow-X is a **Blender extension** (not a legacy `bl_info` add-on): the repo root *is* the
add-on package. `blender_manifest.toml` declares it, `__init__.py` registers
`domain`, `collision`, `solver`, `ui` in that order (unregister runs in reverse).
Minimum Blender is 5.2 LTS. Everything runs on the GPU through Blender's `gpu`
compute API; there is no CPU fallback solver.

## Commands

```sh
# lint / format (what CI runs)
ruff check .
black --check .

# marching-cubes tests — the only tests that run without Blender
python3 scripts/test_marching_cubes.py

# dev loop: symlink the checkout into Blender's user_default extensions repo,
# then restart Blender (or Preferences > Get Extensions > Refresh Local) and Enable
python3 scripts/dev_link.py [--blender-version 5.2]

# headless end-to-end test (requires dev_link first); replays every demos/*.blend
blender --background --python scripts/smoke_test.py

# same script against a packaged zip installed into a throwaway config
BLENDER_USER_RESOURCES=/tmp/blender-config FLOWX_ZIP=dist/flow_x-0.2.0.zip \
  blender --background --python scripts/smoke_test.py

python3 scripts/package.py [source_dir] [out_dir]   # release zip (default out: dist/)
python3 scripts/make_demo.py                        # rebuild demos/*.blend
```

`scripts/reload_on_save.py` runs *inside* Blender (Scripting tab) and re-enables the
extension whenever a `.py`/`.glsl` file changes.

CI (`.github/workflows/ci.yml`) is lint + the marching-cubes tests only. The Blender
smoke test lives in `release.yml` and runs on tag pushes / `workflow_dispatch`, since
it needs to download Blender.

## Architecture

The pipeline, per frame, is: `frame_change_pre` handler → N substeps of GPU compute →
one density splat + CPU marching cubes → rebuilt `<Domain>.FluidSurface` mesh.

- `domain/` — the Fluid Domain object, its `PropertyGroup` (`obj.flowx_domain`), and
  the add operator. One domain per scene; `find_domain`, `world_bounds`, `is_alive`,
  `is_degenerate` are the accessors everything else uses.
- `collision/` — `obj.flowx_collider` tagging plus CPU voxelization (BVH ray-parity
  inside/outside test) into a 3D occupancy texture sized to the domain grid. Every
  tagged collider's occupancy is merged into a single "solver grid" so the shader does
  one lookup regardless of collider count. Rebuilt on depsgraph transform/geometry
  change; *animated* colliders are rebuilt each frame by `rebuild_animated_grids`.
- `solver/sph.py` — the Position Based Fluids core and all timeline handling. Read its
  module docstring first; it is the design document for the solver.
- `solver/surface.py` + `solver/marching_cubes.py` — GPU splat, read-back, CPU
  iso-surface extraction. `marching_cubes.py` deliberately has **no Blender imports**
  so it can be tested standalone; keep it that way.
- `solver/cache.py` — the optional disk cache (see its docstring for the binary
  format and the config-hash validity rules).
- `solver/gpu_util.py` — shader compilation, texture helpers, and the GPU-API
  workarounds; read its docstring before touching any GPU plumbing.
- `shaders/` — one GLSL file per compute pass; `sph_common.glsl` is a prelude
  concatenated onto every SPH pass.

### Solver pass order (per substep)

`sph_normal` (only when surface tension > 0) → `sph_predict` → `sph_grid_key` →
`sph_sort` → `sph_cell_clear` → `sph_cell_range` → [`sph_lambda` → `sph_delta` →
`sph_apply_delta`] × iterations → `sph_velocity` → `sph_xsph` → `sph_finalize`
(collision + domain clamp). `_PASSES` in `solver/sph.py` lists them; adding a pass
means adding a `.glsl` file and an entry there.

## Constraints that will bite you

These are hard-won and non-obvious — violating them produces failures that point at
Blender's own headers, not at this code:

- **Push constants are full.** Every SPH pass declares the same 128-byte block
  (`_PUSH_CONSTANTS` in `solver/sph.py`, documented in `shaders/sph_common.glsl`).
  There is no room left. New parameters must be bit-packed into unused lanes of
  existing slots (see how `scorr_k` and `surface_tension` ride in `i_sort.zw`).
- **Do not declare all images on every pass.** Metal caps read-write textures at 8 per
  shader and keeps every declared slot; OpenGL eliminates unused ones. `_images_for_pass`
  scans the pass source and declares only what it mentions — plus `collider_img`, which
  the shared prelude references by name and therefore must always be declared.
- **Image atomics do not compile on Metal**, which is why the grid build uses a bitonic
  sort instead of a counting sort. Don't "optimize" it back.
- **1D textures cannot be read back** (`GPUTexture.read()` reports zero length), so all
  particle/grid state lives in 2D textures wrapped at `TEXTURE_WIDTH` (256).
- **`GPUShaderCreateInfo.image()` needs explicit `qualifiers`** or Metal generates MSL
  that fails to compile.
- **OpenGL drops push-constant/image slots a pass never reads**, so binding goes through
  the per-shader `missing` sets in `bind_push_constants`/`bind_image` rather than
  assuming a slot exists.
- **Determinism is a feature the cache depends on**: fixed RNG seed, substep count
  derived only from the scene frame rate. Anything that makes a run non-reproducible
  silently breaks cached scrubbing.
- The solver's run/reset operators are deliberately **not** `REGISTER|UNDO` — GPU state
  isn't on Blender's undo stack.
- Handlers must not be removed mid-dispatch; use `stop_deferred()` to tear down from
  inside a frame handler.

## Conventions

- Line length 100; ruff (`E,F,I,UP,B`) and black, both targeting py311.
- Blender naming: operators `FLOWX_OT_*` / `flowx.*`, panels `FLOWX_PT_*`.
- Comments in this codebase explain *why* — especially the "we tried the obvious thing
  and it doesn't work on Metal" cases. Match that density; don't strip them.
- Work lands directly on `main`.

## Docs

`README.md` is user-facing and current. `ROADMAP.md` is the original MVP plan and is
partly stale — notably it specifies WCSPH, which the solver has since been replaced
with PBF. Trust the code and `solver/sph.py`'s docstring over the roadmap.
