# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Flow-X is a **Blender extension** (not a legacy `bl_info` add-on): the repo root *is* the
add-on package. `blender_manifest.toml` declares it, `__init__.py` registers
`domain`, `collision`, `solver`, `ui` in that order (unregister runs in reverse).
Minimum Blender is 5.2 LTS.

Compute does **not** go through Blender's `gpu` module. Flow-X owns a Metal
device via a small Objective-C++ helper dylib (`native/`, built into `bin/`),
and falls back to a numpy CPU engine where that is unavailable. Blender's `gpu`
module is still used for *drawing* - the debug point cloud and the collider
overlay - and only for that.

## Commands

```sh
# lint / format (what CI runs)
ruff check .
black --check .

# build the Metal helper (needed for the GPU engine; not for the CPU one)
python3 scripts/build_native.py [--debug] [--check]

# the tests that run without Blender — all three are in CI
python3 scripts/test_marching_cubes.py   # iso-surface extraction
python3 scripts/test_backend.py          # device, kernels, atomics, dispatch order
python3 scripts/test_cpu_engine.py       # the numpy PBF, and its agreement with the GPU

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

# record a reference run, and check a later one against it. This is the gate
# for anything that touches the solver: bake a reference before the change,
# compare after.
blender --background demos/pool_with_ramp.blend --python scripts/golden.py -- \
    dump|compare <path.npz> [frames] [--whitewater] [--engine=cpu|metal]
```

`scripts/reload_on_save.py` runs *inside* Blender (Scripting tab) and re-enables the
extension whenever a `.py`/`.metal` file changes.

CI (`.github/workflows/ci.yml`) is lint plus the three Blender-free test scripts on
Linux, and a macOS job that builds the dylib and runs the two that need a device.
The Blender smoke test lives in `release.yml` and runs on tag pushes /
`workflow_dispatch`, since it needs to download Blender.

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
- `solver/backend/` — the device abstraction: buffers, a compiled kernel library, a
  recording command queue. Generic — it knows nothing about SPH — and free of `bpy`
  and `gpu` imports, which is what lets `scripts/test_backend.py` run without Blender.
  `metal.py` is ctypes bindings to `bin/libflowx_metal.dylib`; `native/flowx_metal.h`
  is the contract they must match.
- `solver/engine/` — the simulation engines. `metal_engine.py` dispatches the kernels;
  `cpu_engine.py` is a vectorized numpy PBF, used as the fallback *and* as the
  reference the GPU engine is checked against. Both satisfy one protocol, which is at
  pass-group level (`substep`, `splat_surface`, `step_whitewater`), not per-kernel —
  that is what lets the numpy engine exist without mirroring every kernel.
- `solver/sph.py` — timeline handling, the cache, the operators, and the resolved
  `SolverConfig`. Read its module docstring first; it is the design document for the
  solver. The per-substep pass chain lives in `metal_engine.py`.
- `solver/surface.py` + `solver/marching_cubes.py` — GPU splat, read-back, CPU
  iso-surface extraction. `marching_cubes.py` deliberately has **no Blender imports**
  so it can be tested standalone; keep it that way.
- `solver/cache.py` — the optional disk cache (see its docstring for the binary
  format and the config-hash validity rules).
- `kernels/` — one `.metal` file per pass. `flowx_prelude.h` holds the shared
  `Params` struct, the buffer binding table and the macros that keep the pass bodies
  compilable as CUDA later; `sph_common.h` holds the SPH helpers. There are no include
  paths when compiling MSL from source, so `solver/engine/kernels.py` concatenates
  them and strips the local `#include` lines.

### Solver pass order (per substep)

`sph_normal` (only when surface tension > 0) → `sph_predict` → `sph_grid_key` →
`sph_sort` → `sph_cell_clear` → `sph_cell_range` → [`sph_lambda` → `sph_delta` →
`sph_apply_delta`] × iterations → `sph_velocity` → `sph_xsph` → `sph_finalize`
(collision + domain clamp). `MetalEngine.substep` runs them and `SPH_PASSES` in
`solver/engine/kernels.py` lists them; adding a pass means a `.metal` file, an entry
there, and a line in `substep`.

A whole frame — every substep, then the surface splat, then whitewater — is recorded
into one command queue and submitted once. `record()` does no GPU work; `flush()` is
where the frame happens and where a read-back becomes valid.

## Constraints that will bite you

Most of what used to be in this section was Blender-GPU-API damage and is gone with
it. What remains is real:

- **`kernels/flowx_prelude.h`'s `Params` and `solver/engine/params.py` must agree.**
  Field order and types, exactly. Nothing checks it at runtime; `scripts/test_cpu_engine.py`
  has no opinion, but `kernels/flowx_params_probe.metal` reports the struct's real size
  and offsets and the backend test compares them. A drifted field silently corrupts
  every parameter after it, and the symptom is "the fluid behaves oddly".
- **`native/flowx_metal.h` is the ABI.** `solver/backend/metal.py` mirrors it by hand.
  Bump `FLOWX_ABI_VERSION` on any layout change — the Python side refuses to load a
  dylib that disagrees, which is what stops a stale `bin/` being read as garbage.
- **Nothing may throw across the C boundary.** An Objective-C exception unwinding into
  CPython takes Blender with it, so every entry point in `flowx_metal.mm` is wrapped
  and returns a code plus a message.
- **`flowx_submit` encodes into one serial-dispatch encoder.** That is what makes each
  recorded dispatch see the previous one's writes, which the whole pass chain assumes.
  Do not switch it to `MTLDispatchTypeConcurrent`. `scripts/test_backend.py` guards this.
- **A kernel must outlive nothing.** `Kernel` holds its `Program`, because releasing a
  program frees the pipelines it handed out and the failure surfaces later, inside
  submit, as a null kernel.
- **`solver/backend/`, `solver/engine/params.py` and `solver/engine/cpu_engine.py` must
  not import `bpy`** (nor `marching_cubes.py`) — three CI test scripts depend on it.
- **Image atomics used to be impossible**, which is why the grid build is a bitonic
  sort and whitewater uses a ring buffer instead of compaction. Atomics work now; those
  two are simply not yet rewritten. If you rewrite them, re-bake a `scripts/golden.py`
  reference first — results will change, deliberately, and you want to see how.
- **Determinism is a feature the cache depends on**: fixed RNG seed, substep count
  derived only from the scene frame rate. Anything that makes a run non-reproducible
  silently breaks cached scrubbing. The engine's identity is part of the cache's config
  hash, because the two engines do not agree bit for bit.
- The solver's run/reset operators are deliberately **not** `REGISTER|UNDO` — device
  state isn't on Blender's undo stack.
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
