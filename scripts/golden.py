"""Record a reference run, and compare a later run against it.

This exists for the backend migration: the solver's output has to be checked
against how it behaved on Blender's own GPU module *after* that module is
gone, so the reference has to be captured first and compared later.

    blender --background <scene.blend> --python scripts/golden.py -- \
        dump|compare <path.npz> [frames] [--whitewater] [--engine=cpu|metal]

`--engine=` pins the run to one engine instead of letting it pick. A dump
records which engine produced it, and a comparison against a reference from a
*different* engine drops the strict per-particle window automatically: the two
engines are not expected to agree particle by particle (see
solver/engine/cpu_engine.py), only to describe the same fluid.

`--whitewater` turns the domain's whitewater on for the run. The shipped demo
scenes leave it off (it is off by default - see domain/__init__.py), so without
this the reference covers the fluid and the surface but never exercises the
spawn/sort/advect passes at all.

Comparison is deliberately two-tier, because bit-exactness is not achievable
and demanding it would make the tool useless:

* Early frames are compared per particle. A port changes which `pow` and which
  fast-math the compiler picks, so the tolerance is loose, but for the first
  handful of frames the trajectories have not had time to separate and a real
  algorithmic difference shows up immediately.
* Later frames are compared as aggregates - bounds, speed distribution, a
  coarse occupancy histogram, surface/whitewater counts. PBF is chaotic, so two
  runs that differ in the last bits *will* diverge per particle while still
  describing the same fluid. The aggregates are what "the same fluid" means.

Determinism (fixed RNG seed, substep count derived only from the frame rate -
see solver/sph.py's docstring) is what makes the reference reproducible at all.
"""

import sys
from pathlib import Path

import bpy
import numpy as np

ADDON = "bl_ext.user_default.flow_x"

DEFAULT_FRAMES = 24

# Frames compared per particle before the comparison falls back to aggregates.
STRICT_FRAMES = 8

# Per-particle position tolerance over the strict window, in metres. Generous
# on purpose: it is sized to catch a wrong kernel or a wrong pass order, not to
# police the last bits of a float.
STRICT_ATOL = 1e-3

# Aggregate tolerances. Bounds and speeds are compared in absolute terms; the
# occupancy histogram is compared as a fraction of the particle count.
BOUNDS_ATOL = 0.05
SPEED_RTOL = 0.10
HIST_FRAC = 0.05

# The tolerances above are sized for two runs of the *same* engine, where the
# only source of difference is float non-determinism. Comparing two different
# engines is a weaker claim - they solve the same physics, not the same
# arithmetic - so the shape tolerance is relaxed. Measured cross-engine drift
# on the demo scenes is ~0.06 after 24 frames; 0.15 leaves room for the run to
# be chaotic without letting a real disagreement through.
CROSS_ENGINE_HIST_FRAC = 0.15
CROSS_ENGINE_SPEED_RTOL = 0.20

# Edge of the coarse occupancy histogram, per axis.
HIST_BINS = 8


def _enable():
    import addon_utils  # noqa: F401  (import side effect: registers the repo)

    bpy.ops.preferences.addon_enable(module=ADDON)
    import gpu

    # Background Blender has no window to hang a GPU context off; 5.0+ can
    # bring one up explicitly. Without it every compute dispatch fails.
    if hasattr(gpu, "init"):
        gpu.init()


def _solver():
    from bl_ext.user_default.flow_x.solver import sph, surface, whitewater

    return sph, surface, whitewater


def _run(frames, want_whitewater=False, engine=None):
    """Seed the scene's domain and step `frames` frames, yielding per-frame data."""
    sph, surface, whitewater = _solver()
    scene = bpy.context.scene

    if engine is not None:
        from bl_ext.user_default.flow_x.domain import find_domain

        domain = find_domain(scene)
        if domain is None:
            raise SystemExit("--engine: scene has no Flow-X domain")
        domain.flowx_domain.engine = engine.upper()

    if want_whitewater:
        from bl_ext.user_default.flow_x.domain import find_domain

        domain = find_domain(scene)
        if domain is None:
            raise SystemExit("--whitewater: scene has no Flow-X domain")
        # Set before the run starts: solver/__init__.py reads show_whitewater
        # when it decides whether to bring the pool up, not per frame.
        domain.flowx_domain.show_whitewater = True

    bpy.ops.flowx.sph_toggle()
    if not sph.is_running():
        raise SystemExit("solver did not start - no domain, or no GPU context")

    start = scene.frame_start
    scene.frame_set(start)
    for i in range(frames):
        scene.frame_set(start + 1 + i)
        state = sph.read_state()
        if state is None:
            raise SystemExit(f"solver stopped at frame {start + 1 + i}")
        positions, velocities = state
        mesh = surface.last_mesh()
        points = whitewater.last_points()
        yield {
            "pos": np.asarray([p[:3] for p in positions], dtype=np.float32),
            "vel": np.asarray([v[:3] for v in velocities], dtype=np.float32),
            "surface_verts": len(mesh[0]) if mesh else 0,
            "surface_tris": len(mesh[1]) if mesh else 0,
            "whitewater": len(points) if points else 0,
        }


def dump(path, frames, want_whitewater=False, engine=None):
    out = {}
    for i, frame in enumerate(_run(frames, want_whitewater, engine)):
        out[f"pos_{i}"] = frame["pos"]
        out[f"vel_{i}"] = frame["vel"]
        out[f"counts_{i}"] = np.asarray(
            [frame["surface_verts"], frame["surface_tris"], frame["whitewater"]], dtype=np.int64
        )
    out["frames"] = np.asarray([frames], dtype=np.int64)
    # Recorded so a later comparison knows whether a per-particle check is
    # meaningful or whether only the aggregates are.
    out["engine"] = np.asarray(_engine_name(), dtype="U32")
    np.savez_compressed(path, **out)
    print(f"golden: wrote {frames} frames to {path}")


def _engine_name():
    """The engine the running solver actually chose, e.g. "metal" or "cpu"."""
    sph, _surface, _whitewater = _solver()
    stats = sph.stats()
    return (stats["engine"].split(" ")[0] if stats else "unknown") if stats else "unknown"


def _aggregates(pos, vel):
    """The frame summary two runs of the same fluid must agree on."""
    speed = np.linalg.norm(vel, axis=1)
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    # Histogram over the frame's own bounding box rather than the domain: it is
    # the *shape* of the distribution being compared, and a box that tracks the
    # fluid keeps the bins meaningful as it spreads out.
    span = np.maximum(hi - lo, 1e-6)
    idx = np.clip(((pos - lo) / span * HIST_BINS).astype(np.int64), 0, HIST_BINS - 1)
    flat = (idx[:, 0] * HIST_BINS + idx[:, 1]) * HIST_BINS + idx[:, 2]
    hist = np.bincount(flat, minlength=HIST_BINS**3)
    return {
        "lo": lo,
        "hi": hi,
        "centroid": pos.mean(axis=0),
        "speed_mean": float(speed.mean()),
        "speed_max": float(speed.max()),
        "hist": hist,
    }


def compare(path, frames, want_whitewater=False, engine=None):
    ref = np.load(path)
    available = int(ref["frames"][0])
    if frames > available:
        print(f"golden: reference holds {available} frames, comparing that many")
        frames = available
    reference_engine = str(ref["engine"]) if "engine" in ref else "unknown"

    failures = []
    strict_frames = STRICT_FRAMES
    hist_frac, speed_rtol = HIST_FRAC, SPEED_RTOL
    for i, frame in enumerate(_run(frames, want_whitewater, engine)):
        if i == 0:
            running = _engine_name()
            if reference_engine not in ("unknown", running):
                print(
                    f"golden: reference is from '{reference_engine}', this run is "
                    f"'{running}' - comparing aggregates only"
                )
                strict_frames = 0
                hist_frac, speed_rtol = CROSS_ENGINE_HIST_FRAC, CROSS_ENGINE_SPEED_RTOL
        pos, vel = frame["pos"], frame["vel"]
        ref_pos, ref_vel = ref[f"pos_{i}"], ref[f"vel_{i}"]

        if pos.shape != ref_pos.shape:
            failures.append(f"frame {i}: particle count {pos.shape[0]} != {ref_pos.shape[0]}")
            break

        if i < strict_frames:
            worst = float(np.abs(pos - ref_pos).max())
            status = "ok" if worst <= STRICT_ATOL else "FAIL"
            print(f"frame {i:3d} strict   max|dp| = {worst:.3e}  {status}")
            if worst > STRICT_ATOL:
                failures.append(f"frame {i}: max position delta {worst:.3e} > {STRICT_ATOL:.0e}")
            continue

        got, want = _aggregates(pos, vel), _aggregates(ref_pos, ref_vel)
        bounds = max(
            float(np.abs(got["lo"] - want["lo"]).max()),
            float(np.abs(got["hi"] - want["hi"]).max()),
            float(np.abs(got["centroid"] - want["centroid"]).max()),
        )
        speed = abs(got["speed_mean"] - want["speed_mean"]) / max(want["speed_mean"], 1e-6)
        drift = float(np.abs(got["hist"] - want["hist"]).sum()) / max(pos.shape[0], 1) / 2.0
        bad = bounds > BOUNDS_ATOL or speed > speed_rtol or drift > hist_frac
        print(
            f"frame {i:3d} aggregate bounds={bounds:.4f} speed={speed:.3f} "
            f"shape={drift:.3f}  {'FAIL' if bad else 'ok'}"
        )
        if bad:
            failures.append(f"frame {i}: bounds {bounds:.4f}, speed {speed:.3f}, shape {drift:.3f}")

        ref_counts = ref[f"counts_{i}"]
        got_counts = (frame["surface_verts"], frame["surface_tris"], frame["whitewater"])
        if ref_counts[0] and abs(got_counts[0] - ref_counts[0]) > 0.25 * ref_counts[0]:
            failures.append(f"frame {i}: surface vertices {got_counts[0]} vs {int(ref_counts[0])}")

    if failures:
        print(f"\ngolden: FAILED ({len(failures)})")
        for line in failures:
            print("  -", line)
        raise SystemExit(1)
    print(f"\ngolden: matched {frames} frames")


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(argv) < 2:
        raise SystemExit(__doc__)
    want_whitewater = "--whitewater" in argv
    engine = next((a.split("=", 1)[1] for a in argv if a.startswith("--engine=")), None)
    argv = [a for a in argv if not a.startswith("--")]
    mode, path = argv[0], Path(argv[1]).resolve()
    frames = int(argv[2]) if len(argv) > 2 else DEFAULT_FRAMES

    _enable()
    if mode == "dump":
        dump(path, frames, want_whitewater, engine)
    elif mode == "compare":
        compare(path, frames, want_whitewater, engine)
    else:
        raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
