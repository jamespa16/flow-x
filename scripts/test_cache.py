"""Standalone tests for the v4 particle-state cache.

Blender is replaced with the tiny scene/domain surface this module needs, so
the binary format and APIC continuation state stay covered on every CI host.
"""

import importlib.util
import struct
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "flowx_cache_test"


def _install_stubs():
    root = types.ModuleType(PACKAGE)
    root.__path__ = [str(ROOT)]
    solver = types.ModuleType(f"{PACKAGE}.solver")
    solver.__path__ = [str(ROOT / "solver")]
    sys.modules[PACKAGE] = root
    sys.modules[solver.__name__] = solver
    bpy = types.ModuleType("bpy")
    bpy_types = types.ModuleType("bpy.types")
    bpy_types.Operator = type("Operator", (), {})
    bpy.types = bpy_types
    bpy.path = types.SimpleNamespace(abspath=lambda value: value)
    bpy.data = types.SimpleNamespace(filepath="")
    bpy.context = types.SimpleNamespace(scene=None)
    sys.modules["bpy"] = bpy
    sys.modules["bpy.types"] = bpy_types

    collision = types.ModuleType(f"{PACKAGE}.collision")
    collision.mesh_fingerprint = lambda _name: b"mesh"
    sys.modules[collision.__name__] = collision

    domain = types.ModuleType(f"{PACKAGE}.domain")
    domain.find_domain = lambda _scene: None
    domain.world_bounds = lambda _domain: (Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 1.0))
    sys.modules[domain.__name__] = domain
    return bpy


class Vec:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class Settings:
    cache_enabled = True
    solver_method = "APIC"
    engine = "AUTO"
    resolution = 16
    fluid_level = 50.0
    rest_density = 1000.0
    collider_voxel_multiplier = 1.0
    surface_multiplier = 1.0
    max_substeps = 8
    max_particles = 16000
    apic_pressure_iterations = 40
    apic_vorticity_strength = 0.3
    pbf_relaxation = 100.0
    pbf_scorr_k = 0.1
    viscosity = 0.1
    surface_tension = 0.0
    pbf_iterations = 4
    show_surface = True
    surface_iso = 0.5
    show_whitewater = True
    whitewater_capacity = 3
    whitewater_spawn_rate = 100.0
    whitewater_trapped_air_weight = 1.0
    whitewater_wave_crest_weight = 1.0
    whitewater_kinetic_weight = 1.0
    whitewater_kinetic_reference_speed = 2.0
    whitewater_spray_speed_threshold = 1.0
    whitewater_bubble_trapped_threshold = 0.5
    whitewater_jitter_strength = 0.1
    whitewater_normal_offset = 0.01
    whitewater_spray_life_min = 0.5
    whitewater_spray_life_max = 1.5
    whitewater_foam_life_min = 1.0
    whitewater_foam_life_max = 3.0
    whitewater_bubble_life_min = 0.5
    whitewater_bubble_life_max = 2.0
    whitewater_drag = 1.0
    whitewater_buoyancy = 1.0


class Scene:
    def __init__(self):
        self.render = types.SimpleNamespace(fps=24, fps_base=1.0)
        self.frame_current = 1
        self.objects = []
        self.name = "CacheTest"


class Domain:
    def __init__(self, path):
        self.flowx_domain = Settings()
        self.flowx_domain.cache_path = str(path)


class Failure(AssertionError):
    pass


def check(condition, message):
    if not condition:
        raise Failure(message)


bpy = _install_stubs()
spec = importlib.util.spec_from_file_location(f"{PACKAGE}.solver.cache", ROOT / "solver/cache.py")
cache = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cache
spec.loader.exec_module(cache)


def _state(method, whitewater):
    state = {
        "positions": [(0.25, 0.5, 0.75, 1.0), (0.5, 0.25, 0.125, 1.0)],
        "velocities": [(1.0, -0.5, 0.25, 0.0), (-0.25, 0.5, -1.0, 0.0)],
    }
    if method == "pbf":
        state["densities"] = [997.5, 1002.5]
    if method == "apic":
        state["affine"] = [tuple(float(i) / 16.0 for i in range(12))] * 2
    if whitewater:
        state["whitewater_cursor"] = 2
        state["ww_positions"] = [
            (0.0, 0.0, 0.0, 0.0),
            (0.25, 0.5, 0.75, 1.0),
            (0.5, 0.25, 0.125, 0.5),
        ]
        state["ww_velkind"] = [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0, 2.0),
        ]
    return state


def test_v4_roundtrip(method, whitewater):
    with tempfile.TemporaryDirectory(prefix="flowx-cache-test-") as temp:
        path = Path(temp) / "state.flowx_cache"
        scene = Scene()
        domain = Domain(path)
        domain.flowx_domain.show_whitewater = whitewater
        bpy.context.scene = scene
        capacity = 3 if whitewater else 0
        expected = _state(method, whitewater)

        cache.open(scene, domain, 2, method, "cpu", capacity)
        cache.write_frame(2, expected, domain)
        expected_size = cache.header()["header_size"] + cache.header()["frame_size"]
        check(path.stat().st_size == expected_size, "cache frame size disagrees with its header")
        cache.close()

        cache.open(scene, domain, 2, method, "cpu", capacity)
        got = cache.try_load(2, scene, domain)
        check(got == expected, f"{method} cache state did not round-trip exactly")
        header = cache.header()
        check(header["method"] == method and header["device"] == "cpu", "identity was lost")
        cache.close()


def test_hash_uses_resolved_identity_and_active_method_settings():
    with tempfile.TemporaryDirectory(prefix="flowx-cache-test-") as temp:
        scene = Scene()
        domain = Domain(Path(temp) / "state.flowx_cache")

        apic = cache.config_hash(domain, scene, "apic", "cpu")
        domain.flowx_domain.pbf_iterations += 1
        check(
            cache.config_hash(domain, scene, "apic", "cpu") == apic,
            "an inactive PBF setting invalidated an APIC cache",
        )
        domain.flowx_domain.apic_pressure_iterations += 1
        check(
            cache.config_hash(domain, scene, "apic", "cpu") != apic,
            "an active APIC setting did not invalidate its cache",
        )

        pbf = cache.config_hash(domain, scene, "pbf", "cpu")
        domain.flowx_domain.apic_vorticity_strength += 0.1
        check(
            cache.config_hash(domain, scene, "pbf", "cpu") == pbf,
            "an inactive APIC setting invalidated a PBF cache",
        )
        domain.flowx_domain.viscosity += 0.1
        check(
            cache.config_hash(domain, scene, "pbf", "cpu") != pbf,
            "an active PBF setting did not invalidate its cache",
        )
        check(
            cache.config_hash(domain, scene, "pbf", "cpu")
            != cache.config_hash(domain, scene, "pbf", "metal"),
            "the resolved device was omitted from the cache hash",
        )


def test_whitewater_hash_layout_covers_capacity_drag_and_buoyancy():
    with tempfile.TemporaryDirectory(prefix="flowx-cache-test-") as temp:
        scene = Scene()
        domain = Domain(Path(temp) / "state.flowx_cache")
        base = cache.config_hash(domain, scene, "apic", "cpu")
        for name, delta in (
            ("whitewater_capacity", 1),
            ("whitewater_drag", 0.1),
            ("whitewater_buoyancy", 0.1),
        ):
            settings = Settings()
            setattr(settings, name, getattr(settings, name) + delta)
            domain.flowx_domain = settings
            check(
                cache.config_hash(domain, scene, "apic", "cpu") != base,
                f"{name} was omitted or mispacked in the cache hash",
            )


def test_v3_is_recreated():
    with tempfile.TemporaryDirectory(prefix="flowx-cache-test-") as temp:
        path = Path(temp) / "old.flowx_cache"
        path.write_bytes(struct.pack("<8sI", cache.MAGIC, 3) + b"old cache")
        scene = Scene()
        domain = Domain(path)
        bpy.context.scene = scene
        cache.open(scene, domain, 2, "pbf", "cpu", 0)
        check(cache.header()["format_version"] == 4, "v3 cache was not replaced")
        check(path.read_bytes()[8:12] == struct.pack("<I", 4), "v4 header was not written")
        cache.close()


def main():
    tests = (
        ("PBF v4 round-trip", lambda: test_v4_roundtrip("pbf", False)),
        ("APIC + whitewater v4 round-trip", lambda: test_v4_roundtrip("apic", True)),
        (
            "active-method and resolved-device hash",
            test_hash_uses_resolved_identity_and_active_method_settings,
        ),
        (
            "whitewater hash layout",
            test_whitewater_hash_layout_covers_capacity_drag_and_buoyancy,
        ),
        ("v3 is recreated", test_v3_is_recreated),
    )
    failures = 0
    for name, run in tests:
        try:
            run()
        except Failure as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
