"""Headless smoke test for the Flow-X extension.

Verifies the extension installs and enables cleanly in Blender 4.2+, with
zero errors, on a clean checkout - the Phase 0 exit criterion. Later phases
should append their new operators to OPERATORS_TO_SMOKE_TEST below and this
script will exercise each one after enabling the extension.

Run from the repo root (after `python3 scripts/dev_link.py`):
    blender --background --python scripts/smoke_test.py
"""

import sys

import bpy

ADDON_MODULE = "bl_ext.user_default.flow_x"

# (operator idname, kwargs) pairs to run once the extension is enabled.
OPERATORS_TO_SMOKE_TEST = [
    ("flowx.domain_add", {}),
    ("mesh.primitive_cube_add", {"size": 0.5, "location": (0, 0, 0)}),
    ("flowx.toggle_collider", {}),
    ("flowx.solver_gpu_test_toggle", {}),
]


def _get_operator(idname):
    module_name, func_name = idname.split(".")
    return getattr(getattr(bpy.ops, module_name), func_name)


def main():
    bpy.ops.extensions.repo_refresh_all()

    bpy.ops.preferences.addon_enable(module=ADDON_MODULE)
    if ADDON_MODULE not in bpy.context.preferences.addons:
        raise RuntimeError(f"{ADDON_MODULE} did not appear in enabled addons after enable")
    print(f"[smoke_test] enabled {ADDON_MODULE}")

    for idname, kwargs in OPERATORS_TO_SMOKE_TEST:
        result = _get_operator(idname)(**kwargs)
        if "FINISHED" not in result:
            raise RuntimeError(f"{idname} returned {result}, expected FINISHED")
        print(f"[smoke_test] ran {idname} -> {result}")

    bpy.ops.preferences.addon_disable(module=ADDON_MODULE)
    if ADDON_MODULE in bpy.context.preferences.addons:
        raise RuntimeError(f"{ADDON_MODULE} still enabled after disable")
    print(f"[smoke_test] disabled {ADDON_MODULE}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        print("SMOKE TEST FAILED")
        sys.exit(1)
    else:
        print("SMOKE TEST PASSED")
        sys.exit(0)
