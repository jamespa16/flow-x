"""Build the example .blend scenes shipped in demos/.

Each scene reproduces a roadmap setup end to end - a domain, a fluid level,
tagged colliders - so opening one and hitting play is the whole interaction.
The files are committed to the repo and bundled in the release zip; rebuild
them headless from the repo root with the extension linked
(scripts/dev_link.py):

    blender --background --python scripts/make_demo.py

The extension must be linkable, because tagging a domain or collider writes
the extension's custom properties into the file.
"""

import math
from pathlib import Path

import bpy

ADDON_MODULE = "bl_ext.user_default.flow_x"
DEMOS_DIR = Path(__file__).resolve().parent.parent / "demos"


def _require_extension():
    bpy.ops.extensions.repo_refresh_all()
    if ADDON_MODULE not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module=ADDON_MODULE)
    if ADDON_MODULE not in bpy.context.preferences.addons:
        raise RuntimeError(
            f"{ADDON_MODULE} is not installed; run " "`python3 scripts/dev_link.py` first and retry"
        )


def _new_scene(name):
    # Clear the scene by hand rather than reading factory settings: the
    # factory load also resets preferences, which un-enables the extension
    # and strips the flowx_* properties from the RNA.
    scene = bpy.context.scene
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for blocks in (bpy.data.meshes, bpy.data.materials, bpy.data.actions):
        for block in list(blocks):
            if block.users == 0:
                blocks.remove(block)
    scene.name = name
    scene.frame_start = 1
    scene.frame_end = 250
    return scene


def _add_domain(scene, center, size, fluid_level):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
    obj = bpy.context.active_object
    obj.name = "FluidDomain"
    obj.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.display_type = "WIRE"
    obj.flowx_domain.is_domain = True
    obj.flowx_domain.fluid_level = fluid_level
    return obj


def _add_box_collider(name, location, size, rotation=(0.0, 0.0, 0.0)):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = size
    obj.rotation_euler = rotation
    obj.flowx_collider.is_collider = True
    return obj


def _tag_collider(obj):
    obj.flowx_collider.is_collider = True


def build_pool_with_ramp():
    """The canonical demo: a ball drops into a pool with a ramp collider.

    The ball is itself a collider on a two-keyframe fall, which exercises
    the animated-collider grid rebuild as well as the static ramp.
    """
    scene = _new_scene("PoolAndRamp")
    _add_domain(scene, (0.0, 0.0, 1.0), (2.0, 2.0, 2.0), fluid_level=35.0)

    # Sloped slab: low end dipped into the seeded pool, high end above it.
    _add_box_collider(
        "Ramp", (0.35, 0.0, 0.55), (1.4, 0.6, 0.12), rotation=(0.0, math.radians(18.0), 0.0)
    )

    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.22, location=(0.0, 0.35, 1.45), segments=24, ring_count=16
    )
    ball = bpy.context.active_object
    ball.name = "Ball"
    _tag_collider(ball)
    ball.keyframe_insert(data_path="location", frame=1)
    ball.location = (0.0, 0.35, 0.45)
    ball.keyframe_insert(data_path="location", frame=45)

    path = DEMOS_DIR / "pool_with_ramp.blend"
    bpy.ops.wm.save_mainfile(filepath=str(path))
    print(f"[make_demo] wrote {path}")


def build_pool_with_obstacle():
    """A pool settling around two static colliders; no animation.

    The simplest possible first run: open, hit play, watch the seeded fluid
    drop onto the pillar and pool around the wall.
    """
    scene = _new_scene("PoolObstacle")
    _add_domain(scene, (0.0, 0.0, 1.0), (2.0, 2.0, 2.0), fluid_level=50.0)

    _add_box_collider("Pillar", (0.55, 0.0, 0.9), (0.5, 0.5, 1.8))
    _add_box_collider("Wall", (-0.5, 0.0, 0.25), (0.8, 1.6, 0.5))

    path = DEMOS_DIR / "pool_with_obstacle.blend"
    bpy.ops.wm.save_mainfile(filepath=str(path))
    print(f"[make_demo] wrote {path}")


def main():
    _require_extension()
    DEMOS_DIR.mkdir(exist_ok=True)
    build_pool_with_ramp()
    build_pool_with_obstacle()


if __name__ == "__main__":
    main()
