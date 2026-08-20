"""GPU compute plumbing and the WCSPH solver (Phases 3-6).

Phase 3 stands up the compute round-trip: allocate GPU-side particle state,
dispatch a compute shader from Python, and read the results back - proven
here with a trivial gravity/integrate shader, before any SPH math lands.
"""

import math
import random
from pathlib import Path

import bpy
import gpu
from bpy.app.handlers import persistent
from bpy.types import Operator
from gpu_extras.batch import batch_for_shader

from ..domain import find_domain, world_bounds

PARTICLE_COUNT = 512
LOCAL_GROUP_SIZE = 64
GRAVITY = -9.81

_SHADER_PATH = Path(__file__).resolve().parent.parent / "shaders" / "gravity_integrate.glsl"

_PARTICLE_COLOR = (0.2, 0.55, 1.0, 0.9)
_PARTICLE_POINT_SIZE = 5.0

_state = {
    "running": False,
    "shader": None,
    "positions_tex": None,
    "velocities_tex": None,
    "floor_z": 0.0,
    "points": [],
}

_draw_handle = None


def _compile_gravity_shader():
    info = gpu.types.GPUShaderCreateInfo()
    info.image(0, "RGBA32F", "FLOAT_1D", "positions_img")
    info.image(1, "RGBA32F", "FLOAT_1D", "velocities_img")
    info.push_constant("FLOAT", "dt")
    info.push_constant("FLOAT", "gravity")
    info.push_constant("FLOAT", "floor_z")
    info.push_constant("INT", "particle_count")
    info.compute_source(_SHADER_PATH.read_text())
    info.local_group_size(LOCAL_GROUP_SIZE)
    return gpu.shader.create_from_info(info)


def _seed_particles(domain, count):
    """Jittered grid of particles filling the upper ~35% of the domain."""
    lo, hi = world_bounds(domain)
    size = hi - lo
    side = math.ceil(count ** (1 / 3))
    rng = random.Random(0)

    positions = []
    for i in range(count):
        ix = i % side
        iy = (i // side) % side
        iz = i // (side * side)
        fx = (ix + 0.5) / side + rng.uniform(-0.02, 0.02)
        fy = (iy + 0.5) / side + rng.uniform(-0.02, 0.02)
        fz = (iz + 0.5) / side
        x = lo.x + fx * size.x
        y = lo.y + fy * size.y
        z = lo.z + size.z * (0.6 + 0.35 * fz)
        positions.append((x, y, z, 1.0))
    return positions, lo.z


def _flat_buffer(vectors):
    flat = [component for vec in vectors for component in vec]
    return gpu.types.Buffer("FLOAT", [len(flat)], flat)


def _tag_viewports_redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _start(domain):
    positions, floor_z = _seed_particles(domain, PARTICLE_COUNT)
    velocities = [(0.0, 0.0, 0.0, 0.0)] * PARTICLE_COUNT

    positions_tex = gpu.types.GPUTexture(
        PARTICLE_COUNT, format="RGBA32F", data=_flat_buffer(positions)
    )
    velocities_tex = gpu.types.GPUTexture(
        PARTICLE_COUNT, format="RGBA32F", data=_flat_buffer(velocities)
    )

    _state.update(
        {
            "running": True,
            "shader": _compile_gravity_shader(),
            "positions_tex": positions_tex,
            "velocities_tex": velocities_tex,
            "floor_z": floor_z,
            "points": [p[:3] for p in positions],
        }
    )

    if _on_frame_change not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(_on_frame_change)

    global _draw_handle
    if _draw_handle is None:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_particles, (), "WINDOW", "POST_VIEW"
        )
    _tag_viewports_redraw()


def _stop():
    global _draw_handle
    if _on_frame_change in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(_on_frame_change)
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, "WINDOW")
        _draw_handle = None
    _state.update(
        {
            "running": False,
            "shader": None,
            "positions_tex": None,
            "velocities_tex": None,
            "points": [],
        }
    )
    _tag_viewports_redraw()


def _step(dt):
    shader = _state["shader"]
    shader.bind()
    shader.uniform_float("dt", dt)
    shader.uniform_float("gravity", GRAVITY)
    shader.uniform_float("floor_z", _state["floor_z"])
    shader.uniform_int("particle_count", PARTICLE_COUNT)
    shader.image("positions_img", _state["positions_tex"])
    shader.image("velocities_img", _state["velocities_tex"])

    groups_x = math.ceil(PARTICLE_COUNT / LOCAL_GROUP_SIZE)
    gpu.compute.dispatch(shader, groups_x, 1, 1)

    # gpu.compute.dispatch() issues a texture-fetch memory barrier, so this
    # read-back is safe immediately after dispatch.
    rows = _state["positions_tex"].read().to_list()[0]
    _state["points"] = [tuple(row[:3]) for row in rows]


@persistent
def _on_frame_change(scene, _depsgraph):
    if not _state["running"]:
        return
    fps = scene.render.fps / scene.render.fps_base if scene.render.fps_base else 24.0
    _step(1.0 / fps if fps > 0 else 1.0 / 24.0)
    _tag_viewports_redraw()


def _draw_particles():
    points = _state["points"]
    if not points:
        return
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.point_size_set(_PARTICLE_POINT_SIZE)
    gpu.state.blend_set("ALPHA")
    shader.bind()
    shader.uniform_float("color", _PARTICLE_COLOR)
    batch = batch_for_shader(shader, "POINTS", {"pos": points})
    batch.draw(shader)
    gpu.state.blend_set("NONE")


class FLOWX_OT_solver_gpu_test_toggle(Operator):
    """Toggle the Phase 3 GPU compute round-trip smoke test (gravity only, point-cloud viz)"""

    bl_idname = "flowx.solver_gpu_test_toggle"
    bl_label = "Toggle GPU Compute Smoke Test"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return find_domain(context.scene) is not None

    def execute(self, context):
        if _state["running"]:
            _stop()
            self.report({"INFO"}, "Flow-X GPU smoke test stopped")
            return {"FINISHED"}

        domain = find_domain(context.scene)
        try:
            _start(domain)
        except Exception as exc:
            # No active GPU context (e.g. some --background runs) or a shader
            # compile failure shouldn't hard-crash the operator - report and
            # leave the smoke test off.
            _stop()
            self.report({"WARNING"}, f"Could not start GPU compute round-trip: {exc}")
            return {"FINISHED"}

        self.report({"INFO"}, f"Flow-X GPU smoke test running ({PARTICLE_COUNT} particles)")
        return {"FINISHED"}


def is_running():
    return _state["running"]


_classes = (FLOWX_OT_solver_gpu_test_toggle,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    _stop()
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
