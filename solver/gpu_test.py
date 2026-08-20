"""Phase 3 GPU compute round-trip smoke test.

Gravity and integration only - no neighbors, no pressure, no collisions. It
stays in the tree past Phase 4 because it isolates "can we drive GPU compute
from Python on this machine" from "is the SPH math right", which is the first
question to ask when the solver misbehaves on a new GPU or Blender build.
"""

import math
import random

import bpy
from bpy.app.handlers import persistent
from bpy.types import Operator

from ..domain import find_domain, world_bounds
from . import viz
from .gpu_util import build_compute_shader, dispatch_1d, make_texture, read_texture, shader_source

PARTICLE_COUNT = 512
LOCAL_GROUP_SIZE = 64
GRAVITY = -9.81

_IMAGES = (
    ("RGBA32F", "FLOAT_2D", "positions_img"),
    ("RGBA32F", "FLOAT_2D", "velocities_img"),
)
_PUSH_CONSTANTS = (
    ("FLOAT", "dt"),
    ("FLOAT", "gravity"),
    ("FLOAT", "floor_z"),
    ("INT", "particle_count"),
    ("INT", "tex_width"),
)

_state = {
    "running": False,
    "shader": None,
    "positions": None,
    "velocities": None,
    "floor_z": 0.0,
}


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
        positions.extend(
            (
                lo.x + fx * size.x,
                lo.y + fy * size.y,
                lo.z + size.z * (0.6 + 0.35 * fz),
                1.0,
            )
        )
    return positions, lo.z


def _start(domain):
    positions, floor_z = _seed_particles(domain, PARTICLE_COUNT)

    _state.update(
        {
            "running": True,
            "shader": build_compute_shader(
                [shader_source("gravity_integrate")],
                _IMAGES,
                _PUSH_CONSTANTS,
                LOCAL_GROUP_SIZE,
            ),
            "positions": make_texture(PARTICLE_COUNT, values=positions),
            "velocities": make_texture(PARTICLE_COUNT, values=[0.0] * (PARTICLE_COUNT * 4)),
            "floor_z": floor_z,
        }
    )

    viz.set_points([tuple(positions[i * 4 : i * 4 + 3]) for i in range(PARTICLE_COUNT)])
    viz.enable()
    if _on_frame_change not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(_on_frame_change)
    viz.tag_viewports_redraw()


def stop():
    if _on_frame_change in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(_on_frame_change)
    _state.update({"running": False, "shader": None, "positions": None, "velocities": None})
    viz.disable()


def _step(dt):
    from .gpu_util import TEXTURE_WIDTH

    shader = _state["shader"]
    shader.bind()
    shader.uniform_float("dt", dt)
    shader.uniform_float("gravity", GRAVITY)
    shader.uniform_float("floor_z", _state["floor_z"])
    shader.uniform_int("particle_count", PARTICLE_COUNT)
    shader.uniform_int("tex_width", TEXTURE_WIDTH)
    shader.image("positions_img", _state["positions"])
    shader.image("velocities_img", _state["velocities"])
    dispatch_1d(shader, PARTICLE_COUNT, LOCAL_GROUP_SIZE)

    # gpu.compute.dispatch() issues a texture-fetch memory barrier, so this
    # read-back is safe immediately after dispatch.
    viz.set_points([p[:3] for p in read_texture(_state["positions"], PARTICLE_COUNT)])


@persistent
def _on_frame_change(scene, _depsgraph):
    if not _state["running"]:
        return
    fps = scene.render.fps / scene.render.fps_base if scene.render.fps_base else 24.0
    _step(1.0 / fps if fps > 0 else 1.0 / 24.0)
    viz.tag_viewports_redraw()


def is_running():
    return _state["running"]


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
            stop()
            self.report({"INFO"}, "Flow-X GPU smoke test stopped")
            return {"FINISHED"}

        from . import sph

        sph.stop()

        domain = find_domain(context.scene)
        try:
            _start(domain)
        except Exception as exc:
            # No active GPU context (e.g. some --background runs) or a shader
            # compile failure shouldn't hard-crash the operator - report and
            # leave the smoke test off.
            stop()
            self.report({"WARNING"}, f"Could not start GPU compute round-trip: {exc}")
            return {"FINISHED"}

        self.report({"INFO"}, f"Flow-X GPU smoke test running ({PARTICLE_COUNT} particles)")
        return {"FINISHED"}
