"""Debug point-cloud viewport overlay, shared by the solver's run modes.

Phases 3-5 look at particles directly; Phase 6 replaces this with the extracted
surface mesh. Only one run mode draws at a time, so the draw handler and its
point list live here rather than in either mode's module.
"""

import bpy
import gpu
from gpu_extras.batch import batch_for_shader

_COLOR = (0.2, 0.55, 1.0, 0.9)
_POINT_SIZE = 5.0

_points = []
_draw_handle = None


def set_points(points):
    global _points
    _points = points


def points():
    """The point cloud currently being drawn (read-only view, for tests)."""
    return _points


def tag_viewports_redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _draw():
    if not _points:
        return
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.point_size_set(_POINT_SIZE)
    gpu.state.blend_set("ALPHA")
    shader.bind()
    shader.uniform_float("color", _COLOR)
    batch = batch_for_shader(shader, "POINTS", {"pos": _points})
    batch.draw(shader)
    gpu.state.blend_set("NONE")


def enable():
    global _draw_handle
    if _draw_handle is None:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_VIEW")


def disable():
    global _draw_handle, _points
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, "WINDOW")
        _draw_handle = None
    _points = []
    tag_viewports_redraw()
