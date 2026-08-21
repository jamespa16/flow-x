"""Collider tagging and CPU-side voxelization (Phase 2)."""

import math

import bpy
import gpu
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, PointerProperty
from bpy.types import Object, Operator, PropertyGroup
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from ..domain import find_domain, world_bounds

# Safety cap on ray-cast bounces during the inside/outside parity test, so a
# non-manifold/non-closed collider mesh can't spin the voxelizer forever.
_MAX_RAY_BOUNCES = 64

_OVERLAY_COLOR = (1.0, 0.35, 0.1, 0.9)
_OVERLAY_POINT_SIZE = 4.0

_grids = {}
_draw_handle = None

# Union of every tagged collider's occupancy, in the same domain-resolution
# grid, kept ready for the Phase 5 solver to sample. Rebuilt whenever any
# collider's own grid changes rather than read from `_grids` per frame, so a
# multi-collider scene costs the solver one texture lookup, not several.
_solver_grid = {"texture": None, "voxel_size": 0.0, "dims": (1, 1, 1)}


class ColliderGrid:
    """Voxelized occupancy for one collider, sized to the domain's grid."""

    __slots__ = ("dims", "occupancy", "points", "gpu_buf")

    def __init__(self, dims, occupancy, points, gpu_buf):
        self.dims = dims
        self.occupancy = occupancy
        self.points = points
        self.gpu_buf = gpu_buf


class FlowXColliderSettings(PropertyGroup):
    is_collider: BoolProperty(
        name="Is Flow-X Collider",
        description="Marks this object as a Flow-X fluid collider",
        default=False,
    )


class FLOWX_OT_toggle_collider(Operator):
    """Toggle Flow-X collider tagging on the active object"""

    bl_idname = "flowx.toggle_collider"
    bl_label = "Toggle Fluid Collider"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context):
        obj = context.active_object
        settings = obj.flowx_collider
        settings.is_collider = not settings.is_collider

        domain = find_domain(context.scene)
        if settings.is_collider:
            if domain is None:
                settings.is_collider = False
                self.report(
                    {"WARNING"},
                    "No Flow-X fluid domain in scene; add one before tagging colliders.",
                )
                return {"CANCELLED"}
            _rebuild_grid(domain, obj)
            _warn_collisionless(self, context, domain, obj)
        else:
            _grids.pop(obj.name, None)
            if domain is not None:
                _rebuild_solver_grid(domain)

        _tag_viewports_redraw()
        return {"FINISHED"}


def _warn_collisionless(operator, context, domain, obj):
    """Report a collider that cannot possibly collide, as tagged.

    Both cases leave the tag on: the object may gain geometry or move into the
    domain later, and the depsgraph handler rebuilds the grid when it does. A
    silent tag that never collides is worse than a tag that says why.
    """
    depsgraph = context.evaluated_depsgraph_get()
    if len(obj.evaluated_get(depsgraph).data.polygons) == 0:
        operator.report(
            {"WARNING"},
            f"'{obj.name}' has no faces, so it will not collide until it gains geometry.",
        )
        return

    obj_lo, obj_hi = world_bounds(obj)
    dom_lo, dom_hi = world_bounds(domain)
    if (
        obj_lo.x >= dom_hi.x
        or obj_hi.x <= dom_lo.x
        or obj_lo.y >= dom_hi.y
        or obj_hi.y <= dom_lo.y
        or obj_lo.z >= dom_hi.z
        or obj_hi.z <= dom_lo.z
    ):
        operator.report(
            {"WARNING"},
            f"'{obj.name}' lies outside the domain bounds, so it will not "
            "collide until it overlaps the domain.",
        )


def occupied_count(obj_name):
    """Number of occupied voxels in obj_name's collider grid, or 0 if untracked."""
    grid = _grids.get(obj_name)
    return len(grid.points) if grid is not None else 0


def get_solver_grid():
    """(texture, voxel_size, dims) for the SPH solver's collider sampling.

    `texture` is None when there are no tagged colliders, or when the upload
    failed (e.g. no GL context in a headless run) - the solver treats that the
    same way, as "nothing to collide with".
    """
    return _solver_grid["texture"], _solver_grid["voxel_size"], _solver_grid["dims"]


def ensure_grids(scene=None):
    """Rebuild the voxel grids for tagged colliders that don't have one.

    The grids live in memory only, so after a Blender restart - or an
    extension disable/enable, e.g. from scripts/reload_on_save.py - tagged
    colliders come back with their tag but no grid until one of them next
    changes transform or geometry. The solver would then sample an empty
    collider grid and the fluid would pass straight through, so this is
    called at solver start to reconcile tags with grids.
    """
    scene = scene or bpy.context.scene
    domain = find_domain(scene)
    if domain is None:
        return
    origin, voxel_size, dims = _domain_grid_geometry(domain)
    if voxel_size <= 0.0:
        return
    for obj in scene.objects:
        if obj.type != "MESH" or not obj.flowx_collider.is_collider:
            continue
        grid = _grids.get(obj.name)
        if grid is None or grid.dims != dims:
            _rebuild_grid(domain, obj)


def _rebuild_solver_grid(domain):
    if not _grids:
        _solver_grid.update(texture=None, voxel_size=0.0, dims=(1, 1, 1))
        return

    origin, voxel_size, dims = _domain_grid_geometry(domain)
    nx, ny, nz = dims
    union = bytearray(nx * ny * nz)
    for grid in _grids.values():
        # A grid built against a stale domain resolution is skipped until its
        # own transform/geometry update rebuilds it at the current one.
        if grid.dims != dims:
            continue
        for idx, occupied in enumerate(grid.occupancy):
            if occupied:
                union[idx] = 1

    _solver_grid.update(texture=_upload_to_gpu(union, dims), voxel_size=voxel_size, dims=dims)


def _domain_grid_geometry(domain):
    """(origin, voxel_size, dims) for the domain's voxel grid, per its resolution."""
    lo, hi = world_bounds(domain)
    size = hi - lo
    longest = max(size.x, size.y, size.z)
    if longest <= 0.0:
        return lo, 0.0, (0, 0, 0)
    voxel_size = longest / domain.flowx_domain.resolution
    dims = tuple(max(1, round(axis / voxel_size)) for axis in (size.x, size.y, size.z))
    return lo, voxel_size, dims


def _index_range(obj_min, obj_max, origin, voxel_size, count):
    lo = max(0, math.floor((obj_min - origin) / voxel_size))
    hi = min(count, math.ceil((obj_max - origin) / voxel_size))
    return lo, hi


def _point_inside(bvh, point, direction, epsilon=1e-4):
    """Parity test: odd number of ray hits along `direction` means inside."""
    count = 0
    origin = point
    for _ in range(_MAX_RAY_BOUNCES):
        hit, _normal, _index, _dist = bvh.ray_cast(origin, direction)
        if hit is None:
            break
        count += 1
        origin = hit + direction * epsilon
    return count % 2 == 1


def _upload_to_gpu(occupancy, dims):
    # Blender's Python GPU API has no read-write storage buffer type (confirmed
    # while standing up the Phase 3 compute round-trip) - only uniform buffers,
    # samplers and images are bindable. So the occupancy grid uploads as a 3D
    # R32F image instead of a nonexistent GPUStorageBuf; a compute shader can
    # later sample it with imageLoad(...).r > 0.5.
    try:
        values = [float(v) for v in occupancy]
        buf = gpu.types.Buffer("FLOAT", [len(values)], values)
        return gpu.types.GPUTexture(dims, format="R32F", data=buf)
    except Exception as exc:
        # Best-effort so the CPU-side debug overlay keeps working regardless of
        # platform/context quirks (e.g. no GL context in --background runs).
        print(f"[flow-x] Skipping GPU collider texture upload: {exc}")
        return None


def _rebuild_grid(domain, obj):
    origin, voxel_size, dims = _domain_grid_geometry(domain)
    if voxel_size <= 0.0:
        _grids.pop(obj.name, None)
        return

    depsgraph = bpy.context.evaluated_depsgraph_get()
    try:
        bvh = BVHTree.FromObject(obj, depsgraph)
    except Exception as exc:
        # A mesh that cannot be built into a BVH (e.g. malformed geometry)
        # collides with nothing rather than raising on every transform update.
        print(f"[flow-x] Could not build a collider BVH for '{obj.name}': {exc}")
        bvh = None
    if bvh is None:
        _grids.pop(obj.name, None)
        _rebuild_solver_grid(domain)
        return

    # FromObject builds the tree in the object's *local* space, while the
    # voxel centers are world space - so the query ray is transformed into
    # local space. The transform is an invertible affine map, which sends the
    # ray to a ray, so the inside/outside parity of the hit count is exact.
    # (Skipping this is why a collider not sitting at the origin voxelized
    # to nothing: its world-space centers all landed outside the local tree.)
    try:
        inv_matrix = obj.matrix_world.inverted()
    except ValueError:
        # A singular transform (an axis scaled to zero) has no inverse, and
        # Blender raises on it rather than returning garbage. The mesh
        # voxelizes to nothing, like one with no faces, and keeps its tag in
        # case it is scaled back up - which re-triggers this rebuild.
        print(
            f"[flow-x] Cannot voxelize '{obj.name}': its transform is singular "
            "(an axis is scaled to zero)."
        )
        _grids.pop(obj.name, None)
        _rebuild_solver_grid(domain)
        return
    direction = (inv_matrix.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()

    obj_lo, obj_hi = world_bounds(obj)
    nx, ny, nz = dims
    i0, i1 = _index_range(obj_lo.x, obj_hi.x, origin.x, voxel_size, nx)
    j0, j1 = _index_range(obj_lo.y, obj_hi.y, origin.y, voxel_size, ny)
    k0, k1 = _index_range(obj_lo.z, obj_hi.z, origin.z, voxel_size, nz)

    occupancy = bytearray(nx * ny * nz)
    points = []
    for k in range(k0, k1):
        z = origin.z + (k + 0.5) * voxel_size
        for j in range(j0, j1):
            y = origin.y + (j + 0.5) * voxel_size
            for i in range(i0, i1):
                x = origin.x + (i + 0.5) * voxel_size
                if _point_inside(bvh, inv_matrix @ Vector((x, y, z)), direction):
                    occupancy[(k * ny + j) * nx + i] = 1
                    points.append(Vector((x, y, z)))

    gpu_buf = _upload_to_gpu(occupancy, dims)
    _grids[obj.name] = ColliderGrid(dims, occupancy, points, gpu_buf)
    _rebuild_solver_grid(domain)


def _tag_viewports_redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


@persistent
def _on_depsgraph_update(scene, depsgraph):
    domain = None
    for update in depsgraph.updates:
        obj = update.id
        if not isinstance(obj, bpy.types.Object) or obj.name not in _grids:
            continue
        if not (update.is_updated_transform or update.is_updated_geometry):
            continue
        if domain is None:
            domain = find_domain(scene)
            if domain is None:
                break
        _rebuild_grid(domain, obj)
        _tag_viewports_redraw()


def _draw_collider_grids():
    # The overlay's on/off is a domain setting: the grids exist in the
    # domain's space, and the domain panel is where the user tunes the run.
    # (Draw handlers only fire with a live viewport, so the context is valid.)
    domain = find_domain(bpy.context.scene)
    if domain is None or not domain.flowx_domain.show_collider_overlay:
        return
    if not _grids:
        return
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.point_size_set(_OVERLAY_POINT_SIZE)
    gpu.state.blend_set("ALPHA")
    shader.bind()
    shader.uniform_float("color", _OVERLAY_COLOR)
    for grid in _grids.values():
        if not grid.points:
            continue
        batch = batch_for_shader(shader, "POINTS", {"pos": grid.points})
        batch.draw(shader)
    gpu.state.blend_set("NONE")


_classes = (
    FlowXColliderSettings,
    FLOWX_OT_toggle_collider,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    Object.flowx_collider = PointerProperty(type=FlowXColliderSettings)
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
    global _draw_handle
    _draw_handle = bpy.types.SpaceView3D.draw_handler_add(
        _draw_collider_grids, (), "WINDOW", "POST_VIEW"
    )


def unregister():
    global _draw_handle
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, "WINDOW")
        _draw_handle = None
    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    _grids.clear()
    _solver_grid.update(texture=None, voxel_size=0.0, dims=(1, 1, 1))
    del Object.flowx_collider
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
