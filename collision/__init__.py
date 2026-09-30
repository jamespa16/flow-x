"""Collider tagging and CPU-side voxelization (Phase 2)."""

import hashlib
import math
import struct
from array import array

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
_mesh_fingerprints = {}
_motion_baselines = {}
_draw_handle = None

# Union of every tagged collider's occupancy, in the domain's collider grid
# (its resolution times the collider voxel multiplier - same grid as the
# particles by default), kept ready for the Phase 5 solver to sample. Rebuilt
# whenever any collider's own grid changes rather than read from `_grids` per
# frame, so a multi-collider scene costs the solver one texture lookup, not
# several.
_solver_grid = {
    "buffer": None,
    "occupancy": None,
    "velocity": None,
    "voxel_size": 0.0,
    "dims": (1, 1, 1),
}

# Matrix inversion and multiplication of an unchanged animated transform can
# leave a few ulps of noise. Treat that as stationary so an animated collider
# does not switch the static upload layout on and off while it is at rest.
_VELOCITY_EPSILON = 1.0e-12
_MOTION_UNCHANGED = object()


def _reset_state():
    """Drop all in-memory collider state.

    The grids, fingerprints and solver grid are keyed by object name and hold
    no reference to the scene, so they outlive the file they were built for.
    Opening a new .blend does not re-run register(), so without this the
    previous file's colliders would carry into the new one - and, because the
    keys are names, a same-named object in the new file would inherit a
    collider it was never tagged with. Called from the load_post handler and
    from unregister.
    """
    _grids.clear()
    _mesh_fingerprints.clear()
    _motion_baselines.clear()
    _solver_grid.update(
        buffer=None,
        occupancy=None,
        velocity=None,
        voxel_size=0.0,
        dims=(1, 1, 1),
    )


class ColliderGrid:
    """Voxelized occupancy for one collider, sized to the domain's grid."""

    __slots__ = ("dims", "occupancy", "points", "velocities")

    def __init__(self, dims, occupancy, points, velocities=None):
        self.dims = dims
        self.occupancy = occupancy
        self.points = points
        # Sparse map from flat voxel index to a world-space velocity. Keeping
        # only occupied voxels avoids a second full grid per collider; the
        # merged solver field is dense because that is the kernel's indexing
        # contract.
        self.velocities = velocities


def _on_animation_flag_update(settings, context):
    """Reset stale motion immediately when Animated Collider is toggled."""
    obj = getattr(settings, "id_data", None)
    scene = getattr(context, "scene", None)
    if obj is None or scene is None:
        return
    reset_motion_baseline(obj_name=obj.name)
    domain = find_domain(scene)
    if domain is not None and getattr(obj.flowx_collider, "is_collider", False):
        _rebuild_grid(domain, obj, scene=scene)


class FlowXColliderSettings(PropertyGroup):
    is_collider: BoolProperty(
        name="Is Flow-X Collider",
        description="Marks this object as a Flow-X fluid collider",
        default=False,
    )
    is_animated: BoolProperty(
        name="Animated Collider",
        description="Rebuild this collider's voxel grid every frame from its "
        "animation, so a keyframed collider tracks its motion during playback "
        "and transfers one-way normal velocity to the fluid",
        default=False,
        update=_on_animation_flag_update,
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
            # A keyframed collider needs the per-frame rebuild to track its
            # motion, so pre-flag it when the object is animated; the user can
            # still turn it off (e.g. the animation is far outside the shot).
            if obj.animation_data is not None and obj.animation_data.action is not None:
                settings.is_animated = True
            _rebuild_grid(domain, obj)
            _warn_collisionless(self, context, domain, obj)
        else:
            _grids.pop(obj.name, None)
            _mesh_fingerprints.pop(obj.name, None)
            _motion_baselines.pop(obj.name, None)
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

    evaluated_obj = obj.evaluated_get(depsgraph)
    obj_lo, obj_hi = _world_bounds_with_matrix(evaluated_obj, evaluated_obj.matrix_world)
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


def mesh_fingerprint(obj_name):
    """sha256 over a collider's evaluated local-space mesh, or None if untracked.

    Computed at grid build time and stored alongside it, so asking is a dict
    lookup. Unlike the voxel grid - world-space, and rebuilt on every
    transform update, including an animated collider's motion each frame -
    this is transform-invariant: it changes only when the mesh itself is
    edited, so the disk cache can key collider geometry on it without an
    animated collider's rigid motion looking like an edit.
    """
    return _mesh_fingerprints.get(obj_name)


def get_solver_grid():
    """(buffer, voxel_size, dims) for the SPH solver's collider sampling.

    `buffer` begins with one float occupancy value per voxel and, when
    get_solver_velocity() is non-None, appends three velocity floats per
    voxel. It is None when there are no tagged colliders or when upload failed
    (no usable device, say). See get_solver_occupancy() and
    get_solver_velocity() for the host-side forms the CPU engine reads.
    """
    return _solver_grid["buffer"], _solver_grid["voxel_size"], _solver_grid["dims"]


def get_solver_occupancy():
    """The merged occupancy grid as raw bytes, for engines with no device.

    The same union get_solver_grid() uploads; the CPU engine indexes it
    directly rather than through a device buffer.
    """
    return _solver_grid["occupancy"]


def get_solver_velocity():
    """Return the merged flat float3 wall-velocity field, or ``None``.

    The returned :class:`array.array` has three float32 values per collider
    voxel in ``x, y, z`` order and the same z-major indexing as
    :func:`get_solver_occupancy`.  ``None`` means no occupied voxel currently
    has nonzero animated-collider motion; this is also what keeps static scenes
    on the original occupancy-only device allocation.
    """
    return _solver_grid["velocity"]


def reset_motion_baseline(scene=None, obj_name=None):
    """Forget animated-collider transform history and clear wall velocities.

    Cache loads, reseeds, backward timeline jumps and other discontinuities
    call this before installing a new collider state.  The next animated
    sample becomes a baseline and therefore contributes zero velocity.  When a
    scene is supplied, the already voxelized occupancy is uploaded again with
    the static layout so no stale motion is visible to the solver.

    ``obj_name`` optionally limits the reset to one collider.  The default
    resets all baselines, which is the safe operation for a timeline reset.
    """
    if obj_name is None:
        _motion_baselines.clear()
        names = set(_grids)
    else:
        _motion_baselines.pop(obj_name, None)
        names = {obj_name} if obj_name in _grids else set()
    for name in names:
        _grids[name].velocities = None
    if scene is None:
        scene = getattr(bpy.context, "scene", None)
    if scene is not None:
        domain = find_domain(scene)
        if domain is not None:
            _rebuild_solver_grid(domain)


def ensure_grids(scene=None):
    """Rebuild the voxel grids for tagged colliders that don't have one.

    The grids live in memory only, so after a Blender restart - or an
    extension disable/enable, e.g. from scripts/reload_on_save.py - tagged
    colliders come back with their tag but no grid until one of them next
    changes transform or geometry. The solver would then sample an empty
    collider grid and the fluid would pass straight through, so this is
    called at solver start to reconcile tags with grids - in both
    directions: it builds the missing grids and drops the stale ones (for
    objects that were deleted or untagged), so the solver never samples a
    collider that is no longer tagged in this scene.
    """
    scene = scene or bpy.context.scene
    domain = find_domain(scene)
    if domain is None:
        return
    origin, voxel_size, dims = _domain_grid_geometry(domain)
    if voxel_size <= 0.0:
        return
    tagged = [obj for obj in scene.objects if obj.type == "MESH" and obj.flowx_collider.is_collider]
    tagged_names = {obj.name for obj in tagged}
    stale = [name for name in _grids if name not in tagged_names]
    for name in stale:
        _grids.pop(name, None)
        _mesh_fingerprints.pop(name, None)
        _motion_baselines.pop(name, None)
    if stale:
        _rebuild_solver_grid(domain)
    for obj in tagged:
        grid = _grids.get(obj.name)
        if grid is None or grid.dims != dims:
            _rebuild_grid(domain, obj)


def rebuild_animated_grids(scene=None):
    """Rebuild the voxel grids of colliders flagged as animated, at the current frame.

    An animated collider changes transform through the timeline rather than an
    object edit, so the depsgraph-update rebuild path does not fire for it on
    each frame - its grid would freeze at the last interactively-rebuilt
    shape, and the fluid would collide with a stale collider. The solver calls
    this once per simulated frame so the fluid collides with the collider as
    it is on that frame.
    """
    scene = scene or bpy.context.scene
    domain = find_domain(scene)
    if domain is None:
        return
    objs = [
        obj
        for obj in scene.objects
        if obj.type == "MESH" and obj.flowx_collider.is_collider and obj.flowx_collider.is_animated
    ]
    if not objs:
        return
    # The solver calls this from a frame_change_pre handler, which runs before
    # the depsgraph has updated the base transforms for the new frame - force
    # the evaluation so each collider's matrix_world is this frame's transform.
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    # Rebuild all animated colliders first, then merge and upload exactly once
    # for this logical frame.  Sorting makes overlap resolution independent of
    # Blender collection/object iteration order.
    for obj in sorted(objs, key=lambda item: item.name):
        _rebuild_grid(domain, obj, upload=False, scene=scene)
    _rebuild_solver_grid(domain)


def _rebuild_solver_grid(domain):
    if not _grids:
        _solver_grid.update(
            buffer=None,
            occupancy=None,
            velocity=None,
            voxel_size=0.0,
            dims=(1, 1, 1),
        )
        return

    origin, voxel_size, dims = _domain_grid_geometry(domain)
    nx, ny, nz = dims
    union = bytearray(nx * ny * nz)
    velocity_sums = [0.0] * (nx * ny * nz * 3)
    velocity_counts = [0] * (nx * ny * nz)
    for name in sorted(_grids):
        grid = _grids[name]
        # A grid built against a stale domain resolution is skipped until its
        # own transform/geometry update rebuilds it at the current one.
        if grid.dims != dims:
            continue
        for idx, occupied in enumerate(grid.occupancy):
            if occupied:
                union[idx] = 1
                # Every occupying collider contributes to the deterministic
                # overlap average; static/first-sample colliders contribute
                # an explicit zero wall velocity.
                velocity_counts[idx] += 1
                velocity = grid.velocities.get(idx) if grid.velocities else None
                if velocity is not None:
                    base = idx * 3
                    velocity_sums[base] += velocity.x
                    velocity_sums[base + 1] += velocity.y
                    velocity_sums[base + 2] += velocity.z

    # Average every overlapping animated-collider contribution.  A dense
    # float3 field is emitted only when at least one averaged value is
    # materially nonzero; static scenes retain the old N-float upload.
    merged_velocity = array("f", [0.0]) * (nx * ny * nz * 3)
    has_motion = False
    for idx, count in enumerate(velocity_counts):
        if not count:
            continue
        base = idx * 3
        vx = velocity_sums[base] / count
        vy = velocity_sums[base + 1] / count
        vz = velocity_sums[base + 2] / count
        if max(abs(vx), abs(vy), abs(vz)) <= _VELOCITY_EPSILON:
            continue
        merged_velocity[base] = vx
        merged_velocity[base + 1] = vy
        merged_velocity[base + 2] = vz
        has_motion = True
    if not has_motion:
        merged_velocity = None

    # Both host forms are kept beside the packed device buffer. The upload is
    # best-effort, so on a machine with no device the CPU path still has both
    # occupancy and wall motion.
    _solver_grid.update(
        buffer=_upload_to_device(union, dims, merged_velocity),
        occupancy=union,
        velocity=merged_velocity,
        voxel_size=voxel_size,
        dims=dims,
    )


def _domain_grid_geometry(domain):
    """(origin, voxel_size, dims) for the domain's collider voxel grid.

    Sized to the domain's resolution times the collider voxel multiplier, so
    colliders sit on the simulation's own particle grid by default - the
    collider occupancy is only ever queried at particle positions, so a
    finer grid would buy collision detail the fluid cannot express, and a
    coarser one would let thin colliders vanish.
    """
    lo, hi = world_bounds(domain)
    size = hi - lo
    longest = max(size.x, size.y, size.z)
    if longest <= 0.0:
        return lo, 0.0, (0, 0, 0)
    settings = domain.flowx_domain
    grid_resolution = settings.resolution * settings.collider_voxel_multiplier
    voxel_size = longest / grid_resolution
    dims = tuple(max(1, round(axis / voxel_size)) for axis in (size.x, size.y, size.z))
    return lo, voxel_size, dims


def _world_bounds_with_matrix(obj, matrix):
    """Return world bounds using an explicitly evaluated world matrix."""
    corners = [matrix @ Vector(corner) for corner in obj.bound_box]
    xs = [corner.x for corner in corners]
    ys = [corner.y for corner in corners]
    zs = [corner.z for corner in corners]
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def _frame_seconds(scene):
    render = getattr(scene, "render", None)
    fps = float(getattr(render, "fps", 24.0))
    fps_base = float(getattr(render, "fps_base", 1.0))
    if fps <= 0.0 or fps_base <= 0.0:
        return 0.0
    return fps_base / fps


def _animated_voxel_velocities(obj, scene, current_matrix, occupied_points):
    """Return sparse current-voxel velocities and retain this sample baseline.

    A baseline is valid only for adjacent forward timeline frames.  This is
    intentionally stricter than using the elapsed frame count: a frame jump
    would otherwise turn a cache seek or a render catch-up into a teleporting
    wall impulse.  Singular matrices and invalid frame timing clear history so
    the next valid sample starts fresh.
    """
    if not getattr(obj.flowx_collider, "is_animated", False):
        return None
    name = obj.name
    frame = getattr(scene, "frame_current", None)
    try:
        frame = int(frame)
    except (TypeError, ValueError):
        frame = None
    previous = _motion_baselines.get(name)
    if (
        previous is not None
        and frame is not None
        and previous[1] == frame
        and previous[0] == current_matrix
    ):
        # Blender may report the same evaluated transform through both a
        # depsgraph update and the solver's explicit frame rebuild. Keep the
        # already computed velocity for that logical frame instead of erasing
        # it on the duplicate notification.
        return _MOTION_UNCHANGED
    _motion_baselines[name] = (current_matrix.copy(), frame)
    if previous is None or frame is None:
        return None
    previous_matrix, previous_frame = previous
    if previous_frame is None or frame != previous_frame + 1:
        return None
    elapsed = _frame_seconds(scene)
    if elapsed <= 0.0:
        return None
    try:
        current_inverse = current_matrix.inverted()
    except (ValueError, ZeroDivisionError):
        _motion_baselines.pop(name, None)
        return None
    velocities = {}
    for flat_index, point in occupied_points:
        previous_point = previous_matrix @ (current_inverse @ point)
        velocity = (point - previous_point) / elapsed
        if velocity.length_squared > _VELOCITY_EPSILON * _VELOCITY_EPSILON:
            velocities[flat_index] = velocity
    return velocities or None


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


def _upload_to_device(occupancy, dims, velocities=None):
    """Upload occupancy, optionally followed by a flat float3 field.

    It used to go up as a 3D R32F image, because Blender's GPU API had no
    read-write storage buffer type at all and images were the only mutable
    device state available. The solver owns its device now, so this is an
    ordinary buffer and the kernels index it with the z-major arithmetic in
    kernels/sph_common.h.

    Imported here rather than at module scope: solver/ imports this package, so
    a top-level import back into it would be circular.

    Best-effort - a machine with no usable device still gets the CPU-side
    debug overlay, and the solver treats None as "nothing to collide with".
    """
    from ..solver.backend import select

    try:
        backend = select()
        if backend is None:
            return None
        data = array("f", (float(v) for v in occupancy))
        if velocities is not None:
            data.extend(velocities)
        return backend.buffer(len(data) * 4, data)
    except Exception as exc:
        print(f"[flow-x] Skipping collider grid upload: {exc}")
        return None


def _compute_mesh_fingerprint(obj, depsgraph):
    """sha256 over an object's evaluated mesh in local space (vertices + faces)."""
    mesh = obj.evaluated_get(depsgraph).data
    digest = hashlib.sha256()
    for v in mesh.vertices:
        digest.update(struct.pack("<3f", *v.co))
    for p in mesh.polygons:
        digest.update(struct.pack(f"<{len(p.vertices)}I", *p.vertices))
    return digest.digest()


def _rebuild_grid(domain, obj, upload=True, scene=None):
    origin, voxel_size, dims = _domain_grid_geometry(domain)
    if voxel_size <= 0.0:
        _grids.pop(obj.name, None)
        _mesh_fingerprints.pop(obj.name, None)
        _motion_baselines.pop(obj.name, None)
        return

    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_obj = obj.evaluated_get(depsgraph)
    current_matrix = evaluated_obj.matrix_world.copy()
    _mesh_fingerprints[obj.name] = _compute_mesh_fingerprint(obj, depsgraph)
    try:
        bvh = BVHTree.FromObject(obj, depsgraph)
    except Exception as exc:
        # A mesh that cannot be built into a BVH (e.g. malformed geometry)
        # collides with nothing rather than raising on every transform update.
        print(f"[flow-x] Could not build a collider BVH for '{obj.name}': {exc}")
        bvh = None
    if bvh is None:
        _grids.pop(obj.name, None)
        _motion_baselines.pop(obj.name, None)
        if upload:
            _rebuild_solver_grid(domain)
        return

    # FromObject builds the tree in the object's *local* space, while the
    # voxel centers are world space - so the query ray is transformed into
    # local space. The transform is an invertible affine map, which sends the
    # ray to a ray, so the inside/outside parity of the hit count is exact.
    # (Skipping this is why a collider not sitting at the origin voxelized
    # to nothing: its world-space centers all landed outside the local tree.)
    try:
        inv_matrix = current_matrix.inverted()
    except (ValueError, ZeroDivisionError):
        # A singular transform (an axis scaled to zero) has no inverse, and
        # Blender raises on it rather than returning garbage. The mesh
        # voxelizes to nothing, like one with no faces, and keeps its tag in
        # case it is scaled back up - which re-triggers this rebuild.
        print(
            f"[flow-x] Cannot voxelize '{obj.name}': its transform is singular "
            "(an axis is scaled to zero)."
        )
        _grids.pop(obj.name, None)
        _motion_baselines.pop(obj.name, None)
        if upload:
            _rebuild_solver_grid(domain)
        return
    direction = (inv_matrix.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()

    obj_lo, obj_hi = _world_bounds_with_matrix(evaluated_obj, current_matrix)
    nx, ny, nz = dims
    i0, i1 = _index_range(obj_lo.x, obj_hi.x, origin.x, voxel_size, nx)
    j0, j1 = _index_range(obj_lo.y, obj_hi.y, origin.y, voxel_size, ny)
    k0, k1 = _index_range(obj_lo.z, obj_hi.z, origin.z, voxel_size, nz)

    occupancy = bytearray(nx * ny * nz)
    points = []
    occupied_points = []
    for k in range(k0, k1):
        z = origin.z + (k + 0.5) * voxel_size
        for j in range(j0, j1):
            y = origin.y + (j + 0.5) * voxel_size
            for i in range(i0, i1):
                x = origin.x + (i + 0.5) * voxel_size
                if _point_inside(bvh, inv_matrix @ Vector((x, y, z)), direction):
                    flat_index = (k * ny + j) * nx + i
                    point = Vector((x, y, z))
                    occupancy[flat_index] = 1
                    points.append(point)
                    occupied_points.append((flat_index, point))

    velocities = _animated_voxel_velocities(
        obj,
        scene=scene or bpy.context.scene,
        current_matrix=current_matrix,
        occupied_points=occupied_points,
    )
    if velocities is _MOTION_UNCHANGED:
        prior_grid = _grids.get(obj.name)
        velocities = prior_grid.velocities if prior_grid is not None else None

    # Only the merged solver grid is ever bound; a per-object upload would be
    # allocated and never read.
    _grids[obj.name] = ColliderGrid(dims, occupancy, points, velocities)
    if upload:
        _rebuild_solver_grid(domain)


def _tag_viewports_redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


@persistent
def _on_file_open(filepath):
    """Drop the previous file's collider state when a new file is opened.

    Must be @persistent: when Blender loads a file it resets its Python
    state and clears every handler that is not marked persistent, so a plain
    handler would be removed before it could ever observe the load.
    """
    _reset_state()


@persistent
def _on_depsgraph_update(scene, depsgraph):
    domain = None
    for update in depsgraph.updates:
        obj = update.id
        if not isinstance(obj, bpy.types.Object) or obj.name not in _grids:
            continue
        # A stale key (e.g. a same-named object from a file that is no longer
        # open) must not be (re)built into a collider for an object the user
        # never tagged; the prune in ensure_grids drops it at solver start.
        if not obj.flowx_collider.is_collider:
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
    if _on_file_open not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_file_open)
    if _on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
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
    if _on_file_open in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_file_open)
    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    _reset_state()
    del Object.flowx_collider
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
