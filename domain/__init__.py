"""Fluid Domain object: bounds, resolution, fluid level (Phase 1)."""

import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty, PointerProperty
from bpy.types import Object, Operator, PropertyGroup
from mathutils import Vector

DEFAULT_SIZE = 2.0

# Standard cube: 8 corner verts, 6 quad faces (indices into the verts below).
_CUBE_VERTS = (
    (-1, -1, -1),
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, 1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, 1, 1),
)
_CUBE_FACES = (
    (0, 1, 2, 3),
    (4, 7, 6, 5),
    (0, 4, 5, 1),
    (1, 5, 6, 2),
    (2, 6, 7, 3),
    (4, 0, 3, 7),
)


class FlowXDomainSettings(PropertyGroup):
    is_domain: BoolProperty(
        name="Is Fluid Domain",
        description="Marks this object as a Flow-X fluid domain",
        default=False,
    )
    resolution: IntProperty(
        name="Resolution",
        description="Voxel/particle grid resolution along the domain's longest axis",
        default=32,
        min=4,
        max=256,
    )
    fluid_level: FloatProperty(
        name="Fluid Level",
        description="Initial fluid fill height, as a percentage of the domain's height",
        default=50.0,
        min=0.0,
        max=100.0,
        subtype="PERCENTAGE",
    )


def world_bounds(obj):
    """Return (min, max) corners of obj's bounding box, in world space."""
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def find_domain(scene):
    """Return the scene's existing Flow-X domain object, if any."""
    for obj in scene.objects:
        if obj.type == "MESH" and obj.flowx_domain.is_domain:
            return obj
    return None


def _selection_bounds(context):
    """Combined world-space bounds of selected mesh objects, or None."""
    objs = [o for o in context.selected_objects if o.type == "MESH"]
    if not objs:
        return None
    mins, maxs = [], []
    for obj in objs:
        lo, hi = world_bounds(obj)
        mins.append(lo)
        maxs.append(hi)
    lo = Vector((min(v.x for v in mins), min(v.y for v in mins), min(v.z for v in mins)))
    hi = Vector((max(v.x for v in maxs), max(v.y for v in maxs), max(v.z for v in maxs)))
    return lo, hi


class FLOWX_OT_domain_add(Operator):
    """Add a Flow-X fluid domain, sized to the current selection if any"""

    bl_idname = "flowx.domain_add"
    bl_label = "Add Fluid Domain"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        existing = find_domain(context.scene)
        if existing is not None:
            self.report(
                {"ERROR"},
                f"Scene already has a fluid domain ('{existing.name}'). Flow-X MVP "
                "supports a single domain - delete or un-tag it first.",
            )
            return {"CANCELLED"}

        bounds = _selection_bounds(context)
        if bounds is not None:
            lo, hi = bounds
            center = (lo + hi) / 2
            size = hi - lo
            size = Vector((max(size.x, 0.01), max(size.y, 0.01), max(size.z, 0.01)))
        else:
            center = context.scene.cursor.location.copy()
            size = Vector((DEFAULT_SIZE, DEFAULT_SIZE, DEFAULT_SIZE))

        mesh = bpy.data.meshes.new("FluidDomain")
        verts = [(x * size.x / 2, y * size.y / 2, z * size.z / 2) for x, y, z in _CUBE_VERTS]
        mesh.from_pydata(verts, [], _CUBE_FACES)
        mesh.update()

        obj = bpy.data.objects.new("FluidDomain", mesh)
        obj.location = center
        obj.display_type = "WIRE"
        obj.flowx_domain.is_domain = True
        context.collection.objects.link(obj)

        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj

        return {"FINISHED"}


_classes = (
    FlowXDomainSettings,
    FLOWX_OT_domain_add,
)


def _menu_func(self, context):
    self.layout.operator(FLOWX_OT_domain_add.bl_idname, text="Fluid Domain", icon="MOD_FLUID")


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    Object.flowx_domain = PointerProperty(type=FlowXDomainSettings)
    bpy.types.VIEW3D_MT_add.append(_menu_func)


def unregister():
    bpy.types.VIEW3D_MT_add.remove(_menu_func)
    del Object.flowx_domain
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
