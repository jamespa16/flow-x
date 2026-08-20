"""N-panel and Object Properties UI (Phase 1+)."""

import bpy
from bpy.types import Panel

from ..collision import FLOWX_OT_toggle_collider, occupied_count
from ..domain import world_bounds


class FLOWX_PT_domain(Panel):
    bl_label = "Fluid Domain"
    bl_idname = "FLOWX_PT_domain"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Flow-X"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and obj.flowx_domain.is_domain

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        settings = obj.flowx_domain

        layout.prop(settings, "resolution")
        layout.prop(settings, "fluid_level")

        lo, hi = world_bounds(obj)
        box = layout.box()
        box.label(text="World Bounds")
        col = box.column(align=True)
        col.label(text=f"Min: ({lo.x:.2f}, {lo.y:.2f}, {lo.z:.2f})")
        col.label(text=f"Max: ({hi.x:.2f}, {hi.y:.2f}, {hi.z:.2f})")


class FLOWX_PT_collider(Panel):
    bl_label = "Flow-X Collider"
    bl_idname = "FLOWX_PT_collider"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and not obj.flowx_domain.is_domain

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        is_collider = obj.flowx_collider.is_collider

        row = layout.row()
        icon = "CHECKBOX_HLT" if is_collider else "CHECKBOX_DEHLT"
        row.operator(
            FLOWX_OT_toggle_collider.bl_idname,
            text="Fluid Collider",
            icon=icon,
            depress=is_collider,
        )

        if is_collider:
            layout.label(text=f"Voxels: {occupied_count(obj.name)}")


_classes = (FLOWX_PT_domain, FLOWX_PT_collider)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
