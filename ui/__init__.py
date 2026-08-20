"""N-panel and Object Properties UI (Phase 1+)."""

import bpy
from bpy.types import Panel

from ..collision import FLOWX_OT_toggle_collider, occupied_count
from ..domain import world_bounds
from ..solver import (
    PARTICLE_COUNT,
    FLOWX_OT_solver_gpu_test_toggle,
    FLOWX_OT_sph_reset,
    FLOWX_OT_sph_toggle,
    gpu_test,
    sph,
)


def _scene_fps(scene):
    return scene.render.fps / scene.render.fps_base if scene.render.fps_base else 24.0


def _wrap(text, width):
    """Break text into label-sized lines; Blender labels don't wrap themselves."""
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if len(candidate) > width and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


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

        smoke_running = gpu_test.is_running()
        box = layout.box()
        box.label(text="GPU Compute (Phase 3 smoke test)")
        box.operator(
            FLOWX_OT_solver_gpu_test_toggle.bl_idname,
            text="Stop Smoke Test" if smoke_running else "Run Smoke Test",
            icon="PAUSE" if smoke_running else "PLAY",
            depress=smoke_running,
        )
        if smoke_running:
            box.label(text=f"{PARTICLE_COUNT} particles falling under gravity")


class FLOWX_PT_solver(Panel):
    bl_label = "SPH Solver"
    bl_idname = "FLOWX_PT_solver"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Flow-X"
    bl_parent_id = "FLOWX_PT_domain"

    @classmethod
    def poll(cls, context):
        return FLOWX_PT_domain.poll(context)

    def draw(self, context):
        layout = self.layout
        settings = context.active_object.flowx_domain

        col = layout.column(align=True)
        col.prop(settings, "smoothing_radius")
        col.prop(settings, "rest_density")
        col.prop(settings, "stiffness")
        col.prop(settings, "viscosity")
        col.prop(settings, "max_substeps")

        running = sph.is_running()
        layout.operator(
            FLOWX_OT_sph_toggle.bl_idname,
            text="Stop Simulation" if running else "Run Simulation",
            icon="PAUSE" if running else "PLAY",
            depress=running,
        )

        stats = sph.stats()
        if stats is None:
            layout.label(text="Seeds at the fluid level above, then steps on playback.")
            return

        box = layout.box()
        col = box.column(align=True)
        dims = "x".join(str(n) for n in stats["cell_dims"])
        col.label(text=f"Grid: {dims} ({stats['cells']} cells)")
        # The solver coarsens its own spacing when a domain would blow the
        # particle budget, so show what it actually settled on.
        if abs(stats["smoothing_radius"] - settings.smoothing_radius) > 1e-6:
            col.label(text=f"Effective radius: {stats['smoothing_radius']:.4f} m", icon="INFO")


class FLOWX_PT_playback(Panel):
    bl_label = "Playback"
    bl_idname = "FLOWX_PT_playback"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Flow-X"
    bl_parent_id = "FLOWX_PT_domain"

    @classmethod
    def poll(cls, context):
        return FLOWX_PT_domain.poll(context)

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        stats = sph.stats()

        if stats is None:
            layout.label(text="Run the simulation to step it with the timeline.")
            return

        playing = context.screen.is_animation_playing
        row = layout.row(align=True)
        # Blender's own transport operator, so the button reflects and drives
        # the same playback state as the timeline rather than a second one.
        row.operator(
            "screen.animation_play",
            text="Pause" if playing else "Play",
            icon="PAUSE" if playing else "PLAY",
            depress=playing,
        )
        row.operator(FLOWX_OT_sph_reset.bl_idname, text="Reset", icon="FILE_REFRESH")

        col = layout.column(align=True)
        col.label(text=f"Simulated frame: {stats['frame']} (seeded at {stats['seed_frame']})")
        if scene.frame_current != stats["frame"]:
            col.label(text=f"Timeline: frame {scene.frame_current}", icon="TIME")

        # Scrubbing backward has no cache behind it, so the solver holds its
        # last state and says why here rather than showing a frame it never
        # simulated.
        if stats["warning"]:
            box = layout.box()
            box.alert = True
            column = box.column(align=True)
            icon = "ERROR"
            for line in _wrap(stats["warning"], 44):
                column.label(text=line, icon=icon)
                icon = "BLANK1"

        box = layout.box()
        box.label(text="Performance")
        col = box.column(align=True)
        col.label(text=f"Particles: {stats['particles']}")
        col.label(text=f"Substeps/frame: {stats['substeps']}")
        step_ms = stats["step_ms"]
        if step_ms is None:
            col.label(text="ms/step: play or step a frame to measure")
        else:
            col.label(text=f"ms/step: {step_ms:.1f} ({1000.0 / max(step_ms, 1e-3):.1f} fps)")
            target_ms = 1000.0 / max(_scene_fps(scene), 1e-3)
            if step_ms > target_ms:
                col.label(
                    text=(
                        f"Slower than the scene's {_scene_fps(scene):.0f} fps - lower the "
                        "surface resolution or raise the smoothing radius."
                    ),
                    icon="INFO",
                )


class FLOWX_PT_surface(Panel):
    bl_label = "Surface"
    bl_idname = "FLOWX_PT_surface"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Flow-X"
    bl_parent_id = "FLOWX_PT_domain"

    @classmethod
    def poll(cls, context):
        return FLOWX_PT_domain.poll(context)

    def draw(self, context):
        layout = self.layout
        settings = context.active_object.flowx_domain

        layout.prop(settings, "show_surface")

        col = layout.column(align=True)
        col.enabled = settings.show_surface
        col.prop(settings, "surface_resolution")
        col.prop(settings, "surface_iso")

        layout.prop(settings, "show_particles")

        stats = sph.stats()
        surface_stats = stats["surface"] if stats else None
        if surface_stats is None:
            layout.label(text="Extracted into a '.FluidSurface' child on each frame.")
            return

        box = layout.box()
        col = box.column(align=True)
        dims = "x".join(str(n) for n in surface_stats["dims"])
        col.label(text=f"Samples: {dims} ({surface_stats['samples']})")
        col.label(
            text=f"Mesh: {surface_stats['vertices']} verts, " f"{surface_stats['triangles']} tris"
        )
        # The grid coarsens itself to stay inside its sample budget, so show
        # the spacing it actually settled on rather than the requested one.
        col.label(text=f"Sample spacing: {surface_stats['spacing']:.4f} m")


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


_classes = (
    FLOWX_PT_domain,
    FLOWX_PT_solver,
    FLOWX_PT_playback,
    FLOWX_PT_surface,
    FLOWX_PT_collider,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
