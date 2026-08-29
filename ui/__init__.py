"""N-panel and Object Properties UI (Phase 1+)."""

import bpy
from bpy.types import Panel

from ..collision import FLOWX_OT_toggle_collider, occupied_count
from ..domain import FLOWX_OT_domain_add, find_domain, world_bounds
from ..solver import (
    FLOWX_OT_cache_clear,
    FLOWX_OT_sph_bake,
    FLOWX_OT_sph_reset,
    FLOWX_OT_sph_toggle,
    sph,
    surface,
    whitewater,
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


class FLOWX_PT_no_domain(Panel):
    """Shown in the Flow-X tab while the scene has no domain at all.

    Without this, a fresh scene shows an empty Flow-X tab and the only way to
    start is knowing the domain lives in the Add menu.
    """

    bl_label = "Flow-X"
    bl_idname = "FLOWX_PT_no_domain"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Flow-X"

    @classmethod
    def poll(cls, context):
        return find_domain(context.scene) is None

    def draw(self, context):
        box = self.layout.box()
        box.alert = True
        col = box.column(align=True)
        col.label(text="No fluid domain in this scene")
        col.operator(FLOWX_OT_domain_add.bl_idname, text="Add Fluid Domain", icon="MOD_FLUID")


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
        layout.prop(settings, "collider_voxel_multiplier")

        lo, hi = world_bounds(obj)
        box = layout.box()
        box.label(text="World Bounds")
        col = box.column(align=True)
        col.label(text=f"Min: ({lo.x:.2f}, {lo.y:.2f}, {lo.z:.2f})")
        col.label(text=f"Max: ({hi.x:.2f}, {hi.y:.2f}, {hi.z:.2f})")

        box = layout.box()
        box.label(text="Debug Overlays")
        box.prop(settings, "show_collider_overlay")


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

        # Changing the engine mid-run would leave the timeline half-simulated
        # by one and half by the other, so it is locked while the solver runs.
        row = layout.row()
        row.enabled = not sph.is_running()
        row.prop(settings, "engine")

        col = layout.column(align=True)
        col.prop(settings, "rest_density")
        col.prop(settings, "pbf_iterations")
        col.prop(settings, "pbf_relaxation")
        col.prop(settings, "pbf_scorr_k")
        col.prop(settings, "surface_tension")
        col.prop(settings, "viscosity")
        col.prop(settings, "max_substeps")
        col.prop(settings, "max_particles")

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
        col.label(text=f"Engine: {stats['engine']}")
        dims = "x".join(str(n) for n in stats["cell_dims"])
        col.label(text=f"Grid: {dims} ({stats['cells']} cells)")
        # The solver coarsens its own spacing when a domain would blow the
        # particle budget, so show what it actually settled on.
        lo, hi = world_bounds(context.active_object)
        size = hi - lo
        requested_radius = 2.0 * max(size.x, size.y, size.z) / max(settings.resolution, 1)
        if stats["smoothing_radius"] > requested_radius * (1.0 + 1e-3):
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
        settings = context.active_object.flowx_domain
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
        box.label(text="Cache")
        col = box.column(align=True)
        col.prop(settings, "cache_enabled")
        col.prop(settings, "cache_path")
        cache = stats["cache"]
        if cache["frames"] is not None:
            first, last = cache["frames"]
            size = (cache["size"] or 0) / 1048576.0
            col.label(text=f"Frames {first}-{last} on disk ({size:.1f} MB)")
        elif cache["open"]:
            col.label(text="No frames written yet - the first simulated frame starts the file.")
        else:
            col.label(
                text="Off until enabled - then scrubbing back loads frames instead of re-running."
            )
        if cache["mesh_frames"] is not None:
            mfirst, mlast = cache["mesh_frames"]
            msize = (cache["mesh_size"] or 0) / 1048576.0
            col.label(text=f"Surface baked {mfirst}-{mlast} ({msize:.1f} MB)")
        elif cache["frames"] is None:
            col.label(text="No baked surface yet - renders freeze without it. Bake to record one.")
        if cache["warning"]:
            for line in _wrap(cache["warning"], 44):
                col.label(text=line, icon="ERROR")
        # The render can't step the sim on the GPU (the render owns it), so it
        # replays a baked surface. Bake records the whole range first; the
        # button doubles as the cancel while it runs.
        baking = sph.is_baking()
        col.operator(
            FLOWX_OT_sph_bake.bl_idname,
            text="Stop Baking" if baking else "Bake Cache",
            icon="PAUSE" if baking else "FILE_TICK",
            depress=baking,
        )
        col.operator(FLOWX_OT_cache_clear.bl_idname, text="Clear Cache", icon="TRASH")

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
                        "resolution or the surface multiplier."
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
        col.prop(settings, "surface_multiplier")
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


class FLOWX_PT_whitewater(Panel):
    bl_label = "Whitewater"
    bl_idname = "FLOWX_PT_whitewater"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Flow-X"
    bl_parent_id = "FLOWX_PT_domain"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return FLOWX_PT_domain.poll(context)

    def draw(self, context):
        layout = self.layout
        settings = context.active_object.flowx_domain

        layout.prop(settings, "show_whitewater")

        col = layout.column(align=True)
        col.enabled = settings.show_whitewater
        col.prop(settings, "whitewater_capacity")
        col.prop(settings, "whitewater_spawn_rate")

        box = col.box()
        box.label(text="Spawn Potential")
        potential = box.column(align=True)
        potential.prop(settings, "whitewater_trapped_air_weight")
        potential.prop(settings, "whitewater_wave_crest_weight")
        potential.prop(settings, "whitewater_kinetic_weight")
        potential.prop(settings, "whitewater_kinetic_reference_speed")

        box = col.box()
        box.label(text="Classification")
        classify = box.column(align=True)
        classify.prop(settings, "whitewater_spray_speed_threshold")
        classify.prop(settings, "whitewater_bubble_trapped_threshold")

        box = col.box()
        box.label(text="Motion")
        motion = box.column(align=True)
        motion.prop(settings, "whitewater_jitter_strength")
        motion.prop(settings, "whitewater_normal_offset")
        motion.prop(settings, "whitewater_drag")
        motion.prop(settings, "whitewater_buoyancy")

        box = col.box()
        box.label(text="Lifetime (seconds)")
        life = box.column(align=True)
        row = life.row(align=True)
        row.prop(settings, "whitewater_spray_life_min", text="Spray")
        row.prop(settings, "whitewater_spray_life_max", text="")
        row = life.row(align=True)
        row.prop(settings, "whitewater_foam_life_min", text="Foam")
        row.prop(settings, "whitewater_foam_life_max", text="")
        row = life.row(align=True)
        row.prop(settings, "whitewater_bubble_life_min", text="Bubble")
        row.prop(settings, "whitewater_bubble_life_max", text="")

        stats = whitewater.stats()
        if stats is not None:
            stat_box = layout.box()
            stat_box.label(text=f"Live: {stats['live']} / {stats['capacity']}")


class FLOWX_PT_collider(Panel):
    bl_label = "Flow-X Collider"
    bl_idname = "FLOWX_PT_collider"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None or obj.type != "MESH" or obj.flowx_domain.is_domain:
            return False
        # The auto-generated '<Domain>.FluidSurface'/'<Domain>.Whitewater'
        # children are not collider candidates: they're the fluid's own
        # output, and tagging one would carve the fluid out of itself.
        return not obj.name.endswith((surface.SURFACE_SUFFIX, whitewater.WHITEWATER_SUFFIX))

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
            layout.prop(obj.flowx_collider, "is_animated")


_classes = (
    FLOWX_PT_no_domain,
    FLOWX_PT_domain,
    FLOWX_PT_solver,
    FLOWX_PT_playback,
    FLOWX_PT_surface,
    FLOWX_PT_whitewater,
    FLOWX_PT_collider,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
