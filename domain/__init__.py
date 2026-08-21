"""Fluid Domain object: bounds, resolution, fluid level (Phase 1)."""

import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
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

    # Phase 4 solver parameters. Enough knobs to keep the physics out of the
    # source, not a tuning UI - that's post-MVP. Defaults are sized for a
    # roughly 2m domain in Blender's default metric units.
    smoothing_radius: FloatProperty(
        name="Smoothing Radius",
        description=(
            "SPH kernel support radius, in metres. Particles are seeded half a radius "
            "apart, so halving this roughly octuples the particle count - the solver "
            "coarsens it back if that exceeds its particle budget"
        ),
        default=0.1,
        min=0.005,
        max=2.0,
        subtype="DISTANCE",
    )
    rest_density: FloatProperty(
        name="Rest Density",
        description="Target density of the fluid at rest, in kg/m^3 (water is 1000)",
        default=1000.0,
        min=1.0,
        max=20000.0,
    )
    pbf_iterations: IntProperty(
        name="Solver Iterations",
        description=(
            "Jacobi density-constraint solves per substep. More reduces density "
            "error but costs proportionally more GPU dispatches"
        ),
        default=4,
        min=1,
        max=20,
    )
    pbf_relaxation: FloatProperty(
        name="Constraint Relaxation",
        description=(
            "CFM-style epsilon added to the density constraint's lambda "
            "denominator. Prevents a division blow-up when a particle's "
            "neighborhood is nearly empty; higher softens the constraint"
        ),
        default=100.0,
        min=0.0,
        max=100000.0,
    )
    viscosity: FloatProperty(
        name="Viscosity",
        description="Dynamic viscosity coefficient; higher is thicker and more damped",
        default=0.1,
        min=0.0,
        max=10.0,
    )
    pbf_scorr_k: FloatProperty(
        name="Anti-Clustering Strength",
        description=(
            "s_corr tensile-instability correction. Resists particles clumping "
            "into tight clusters under tension (e.g. a stretched sheet of fluid); "
            "0 disables the term"
        ),
        default=0.1,
        min=0.0,
        max=1.0,
    )
    # Phase 6 surface reconstruction. The grid the surface is extracted from is
    # deliberately independent of the solver's, so the look can be refined
    # without disturbing physics that already behaves.
    show_surface: BoolProperty(
        name="Surface Mesh",
        description=(
            "Extract a liquid surface mesh from the particles each frame, into a "
            "child object named '<Domain>.FluidSurface'"
        ),
        default=True,
    )
    show_particles: BoolProperty(
        name="Debug Particles",
        description=(
            "Draw the raw SPH particles as a point-cloud overlay. Costs a per-frame "
            "read-back of every particle position, so it is off unless you're "
            "checking what the solver is doing underneath the surface"
        ),
        default=False,
    )
    show_collider_overlay: BoolProperty(
        name="Collider Overlay",
        description=(
            "Draw the tagged colliders' voxel grids as a point overlay. One point "
            "per occupied voxel, so a domain-filling collider at high resolution is "
            "tens of thousands of points every viewport frame"
        ),
        default=True,
    )
    surface_resolution: IntProperty(
        name="Surface Resolution",
        description=(
            "Sample grid resolution along the domain's longest axis, for surface "
            "extraction only. Cost grows with the cube of this and the extraction "
            "runs on the CPU, so raise it for a final look rather than while "
            "setting the shot up"
        ),
        default=48,
        min=8,
        max=192,
    )
    surface_iso: FloatProperty(
        name="Surface Iso-Value",
        description=(
            "Density threshold the surface is extracted at, as a fraction of rest "
            "density. Lower fattens the fluid and closes gaps between particles; "
            "higher shrinks it toward the particle centres"
        ),
        default=0.5,
        min=0.01,
        max=4.0,
    )
    max_substeps: IntProperty(
        name="Max Substeps",
        description=(
            "Upper bound on solver substeps per frame. The solver takes as many as "
            "stability requires up to this limit; past it, it advances less than a "
            "full frame of simulated time rather than going unstable"
        ),
        default=24,
        min=1,
        max=200,
    )
    # Post-MVP disk cache. Off by default: it writes a file into the user's
    # project on every simulated frame, which should be a choice, not a side
    # effect.
    cache_enabled: BoolProperty(
        name="Cache to Disk",
        description=(
            "Write each simulated frame's particle state to a file, so scrubbing "
            "back through the timeline loads that frame instead of re-simulating. "
            "The file accumulates across Blender restarts until the simulation "
            "settings or colliders change"
        ),
        default=False,
    )
    cache_path: StringProperty(
        name="Cache Path",
        description=(
            "File the cache is written to. Leave empty to write '<scene>.flowx_cache' "
            "next to the saved .blend file (an unsaved scene uses the system temp dir)"
        ),
        subtype="FILE_PATH",
        default="",
    )


def world_bounds(obj):
    """Return (min, max) corners of obj's bounding box, in world space."""
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def is_degenerate(obj):
    """Whether the domain's world bounds have no volume.

    True when any axis is scaled to zero (or below a size the solver could
    meaningfully resolve), e.g. the user hit Ctrl+2 on the domain. Starting
    the solver on such a domain would seed a single particle in a zero-thick
    box, so the operators refuse instead.
    """
    lo, hi = world_bounds(obj)
    size = hi - lo
    return min(size.x, size.y, size.z) <= 1e-6


def is_alive(datablock):
    """Whether a datablock reference still points at live data.

    Blender invalidates the Python wrapper when the data is removed, and then
    touching any field - `.name` included - raises ReferenceError rather than
    returning None. The solver holds references across frames, so it has to ask
    before it dereferences.
    """
    if datablock is None:
        return False
    try:
        return bool(datablock.name)
    except ReferenceError:
        return False


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
        # _CUBE_FACES winds every quad inward; flip so the domain's normals
        # point outward like a normal mesh.
        mesh.flip_normals()
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
