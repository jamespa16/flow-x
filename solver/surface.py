"""Phase 6: turn the particles into a liquid surface mesh.

Once per frame, after the substeps have finished:

    density splat (GPU) -> read the grid back -> marching cubes (CPU)
                        -> rebuild <Domain>.FluidSurface via bmesh

The splat grid's resolution tracks the solver's through the surface multiplier
(so refining the physics refines the look), though the multiplier can be
pulled below 1.0 to keep the extraction cheap under a high-res sim. Its cost
is the one CPU round-trip in the whole pipeline, which is the tradeoff the
roadmap signs up for: a GPU marching cubes needs histopyramid triangle
compaction, and this ships an actual surface now.

Perf: everything here scales with the *cube* of the effective surface grid
(Resolution x the surface multiplier), and the marching cubes pass is Python.
Measured on a 2m domain with 15k particles, the default grid of 48 (125k
samples) costs ~35 ms of extraction and ~2 ms of read-back on top of a ~35 ms
solver step - so the surface roughly doubles the frame, and doubling the grid
would multiply its share by eight. Raise it for a final look, not while
setting the shot up.
"""

import bmesh
import bpy
from mathutils import Vector

from ..domain import is_alive
from . import marching_cubes

# Suffix on the domain's name, so the surface object is findable and obviously
# owned by its domain rather than looking like something the user made.
SURFACE_SUFFIX = ".FluidSurface"
MATERIAL_NAME = "FlowXWater"

# Sample budget. Past this the grid coarsens itself rather than spending
# unbounded time in the CPU-side extraction below.
MAX_SAMPLES = 1 << 20

# Must match SURFACE_CELL_RADIUS in kernels/surface_splat.metal: the kernel is
# clamped so that many spatial-hash cells still cover its full support.
SURFACE_CELL_RADIUS = 2

_state = {
    "running": False,
    # The scalar field the splat writes, one float per lattice point.
    # The engine this run is bound to, kept so extract() can read the field
    # back without sph.py having to hand it over a second time.
    "engine": None,
    "config": None,
    "object": None,
    "vertices": 0,
    "triangles": 0,
    # The last extracted (vertices, triangles), kept so the disk cache can
    # store the frame's surface and the CPU render path can reinstall it.
    "last_vertices": None,
    "last_triangles": None,
}


class SurfaceConfig:
    """Resolved surface-grid parameters for one run."""

    __slots__ = ("dims", "spacing", "sample_count", "kernel_radius", "iso", "lo")


def resolve(domain, config):
    """Surface grid sized from the domain's settings and the solver's bounds.

    The grid starts one kernel radius *before* the domain's low corner and
    runs one past its high corner. The fluid's bled kernel density falls to
    zero only a kernel radius outside each wall, so a grid that stops at the
    domain bounds has no "outside" sample where a wall's iso-surface closes -
    marching clips the surface open there and the fluid renders as a tray
    missing every side whose wall sits on the boundary.
    """
    settings = domain.flowx_domain
    size = config.hi - config.lo
    longest = max(size.x, size.y, size.z)

    surface = SurfaceConfig()
    surface.iso = settings.surface_iso
    # Grid tracks the solver's lattice through the multiplier, like the
    # collider grid does; the kernel clamp below absorbs the difference when
    # the solver has coarsened itself under its particle budget.
    surface.spacing = max(
        longest / max(settings.resolution * settings.surface_multiplier, 1.0), 1e-6
    )

    def _kernel_radius(spacing):
        # A grid coarser than the fluid needs a wider kernel or the field turns
        # into isolated blobs at the sample points; a finer one gains nothing
        # from going below the solver's own smoothing radius.
        return min(
            max(config.smoothing_radius, spacing * 1.5),
            SURFACE_CELL_RADIUS * config.cell_size,
        )

    for _ in range(8):
        # Samples sit on lattice points, so covering N cells along an axis
        # takes N+1 of them. Rounding up means the lattice reaches past the
        # margin on every side rather than clipping the surface at a wall.
        margin = _kernel_radius(surface.spacing)
        extent = size + Vector((2 * margin, 2 * margin, 2 * margin))
        dims = tuple(max(2, int(-(-axis // surface.spacing)) + 1) for axis in extent)
        if dims[0] * dims[1] * dims[2] <= MAX_SAMPLES:
            break
        surface.spacing *= 1.25

    surface.dims = dims
    surface.sample_count = dims[0] * dims[1] * dims[2]
    surface.kernel_radius = _kernel_radius(surface.spacing)
    margin = surface.kernel_radius
    surface.lo = config.lo - Vector((margin, margin, margin))
    return surface


def start(engine, domain, config):
    """Allocate the splat grid. Safe to call repeatedly.

    There is no per-pass compile step any more: the engine compiles one library
    holding every kernel, this one included, when the solver starts.
    """
    _state["running"] = True
    reseed(engine, domain, config)


def reseed(engine, domain, config):
    """Re-resolve the grid for a re-seeded solver.

    Runs on every playback loop, so it only resizes the grid to whatever the
    domain's surface settings now ask for.
    """
    if not _state["running"]:
        return
    surface = resolve(domain, config)
    _state["config"] = surface
    _state["engine"] = engine
    # A flat float per lattice point. The old texture had to borrow the
    # solver's shared texture width, because the splat addressed it through the
    # same 2D wrap the particle state used; the field is just indexed now.
    engine.alloc_surface(surface.sample_count)
    _state["vertices"] = 0
    _state["triangles"] = 0
    _state["object"] = _surface_object(domain)


def stop():
    """Drop the device state. The surface object is left in the scene as-is.

    Deleting it would throw away the last extracted frame, which is usually the
    thing the user just hit stop to look at.
    """
    _state.update({"running": False, "engine": None, "config": None, "object": None})


def is_running():
    return _state["running"]


def stats():
    """Surface grid/mesh figures for the panel, or None when not running."""
    surface = _state["config"]
    if surface is None:
        return None
    return {
        "dims": surface.dims,
        "samples": surface.sample_count,
        "spacing": surface.spacing,
        "iso": surface.iso,
        "vertices": _state["vertices"],
        "triangles": _state["triangles"],
    }


def last_mesh():
    """The (vertices, triangles) last extracted, or None before the first.

    The vertices are 3-tuples and the triangles 3-tuples of vertex indices -
    exactly the shapes _rebuild_mesh() consumes, so the disk cache can round-
    trip a frame's surface through them without re-extraction.
    """
    if _state["last_vertices"] is None:
        return None
    return _state["last_vertices"], _state["last_triangles"]


def install_mesh(domain, vertices, triangles):
    """Rebuild the surface object from an extracted mesh, with no GPU.

    The CPU render path: during a render the compute context is owned by the
    render, so the frame's surface is replayed from the disk cache by writing
    straight into the child object's mesh. `domain` only locates the child -
    the mesh data came from the cache, not from a splat.
    """
    obj = _surface_object(domain)
    _rebuild_mesh(obj, vertices, triangles)
    _state["vertices"] = len(vertices)
    _state["triangles"] = len(triangles)
    _state["last_vertices"] = vertices
    _state["last_triangles"] = triangles


def record(engine, config, dt):
    """Ask the engine to splat the field. On a GPU engine nothing runs yet.

    Split from extract() because the splat is engine work that belongs in the
    same submission as the substeps, while the extraction below is CPU work on
    what that submission produced.
    """
    surface = _state["config"]
    if surface is None:
        return
    engine.splat_surface(surface)


def extract():
    """Read the splatted field back and rebuild the mesh from it.

    Must follow the engine's flush: it reads what the recorded splat wrote.
    """
    surface = _state["config"]
    engine = _state["engine"]
    if surface is None or engine is None:
        return

    field = engine.read_surface(surface.sample_count)
    vertices, triangles = marching_cubes.extract(
        field, surface.dims, surface.lo, surface.spacing, surface.iso
    )

    _state["vertices"] = len(vertices)
    _state["triangles"] = len(triangles)
    _state["last_vertices"] = vertices
    _state["last_triangles"] = triangles

    obj = _state["object"]
    if is_alive(obj):
        _rebuild_mesh(obj, vertices, triangles)


def _surface_object(domain):
    """Find or create the domain's child surface object."""
    name = domain.name + SURFACE_SUFFIX
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != "MESH":
        obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))
        obj.parent = domain
        obj.data.materials.append(_water_material())

    # Extraction emits world-space vertices, so the parent's transform has
    # to be cancelled out or it would be applied to them a second time. This
    # runs on every re-seed, not just creation, so a domain moved mid-run is
    # re-anchored on the next Reset / playback loop instead of keeping the
    # mesh where it used to be.
    obj.matrix_parent_inverse = domain.matrix_world.inverted()

    collection = domain.users_collection[0] if domain.users_collection else None
    if collection is None:
        collection = bpy.context.scene.collection
    if obj.name not in collection.objects:
        collection.objects.link(obj)
    return obj


def _rebuild_mesh(obj, vertices, triangles):
    """Replace the object's geometry with a fresh triangle mesh."""
    mesh = obj.data
    if not triangles:
        mesh.clear_geometry()
        mesh.update()
        return

    bm = bmesh.new()
    bm_verts = [bm.verts.new(vertex) for vertex in vertices]
    for a, b, c in triangles:
        try:
            face = bm.faces.new((bm_verts[a], bm_verts[b], bm_verts[c]))
        except ValueError:
            # Marching cubes can emit the same triangle twice where a surface
            # pinches to zero thickness; bmesh rejects the duplicate and the
            # first one already covers it.
            continue
        # Smooth shading hides the grid's faceting, which is what makes a
        # modest surface resolution read as liquid rather than as voxels.
        face.smooth = True
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def _water_material():
    """A plain water-ish material, so the surface reads as liquid untouched."""
    material = bpy.data.materials.get(MATERIAL_NAME)
    if material is not None:
        return material

    material = bpy.data.materials.new(MATERIAL_NAME)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        # Set by name and only where present: Principled's sockets were renamed
        # in 4.x, and MVP shading isn't worth a version check.
        for socket, value in (
            ("Base Color", (0.24, 0.52, 0.78, 1.0)),
            ("Roughness", 0.05),
            ("IOR", 1.333),
            ("Transmission Weight", 0.9),
            ("Alpha", 0.65),
        ):
            if socket in bsdf.inputs:
                bsdf.inputs[socket].default_value = value

    # Solid-mode viewport colour, so the surface looks like water before anyone
    # switches to a rendered view.
    material.diffuse_color = (0.24, 0.52, 0.78, 0.65)
    if hasattr(material, "blend_method"):
        material.blend_method = "BLEND"
    return material
