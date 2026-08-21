"""Phase 6: turn the particles into a liquid surface mesh.

Once per frame, after the substeps have finished:

    density splat (GPU) -> read the grid back -> marching cubes (CPU)
                        -> rebuild <Domain>.FluidSurface via bmesh

The splat grid's resolution is independent of the solver's, so the surface can
be refined without touching the physics (and vice versa). Its cost is the one
CPU round-trip in the whole pipeline, which is the tradeoff the roadmap signs
up for: a GPU marching cubes needs histopyramid triangle compaction, and this
ships an actual surface now.

Perf: everything here scales with the *cube* of `surface_resolution`, and the
marching cubes pass is Python. Measured on a 2m domain with 15k particles, the
default resolution of 48 (125k samples) costs ~35 ms of extraction and ~2 ms of
read-back on top of a ~35 ms solver step - so the surface roughly doubles the
frame, and doubling the resolution would multiply its share by eight. Raise it
for a final look, not while setting the shot up.
"""

import struct

import bmesh
import bpy
from mathutils import Vector

from ..domain import is_alive
from . import marching_cubes
from .gpu_util import (
    bind_push_constants,
    build_compute_shader,
    dispatch_1d,
    make_texture,
    read_scalar_texture,
    shader_source,
)

LOCAL_GROUP_SIZE = 64

# Suffix on the domain's name, so the surface object is findable and obviously
# owned by its domain rather than looking like something the user made.
SURFACE_SUFFIX = ".FluidSurface"
MATERIAL_NAME = "FlowXWater"

# Sample budget. Past this the grid coarsens itself rather than spending
# unbounded time in the CPU-side extraction below.
MAX_SAMPLES = 1 << 20

# Must match SURFACE_CELL_RADIUS in shaders/surface_splat.glsl: the kernel is
# clamped so that many spatial-hash cells still cover its full support.
SURFACE_CELL_RADIUS = 2

_IMAGES = (
    ("RGBA32F", "FLOAT_2D", "positions_img"),
    ("RGBA32F", "FLOAT_2D", "keys_img"),
    ("R32F", "FLOAT_2D", "cell_start_img"),
    ("R32F", "FLOAT_2D", "cell_end_img"),
    ("R32F", "FLOAT_3D", "collider_img"),
    ("R32F", "FLOAT_2D", "surface_img"),
)

# The SPH block with its two unused-here slots repurposed; see the header of
# shaders/surface_splat.glsl. Order and total size must still match, because
# sph_common.glsl is compiled against it.
_PUSH_CONSTANTS = (
    ("IVEC4", "i_layout"),
    ("IVEC4", "i_grid"),
    ("IVEC4", "i_surface"),
    ("VEC4", "f_lo"),
    ("VEC4", "f_hi"),
    ("VEC4", "f_sph"),
    ("VEC4", "f_sim"),
    ("IVEC4", "i_collider"),
)

_state = {
    "shader": None,
    "texture": None,
    "config": None,
    "object": None,
    "vertices": 0,
    "triangles": 0,
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
    surface.spacing = max(longest / max(settings.surface_resolution, 1), 1e-6)

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


def start(domain, config):
    """Compile the splat pass and allocate its grid. Safe to call repeatedly."""
    _state["shader"] = build_compute_shader(
        [shader_source("sph_common"), shader_source("surface_splat")],
        _IMAGES,
        _PUSH_CONSTANTS,
        LOCAL_GROUP_SIZE,
    )
    reseed(domain, config)


def reseed(domain, config):
    """Re-resolve the grid for a re-seeded solver, keeping the compiled shader.

    The splat shader depends only on its binding layout, so a re-seed - which
    happens on every playback loop - only has to resize the grid to whatever
    the domain's surface settings now ask for.
    """
    if _state["shader"] is None:
        return
    surface = resolve(domain, config)
    _state["config"] = surface
    _state["texture"] = make_texture(surface.sample_count, channels=1, fmt="R32F")
    _state["vertices"] = 0
    _state["triangles"] = 0
    _state["object"] = _surface_object(domain)


def stop():
    """Drop the GPU state. The surface object is left in the scene as-is.

    Deleting it would throw away the last extracted frame, which is usually the
    thing the user just hit stop to look at.
    """
    _state.update({"shader": None, "texture": None, "config": None, "object": None})


def is_running():
    return _state["shader"] is not None


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


def update(config, textures, constants, collider):
    """Splat, extract and rebuild the surface mesh for the current frame.

    `constants` is the SPH push-constant block sph.py already assembled for
    this step; only the two slots this pass repurposes are overridden.
    """
    surface = _state["config"]
    shader = _state["shader"]
    if shader is None or surface is None:
        return

    collider_texture, i_collider = collider
    spacing_bits = struct.unpack("<i", struct.pack("<f", surface.spacing))[0]

    values = dict(constants)
    values.pop("i_sort")
    values["i_surface"] = (*surface.dims, spacing_bits)
    values["i_collider"] = i_collider
    # Same block, but the kernel radius is the surface grid's, not the solver's.
    values["f_sph"] = (surface.kernel_radius, *constants["f_sph"][1:])
    # The grid starts one kernel radius before the domain's low corner (see
    # resolve); the shader must splat from that true origin or the bled density
    # at the walls is sampled from the wrong cells and the surface clips open.
    values["f_lo"] = (surface.lo.x, surface.lo.y, surface.lo.z, constants["f_lo"][3])

    shader.bind()
    bind_push_constants(shader, values)
    for _fmt, _type, name in _IMAGES:
        if name == "surface_img":
            shader.image(name, _state["texture"])
        elif name == "collider_img":
            shader.image(name, collider_texture)
        else:
            shader.image(name, textures[name])
    dispatch_1d(shader, surface.sample_count, LOCAL_GROUP_SIZE)

    field = read_scalar_texture(_state["texture"], surface.sample_count)
    vertices, triangles = marching_cubes.extract(
        field, surface.dims, surface.lo, surface.spacing, surface.iso
    )

    _state["vertices"] = len(vertices)
    _state["triangles"] = len(triangles)

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
        # Extraction emits world-space vertices, so the parent's transform has
        # to be cancelled out or it would be applied to them a second time.
        obj.matrix_parent_inverse = domain.matrix_world.inverted()
        obj.data.materials.append(_water_material())

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
