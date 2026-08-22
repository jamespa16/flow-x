"""Disk cache: per-frame particle-state snapshots for scrubbing back.

The solver's only state-carrying GPU textures are positions and velocities -
everything else (density, forces, the spatial hash, the collider grid) is
scratch, recomputed every substep - and the run is deterministic (fixed RNG
seed, substep count derived only from the scene's frame rate). A cache of
(positions, velocities) at every frame boundary is therefore a complete
snapshot: loading one and rebuilding the scratch state reproduces that exact
frame, and the collider grid the loaded frame depends on is re-derived from
the scene's state by the depsgraph path before the frame handler runs.

File format (little-endian, one file per run, random access by arithmetic
offset - no index table):

    <header>
    <frame seed+1> ... <frame last>

Header (fixed part, then the collider name list):

    magic "FLWXCA01"   (8s)
    format_version     (I)
    flowx_version      (16s, the extension's version string)
    particle_count     (I)
    seed_frame         (i)
    last_frame         (i)
    fps                (f)
    collider_count     (I)
    per collider: name length (I) + name bytes
    config_hash        (32s, sha256)

Each frame (fixed size):

    positions      (N*4 f32)
    velocities     (N*4 f32)

`last_frame` in the header is rewritten after every frame write, so a torn
tail from a crash is ignored on load. The seed frame itself is never stored:
its state is the deterministic seed, and the handler re-seeds there anyway.

Paired surface-mesh file. Next to the particle file sits
`<base>.flowx_cache.mesh`, an append-only log (no header) of the extracted
surface and whitewater for every frame, seed frame included. Mesh records are
variable-sized, so unlike the particle file they are addressed through an
in-memory {frame: offset} index built by scanning the file on open, and a torn
tail is detected and truncated the same way. This file exists for the CPU
render path: while a render owns the GPU the sim cannot step, so the surface
is replayed from here (load_mesh) instead of re-extracted. It is validated
through the particle file's header - the config hash now covers the surface
and whitewater settings, so a look change invalidates the pair.

Validity. The config hash covers everything that changes the physics: the
extension version (code constants live in it), the frame rate, the domain's
world bounds, resolution and collider voxel multiplier, every solver
parameter, and per tagged collider its name, a transform-invariant mesh
fingerprint (so only an actual edit to
the geometry counts, not the collider simply moving), and a motion
fingerprint - the active action's keyframes when the collider is animated,
or its current world matrix when it isn't - so re-keying a collider's
animation or manually moving a static one invalidates the cache, but an
animated collider simply playing forward does not. A file is only trusted
while the scene still hashes to its header's hash, checked once per write
and once per load rather than by walking historical frames.

Writes are write-through: every simulated frame is appended as it runs, so
the cache accumulates across Blender restarts for as long as the config hash
keeps matching.
"""

import hashlib
import struct
import tempfile
import tomllib
from pathlib import Path

import bpy
from bpy.types import Operator

from ..collision import mesh_fingerprint
from ..domain import find_domain, world_bounds

MAGIC = b"FLWXCA01"
FORMAT_VERSION = 2

# magic, format_version, flowx_version, particle_count, seed_frame, last_frame,
# fps, collider_count
_FIXED_HEADER = struct.Struct("<8sI16sIiifI")

# A header is the fixed part plus one (length + name) pair per collider; 8 KiB
# covers any collider list this add-on will ever see.
_HEADER_READ_BYTES = 8192

_state = {
    "file": None,
    "path": None,
    "header": None,
    # The paired surface-mesh cache: an append-only log of per-frame extracted
    # meshes (variable size, so no arithmetic offsets - a {frame: offset} index
    # is built by scanning the file on open). It carries no header of its own;
    # it is validated through the particle file's header next door.
    "mesh_file": None,
    "mesh_path": None,
    "mesh_index": None,
    "warning": None,
}

_VERSION = None


def _extension_version():
    global _VERSION
    if _VERSION is None:
        try:
            manifest = (
                Path(__file__).resolve().parent.parent / "blender_manifest.toml"
            ).read_text()
            _VERSION = str(tomllib.loads(manifest)["version"])
        except Exception:
            _VERSION = "unknown"
    return _VERSION


def _fps(scene):
    fps = scene.render.fps / scene.render.fps_base if scene.render.fps_base else 24.0
    return fps


def collider_names(scene):
    """Tagged collider object names, sorted - the cache's canonical order."""
    return sorted(
        obj.name for obj in scene.objects if obj.type == "MESH" and obj.flowx_collider.is_collider
    )


def _action_fcurves(action, slot):
    """An action's fcurves for one slot, layered or legacy (Blender 4.4+/pre-4.4)."""
    if hasattr(action, "fcurves"):
        yield from action.fcurves
        return
    for layer in action.layers:
        for strip in layer.strips:
            if strip.type != "KEYFRAME":
                continue
            channelbag = strip.channelbag(slot, ensure=False)
            if channelbag is not None:
                yield from channelbag.fcurves


def _motion_fingerprint(obj):
    """Bytes identifying a collider's motion, for the config hash.

    An animated collider's world matrix legitimately differs frame to frame,
    so hashing a live sample would make the config hash disagree with itself
    across a single run. Instead this hashes the *definition* of the motion:
    the active action's keyframes when the object is animated, or its
    current (constant for the run) world matrix when it isn't. Either one
    only changes when the collider's actual motion changes - a re-key or a
    manual move - not by simply playing the animation forward.
    """
    anim = obj.animation_data
    action = anim.action if anim else None
    if action is None:
        return struct.pack("<16f", *[value for row in obj.matrix_world for value in row])
    digest = hashlib.sha256()
    for fcurve in _action_fcurves(action, anim.action_slot):
        digest.update(fcurve.data_path.encode("utf-8"))
        digest.update(struct.pack("<i", fcurve.array_index))
        for point in fcurve.keyframe_points:
            digest.update(struct.pack("<2f", point.co[0], point.co[1]))
    return digest.digest()


def config_hash(domain, scene):
    """sha256 over everything that changes the physics, for one scene state."""
    settings = domain.flowx_domain
    lo, hi = world_bounds(domain)
    digest = hashlib.sha256()
    digest.update(
        struct.pack(
            "<f3f3fI",
            _fps(scene),
            lo.x,
            lo.y,
            lo.z,
            hi.x,
            hi.y,
            hi.z,
            settings.resolution,
        )
    )
    digest.update(
        struct.pack(
            "<fffffffIII",
            settings.fluid_level,
            settings.rest_density,
            settings.pbf_relaxation,
            settings.pbf_scorr_k,
            settings.viscosity,
            settings.surface_tension,
            settings.collider_voxel_multiplier,
            settings.max_substeps,
            settings.pbf_iterations,
            settings.max_particles,
        )
    )
    # The surface and whitewater settings do not change the physics (the
    # particle cache is valid regardless), but they change the *extracted*
    # surface and whitewater the mesh cache stores - so a look change must
    # invalidate the mesh file too, or a render would replay a stale surface.
    # Both files share this one hash, so toggling either re-bakes the pair.
    s = settings
    digest.update(
        struct.pack(
            "<fffI" + "f" * 18,
            float(s.show_surface),
            s.surface_multiplier,
            s.surface_iso,
            int(s.whitewater_capacity),
            float(s.show_whitewater),
            s.whitewater_spawn_rate,
            s.whitewater_trapped_air_weight,
            s.whitewater_wave_crest_weight,
            s.whitewater_kinetic_weight,
            s.whitewater_kinetic_reference_speed,
            s.whitewater_spray_speed_threshold,
            s.whitewater_bubble_trapped_threshold,
            s.whitewater_jitter_strength,
            s.whitewater_normal_offset,
            s.whitewater_spray_life_min,
            s.whitewater_spray_life_max,
            s.whitewater_foam_life_min,
            s.whitewater_foam_life_max,
            s.whitewater_bubble_life_min,
            s.whitewater_bubble_life_max,
            s.whitewater_drag,
            s.whitewater_buoyancy,
        )
    )
    for name in collider_names(scene):
        digest.update(name.encode("utf-8"))
        mesh_fp = mesh_fingerprint(name)
        digest.update(mesh_fp if mesh_fp is not None else b"")
        digest.update(_motion_fingerprint(scene.objects[name]))
    return digest.digest()


def cache_path(domain, scene):
    """Where the run's cache file lives (an override, or next to the .blend)."""
    override = domain.flowx_domain.cache_path
    if override:
        return Path(bpy.path.abspath(override)).expanduser()
    if bpy.data.filepath:
        return Path(bpy.data.filepath).with_suffix(".flowx_cache")
    return Path(tempfile.gettempdir()) / "flow-x" / f"{scene.name}.flowx_cache"


def _mesh_path(particle_path):
    """The paired surface-mesh cache, next to the particle file.

    `<base>.flowx_cache` holds the particle snapshots; `<base>.flowx_cache.mesh`
    holds the extracted surface + whitewater for the CPU render path. Appending
    `.mesh` to the full name (rather than swapping the suffix) keeps the pair
    together regardless of what the particle file is called.
    """
    return particle_path.parent / (particle_path.name + ".mesh")


def _pack_header(header):
    parts = [
        _FIXED_HEADER.pack(
            MAGIC,
            header["format_version"],
            header["flowx_version"].encode("utf-8")[:16].ljust(16, b"\x00"),
            header["particle_count"],
            header["seed_frame"],
            header["last_frame"],
            header["fps"],
            len(header["colliders"]),
        )
    ]
    for name in header["colliders"]:
        raw = name.encode("utf-8")
        parts.append(struct.pack("<I", len(raw)))
        parts.append(raw)
    parts.append(header["config_hash"])
    return b"".join(parts)


def _unpack_header(data):
    """Parse a header from a byte string, or None when the file is not ours."""
    if len(data) < _FIXED_HEADER.size:
        return None
    try:
        (
            magic,
            format_version,
            version,
            particle_count,
            seed_frame,
            last_frame,
            fps,
            collider_count,
        ) = _FIXED_HEADER.unpack_from(data, 0)
        if magic != MAGIC or format_version != FORMAT_VERSION:
            return None
        offset = _FIXED_HEADER.size
        colliders = []
        for _ in range(collider_count):
            (name_len,) = struct.unpack_from("<I", data, offset)
            raw = data[offset + 4 : offset + 4 + name_len]
            if len(raw) < name_len:
                return None
            colliders.append(raw.decode("utf-8"))
            offset += 4 + name_len
        config_hash = data[offset : offset + 32]
        if len(config_hash) < 32:
            return None
    except (struct.error, UnicodeDecodeError):
        return None
    return {
        "format_version": format_version,
        "flowx_version": version.decode("utf-8", "replace").rstrip("\x00"),
        "particle_count": particle_count,
        "seed_frame": seed_frame,
        "last_frame": last_frame,
        "fps": fps,
        "colliders": colliders,
        "config_hash": config_hash,
        "header_size": offset + 32,
        "frame_size": 8 * 4 * particle_count,
    }


def _read_existing_header(path):
    try:
        with path.open("rb") as handle:
            data = handle.read(_HEADER_READ_BYTES)
    except OSError:
        return None
    return _unpack_header(data)


def _fail(message):
    """Give up on the open file, but keep the run going and say why."""
    _state["warning"] = message
    close()


def open(scene, domain, particle_count):
    """(Re)open the run's cache file, or drop the handle when caching is off.

    Called at every re-seed, so a mid-run edit to any hashed setting (or to
    the cache toggle itself) takes effect on the next Reset or playback loop.
    The seed frame is the frame the run was (re)started at - a Reset away
    from the timeline's start frame restarts the run there, and the seeded
    state is what that frame shows - so the file records it, not
    frame_start. An existing file is reused for appending only while its
    config hash, particle count and seed frame still match the run -
    deterministic runs rewrite identical frames, so appending across Blender
    restarts is safe.
    """
    close()
    _state["warning"] = None
    if not domain.flowx_domain.cache_enabled:
        return
    try:
        path = cache_path(domain, scene)
        path.parent.mkdir(parents=True, exist_ok=True)
        current_hash = config_hash(domain, scene)
        existing = _read_existing_header(path)
        if (
            existing is not None
            and existing["config_hash"] == current_hash
            and existing["particle_count"] == particle_count
            and existing["seed_frame"] == scene.frame_current
        ):
            file = path.open("r+b")
            header = existing
        else:
            file = path.open("w+b")
            header = {
                "format_version": FORMAT_VERSION,
                "flowx_version": _extension_version(),
                "particle_count": particle_count,
                "seed_frame": scene.frame_current,
                "last_frame": scene.frame_current,
                "fps": _fps(scene),
                "colliders": collider_names(scene),
                "config_hash": current_hash,
                "header_size": 0,
                "frame_size": 8 * 4 * particle_count,
            }
            packed = _pack_header(header)
            header["header_size"] = len(packed)
            file.write(packed)
            file.flush()
        _state.update(file=file, path=str(path), header=header)
        _open_mesh_file(path)
    except OSError as exc:
        _state["warning"] = (
            f"The cache file could not be opened ({exc}) - the run continues " "without a cache"
        )


def _open_mesh_file(particle_path):
    """Open (or remember the path of) the paired surface-mesh cache.

    An existing file is opened read+append and its {frame: offset} index is
    built by scanning the records; a file that does not exist yet is only
    remembered, so the first bake creates it rather than a read-only run
    leaving an empty file behind.
    """
    mesh_path = _mesh_path(particle_path)
    _state["mesh_path"] = str(mesh_path)
    _state["mesh_file"] = None
    _state["mesh_index"] = None
    if mesh_path.exists():
        _state["mesh_file"] = mesh_path.open("a+b")
        _state["mesh_index"] = _scan_mesh_index(_state["mesh_file"])


def _scan_mesh_index(file):
    """The {frame: byte offset} of every complete record in the mesh file.

    Mesh records are variable-sized, so there are no arithmetic offsets: walk
    the file, reading each record's 16-byte header (frame, then the vert, tri
    and whitewater counts) to compute its length and skip to the next. A
    partial record at the end - a torn tail from a crash mid-write - is
    detected when the file ends before a record's declared size and is
    truncated away, so a restart resumes from the last complete frame.
    """
    index = {}
    file.seek(0)
    size = file.seek(0, 2)
    file.seek(0)
    offset = 0
    last_good = 0
    while offset + 16 <= size:
        file.seek(offset)
        head = file.read(16)
        if len(head) < 16:
            break
        frame, vn, tn, wn = struct.unpack("<4i", head)
        if vn < 0 or tn < 0 or wn < 0:
            break  # corrupt header - stop at the last good record
        record = 16 + vn * 12 + tn * 12 + wn * 20
        if offset + record > size:
            break  # torn tail
        index[frame] = offset
        last_good = offset + record
        offset += record
    if last_good < size:
        file.seek(last_good)
        file.truncate()
    return index


def _append_mesh(frame, mesh, ww):
    """Append one frame's extracted surface to the mesh cache (or skip it).

    A no-op when the cache is closed, there is no mesh to store, or the frame
    is already recorded - a deterministic re-run produces the identical mesh,
    so re-simulating a cached frame must not append a duplicate. The first
    call creates the file if open() only remembered its path.
    """
    if mesh is None or _state["mesh_path"] is None:
        return
    if _state["mesh_file"] is None:
        try:
            _state["mesh_file"] = Path(_state["mesh_path"]).open("a+b")
            _state["mesh_index"] = _scan_mesh_index(_state["mesh_file"])
        except OSError:
            return
    index = _state["mesh_index"]
    if index is None or frame in index:
        return
    vertices, triangles = mesh
    ww = ww or []
    mesh_file = _state["mesh_file"]
    try:
        mesh_file.seek(0, 2)
        offset = mesh_file.tell()
        vn, tn, wn = len(vertices), len(triangles), len(ww)
        mesh_file.write(struct.pack("<4i", frame, vn, tn, wn))
        if vn:
            mesh_file.write(struct.pack(f"<{vn * 3}f", *[c for v in vertices for c in v]))
        if tn:
            mesh_file.write(struct.pack(f"<{tn * 3}I", *[c for t in triangles for c in t]))
        if wn:
            mesh_file.write(struct.pack(f"<{wn * 5}f", *[c for p in ww for c in p]))
        mesh_file.flush()
        index[frame] = offset
    except OSError:
        # A mesh write failure must not kill the run: the particle cache is the
        # primary one, and a missing frame only warns at render time.
        pass


def write_frame(frame, positions, velocities, domain, mesh=None, ww=None):
    """Append one simulated frame (per-item 4-tuples, as read_texture returns).

    `mesh` is the frame's (vertices, triangles) and `ww` its whitewater
    points, stored in the paired mesh file for the CPU render path - they are
    written after the surface and whitewater have been extracted for this
    frame, so the caller must pass the frame's own, not a previous frame's.

    A no-op while the cache is closed; a failure closes the cache and records
    a panel warning rather than interrupting the run. The config hash is
    re-checked on every write, not trusted from the open: a settings edit
    made after the last re-seed must not land in a file opened under the old
    settings, whether or not a load was attempted in between.
    """
    file = _state["file"]
    header = _state["header"]
    if file is None or header is None:
        return
    scene = bpy.context.scene
    names = collider_names(scene)
    if names != header["colliders"]:
        _fail("collider tagging changed - the cache file no longer matches the scene")
        return
    count = header["particle_count"]
    if len(positions) != count or len(velocities) != count:
        _fail("particle count changed - the cache file no longer matches the run")
        return
    if frame <= header["seed_frame"]:
        return
    if config_hash(domain, scene) != header["config_hash"]:
        _fail(
            "the simulation settings changed - the cache file no longer matches; "
            "Reset re-opens a fresh one"
        )
        return
    try:
        offset = header["header_size"] + (frame - header["seed_frame"] - 1) * header["frame_size"]
        file.seek(offset)
        file.write(struct.pack(f"<{count * 4}f", *[c for item in positions for c in item]))
        file.write(struct.pack(f"<{count * 4}f", *[c for item in velocities for c in item]))
        header["last_frame"] = max(header["last_frame"], frame)
        file.seek(0)
        file.write(_pack_header(header))
        file.flush()
    except OSError as exc:
        _fail(f"writing the cache failed ({exc}) - the run continues without a cache")
    _append_mesh(frame, mesh, ww)


def write_seed_mesh(domain, scene, mesh, ww):
    """Write the seed frame's surface to the mesh cache.

    The particle file never stores the seed frame (its state is the
    deterministic seed, re-derived on load), but the render path still needs
    the seed frame's extracted surface - it is the first rendered frame - so
    it is stored in the mesh file alone. Called right after open(), once the
    seed surface has been extracted.
    """
    if _state["file"] is None or _state["header"] is None:
        return
    if config_hash(domain, scene) != _state["header"]["config_hash"]:
        return
    _append_mesh(_state["header"]["seed_frame"], mesh, ww)


def try_load(frame, scene, domain):
    """(positions, velocities) for a cached frame, or None with a warning set.

    The frame must be inside the file's covered range and the scene must
    still hash to the file's config hash - which covers each collider's
    motion definition, not just its live transform - otherwise the scene
    changed after the run and the stored state is not honest.
    """
    file = _state["file"]
    header = _state["header"]
    if file is None or header is None:
        return None
    if not header["seed_frame"] < frame <= header["last_frame"]:
        return None
    if config_hash(domain, scene) != header["config_hash"]:
        _state["warning"] = (
            "The cache no longer matches the simulation settings - return to "
            f"frame {header['seed_frame']} to re-run and rebuild it."
        )
        return None
    if collider_names(scene) != header["colliders"]:
        _state["warning"] = (
            "The cache no longer matches the scene's colliders - return to "
            f"frame {header['seed_frame']} to re-run and rebuild it."
        )
        return None
    try:
        first = header["seed_frame"] + 1
        file.seek(header["header_size"] + (frame - first) * header["frame_size"])
        n4 = header["particle_count"] * 4
        positions = list(struct.unpack(f"<{n4}f", file.read(4 * n4)))
        velocities = list(struct.unpack(f"<{n4}f", file.read(4 * n4)))
    except (OSError, struct.error):
        _state["warning"] = (
            "The cache file could not be read - return to "
            f"frame {header['seed_frame']} to re-run."
        )
        return None
    _state["warning"] = None
    return positions, velocities


def load_mesh(frame, scene, domain):
    """(vertices, triangles, whitewater) for a baked frame, or None + warning.

    The CPU render path: during a render the compute context is owned by the
    render, so the frame's surface is replayed from the mesh file instead of
    being re-extracted on the GPU. The frame must be in the file's index and
    the scene must still hash to the particle file's config hash - which now
    covers the surface and whitewater settings the stored mesh depends on.
    """
    index = _state["mesh_index"]
    file = _state["mesh_file"]
    header = _state["header"]
    if index is None or file is None:
        _state["warning"] = (
            "No baked surface cache - bake the cache (Playback > Bake Cache) " "before rendering."
        )
        return None
    if header is not None and config_hash(domain, scene) != header["config_hash"]:
        _state["warning"] = (
            "The baked cache no longer matches the settings - bake again to " "update it."
        )
        return None
    if frame not in index:
        _state["warning"] = (
            f"Frame {frame} is not in the baked surface cache - render a range "
            "the bake covered, or bake further."
        )
        return None
    try:
        file.seek(index[frame])
        _frame, vn, tn, wn = struct.unpack("<4i", file.read(16))
        flat = list(struct.unpack(f"<{vn * 3}f", file.read(vn * 12))) if vn else []
        vertices = [flat[i : i + 3] for i in range(0, len(flat), 3)]
        flat = list(struct.unpack(f"<{tn * 3}I", file.read(tn * 12))) if tn else []
        triangles = [flat[i : i + 3] for i in range(0, len(flat), 3)]
        flat = list(struct.unpack(f"<{wn * 5}f", file.read(wn * 20))) if wn else []
        ww = [flat[i : i + 5] for i in range(0, len(flat), 5)]
    except (OSError, struct.error):
        _state["warning"] = "The baked surface cache could not be read."
        return None
    _state["warning"] = None
    return vertices, triangles, ww


def close():
    """Flush and drop the open files. A recorded warning is kept for the panel."""
    file = _state["file"]
    if file is not None:
        try:
            file.close()
        except OSError:
            pass
    mesh_file = _state["mesh_file"]
    if mesh_file is not None:
        try:
            mesh_file.close()
        except OSError:
            pass
    _state.update(
        file=None,
        path=None,
        header=None,
        mesh_file=None,
        mesh_path=None,
        mesh_index=None,
    )


def clear(domain, scene):
    """Delete the cache files (particles and the paired surface mesh).

    Returns (ok, message) for the operator's report.
    """
    close()
    _state["warning"] = None
    path = cache_path(domain, scene)
    deleted = []
    for p in (path, _mesh_path(path)):
        try:
            if p.exists():
                p.unlink()
                deleted.append(str(p))
        except OSError as exc:
            return False, f"The cache file could not be deleted ({exc})."
    if not deleted:
        return False, f"No cache file to delete at {path}."
    return True, f"Deleted {', '.join(deleted)}."


def is_open():
    return _state["file"] is not None


def header():
    return _state["header"]


def warning():
    return _state["warning"]


def info():
    """Cache status for the panel."""
    header = _state["header"]
    path = _state["path"]
    size = None
    if path is not None:
        try:
            size = Path(path).stat().st_size
        except OSError:
            pass
    mesh_size = None
    mesh_path = _state["mesh_path"]
    if mesh_path is not None:
        try:
            mesh_size = Path(mesh_path).stat().st_size
        except OSError:
            pass
    frames = None
    if header is not None and header["last_frame"] > header["seed_frame"]:
        frames = (header["seed_frame"] + 1, header["last_frame"])
    # The mesh file's coverage includes the seed frame (the particle file's
    # does not), so report it from the index rather than the header.
    index = _state["mesh_index"]
    mesh_frames = (min(index), max(index)) if index else None
    return {
        "open": _state["file"] is not None,
        "path": path,
        "frames": frames,
        "size": size,
        "mesh_frames": mesh_frames,
        "mesh_size": mesh_size,
        "warning": _state["warning"],
    }


class FLOWX_OT_cache_clear(Operator):
    """Delete the simulation's disk cache file"""

    bl_idname = "flowx.cache_clear"
    bl_label = "Clear Cache"
    # Not UNDO: the file is outside Blender's undo stack, and deleting it twice
    # is harmless while "undoing" it is impossible.
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return find_domain(context.scene) is not None

    def execute(self, context):
        ok, message = clear(find_domain(context.scene), context.scene)
        self.report({"INFO" if ok else "WARNING"}, message)
        return {"FINISHED"} if ok else {"CANCELLED"}
