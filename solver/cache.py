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

Validity. The config hash covers everything that changes the physics: the
extension version (code constants live in it), the frame rate, the domain's
world bounds and resolution, every solver parameter, and per tagged collider
its name, a transform-invariant mesh fingerprint (so only an actual edit to
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
            "<fffffffII",
            settings.fluid_level,
            settings.smoothing_radius,
            settings.rest_density,
            settings.pbf_relaxation,
            settings.pbf_scorr_k,
            settings.viscosity,
            settings.surface_tension,
            settings.max_substeps,
            settings.pbf_iterations,
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
    except OSError as exc:
        _state["warning"] = (
            f"The cache file could not be opened ({exc}) - the run continues " "without a cache"
        )


def write_frame(frame, positions, velocities, domain):
    """Append one simulated frame (per-item 4-tuples, as read_texture returns).

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


def close():
    """Flush and drop the open file. A recorded warning is kept for the panel."""
    file = _state["file"]
    if file is not None:
        try:
            file.close()
        except OSError:
            pass
    _state.update(file=None, path=None, header=None)


def clear(domain, scene):
    """Delete the cache file. Returns (ok, message) for the operator's report."""
    close()
    _state["warning"] = None
    path = cache_path(domain, scene)
    try:
        if not path.exists():
            return False, f"No cache file to delete at {path}."
        path.unlink()
    except OSError as exc:
        return False, f"The cache file could not be deleted ({exc})."
    return True, f"Deleted {path}."


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
    frames = None
    if header is not None and header["last_frame"] > header["seed_frame"]:
        frames = (header["seed_frame"] + 1, header["last_frame"])
    return {
        "open": _state["file"] is not None,
        "path": path,
        "frames": frames,
        "size": size,
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
