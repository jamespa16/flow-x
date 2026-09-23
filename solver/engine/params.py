"""The parameter block every kernel reads.

One struct, mirroring `struct Params` in kernels/flowx_prelude.h. Field order
and types must match it exactly; nothing checks that at runtime, so
scripts/test_backend.py compiles a probe kernel that reports the struct's real
size and a few offsets and compares them against PARAMS_FORMAT. A silent
mismatch would corrupt every parameter past the drift point, which is not a
failure anyone should have to debug from the symptoms.

The old push-constant block this replaces was exactly 128 bytes, the budget
Blender's backends guaranteed, and completely full - so `scorr_k` and
`surface_tension` were bit-packed into unused lanes of the bitonic-sort slot,
the collider voxel size into a lane of the collider dims, and the surface pass
reinterpreted the domain-max lanes as its own grid corner. None of that is
needed now: Metal's setBytes takes 4 KB, this struct is a few hundred bytes,
and a new parameter is a new field.

Scalars only, no vectors, so the C layout is exactly what struct.pack produces
- MSL aligns float3 to 16 bytes and any vector field would need matching
padding here.
"""

import struct

# (name, struct code). Order must match struct Params in kernels/flowx_prelude.h.
PARAMS_FIELDS = (
    ("particle_count", "i"),
    ("sorted_count", "i"),
    ("cell_count", "i"),
    ("cells_x", "i"),
    ("cells_y", "i"),
    ("cells_z", "i"),
    ("bitonic_k", "i"),
    ("bitonic_j", "i"),
    ("lo_x", "f"),
    ("lo_y", "f"),
    ("lo_z", "f"),
    ("cell_size", "f"),
    ("hi_x", "f"),
    ("hi_y", "f"),
    ("hi_z", "f"),
    ("particle_radius", "f"),
    ("smoothing_radius", "f"),
    ("mass", "f"),
    ("rest_density", "f"),
    ("relaxation", "f"),
    ("viscosity", "f"),
    ("dt", "f"),
    ("gravity", "f"),
    ("boundary_damping", "f"),
    ("scorr_k", "f"),
    ("surface_tension", "f"),
    ("collider_x", "i"),
    ("collider_y", "i"),
    ("collider_z", "i"),
    ("collider_voxel", "f"),
    ("surface_x", "i"),
    ("surface_y", "i"),
    ("surface_z", "i"),
    ("surface_spacing", "f"),
    ("surface_lo_x", "f"),
    ("surface_lo_y", "f"),
    ("surface_lo_z", "f"),
    ("surface_kernel_radius", "f"),
    ("ww_capacity", "i"),
    ("ww_cursor", "i"),
    ("ww_spawn_count", "i"),
    ("frame", "i"),
    ("trapped_air_weight", "f"),
    ("wave_crest_weight", "f"),
    ("kinetic_weight", "f"),
    ("kinetic_reference_speed", "f"),
    ("spray_speed_threshold", "f"),
    ("bubble_trapped_threshold", "f"),
    ("jitter_strength", "f"),
    ("normal_offset", "f"),
    ("spray_life_min", "f"),
    ("spray_life_max", "f"),
    ("foam_life_min", "f"),
    ("foam_life_max", "f"),
    ("bubble_life_min", "f"),
    ("bubble_life_max", "f"),
    ("ww_drag", "f"),
    ("ww_buoyancy", "f"),
    ("frame_dt", "f"),
    ("nodes_x", "i"),
    ("nodes_y", "i"),
    ("nodes_z", "i"),
    ("grid_spacing", "f"),
    ("vorticity_epsilon", "f"),
    ("grid_max_speed", "f"),
    ("pressure_ping", "i"),
)

# "<" - explicit little-endian and no padding. Every field is 4 bytes, so this
# matches the C struct's natural layout exactly.
PARAMS_FORMAT = "<" + "".join(code for _name, code in PARAMS_FIELDS)
PARAMS_SIZE = struct.calcsize(PARAMS_FORMAT)
PARAMS_NAMES = tuple(name for name, _code in PARAMS_FIELDS)

_STRUCT = struct.Struct(PARAMS_FORMAT)
_DEFAULTS = {name: (0 if code == "i" else 0.0) for name, code in PARAMS_FIELDS}


def offset_of(name):
    """Byte offset of a field, for the struct-layout probe test."""
    index = PARAMS_NAMES.index(name)
    return struct.calcsize("<" + "".join(code for _n, code in PARAMS_FIELDS[:index]))


# Per-field (offset, Struct) for patching one value without repacking the rest.
_FIELD_STRUCTS = {
    name: (offset, struct.Struct("<" + code))
    for name, code, offset in (
        (n, c, struct.calcsize("<" + "".join(x[1] for x in PARAMS_FIELDS[:i])))
        for i, (n, c) in enumerate(PARAMS_FIELDS)
    )
}


class ParamBlock:
    """A mutable parameter block that packs to the kernels' `Params`.

    Held across a whole frame and mutated in place, backed by one bytearray
    that is patched in place rather than rebuilt.

    That matters more than it looks: the bitonic sort records on the order of
    three hundred dispatches per frame, each of which carries a copy of this
    block, and repacking all 59 fields each time was measurably more expensive
    than the GPU work those dispatches represent. update() repacks only the
    fields it is given, and pack() with overrides patches, reads and restores
    just those - which is what makes a per-dispatch override cost two struct
    writes instead of fifty-nine.
    """

    __slots__ = ("values", "_buffer")

    def __init__(self, **values):
        self.values = dict(_DEFAULTS)
        self._buffer = bytearray(PARAMS_SIZE)
        _STRUCT.pack_into(self._buffer, 0, *(self.values[n] for n in PARAMS_NAMES))
        self.update(**values)

    def update(self, **values):
        for name, value in values.items():
            field = _FIELD_STRUCTS.get(name)
            if field is None:
                raise KeyError(f"no parameter named {name!r}")
            self.values[name] = value
            offset, packer = field
            packer.pack_into(self._buffer, offset, value)
        return self

    def pack(self, **overrides):
        """Bytes for setBytes, optionally with per-dispatch overrides.

        The overrides are patched in, the buffer is copied, and the previous
        values are put back - so the block itself is unchanged and the caller
        does not have to restore anything.
        """
        if not overrides:
            return bytes(self._buffer)

        restore = []
        for name, value in overrides.items():
            field = _FIELD_STRUCTS.get(name)
            if field is None:
                raise KeyError(f"no parameter named {name!r}")
            offset, packer = field
            restore.append((offset, packer, self.values[name]))
            packer.pack_into(self._buffer, offset, value)
        out = bytes(self._buffer)
        for offset, packer, previous in restore:
            packer.pack_into(self._buffer, offset, previous)
        return out

    def __getitem__(self, name):
        return self.values[name]
