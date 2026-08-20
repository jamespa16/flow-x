"""Standalone tests for solver/marching_cubes.py.

The marching cubes table is derived rather than transcribed, so it needs to be
checked rather than trusted. These run without Blender:

    python3 scripts/test_marching_cubes.py

The important property is watertightness. A surface extracted from a closed
field must be a closed manifold, which means every mesh edge is shared by
exactly two triangles - the check that catches both a mis-derived case and a
pairing rule that disagrees between neighbouring cubes.
"""

import importlib.util
import math
import sys
from pathlib import Path

# Loaded by path rather than as `solver.marching_cubes`: importing the package
# would pull in `solver/__init__.py`, which needs `bpy`. The module under test
# deliberately has no Blender imports of its own.
_SOURCE = Path(__file__).resolve().parent.parent / "solver" / "marching_cubes.py"
_spec = importlib.util.spec_from_file_location("flowx_marching_cubes", _SOURCE)
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def test_table_covers_sign_changes():
    """Every case triangulates exactly the edges that have a sign change."""
    for case in range(256):
        used = set()
        for triangle in mc.TRI_TABLE[case]:
            _check(len(set(triangle)) == 3, f"case {case}: degenerate triangle {triangle}")
            used.update(triangle)

        expected = {index for index in range(12) if mc.EDGE_TABLE[case] & (1 << index)}
        _check(
            used == expected,
            f"case {case}: triangulated edges {sorted(used)} != sign-change edges "
            f"{sorted(expected)}",
        )

        for edge in used:
            a, b = mc.EDGES[edge]
            inside_a = bool(case & (1 << a))
            inside_b = bool(case & (1 << b))
            _check(inside_a != inside_b, f"case {case}: edge {edge} has no sign change")

    empty = [case for case in range(256) if not mc.TRI_TABLE[case]]
    _check(empty == [0, 255], f"only the all-in/all-out cases should be empty, got {empty}")


def test_patch_boundaries_are_closed():
    """Each case's triangles form loops closed within the cube's faces.

    A surface patch that isn't closed off by the cube's own faces would leave a
    hole no neighbouring cube can fill. Interior edges of the patch (both ends
    interpolated on the same face) must be traversed exactly twice, in opposite
    directions; the loop's own boundary edges lie on faces and appear once.
    """
    for case in range(256):
        counts = {}
        for a, b, c in mc.TRI_TABLE[case]:
            for edge in ((a, b), (b, c), (c, a)):
                counts[edge] = counts.get(edge, 0) + 1

        for (a, b), count in counts.items():
            _check(count == 1, f"case {case}: directed edge ({a},{b}) used {count} times")
            reverse = counts.get((b, a), 0)
            _check(
                reverse <= 1,
                f"case {case}: reverse of edge ({a},{b}) used {reverse} times",
            )


def _sample_field(dims, origin, spacing, function):
    nx, ny, nz = dims
    field = []
    for k in range(nz):
        z = origin[2] + k * spacing
        for j in range(ny):
            y = origin[1] + j * spacing
            for i in range(nx):
                field.append(function(origin[0] + i * spacing, y, z))
    return field


def _assert_watertight(triangles, label):
    counts = {}
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            counts[key] = counts.get(key, 0) + 1
    unshared = [edge for edge, count in counts.items() if count != 2]
    _check(
        not unshared,
        f"{label}: {len(unshared)} of {len(counts)} edges are not shared by exactly "
        f"two triangles (mesh has holes)",
    )


def _signed_volume(vertices, triangles):
    """Six times the signed volume; positive when triangles wind outward."""
    total = 0.0
    for ia, ib, ic in triangles:
        ax, ay, az = vertices[ia]
        bx, by, bz = vertices[ib]
        cx, cy, cz = vertices[ic]
        total += ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
    return total / 6.0


def test_sphere_is_watertight_and_correctly_sized():
    """A sphere field extracts to a closed mesh of about the right volume."""
    radius = 0.7
    spacing = 0.05
    dims = (41, 41, 41)
    origin = (-1.0, -1.0, -1.0)

    field = _sample_field(
        dims, origin, spacing, lambda x, y, z: radius - math.sqrt(x * x + y * y + z * z)
    )
    vertices, triangles = mc.extract(field, dims, origin, spacing, 0.0)

    _check(len(triangles) > 1000, f"expected a dense sphere mesh, got {len(triangles)} triangles")
    _assert_watertight(triangles, "sphere")

    volume = _signed_volume(vertices, triangles)
    expected = 4.0 / 3.0 * math.pi * radius**3
    _check(volume > 0.0, f"sphere windings point inward (signed volume {volume:.4f})")
    _check(
        abs(volume - expected) / expected < 0.01,
        f"sphere volume {volume:.4f} differs from {expected:.4f} by more than 1%",
    )


def test_every_case_appears_and_stays_watertight():
    """Exercise all 256 cases at once and confirm the result is still closed.

    A field of random-ish signs on a small lattice hits every case many times,
    including the ambiguous faces the published table gets wrong. Sign changes
    are what matter, so exact values don't need to be meaningful.

    The lattice's outermost samples are forced below the iso-value. Without
    that the surface would run off the edge of the sampled block and be open
    there by construction, which has nothing to do with the table.
    """
    dims = (18, 18, 18)
    origin = (0.0, 0.0, 0.0)
    spacing = 0.25
    nx, ny, nz = dims

    # A deterministic hash, so a failure is reproducible. It needs to mix well
    # enough that neighbouring samples are independent, or whole families of
    # cases never come up - hence a full avalanche rather than the usual
    # multiply-and-xor spatial hash.
    mask = (1 << 64) - 1

    def noise(x, y, z):
        i = int(round(x / spacing))
        j = int(round(y / spacing))
        k = int(round(z / spacing))
        if i in (0, nx - 1) or j in (0, ny - 1) or k in (0, nz - 1):
            return 0.0
        h = (i * 0x9E3779B97F4A7C15 + j * 0xBF58476D1CE4E5B9 + k * 0x94D049BB133111EB) & mask
        h ^= h >> 30
        h = (h * 0xBF58476D1CE4E5B9) & mask
        h ^= h >> 27
        h = (h * 0x94D049BB133111EB) & mask
        h ^= h >> 31
        return h / mask

    field = _sample_field(dims, origin, spacing, noise)
    vertices, triangles = mc.extract(field, dims, origin, spacing, 0.5)
    _check(len(triangles) > 5000, f"expected a busy surface, got {len(triangles)} triangles")
    _assert_watertight(triangles, "noise field")

    seen = set()
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                case = 0
                for corner, (dx, dy, dz) in enumerate(mc.CORNERS):
                    index = ((k + dz) * ny + (j + dy)) * nx + (i + dx)
                    if field[index] >= 0.5:
                        case |= 1 << corner
                seen.add(case)
    _check(len(seen) == 256, f"only {len(seen)} of 256 cases exercised")


def test_field_below_iso_extracts_nothing():
    dims = (8, 8, 8)
    field = [0.0] * (8 * 8 * 8)
    vertices, triangles = mc.extract(field, dims, (0.0, 0.0, 0.0), 0.1, 0.5)
    _check(not vertices and not triangles, "an empty field should extract no geometry")


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"[marching_cubes] {test.__name__} passed")
    print(f"MARCHING CUBES TESTS PASSED ({len(tests)} tests)")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"MARCHING CUBES TESTS FAILED: {exc}")
        sys.exit(1)
