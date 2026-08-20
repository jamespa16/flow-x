"""Phase 6: a self-contained marching cubes implementation.

The add-on shouldn't bundle `scikit-image` just for one algorithm, so the
classic 256-case lookup table lives here. It is *derived* at import rather
than transcribed: `_build_case_table()` builds all 256 entries from the cube's
own geometry in a few milliseconds, which avoids a 256-row data blob whose
typos would only ever show up as an occasional hole in the fluid surface.

Deriving it also fixes the well-known flaw in the widely-copied table: on a
face whose two diagonal corners are inside and the other two outside, the
contour can be paired two ways, and the published table doesn't always pick
the same one for the two cubes sharing that face - so the extracted mesh has
holes there. The construction below builds each case out of per-face contour
segments, and the pairing rule depends only on the four corner signs of the
face, in the same cyclic order up to reversal for either cube touching it.
Both cubes therefore always agree and the output is watertight (which
`scripts/test_marching_cubes.py` checks over every case).

The module is deliberately free of `bpy`/`gpu` imports so it can be tested
with a plain Python interpreter, outside Blender.
"""

# Corner numbering, and the 12 edges joining them, in the conventional
# marching-cubes order. Only this module depends on the ordering.
CORNERS = (
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
)

EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)

# The six faces as corner quads. The cyclic order here is arbitrary;
# _oriented_faces() flips each one as needed so they all wind counterclockwise
# seen from outside the cube, which is what makes the contour pairing rule
# consistent between neighbouring cubes.
_FACES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (3, 2, 6, 7),
    (0, 3, 7, 4),
    (1, 2, 6, 5),
)


def _newell_normal(points):
    nx = ny = nz = 0.0
    for i, (ax, ay, az) in enumerate(points):
        bx, by, bz = points[(i + 1) % len(points)]
        nx += (ay - by) * (az + bz)
        ny += (az - bz) * (ax + bx)
        nz += (ax - bx) * (ay + by)
    return nx, ny, nz


def _oriented_faces():
    """The six faces, each wound counterclockwise as seen from outside."""
    faces = []
    for face in _FACES:
        points = [CORNERS[c] for c in face]
        nx, ny, nz = _newell_normal(points)
        # Face centre relative to the cube centre is the outward direction.
        cx = sum(p[0] for p in points) / 4.0 - 0.5
        cy = sum(p[1] for p in points) / 4.0 - 0.5
        cz = sum(p[2] for p in points) / 4.0 - 0.5
        outward = nx * cx + ny * cy + nz * cz
        faces.append(face if outward > 0.0 else tuple(reversed(face)))
    return tuple(faces)


def _face_segments(face, inside, edge_of):
    """Directed contour segments crossing one face, as (from_edge, to_edge).

    Walking the face's corners in order, an inside->outside step opens a
    segment and an outside->inside step closes one; because the signs alternate
    along the walk, so do the transitions, and pairing each opening with the
    next closing is unambiguous. Reversing the walk (which is how the cube on
    the other side of this face sees it) swaps both the roles and the direction,
    so it produces the same pairs with each segment reversed - exactly the
    shared-edge relationship two adjacent triangles need.
    """
    transitions = []
    for idx, a in enumerate(face):
        b = face[(idx + 1) % len(face)]
        if inside[a] and not inside[b]:
            transitions.append((True, edge_of[frozenset((a, b))]))
        elif inside[b] and not inside[a]:
            transitions.append((False, edge_of[frozenset((a, b))]))

    segments = []
    count = len(transitions)
    for idx, (opening, edge) in enumerate(transitions):
        if not opening:
            continue
        for step in range(1, count):
            closing, other = transitions[(idx + step) % count]
            if not closing:
                segments.append((edge, other))
                break
    return segments


def _build_case_table():
    """Triangle lists, as edge-index triples, for all 256 corner sign cases."""
    faces = _oriented_faces()
    edge_of = {frozenset(edge): index for index, edge in enumerate(EDGES)}

    table = []
    for case in range(256):
        inside = [bool(case & (1 << corner)) for corner in range(8)]

        # Every edge with a sign change is an opening on exactly one of its two
        # faces and a closing on the other, so the segments form disjoint closed
        # loops around the surface patch.
        successor = {}
        for face in faces:
            for start, end in _face_segments(face, inside, edge_of):
                successor[start] = end

        triangles = []
        visited = set()
        for first in successor:
            if first in visited:
                continue
            loop = []
            edge = first
            while edge not in visited:
                visited.add(edge)
                loop.append(edge)
                edge = successor[edge]
            # Fan-triangulate; the loop is planar only by accident, which is
            # true of the published table's triangulations too. The loop runs
            # clockwise seen from outside the surface, so the fan is emitted
            # reversed to leave normals pointing away from the fluid - which
            # scripts/test_marching_cubes.py pins down via the sign of the
            # extracted sphere's volume.
            for i in range(1, len(loop) - 1):
                triangles.append((loop[0], loop[i + 1], loop[i]))
        table.append(tuple(triangles))
    return tuple(table)


TRI_TABLE = _build_case_table()

# Which of the 12 edges each case interpolates a vertex on. Derived from the
# corner signs directly, and used to skip cases with nothing to emit.
EDGE_TABLE = tuple(
    sum(
        1 << index
        for index, (a, b) in enumerate(EDGES)
        if bool(case & (1 << a)) != bool(case & (1 << b))
    )
    for case in range(256)
)


# For each edge: the corner it starts at, the corner it ends at, and the axis
# it runs along. Ordered so the start corner is always the lower one on that
# axis, which makes the interpolation parameter and the shared-vertex key below
# independent of which cube is asking.
def _build_edge_walk():
    walk = []
    for a, b in EDGES:
        axis = next(i for i in range(3) if CORNERS[a][i] != CORNERS[b][i])
        walk.append((a, b, axis) if CORNERS[a][axis] < CORNERS[b][axis] else (b, a, axis))
    return tuple(walk)


_EDGE_WALK = _build_edge_walk()


def extract(field, dims, origin, spacing, iso):
    """Extract an iso-surface mesh from a scalar field sampled on a lattice.

    `field` is a flat sequence of `nx * ny * nz` samples indexed as
    ``(k * ny + j) * nx + i``, taken at ``origin + (i, j, k) * spacing``.
    Samples at or above `iso` are inside the surface.

    Returns ``(vertices, triangles)``: world-space vertex tuples, and index
    triples into them. Vertices are shared between the cubes meeting on a
    lattice edge, so the result is an indexed mesh rather than a triangle soup.
    """
    nx, ny, nz = dims
    if nx < 2 or ny < 2 or nz < 2:
        return [], []

    ox, oy, oz = origin
    nxy = nx * ny
    corner_offset = [dx + dy * nx + dz * nxy for dx, dy, dz in CORNERS]
    # Hoisted into locals: this is the add-on's hottest Python loop.
    tri_table = TRI_TABLE
    edge_walk = _EDGE_WALK
    corners = CORNERS

    vertices = []
    triangles = []
    vertex_of_edge = {}

    for k in range(nz - 1):
        for j in range(ny - 1):
            row = j * nx + k * nxy
            for i in range(nx - 1):
                base = row + i
                v0 = field[base + corner_offset[0]]
                v1 = field[base + corner_offset[1]]
                v2 = field[base + corner_offset[2]]
                v3 = field[base + corner_offset[3]]
                v4 = field[base + corner_offset[4]]
                v5 = field[base + corner_offset[5]]
                v6 = field[base + corner_offset[6]]
                v7 = field[base + corner_offset[7]]

                case = 0
                if v0 >= iso:
                    case |= 1
                if v1 >= iso:
                    case |= 2
                if v2 >= iso:
                    case |= 4
                if v3 >= iso:
                    case |= 8
                if v4 >= iso:
                    case |= 16
                if v5 >= iso:
                    case |= 32
                if v6 >= iso:
                    case |= 64
                if v7 >= iso:
                    case |= 128

                # Wholly inside or wholly outside: no surface crosses here, and
                # this is the overwhelmingly common case.
                if case == 0 or case == 255:
                    continue

                values = (v0, v1, v2, v3, v4, v5, v6, v7)
                for triangle in tri_table[case]:
                    face = []
                    for edge in triangle:
                        lo_corner, hi_corner, axis = edge_walk[edge]
                        key = (base + corner_offset[lo_corner], axis)
                        index = vertex_of_edge.get(key)
                        if index is None:
                            lo_value = values[lo_corner]
                            hi_value = values[hi_corner]
                            delta = hi_value - lo_value
                            t = 0.5 if delta == 0.0 else (iso - lo_value) / delta
                            dx, dy, dz = corners[lo_corner]
                            point = [
                                ox + (i + dx) * spacing,
                                oy + (j + dy) * spacing,
                                oz + (k + dz) * spacing,
                            ]
                            point[axis] += t * spacing
                            index = len(vertices)
                            vertices.append(tuple(point))
                            vertex_of_edge[key] = index
                        face.append(index)
                    triangles.append(tuple(face))

    return vertices, triangles
