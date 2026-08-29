/* Splat particle mass onto the surface grid.
 *
 * One invocation per surface lattice point, gathering from the particles
 * rather than scattering onto them. Under Blender's `gpu` module a scatter was
 * impossible - it needs an atomic add, and image atomics would not compile on
 * that Metal backend. A scatter is available now; the gather is kept because
 * it also reuses the spatial hash the SPH passes already built this substep,
 * so it costs one dispatch and no extra bookkeeping either way.
 *
 * The surface grid's own geometry used to be smuggled through slots borrowed
 * from the SPH block (the grid corner rode in the domain-max lanes, the kernel
 * radius overwrote the smoothing radius) because that block was full at
 * exactly 128 bytes. It has its own fields in Params now.
 */

/* The spatial hash's cells are at least one solver smoothing radius across, so
 * this radius of cells covers a surface kernel up to twice that. surface.py
 * clamps the kernel to match; going wider costs cells cubed for little gain. */
#define SURFACE_CELL_RADIUS 2

FLOWX_KERNEL void surface_splat(FLOWX_CONST_DEVICE float4 *positions [[buffer(BUF_POSITIONS)]],
                                FLOWX_CONST_DEVICE float4 *keys [[buffer(BUF_KEYS)]],
                                FLOWX_CONST_DEVICE float *cell_start [[buffer(BUF_CELL_START)]],
                                FLOWX_CONST_DEVICE float *cell_end [[buffer(BUF_CELL_END)]],
                                FLOWX_CONST_DEVICE float *collider [[buffer(BUF_COLLIDER)]],
                                FLOWX_DEVICE float *surface [[buffer(BUF_SURFACE)]],
                                FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                                FLOWX_TID)
{
  int s = int(tid);
  int3 samples = int3(P.surface_x, P.surface_y, P.surface_z);
  if (s >= samples.x * samples.y * samples.z) {
    return;
  }

  int3 g = int3(s % samples.x, (s / samples.x) % samples.y, s / (samples.x * samples.y));
  float spacing = P.surface_spacing;

  /* Samples sit on lattice points starting at the surface grid's low corner
   * (one kernel radius before the domain's low corner), so the marching cubes
   * cells between them tile the domain - and the kernel's bled support past
   * every wall - exactly. */
  float3 p = float3(P.surface_lo_x, P.surface_lo_y, P.surface_lo_z) + float3(g) * spacing;

  /* The *surface* kernel radius, not the solver's smoothing radius:
   * solver/surface.py sizes it to the coarser of the two grids so a surface
   * grid coarser than the fluid still gets a smooth field, and clamps it so
   * the fixed cell search below stays exhaustive. */
  float h = P.surface_kernel_radius;
  float h2 = h * h;
  float mass = P.mass;
  float rest_density = P.rest_density;
  float coef = poly6_coef(h);

  float density = 0.0f;
  int3 base = cell_coord(P, p);

  for (int dz = -SURFACE_CELL_RADIUS; dz <= SURFACE_CELL_RADIUS; ++dz) {
    for (int dy = -SURFACE_CELL_RADIUS; dy <= SURFACE_CELL_RADIUS; ++dy) {
      for (int dx = -SURFACE_CELL_RADIUS; dx <= SURFACE_CELL_RADIUS; ++dx) {
        int3 c = base + int3(dx, dy, dz);
        if (!cell_in_bounds(P, c)) {
          continue;
        }
        int cell = cell_index(P, c);
        int start = int(cell_start[cell]);
        if (start < 0) {
          continue;
        }
        int end = min(int(cell_end[cell]), start + MAX_CELL_SCAN);

        for (int k = start; k < end; ++k) {
          int j = int(keys[k].y);
          float3 d = p - positions[j].xyz;
          density += mass * w_poly6(dot(d, d), h2, coef);
        }
      }
    }
  }

  /* Normalized by rest density, so the iso-value the user sets is a fraction
   * of "fully dense fluid" and stays meaningful across resolutions and fluid
   * parameters rather than being a raw kg/m^3 figure to re-tune. */
  float value = density / max(rest_density, 1e-6f);

  /* Carve the colliders back out. The kernel has a radius of support, so
   * without this the field bleeds a smoothing radius into a collider and the
   * extracted surface buries its geometry instead of pooling against it. */
  if (P.collider_voxel > 0.0f && collider_occupied(P, collider, collider_coord(P, p))) {
    value = 0.0f;
  }

  surface[s] = value;
}
