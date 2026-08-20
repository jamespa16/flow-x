/* Phase 6 pass: splat particle mass onto the surface grid.
 *
 * One invocation per surface lattice point, gathering from the particles
 * rather than scattering from them: a scatter would need imageAtomicAdd, which
 * doesn't compile on Blender's Metal backend (the same constraint that made
 * the neighbour search a bitonic sort - see solver/sph.py).
 *
 * The gather reuses the spatial hash the SPH passes already built this
 * substep, so this costs one more dispatch and no extra bookkeeping.
 *
 * This pass shares the SPH push-constant block, with two slots repurposed -
 * the block is exactly at the 128-byte budget, so there's no room to add:
 *
 *   i_surface = (samples.xyz, floatBitsToInt(sample_spacing))  [was i_sort]
 *   f_sph.x   = the *surface* kernel radius, not the solver's smoothing
 *               radius. solver/surface.py sizes it to the coarser of the two
 *               grids so a surface grid coarser than the fluid still gets a
 *               smooth field, and clamps it so the fixed cell search below
 *               stays exhaustive.
 */

/* The spatial hash's cells are at least one solver smoothing radius across, so
 * this radius of cells covers a surface kernel up to twice that. surface.py
 * clamps the kernel to match; going wider costs cells cubed for little gain. */
#define SURFACE_CELL_RADIUS 2

void main()
{
  int s = int(gl_GlobalInvocationID.x);
  ivec3 samples = i_surface.xyz;
  if (s >= samples.x * samples.y * samples.z) {
    return;
  }

  ivec3 g = ivec3(s % samples.x, (s / samples.x) % samples.y, s / (samples.x * samples.y));
  float spacing = intBitsToFloat(i_surface.w);

  /* Samples sit on lattice points starting at the domain's low corner, so the
   * marching cubes cells between them tile the domain exactly. */
  vec3 p = f_lo.xyz + vec3(g) * spacing;

  float h = f_sph.x;
  float h2 = h * h;
  float mass = f_sph.y;
  float rest_density = f_sph.z;
  float coef = poly6_coef(h);

  float density = 0.0;
  ivec3 base = cell_coord(p);

  for (int dz = -SURFACE_CELL_RADIUS; dz <= SURFACE_CELL_RADIUS; ++dz) {
    for (int dy = -SURFACE_CELL_RADIUS; dy <= SURFACE_CELL_RADIUS; ++dy) {
      for (int dx = -SURFACE_CELL_RADIUS; dx <= SURFACE_CELL_RADIUS; ++dx) {
        ivec3 c = base + ivec3(dx, dy, dz);
        if (!cell_in_bounds(c)) {
          continue;
        }
        int cell = cell_index(c);
        int start = int(imageLoad(cell_start_img, cell_texel(cell)).x);
        if (start < 0) {
          continue;
        }
        int end = min(int(imageLoad(cell_end_img, cell_texel(cell)).x), start + MAX_CELL_SCAN);

        for (int k = start; k < end; ++k) {
          int j = int(imageLoad(keys_img, particle_texel(k)).y);
          vec3 d = p - imageLoad(positions_img, particle_texel(j)).xyz;
          density += mass * w_poly6(dot(d, d), h2, coef);
        }
      }
    }
  }

  /* Normalized by rest density, so the iso-value the user sets is a fraction
   * of "fully dense fluid" and stays meaningful across resolutions and fluid
   * parameters rather than being a raw kg/m^3 figure to re-tune. */
  float value = density / max(rest_density, 1e-6);

  /* Carve the colliders back out. The kernel has a radius of support, so
   * without this the field bleeds a smoothing radius into a collider and the
   * extracted surface buries its geometry instead of pooling against it. */
  if (collider_voxel_size() > 0.0 && collider_occupied(collider_coord(p))) {
    value = 0.0;
  }

  /* Surface samples are indexed like cells - a flat array wrapped into a 2D
   * texture at the same width - because 1D textures can't be read back, and
   * this grid has to reach the CPU for marching cubes. */
  imageStore(surface_img, cell_texel(s), vec4(value));
}
