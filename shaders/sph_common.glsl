/* Shared prelude for every Phase 4/5 SPH compute pass.
 *
 * solver/gpu_util.py prepends this file to each pass's own source, and
 * solver/sph.py declares an identical push-constant block on every SPH shader
 * so the helpers below can be written once. The block is exactly 128 bytes,
 * the push-constant budget Blender's backends guarantee - there is no spare
 * room left, so packing (see i_collider.w below) is how new parameters have
 * to arrive from here on.
 *
 *   i_layout   = (particle_count, particle_tex_width, cell_tex_width, sorted_count)
 *   i_grid     = (cells_x, cells_y, cells_z, cell_count)
 *   i_sort     = (bitonic_k, bitonic_j, unused, unused)
 *   f_lo       = (domain_min.xyz, cell_size)
 *   f_hi       = (domain_max.xyz, particle_radius)
 *   f_sph      = (smoothing_radius, particle_mass, rest_density, stiffness)
 *   f_sim      = (viscosity, dt, gravity, boundary_damping)
 *   i_collider = (collider_dims.xyz, floatBitsToInt(collider_voxel_size))
 */

#define PI 3.14159265358979323846

/* Defensive bound on a single cell's particle run. A cell range is built from
 * sorted keys and should always be small, but an unbounded data-dependent loop
 * in a compute shader is a hang waiting to happen. */
#define MAX_CELL_SCAN 512

int particle_count()
{
  return i_layout.x;
}

/* Particle state lives in 2D textures rather than 1D: Blender's GPUTexture
 * read-back reports a zero-length first dimension for 1D textures, so nothing
 * can be pulled back to the CPU for the viewport viz. */
ivec2 particle_texel(int i)
{
  return ivec2(i % i_layout.y, i / i_layout.y);
}

ivec2 cell_texel(int c)
{
  return ivec2(c % i_layout.z, c / i_layout.z);
}

ivec3 cell_coord(vec3 p)
{
  ivec3 g = ivec3(floor((p - f_lo.xyz) / f_lo.w));
  return clamp(g, ivec3(0), i_grid.xyz - 1);
}

int cell_index(ivec3 g)
{
  return (g.z * i_grid.y + g.y) * i_grid.x + g.x;
}

bool cell_in_bounds(ivec3 g)
{
  return all(greaterThanEqual(g, ivec3(0))) && all(lessThan(g, i_grid.xyz));
}

/* Kernel normalization factors. Each involves a pow() of the smoothing radius,
 * so callers hoist these out of the neighbor loop and pass the result down. */
float poly6_coef(float h)
{
  return 315.0 / (64.0 * PI * pow(h, 9.0));
}

float spiky_grad_coef(float h)
{
  return -45.0 / (PI * pow(h, 6.0));
}

float visc_lap_coef(float h)
{
  return 45.0 / (PI * pow(h, 6.0));
}

/* Poly6 density kernel, in terms of squared distance to avoid a sqrt. */
float w_poly6(float r2, float h2, float coef)
{
  if (r2 >= h2) {
    return 0.0;
  }
  float d = h2 - r2;
  return coef * d * d * d;
}

/* Magnitude of the Spiky gradient along the separation direction. Already
 * carries the kernel's negative sign via spiky_grad_coef(). */
float w_spiky_grad(float r, float h, float coef)
{
  if (r >= h) {
    return 0.0;
  }
  float d = h - r;
  return coef * d * d;
}

/* Laplacian of the viscosity kernel. */
float w_visc_lap(float r, float h, float coef)
{
  if (r >= h) {
    return 0.0;
  }
  return coef * (h - r);
}

/* Phase 5: the collider occupancy grid built in collision/__init__.py, shares
 * f_lo.xyz as its origin so only a voxel size is needed to locate a point in
 * it. A voxel size of 0 means "no collider tagged" - the empty placeholder
 * texture bound in that case reads as unoccupied everywhere. */

ivec3 collider_dims()
{
  return i_collider.xyz;
}

float collider_voxel_size()
{
  return intBitsToFloat(i_collider.w);
}

ivec3 collider_coord(vec3 p)
{
  return ivec3(floor((p - f_lo.xyz) / collider_voxel_size()));
}

bool collider_in_bounds(ivec3 c)
{
  return all(greaterThanEqual(c, ivec3(0))) && all(lessThan(c, collider_dims()));
}

bool collider_occupied(ivec3 c)
{
  if (!collider_in_bounds(c)) {
    return false;
  }
  return imageLoad(collider_img, c).r > 0.5;
}
