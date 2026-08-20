/* SPH pass 1: density from the Poly6 kernel, then pressure from the Tait
 * equation of state.
 *
 * Cells are sized at least one smoothing radius across, so the 3x3x3 block
 * around a particle's own cell contains every neighbor within h.
 */

void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= particle_count()) {
    return;
  }

  vec3 pi = imageLoad(positions_img, particle_texel(i)).xyz;

  float h = f_sph.x;
  float h2 = h * h;
  float mass = f_sph.y;
  float rest_density = f_sph.z;
  float stiffness = f_sph.w;
  float coef = poly6_coef(h);

  float density = 0.0;
  ivec3 base = cell_coord(pi);

  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        ivec3 g = base + ivec3(dx, dy, dz);
        if (!cell_in_bounds(g)) {
          continue;
        }
        int c = cell_index(g);
        int start = int(imageLoad(cell_start_img, cell_texel(c)).x);
        if (start < 0) {
          continue;
        }
        int end = min(int(imageLoad(cell_end_img, cell_texel(c)).x), start + MAX_CELL_SCAN);

        for (int k = start; k < end; ++k) {
          int j = int(imageLoad(keys_img, particle_texel(k)).y);
          vec3 d = pi - imageLoad(positions_img, particle_texel(j)).xyz;
          density += mass * w_poly6(dot(d, d), h2, coef);
        }
      }
    }
  }

  /* Tait EOS with gamma = 7. Clamped at zero so a particle that strays below
   * rest density can't develop an attractive pressure and clump. */
  float ratio = max(density / rest_density, 0.0);
  float pressure = max(0.0, stiffness * (pow(ratio, 7.0) - 1.0));

  imageStore(density_img, particle_texel(i), vec4(max(density, 1e-6), pressure, 0.0, 0.0));
}
