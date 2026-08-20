/* SPH pass 2: pressure + viscosity forces, accumulated into an acceleration.
 *
 * Kept separate from integration because a particle reads its neighbors'
 * positions and velocities here - integrating in the same pass would let some
 * particles move before others had read them.
 *
 * Also computes the XSPH correction, which pulls each particle's velocity
 * partway towards the kernel-weighted average of its neighbors'. Without it
 * this solver stays plausible for about ten frames and then gains energy
 * without bound: nothing else damps velocity noise at the particle scale, and
 * with a gamma-7 Tait pressure that noise feeds straight back into itself.
 * Unlike raising the viscosity, XSPH barely touches the bulk flow.
 */

#define XSPH_EPSILON 0.5

void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= particle_count()) {
    return;
  }

  ivec2 ti = particle_texel(i);
  vec3 pi = imageLoad(positions_img, ti).xyz;
  vec3 vi = imageLoad(velocities_img, ti).xyz;
  vec2 di = imageLoad(density_img, ti).xy;
  float density_i = di.x;
  float pressure_i = di.y;

  float h = f_sph.x;
  float h2 = h * h;
  float mass = f_sph.y;
  float viscosity = f_sim.x;
  float dt = f_sim.y;
  float spiky = spiky_grad_coef(h);
  float visc = visc_lap_coef(h);
  float poly6 = poly6_coef(h);

  vec3 f_pressure = vec3(0.0);
  vec3 f_viscous = vec3(0.0);
  vec3 v_xsph = vec3(0.0);
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
          if (j == i) {
            continue;
          }
          ivec2 tj = particle_texel(j);
          vec3 d = pi - imageLoad(positions_img, tj).xyz;
          float r = length(d);
          if (r >= h || r <= 1e-6) {
            continue;
          }

          vec2 dj = imageLoad(density_img, tj).xy;
          vec3 vj = imageLoad(velocities_img, tj).xyz;

          /* w_spiky_grad already carries the kernel's negative sign, so the
           * leading minus of the pressure-force term leaves a repulsion along
           * (pi - pj) whenever both pressures are positive. */
          f_pressure += -mass * (pressure_i + dj.y) / (2.0 * dj.x) *
                        w_spiky_grad(r, h, spiky) * (d / r);
          f_viscous += viscosity * mass * (vj - vi) / dj.x * w_visc_lap(r, h, visc);
          v_xsph += (mass / dj.x) * (vj - vi) * w_poly6(r * r, h2, poly6);
        }
      }
    }
  }

  vec3 accel = (f_pressure + f_viscous) / density_i + vec3(0.0, 0.0, f_sim.z);

  /* XSPH is a velocity correction, not a force; dividing by dt turns it into
   * an equivalent acceleration so it can ride along with everything else and
   * pass through the integrator's speed clamp. */
  accel += XSPH_EPSILON * v_xsph / max(dt, 1e-6);

  imageStore(forces_img, ti, vec4(accel, 0.0));
}
