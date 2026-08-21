/* PBF pass: XSPH velocity smoothing (Schechter & Bridson), ported from
 * WCSPH's sph_force.glsl. Pulls each particle's velocity partway towards the
 * kernel-weighted average of its neighbors', which is what keeps the
 * constraint-solve's particle-scale velocity noise from feeding back into
 * itself frame over frame - without it this solver drifted the same way the
 * WCSPH one did before XSPH was added there.
 *
 * Reads velocities_img fully raw (every particle already holds its
 * (p_pred - p_old)/dt velocity from sph_velocity.glsl, untouched since).
 * Writes the correction to delta_img rather than velocities_img directly:
 * every invocation here reads its neighbors' velocities, so nothing may
 * overwrite velocities_img until this whole pass has finished reading it -
 * sph_finalize.glsl is what applies this correction.
 *
 * density_img is now lambda_img (see sph_lambda.glsl); its density channel
 * (.x) was computed against this substep's *pre-correction* predicted
 * positions, so the weighting here is one iteration-loop stale. That is the
 * same order of approximation XSPH already tolerated under WCSPH (there it
 * read a density computed before force integration moved anything) and is
 * not worth a second density pass to correct.
 */

#define XSPH_EPSILON 0.5

void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= particle_count()) {
    return;
  }

  ivec2 ti = particle_texel(i);
  vec3 pi = imageLoad(predicted_img, ti).xyz;
  vec3 vi = imageLoad(velocities_img, ti).xyz;

  float h = f_sph.x;
  float h2 = h * h;
  float mass = f_sph.y;
  float poly6 = poly6_coef(h);

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
          vec3 d = pi - imageLoad(predicted_img, tj).xyz;
          float r2 = dot(d, d);
          if (r2 >= h2) {
            continue;
          }
          float density_j = max(imageLoad(lambda_img, tj).x, 1e-6);
          vec3 vj = imageLoad(velocities_img, tj).xyz;
          v_xsph += (mass / density_j) * (vj - vi) * w_poly6(r2, h2, poly6);
        }
      }
    }
  }

  imageStore(delta_img, ti, vec4(XSPH_EPSILON * v_xsph, 0.0));
}
