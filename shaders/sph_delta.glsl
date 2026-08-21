/* PBF pass: position correction from the density-constraint lambdas
 * (Macklin & Muller 2013, eq. 12-14), including the s_corr tensile-instability
 * fix (eq. 13) that keeps particles from clumping into tight clusters under
 * negative pressure (e.g. a thin sheet of fluid stretched by gravity).
 *
 * s_corr's exponent (n=4) and evaluation point (delta_q = 0.2h) are the
 * paper's own defaults, hardcoded here rather than exposed as domain
 * properties - only its strength (i_sort.z, see sph.py's push_constant_values
 * for why it rides in an otherwise-unused bitonic-sort lane) is user-tunable.
 * i_sort.z holds floatBitsToInt(scorr_k); 0 disables the term entirely.
 *
 * Writes to delta_img rather than predicted_img directly: this pass reads
 * every neighbor's predicted position, so nothing here may write predicted_img
 * until every invocation that still needs to read it has finished - the same
 * Jacobi hazard the codebase already handles by splitting sph_force from
 * sph_integrate. sph_apply_delta.glsl is the pass that commits this delta.
 */

#define SCORR_N 4.0
#define SCORR_DELTA_Q 0.2

void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= particle_count()) {
    return;
  }

  vec3 pi = imageLoad(predicted_img, particle_texel(i)).xyz;
  float lambda_i = imageLoad(lambda_img, particle_texel(i)).y;

  float h = f_sph.x;
  float h2 = h * h;
  float rest_density = f_sph.z;
  float spiky = spiky_grad_coef(h);
  float poly6 = poly6_coef(h);
  float scorr_k = intBitsToFloat(i_sort.z);

  float w_q = w_poly6(pow(SCORR_DELTA_Q * h, 2.0), h2, poly6);

  vec3 delta = vec3(0.0);
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
          vec3 pj = imageLoad(predicted_img, particle_texel(j)).xyz;
          vec3 d = pi - pj;
          float r2 = dot(d, d);
          if (r2 >= h2 || r2 <= 1e-12) {
            continue;
          }
          float r = sqrt(r2);
          float lambda_j = imageLoad(lambda_img, particle_texel(j)).y;

          float s_corr = 0.0;
          if (scorr_k > 0.0 && w_q > 1e-9) {
            float ratio = w_poly6(r2, h2, poly6) / w_q;
            s_corr = -scorr_k * pow(max(ratio, 0.0), SCORR_N);
          }

          delta += (lambda_i + lambda_j + s_corr) * w_spiky_grad(r, h, spiky) * (d / r);
        }
      }
    }
  }

  imageStore(delta_img, particle_texel(i), vec4(delta / rest_density, 0.0));
}
