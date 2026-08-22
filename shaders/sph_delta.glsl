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
 * Two things here are *not* transcribed from the paper's equations as written,
 * because the paper works in a normalization (mass = 1, rest density as a
 * number density) this solver does not:
 *
 *   - The `mass` factor on the kernel gradient. grad_pk(C_i) carries the same
 *     mass/rest_density that sph_lambda.glsl's density sum does, so dropping
 *     it here does not merely rescale the step - it makes delta dimensionally
 *     wrong (length/mass rather than length) and over-relaxes every Jacobi
 *     iteration by 1/mass, which at the shipped defaults is 8x. The loop then
 *     oscillates instead of converging.
 *
 *   - The (inv_denom_i + inv_denom_j) factor on s_corr. lambda has units of
 *     length^2 here and runs ~1e-4 at the shipped defaults, so the paper's
 *     bare k = 0.1 added to (lambda_i + lambda_j) is not the small nudge it is
 *     in the paper - it is ~9x the density term, and the constraint stops
 *     mattering. Dividing by the same denominator lambda was built from puts
 *     s_corr back in lambda's units, which makes the term exactly "as if C_i
 *     carried an extra artificial surplus of k*(W/W_q)^n" - its actual intent -
 *     and restores k as a dimensionless 0..1 knob. Applying each particle's
 *     own inv_denom keeps the pair bracket symmetric, so the delta stays
 *     antisymmetric in i<->j and momentum is still conserved.
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
  vec4 li = imageLoad(lambda_img, particle_texel(i));
  float lambda_i = li.y;
  float inv_denom_i = li.z;

  float h = f_sph.x;
  float h2 = h * h;
  float mass = f_sph.y;
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
          vec4 lj = imageLoad(lambda_img, particle_texel(j));
          float lambda_j = lj.y;

          float s_corr = 0.0;
          if (scorr_k > 0.0 && w_q > 1e-9) {
            float ratio = w_poly6(r2, h2, poly6) / w_q;
            s_corr = -scorr_k * pow(max(ratio, 0.0), SCORR_N) * (inv_denom_i + lj.z);
          }

          delta += (lambda_i + lambda_j + s_corr) * mass * w_spiky_grad(r, h, spiky) * (d / r);
        }
      }
    }
  }

  imageStore(delta_img, particle_texel(i), vec4(delta / rest_density, 0.0));
}
