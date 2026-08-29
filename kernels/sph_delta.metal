/* PBF pass: position correction from the density-constraint lambdas (Macklin &
 * Muller 2013, eq. 12-14), including the s_corr tensile-instability fix
 * (eq. 13) that keeps particles from clumping into tight clusters under
 * negative pressure (e.g. a thin sheet of fluid stretched by gravity).
 *
 * s_corr's exponent (n=4) and evaluation point (delta_q = 0.2h) are the
 * paper's own defaults, hardcoded here rather than exposed as domain
 * properties - only its strength, P.scorr_k, is user-tunable; 0 disables the
 * term entirely.
 *
 * Two things here are *not* transcribed from the paper's equations as written,
 * because the paper works in a normalization (mass = 1, rest density as a
 * number density) this solver does not:
 *
 *   - The `mass` factor on the kernel gradient. grad_pk(C_i) carries the same
 *     mass/rest_density that sph_lambda's density sum does, so dropping it here
 *     does not merely rescale the step - it makes delta dimensionally wrong
 *     (length/mass rather than length) and over-relaxes every Jacobi iteration
 *     by 1/mass, which at the shipped defaults is 8x. The loop then oscillates
 *     instead of converging.
 *
 *   - The (inv_denom_i + inv_denom_j) factor on s_corr. lambda has units of
 *     length^2 here and runs ~1e-4 at the shipped defaults, so the paper's bare
 *     k = 0.1 added to (lambda_i + lambda_j) is not the small nudge it is in
 *     the paper - it is ~9x the density term, and the constraint stops
 *     mattering. Dividing by the same denominator lambda was built from puts
 *     s_corr back in lambda's units, which makes the term exactly "as if C_i
 *     carried an extra artificial surplus of k*(W/W_q)^n" - its actual intent -
 *     and restores k as a dimensionless 0..1 knob. Applying each particle's own
 *     inv_denom keeps the pair bracket symmetric, so the delta stays
 *     antisymmetric in i<->j and momentum is still conserved.
 *
 * Writes to the delta buffer rather than the predicted one directly: this pass
 * reads every neighbour's predicted position, so nothing here may write it
 * until every invocation that still needs to read it has finished.
 * sph_apply_delta is the pass that commits this.
 */

#define SCORR_N 4.0f
#define SCORR_DELTA_Q 0.2f

FLOWX_KERNEL void sph_delta(FLOWX_CONST_DEVICE float4 *predicted [[buffer(BUF_PREDICTED)]],
                            FLOWX_CONST_DEVICE float4 *lambda [[buffer(BUF_LAMBDA)]],
                            FLOWX_DEVICE float4 *delta [[buffer(BUF_DELTA)]],
                            FLOWX_CONST_DEVICE float4 *keys [[buffer(BUF_KEYS)]],
                            FLOWX_CONST_DEVICE float *cell_start [[buffer(BUF_CELL_START)]],
                            FLOWX_CONST_DEVICE float *cell_end [[buffer(BUF_CELL_END)]],
                            FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                            FLOWX_TID)
{
  int i = int(tid);
  if (i >= P.particle_count) {
    return;
  }

  float3 pi = predicted[i].xyz;
  float4 li = lambda[i];
  float lambda_i = li.y;
  float inv_denom_i = li.z;

  float h = P.smoothing_radius;
  float h2 = h * h;
  float mass = P.mass;
  float rest_density = P.rest_density;
  float spiky = spiky_grad_coef(h);
  float poly6 = poly6_coef(h);
  float scorr_k = P.scorr_k;

  float w_q = w_poly6(pow(SCORR_DELTA_Q * h, 2.0f), h2, poly6);

  float3 d_sum = float3(0.0f);
  int3 base = cell_coord(P, pi);

  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        int3 g = base + int3(dx, dy, dz);
        if (!cell_in_bounds(P, g)) {
          continue;
        }
        int c = cell_index(P, g);
        int start = int(cell_start[c]);
        if (start < 0) {
          continue;
        }
        int end = min(int(cell_end[c]), start + MAX_CELL_SCAN);

        for (int k = start; k < end; ++k) {
          int j = int(keys[k].y);
          if (j == i) {
            continue;
          }
          float3 pj = predicted[j].xyz;
          float3 d = pi - pj;
          float r2 = dot(d, d);
          if (r2 >= h2 || r2 <= 1e-12f) {
            continue;
          }
          float r = sqrt(r2);
          float4 lj = lambda[j];
          float lambda_j = lj.y;

          float s_corr = 0.0f;
          if (scorr_k > 0.0f && w_q > 1e-9f) {
            float ratio = w_poly6(r2, h2, poly6) / w_q;
            s_corr = -scorr_k * pow(max(ratio, 0.0f), SCORR_N) * (inv_denom_i + lj.z);
          }

          d_sum += (lambda_i + lambda_j + s_corr) * mass * w_spiky_grad(r, h, spiky) * (d / r);
        }
      }
    }
  }

  delta[i] = float4(d_sum / rest_density, 0.0f);
}
