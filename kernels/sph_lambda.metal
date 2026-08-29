/* PBF pass: density (Poly6) from the predicted positions, then the
 * density-constraint Lagrange multiplier lambda_i (Macklin & Muller 2013,
 * eq. 9-11).
 *
 * C_i = density_i / rest_density - 1 is the constraint each particle should
 * satisfy exactly; lambda_i is how far a first-order position step would need
 * to move to satisfy it, weighted by how "responsive" the local neighbourhood
 * is to that move (the denominator - a nearly-empty neighbourhood has a small
 * gradient sum and would demand an enormous, destabilizing step, which is
 * exactly what P.relaxation, the CFM-style epsilon, is there to bound).
 *
 * Reads the predicted positions (this substep's unconstrained guess, already
 * grid-built against by the time this pass runs) rather than the committed
 * ones.
 */
FLOWX_KERNEL void sph_lambda(FLOWX_CONST_DEVICE float4 *predicted [[buffer(BUF_PREDICTED)]],
                             FLOWX_DEVICE float4 *lambda [[buffer(BUF_LAMBDA)]],
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

  float h = P.smoothing_radius;
  float h2 = h * h;
  float mass = P.mass;
  float rest_density = P.rest_density;
  float epsilon = P.relaxation;
  float poly6 = poly6_coef(h);
  float spiky = spiky_grad_coef(h);

  float density = 0.0f;
  float3 grad_self = float3(0.0f);
  float grad_norm2_sum = 0.0f;
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
          float3 pj = predicted[j].xyz;
          float3 d = pi - pj;
          float r2 = dot(d, d);
          density += mass * w_poly6(r2, h2, poly6);

          if (j == i || r2 >= h2 || r2 <= 1e-12f) {
            continue;
          }
          float r = sqrt(r2);
          float3 grad_ij = mass * w_spiky_grad(r, h, spiky) * (d / r);
          grad_self += grad_ij;
          grad_norm2_sum += dot(grad_ij, grad_ij);
        }
      }
    }
  }

  float constraint = max(density, 1e-6f) / rest_density - 1.0f;
  float denom =
      (dot(grad_self, grad_self) + grad_norm2_sum) / (rest_density * rest_density) + epsilon;
  float inv_denom = 1.0f / max(denom, 1e-9f);
  float lambda_i = -constraint * inv_denom;

  /* z carries 1/denom so sph_delta can express s_corr in lambda's own units -
   * see that file. It is this pass's denominator, not a new quantity, so it is
   * published here rather than recomputed there over the same neighbour loop. */
  lambda[i] = float4(max(density, 1e-6f), lambda_i, inv_denom, 0.0f);
}
