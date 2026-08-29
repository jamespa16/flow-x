/* PBF pass: XSPH velocity smoothing (Schechter & Bridson). Pulls each
 * particle's velocity partway towards the kernel-weighted average of its
 * neighbours', which is what keeps the constraint-solve's particle-scale
 * velocity noise from feeding back into itself frame over frame - without it
 * this solver drifted the same way the WCSPH one did before XSPH was added
 * there.
 *
 * Reads velocities fully raw (every particle already holds its
 * (p_pred - p_old)/dt velocity from sph_velocity, untouched since). Writes the
 * correction to the delta buffer rather than the velocity buffer directly:
 * every invocation here reads its neighbours' velocities, so nothing may
 * overwrite them until this whole pass has finished reading - sph_finalize is
 * what applies this correction.
 *
 * The density channel of the lambda buffer was computed against this substep's
 * *pre-correction* predicted positions, so the weighting here is one
 * iteration-loop stale. That is the same order of approximation XSPH already
 * tolerated under WCSPH and is not worth a second density pass to correct.
 *
 * The blend strength is P.viscosity (the Viscosity domain property) rather
 * than a fixed constant, so raising it visibly thickens/damps the fluid as its
 * description promises; the solver's CFL term already bounds dt against this
 * same value to keep the blend stable.
 */
FLOWX_KERNEL void sph_xsph(FLOWX_CONST_DEVICE float4 *velocities [[buffer(BUF_VELOCITIES)]],
                           FLOWX_CONST_DEVICE float4 *lambda [[buffer(BUF_LAMBDA)]],
                           FLOWX_CONST_DEVICE float4 *predicted [[buffer(BUF_PREDICTED)]],
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
  float3 vi = velocities[i].xyz;

  float h = P.smoothing_radius;
  float h2 = h * h;
  float mass = P.mass;
  float poly6 = poly6_coef(h);

  float3 v_xsph = float3(0.0f);
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
          float3 d = pi - predicted[j].xyz;
          float r2 = dot(d, d);
          if (r2 >= h2) {
            continue;
          }
          float density_j = max(lambda[j].x, 1e-6f);
          float3 vj = velocities[j].xyz;
          v_xsph += (mass / density_j) * (vj - vi) * w_poly6(r2, h2, poly6);
        }
      }
    }
  }

  delta[i] = float4(P.viscosity * v_xsph, 0.0f);
}
