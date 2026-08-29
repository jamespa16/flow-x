/* Surface tension pass 1 (Morris 2000): per-particle color-field gradient
 * (surface normal) and curvature, read by sph_predict as a cohesion force
 * alongside gravity.
 *
 * color_i = sum_j (m_j / density_j) * W_poly6(r) is a smoothed indicator that
 * is ~1 deep inside the fluid and falls off near the surface; its gradient n_i
 * therefore points inward wherever a particle is near the surface, and is ~0
 * for interior particles - which is exactly the "only pull at the surface"
 * behaviour surface tension needs, for free from the field shape. Curvature is
 * the color field's Laplacian, using the same viscosity-kernel Laplacian
 * machinery (w_visc_lap/visc_lap_coef) that XSPH's WCSPH-era ancestor already
 * used for a different purpose - no new kernel family needed for this model,
 * unlike Akinci-style cohesion.
 *
 * Runs on the predicted positions and the grid built from them *last* substep:
 * normals are therefore one substep stale. That is a deliberate tradeoff, not
 * an oversight - genuinely current normals would need a second grid build
 * before sph_predict, every substep, just for this.
 */
FLOWX_KERNEL void sph_normal(FLOWX_CONST_DEVICE float4 *predicted [[buffer(BUF_PREDICTED)]],
                             FLOWX_CONST_DEVICE float4 *lambda [[buffer(BUF_LAMBDA)]],
                             FLOWX_DEVICE float4 *normal [[buffer(BUF_NORMAL)]],
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
  float poly6 = poly6_coef(h);
  float visc_lap = visc_lap_coef(h);

  float3 n_sum = float3(0.0f);
  float laplacian = 0.0f;
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
          if (r2 >= h2) {
            continue;
          }
          float density_j = max(lambda[j].x, 1e-6f);
          float weight = mass / density_j;

          n_sum += weight * w_poly6_grad(d, r2, h2, poly6);
          laplacian += weight * w_visc_lap(sqrt(r2), h, visc_lap);
        }
      }
    }
  }

  float mag = length(n_sum);
  float curvature = (mag > 1e-4f) ? -laplacian / mag : 0.0f;
  normal[i] = float4(n_sum, curvature);
}
