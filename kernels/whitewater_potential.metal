/* Whitewater pass 1: score every fluid particle's likelihood of throwing off
 * secondary (spray/foam/bubble) particles this frame, for the sort that
 * follows.
 *
 * A simplified, real-time-budget version of Ihmsen et al.'s three-potential
 * classification (trapped-air, wave-crest, kinetic-energy), computed from the
 * same neighbour loop and kernel gradient sph_normal already uses for surface
 * tension - but run unconditionally, since whitewater needs the surface shape
 * whether or not surface tension itself is enabled. It reuses the frame's
 * *last substep* grid, exactly like surface_splat does and for the same
 * reason: rebuilding a grid just for this would cost more than the
 * one-substep staleness is worth.
 *
 * Output, per fluid particle: x = combined score (sort key), y = source
 * particle index, z = trapped-air sub-score (bubble driver), w = wave-crest
 * sub-score (spray/foam driver). Padding slots - past particle_count, out to
 * the power-of-two sorted_count the bitonic sort needs - get a sentinel score
 * of -1 so a *descending* sort pushes them to the tail, the same trick
 * sph_grid_key uses for its own padding.
 */
FLOWX_KERNEL void whitewater_potential(
    FLOWX_CONST_DEVICE float4 *positions [[buffer(BUF_POSITIONS)]],
    FLOWX_CONST_DEVICE float4 *velocities [[buffer(BUF_VELOCITIES)]],
    FLOWX_CONST_DEVICE float4 *keys [[buffer(BUF_KEYS)]],
    FLOWX_CONST_DEVICE float *cell_start [[buffer(BUF_CELL_START)]],
    FLOWX_CONST_DEVICE float *cell_end [[buffer(BUF_CELL_END)]],
    FLOWX_DEVICE float4 *ww_keys [[buffer(BUF_WW_KEYS)]],
    FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
    FLOWX_TID)
{
  int i = int(tid);
  if (i >= P.sorted_count) {
    return;
  }
  if (i >= P.particle_count) {
    ww_keys[i] = float4(-1.0f, -1.0f, 0.0f, 0.0f);
    return;
  }

  float3 pi = positions[i].xyz;
  float3 vi = velocities[i].xyz;
  float speed = length(vi);

  float h = P.smoothing_radius;
  float h2 = h * h;
  float poly6 = poly6_coef(h);

  float3 gradient = float3(0.0f);
  float divergence = 0.0f;
  int neighbours = 0;
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
          float3 pj = positions[j].xyz;
          float3 d = pi - pj;
          float r2 = dot(d, d);
          if (r2 >= h2) {
            continue;
          }
          gradient += w_poly6_grad(d, r2, h2, poly6);

          /* Trapped-air proxy: neighbours closing fast and from divergent
           * directions - the signature of a turbulent impact that would
           * entrain air in a real fluid. */
          float3 vj = velocities[j].xyz;
          float3 rel = vi - vj;
          float rel_len = length(rel);
          if (rel_len > 1e-5f && r2 > 1e-10f) {
            divergence += max(0.0f, -dot(rel / rel_len, d / sqrt(r2))) * rel_len;
          }
          neighbours++;
        }
      }
    }
  }

  /* gradient's magnitude is ~0 deep inside the fluid and grows near the
   * surface (the same field sph_normal builds for surface tension), so it
   * doubles here as a cheap "is this particle near the surface" gate: a
   * particle riding outward along that gradient is cresting a wave. */
  float grad_mag = length(gradient);
  float3 outward = (grad_mag > 1e-4f) ? gradient / grad_mag : float3(0.0f);
  float reference = max(P.kinetic_reference_speed, 1e-3f);
  float wave_crest = grad_mag * max(0.0f, dot(vi, -outward)) / reference;

  float trapped_air = divergence / max(float(neighbours), 1.0f);
  float kinetic = clamp(speed / reference, 0.0f, 1.0f);

  float score = P.trapped_air_weight * trapped_air + P.wave_crest_weight * wave_crest +
                P.kinetic_weight * kinetic;

  ww_keys[i] = float4(score, float(i), trapped_air, wave_crest);
}
