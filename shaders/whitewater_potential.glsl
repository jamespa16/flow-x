/* Whitewater pass 1: score every fluid particle's likelihood of throwing off
 * secondary (spray/foam/bubble) particles this frame, and stash the result
 * for the sort pass that follows.
 *
 * A simplified, real-time-budget version of Ihmsen et al.'s three-potential
 * classification (trapped-air, wave-crest, kinetic-energy), computed from the
 * same neighbor loop and kernel gradient sph_normal.glsl already uses for
 * surface tension - but run unconditionally, since whitewater needs the
 * surface shape whether or not surface tension itself is enabled. It reuses
 * the frame's *last substep* grid (keys_img/cell_start_img/cell_end_img),
 * exactly like surface_splat.glsl does for the same reason: rebuilding a grid
 * just for this would cost more than the one-substep staleness is worth.
 *
 * Output (per fluid particle, into ww_keys_img at the same texel the fluid's
 * own keys_img would use): x = combined score (sort key), y = source
 * particle index, z = trapped-air sub-score (bubble driver), w = wave-crest
 * sub-score (spray/foam driver). Padding slots - i is a fluid index but past
 * particle_count(), out to the power-of-two sorted_count the bitonic sort
 * needs - get a sentinel score of -1 so a *descending* sort pushes them to
 * the tail, same trick sph_grid_key.glsl uses for its own padding.
 */

void main()
{
  int i = int(gl_GlobalInvocationID.x);
  int n = i_layout.w;
  if (i >= n) {
    return;
  }
  if (i >= particle_count()) {
    imageStore(ww_keys_img, particle_texel(i), vec4(-1.0, -1.0, 0.0, 0.0));
    return;
  }

  vec3 pi = imageLoad(positions_img, particle_texel(i)).xyz;
  vec3 vi = imageLoad(velocities_img, particle_texel(i)).xyz;
  float speed = length(vi);

  float h = f_sph.x;
  float h2 = h * h;
  float poly6 = poly6_coef(h);

  vec3 gradient = vec3(0.0);
  float divergence = 0.0;
  int neighbors = 0;
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
          vec3 pj = imageLoad(positions_img, particle_texel(j)).xyz;
          vec3 d = pi - pj;
          float r2 = dot(d, d);
          if (r2 >= h2) {
            continue;
          }
          gradient += w_poly6_grad(d, r2, h2, poly6);

          /* Trapped-air proxy: neighbors closing fast and from divergent
           * directions - the signature of a turbulent impact that would
           * entrain air in a real fluid. */
          vec3 vj = imageLoad(velocities_img, particle_texel(j)).xyz;
          vec3 rel = vi - vj;
          float rel_len = length(rel);
          if (rel_len > 1e-5 && r2 > 1e-10) {
            divergence += max(0.0, -dot(rel / rel_len, d / sqrt(r2))) * rel_len;
          }
          neighbors++;
        }
      }
    }
  }

  /* gradient's magnitude is ~0 deep inside the fluid and grows near the
   * surface (same field sph_normal.glsl builds for surface tension), so it
   * doubles here as a cheap "is this particle near the surface" gate: a
   * particle riding outward along that gradient is cresting a wave. */
  float grad_mag = length(gradient);
  vec3 outward = (grad_mag > 1e-4) ? gradient / grad_mag : vec3(0.0);
  float wave_crest = grad_mag * max(0.0, dot(vi, -outward)) / max(f_potential.w, 1e-3);

  float trapped_air = divergence / max(float(neighbors), 1.0);
  float kinetic = clamp(speed / max(f_potential.w, 1e-3), 0.0, 1.0);

  float score =
      f_potential.x * trapped_air + f_potential.y * wave_crest + f_potential.z * kinetic;

  imageStore(ww_keys_img, particle_texel(i), vec4(score, float(i), trapped_air, wave_crest));
}
