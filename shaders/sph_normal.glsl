/* Surface tension pass 1 (Morris 2000): per-particle color-field gradient
 * (surface normal) and curvature, read by sph_predict.glsl as a cohesion
 * force alongside gravity.
 *
 * color_i = sum_j (m_j / density_j) * W_poly6(r) is a smoothed indicator that
 * is ~1 deep inside the fluid and falls off near the surface; its gradient
 * n_i therefore points inward wherever a particle is near the surface, and is
 * ~0 for interior particles - which is exactly the "only pull at the
 * surface" behavior surface tension needs, for free from the field shape.
 * Curvature is the color field's Laplacian, using the same viscosity-kernel
 * Laplacian machinery (w_visc_lap/visc_lap_coef) sph_xsph's WCSPH-era
 * ancestor already used for a different purpose - no new kernel family
 * needed for this model, unlike Akinci-style cohesion.
 *
 * Runs on predicted_img and the grid built from it *last* substep (see
 * sph.py's module docstring): normals are therefore one substep stale. That
 * is a deliberate tradeoff, not an oversight - computing genuinely current
 * normals would need a second grid build (the dominant per-substep cost)
 * before sph_predict, every substep, just for this. One substep of lag on a
 * force whose whole job is a slow surface-shape effect is the same order of
 * approximation this codebase already accepts for animated-collider grids.
 */

void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= particle_count()) {
    return;
  }

  vec3 pi = imageLoad(predicted_img, particle_texel(i)).xyz;

  float h = f_sph.x;
  float h2 = h * h;
  float mass = f_sph.y;
  float poly6 = poly6_coef(h);
  float visc_lap = visc_lap_coef(h);

  vec3 normal = vec3(0.0);
  float laplacian = 0.0;
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
          if (r2 >= h2) {
            continue;
          }
          float density_j = max(imageLoad(lambda_img, particle_texel(j)).x, 1e-6);
          float weight = mass / density_j;

          normal += weight * w_poly6_grad(d, r2, h2, poly6);
          laplacian += weight * w_visc_lap(sqrt(r2), h, visc_lap);
        }
      }
    }
  }

  float mag = length(normal);
  float curvature = (mag > 1e-4) ? -laplacian / mag : 0.0;
  imageStore(normal_img, particle_texel(i), vec4(normal, curvature));
}
