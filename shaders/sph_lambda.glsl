/* PBF pass: density (Poly6, same kernel WCSPH used) from the predicted
 * positions, then the density-constraint Lagrange multiplier lambda_i
 * (Macklin & Muller 2013, eq. 9-11).
 *
 * C_i = density_i / rest_density - 1 is the constraint each particle should
 * satisfy exactly; lambda_i is how far a first-order position step would need
 * to move to satisfy it, weighted by how "responsive" the local neighborhood
 * is to that move (the denominator - a nearly-empty neighborhood has a small
 * gradient sum and would demand an enormous, destabilizing step, which is
 * exactly what f_sph.w's relaxation epsilon (formerly WCSPH's stiffness slot,
 * see sph_common.glsl) is there to bound).
 *
 * Reads predicted_img (this substep's unconstrained guess, already grid-built
 * against by the time this pass runs) rather than positions_img.
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
  float rest_density = f_sph.z;
  float epsilon = f_sph.w;
  float poly6 = poly6_coef(h);
  float spiky = spiky_grad_coef(h);

  float density = 0.0;
  vec3 grad_self = vec3(0.0);
  float grad_norm2_sum = 0.0;
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
          vec3 pj = imageLoad(predicted_img, particle_texel(j)).xyz;
          vec3 d = pi - pj;
          float r2 = dot(d, d);
          density += mass * w_poly6(r2, h2, poly6);

          if (j == i || r2 >= h2 || r2 <= 1e-12) {
            continue;
          }
          float r = sqrt(r2);
          vec3 grad_ij = mass * w_spiky_grad(r, h, spiky) * (d / r);
          grad_self += grad_ij;
          grad_norm2_sum += dot(grad_ij, grad_ij);
        }
      }
    }
  }

  float constraint = max(density, 1e-6) / rest_density - 1.0;
  float denom = (dot(grad_self, grad_self) + grad_norm2_sum) / (rest_density * rest_density) + epsilon;
  float lambda = -constraint / max(denom, 1e-9);

  imageStore(lambda_img, particle_texel(i), vec4(max(density, 1e-6), lambda, 0.0, 0.0));
}
