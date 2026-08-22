/* PBF pass: apply XSPH's correction, then Phase 5's collider push-out, then
 * the domain-box clamp - the collision/boundary handling body is ported
 * verbatim from WCSPH's sph_integrate.glsl, which this pass replaces.
 *
 * PBF's per-substep position change is bounded by the constraint loop's
 * kernel-support corrections rather than an unclamped velocity integration
 * (that clamp now happens earlier, in sph_predict.glsl, since the
 * constraint loop's correctness depends on a complete neighbor grid), so a
 * collider penetration reaching here is no likelier than it was under WCSPH -
 * MAX_COLLIDER_SEARCH's recovery window is unchanged.
 */

#define MAX_COLLIDER_SEARCH 2

void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= particle_count()) {
    return;
  }

  ivec2 t = particle_texel(i);
  vec3 v = imageLoad(velocities_img, t).xyz + imageLoad(delta_img, t).xyz;
  vec3 p = imageLoad(predicted_img, t).xyz;

  float radius = f_hi.w;
  float damping = f_sim.w;

  if (collider_voxel_size() > 0.0 && collider_occupied(collider_coord(p))) {
    float voxel = collider_voxel_size();
    vec3 nearest_free = vec3(0.0);
    float best_dist2 = 1e30;
    bool found = false;

    ivec3 c = collider_coord(p);
    for (int dz = -MAX_COLLIDER_SEARCH; dz <= MAX_COLLIDER_SEARCH; ++dz) {
      for (int dy = -MAX_COLLIDER_SEARCH; dy <= MAX_COLLIDER_SEARCH; ++dy) {
        for (int dx = -MAX_COLLIDER_SEARCH; dx <= MAX_COLLIDER_SEARCH; ++dx) {
          ivec3 nc = c + ivec3(dx, dy, dz);
          if (!collider_in_bounds(nc) || collider_occupied(nc)) {
            continue;
          }
          vec3 center = f_lo.xyz + (vec3(nc) + 0.5) * voxel;
          float d2 = dot(center - p, center - p);
          if (d2 < best_dist2) {
            best_dist2 = d2;
            nearest_free = center;
            found = true;
          }
        }
      }
    }

    if (found) {
      float dist = sqrt(best_dist2);
      vec3 push_dir = (dist > 1e-6) ? (nearest_free - p) / dist : vec3(0.0, 0.0, 1.0);
      p += push_dir * (dist + radius);

      float vn = dot(v, push_dir);
      if (vn < 0.0) {
        v -= vn * (1.0 + damping) * push_dir;
      }
    }
  }

  vec3 lo = f_lo.xyz + vec3(radius);
  vec3 hi = f_hi.xyz - vec3(radius);

  if (p.x < lo.x) {
    p.x = lo.x;
    v.x = -v.x * damping;
  }
  else if (p.x > hi.x) {
    p.x = hi.x;
    v.x = -v.x * damping;
  }
  if (p.y < lo.y) {
    p.y = lo.y;
    v.y = -v.y * damping;
  }
  else if (p.y > hi.y) {
    p.y = hi.y;
    v.y = -v.y * damping;
  }
  if (p.z < lo.z) {
    p.z = lo.z;
    v.z = -v.z * damping;
  }
  else if (p.z > hi.z) {
    p.z = hi.z;
    v.z = -v.z * damping;
  }

  imageStore(positions_img, t, vec4(p, 1.0));
  imageStore(velocities_img, t, vec4(v, 0.0));
}
