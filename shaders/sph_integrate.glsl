/* SPH pass 3: semi-implicit Euler integration, Phase 5 collider push-out, then
 * a clamp to the domain box.
 *
 * The collider search only runs for a particle that actually lands inside an
 * occupied voxel - rare, since it means a particle got through a whole
 * substep of SPH pressure force without being pushed back already - so a
 * small brute-force box scan for the nearest free voxel is cheap in the
 * aggregate despite being O(n^3) in the search radius. MAX_COLLIDER_SEARCH
 * bounds how deep a penetration it can recover from in one substep; deeper
 * than that and the particle is left in place for this substep - SPH surface
 * pressure will push it toward a free voxel over the next few instead.
 */

#define MAX_COLLIDER_SEARCH 2

void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= particle_count()) {
    return;
  }

  ivec2 t = particle_texel(i);
  float dt = f_sim.y;
  float h = f_sph.x;

  vec3 v = imageLoad(velocities_img, t).xyz + imageLoad(forces_img, t).xyz * dt;

  /* A particle must not cross more than a fraction of a smoothing radius in
   * one substep, or it tunnels past the neighbors that would have pushed back
   * and the simulation blows up. Clamping speed is a cruder guard than
   * shrinking dt, but it keeps a bad parameter set recoverable. */
  float v_max = 0.4 * h / max(dt, 1e-6);
  float speed = length(v);
  if (speed > v_max) {
    v *= v_max / speed;
  }

  vec3 p = imageLoad(positions_img, t).xyz + v * dt;

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

      /* Reflect and damp only the velocity component driving the particle
       * into the collider, same as the wall clamp below does per-axis. */
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
