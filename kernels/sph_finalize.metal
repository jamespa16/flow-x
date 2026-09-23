/* PBF pass: apply XSPH's correction, then the collider push-out, then the
 * domain-box clamp.
 *
 * PBF's per-substep position change is bounded by the constraint loop's
 * kernel-support corrections rather than an unclamped velocity integration
 * (that clamp happens earlier, in sph_predict, since the constraint loop's
 * correctness depends on a complete neighbour grid), so a collider penetration
 * reaching here is no likelier than it was under WCSPH - MAX_COLLIDER_SEARCH's
 * recovery window is unchanged.
 */

#define MAX_COLLIDER_SEARCH 2

FLOWX_KERNEL void sph_finalize(FLOWX_DEVICE float4 *positions [[buffer(BUF_POSITIONS)]],
                               FLOWX_DEVICE float4 *velocities [[buffer(BUF_VELOCITIES)]],
                               FLOWX_DEVICE float4 *predicted [[buffer(BUF_PREDICTED)]],
                               FLOWX_CONST_DEVICE float4 *delta [[buffer(BUF_DELTA)]],
                               FLOWX_CONST_DEVICE float *collider [[buffer(BUF_COLLIDER)]],
                               FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                               FLOWX_TID)
{
  int i = int(tid);
  if (i >= P.particle_count) {
    return;
  }

  float3 v = velocities[i].xyz + delta[i].xyz;
  float3 p = predicted[i].xyz;

  float radius = P.particle_radius;
  float damping = P.boundary_damping;
  float voxel = P.collider_voxel;

  if (voxel > 0.0f && collider_occupied(P, collider, collider_coord(P, p))) {
    float3 nearest_free = float3(0.0f);
    float best_dist2 = 1e30f;
    bool found = false;

    int3 c = collider_coord(P, p);
    for (int dz = -MAX_COLLIDER_SEARCH; dz <= MAX_COLLIDER_SEARCH; ++dz) {
      for (int dy = -MAX_COLLIDER_SEARCH; dy <= MAX_COLLIDER_SEARCH; ++dy) {
        for (int dx = -MAX_COLLIDER_SEARCH; dx <= MAX_COLLIDER_SEARCH; ++dx) {
          int3 nc = c + int3(dx, dy, dz);
          if (!collider_in_bounds(P, nc) || collider_occupied(P, collider, nc)) {
            continue;
          }
          float3 center = params_lo(P) + (float3(nc) + 0.5f) * voxel;
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
      float3 push_dir = (dist > 1e-6f) ? (nearest_free - p) / dist : float3(0.0f, 0.0f, 1.0f);
      p += push_dir * (dist + radius);

      float vn = dot(v, push_dir);
      if (vn < 0.0f) {
        v -= vn * (1.0f + damping) * push_dir;
      }
    }
  }

  float3 lo = params_lo(P) + float3(radius);
  float3 hi = params_hi(P) - float3(radius);

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

  positions[i] = float4(p, 1.0f);
  // The next surface-normal pass hashes predicted positions, so it must see
  // the collider/domain-corrected position rather than this pass's input.
  predicted[i] = float4(p, 1.0f);
  velocities[i] = float4(v, 0.0f);
}
