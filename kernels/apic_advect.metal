/* Move particles with the previous projected velocity, then recover any
 * collider/domain penetration before the new particle-to-grid transfer.
 *
 * Under FLIP blending this runs at the end of the substep instead, and moves
 * particles by the grid velocity apic_g2p left in the delta buffer rather than
 * by the carried velocity; the carried velocity only takes the collision
 * response. See ApicMetalEngine.substep for why. */

#define APIC_MAX_COLLIDER_SEARCH 2

FLOWX_KERNEL void apic_advect(FLOWX_DEVICE float4 *positions [[buffer(BUF_POSITIONS)]],
                              FLOWX_DEVICE float4 *velocities [[buffer(BUF_VELOCITIES)]],
                              FLOWX_DEVICE float4 *predicted [[buffer(BUF_PREDICTED)]],
                              FLOWX_CONST_DEVICE float4 *motion [[buffer(BUF_DELTA)]],
                              FLOWX_CONST_DEVICE float *collider [[buffer(BUF_COLLIDER)]],
                              FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                              FLOWX_TID)
{
  int i = int(tid);
  if (i >= P.particle_count) {
    return;
  }
  bool flip = P.flip_blend > 0.0f;
  float3 v = velocities[i].xyz;
  float3 p = positions[i].xyz + (flip ? motion[i].xyz : v) * P.dt;
  float radius = P.particle_radius;

  if (P.collider_voxel > 0.0f &&
      collider_occupied(P, collider, collider_coord(P, p))) {
    int3 c = collider_coord(P, p);
    float3 nearest = float3(0.0f);
    float best = 1e30f;
    bool found = false;
    for (int dz = -APIC_MAX_COLLIDER_SEARCH; dz <= APIC_MAX_COLLIDER_SEARCH; ++dz) {
      for (int dy = -APIC_MAX_COLLIDER_SEARCH; dy <= APIC_MAX_COLLIDER_SEARCH; ++dy) {
        for (int dx = -APIC_MAX_COLLIDER_SEARCH; dx <= APIC_MAX_COLLIDER_SEARCH; ++dx) {
          int3 nc = c + int3(dx, dy, dz);
          if (!collider_in_bounds(P, nc) || collider_occupied(P, collider, nc)) {
            continue;
          }
          float3 center = params_lo(P) + (float3(nc) + 0.5f) * P.collider_voxel;
          float d2 = dot(center - p, center - p);
          if (d2 < best) {
            nearest = center;
            best = d2;
            found = true;
          }
        }
      }
    }
    if (found) {
      float dist = sqrt(best);
      float3 normal = dist > 1e-6f ? (nearest - p) / dist : float3(0.0f, 0.0f, 1.0f);
      float3 wall = apic_collider_velocity(P, collider, c);
      p += normal * (dist + radius);
      float vn = dot(v - wall, normal);
      if (vn < 0.0f) {
        v -= vn * (1.0f + P.boundary_damping) * normal;
      }
    }
  }

  float3 lo = params_lo(P) + radius;
  float3 hi = params_hi(P) - radius;
  for (int axis = 0; axis < 3; ++axis) {
    /* The carried FLIP velocity need not point the way the motion did, so
     * only its into-wall component is reflected. Under APIC they are the
     * same vector and the reflection is unconditional, as it always was. */
    if (p[axis] < lo[axis]) {
      p[axis] = lo[axis];
      if (!flip || v[axis] < 0.0f) {
        v[axis] *= -P.boundary_damping;
      }
    }
    else if (p[axis] > hi[axis]) {
      p[axis] = hi[axis];
      if (!flip || v[axis] > 0.0f) {
        v[axis] *= -P.boundary_damping;
      }
    }
  }
  positions[i] = float4(p, 1.0f);
  predicted[i] = float4(p, 1.0f);
  velocities[i] = float4(v, 0.0f);
}
