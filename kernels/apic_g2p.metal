FLOWX_INLINE float3 apic_solve_symmetric(float3 diagonal, float3 off_diagonal,
                                         float3 rhs)
{
  /* Matrix rows: (a,b,c), (b,d,e), (c,e,f). */
  float a = diagonal.x;
  float d = diagonal.y;
  float f = diagonal.z;
  float b = off_diagonal.x;
  float c = off_diagonal.y;
  float e = off_diagonal.z;
  float determinant = a * (d * f - e * e) - b * (b * f - c * e) +
                      c * (b * e - c * d);
  if (abs(determinant) < 1e-8f) {
    return float3(0.0f);
  }
  float3 result;
  result.x = ((d * f - e * e) * rhs.x + (c * e - b * f) * rhs.y +
              (b * e - c * d) * rhs.z) / determinant;
  result.y = ((c * e - b * f) * rhs.x + (a * f - c * c) * rhs.y +
              (b * c - a * e) * rhs.z) / determinant;
  result.z = ((b * e - c * d) * rhs.x + (b * c - a * e) * rhs.y +
              (a * d - b * b) * rhs.z) / determinant;
  return result;
}

FLOWX_KERNEL void apic_g2p(FLOWX_CONST_DEVICE float4 *positions [[buffer(BUF_POSITIONS)]],
                           FLOWX_DEVICE float4 *velocities [[buffer(BUF_VELOCITIES)]],
                           FLOWX_DEVICE float4 *affine [[buffer(BUF_AFFINE)]],
                           FLOWX_CONST_DEVICE float4 *grid_velocity [[buffer(BUF_GRID_VELOCITY)]],
                           FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                           FLOWX_TID)
{
  int particle = int(tid);
  if (particle >= P.particle_count) {
    return;
  }
  float3 xp = positions[particle].xyz;
  float3 particle_velocity = float3(0.0f);

  for (int axis = 0; axis < 3; ++axis) {
    float3 local = (xp - params_lo(P)) / P.grid_spacing - apic_face_offset(axis);
    int3 base = int3(floor(local));
    float component = 0.0f;
    float3 covariance = float3(0.0f);
    float3 diagonal = float3(0.0f);
    float3 off_diagonal = float3(0.0f); /* xy, xz, yz */

    for (int dz = 0; dz <= 1; ++dz) {
      for (int dy = 0; dy <= 1; ++dy) {
        for (int dx = 0; dx <= 1; ++dx) {
          int3 node = base + int3(dx, dy, dz);
          if (!apic_face_valid(P, node, axis)) {
            continue;
          }
          float weight = apic_weight(P, xp, node, axis);
          if (weight <= 0.0f) {
            continue;
          }
          float value = grid_velocity[apic_node_index(P, node)][axis];
          float3 r = (apic_face_position(P, node, axis) - xp) / P.grid_spacing;
          component += weight * value;
          covariance += weight * value * r;
          diagonal += weight * r * r;
          off_diagonal += weight * float3(r.x * r.y, r.x * r.z, r.y * r.z);
        }
      }
    }
    particle_velocity[axis] = component;
    float3 row = apic_solve_symmetric(diagonal, off_diagonal, covariance) /
                 P.grid_spacing;
    affine[particle * 3 + axis] = float4(row, 0.0f);
  }
  velocities[particle] = float4(particle_velocity, 0.0f);
}
