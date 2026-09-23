FLOWX_INLINE float3 apic_confinement_force(FLOWX_CONSTANT Params &P,
                                           FLOWX_CONST_DEVICE float4 *vorticity,
                                           int3 cell)
{
  if (!cell_in_bounds(P, cell)) {
    return float3(0.0f);
  }
  float inv = 0.5f / P.grid_spacing;
  float gx = apic_clamped_vorticity(P, vorticity, cell + int3(1, 0, 0)).w -
             apic_clamped_vorticity(P, vorticity, cell - int3(1, 0, 0)).w;
  float gy = apic_clamped_vorticity(P, vorticity, cell + int3(0, 1, 0)).w -
             apic_clamped_vorticity(P, vorticity, cell - int3(0, 1, 0)).w;
  float gz = apic_clamped_vorticity(P, vorticity, cell + int3(0, 0, 1)).w -
             apic_clamped_vorticity(P, vorticity, cell - int3(0, 0, 1)).w;
  float3 gradient = float3(gx, gy, gz) * inv;
  float magnitude = length(gradient);
  float3 normal = magnitude > 1e-8f ? gradient / magnitude : float3(0.0f);
  float3 curl = vorticity[cell_index(P, cell)].xyz;
  return P.vorticity_epsilon * P.grid_spacing * cross(normal, curl);
}

FLOWX_KERNEL void apic_confinement(
    FLOWX_DEVICE float4 *velocity [[buffer(BUF_GRID_VELOCITY)]],
    FLOWX_CONST_DEVICE float4 *vorticity [[buffer(BUF_GRID_VORT)]],
    FLOWX_CONST_DEVICE float4 *cell_data [[buffer(BUF_GRID_SCRATCH)]],
    FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
    FLOWX_TID)
{
  int index = int(tid);
  if (index >= apic_node_count(P)) {
    return;
  }
  int3 node = apic_node_coord(P, index);
  float4 value = velocity[index];
  for (int axis = 0; axis < 3; ++axis) {
    if (!apic_face_valid(P, node, axis) || apic_face_solid(P, cell_data, node, axis)) {
      value[axis] = 0.0f;
      continue;
    }
    int3 low = node;
    low[axis] -= 1;
    int3 high = node;
    float3 force = 0.5f * (apic_confinement_force(P, vorticity, low) +
                           apic_confinement_force(P, vorticity, high));
    value[axis] = clamp(value[axis] + P.dt * force[axis],
                        -P.grid_max_speed, P.grid_max_speed);
  }
  velocity[index] = value;
}
