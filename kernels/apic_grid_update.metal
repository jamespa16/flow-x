FLOWX_KERNEL void apic_grid_update(FLOWX_CONST_DEVICE float4 *mass [[buffer(BUF_GRID_MASS)]],
                                   FLOWX_CONST_DEVICE float4 *momentum [[buffer(BUF_GRID_MOMENTUM)]],
                                   FLOWX_DEVICE float4 *velocity [[buffer(BUF_GRID_VELOCITY)]],
                                   FLOWX_CONST_DEVICE float4 *cell_data [[buffer(BUF_GRID_SCRATCH)]],
                                   FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                                   FLOWX_TID)
{
  int index = int(tid);
  if (index >= apic_node_count(P)) {
    return;
  }
  int3 node = apic_node_coord(P, index);
  float4 value = float4(0.0f);
  for (int axis = 0; axis < 3; ++axis) {
    if (!apic_face_valid(P, node, axis) || apic_face_solid(P, cell_data, node, axis)) {
      continue;
    }
    float m = mass[index][axis];
    if (m <= 1e-12f) {
      continue;
    }
    float v = momentum[index][axis] / m;
    if (axis == 2) {
      v += P.gravity * P.dt;
    }
    value[axis] = clamp(v, -P.grid_max_speed, P.grid_max_speed);
  }
  velocity[index] = value;
}
