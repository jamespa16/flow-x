FLOWX_KERNEL void apic_project(FLOWX_DEVICE float4 *velocity [[buffer(BUF_GRID_VELOCITY)]],
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
  float scale = P.dt / (max(P.rest_density, 1e-6f) * P.grid_spacing);
  for (int axis = 0; axis < 3; ++axis) {
    if (!apic_face_valid(P, node, axis) || apic_face_solid(P, cell_data, node, axis)) {
      value[axis] = 0.0f;
      continue;
    }
    int3 low = node;
    low[axis] -= 1;
    int3 high = node;
    float low_type = apic_cell_type(P, cell_data, low);
    float high_type = apic_cell_type(P, cell_data, high);
    if (low_type <= 0.0f && high_type <= 0.0f) {
      continue;
    }
    float low_pressure = low_type > 0.0f ? apic_pressure(P, cell_data, low) : 0.0f;
    float high_pressure = high_type > 0.0f ? apic_pressure(P, cell_data, high) : 0.0f;
    value[axis] -= scale * (high_pressure - low_pressure);
  }
  velocity[index] = value;
}
