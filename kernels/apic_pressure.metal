FLOWX_KERNEL void apic_pressure(FLOWX_DEVICE float4 *cell_data [[buffer(BUF_GRID_SCRATCH)]],
                                FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                                FLOWX_TID)
{
  int index = int(tid);
  if (index >= P.cell_count) {
    return;
  }
  float4 current = cell_data[index];
  if (current.w <= 0.0f) {
    if (P.pressure_ping == 0) {
      cell_data[index].z = 0.0f;
    }
    else {
      cell_data[index].y = 0.0f;
    }
    return;
  }

  int3 c = apic_cell_coord(P, index);
  const int3 offsets[6] = {int3(-1, 0, 0), int3(1, 0, 0),
                           int3(0, -1, 0), int3(0, 1, 0),
                           int3(0, 0, -1), int3(0, 0, 1)};
  float total = 0.0f;
  float diagonal = 0.0f;
  for (int n = 0; n < 6; ++n) {
    int3 neighbour = c + offsets[n];
    if (!cell_in_bounds(P, neighbour)) {
      continue;
    }
    float type = apic_cell_type(P, cell_data, neighbour);
    if (type < 0.0f) {
      continue;
    }
    diagonal += 1.0f;
    if (type > 0.0f) {
      total += apic_pressure(P, cell_data, neighbour);
    }
  }
  float old_pressure = P.pressure_ping == 0 ? current.y : current.z;
  float rhs = P.rest_density * P.grid_spacing * P.grid_spacing * current.x /
              max(P.dt, 1e-8f);
  float candidate = diagonal > 0.0f ? (total - rhs) / diagonal : 0.0f;
  float next = (1.0f / 3.0f) * old_pressure + (2.0f / 3.0f) * candidate;
  if (P.pressure_ping == 0) {
    cell_data[index].z = next;
  }
  else {
    cell_data[index].y = next;
  }
}
