FLOWX_KERNEL void apic_vorticity(FLOWX_CONST_DEVICE float4 *velocity [[buffer(BUF_GRID_VELOCITY)]],
                                 FLOWX_DEVICE float4 *vorticity [[buffer(BUF_GRID_VORT)]],
                                 FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                                 FLOWX_TID)
{
  int index = int(tid);
  if (index >= P.cell_count) {
    return;
  }
  int3 c = apic_cell_coord(P, index);
  float inv = 0.5f / P.grid_spacing;
  float3 vx0 = apic_clamped_cell_velocity(P, velocity, c - int3(1, 0, 0));
  float3 vx1 = apic_clamped_cell_velocity(P, velocity, c + int3(1, 0, 0));
  float3 vy0 = apic_clamped_cell_velocity(P, velocity, c - int3(0, 1, 0));
  float3 vy1 = apic_clamped_cell_velocity(P, velocity, c + int3(0, 1, 0));
  float3 vz0 = apic_clamped_cell_velocity(P, velocity, c - int3(0, 0, 1));
  float3 vz1 = apic_clamped_cell_velocity(P, velocity, c + int3(0, 0, 1));
  float3 curl = float3((vy1.z - vy0.z) - (vz1.y - vz0.y),
                       (vz1.x - vz0.x) - (vx1.z - vx0.z),
                       (vx1.y - vx0.y) - (vy1.x - vy0.x)) * inv;
  vorticity[index] = float4(curl, length(curl));
}
