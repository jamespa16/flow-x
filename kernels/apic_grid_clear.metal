FLOWX_KERNEL void apic_grid_clear(FLOWX_DEVICE float4 *mass [[buffer(BUF_GRID_MASS)]],
                                  FLOWX_DEVICE float4 *momentum [[buffer(BUF_GRID_MOMENTUM)]],
                                  FLOWX_DEVICE float4 *velocity [[buffer(BUF_GRID_VELOCITY)]],
                                  FLOWX_DEVICE float4 *vorticity [[buffer(BUF_GRID_VORT)]],
                                  FLOWX_DEVICE float4 *cell_data [[buffer(BUF_GRID_SCRATCH)]],
                                  FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                                  FLOWX_TID)
{
  int i = int(tid);
  if (i < apic_node_count(P)) {
    mass[i] = float4(0.0f);
    momentum[i] = float4(0.0f);
    velocity[i] = float4(0.0f);
  }
  if (i < P.cell_count) {
    vorticity[i] = float4(0.0f);
    cell_data[i] = float4(0.0f);
  }
}
