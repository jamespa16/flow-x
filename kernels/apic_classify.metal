FLOWX_KERNEL void apic_classify(FLOWX_CONST_DEVICE float *cell_start [[buffer(BUF_CELL_START)]],
                                FLOWX_CONST_DEVICE float *collider [[buffer(BUF_COLLIDER)]],
                                FLOWX_DEVICE float4 *cell_data [[buffer(BUF_GRID_SCRATCH)]],
                                FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                                FLOWX_TID)
{
  int i = int(tid);
  if (i >= P.cell_count) {
    return;
  }
  int3 cell = apic_cell_coord(P, i);
  float3 center = params_lo(P) + (float3(cell) + 0.5f) * P.grid_spacing;
  bool solid = P.collider_voxel > 0.0f &&
               collider_occupied(P, collider, collider_coord(P, center));
  float type = solid ? -1.0f : (cell_start[i] >= 0.0f ? 1.0f : 0.0f);
  cell_data[i] = float4(0.0f, 0.0f, 0.0f, type);
}
