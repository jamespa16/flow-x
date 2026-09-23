FLOWX_KERNEL void apic_divergence(
    FLOWX_CONST_DEVICE float4 *velocity [[buffer(BUF_GRID_VELOCITY)]],
    FLOWX_DEVICE float4 *cell_data [[buffer(BUF_GRID_SCRATCH)]],
    FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
    FLOWX_TID)
{
  int index = int(tid);
  if (index >= P.cell_count) {
    return;
  }
  int3 c = apic_cell_coord(P, index);
  int base = apic_node_index(P, c);
  int ix = apic_node_index(P, c + int3(1, 0, 0));
  int iy = apic_node_index(P, c + int3(0, 1, 0));
  int iz = apic_node_index(P, c + int3(0, 0, 1));
  float divergence = ((velocity[ix].x - velocity[base].x) +
                      (velocity[iy].y - velocity[base].y) +
                      (velocity[iz].z - velocity[base].z)) / P.grid_spacing;
  float type = cell_data[index].w;
  cell_data[index] = float4(divergence, 0.0f, 0.0f, type);
}
