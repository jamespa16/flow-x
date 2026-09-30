/* q = A p for the pressure matrix, and the partial sums of p.q.
 *
 * A is the Jacobi system: the diagonal from apic_pcg_init, minus each fluid
 * neighbour's value; air neighbours are Dirichlet zero and solid ones drop
 * out. Neighbours are summed in a fixed order (-x, +x, -y, +y, -z, +z), the
 * order apic_cpu's stencil uses. */

FLOWX_KERNEL void apic_pcg_matvec(FLOWX_CONST_DEVICE float4 *cell_data [[buffer(BUF_GRID_SCRATCH)]],
                                  FLOWX_DEVICE float4 *pcg [[buffer(BUF_PCG)]],
                                  FLOWX_DEVICE float4 *partials [[buffer(BUF_PCG_PARTIALS)]],
                                  FLOWX_CONST_DEVICE float *scalars [[buffer(BUF_PCG_SCALARS)]],
                                  FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                                  FLOWX_GROUP_TID)
{
  /* Uniform across the whole dispatch, so returning before the barriers in
   * the group sum is safe. */
  if (!apic_pcg_active(scalars)) {
    return;
  }
  FLOWX_SHARED float2 shared[FLOWX_GROUP_SIZE];
  int index = int(tid);
  float2 local = float2(0.0f);
  if (index < P.cell_count && cell_data[index].w > 0.0f) {
    int3 c = apic_cell_coord(P, index);
    const int3 offsets[6] = {int3(-1, 0, 0), int3(1, 0, 0),
                             int3(0, -1, 0), int3(0, 1, 0),
                             int3(0, 0, -1), int3(0, 0, 1)};
    float neighbours = 0.0f;
    for (int n = 0; n < 6; ++n) {
      int3 neighbour = c + offsets[n];
      if (apic_cell_type(P, cell_data, neighbour) > 0.0f) {
        neighbours += pcg[cell_index(P, neighbour)].y;
      }
    }
    float p = pcg[index].y;
    float q = pcg[index].w * p - neighbours;
    /* Only .z is written; neighbours read only .y. */
    pcg[index].z = q;
    local = float2(p * q, 0.0f);
  }
  float2 total = flowx_group_sum(shared, lid, local);
  if (lid == 0) {
    partials[group] = float4(total, 0.0f, 0.0f);
  }
}
