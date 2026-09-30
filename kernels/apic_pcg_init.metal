/* Start a PCG pressure solve from zero pressure.
 *
 * With x = 0 the residual is the right-hand side itself, b = -rhs (the sign
 * that makes the matrix positive definite rather than negative), and the first
 * direction is the preconditioned residual. Also records each cell's diagonal,
 * and reduces r.r and r.z for apic_pcg_reduce's INIT stage.
 *
 * Pressure lives in cell_data.y, where apic_divergence has just written zero
 * and where apic_project reads it with pressure_ping = 0. */

FLOWX_KERNEL void apic_pcg_init(FLOWX_CONST_DEVICE float4 *cell_data [[buffer(BUF_GRID_SCRATCH)]],
                                FLOWX_DEVICE float4 *pcg [[buffer(BUF_PCG)]],
                                FLOWX_DEVICE float4 *partials [[buffer(BUF_PCG_PARTIALS)]],
                                FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                                FLOWX_GROUP_TID)
{
  FLOWX_SHARED float2 shared[FLOWX_GROUP_SIZE];
  int index = int(tid);
  float2 local = float2(0.0f);
  if (index < P.cell_count) {
    float4 cell = cell_data[index];
    float4 value = float4(0.0f);
    if (cell.w > 0.0f) {
      float diagonal = apic_pressure_diagonal(P, cell_data, apic_cell_coord(P, index));
      float b = -(P.rest_density * P.grid_spacing * P.grid_spacing * cell.x /
                  max(P.dt, 1e-8f));
      float z = diagonal > 0.0f ? b / diagonal : 0.0f;
      value = float4(b, z, 0.0f, diagonal);
      local = float2(b * b, b * z);
    }
    pcg[index] = value;
  }
  float2 total = flowx_group_sum(shared, lid, local);
  if (lid == 0) {
    partials[group] = float4(total, 0.0f, 0.0f);
  }
}
