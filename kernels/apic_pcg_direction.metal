/* p = z + beta p, with z = r / diagonal: the diagonal preconditioner applied
 * on the fly rather than stored. */

FLOWX_KERNEL void apic_pcg_direction(FLOWX_CONST_DEVICE float4 *cell_data [[buffer(BUF_GRID_SCRATCH)]],
                                     FLOWX_DEVICE float4 *pcg [[buffer(BUF_PCG)]],
                                     FLOWX_CONST_DEVICE float *scalars [[buffer(BUF_PCG_SCALARS)]],
                                     FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                                     FLOWX_TID)
{
  int index = int(tid);
  if (index >= P.cell_count || !apic_pcg_active(scalars) || cell_data[index].w <= 0.0f) {
    return;
  }
  float4 value = pcg[index];
  float z = value.w > 0.0f ? value.x / value.w : 0.0f;
  pcg[index].y = z + scalars[PCG_BETA] * value.y;
}
