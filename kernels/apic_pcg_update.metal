/* x += alpha p, r -= alpha q, and the partial sums of r.r and r.z for the
 * RESIDUAL stage that decides whether to stop and computes beta. */

FLOWX_KERNEL void apic_pcg_update(FLOWX_DEVICE float4 *cell_data [[buffer(BUF_GRID_SCRATCH)]],
                                  FLOWX_DEVICE float4 *pcg [[buffer(BUF_PCG)]],
                                  FLOWX_DEVICE float4 *partials [[buffer(BUF_PCG_PARTIALS)]],
                                  FLOWX_CONST_DEVICE float *scalars [[buffer(BUF_PCG_SCALARS)]],
                                  FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                                  FLOWX_GROUP_TID)
{
  if (!apic_pcg_active(scalars)) {
    return;
  }
  FLOWX_SHARED float2 shared[FLOWX_GROUP_SIZE];
  int index = int(tid);
  float2 local = float2(0.0f);
  if (index < P.cell_count && cell_data[index].w > 0.0f) {
    float alpha = scalars[PCG_ALPHA];
    float4 value = pcg[index];
    cell_data[index].y += alpha * value.y;
    float r = value.x - alpha * value.z;
    float z = value.w > 0.0f ? r / value.w : 0.0f;
    pcg[index].x = r;
    local = float2(r * r, r * z);
  }
  float2 total = flowx_group_sum(shared, lid, local);
  if (lid == 0) {
    partials[group] = float4(total, 0.0f, 0.0f);
  }
}
