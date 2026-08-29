/* PBF pass: commit the position correction computed by sph_delta.
 *
 * Split from sph_delta because that pass reads every particle's predicted
 * position while computing deltas, so nothing may write the predicted buffer
 * until the whole pass is done. This one runs after it, one slot per particle,
 * no cross-particle reads - the write is race-free.
 */
FLOWX_KERNEL void sph_apply_delta(FLOWX_DEVICE float4 *predicted [[buffer(BUF_PREDICTED)]],
                                  FLOWX_CONST_DEVICE float4 *delta [[buffer(BUF_DELTA)]],
                                  FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                                  FLOWX_TID)
{
  int i = int(tid);
  if (i >= P.particle_count) {
    return;
  }
  float3 p = predicted[i].xyz + delta[i].xyz;
  predicted[i] = float4(p, 1.0f);
}
