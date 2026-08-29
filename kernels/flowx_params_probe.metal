/* Test-only kernel: report the real layout of `struct Params`.
 *
 * Not part of the solver. scripts/test_engine.py compiles it to check that
 * solver/engine/params.py's PARAMS_FORMAT still agrees with the struct in
 * flowx_prelude.h. Nothing else validates that, and a drifted field would
 * silently corrupt every parameter after it - a bug whose symptoms (a fluid
 * that behaves oddly) point nowhere near the cause.
 *
 * Writes sizeof(Params) followed by the offset of each probed field.
 */
FLOWX_KERNEL void flowx_params_probe(FLOWX_DEVICE uint *out [[buffer(0)]],
                                     FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                                     FLOWX_TID)
{
  if (tid != 0) {
    return;
  }
  FLOWX_CONSTANT char *base = (FLOWX_CONSTANT char *)&P;
  out[0] = uint(sizeof(Params));
  out[1] = uint((FLOWX_CONSTANT char *)&P.particle_count - base);
  out[2] = uint((FLOWX_CONSTANT char *)&P.lo_x - base);
  out[3] = uint((FLOWX_CONSTANT char *)&P.smoothing_radius - base);
  out[4] = uint((FLOWX_CONSTANT char *)&P.collider_voxel - base);
  out[5] = uint((FLOWX_CONSTANT char *)&P.surface_kernel_radius - base);
  out[6] = uint((FLOWX_CONSTANT char *)&P.ww_capacity - base);
  out[7] = uint((FLOWX_CONSTANT char *)&P.frame_dt - base);
}
