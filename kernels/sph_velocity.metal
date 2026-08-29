/* PBF pass: derive velocity from the position change the constraint loop
 * produced (Macklin & Muller 2013, eq. 17): v_i = (p_pred - p_old) / dt.
 *
 * Split from XSPH smoothing (sph_xsph) and finalization (sph_finalize) because
 * XSPH needs every particle's *raw* velocity from this formula before any of
 * them are touched again - writing a smoothed velocity back into the same
 * buffer that other invocations still read from would race, the same Jacobi
 * hazard the grid-build and constraint-solve passes already avoid by writing
 * corrections to a separate buffer first.
 */
FLOWX_KERNEL void sph_velocity(FLOWX_CONST_DEVICE float4 *positions [[buffer(BUF_POSITIONS)]],
                               FLOWX_DEVICE float4 *velocities [[buffer(BUF_VELOCITIES)]],
                               FLOWX_CONST_DEVICE float4 *predicted [[buffer(BUF_PREDICTED)]],
                               FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                               FLOWX_TID)
{
  int i = int(tid);
  if (i >= P.particle_count) {
    return;
  }
  float dt = P.dt;
  float3 p_pred = predicted[i].xyz;
  float3 p_old = positions[i].xyz;
  velocities[i] = float4((p_pred - p_old) / max(dt, 1e-6f), 0.0f);
}
