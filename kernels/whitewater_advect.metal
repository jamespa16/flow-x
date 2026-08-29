/* Whitewater pass 4: integrate every pool slot one frame forward.
 *
 * Dispatched over the full pool capacity, not just the live particles - there
 * is no compacted "alive list" (see whitewater_spawn: the ring buffer's
 * overwrite-on-wrap is the only bookkeeping this system does), so a dead slot
 * (life <= 0) is skipped cheaply rather than iterated around.
 *
 * Per-kind behaviour is a coarse stand-in for real whitewater physics -
 * visually distinct, not a second SPH solve - and it does not sample the
 * fluid's own velocity field. Under Blender's `gpu` module that was not a
 * choice: doing so would have needed the fluid's five state images on top of
 * this pass's own three, landing exactly at Metal's 8 read-write-image cap.
 * Buffers have no such cap, so the field is now reachable; taking it is left
 * for the pass that changes this behaviour deliberately, not for the port that
 * is meant to reproduce it.
 *
 *   spray  - ballistic: gravity only.
 *   foam   - drag damps horizontal motion (it settles where it lands), tiny
 *            gravity so it doesn't stack forever.
 *   bubble - buoyant (gravity partly or fully cancelled), light drag so it
 *            doesn't accelerate forever on the way up.
 */
FLOWX_KERNEL void whitewater_advect(FLOWX_DEVICE float4 *ww_positions [[buffer(BUF_WW_POSITIONS)]],
                                    FLOWX_DEVICE float4 *ww_velkind [[buffer(BUF_WW_VELKIND)]],
                                    FLOWX_CONST_DEVICE float *collider [[buffer(BUF_COLLIDER)]],
                                    FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                                    FLOWX_TID)
{
  int slot = int(tid);
  if (slot >= P.ww_capacity) {
    return;
  }

  float4 pos_life = ww_positions[slot];
  float life = pos_life.w;
  if (life <= 0.0f) {
    return;
  }

  float4 vel_kind = ww_velkind[slot];
  float3 p = pos_life.xyz;
  float3 v = vel_kind.xyz;
  int kind = int(vel_kind.w);

  float gravity = P.gravity;
  float dt = P.frame_dt;
  float drag = P.ww_drag;
  float buoyancy = P.ww_buoyancy;

  if (kind == 0) {
    v.z += gravity * dt;
  }
  else if (kind == 2) {
    v.z += gravity * (1.0f - buoyancy) * dt;
    v.xy *= clamp(1.0f - drag * dt, 0.0f, 1.0f);
  }
  else {
    v *= clamp(1.0f - drag * dt, 0.0f, 1.0f);
    v.z += gravity * 0.1f * dt;
  }

  p += v * dt;
  life -= dt;

  /* Leaving the domain by more than a couple of cells retires the particle
   * outright rather than clamping it to a wall it was never simulated
   * against - unlike the fluid particles, whitewater has no obligation to stay
   * inside the box. */
  float3 margin = float3(0.2f);
  if (any(p < params_lo(P) - margin) || any(p > params_hi(P) + margin)) {
    life = 0.0f;
  }

  if (P.collider_voxel > 0.0f && collider_occupied(P, collider, collider_coord(P, p))) {
    p -= v * dt;
    v *= 0.2f;
  }

  ww_positions[slot] = float4(p, life);
  ww_velkind[slot] = float4(v, float(kind));
}
