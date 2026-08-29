/* PBF pass 1: apply external forces to velocity, then predict a position.
 *
 * This is PBF's replacement for WCSPH's pressure force: there is no pressure
 * term here at all, only gravity and the surface-tension cohesion force
 * (Morris 2000) read from the normal buffer, which sph_normal computed from
 * last substep's state before this substep's grid was rebuilt. Incompressi-
 * bility is restored afterwards by the constraint-solve loop (sph_lambda/
 * sph_delta/sph_apply_delta), not by anything in this pass.
 *
 * P.surface_tension is sigma; 0 makes the term a no-op without needing a
 * branch to skip the pass entirely. (It used to arrive bit-packed into a spare
 * lane of the bitonic-sort slot, for want of room in the push-constant block.)
 *
 * The speed clamp lives here rather than after the constraint loop, because
 * the constraint loop's correctness depends on the neighbour grid built from
 * this pass's output being complete - a particle that jumps more than a
 * smoothing radius before the grid is built can miss neighbours that should
 * have constrained it, which is a correctness problem for PBF's density
 * constraint, not just a stability one the way it was for WCSPH.
 */
FLOWX_KERNEL void sph_predict(FLOWX_CONST_DEVICE float4 *positions [[buffer(BUF_POSITIONS)]],
                              FLOWX_DEVICE float4 *velocities [[buffer(BUF_VELOCITIES)]],
                              FLOWX_DEVICE float4 *predicted [[buffer(BUF_PREDICTED)]],
                              FLOWX_CONST_DEVICE float4 *normal [[buffer(BUF_NORMAL)]],
                              FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                              FLOWX_TID)
{
  int i = int(tid);
  if (i >= P.particle_count) {
    return;
  }

  float dt = P.dt;
  float h = P.smoothing_radius;
  float mass = P.mass;
  float sigma = P.surface_tension;

  float3 v = velocities[i].xyz;
  v.z += P.gravity * dt;

  if (sigma > 0.0f) {
    float4 n = normal[i];
    float mag = length(n.xyz);
    if (mag > 1e-4f) {
      v += (-sigma * n.w * n.xyz / mag / mass) * dt;
    }
  }

  float v_max = 0.4f * h / max(dt, 1e-6f);
  float speed = length(v);
  if (speed > v_max) {
    v *= v_max / speed;
  }

  float3 p = positions[i].xyz + v * dt;

  velocities[i] = float4(v, 0.0f);
  predicted[i] = float4(p, 1.0f);
}
