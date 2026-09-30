/* Second level of every PCG dot product, and the scalar bookkeeping.
 *
 * Dispatched as exactly one threadgroup. Lane i sums partials i, i + 64,
 * i + 128, ... in that order, then the lanes are combined by the same fixed
 * tree the first level used - so the total is a function of the inputs only,
 * never of scheduling. See flowx_group_sum for why that matters.
 *
 * Lane 0 then does what the host would do between CG steps, per stage:
 *
 *   INIT      r.r and r.z of the starting residual: the stopping threshold, the
 *             breakdown guard, and whether there is anything to solve.
 *   CURVATURE p.Ap: alpha, or stop if p.Ap is not safely positive.
 *   RESIDUAL  r.r and r.z after the update: stop at the tolerance, else beta.
 *
 * Same rules, same order, as ApicCpuEngine._solve_pcg. */

FLOWX_KERNEL void apic_pcg_reduce(FLOWX_CONST_DEVICE float4 *partials [[buffer(BUF_PCG_PARTIALS)]],
                                  FLOWX_DEVICE float *scalars [[buffer(BUF_PCG_SCALARS)]],
                                  FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                                  FLOWX_GROUP_TID)
{
  FLOWX_SHARED float2 shared[FLOWX_GROUP_SIZE];
  if (P.pcg_stage != PCG_STAGE_INIT && !apic_pcg_active(scalars)) {
    return;
  }
  int groups = (P.cell_count + FLOWX_GROUP_SIZE - 1) / FLOWX_GROUP_SIZE;
  float2 local = float2(0.0f);
  for (int g = int(lid); g < groups; g += FLOWX_GROUP_SIZE) {
    local += partials[g].xy;
  }
  float2 total = flowx_group_sum(shared, lid, local);
  if (lid != 0 || tid != 0) {
    return;
  }

  if (P.pcg_stage == PCG_STAGE_INIT) {
    float rz = total.y;
    scalars[PCG_RZ] = rz;
    scalars[PCG_ALPHA] = 0.0f;
    scalars[PCG_BETA] = 0.0f;
    scalars[PCG_THRESHOLD] = P.pressure_tolerance * P.pressure_tolerance * total.x;
    scalars[PCG_CONVERGED] = rz > 0.0f ? 0.0f : 1.0f;
    scalars[PCG_ITERATIONS] = 0.0f;
    scalars[PCG_GUARD] = PCG_BREAKDOWN * rz;
    scalars[PCG_RR] = total.x;
  }
  else if (P.pcg_stage == PCG_STAGE_CURVATURE) {
    float curvature = total.x;
    /* Written as !(a > b) so a NaN also stops the solve. */
    if (!(curvature > scalars[PCG_GUARD])) {
      scalars[PCG_CONVERGED] = 1.0f;
      scalars[PCG_ALPHA] = 0.0f;
    }
    else {
      scalars[PCG_ALPHA] = scalars[PCG_RZ] / curvature;
      scalars[PCG_ITERATIONS] += 1.0f;
    }
  }
  else {
    float rr = total.x;
    scalars[PCG_RR] = rr;
    if (rr <= scalars[PCG_THRESHOLD]) {
      scalars[PCG_CONVERGED] = 1.0f;
    }
    else {
      scalars[PCG_BETA] = total.y / scalars[PCG_RZ];
      scalars[PCG_RZ] = total.y;
    }
  }
}
