FLOWX_KERNEL void apic_g2p(FLOWX_CONST_DEVICE float4 *positions [[buffer(BUF_POSITIONS)]],
                           FLOWX_DEVICE float4 *velocities [[buffer(BUF_VELOCITIES)]],
                           FLOWX_DEVICE float4 *affine [[buffer(BUF_AFFINE)]],
                           FLOWX_DEVICE float4 *motion [[buffer(BUF_DELTA)]],
                           FLOWX_CONST_DEVICE float4 *grid_velocity [[buffer(BUF_GRID_VELOCITY)]],
                           FLOWX_CONST_DEVICE float4 *grid_velocity_old [[buffer(BUF_GRID_VELOCITY_OLD)]],
                           FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                           FLOWX_TID)
{
  int particle = int(tid);
  if (particle >= P.particle_count) {
    return;
  }
  float3 xp = positions[particle].xyz;
  float3 particle_velocity = float3(0.0f);
  float3 old_velocity = float3(0.0f);
  bool flip = P.flip_blend > 0.0f;

  for (int axis = 0; axis < 3; ++axis) {
    float3 local = (xp - params_lo(P)) / P.grid_spacing - apic_face_offset(axis);
    int3 base = int3(floor(local));
    float component = 0.0f;
    float3 row = float3(0.0f);
    float3 frac = local - float3(base);

    for (int dz = 0; dz <= 1; ++dz) {
      for (int dy = 0; dy <= 1; ++dy) {
        for (int dx = 0; dx <= 1; ++dx) {
          int3 node = base + int3(dx, dy, dz);
          if (!apic_face_valid(P, node, axis)) {
            continue;
          }
          /* Trilinear weight and its gradient. The corner is kept even when its
           * weight is zero: on a face plane the far corner carries no weight but
           * still carries a gradient, and dropping it is what made the old
           * B.D^-1 form singular there (rows of ~27000 1/s). C = sum v grad(w)
           * equals B.D^-1 wherever D is invertible. Keep in sync with
           * solver/engine/apic_cpu.py. */
          float3 w1 = float3(dx ? frac.x : 1.0f - frac.x, dy ? frac.y : 1.0f - frac.y,
                             dz ? frac.z : 1.0f - frac.z);
          float3 sign = float3(dx ? 1.0f : -1.0f, dy ? 1.0f : -1.0f, dz ? 1.0f : -1.0f);
          float weight = w1.x * w1.y * w1.z;
          float3 grad = sign * float3(w1.y * w1.z, w1.x * w1.z, w1.x * w1.y);
          float value = grid_velocity[apic_node_index(P, node)][axis];
          if (flip) {
            old_velocity[axis] += weight * grid_velocity_old[apic_node_index(P, node)][axis];
          }
          component += weight * value;
          row += value * grad;
        }
      }
    }
    particle_velocity[axis] = component;
    affine[particle * 3 + axis] = float4(row / P.grid_spacing, 0.0f);
  }
  if (flip) {
    /* FLIP hands the particle the grid's change rather than the grid's value,
     * which keeps detail the grid is too coarse to hold - and the noise that
     * comes with it, hence a blend. The affine rows stay APIC's. Behind a
     * branch, not a multiply by zero, so blend 0 is exactly APIC.
     *
     * The grid's own value is kept in the (PBF-only) delta buffer for the
     * advection that follows this pass: positions move through the grid
     * field, not the carried velocity. See ApicMetalEngine.substep. */
    motion[particle] = float4(particle_velocity, 0.0f);
    float3 flip_velocity = velocities[particle].xyz + particle_velocity - old_velocity;
    particle_velocity = mix(particle_velocity, flip_velocity, P.flip_blend);
  }
  velocities[particle] = float4(particle_velocity, 0.0f);
}
