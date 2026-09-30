FLOWX_KERNEL void apic_grid_update(FLOWX_CONST_DEVICE float4 *mass [[buffer(BUF_GRID_MASS)]],
                                   FLOWX_CONST_DEVICE float4 *momentum [[buffer(BUF_GRID_MOMENTUM)]],
                                   FLOWX_DEVICE float4 *velocity [[buffer(BUF_GRID_VELOCITY)]],
                                   FLOWX_DEVICE float4 *velocity_old [[buffer(BUF_GRID_VELOCITY_OLD)]],
                                   FLOWX_CONST_DEVICE float4 *cell_data [[buffer(BUF_GRID_SCRATCH)]],
                                   FLOWX_CONST_DEVICE float *collider [[buffer(BUF_COLLIDER)]],
                                   FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                                   FLOWX_TID)
{
  int index = int(tid);
  if (index >= apic_node_count(P)) {
    return;
  }
  int3 node = apic_node_coord(P, index);
  float4 value = float4(0.0f);
  float4 old_value = float4(0.0f);
  for (int axis = 0; axis < 3; ++axis) {
    if (!apic_face_valid(P, node, axis)) {
      continue;
    }
    bool solid = apic_face_solid(P, cell_data, node, axis);
    if (solid) {
      /* Solid faces carry the wall's own velocity (zero for static colliders), even
       * where no particle has deposited mass on them. */
      value[axis] = apic_solid_face_velocity(P, cell_data, collider, node, axis)[axis];
    }
    float m = mass[index][axis];
    if (m <= 1e-12f) {
      continue;
    }
    float v = momentum[index][axis] / m;
    /* FLIP's reference: the transferred velocity under the same speed cap,
     * but before gravity and before solid faces are zeroed - so the change
     * FLIP hands back is gravity, walls and pressure, and not the cap. The
     * cap is a numerical limit on how far the grid moves particles per
     * substep; positions move through the (capped) grid field under FLIP
     * too, so it still does that job, without also bleeding momentum out of
     * every particle faster than it. Measured on the dam break, a reference
     * taken before the cap made FLIP lose energy faster than APIC. */
    old_value[axis] = clamp(v, -P.grid_max_speed, P.grid_max_speed);
    if (solid) {
      continue;
    }
    if (axis == 2) {
      v += P.gravity * P.dt;
    }
    value[axis] = clamp(v, -P.grid_max_speed, P.grid_max_speed);
  }
  velocity[index] = value;
  velocity_old[index] = old_value;
}
