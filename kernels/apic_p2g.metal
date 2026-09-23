/* Deterministic gather from particles in the surrounding hash cells to each
 * staggered face. */

FLOWX_KERNEL void apic_p2g(FLOWX_CONST_DEVICE float4 *positions [[buffer(BUF_POSITIONS)]],
                           FLOWX_CONST_DEVICE float4 *velocities [[buffer(BUF_VELOCITIES)]],
                           FLOWX_CONST_DEVICE float4 *affine [[buffer(BUF_AFFINE)]],
                           FLOWX_CONST_DEVICE float4 *keys [[buffer(BUF_KEYS)]],
                           FLOWX_CONST_DEVICE float *cell_start [[buffer(BUF_CELL_START)]],
                           FLOWX_CONST_DEVICE float *cell_end [[buffer(BUF_CELL_END)]],
                           FLOWX_DEVICE float4 *mass [[buffer(BUF_GRID_MASS)]],
                           FLOWX_DEVICE float4 *momentum [[buffer(BUF_GRID_MOMENTUM)]],
                           FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                           FLOWX_TID)
{
  int index = int(tid);
  if (index >= apic_node_count(P)) {
    return;
  }
  int3 node = apic_node_coord(P, index);
  float4 out_mass = float4(0.0f);
  float4 out_momentum = float4(0.0f);

  for (int axis = 0; axis < 3; ++axis) {
    if (!apic_face_valid(P, node, axis)) {
      continue;
    }
    float3 face = apic_face_position(P, node, axis);
    int3 base = cell_coord(P, face);
    float m = 0.0f;
    float q = 0.0f;
    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          int3 cell = base + int3(dx, dy, dz);
          if (!cell_in_bounds(P, cell)) {
            continue;
          }
          int c = cell_index(P, cell);
          int start = int(cell_start[c]);
          if (start < 0) {
            continue;
          }
          int end = min(int(cell_end[c]), start + MAX_CELL_SCAN);
          for (int slot = start; slot < end; ++slot) {
            int p = int(keys[slot].y);
            float3 xp = positions[p].xyz;
            float weight = apic_weight(P, xp, node, axis);
            if (weight <= 0.0f) {
              continue;
            }
            float3 row = affine[p * 3 + axis].xyz;
            float value = velocities[p][axis] + dot(row, face - xp);
            float wm = weight * P.mass;
            m += wm;
            q += wm * value;
          }
        }
      }
    }
    out_mass[axis] = m;
    out_momentum[axis] = q;
  }
  mass[index] = out_mass;
  momentum[index] = out_momentum;
}
