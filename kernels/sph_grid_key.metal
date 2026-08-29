/* Spatial-hash build, pass 1: emit one (cell key, particle index) pair per
 * slot of the sort array.
 *
 * The array is padded up to a power of two for the bitonic sort that follows;
 * padding slots get a key of `cell_count`, one past every real cell, so they
 * sort to the tail and are skipped when cell ranges are read off.
 *
 * Keys and indices stay floats, as they were under GLSL. Nothing forces that
 * any more - a buffer of uint2 would do - but both values are well inside
 * float32's exact-integer range (2^24) and changing the representation would
 * be a behavioural change in a port whose whole point is that it is not one.
 */
FLOWX_KERNEL void sph_grid_key(FLOWX_CONST_DEVICE float4 *predicted [[buffer(BUF_PREDICTED)]],
                               FLOWX_DEVICE float4 *keys [[buffer(BUF_KEYS)]],
                               FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                               FLOWX_TID)
{
  int i = int(tid);
  if (i >= P.sorted_count) {
    return;
  }

  float key = float(P.cell_count);
  if (i < P.particle_count) {
    float3 p = predicted[i].xyz;
    key = float(cell_index(P, cell_coord(P, p)));
  }

  keys[i] = float4(key, float(i), 0.0f, 0.0f);
}
