/* Spatial-hash build, pass 4: turn the sorted key array into per-cell
 * [start, end) ranges by spotting the boundaries where the key changes.
 *
 * start and end live in two separate buffers rather than two channels of one:
 * they're written by different invocations, and a read-modify-write of shared
 * channels would race. As written, every slot has exactly one writer.
 */
FLOWX_KERNEL void sph_cell_range(FLOWX_CONST_DEVICE float4 *keys [[buffer(BUF_KEYS)]],
                                 FLOWX_DEVICE float *cell_start [[buffer(BUF_CELL_START)]],
                                 FLOWX_DEVICE float *cell_end [[buffer(BUF_CELL_END)]],
                                 FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                                 FLOWX_TID)
{
  int i = int(tid);
  int n = P.sorted_count;
  if (i >= n) {
    return;
  }

  int key = int(keys[i].x);
  if (key >= P.cell_count) {
    return; /* padding slot */
  }

  int prev = (i == 0) ? -1 : int(keys[i - 1].x);
  int next = (i == n - 1) ? -1 : int(keys[i + 1].x);

  if (key != prev) {
    cell_start[key] = float(i);
  }
  if (key != next) {
    cell_end[key] = float(i + 1);
  }
}
