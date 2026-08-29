/* Spatial-hash build, pass 3: reset every cell's particle run to empty.
 *
 * sph_cell_range only writes cells that actually contain particles, so the -1
 * sentinel written here is what marks a cell empty for the neighbour loops in
 * sph_lambda/sph_delta/sph_normal/sph_xsph.
 */
FLOWX_KERNEL void sph_cell_clear(FLOWX_DEVICE float *cell_start [[buffer(BUF_CELL_START)]],
                                 FLOWX_DEVICE float *cell_end [[buffer(BUF_CELL_END)]],
                                 FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                                 FLOWX_TID)
{
  int c = int(tid);
  if (c >= P.cell_count) {
    return;
  }
  cell_start[c] = -1.0f;
  cell_end[c] = -1.0f;
}
