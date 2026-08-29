/* Whitewater pass 2: one compare-exchange step of a bitonic sort of the
 * potential scores, descending.
 *
 * The same algorithm sph_sort uses for the fluid's spatial-hash build, with
 * the comparison flipped to sort largest-first. See sph_sort for why a bitonic
 * sort rather than a counting sort, and why that reason no longer holds.
 *
 * The array length is the fluid's own sorted_count - whitewater_potential
 * fills exactly that many slots.
 */
FLOWX_KERNEL void whitewater_sort(FLOWX_DEVICE float4 *ww_keys [[buffer(BUF_WW_KEYS)]],
                                  FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                                  FLOWX_TID)
{
  int i = int(tid);
  if (i >= P.sorted_count) {
    return;
  }

  int partner = i ^ P.bitonic_j;
  if (partner <= i) {
    return;
  }

  float4 a = ww_keys[i];
  float4 b = ww_keys[partner];

  bool want_descending = (i & P.bitonic_k) == 0;
  if ((a.x < b.x) == want_descending) {
    ww_keys[i] = b;
    ww_keys[partner] = a;
  }
}
