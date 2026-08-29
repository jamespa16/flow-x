/* Spatial-hash build, pass 2: one compare-exchange step of a bitonic sort of
 * the (cell key, particle index) pairs, ascending by key.
 *
 * A counting sort is the usual GPU choice and needs an atomic add. Under
 * Blender's `gpu` module that was impossible - image atomics would not compile
 * on its Metal backend at all - so this bitonic sort was the only option, at
 * O(log^2 n) host-driven passes (~105 per substep, the dominant cost). Owning
 * the device removes that constraint, and the counting sort is the intended
 * replacement; this pass is kept for now so the port can be checked against
 * reference runs before the algorithm changes underneath it.
 *
 * Only the lower element of each pair does the swap, and the pairs within one
 * pass are disjoint, so no two invocations touch the same slot.
 */
FLOWX_KERNEL void sph_sort(FLOWX_DEVICE float4 *keys [[buffer(BUF_KEYS)]],
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

  float4 a = keys[i];
  float4 b = keys[partner];

  bool ascending = (i & P.bitonic_k) == 0;
  if ((a.x > b.x) == ascending) {
    keys[i] = b;
    keys[partner] = a;
  }
}
