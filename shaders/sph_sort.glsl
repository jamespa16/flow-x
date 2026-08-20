/* Spatial-hash build, pass 2: one compare-exchange step of a bitonic sort of
 * the (cell key, particle index) pairs, ascending by key.
 *
 * A counting sort would be the usual GPU choice, but it needs imageAtomicAdd,
 * and image atomics fail to compile on Blender's Metal backend (its MSL atomic
 * header errors out before our source is even reached). Bitonic sort is pure
 * compare-exchange, so it needs no atomics at all. The host drives the
 * O(log^2 n) passes, setting i_sort = (k, j) per dispatch.
 *
 * Only the lower element of each pair does the swap, and the pairs within one
 * pass are disjoint, so no two invocations touch the same texel.
 */

void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= i_layout.w) {
    return;
  }

  int partner = i ^ i_sort.y;
  if (partner <= i) {
    return;
  }

  vec4 a = imageLoad(keys_img, particle_texel(i));
  vec4 b = imageLoad(keys_img, particle_texel(partner));

  bool ascending = (i & i_sort.x) == 0;
  if ((a.x > b.x) == ascending) {
    imageStore(keys_img, particle_texel(i), b);
    imageStore(keys_img, particle_texel(partner), a);
  }
}
