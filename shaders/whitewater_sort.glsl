/* Whitewater pass 2: one compare-exchange step of a bitonic sort of
 * ww_keys_img, descending by score (x channel).
 *
 * Standalone - unlike the other three whitewater passes it does not
 * concatenate sph_common.glsl, since it needs none of the fluid's kernel or
 * collider helpers and declaring collider_img just for this pass would waste
 * a slot for nothing. It is otherwise the same algorithm sph_sort.glsl uses
 * for the fluid's own spatial-hash build (see that file's header for why a
 * bitonic sort rather than a counting sort: no imageAtomicAdd, since image
 * atomics do not compile on Blender's Metal backend), with the comparison
 * flipped to sort largest-first and its own texel addressing since the
 * whitewater key array's width is independent of the fluid's.
 *
 * i_sort = (k, j, count, width): count is the power-of-two array length
 * (the fluid's own sorted_count - whitewater_potential.glsl sizes
 * ww_keys_img identically), width is the texture's row width.
 */

ivec2 ww_key_texel(int i, int width)
{
  return ivec2(i % width, i / width);
}

void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= i_sort.z) {
    return;
  }

  int partner = i ^ i_sort.y;
  if (partner <= i) {
    return;
  }

  int width = i_sort.w;
  vec4 a = imageLoad(ww_keys_img, ww_key_texel(i, width));
  vec4 b = imageLoad(ww_keys_img, ww_key_texel(partner, width));

  bool want_descending = (i & i_sort.x) == 0;
  if ((a.x < b.x) == want_descending) {
    imageStore(ww_keys_img, ww_key_texel(i, width), b);
    imageStore(ww_keys_img, ww_key_texel(partner, width), a);
  }
}
