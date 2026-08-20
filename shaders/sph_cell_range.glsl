/* Spatial-hash build, pass 4: turn the sorted key array into per-cell
 * [start, end) ranges by spotting the boundaries where the key changes.
 *
 * start and end live in two separate single-channel images rather than two
 * channels of one image: they're written by different invocations, and a
 * read-modify-write of shared channels would race. As written, every texel has
 * exactly one writer.
 */

void main()
{
  int i = int(gl_GlobalInvocationID.x);
  int n = i_layout.w;
  if (i >= n) {
    return;
  }

  int key = int(imageLoad(keys_img, particle_texel(i)).x);
  if (key >= i_grid.w) {
    return; /* padding slot */
  }

  int prev = (i == 0) ? -1 : int(imageLoad(keys_img, particle_texel(i - 1)).x);
  int next = (i == n - 1) ? -1 : int(imageLoad(keys_img, particle_texel(i + 1)).x);

  if (key != prev) {
    imageStore(cell_start_img, cell_texel(key), vec4(float(i)));
  }
  if (key != next) {
    imageStore(cell_end_img, cell_texel(key), vec4(float(i + 1)));
  }
}
