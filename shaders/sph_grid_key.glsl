/* Spatial-hash build, pass 1: emit one (cell key, particle index) pair per
 * slot of the sort array.
 *
 * The array is padded up to a power of two for the bitonic sort that follows;
 * padding slots get a key of `cell_count`, one past every real cell, so they
 * sort to the tail and are skipped when cell ranges are read off.
 *
 * Keys and indices are stored as floats in an RGBA32F image rather than in an
 * integer image: Blender's Python GPU API can only upload/initialize textures
 * from FLOAT buffers, and both values stay well inside float32's exact-integer
 * range (2^24).
 */

void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= i_layout.w) {
    return;
  }

  float key = float(i_grid.w);
  if (i < particle_count()) {
    vec3 p = imageLoad(positions_img, particle_texel(i)).xyz;
    key = float(cell_index(cell_coord(p)));
  }

  imageStore(keys_img, particle_texel(i), vec4(key, float(i), 0.0, 0.0));
}
