/* Spatial-hash build, pass 3: reset every cell's particle run to empty.
 *
 * sph_cell_range.glsl only writes cells that actually contain particles, so
 * the -1 sentinel written here is what marks a cell empty for the neighbor
 * loops in sph_density/sph_force.
 */

void main()
{
  int c = int(gl_GlobalInvocationID.x);
  if (c >= i_grid.w) {
    return;
  }
  imageStore(cell_start_img, cell_texel(c), vec4(-1.0));
  imageStore(cell_end_img, cell_texel(c), vec4(-1.0));
}
