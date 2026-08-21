/* PBF pass: derive velocity from the position change the constraint loop
 * produced (Macklin & Muller 2013, eq. 17): v_i = (p_pred - p_old) / dt.
 *
 * Split from XSPH smoothing (sph_xsph.glsl) and finalization (sph_finalize.glsl)
 * because XSPH needs every particle's *raw* velocity from this formula before
 * any of them are touched again - writing a smoothed velocity back into
 * velocities_img in the same pass that other invocations still read it from
 * would race, the same Jacobi hazard the grid-build and constraint-solve
 * passes already avoid by writing corrections to a separate image first.
 */

void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= particle_count()) {
    return;
  }

  ivec2 t = particle_texel(i);
  float dt = f_sim.y;
  vec3 p_pred = imageLoad(predicted_img, t).xyz;
  vec3 p_old = imageLoad(positions_img, t).xyz;
  imageStore(velocities_img, t, vec4((p_pred - p_old) / max(dt, 1e-6), 0.0));
}
