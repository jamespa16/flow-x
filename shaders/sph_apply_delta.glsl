/* PBF pass: commit the position correction computed by sph_delta.glsl.
 *
 * Split from sph_delta.glsl for the same reason sph_force is split from
 * sph_integrate: sph_delta reads every particle's predicted_img while
 * computing deltas, so nothing may write predicted_img until that whole pass
 * is done. This pass runs after it, one texel per particle, no cross-particle
 * reads - the write is race-free.
 */

void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= particle_count()) {
    return;
  }

  ivec2 t = particle_texel(i);
  vec3 p = imageLoad(predicted_img, t).xyz + imageLoad(delta_img, t).xyz;
  imageStore(predicted_img, t, vec4(p, 1.0));
}
