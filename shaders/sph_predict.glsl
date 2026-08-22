/* PBF pass 1: apply external forces to velocity, then predict a position.
 *
 * This is PBF's replacement for WCSPH's pressure force: there is no pressure
 * term here at all, only gravity and the surface-tension cohesion force
 * (Morris 2000) read from normal_img, which sph_normal.glsl computed from
 * last substep's state before this substep's grid was rebuilt. Incompressi-
 * bility is restored afterwards by the constraint-solve loop (sph_lambda/
 * sph_delta/sph_apply_delta), not by anything in this pass.
 *
 * i_sort.w carries the surface tension coefficient sigma (bit-packed as a
 * float - see sph.py's push_constant_values for why it rides in an otherwise-
 * unused sort lane); 0 makes the term a no-op without needing a branch to
 * skip the pass entirely.
 *
 * The speed clamp lives here rather than after the constraint loop, because
 * the constraint loop's correctness depends on the neighbor grid built from
 * this pass's output (predicted_img) being complete - a particle that jumps
 * more than a smoothing radius before the grid is built can miss neighbors
 * that should have constrained it, which is a correctness problem for PBF's
 * density constraint, not just a stability one the way it was for WCSPH.
 */

void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= particle_count()) {
    return;
  }

  ivec2 t = particle_texel(i);
  float dt = f_sim.y;
  float h = f_sph.x;
  float mass = f_sph.y;
  float sigma = intBitsToFloat(i_sort.w);

  vec3 v = imageLoad(velocities_img, t).xyz;
  v.z += f_sim.z * dt;

  if (sigma > 0.0) {
    vec4 n = imageLoad(normal_img, t);
    float mag = length(n.xyz);
    if (mag > 1e-4) {
      v += (-sigma * n.w * n.xyz / mag / mass) * dt;
    }
  }

  float v_max = 0.4 * h / max(dt, 1e-6);
  float speed = length(v);
  if (speed > v_max) {
    v *= v_max / speed;
  }

  vec3 p = imageLoad(positions_img, t).xyz + v * dt;

  imageStore(velocities_img, t, vec4(v, 0.0));
  imageStore(predicted_img, t, vec4(p, 1.0));
}
