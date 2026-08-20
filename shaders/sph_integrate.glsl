/* SPH pass 3: semi-implicit Euler integration plus a clamp to the domain box.
 *
 * Collider-aware boundary handling arrives in Phase 5; for now the only
 * obstacles are the six domain walls.
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

  vec3 v = imageLoad(velocities_img, t).xyz + imageLoad(forces_img, t).xyz * dt;

  /* A particle must not cross more than a fraction of a smoothing radius in
   * one substep, or it tunnels past the neighbors that would have pushed back
   * and the simulation blows up. Clamping speed is a cruder guard than
   * shrinking dt, but it keeps a bad parameter set recoverable. */
  float v_max = 0.4 * h / max(dt, 1e-6);
  float speed = length(v);
  if (speed > v_max) {
    v *= v_max / speed;
  }

  vec3 p = imageLoad(positions_img, t).xyz + v * dt;

  float radius = f_hi.w;
  float damping = f_sim.w;
  vec3 lo = f_lo.xyz + vec3(radius);
  vec3 hi = f_hi.xyz - vec3(radius);

  if (p.x < lo.x) {
    p.x = lo.x;
    v.x = -v.x * damping;
  }
  else if (p.x > hi.x) {
    p.x = hi.x;
    v.x = -v.x * damping;
  }
  if (p.y < lo.y) {
    p.y = lo.y;
    v.y = -v.y * damping;
  }
  else if (p.y > hi.y) {
    p.y = hi.y;
    v.y = -v.y * damping;
  }
  if (p.z < lo.z) {
    p.z = lo.z;
    v.z = -v.z * damping;
  }
  else if (p.z > hi.z) {
    p.z = hi.z;
    v.z = -v.z * damping;
  }

  imageStore(positions_img, t, vec4(p, 1.0));
  imageStore(velocities_img, t, vec4(v, 0.0));
}
