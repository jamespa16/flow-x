/* Whitewater pass 4: integrate every pool slot one frame forward.
 *
 * Dispatched over the full pool capacity, not just the live particles - there
 * is no compacted "alive list" (see whitewater_spawn.glsl for why: the ring
 * buffer's overwrite-on-wrap is the only bookkeeping this system does), so a
 * dead slot (life <= 0) is skipped cheaply rather than iterated around.
 *
 * Per-kind behavior is a coarse stand-in for real whitewater physics, in the
 * same spirit as the rest of this pass family - visually distinct, not a
 * second SPH solve, and deliberately not sampling the fluid's own velocity
 * field: that would need positions_img/velocities_img/keys_img/cell_start_img/
 * cell_end_img on top of the pool's own two images and collider_img, landing
 * exactly at Metal's 8 read-write-image cap rather than safely under it (see
 * shaders/sph_common.glsl's header and solver/sph.py's _images_for_pass,
 * which both treat 8 as the number to stay under, not to touch). A drag term
 * toward a fixed damping target reads close enough to "rides the surface" for
 * foam and "gets dragged along" for bubbles without needing the fluid grid.
 *   spray  - ballistic: gravity only.
 *   foam   - drag damps horizontal motion (it settles where it lands), tiny
 *            gravity so it doesn't stack forever.
 *   bubble - buoyant (gravity partly or fully cancelled), light drag so it
 *            doesn't accelerate forever on the way up.
 *
 * Push constants (this pass's own 128-byte block):
 *   i_ww     = (pool_tex_width, pool_capacity, unused, unused)
 *   i_layout, i_grid = the fluid's own blocks. This pass's *body* doesn't use
 *              them, but sph_common.glsl's particle_texel()/cell_texel()/
 *              cell_coord()/cell_index()/cell_in_bounds() still reference
 *              i_layout/i_grid even though nothing here calls those
 *              functions - the concatenated prelude is one compilation unit,
 *              and an identifier it mentions must be declared regardless of
 *              whether the pass's own body reaches it (see sph.py's
 *              _images_for_pass docstring for the image-side version of the
 *              same rule; it applies to push constants too).
 *   f_lo     = (domain_min.xyz, unused) - collider_coord()'s origin
 *   f_hi     = (domain_max.xyz, unused)
 *   f_sim    = (gravity, dt, drag, buoyancy) - buoyancy is the fraction of
 *              gravity cancelled for bubbles (1.0 = neutral, >1.0 = rises)
 */

void main()
{
  int slot = int(gl_GlobalInvocationID.x);
  if (slot >= i_ww.y) {
    return;
  }

  ivec2 t = ivec2(slot % i_ww.x, slot / i_ww.x);
  vec4 pos_life = imageLoad(ww_positions_img, t);
  float life = pos_life.w;
  if (life <= 0.0) {
    return;
  }

  vec4 vel_kind = imageLoad(ww_velocity_kind_img, t);
  vec3 p = pos_life.xyz;
  vec3 v = vel_kind.xyz;
  int kind = int(vel_kind.w);

  float gravity = f_sim.x;
  float dt = f_sim.y;
  float drag = f_sim.z;
  float buoyancy = f_sim.w;

  if (kind == 0) {
    v.z += gravity * dt;
  }
  else if (kind == 2) {
    v.z += gravity * (1.0 - buoyancy) * dt;
    v.xy *= clamp(1.0 - drag * dt, 0.0, 1.0);
  }
  else {
    v *= clamp(1.0 - drag * dt, 0.0, 1.0);
    v.z += gravity * 0.1 * dt;
  }

  p += v * dt;
  life -= dt;

  /* Leaving the domain by more than a couple of cells retires the particle
   * outright rather than clamping it to a wall it was never simulated
   * against - unlike the fluid particles, whitewater has no obligation to
   * stay inside the box. */
  vec3 margin = vec3(0.2);
  if (any(lessThan(p, f_lo.xyz - margin)) || any(greaterThan(p, f_hi.xyz + margin))) {
    life = 0.0;
  }

  if (collider_voxel_size() > 0.0 && collider_occupied(collider_coord(p))) {
    p -= v * dt;
    v *= 0.2;
  }

  imageStore(ww_positions_img, t, vec4(p, life));
  imageStore(ww_velocity_kind_img, t, vec4(v, float(kind)));
}
