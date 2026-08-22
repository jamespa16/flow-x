/* Whitewater pass 3: birth this frame's new secondary particles from the
 * top of the sorted potential-score array into a fixed-size ring buffer.
 *
 * The spawn budget K (i_ww.w) is a domain setting, not something read back
 * from the GPU, which is what lets this dispatch a fixed K threads and skip
 * both an atomic allocation counter (unavailable - see whitewater_sort.glsl)
 * and a GPU->CPU readback of "how many qualified this frame". Thread t just
 * takes sorted slot t (already descending by score after the sort passes)
 * and writes into ring slot (cursor + t) mod capacity, unconditionally
 * overwriting whatever lived there - which is also this system's entire
 * garbage-collection story: once the ring wraps, stale particles are simply
 * replaced rather than explicitly killed. solver/whitewater.py advances the
 * CPU-side cursor by K every frame, deterministically, the same way
 * solver/sph.py tracks seed_frame/last_frame.
 *
 * Classification is a coarse approximation of Ihmsen et al.'s spray/foam/
 * bubble split, from the two sub-scores whitewater_potential.glsl already
 * computed: a particle whose trapped-air sub-score dominates becomes a
 * bubble; otherwise fast particles become spray and slow ones foam.
 *
 * Push constants (this pass's own 128-byte block, separate from both the
 * SPH block and whitewater_advect.glsl's - see solver/whitewater.py):
 *   i_layout = the fluid's own block, reused unmodified so particle_texel()
 *              addresses both the fluid's textures and ww_keys_img (which
 *              whitewater_potential.glsl sized identically to keys_img)
 *   i_grid   = the fluid's own block. This pass's body never calls
 *              cell_index()/cell_in_bounds(), but sph_common.glsl still
 *              defines them referencing i_grid, and the concatenated
 *              prelude is one compilation unit - an identifier it mentions
 *              anywhere must be declared, whether or not this pass's body
 *              reaches it (see whitewater_advect.glsl's header for the same
 *              rule, and sph.py's _images_for_pass docstring for the
 *              image-side version of it).
 *   i_ww     = (pool_tex_width, pool_capacity, ring_cursor, spawn_count)
 *   f_lo     = (domain_min.xyz, cell_size) - collider_coord()'s origin
 *   f_kind   = (spray_speed_threshold, bubble_trapped_air_threshold,
 *               jitter_strength, normal_offset)
 *   f_life   = (spray_life_min, spray_life_max, foam_life_min, foam_life_max)
 *   f_life2  = (bubble_life_min, bubble_life_max, unused, frame_seed)
 */

vec3 ww_hash3(float seed)
{
  vec3 h = vec3(sin(seed * 12.9898), sin(seed * 78.233), sin(seed * 37.719)) * 43758.5453;
  return fract(h) * 2.0 - 1.0;
}

void main()
{
  int t = int(gl_GlobalInvocationID.x);
  int spawn_count = i_ww.w;
  if (t >= spawn_count) {
    return;
  }

  vec4 key = imageLoad(ww_keys_img, particle_texel(t));
  float score = key.x;
  int src = int(key.y);
  if (src < 0 || score <= 0.0) {
    return; /* padding slot, or nothing scored above zero at this rank */
  }

  vec3 p = imageLoad(positions_img, particle_texel(src)).xyz;
  vec3 v = imageLoad(velocities_img, particle_texel(src)).xyz;
  float trapped_air = key.z;
  float speed = length(v);

  int kind; /* 0 = spray, 1 = foam, 2 = bubble */
  if (trapped_air > f_kind.y) {
    kind = 2;
  }
  else if (speed > f_kind.x) {
    kind = 0;
  }
  else {
    kind = 1;
  }

  float seed = f_life2.w + float(t) * 0.6180339887;
  vec3 jitter = ww_hash3(seed);

  vec3 spawn_pos = p + jitter * f_kind.w;
  vec3 spawn_vel = v + jitter * f_kind.z;

  float life;
  if (kind == 0) {
    life = mix(f_life.x, f_life.y, fract(seed * 1.618034));
  }
  else if (kind == 1) {
    life = mix(f_life.z, f_life.w, fract(seed * 2.718282));
  }
  else {
    life = mix(f_life2.x, f_life2.y, fract(seed * 3.141593));
  }

  if (collider_voxel_size() > 0.0 && collider_occupied(collider_coord(spawn_pos))) {
    life = 0.0; /* don't spawn inside geometry; the slot just stays retired */
  }

  int slot = (i_ww.z + t) % i_ww.y;
  ivec2 texel = ivec2(slot % i_ww.x, slot / i_ww.x);
  imageStore(ww_positions_img, texel, vec4(spawn_pos, life));
  imageStore(ww_velocity_kind_img, texel, vec4(spawn_vel, float(kind)));
}
