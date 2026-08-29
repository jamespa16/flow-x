/* Whitewater pass 3: birth this frame's new secondary particles from the top
 * of the sorted potential-score array into a fixed-size ring buffer.
 *
 * The spawn budget K is a domain setting, not something read back from the
 * GPU, which is what lets this dispatch a fixed K threads and skip both an
 * atomic allocation counter and a device-to-host readback of "how many
 * qualified this frame". Thread t takes sorted slot t (already descending by
 * score) and writes into ring slot (cursor + t) mod capacity, unconditionally
 * overwriting whatever lived there - which is also this system's entire
 * garbage-collection story: once the ring wraps, stale particles are replaced
 * rather than explicitly killed.
 *
 * The atomic counter that would make this a real compaction - particles dying
 * when their lifetime expires rather than when the cursor laps them - is
 * available now that the solver owns its device. It is deliberately not used
 * yet: this pass is a transcription of what the GLSL did, so the port can be
 * checked against reference runs before the behaviour changes.
 *
 * Classification is a coarse approximation of Ihmsen et al.'s spray/foam/
 * bubble split, from the two sub-scores whitewater_potential computed: a
 * particle whose trapped-air sub-score dominates becomes a bubble; otherwise
 * fast particles become spray and slow ones foam.
 */

FLOWX_INLINE float3 ww_hash3(float seed)
{
  float3 h = float3(sin(seed * 12.9898f), sin(seed * 78.233f), sin(seed * 37.719f)) * 43758.5453f;
  return fract(h) * 2.0f - 1.0f;
}

FLOWX_KERNEL void whitewater_spawn(FLOWX_CONST_DEVICE float4 *positions [[buffer(BUF_POSITIONS)]],
                                   FLOWX_CONST_DEVICE float4 *velocities [[buffer(BUF_VELOCITIES)]],
                                   FLOWX_CONST_DEVICE float4 *ww_keys [[buffer(BUF_WW_KEYS)]],
                                   FLOWX_DEVICE float4 *ww_positions [[buffer(BUF_WW_POSITIONS)]],
                                   FLOWX_DEVICE float4 *ww_velkind [[buffer(BUF_WW_VELKIND)]],
                                   FLOWX_CONST_DEVICE float *collider [[buffer(BUF_COLLIDER)]],
                                   FLOWX_CONSTANT Params &P [[buffer(BUF_PARAMS)]],
                                   FLOWX_TID)
{
  int t = int(tid);
  if (t >= P.ww_spawn_count) {
    return;
  }

  float4 key = ww_keys[t];
  float score = key.x;
  int src = int(key.y);
  if (src < 0 || score <= 0.0f) {
    return; /* padding slot, or nothing scored above zero at this rank */
  }

  float3 p = positions[src].xyz;
  float3 v = velocities[src].xyz;
  float trapped_air = key.z;
  float speed = length(v);

  int kind; /* 0 = spray, 1 = foam, 2 = bubble */
  if (trapped_air > P.bubble_trapped_threshold) {
    kind = 2;
  }
  else if (speed > P.spray_speed_threshold) {
    kind = 0;
  }
  else {
    kind = 1;
  }

  float seed = float(P.frame) + float(t) * 0.6180339887f;
  float3 jitter = ww_hash3(seed);

  float3 spawn_pos = p + jitter * P.normal_offset;
  float3 spawn_vel = v + jitter * P.jitter_strength;

  float life;
  if (kind == 0) {
    life = mix(P.spray_life_min, P.spray_life_max, fract(seed * 1.618034f));
  }
  else if (kind == 1) {
    life = mix(P.foam_life_min, P.foam_life_max, fract(seed * 2.718282f));
  }
  else {
    life = mix(P.bubble_life_min, P.bubble_life_max, fract(seed * 3.141593f));
  }

  if (P.collider_voxel > 0.0f && collider_occupied(P, collider, collider_coord(P, spawn_pos))) {
    life = 0.0f; /* don't spawn inside geometry; the slot just stays retired */
  }

  int slot = (P.ww_cursor + t) % P.ww_capacity;
  ww_positions[slot] = float4(spawn_pos, life);
  ww_velkind[slot] = float4(spawn_vel, float(kind));
}
