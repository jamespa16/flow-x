/* Phase 3 GPU compute smoke test: gravity + semi-implicit Euler integration,
 * no neighbors, no collisions. Isolates "can we drive GPU compute from
 * Python" from "is the SPH math correct" (that's Phase 4+).
 *
 * Blender's Python GPU API has no read-write storage buffer type - only
 * uniform buffers, samplers and images can be bound to a GPUShaderCreateInfo.
 * So particle position/velocity live in RGBA32F 1D images instead of SSBOs
 * (xyz used, w unused). positions_img/velocities_img and the push constants
 * below are declared for us by GPUShaderCreateInfo.image()/.push_constant()
 * in solver/__init__.py - this file only provides main().
 */

void main() {
  int idx = int(gl_GlobalInvocationID.x);
  if (idx >= particle_count) {
    return;
  }

  vec4 pos = imageLoad(positions_img, idx);
  vec4 vel = imageLoad(velocities_img, idx);

  vel.z += gravity * dt;
  pos.xyz += vel.xyz * dt;

  /* Minimal floor clamp so particles stay visibly inside the domain for the
   * smoke test - full boundary/collision handling is Phase 4/5. */
  if (pos.z < floor_z) {
    pos.z = floor_z;
    vel.z = 0.0;
  }

  imageStore(positions_img, idx, pos);
  imageStore(velocities_img, idx, vel);
}
