/* Portability layer shared by every Flow-X compute kernel.
 *
 * The kernels are written in the subset of C++ that Metal Shading Language and
 * CUDA both accept, with everything that actually differs between them hidden
 * behind the macros below. A CUDA backend then reuses the pass bodies as they
 * stand and only has to supply the other half of each #ifdef here - which is
 * the whole reason this file exists rather than the address-space keywords
 * being written inline.
 *
 * These kernels replaced a set of GLSL passes that ran on Blender's `gpu`
 * module. Two differences are worth knowing when reading them against that
 * history:
 *
 * * State is in flat device buffers, not 2D images. The old code wrapped every
 *   array into a 2D texture at a shared width because Blender could not read a
 *   1D texture back, and every pass paid for that with an index round-trip.
 *   Indexing here is just the thread id.
 * * There is one parameter struct, and it has room. The old passes shared a
 *   128-byte push-constant block that was completely full, so new parameters
 *   had to be bit-packed into unused lanes of existing ones. Params below is an
 *   ordinary struct; add a field.
 */

#ifndef FLOWX_PRELUDE_H
#define FLOWX_PRELUDE_H

#ifdef __METAL_VERSION__

#include <metal_stdlib>
using namespace metal;

#define FLOWX_DEVICE device
#define FLOWX_CONST_DEVICE device const
#define FLOWX_CONSTANT constant
#define FLOWX_KERNEL kernel
#define FLOWX_INLINE inline
#define FLOWX_TID uint tid [[thread_position_in_grid]]

#else /* CUDA - not built yet; the shape the port has to fill in. */

#define FLOWX_DEVICE
#define FLOWX_CONST_DEVICE const
#define FLOWX_CONSTANT const
#define FLOWX_KERNEL extern "C" __global__
#define FLOWX_INLINE __device__ inline
#define FLOWX_TID
/* CUDA has float3/float4 but no operators on them; a port supplies those here
 * along with a thread-id definition, and the pass bodies stay untouched. */

#endif

/* Buffer binding indices, shared by every pass.
 *
 * One table for all kernels, rather than a per-pass list. The old GLSL passes
 * could not do this: Metal caps read-write *images* at 8 per shader and keeps
 * every declared slot, so each pass had to declare only the images its source
 * mentioned, worked out by scanning the source text. Buffers have no such cap,
 * so a pass simply declares the slots it wants at their fixed index and ignores
 * the rest. Sparse binding is fine.
 */
#define BUF_POSITIONS 0
#define BUF_VELOCITIES 1
#define BUF_LAMBDA 2
#define BUF_PREDICTED 3
#define BUF_DELTA 4
#define BUF_NORMAL 5
#define BUF_KEYS 6
#define BUF_CELL_START 7
#define BUF_CELL_END 8
#define BUF_COLLIDER 9
#define BUF_SURFACE 10
#define BUF_WW_KEYS 11
#define BUF_WW_POSITIONS 12
#define BUF_WW_VELKIND 13
#define BUF_PARAMS 14

/* Every parameter every pass needs, in one block.
 *
 * Scalars only, deliberately: MSL aligns float3 to 16 bytes, and a struct with
 * vectors in it would need matching padding on the Python side. All-scalar
 * means the layout is exactly what struct.pack produces, and PARAMS_FORMAT in
 * solver/engine/params.py is the one place that has to agree with it.
 *
 * Keep this in sync with PARAMS_FORMAT and PARAMS_FIELDS. Field order matters.
 */
struct Params {
  /* layout */
  int particle_count;
  int sorted_count;
  int cell_count;
  /* grid */
  int cells_x;
  int cells_y;
  int cells_z;
  /* bitonic sort step, set per dispatch */
  int bitonic_k;
  int bitonic_j;
  /* domain bounds */
  float lo_x;
  float lo_y;
  float lo_z;
  float cell_size;
  float hi_x;
  float hi_y;
  float hi_z;
  float particle_radius;
  /* SPH */
  float smoothing_radius;
  float mass;
  float rest_density;
  float relaxation;
  /* integration */
  float viscosity;
  float dt;
  float gravity;
  float boundary_damping;
  /* These two used to ride bit-packed in spare lanes of the sort slot, for
   * want of 8 spare bytes in the push-constant block. They are just floats. */
  float scorr_k;
  float surface_tension;
  /* collider occupancy grid; shares the domain origin, so only a voxel size */
  int collider_x;
  int collider_y;
  int collider_z;
  float collider_voxel;
  /* surface splat lattice */
  int surface_x;
  int surface_y;
  int surface_z;
  float surface_spacing;
  float surface_lo_x;
  float surface_lo_y;
  float surface_lo_z;
  float surface_kernel_radius;
  /* whitewater */
  int ww_capacity;
  int ww_cursor;
  int ww_spawn_count;
  int frame;
  float trapped_air_weight;
  float wave_crest_weight;
  float kinetic_weight;
  float kinetic_reference_speed;
  float spray_speed_threshold;
  float bubble_trapped_threshold;
  float jitter_strength;
  float normal_offset;
  float spray_life_min;
  float spray_life_max;
  float foam_life_min;
  float foam_life_max;
  float bubble_life_min;
  float bubble_life_max;
  float ww_drag;
  float ww_buoyancy;
  float frame_dt;
};

#define FLOWX_PI 3.14159265358979323846f

/* Defensive bound on a single cell's particle run. A cell range is built from
 * sorted keys and should always be small, but an unbounded data-dependent loop
 * in a compute kernel is a hang waiting to happen. */
#define MAX_CELL_SCAN 512

FLOWX_INLINE float3 params_lo(FLOWX_CONSTANT Params &P)
{
  return float3(P.lo_x, P.lo_y, P.lo_z);
}

FLOWX_INLINE float3 params_hi(FLOWX_CONSTANT Params &P)
{
  return float3(P.hi_x, P.hi_y, P.hi_z);
}

FLOWX_INLINE int3 params_cells(FLOWX_CONSTANT Params &P)
{
  return int3(P.cells_x, P.cells_y, P.cells_z);
}

#endif /* FLOWX_PRELUDE_H */
