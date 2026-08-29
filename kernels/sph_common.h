/* Shared SPH helpers: grid indexing, smoothing kernels, collider lookups.
 *
 * Ported from the GLSL shaders/sph_common.glsl these replaced. The maths is
 * unchanged - this is a transcription, and it should stay one, because the
 * reference runs in scripts/golden.py are checked against results the GLSL
 * produced.
 *
 * What did change is how state is addressed. The GLSL versions took no
 * arguments and read the push-constant block as globals; here the parameter
 * block and any buffer are passed in explicitly, because MSL has no globals
 * across address spaces and CUDA will want the same shape.
 *
 * The particle_texel()/cell_texel() pair the GLSL prelude needed is simply
 * gone. Those existed to map a flat index into the 2D texture that state was
 * wrapped into, because Blender could not read a 1D texture back. Buffers
 * index directly.
 */

#ifndef FLOWX_SPH_COMMON_H
#define FLOWX_SPH_COMMON_H

#include "flowx_prelude.h"

/* --- neighbour grid ------------------------------------------------------ */

FLOWX_INLINE int3 cell_coord(FLOWX_CONSTANT Params &P, float3 p)
{
  int3 g = int3(floor((p - params_lo(P)) / P.cell_size));
  return clamp(g, int3(0), params_cells(P) - 1);
}

FLOWX_INLINE int cell_index(FLOWX_CONSTANT Params &P, int3 g)
{
  return (g.z * P.cells_y + g.y) * P.cells_x + g.x;
}

FLOWX_INLINE bool cell_in_bounds(FLOWX_CONSTANT Params &P, int3 g)
{
  return all(g >= int3(0)) && all(g < params_cells(P));
}

/* --- smoothing kernels ---------------------------------------------------- */

/* Each normalization involves a pow() of the smoothing radius, so callers
 * hoist these out of the neighbour loop and pass the result down. */

FLOWX_INLINE float poly6_coef(float h)
{
  return 315.0f / (64.0f * FLOWX_PI * pow(h, 9.0f));
}

FLOWX_INLINE float spiky_grad_coef(float h)
{
  return -45.0f / (FLOWX_PI * pow(h, 6.0f));
}

FLOWX_INLINE float visc_lap_coef(float h)
{
  return 45.0f / (FLOWX_PI * pow(h, 6.0f));
}

/* Poly6 density kernel, in terms of squared distance to avoid a sqrt. */
FLOWX_INLINE float w_poly6(float r2, float h2, float coef)
{
  if (r2 >= h2) {
    return 0.0f;
  }
  float d = h2 - r2;
  return coef * d * d * d;
}

/* Gradient of the Poly6 kernel along the separation direction. Poly6 is C2
 * (its gradient vanishes smoothly at both r=0 and r=h), which is what makes it
 * usable for a surface normal - unlike Spiky's gradient, which blows up as
 * r -> 0 and is only ever evaluated for particle *pairs* (r > 1e-6 guarded by
 * callers), Poly6's gradient is well-behaved for a single particle's own
 * neighbourhood sum. coef is poly6_coef(h) - the density kernel's own
 * normalization, reused here rather than a second one. */
FLOWX_INLINE float3 w_poly6_grad(float3 d, float r2, float h2, float coef)
{
  if (r2 >= h2) {
    return float3(0.0f);
  }
  float t = h2 - r2;
  return coef * -6.0f * t * t * d;
}

/* Magnitude of the Spiky gradient along the separation direction. Already
 * carries the kernel's negative sign via spiky_grad_coef(). */
FLOWX_INLINE float w_spiky_grad(float r, float h, float coef)
{
  if (r >= h) {
    return 0.0f;
  }
  float d = h - r;
  return coef * d * d;
}

/* Laplacian of the viscosity kernel. */
FLOWX_INLINE float w_visc_lap(float r, float h, float coef)
{
  if (r >= h) {
    return 0.0f;
  }
  return coef * (h - r);
}

/* --- collider occupancy grid ---------------------------------------------- */

/* Built on the CPU in collision/__init__.py and shares the domain origin, so
 * only a voxel size is needed to locate a point in it. A voxel size of 0 means
 * "no collider tagged"; the 1-element placeholder bound in that case reads as
 * unoccupied everywhere.
 *
 * The grid is a flat buffer now rather than a 3D image, so the z-major index
 * is spelled out here instead of coming free from imageLoad. */

FLOWX_INLINE int3 collider_coord(FLOWX_CONSTANT Params &P, float3 p)
{
  return int3(floor((p - params_lo(P)) / P.collider_voxel));
}

FLOWX_INLINE bool collider_in_bounds(FLOWX_CONSTANT Params &P, int3 c)
{
  int3 dims = int3(P.collider_x, P.collider_y, P.collider_z);
  return all(c >= int3(0)) && all(c < dims);
}

FLOWX_INLINE bool collider_occupied(FLOWX_CONSTANT Params &P,
                                    FLOWX_CONST_DEVICE float *collider,
                                    int3 c)
{
  if (!collider_in_bounds(P, c)) {
    return false;
  }
  int index = (c.z * P.collider_y + c.y) * P.collider_x + c.x;
  return collider[index] > 0.5f;
}

#endif /* FLOWX_SPH_COMMON_H */
