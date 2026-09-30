/* Shared staggered-grid helpers for the APIC passes. */

#ifndef FLOWX_APIC_COMMON_H
#define FLOWX_APIC_COMMON_H

#include "sph_common.h"

FLOWX_INLINE int3 apic_nodes(FLOWX_CONSTANT Params &P)
{
  return int3(P.nodes_x, P.nodes_y, P.nodes_z);
}

FLOWX_INLINE int apic_node_count(FLOWX_CONSTANT Params &P)
{
  return P.nodes_x * P.nodes_y * P.nodes_z;
}

FLOWX_INLINE int3 apic_node_coord(FLOWX_CONSTANT Params &P, int index)
{
  int x = index % P.nodes_x;
  int y = (index / P.nodes_x) % P.nodes_y;
  int z = index / (P.nodes_x * P.nodes_y);
  return int3(x, y, z);
}

FLOWX_INLINE int apic_node_index(FLOWX_CONSTANT Params &P, int3 node)
{
  return (node.z * P.nodes_y + node.y) * P.nodes_x + node.x;
}

FLOWX_INLINE int3 apic_cell_coord(FLOWX_CONSTANT Params &P, int index)
{
  int x = index % P.cells_x;
  int y = (index / P.cells_x) % P.cells_y;
  int z = index / (P.cells_x * P.cells_y);
  return int3(x, y, z);
}

FLOWX_INLINE bool apic_face_valid(FLOWX_CONSTANT Params &P, int3 node, int axis)
{
  if (axis == 0) {
    return node.x <= P.cells_x && node.y < P.cells_y && node.z < P.cells_z;
  }
  if (axis == 1) {
    return node.x < P.cells_x && node.y <= P.cells_y && node.z < P.cells_z;
  }
  return node.x < P.cells_x && node.y < P.cells_y && node.z <= P.cells_z;
}

FLOWX_INLINE float3 apic_face_offset(int axis)
{
  if (axis == 0) {
    return float3(0.0f, 0.5f, 0.5f);
  }
  if (axis == 1) {
    return float3(0.5f, 0.0f, 0.5f);
  }
  return float3(0.5f, 0.5f, 0.0f);
}

FLOWX_INLINE float3 apic_face_position(FLOWX_CONSTANT Params &P, int3 node, int axis)
{
  return params_lo(P) + (float3(node) + apic_face_offset(axis)) * P.grid_spacing;
}

FLOWX_INLINE float apic_weight(FLOWX_CONSTANT Params &P, float3 particle,
                               int3 node, int axis)
{
  float3 q = (particle - params_lo(P)) / P.grid_spacing - apic_face_offset(axis);
  float3 d = abs(q - float3(node));
  if (any(d >= float3(1.0f))) {
    return 0.0f;
  }
  float3 w = float3(1.0f) - d;
  return w.x * w.y * w.z;
}

FLOWX_INLINE float apic_cell_type(FLOWX_CONSTANT Params &P,
                                  FLOWX_CONST_DEVICE float4 *cell_data,
                                  int3 cell)
{
  if (!cell_in_bounds(P, cell)) {
    return -1.0f;
  }
  return cell_data[cell_index(P, cell)].w;
}

FLOWX_INLINE bool apic_face_solid(FLOWX_CONSTANT Params &P,
                                  FLOWX_CONST_DEVICE float4 *cell_data,
                                  int3 node, int axis)
{
  int3 low = node;
  low[axis] -= 1;
  int3 high = node;
  if (!cell_in_bounds(P, low) || !cell_in_bounds(P, high)) {
    return true;
  }
  return apic_cell_type(P, cell_data, low) < 0.0f ||
         apic_cell_type(P, cell_data, high) < 0.0f;
}

/* Animated colliders are packed as N occupancy floats followed by 3N wall
 * velocity floats.  Static colliders keep the old occupancy-only allocation;
 * collider_motion keeps this read optional in every APIC pass. */
FLOWX_INLINE float3 apic_collider_velocity(FLOWX_CONSTANT Params &P,
                                           FLOWX_CONST_DEVICE float *collider,
                                           int3 c)
{
  if (P.collider_motion == 0 || !collider_in_bounds(P, c)) {
    return float3(0.0f);
  }
  int count = P.collider_x * P.collider_y * P.collider_z;
  int index = (c.z * P.collider_y + c.y) * P.collider_x + c.x;
  int offset = count + index * 3;
  return float3(collider[offset], collider[offset + 1], collider[offset + 2]);
}

FLOWX_INLINE float3 apic_wall_velocity(FLOWX_CONSTANT Params &P,
                                       FLOWX_CONST_DEVICE float *collider,
                                       float3 point)
{
  if (P.collider_motion == 0 || P.collider_voxel <= 0.0f) {
    return float3(0.0f);
  }
  return apic_collider_velocity(P, collider, collider_coord(P, point));
}

FLOWX_INLINE float3 apic_solid_face_velocity(FLOWX_CONSTANT Params &P,
                                             FLOWX_CONST_DEVICE float4 *cell_data,
                                             FLOWX_CONST_DEVICE float *collider,
                                             int3 node, int axis)
{
  int3 low = node;
  low[axis] -= 1;
  int3 high = node;
  float3 result = float3(0.0f);
  int count = 0;
  if (cell_in_bounds(P, low) && apic_cell_type(P, cell_data, low) < 0.0f) {
    float3 center = params_lo(P) + (float3(low) + 0.5f) * P.grid_spacing;
    result += apic_wall_velocity(P, collider, center);
    count++;
  }
  if (cell_in_bounds(P, high) && apic_cell_type(P, cell_data, high) < 0.0f) {
    float3 center = params_lo(P) + (float3(high) + 0.5f) * P.grid_spacing;
    result += apic_wall_velocity(P, collider, center);
    count++;
  }
  return count > 0 ? result / float(count) : float3(0.0f);
}

FLOWX_INLINE float apic_pressure(FLOWX_CONSTANT Params &P,
                                 FLOWX_CONST_DEVICE float4 *cell_data,
                                 int3 cell)
{
  if (!cell_in_bounds(P, cell)) {
    return 0.0f;
  }
  float4 value = cell_data[cell_index(P, cell)];
  return P.pressure_ping == 0 ? value.y : value.z;
}

FLOWX_INLINE float3 apic_cell_velocity(FLOWX_CONSTANT Params &P,
                                       FLOWX_CONST_DEVICE float4 *velocity,
                                       int3 cell)
{
  int i = apic_node_index(P, cell);
  int ix = apic_node_index(P, cell + int3(1, 0, 0));
  int iy = apic_node_index(P, cell + int3(0, 1, 0));
  int iz = apic_node_index(P, cell + int3(0, 0, 1));
  return float3(0.5f * (velocity[i].x + velocity[ix].x),
                0.5f * (velocity[i].y + velocity[iy].y),
                0.5f * (velocity[i].z + velocity[iz].z));
}

FLOWX_INLINE float3 apic_clamped_cell_velocity(FLOWX_CONSTANT Params &P,
                                               FLOWX_CONST_DEVICE float4 *velocity,
                                               int3 cell)
{
  return apic_cell_velocity(P, velocity,
                            clamp(cell, int3(0), params_cells(P) - 1));
}

FLOWX_INLINE float4 apic_clamped_vorticity(FLOWX_CONSTANT Params &P,
                                           FLOWX_CONST_DEVICE float4 *vorticity,
                                           int3 cell)
{
  int3 c = clamp(cell, int3(0), params_cells(P) - 1);
  return vorticity[cell_index(P, c)];
}

/* --- PCG pressure solve ---------------------------------------------------
 *
 * Slots of the BUF_PCG_SCALARS array. apic_pcg_reduce's lane 0 is the only
 * writer; the vector passes read them, which serial dispatch makes safe. The
 * loop is recorded at a fixed iteration count and a converged solve turns the
 * remaining passes into no-ops through PCG_CONVERGED, so nothing has to come
 * back to the host mid-frame.
 */
#define PCG_RZ 0         /* r.z of the current residual */
#define PCG_ALPHA 1      /* step length for this iteration */
#define PCG_BETA 2       /* direction update for this iteration */
#define PCG_THRESHOLD 3  /* tolerance^2 * r0.r0: stop once r.r is at or below */
#define PCG_CONVERGED 4  /* nonzero once the solve has stopped */
#define PCG_ITERATIONS 5 /* iterations that actually updated the pressure */
#define PCG_GUARD 6      /* breakdown bound on p.Ap */
#define PCG_RR 7         /* r.r of the latest residual (initial one after init) */
#define PCG_SCALAR_COUNT 8

/* Stages of apic_pcg_reduce, set per dispatch through Params.pcg_stage. */
#define PCG_STAGE_INIT 0
#define PCG_STAGE_CURVATURE 1
#define PCG_STAGE_RESIDUAL 2

/* Relative to the initial r.z; solver/engine/apic_cpu.py's _PCG_BREAKDOWN. */
#define PCG_BREAKDOWN 1e-12f

/* The pressure matrix's diagonal: non-solid neighbours, walls counting as
 * solid. The same count apic_pressure accumulates inline for Jacobi. */
FLOWX_INLINE float apic_pressure_diagonal(FLOWX_CONSTANT Params &P,
                                          FLOWX_CONST_DEVICE float4 *cell_data,
                                          int3 c)
{
  const int3 offsets[6] = {int3(-1, 0, 0), int3(1, 0, 0),
                           int3(0, -1, 0), int3(0, 1, 0),
                           int3(0, 0, -1), int3(0, 0, 1)};
  float diagonal = 0.0f;
  for (int n = 0; n < 6; ++n) {
    if (apic_cell_type(P, cell_data, c + offsets[n]) >= 0.0f) {
      diagonal += 1.0f;
    }
  }
  return diagonal;
}

/* Sum `value` over the threadgroup in a fixed tree order; lane 0 gets it.
 *
 * Every thread of the group must call this - the barriers are collective -
 * so callers compute a zero for out-of-range threads rather than returning
 * early, and the host dispatches whole groups. The pairing (lane i takes
 * lane i + stride, stride halving) depends only on lane indices, never on
 * which thread happens to run first, which is what makes the sum
 * reproducible. Float atomics would not be: their accumulation order is the
 * hardware's, and a different order is a different rounding - which would
 * silently break cached scrubbing, for the same reason P2G is a gather. */
FLOWX_INLINE float2 flowx_group_sum(FLOWX_SHARED float2 *shared, uint lid, float2 value)
{
  shared[lid] = value;
  FLOWX_BARRIER();
  for (uint stride = FLOWX_GROUP_SIZE / 2; stride > 0; stride >>= 1) {
    if (lid < stride) {
      shared[lid] += shared[lid + stride];
    }
    FLOWX_BARRIER();
  }
  return shared[0];
}

FLOWX_INLINE bool apic_pcg_active(FLOWX_CONST_DEVICE float *scalars)
{
  return scalars[PCG_CONVERGED] == 0.0f;
}

#endif /* FLOWX_APIC_COMMON_H */
