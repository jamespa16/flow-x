# Flow-X APIC implementation

Status: implemented for Flow-X 0.3.0. PBF remains the default; APIC is opt-in.

The implementation follows the [staggered MAC APIC formulation](https://www.cs.ucr.edu/~shinar/papers/2019-mac-apic.pdf). It is available through matching NumPy and Metal engines and shares Flow-X's particle seeding, deterministic spatial hash, collider occupancy, surface extraction, whitewater, timeline, and render replay.

## Per-substep algorithm

1. Advect particles with the previous projected velocity, recover collider penetration, clamp the domain, and mirror positions into `predicted`.
2. Rebuild the particle hash.
3. Deterministically gather mass and affine momentum to staggered MAC faces.
4. Normalize velocity, apply gravity and the grid CFL clamp.
5. Optionally compute curl and apply vorticity confinement.
6. Apply solid/domain no-penetration conditions.
7. Classify solid, fluid, and air pressure cells and compute divergence.
8. Run weighted Jacobi (`omega = 2/3`) with 40 iterations by default.
9. Apply the pressure gradient and solid-face conditions.
10. Gather projected velocities back to particles and reconstruct three padded affine rows.

The pressure cell dimensions equal the particle-hash dimensions. MAC arrays use a padded `(nx+1) x (ny+1) x (nz+1)` allocation with component-specific valid face regions. P2G is a face-owned gather rather than a float-atomic scatter, preserving deterministic cache replay.

## Shared state contract

Buffer slots 0–13 retain the PBF/surface/whitewater layout. APIC uses:

- 14: three padded affine `float4` rows per particle
- 15: packed MAC face mass
- 16: packed MAC face momentum
- 17: packed MAC face velocity
- 18: cell curl and magnitude
- 19: divergence, pressure ping/pong, and cell type
- 20: `Params`

PBF and APIC compile separate Metal libraries, so an APIC compilation failure cannot prevent PBF from starting. Auto selection falls back only across devices for the selected method; it never substitutes PBF for APIC.

## Cache and output

Cache v3 records the method, resolved device, extension version, state flags, particle count, and whitewater capacity. PBF frames store position/velocity; APIC adds affine rows; enabled whitewater adds the full pool and ring cursor. Earlier cache formats are intentionally recreated. The paired mesh cache remains the render replay path.

## Validation

Standalone coverage includes bounded dam-break motion, 90% pressure-divergence reduction, translation/affine transfer invariants, less than 5% rotating-block angular-momentum drift over 300 steps, collider response, surface extraction, whitewater lifecycle, exact snapshot continuation, cache-v3 round trips, and optional Metal compilation/agreement tests. `scripts/smoke_test.py` exercises both methods in Blender and `scripts/golden.py` records method and device separately.

## Deferred work

- FLIP blending and PCG pressure solving
- particle reseeding/remeshing
- collider velocity transfer and two-way coupling
- APIC viscosity and surface tension
- CUDA and multi-domain interaction
