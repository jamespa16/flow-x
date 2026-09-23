"""The PBF/APIC solvers and the compute plumbing under them.

* `backend` - the device abstraction: buffers, compiled kernels, a command
  queue. Generic, and free of Blender imports.
* `engine` - simulation methods that sit on a backend and advance the fluid.
* `viz` - the debug point-cloud overlay.
* `sph` - shared configuration, timeline and Blender operators.
* `marching_cubes` - Phase 6's iso-surface extraction, pure Python and free of
  Blender imports so it can be tested outside Blender.
* `surface` - Phase 6's density splat, extraction and FluidSurface mesh.
* `cache` - the disk cache: per-frame particle-state snapshots that let the
  timeline be scrubbed back without re-simulating.
* `whitewater` - secondary spray/foam/bubble particles driven from either
  method's positions, velocities and particle hash.

The Phase 3 gravity-only compute round-trip that used to live here as
`gpu_test` is now scripts/test_backend.py, which answers the same question -
"can we drive the GPU on this machine at all" - without needing Blender.
"""

import bpy

from . import backend, cache, engine, sph, surface, viz, whitewater
from .cache import FLOWX_OT_cache_clear
from .sph import FLOWX_OT_sph_bake, FLOWX_OT_sph_reset, FLOWX_OT_sph_toggle

__all__ = [
    "FLOWX_OT_cache_clear",
    "FLOWX_OT_sph_bake",
    "FLOWX_OT_sph_reset",
    "FLOWX_OT_sph_toggle",
    "backend",
    "cache",
    "engine",
    "sph",
    "surface",
    "whitewater",
]

_classes = (
    FLOWX_OT_sph_toggle,
    FLOWX_OT_sph_reset,
    FLOWX_OT_sph_bake,
    FLOWX_OT_cache_clear,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    sph.stop()
    surface.stop()
    whitewater.stop()
    viz.disable()
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
