"""The PBF solver and the compute plumbing under it.

* `backend` - the device abstraction: buffers, compiled kernels, a command
  queue. Generic, and free of Blender imports.
* `engine` - the simulation engines that sit on a backend and know what SPH is.
* `viz` - the debug point-cloud overlay.
* `sph` - Phase 4's Position Based Fluids solver core, plus Phase 7's timeline
  handling.
* `marching_cubes` - Phase 6's iso-surface extraction, pure Python and free of
  Blender imports so it can be tested outside Blender.
* `surface` - Phase 6's density splat, extraction and FluidSurface mesh.
* `cache` - the disk cache: per-frame particle-state snapshots that let the
  timeline be scrubbed back without re-simulating.
* `whitewater` - Phase 8's secondary spray/foam/bubble particles, driven off
  the PBF state each frame into a Whitewater point-cloud child object.

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
