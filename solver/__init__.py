"""GPU compute plumbing and the WCSPH solver (Phases 3-6).

* `gpu_util` - shared shader/texture helpers and the GPU-API constraints they
  work around.
* `viz` - the debug point-cloud overlay, shared by the run modes below.
* `gpu_test` - Phase 3's compute round-trip smoke test (gravity only).
* `sph` - Phase 4's WCSPH solver core.

The two run modes are mutually exclusive; starting either stops the other.
"""

import bpy

from . import gpu_test, sph, viz
from .gpu_test import PARTICLE_COUNT, FLOWX_OT_solver_gpu_test_toggle
from .sph import FLOWX_OT_sph_toggle

__all__ = [
    "FLOWX_OT_solver_gpu_test_toggle",
    "FLOWX_OT_sph_toggle",
    "PARTICLE_COUNT",
    "gpu_test",
    "sph",
]

_classes = (FLOWX_OT_solver_gpu_test_toggle, FLOWX_OT_sph_toggle)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    gpu_test.stop()
    sph.stop()
    viz.disable()
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
