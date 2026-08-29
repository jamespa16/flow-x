"""Loading and assembling the kernel sources in kernels/.

The device compiles one library holding every pass, rather than one library per
pass. Metal has no include paths when compiling from source, so the files are
concatenated here - prelude first, then the shared helpers, then the passes -
and any local `#include "..."` line is dropped on the way in. The headers keep
their include lines so that they stay valid C++ for an editor or a linter;
`<metal_stdlib>` and friends are left alone, since those the compiler does
resolve.

One library also means one compile: the passes share a prelude, so compiling
them separately would re-parse it a dozen times for nothing.
"""

import re
from pathlib import Path

KERNEL_DIR = Path(__file__).resolve().parent.parent.parent / "kernels"

# Order matters: definitions before use.
HEADERS = ("flowx_prelude.h", "sph_common.h")

# The SPH substep chain, in dispatch order. Mirrors what _PASSES used to be.
SPH_PASSES = (
    "sph_normal",
    "sph_predict",
    "sph_grid_key",
    "sph_sort",
    "sph_cell_clear",
    "sph_cell_range",
    "sph_lambda",
    "sph_delta",
    "sph_apply_delta",
    "sph_velocity",
    "sph_xsph",
    "sph_finalize",
)

SURFACE_PASSES = ("surface_splat",)

WHITEWATER_PASSES = (
    "whitewater_potential",
    "whitewater_sort",
    "whitewater_spawn",
    "whitewater_advect",
)

_LOCAL_INCLUDE = re.compile(r'^\s*#\s*include\s*"[^"]+"\s*$', re.MULTILINE)


def read(name, suffix=".metal"):
    return (KERNEL_DIR / f"{name}{suffix}").read_text()


def library_source(passes=None):
    """The full source for one device library.

    `passes` defaults to every pass the solver has. Narrowing it is for tests
    that want a faster compile, not for the solver - a partial library would
    just mean compiling again later.
    """
    if passes is None:
        passes = SPH_PASSES + SURFACE_PASSES + WHITEWATER_PASSES

    parts = [read(header, suffix="") for header in HEADERS]
    parts += [read(name) for name in passes]
    return _LOCAL_INCLUDE.sub("", "\n".join(parts))


def entry_points(passes=None):
    """Every kernel name in the library, for eager compilation at startup.

    Compiling every pipeline up front rather than on first use means a bad
    kernel is reported when the solver starts, not three frames into playback.
    """
    if passes is None:
        passes = SPH_PASSES + SURFACE_PASSES + WHITEWATER_PASSES
    return tuple(passes)
