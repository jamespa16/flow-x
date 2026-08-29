"""Import Flow-X's Blender-free modules outside Blender.

`solver/__init__.py` imports bpy, so importing `solver.backend` the ordinary
way drags Blender in. But solver/backend and solver/engine (except where an
engine needs a device) import only stdlib and numpy, on purpose - that is what
lets scripts/test_backend.py and scripts/test_cpu_engine.py run in CI on a
machine with neither Blender nor a GPU.

This registers `solver` as a namespace package pointing at the real directory
*without* executing its `__init__.py`, so relative imports inside it (`from
..backend import select`) resolve normally and the submodules can be imported
with plain importlib.

    from flowx_standalone import load
    backend = load("solver.backend")
"""

import importlib
import importlib.machinery
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "flowx_solver"


def _bootstrap():
    """Register `flowx_solver` as solver/, with its __init__ deliberately unrun."""
    if PACKAGE in sys.modules:
        return
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(ROOT / "solver")]
    package.__package__ = PACKAGE
    # A spec with submodule_search_locations is what makes importlib treat this
    # as a package it can import submodules of.
    package.__spec__ = importlib.machinery.ModuleSpec(
        PACKAGE, loader=None, is_package=True, origin=str(ROOT / "solver")
    )
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules[PACKAGE] = package


def load(dotted):
    """Import a module by its in-repo path, e.g. "solver.engine.cpu_engine"."""
    _bootstrap()
    if not dotted.startswith("solver."):
        raise ValueError(f"expected a solver.* module, got {dotted!r}")
    return importlib.import_module(PACKAGE + dotted[len("solver") :])
