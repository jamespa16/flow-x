"""Flow-X: GPU-simulated PBF/APIC fluids with a live surface mesh, for Blender."""

from . import collision, domain, solver, ui

_modules = (
    domain,
    collision,
    solver,
    ui,
)


def register():
    for module in _modules:
        module.register()


def unregister():
    for module in reversed(_modules):
        module.unregister()
