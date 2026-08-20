"""Auto-reload Flow-X on file save, for local dev.

Run *inside* Blender (Scripting tab > Open this file > Run Script, or
`blender --python scripts/reload_on_save.py`). Polls the extension's
source files every second; on change, disables and re-enables the
extension so edits show up without restarting Blender.

Requires the extension to already be linked and enabled - see
scripts/dev_link.py.
"""

import os

import bpy

ADDON_MODULE = "bl_ext.user_default.flow_x"
WATCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WATCH_EXTENSIONS = (".py", ".glsl")
POLL_SECONDS = 1.0

_last_mtime = 0.0


def _source_mtime():
    latest = 0.0
    for root, dirs, files in os.walk(WATCH_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if name.endswith(WATCH_EXTENSIONS):
                latest = max(latest, os.path.getmtime(os.path.join(root, name)))
    return latest


def _reload():
    global _last_mtime
    print(f"[flow-x] change detected, reloading {ADDON_MODULE}")
    try:
        bpy.ops.preferences.addon_disable(module=ADDON_MODULE)
        bpy.ops.preferences.addon_enable(module=ADDON_MODULE)
    except Exception as exc:
        print(f"[flow-x] reload failed: {exc}")
    _last_mtime = _source_mtime()


def _poll():
    global _last_mtime
    mtime = _source_mtime()
    if mtime > _last_mtime:
        _reload()
    return POLL_SECONDS


def register():
    global _last_mtime
    _last_mtime = _source_mtime()
    bpy.app.timers.register(_poll, persistent=True)
    print("[flow-x] reload-on-save watcher active")


def unregister():
    if bpy.app.timers.is_registered(_poll):
        bpy.app.timers.unregister(_poll)


if __name__ == "__main__":
    register()
