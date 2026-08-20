#!/usr/bin/env python3
"""Symlink this repo into Blender's extensions directory, for local dev.

Usage:
    python3 scripts/dev_link.py [--blender-version 4.2]

Creates a symlink named after the extension id (see blender_manifest.toml)
inside Blender's "user_default" local extensions repository, pointing back
at this checkout. Blender then loads the extension straight from the
working tree - edit and (re)enable, no packaging/reinstall step.

Honors the BLENDER_USER_RESOURCES environment variable, matching Blender's
own resource-path override, so CI can point this at a throwaway config dir
instead of a real user profile.
"""

import argparse
import os
import platform
import sys

EXTENSION_ID = "flow_x"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _user_extensions_dir(blender_version: str) -> str:
    # BLENDER_USER_RESOURCES overrides the whole per-version resources root
    # (i.e. it already stands in for ".../Blender/<version>"), so don't
    # append the version segment again when it's set.
    override = os.environ.get("BLENDER_USER_RESOURCES")
    if override:
        return os.path.join(override, "extensions", "user_default")

    system = platform.system()
    if system == "Darwin":
        base = os.path.expanduser("~/Library/Application Support/Blender")
    elif system == "Windows":
        base = os.path.join(os.environ["APPDATA"], "Blender Foundation", "Blender")
    elif system == "Linux":
        base = os.path.expanduser("~/.config/blender")
    else:
        raise RuntimeError(f"Unsupported platform: {system}")
    return os.path.join(base, blender_version, "extensions", "user_default")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blender-version",
        default="4.2",
        help="Blender resource-version directory to link into, e.g. 4.2 or 5.2 "
        "(match whatever `blender --version` reports as the major.minor)",
    )
    args = parser.parse_args()

    target_dir = _user_extensions_dir(args.blender_version)
    os.makedirs(target_dir, exist_ok=True)
    link_path = os.path.join(target_dir, EXTENSION_ID)

    if os.path.islink(link_path):
        if os.path.realpath(link_path) == os.path.realpath(REPO_ROOT):
            print(f"Already linked: {link_path} -> {REPO_ROOT}")
            return
        os.remove(link_path)
    elif os.path.exists(link_path):
        print(f"Refusing to overwrite non-symlink at {link_path}", file=sys.stderr)
        sys.exit(1)

    os.symlink(REPO_ROOT, link_path, target_is_directory=True)
    print(f"Linked {link_path} -> {REPO_ROOT}")
    print("Restart Blender (or Preferences > Get Extensions > Refresh Local) to pick it up.")


if __name__ == "__main__":
    main()
