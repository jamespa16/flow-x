"""Package the extension as an installable .zip for a release.

The zip layout Blender expects: a single top-level directory named after the
extension id (from blender_manifest.toml), containing the manifest and the
rest of the add-on. Plain stdlib only, so this runs in any CPython 3.11+
without Blender:

    python3 scripts/package.py [source_dir] [out_dir]

Defaults: source_dir = the repo root, out_dir = <repo>/dist. VCS metadata
and dev junk are excluded; the demo .blend files are included on purpose -
they ship with the release.
"""

import os
import re
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EXCLUDE_DIRS = {".git", ".github", ".qwen", "__pycache__", ".ruff_cache", ".venv", "dist"}
EXCLUDE_FILES = {".DS_Store"}
EXCLUDE_SUFFIXES = {".pyc", ".blend1", ".blend2"}


def _read_manifest(manifest_path: Path):
    """(id, version) from blender_manifest.toml.

    tomllib on Python 3.11+, a line-parse fallback below that - the manifest
    values are simple strings, and packaging should work on any runner Python.
    """
    try:
        import tomllib
    except ModuleNotFoundError:
        text = manifest_path.read_text()

        def grab(key: str) -> str:
            match = re.search(rf'^{key}\s*=\s*"([^"]+)"', text, re.M)
            if match is None:
                raise ValueError(f"{key} not found in {manifest_path}")
            return match.group(1)

        return grab("id"), grab("version")

    with open(manifest_path, "rb") as f:
        manifest = tomllib.load(f)
    return manifest["id"], manifest["version"]


def build_zip(source_dir: Path, out_dir: Path) -> Path:
    ext_id, version = _read_manifest(source_dir / "blender_manifest.toml")

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{ext_id}-{version}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(source_dir):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for name in sorted(files):
                if name in EXCLUDE_FILES or Path(name).suffix in EXCLUDE_SUFFIXES:
                    continue
                full = Path(root) / name
                zf.write(full, os.path.join(ext_id, full.relative_to(source_dir).as_posix()))
    return zip_path


def main():
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO_ROOT / "dist"
    zip_path = build_zip(source, out)
    print(f"Wrote {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
