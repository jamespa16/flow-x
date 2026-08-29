"""Build the Metal helper dylib that the solver dispatches through.

    python3 scripts/build_native.py [--debug] [--check]

Output is bin/libflowx_metal.dylib, which solver/backend/metal.py loads with
ctypes. The build is plain clang++ - no Xcode project, no CMake - because the
whole native surface is one translation unit (native/flowx_metal.mm) and adding
a build system would cost more than it saves.

The dylib is ad-hoc code-signed. On a locally-built file that is enough for
dlopen to succeed; a dylib that arrives inside a *downloaded* zip carries a
quarantine attribute and needs real Developer ID signing plus notarization of
the whole artifact. That is a release-engineering problem, deliberately not
solved here - the add-on falls back to the CPU engine when the dylib will not
load, so an unsigned build degrades instead of breaking.

Note this needs only clang and the macOS SDK. It does *not* need Xcode's
separately-downloadable Metal toolchain, because the kernels are compiled from
source at runtime rather than shipped as a .metallib.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "native" / "flowx_metal.mm"
OUT_DIR = REPO_ROOT / "bin"
OUT = OUT_DIR / "libflowx_metal.dylib"

# 13.0 is where the Metal 3 API this uses settled; it is well below anything
# that ships Blender 5.2, so it costs nothing and keeps the artifact portable.
DEPLOYMENT_TARGET = "13.0"


def _run(cmd):
    print("$", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"command failed with {result.returncode}")


def build(debug=False):
    if sys.platform != "darwin":
        raise SystemExit("the Metal helper is macOS-only; other platforms use the CPU engine")
    if not SOURCE.exists():
        raise SystemExit(f"missing {SOURCE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "clang++",
        "-x",
        "objective-c++",
        "-std=c++17",
        "-fobjc-arc",
        # The ABI is declared extern "C" with default visibility in the header;
        # hiding everything else means the dylib exports only those entry points.
        "-fvisibility=hidden",
        "-dynamiclib",
        f"-mmacosx-version-min={DEPLOYMENT_TARGET}",
        "-Wall",
        "-Wextra",
        "-Werror",
    ]
    cmd += ["-O0", "-g"] if debug else ["-O2"]
    cmd += ["-framework", "Metal", "-framework", "Foundation"]
    cmd += ["-o", str(OUT), str(SOURCE)]
    _run(cmd)

    # Ad-hoc signature. Without it a freshly built dylib still loads, but macOS
    # increasingly wants *some* signature and the cost here is nil.
    _run(["codesign", "--force", "--sign", "-", str(OUT)])
    print(f"built {OUT} ({OUT.stat().st_size} bytes)")


def check():
    """Load the built dylib and report what it says about the device."""
    sys.path.insert(0, str(REPO_ROOT.parent))
    import ctypes

    if not OUT.exists():
        raise SystemExit(f"{OUT} not built yet")
    lib = ctypes.CDLL(str(OUT))
    lib.flowx_abi_version.restype = ctypes.c_int32
    lib.flowx_device_create.restype = ctypes.c_void_p
    lib.flowx_device_create.argtypes = [ctypes.POINTER(ctypes.c_char_p)]
    lib.flowx_device_name.restype = ctypes.c_char_p
    lib.flowx_device_name.argtypes = [ctypes.c_void_p]
    err = ctypes.c_char_p()
    device = lib.flowx_device_create(ctypes.byref(err))
    if not device:
        raise SystemExit(f"device creation failed: {err.value!r}")
    print(f"abi {lib.flowx_abi_version()}  device {lib.flowx_device_name(device).decode()}")


def main():
    args = sys.argv[1:]
    if "--check" in args:
        check()
        return
    build(debug="--debug" in args)


if __name__ == "__main__":
    main()
