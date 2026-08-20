"""Shared GPU compute plumbing for the solver passes.

Two Blender GPU-API constraints shape everything here, both established by
probing the module directly rather than from the docs:

* ``GPUShaderCreateInfo.image()`` must be given explicit ``qualifiers``. Left
  at the default, the Metal backend generates MSL that fails to compile inside
  Blender's own headers, with errors that point at those headers rather than
  at our source.
* 1D textures cannot be read back - ``GPUTexture.read()`` reports a
  zero-length first dimension for them. All particle/grid state therefore
  lives in 2D textures, addressed as a flat array wrapped at ``TEXTURE_WIDTH``.

There is no read-write storage buffer in the Python API at all, so images are
the only mutable GPU state available.
"""

import math
from pathlib import Path

import gpu

# Flat arrays are wrapped into 2D textures at this width. 256 keeps the height
# well inside any sane GL_MAX_TEXTURE_SIZE for the array lengths we use.
TEXTURE_WIDTH = 256

_READ_WRITE = {"READ", "WRITE"}
_SHADER_DIR = Path(__file__).resolve().parent.parent / "shaders"


def shader_source(name):
    """Read a GLSL file from the add-on's shaders/ directory."""
    return (_SHADER_DIR / f"{name}.glsl").read_text()


def texture_size(count):
    """(width, height) of the 2D texture backing a flat array of `count` items."""
    return TEXTURE_WIDTH, max(1, math.ceil(count / TEXTURE_WIDTH))


def make_texture(count, channels=4, values=None, fmt="RGBA32F"):
    """Allocate a 2D texture holding a flat array of `count` items.

    `values` is a flat sequence of `count * channels` floats; padding out to
    the texture's full width is handled here. Passing None leaves the contents
    undefined, which is fine for buffers a compute pass fully overwrites.
    """
    width, height = texture_size(count)
    if values is None:
        return gpu.types.GPUTexture((width, height), format=fmt)

    padded = list(values) + [0.0] * (width * height * channels - len(values))
    buffer = gpu.types.Buffer("FLOAT", [len(padded)], padded)
    return gpu.types.GPUTexture((width, height), format=fmt, data=buffer)


def read_texture(texture, count, channels=4):
    """Read a 2D texture back as a flat list of `count` per-item tuples."""
    rows = texture.read().to_list()
    out = []
    for i in range(count):
        texel = rows[i // TEXTURE_WIDTH][i % TEXTURE_WIDTH]
        out.append(tuple(texel[:channels]) if channels > 1 else texel[0])
    return out


def build_compute_shader(sources, images, push_constants, local_size):
    """Compile a compute shader from GLSL sources plus its resource bindings.

    `sources` are concatenated in order (a shared prelude followed by the
    pass's own body), `images` are (format, type, name) triples bound to
    successive slots, and `push_constants` are (type, name) pairs.
    """
    info = gpu.types.GPUShaderCreateInfo()
    for slot, (fmt, image_type, name) in enumerate(images):
        info.image(slot, fmt, image_type, name, qualifiers=_READ_WRITE)
    for const_type, name in push_constants:
        info.push_constant(const_type, name)
    info.compute_source("\n".join(sources))
    info.local_group_size(local_size)
    return gpu.shader.create_from_info(info)


def dispatch_1d(shader, count, local_size):
    """Dispatch enough workgroups to cover `count` invocations along x."""
    gpu.compute.dispatch(shader, max(1, math.ceil(count / local_size)), 1, 1)
