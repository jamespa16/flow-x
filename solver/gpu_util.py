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
* The OpenGL backend dead-code-eliminates push-constant uniforms and images
  that a compiled pass never reads, while Metal keeps every declared slot.
  A shared block therefore cannot be assumed present in every pass: bind it
  through the per-shader ``missing`` sets in bind_push_constants()/
  bind_image(), which remember what this particular compile has no slot for.

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


def read_scalar_texture(texture, count):
    """Read a single-channel 2D texture back as a flat list of `count` floats.

    Cheaper than read_texture() for the large grids the surface pass produces:
    it extends row by row instead of indexing every item, and a single-channel
    format is a quarter of the read-back an RGBA32F one would be. Blender hands
    back an R32F texel as a bare float on some backends and as a 1-element
    sequence on others, so both are unwrapped.
    """
    values = []
    for row in texture.read().to_list():
        values.extend(row)
        if len(values) >= count:
            break
    del values[count:]
    if values and isinstance(values[0], (list, tuple)):
        values = [value[0] for value in values]
    return values


def _remember_missing(missing, name, exc):
    """Record a resource the compiled shader has no slot for, or re-raise.

    Only a "not found" lookup is remembered: that is the OpenGL backend
    having eliminated a slot the pass never reads, and a constant the pass
    does read can never be looked up as missing, so the skip is a no-op.
    Anything else is a real error and must still surface.
    """
    if "not found" in str(exc):
        missing.add(name)
    else:
        raise


def bind_push_constants(shader, values, missing):
    """Bind a {name: 4-tuple} push-constant block onto an already-bound shader.

    `missing` is a per-compiled-shader set of names the backend has already
    reported as absent; it is updated as bindings are attempted (see
    _remember_missing), so only the first dispatch pays each lookup. Whether
    a slot is an int or a float vector follows the `i_`/`f_` naming the GLSL
    block uses (see shaders/sph_common.glsl); passing a float tuple that
    happens to hold whole numbers therefore still binds as floats.
    """
    for name, value in values.items():
        if name in missing:
            continue
        try:
            if name.startswith("i_"):
                shader.uniform_int(name, value)
            elif name.startswith("f_"):
                shader.uniform_float(name, value)
            else:
                raise ValueError(f"push constant {name!r} must be named i_* or f_*")
        except Exception as exc:
            _remember_missing(missing, name, exc)


def bind_image(shader, name, texture, missing):
    """Bind an image, remembering names the compiled pass has no slot for."""
    if name in missing:
        return
    try:
        shader.image(name, texture)
    except Exception as exc:
        _remember_missing(missing, name, exc)


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
