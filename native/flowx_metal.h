/* Flow-X's Metal helper: a flat C ABI over just enough of Metal to run the
 * solver's compute passes.
 *
 * Why a helper at all: Flow-X used to dispatch through Blender's `gpu` module,
 * whose Metal backend has no image atomics, caps read-write images at 8 per
 * shader, and offers no read-write storage buffers. Those limits had started
 * dictating the solver's algorithms rather than merely annoying it. Owning a
 * MTLDevice directly removes all three.
 *
 * Why C and not Objective-C objects across the boundary: the caller is CPython
 * via ctypes, and a flat C ABI with opaque handles is the only thing ctypes can
 * bind without a compiled Python extension - which would tie the build to one
 * CPython ABI and one Blender version.
 *
 * Scope discipline: this file knows nothing about SPH. It is a generic device
 * layer, so that solver/engine/ can sit on top of it and a CUDA sibling can
 * implement the same shape later. Do not add solver logic here.
 *
 * Error convention: functions that can fail return NULL or a non-zero code and,
 * when `err` is non-NULL, write a malloc'd message into it for the caller to
 * release with flowx_string_free(). Nothing throws across the boundary - an
 * Objective-C exception escaping into CPython would take the process with it.
 */

#ifndef FLOWX_METAL_H
#define FLOWX_METAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The dylib is built with -fvisibility=hidden so that nothing but the ABI
 * below is exported; each entry point therefore has to opt back in explicitly.
 * extern "C" alone only fixes the mangling, not the visibility. */
#define FLOWX_API __attribute__((visibility("default")))

/* Bumped whenever the layout of FlowxDispatch or the meaning of an entry point
 * changes. solver/backend/metal.py refuses to load a dylib that disagrees,
 * which is what stops a stale build in bin/ from being read as garbage. */
#define FLOWX_ABI_VERSION 1

typedef struct FlowxDevice FlowxDevice;
typedef struct FlowxLibrary FlowxLibrary;
typedef struct FlowxKernel FlowxKernel;
typedef struct FlowxBuffer FlowxBuffer;
typedef struct FlowxQueue FlowxQueue;

FLOWX_API int32_t flowx_abi_version(void);
FLOWX_API void flowx_string_free(char *s);

/* --- device ------------------------------------------------------------- */

FLOWX_API FlowxDevice *flowx_device_create(char **err);
FLOWX_API void flowx_device_destroy(FlowxDevice *device);
/* Borrowed, NUL-terminated, valid for the device's lifetime. */
FLOWX_API const char *flowx_device_name(FlowxDevice *device);
FLOWX_API uint64_t flowx_device_max_buffer_length(FlowxDevice *device);
FLOWX_API int32_t flowx_device_has_unified_memory(FlowxDevice *device);

/* --- programs ----------------------------------------------------------- */

/* Compiles MSL at runtime. Deliberately source, not a precompiled .metallib:
 * building one needs Xcode's separately-downloaded Metal toolchain, whereas
 * newLibraryWithSource: goes through the Metal framework and is always there. */
FLOWX_API FlowxLibrary *flowx_library_new(FlowxDevice *device, const char *source,
                                          char **err);
FLOWX_API void flowx_library_destroy(FlowxLibrary *library);

FLOWX_API FlowxKernel *flowx_kernel_new(FlowxLibrary *library, const char *name, char **err);
FLOWX_API void flowx_kernel_destroy(FlowxKernel *kernel);
FLOWX_API uint32_t flowx_kernel_max_threads_per_group(FlowxKernel *kernel);
FLOWX_API uint32_t flowx_kernel_execution_width(FlowxKernel *kernel);

/* --- buffers ------------------------------------------------------------ */

/* Shared storage: on Apple silicon the CPU and GPU address the same physical
 * memory, so flowx_buffer_contents() is a real pointer into the live buffer and
 * read-back is a memoryview, not a copy. `data` may be NULL for zero-filled. */
FLOWX_API FlowxBuffer *flowx_buffer_new(FlowxDevice *device, uint64_t nbytes, const void *data,
                                        char **err);
FLOWX_API void flowx_buffer_destroy(FlowxBuffer *buffer);
FLOWX_API void *flowx_buffer_contents(FlowxBuffer *buffer);
FLOWX_API uint64_t flowx_buffer_length(FlowxBuffer *buffer);

/* --- dispatch ----------------------------------------------------------- */

/* One recorded dispatch. `buffers` are bound at indices 0..nbuffers-1, with a
 * NULL entry meaning "leave that index unbound" - a pass that ignores a slot
 * costs nothing for it. `constants` is bound via setBytes at
 * `constants_index`, or skipped when `constants_size` is 0. */
typedef struct {
  FlowxKernel *kernel;
  FlowxBuffer *const *buffers;
  uint32_t nbuffers;
  const void *constants;
  uint32_t constants_size;
  uint32_t constants_index;
  uint32_t threads;
  uint32_t threads_per_group;
} FlowxDispatch;

/* Encodes every record into a single command buffer and commits it.
 *
 * The whole point of taking an array: the solver's grid build is ~105
 * compare-exchange dispatches per substep, and crossing into Python for each
 * one was the dominant cost. Here the loop runs in C++.
 *
 * Records go into one *serial-dispatch* compute encoder, which is Metal's
 * default. That matters: serial dispatch makes each record observe the previous
 * record's writes, which is exactly the barrier semantics the solver's pass
 * chain assumes. Do not switch to MTLDispatchTypeConcurrent.
 *
 * Returns 0 on success. `wait` blocks until the GPU is done, which is required
 * before reading a buffer back. */
FLOWX_API int32_t flowx_submit(FlowxQueue *queue, const FlowxDispatch *records, uint32_t count,
                               int32_t wait, char **err);

FLOWX_API FlowxQueue *flowx_queue_new(FlowxDevice *device, char **err);
FLOWX_API void flowx_queue_destroy(FlowxQueue *queue);

#ifdef __cplusplus
}
#endif

#endif /* FLOWX_METAL_H */
