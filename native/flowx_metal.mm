/* Objective-C++ implementation of Flow-X's Metal helper. See flowx_metal.h for
 * the contract and for why this exists at all.
 *
 * Built by scripts/build_native.py into bin/libflowx_metal.dylib.
 *
 * Two things are load-bearing and easy to undo by accident:
 *
 * * Every entry point is wrapped in @try/@catch. An Objective-C exception
 *   unwinding into CPython's C frames is undefined behaviour and takes Blender
 *   down with it, so nothing is allowed to escape - failures come back as a
 *   NULL/non-zero plus a message.
 * * The handles are C++ structs holding strong `id<MTL...>` members. Under ARC
 *   (-fobjc-arc) that keeps the Metal objects alive exactly as long as the
 *   handle, so the Python side's lifetime management is ordinary refcounting
 *   with no autorelease-pool subtleties leaking across the boundary.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "flowx_metal.h"

#include <cstdlib>
#include <cstring>
#include <string>

namespace {

void set_error(char **err, NSString *message) {
  if (err == nullptr) {
    return;
  }
  const char *utf8 = [message UTF8String];
  *err = strdup(utf8 != nullptr ? utf8 : "unknown Metal error");
}

void set_error(char **err, NSError *error, NSString *context) {
  set_error(err, [NSString stringWithFormat:@"%@: %@", context,
                                            error != nil ? [error localizedDescription]
                                                         : @"no error detail"]);
}

}  // namespace

struct FlowxDevice {
  id<MTLDevice> device;
  /* Held so flowx_device_name() can hand out a borrowed pointer with the
   * device's lifetime rather than a buffer the caller has to free. */
  std::string name;
};

struct FlowxLibrary {
  id<MTLDevice> device;
  id<MTLLibrary> library;
};

struct FlowxKernel {
  id<MTLComputePipelineState> pipeline;
};

struct FlowxBuffer {
  id<MTLBuffer> buffer;
};

struct FlowxQueue {
  id<MTLCommandQueue> queue;
};

extern "C" {

FLOWX_API int32_t flowx_abi_version(void) { return FLOWX_ABI_VERSION; }

FLOWX_API void flowx_string_free(char *s) { free(s); }

/* --- device ------------------------------------------------------------- */

FLOWX_API FlowxDevice *flowx_device_create(char **err) {
  @try {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      set_error(err, @"no Metal device available");
      return nullptr;
    }
    auto *handle = new FlowxDevice();
    handle->device = device;
    const char *name = [[device name] UTF8String];
    handle->name = name != nullptr ? name : "unknown";
    return handle;
  } @catch (NSException *e) {
    set_error(err, [e reason] ?: @"exception creating device");
    return nullptr;
  }
}

FLOWX_API void flowx_device_destroy(FlowxDevice *device) { delete device; }

FLOWX_API const char *flowx_device_name(FlowxDevice *device) {
  return device != nullptr ? device->name.c_str() : "";
}

FLOWX_API uint64_t flowx_device_max_buffer_length(FlowxDevice *device) {
  return device != nullptr ? (uint64_t)[device->device maxBufferLength] : 0;
}

FLOWX_API int32_t flowx_device_has_unified_memory(FlowxDevice *device) {
  return device != nullptr && [device->device hasUnifiedMemory] ? 1 : 0;
}

/* --- programs ----------------------------------------------------------- */

FLOWX_API FlowxLibrary *flowx_library_new(FlowxDevice *device, const char *source, char **err) {
  if (device == nullptr || source == nullptr) {
    set_error(err, @"library: null device or source");
    return nullptr;
  }
  @try {
    NSError *error = nil;
    MTLCompileOptions *options = [MTLCompileOptions new];
    /* Default (fast) math is what a GLSL compute pass got through Blender too,
     * so keeping it here is the closer match to the behaviour being ported. */
    id<MTLLibrary> library =
        [device->device newLibraryWithSource:[NSString stringWithUTF8String:source]
                                     options:options
                                       error:&error];
    if (library == nil) {
      set_error(err, error, @"MSL compile failed");
      return nullptr;
    }
    auto *handle = new FlowxLibrary();
    handle->device = device->device;
    handle->library = library;
    return handle;
  } @catch (NSException *e) {
    set_error(err, [e reason] ?: @"exception compiling library");
    return nullptr;
  }
}

FLOWX_API void flowx_library_destroy(FlowxLibrary *library) { delete library; }

FLOWX_API FlowxKernel *flowx_kernel_new(FlowxLibrary *library, const char *name, char **err) {
  if (library == nullptr || name == nullptr) {
    set_error(err, @"kernel: null library or name");
    return nullptr;
  }
  @try {
    NSString *entry = [NSString stringWithUTF8String:name];
    id<MTLFunction> function = [library->library newFunctionWithName:entry];
    if (function == nil) {
      set_error(err, [NSString stringWithFormat:@"no kernel named '%@' in library", entry]);
      return nullptr;
    }
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline =
        [library->device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
      set_error(err, error, [NSString stringWithFormat:@"pipeline for '%@'", entry]);
      return nullptr;
    }
    auto *handle = new FlowxKernel();
    handle->pipeline = pipeline;
    return handle;
  } @catch (NSException *e) {
    set_error(err, [e reason] ?: @"exception creating kernel");
    return nullptr;
  }
}

FLOWX_API void flowx_kernel_destroy(FlowxKernel *kernel) { delete kernel; }

FLOWX_API uint32_t flowx_kernel_max_threads_per_group(FlowxKernel *kernel) {
  return kernel != nullptr ? (uint32_t)[kernel->pipeline maxTotalThreadsPerThreadgroup] : 0;
}

FLOWX_API uint32_t flowx_kernel_execution_width(FlowxKernel *kernel) {
  return kernel != nullptr ? (uint32_t)[kernel->pipeline threadExecutionWidth] : 0;
}

/* --- buffers ------------------------------------------------------------ */

FLOWX_API FlowxBuffer *flowx_buffer_new(FlowxDevice *device, uint64_t nbytes, const void *data,
                                        char **err) {
  if (device == nullptr) {
    set_error(err, @"buffer: null device");
    return nullptr;
  }
  if (nbytes == 0) {
    set_error(err, @"buffer: zero length");
    return nullptr;
  }
  @try {
    /* Shared, not private: the solver reads particle state back every frame,
     * and on unified memory shared storage makes that a pointer rather than a
     * blit. newBufferWithLength: already returns zero-filled memory, which the
     * solver relies on for the buffers a pass accumulates into. */
    const MTLResourceOptions options = MTLResourceStorageModeShared;
    id<MTLBuffer> buffer =
        data != nullptr
            ? [device->device newBufferWithBytes:data length:(NSUInteger)nbytes options:options]
            : [device->device newBufferWithLength:(NSUInteger)nbytes options:options];
    if (buffer == nil) {
      set_error(err, [NSString stringWithFormat:@"allocation of %llu bytes failed", nbytes]);
      return nullptr;
    }
    auto *handle = new FlowxBuffer();
    handle->buffer = buffer;
    return handle;
  } @catch (NSException *e) {
    set_error(err, [e reason] ?: @"exception allocating buffer");
    return nullptr;
  }
}

FLOWX_API void flowx_buffer_destroy(FlowxBuffer *buffer) { delete buffer; }

FLOWX_API void *flowx_buffer_contents(FlowxBuffer *buffer) {
  return buffer != nullptr ? [buffer->buffer contents] : nullptr;
}

FLOWX_API uint64_t flowx_buffer_length(FlowxBuffer *buffer) {
  return buffer != nullptr ? (uint64_t)[buffer->buffer length] : 0;
}

/* --- dispatch ----------------------------------------------------------- */

FLOWX_API FlowxQueue *flowx_queue_new(FlowxDevice *device, char **err) {
  if (device == nullptr) {
    set_error(err, @"queue: null device");
    return nullptr;
  }
  @try {
    id<MTLCommandQueue> queue = [device->device newCommandQueue];
    if (queue == nil) {
      set_error(err, @"newCommandQueue returned nil");
      return nullptr;
    }
    auto *handle = new FlowxQueue();
    handle->queue = queue;
    return handle;
  } @catch (NSException *e) {
    set_error(err, [e reason] ?: @"exception creating queue");
    return nullptr;
  }
}

FLOWX_API void flowx_queue_destroy(FlowxQueue *queue) { delete queue; }

FLOWX_API int32_t flowx_submit(FlowxQueue *queue, const FlowxDispatch *records, uint32_t count,
                               int32_t wait, char **err) {
  if (queue == nullptr || (records == nullptr && count > 0)) {
    set_error(err, @"submit: null queue or records");
    return 1;
  }
  if (count == 0) {
    return 0;
  }
  @try {
    /* One pool around the whole submission: command buffers and encoders are
     * autoreleased, and without this they would accumulate for the life of the
     * calling thread - which, driven from a frame handler, is forever. */
    @autoreleasepool {
      id<MTLCommandBuffer> commands = [queue->queue commandBuffer];
      /* Serial dispatch (the default) is what gives each record visibility of
       * the previous record's writes. The solver's pass chain depends on it. */
      id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];

      for (uint32_t r = 0; r < count; ++r) {
        const FlowxDispatch &d = records[r];
        if (d.kernel == nullptr) {
          [encoder endEncoding];
          set_error(err, [NSString stringWithFormat:@"submit: record %u has no kernel", r]);
          return 2;
        }
        [encoder setComputePipelineState:d.kernel->pipeline];
        for (uint32_t i = 0; i < d.nbuffers; ++i) {
          FlowxBuffer *b = d.buffers[i];
          if (b != nullptr) {
            [encoder setBuffer:b->buffer offset:0 atIndex:i];
          }
        }
        if (d.constants_size > 0 && d.constants != nullptr) {
          [encoder setBytes:d.constants
                     length:(NSUInteger)d.constants_size
                    atIndex:(NSUInteger)d.constants_index];
        }
        if (d.threads == 0) {
          continue;
        }
        NSUInteger group = d.threads_per_group;
        const NSUInteger cap = [d.kernel->pipeline maxTotalThreadsPerThreadgroup];
        if (group == 0) {
          group = 1;
        }
        if (group > cap) {
          group = cap;
        }
        /* dispatchThreads: (not dispatchThreadgroups:) so a count that is not a
         * multiple of the group size launches exactly `threads` invocations.
         * Apple silicon supports non-uniform threadgroups; the kernels keep
         * their own bounds guards anyway, both defensively and because a CUDA
         * port will need them. */
        [encoder dispatchThreads:MTLSizeMake(d.threads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
      }

      [encoder endEncoding];
      [commands commit];
      if (wait != 0) {
        [commands waitUntilCompleted];
        NSError *error = [commands error];
        if (error != nil) {
          set_error(err, error, @"command buffer failed");
          return 3;
        }
      }
    }
    return 0;
  } @catch (NSException *e) {
    set_error(err, [e reason] ?: @"exception during submit");
    return 4;
  }
}

}  // extern "C"
