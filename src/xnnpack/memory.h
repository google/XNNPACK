// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <string.h>

#include <xnnpack.h>
#include <xnnpack/common.h>


#ifdef __cplusplus
extern "C" {
#endif

// Default size for buffer to hold all generated microkernels, 16KB.
#define XNN_DEFAULT_CODE_BUFFER_SIZE    16384
// Default size required for generating one microkernel, 4KB.
#define XNN_DEFAULT_MICROKERNEL_SIZE    4096
// Default size for buffer to hold repacked weights, 1MB.
#define XNN_DEFAULT_WEIGHTS_BUFFER_SIZE 1048576

#define XNN_INVALID_FUNCTION_INDEX -1

struct xnn_code_buffer {
  // Pointer to allocated, externally managed memory.
  void* start;
  // Actual size of instructions (bytes). It is only safe to access code within
  // this size.
  size_t size;
  // Maximum capacity of the buffer pointer to by `code`. This is the size of
  // the currently mapped memory.
  size_t capacity;
};

// Allocates a code region and associates it with `buffer`.
enum xnn_status xnn_allocate_code_memory(struct xnn_code_buffer* buffer, size_t size);
// Ensure that buffer has at least min_available_size bytes free (i.e. buffer->capacity - buffer->size >= min_available_size), grows if not.
enum xnn_status xnn_reserve_code_memory(struct xnn_code_buffer* buffer, size_t min_available_size);
// Free all memory associated with `buffer`.
enum xnn_status xnn_release_code_memory(struct xnn_code_buffer* buffer);
#if XNN_PLATFORM_JIT
// Finalize buffer, users won't need to call this directly, called by Assembler.
enum xnn_status xnn_finalize_code_memory(struct xnn_code_buffer* buffer);
// Returns a pointer (casted to integer) to the first function in the code block
// between `code + offset` and `code + offset_end`.
uintptr_t xnn_first_function_in_chunk_ptr(struct xnn_code_buffer* buffer, size_t offset, size_t offset_end);
// Returns a pointer (casted to integer) to the first function in the buffer.
// On WASM the buffer is presumed to contain a single module.
inline uintptr_t xnn_first_function_ptr(struct xnn_code_buffer* buffer) {
  return xnn_first_function_in_chunk_ptr(buffer, 0, buffer->size);
}
#endif

// Buffer to hold repacked weights.
struct xnn_weights_buffer {
  // Pointer to allocated memory for weights.
  void* start;
  // Size of weights.
  size_t size;
  // Maximum capacity of this buffer pointed to by `code`. This is the size of the allcoated memory.
  size_t capacity;
};

// Allocates a weights region and associates it with `buffer`.
enum xnn_status xnn_allocate_weights_memory(struct xnn_weights_buffer* buffer, size_t size);
// Free all memory associated with `buffer`.
enum xnn_status xnn_release_weights_memory(struct xnn_weights_buffer* buffer);
// Ensure that buffer has at least min_available_size bytes free (i.e. buffer->capacity - buffer->size >= min_available_size), grows if not.
enum xnn_status xnn_reserve_weights_memory(struct xnn_weights_buffer* buffer, size_t min_available_size);
// Releases unused memory in `buffer`, and sets used memory to read-only. The address of allocated memory (`buffer->start`)
// is fixed after this call. This should only be called after all the weights have been written.
enum xnn_status xnn_finalize_weights_memory(struct xnn_weights_buffer* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif
