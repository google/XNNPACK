// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <string.h>

#include "xnnpack.h"
#include "xnnpack/common.h"


#ifdef __cplusplus
extern "C" {
#endif

// Default size for buffer to hold repacked weights, 1MB.
#define XNN_DEFAULT_WEIGHTS_BUFFER_SIZE 1048576

#define XNN_INVALID_FUNCTION_INDEX -1

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
