// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <limits.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
  #include <malloc.h>
#elif !defined(__GNUC__)
  #include <alloca.h>
#endif

#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/params.h"

#ifdef __cplusplus
extern "C" {
#endif

XNN_INTERNAL extern const struct xnn_allocator xnn_default_allocator;

inline static void* xnn_allocate_memory(size_t memory_size) {
  return xnn_params.allocator.allocate(xnn_params.allocator.context, memory_size);
}

inline static void* xnn_allocate_zero_memory(size_t memory_size) {
  void* memory_pointer = xnn_params.allocator.allocate(xnn_params.allocator.context, memory_size);
  if (memory_pointer != NULL) {
    memset(memory_pointer, 0, memory_size);
  }
  return memory_pointer;
}

inline static void* xnn_reallocate_memory(void* memory_pointer, size_t memory_size) {
  return xnn_params.allocator.reallocate(xnn_params.allocator.context, memory_pointer, memory_size);
}

inline static void xnn_release_memory(void* memory_pointer) {
  xnn_params.allocator.deallocate(xnn_params.allocator.context, memory_pointer);
}

inline static void* xnn_allocate_simd_memory(size_t memory_size) {
  return xnn_params.allocator.aligned_allocate(xnn_params.allocator.context, XNN_ALLOCATION_ALIGNMENT, memory_size);
}

inline static void* xnn_allocate_zero_simd_memory(size_t memory_size) {
  void* memory_pointer = xnn_params.allocator.aligned_allocate(
    xnn_params.allocator.context, XNN_ALLOCATION_ALIGNMENT, memory_size);
  if (memory_pointer != NULL) {
    memset(memory_pointer, 0, memory_size);
  }
  return memory_pointer;
}

inline static void xnn_release_simd_memory(void* memory_pointer) {
  xnn_params.allocator.aligned_deallocate(xnn_params.allocator.context, memory_pointer);
}

#ifdef __cplusplus
}  // extern "C"
#endif

