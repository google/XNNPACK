// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#ifdef __ANDROID__
  #include <malloc.h>
#endif

#include <cpuinfo.h>

extern int posix_memalign(void **memptr, size_t alignment, size_t size);


#define XNN_ALLOCATION_ALIGNMENT 16


inline static void* xnn_allocate_memory(size_t memory_size) {
  void* memory_ptr = NULL;
#if CPUINFO_ARCH_ASMJS || CPUINFO_ARCH_WASM
  memory_ptr = malloc(memory_size);
#elif defined(__ANDROID__)
  memory_ptr = memalign(XNN_ALLOCATION_ALIGNMENT, memory_size);
#else
  if (posix_memalign(&memory_ptr, XNN_ALLOCATION_ALIGNMENT, memory_size) != 0) {
    return NULL;
  }
#endif
  return memory_ptr;
}

inline static void* xnn_allocate_zero_memory(size_t memory_size) {
  void* memory_ptr = xnn_allocate_memory(memory_size);
  if (memory_ptr != NULL) {
    memset(memory_ptr, 0, memory_size);
  }
  return memory_ptr;
}

inline static void xnn_release_memory(void* memory_ptr) {
  free(memory_ptr);
}
