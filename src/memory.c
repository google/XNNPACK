// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#ifdef __ANDROID__
  #include <malloc.h>
#endif

#include <xnnpack/common.h>
#include <xnnpack/memory.h>


extern int posix_memalign(void **memptr, size_t alignment, size_t size);


void* xnn_allocate(void* context, size_t size) {
  return malloc(size);
}

void* xnn_reallocate(void* context, void* pointer, size_t size) {
  return realloc(pointer, size);
}

void xnn_deallocate(void* context, void* pointer) {
  if XNN_LIKELY(pointer != NULL) {
    free(pointer);
  }
}

void* xnn_aligned_allocate(void* context, size_t alignment, size_t size) {
#if XNN_ARCH_ASMJS || XNN_ARCH_WASM
  assert(alignment <= 2 * sizeof(void*));
  return malloc(size);
#elif defined(__ANDROID__)
  return memalign(alignment, size);
#elif defined(_WIN32)
  return _aligned_malloc(size, alignment);
#else
  void* memory_ptr = NULL;
  if (posix_memalign(&memory_ptr, alignment, size) != 0) {
    return NULL;
  }
  return memory_ptr;
#endif
}

void xnn_aligned_deallocate(void* context, void* pointer) {
  if XNN_LIKELY(pointer != NULL) {
    #if defined(_WIN32)
      _aligned_free(pointer);
    #else
      free(pointer);
    #endif
  }
}
