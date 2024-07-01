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

#include "xnnpack.h"
#include "xnnpack/allocator.h"
#include "xnnpack/common.h"


extern int posix_memalign(void **memptr, size_t alignment, size_t size);


static void* xnn_allocate(void* context, size_t size) {
  return malloc(size);
}

static void* xnn_reallocate(void* context, void* pointer, size_t size) {
  return realloc(pointer, size);
}

static void xnn_deallocate(void* context, void* pointer) {
  if XNN_LIKELY(pointer != NULL) {
    free(pointer);
  }
}

static void* xnn_aligned_allocate(void* context, size_t alignment, size_t size) {
#if XNN_ARCH_WASM
  assert(alignment <= 2 * sizeof(void*));
  return malloc(size);
#elif XNN_PLATFORM_ANDROID
  return memalign(alignment, size);
#elif XNN_PLATFORM_WINDOWS
  return _aligned_malloc(size, alignment);
#else
  void* memory_ptr = NULL;
  if (posix_memalign(&memory_ptr, alignment, size) != 0) {
    return NULL;
  }
  return memory_ptr;
#endif
}

static void xnn_aligned_deallocate(void* context, void* pointer) {
  if XNN_LIKELY(pointer != NULL) {
    #if defined(_WIN32)
      _aligned_free(pointer);
    #else
      free(pointer);
    #endif
  }
}

const struct xnn_allocator xnn_default_allocator = {
  .allocate = xnn_allocate,
  .reallocate = xnn_reallocate,
  .deallocate = xnn_deallocate,
  .aligned_allocate = xnn_aligned_allocate,
  .aligned_deallocate = xnn_aligned_deallocate,
};
