// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Include first for the platform detection macros.
#include "xnnpack/common.h"

#if XNN_PLATFORM_WEB
#include <emscripten/emscripten.h>
#endif  // XNN_PLATFORM_WEB

#if XNN_PLATFORM_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <inttypes.h>
#include <windows.h>

#else
// This define needs to come first because errno include features.h and would have defined macros that lead to
// sys/mman.h not having mremap.
#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include <errno.h>
#if XNN_HAS_MMAP
#include <sys/mman.h>
#endif
#include <unistd.h>
#endif  // XNN_PLATFORM_WINDOWS

#if XNN_PLATFORM_QURT
#include "qurt/qurt_alloc.h"  // NOLINT - header provided by Hexagon toolchain
#endif

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "xnnpack.h"
#include "xnnpack/log.h"
#include "xnnpack/math.h"
#include "xnnpack/memory.h"

// Helpers to allocate/mmap and release memory used by both code and weights cache.

static size_t system_page_size = 0;

static size_t get_page_size() {
  if (system_page_size == 0) {
    // Get page size.
    #if XNN_PLATFORM_WINDOWS
      SYSTEM_INFO sysinfo;
      GetSystemInfo(&sysinfo);
      assert(sysinfo.dwPageSize != 0);
      system_page_size = (size_t) sysinfo.dwPageSize;
    #elif XNN_PLATFORM_QURT || !XNN_HAS_MMAP
      // sysconf(_SC_PAGESIZE) will fail, but we're just using malloc anyway.
      system_page_size = 4096;
    #else
      long result = sysconf(_SC_PAGESIZE);
      if (result == -1) {
        xnn_log_fatal("failed to get page size, error code: %d", errno);
      }
      assert(result >= 0);
      system_page_size = (size_t) result;
    #endif
  }
  assert(is_po2(system_page_size));
  return system_page_size;
}

// Maps `size` bytes of memory, returns pointer to allocation, NULL if failed.
static void* allocate_buffer(size_t size) {
  xnn_log_debug("allocating buffer of size %zu", size);
  assert(size % get_page_size() == 0);
#if XNN_PLATFORM_WINDOWS
  void* p = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  if (p == NULL) {
    xnn_log_error("failed to allocate %zu bytes for code/weights buffer, error code: %" PRIu32,
                  size, (uint32_t) GetLastError());
    return NULL;
  }
#elif XNN_PLATFORM_QURT
  // Emulate through qurt_malloc.
  void* p = qurt_malloc(size);
  if (p == NULL) {
    xnn_log_error("failed to allocate %zu bytes for code/weights buffer, error code: %d", size, errno);
    return NULL;
  }
#elif !XNN_HAS_MMAP
  // Emulate through malloc.
  void* p = malloc(size);
  if (p == NULL) {
    xnn_log_error("failed to allocate %zu bytes for code/weights buffer, error code: %d", size, errno);
    return NULL;
  }
#else
  void* p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) {
    xnn_log_error("failed to allocate %zu bytes for code/weights buffer, error code: %d", size, errno);
    return NULL;
  }
#endif
  return p;
}

// Releases memory previously mapped by `allocate_buffer`, returns xnn_status_success on success.
static enum xnn_status release_memory(void* start, size_t capacity) {
#if XNN_PLATFORM_WINDOWS
  // We only decommitted any unused capacity, so we release all of it now.
  if (!VirtualFree(start, 0, MEM_RELEASE)) {
    xnn_log_error("failed to release code/weights buffer, error code: %" PRIu32, (uint32_t) GetLastError());
    return xnn_status_invalid_state;
  }
#elif XNN_PLATFORM_QURT
  // Emulate through qurt_free.
  qurt_free(start);
#elif !XNN_HAS_MMAP
  // Emulate through free.
  free(start);
#else
  if (munmap(start, capacity) == -1) {
    xnn_log_error("failed to release code/weights buffer, error code: %d", errno);
    return xnn_status_invalid_state;
  }
#endif
  return xnn_status_success;
}

// Resize a buffer at old_pointer of size old_bytes to new_size. The actual new size of the resized buffer is written to
// new_capacity_out, which can be >= new_size due to page alignment requirements.
// Returns a pointer to a buffer which might be the same as old_pointer if we can remap virtual memory, otherwise we
// allocate a new buffer and copy contents of old_buffer over.
static void* resize_buffer(
  void* old_pointer, size_t old_size, size_t old_capacity, size_t new_size, size_t* new_capacity_out)
{
  const size_t new_capacity = round_up_po2(new_size, get_page_size());
  #if XNN_PLATFORM_LINUX
    void* new_pointer = mremap(old_pointer, old_capacity, new_capacity, MREMAP_MAYMOVE, NULL);
    if (new_pointer == MAP_FAILED) {
      xnn_log_error("mremap failed with errno: %d", errno);
      return NULL;
    }
    xnn_log_debug("resize_buffer: remap, old capacity %zu to new capacity %zu", old_capacity, new_capacity);
  #else
    void* new_pointer = allocate_buffer(new_capacity);
    if (new_pointer == NULL) {
      xnn_log_error("allocate_buffer failed");
      return NULL;
    }
    memcpy(new_pointer, old_pointer, old_size);
    // Release old code_buffer.
    const enum xnn_status status = release_memory(old_pointer, old_capacity);
    if (status != xnn_status_success) {
      xnn_log_error("releasing old buffer failed, this could be a leak of %zu bytes", old_capacity);
      // Log but proceed as per normal since we successfully allocated a new memory that can be used by the caller.
    }
    xnn_log_debug("resize_buffer: allocate memory, old capacity %zu to new capacity %zu", old_capacity, new_capacity);
  #endif
  *new_capacity_out = new_capacity;
  return new_pointer;
}

// Releases unused memory. Will write the new capacity to `capacity`.
static enum xnn_status release_unused_memory(size_t size, void* start, size_t* capacity) {
  // Release all unused pages.
  const size_t page_aligned_size = round_up_po2(size, get_page_size());
  const uint8_t* mem_start = (uint8_t*) start;
  const uint8_t* unused_start = mem_start + page_aligned_size;
  assert(*capacity >= page_aligned_size);
  const size_t unused_capacity = *capacity - page_aligned_size;

  xnn_log_debug("releasing memory, start %p, used: %zu, capacity: %zu, unused %zu", mem_start, size, *capacity,
                unused_capacity);

  if (unused_capacity != 0) {
    // Free unused pages.
    #if XNN_PLATFORM_WINDOWS
      // We cannot selectively release pages inside the region of pages, so just decommit them.
      if (!VirtualFree((void*) unused_start, unused_capacity, MEM_DECOMMIT)) {
        xnn_log_error("failed to unmap weights buffer, error code: %" PRIu32, (uint32_t) GetLastError());
        return xnn_status_invalid_state;
      }
      *capacity = page_aligned_size;
    #elif XNN_PLATFORM_QURT || !XNN_HAS_MMAP
      // We can't selectively release parts of this memory --
      // we *could* release the entire block if unused_capacity == *capacity,
      // but that would invalidate the start pointer. Just do nothing.
      (void) unused_start;
    #elif !XNN_PLATFORM_WEB
      // Web does not support partial unmapping.
      if (munmap((void*) unused_start, unused_capacity) == -1) {
        xnn_log_error("failed to unmap weights buffer, error code: %d", errno);
        return xnn_status_invalid_state;
      }
      *capacity = page_aligned_size;
    #else
      if (unused_capacity == *capacity) {
        if (munmap((void*) unused_start, unused_capacity) == -1) {
          xnn_log_error("failed to unmap weights buffer, error code: %d", errno);
          return xnn_status_invalid_state;
        } else {
          *capacity = 0;
        }
      }
    #endif
  }

  return xnn_status_success;
}

enum xnn_memory_permission {
  xnn_memory_permission_read_only,
  xnn_memory_permission_read_execute,
};

static enum xnn_status set_memory_permission(void* start, size_t size, enum xnn_memory_permission permission) {
  #if XNN_PLATFORM_WINDOWS
    DWORD old = 0, prot = 0;
    switch (permission) {
      case xnn_memory_permission_read_only:
        prot = PAGE_READONLY;
        break;
      case xnn_memory_permission_read_execute:
        prot = PAGE_EXECUTE_READ;
        break;
      default:
        XNN_UNREACHABLE;
    }
    if (!VirtualProtect(start, size, prot, &old)) {
      xnn_log_error(
        "failed to set memory permission (%d), error code: %" PRIu32, permission, (uint32_t) GetLastError());
      return xnn_status_invalid_state;
    }
  #elif XNN_PLATFORM_WEB || XNN_PLATFORM_QURT || !XNN_HAS_MMAP
    // Memory protection not supported.
    return xnn_status_success;
  #else
    int prot = 0;
    switch (permission) {
      case xnn_memory_permission_read_only:
        prot = PROT_READ;
        break;
      case xnn_memory_permission_read_execute:
        prot = PROT_READ | PROT_EXEC;
        break;
      default:
        XNN_UNREACHABLE;
    }
    if (mprotect(start, size, prot) == -1) {
      xnn_log_error("failed to set memory permission (%d), error code: %d", permission, errno);
      return xnn_status_invalid_state;
    }
  #endif
  return xnn_status_success;
}

enum xnn_status xnn_allocate_weights_memory(struct xnn_weights_buffer* buffer, size_t size) {
  memset(buffer, 0, sizeof(struct xnn_weights_buffer));
  const size_t page_aligned_size = round_up_po2(size, get_page_size());
  buffer->start = allocate_buffer(page_aligned_size);
  if (buffer->start == NULL) {
    return xnn_status_out_of_memory;
  }

  buffer->size = 0;
  buffer->capacity = page_aligned_size;
  return xnn_status_success;
}

enum xnn_status xnn_release_weights_memory(struct xnn_weights_buffer* buffer) {
  if (buffer->capacity == 0) {
    return xnn_status_success;
  }
  const enum xnn_status status = release_memory(buffer->start, buffer->capacity);
  if (status != xnn_status_success) {
    return status;
  }
  memset(buffer, 0, sizeof(struct xnn_weights_buffer));
  return xnn_status_success;
}

enum xnn_status xnn_reserve_weights_memory(struct xnn_weights_buffer* buffer, size_t min_available_size) {
  if (buffer->size + min_available_size <= buffer->capacity) {
    xnn_log_debug("reserving weights memory of size %zu without growing buffer", min_available_size);
    return xnn_status_success;
  }

  size_t new_capacity = 0;
  void* new_start =
    resize_buffer(buffer->start, buffer->size, buffer->capacity, buffer->size + min_available_size, &new_capacity);
  if (new_start == NULL) {
    xnn_log_error("failed to reserve weights memory");
    return xnn_status_out_of_memory;
  }
  buffer->start = new_start;
  buffer->capacity = new_capacity;

  return xnn_status_success;
}

enum xnn_status xnn_finalize_weights_memory(struct xnn_weights_buffer* buffer) {
  const enum xnn_status status = release_unused_memory(buffer->size, buffer->start, &buffer->capacity);
  if (status != xnn_status_success) {
    return status;
  }

  if (buffer->capacity == 0) {
    return xnn_status_success;
  }

  return set_memory_permission(buffer->start, buffer->size, xnn_memory_permission_read_only);
}
