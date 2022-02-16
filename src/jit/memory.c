// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Include first for the platform detection macros.
#include "xnnpack/common.h"

#if XNN_PLATFORM_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <errno.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <stddef.h>
#include <stdint.h>
#include <xnnpack.h>

#include "xnnpack/allocator.h"
#include "xnnpack/log.h"
#include "xnnpack/math.h"

enum xnn_status xnn_allocate_code_memory(struct xnn_code_buffer* buf, size_t size) {
#if XNN_PLATFORM_WINDOWS
  void* p = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  if (p == NULL) {
    xnn_log_error("failed to allocate %zu bytes for JIT code buffer, error code: %" PRIu32,
                  size, (uint32_t) GetLastError());
    return xnn_status_out_of_memory;
  }
#else
  void* p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) {
    xnn_log_error("failed to allocate %zu bytes for JIT code buffer, error code: %d", size, errno);
    return xnn_status_out_of_memory;
  }
#endif

  buf->code = p;
  buf->size = 0;
  buf->capacity = size;
  return xnn_status_success;
}

enum xnn_status xnn_finalize_code_memory(struct xnn_code_buffer* buf) {
  // Get page size.
#if XNN_PLATFORM_WINDOWS
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  const size_t page_size = sysinfo.dwPageSize;
#else
  const long res = sysconf(_SC_PAGESIZE);
  if (res == -1) {
    xnn_log_error("failed to get page size, error code: %d", errno);
    return xnn_status_invalid_state;
  }
  const size_t page_size = res;
#endif

  // Release all unused pages.
  const size_t page_aligned_code_size = round_up_po2(buf->size, page_size);
  const uint8_t* start = (uint8_t*) buf->code;
  const uint8_t* unused_start = start + page_aligned_code_size;
  const size_t unused_capacity = buf->capacity - page_aligned_code_size;

  xnn_log_debug("JIT code memory start %p, used: %zu, capacity: %zu, unused %zu", start, buf->size, buf->capacity,
                unused_capacity);

  if (unused_capacity != 0) {
    // Free unused pages.
    #if XNN_PLATFORM_WINDOWS
      // We cannot selectively release pages inside the region of pages, so just decommit them.
      if (!VirtualFree((void*) unused_start, unused_capacity, MEM_DECOMMIT)) {
        xnn_log_error("failed to unmap code buffer, error code: %" PRIu32, (uint32_t) GetLastError());
        return xnn_status_invalid_state;
      }
    #else
      if (munmap((void*) unused_start, unused_capacity) == -1) {
        xnn_log_error("failed to unmap code buffer, error code: %d", errno);
        return xnn_status_invalid_state;
      }
    #endif
  }

  buf->capacity = page_aligned_code_size;

  if (buf->capacity == 0) {
    return xnn_status_success;
  }

  // Flush icache, do it before changing permissions due to bugs on older ARM64 kernels.
#if (XNN_ARCH_ARM || XNN_ARCH_ARM64) && !XNN_PLATFORM_IOS
  // iOS toolchain doesn't support this, use sys_icache_invalidate, when we support iOS.
  __builtin___clear_cache(buf->code, (void*) ((uint8_t*) buf->code + buf->capacity));
#endif  // (XNN_ARCH_ARM || XNN_ARCH_ARM64) && !XNN_PLATFORM_IOS

  // Set permissions to RX (no write).
#if XNN_PLATFORM_WINDOWS
  DWORD old = 0;
  if (!VirtualProtect(buf->code, buf->size, PAGE_EXECUTE_READ, &old)) {
    xnn_log_error("failed to make code buffer read+execute, error code: %" PRIu32, (uint32_t) GetLastError());
    return xnn_status_invalid_state;
  }
#else
  if (mprotect(buf->code, buf->size, PROT_READ | PROT_EXEC) == -1) {
    xnn_log_error("failed to make code buffer read+execute, error code: %d", errno);
    return xnn_status_invalid_state;
  }
#endif
  return xnn_status_success;
}

enum xnn_status xnn_release_code_memory(struct xnn_code_buffer* buf) {
  if (buf->capacity == 0) {
    return xnn_status_success;
  }
#if XNN_PLATFORM_WINDOWS
  // We only decommited any unused capacity, so we release all of it now.
  if (!VirtualFree(buf->code, 0, MEM_RELEASE)) {
    xnn_log_error("failed to release code buffer for JIT, error code: %" PRIu32, (uint32_t) GetLastError());
    return xnn_status_invalid_state;
  }
#else
  if (munmap(buf->code, buf->capacity) == -1) {
    xnn_log_error("failed to release code buffer for JIT, error code: %d", errno);
    return xnn_status_invalid_state;
  }
#endif
  buf->code = NULL;
  buf->size = 0;
  buf->capacity = 0;
  return xnn_status_success;
}
