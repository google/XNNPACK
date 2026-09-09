// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root of this source tree.

// Wall-clock timer shared between the runtime profiling path in
// src/runtime.c and the per-microkernel timing in src/operator-run.c.
// Deliberately does not include src/xnnpack/log.h: this header reaches
// per-microkernel TUs that do not define XNN_LOG_LEVEL. Clock failures
// silently zero the timestamp; the operator-level path keeps its
// existing logging in src/runtime.c.

#pragma once

#include <stdint.h>
#include <string.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/subgraph.h"

#ifdef __MACH__
#include <time.h>
#elif defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#elif XNN_PLATFORM_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <time.h>
#endif

static inline xnn_timestamp xnn_read_timer(void) {
  xnn_timestamp timestamp;
#ifdef __MACH__
  timestamp = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#elif defined(__EMSCRIPTEN__)
  timestamp = emscripten_get_now();
#elif XNN_PLATFORM_WINDOWS
  if (!QueryPerformanceCounter(&timestamp)) {
    memset(&timestamp, 0, sizeof(timestamp));
  }
#else
  if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0) {
    memset(&timestamp, 0, sizeof(timestamp));
  }
#endif
  return timestamp;
}

// Returns elapsed microseconds. The macOS branch divides nanoseconds; the
// Emscripten branch scales milliseconds; the Windows branch normalises by
// QueryPerformanceFrequency.
static inline uint64_t xnn_get_elapsed_time(const xnn_timestamp* start,
                                            const xnn_timestamp* end) {
#ifdef __MACH__
  return (*end - *start) / UINT64_C(1000);
#elif defined(__EMSCRIPTEN__)
  return (uint64_t)((*end - *start) * 1.0e3);
#elif XNN_PLATFORM_WINDOWS
  LARGE_INTEGER frequency;
  if (!QueryPerformanceFrequency(&frequency)) {
    return 0;
  }
  return ((end->QuadPart - start->QuadPart) * UINT64_C(1000000)) /
         frequency.QuadPart;
#else
  const uint64_t secs = (end->tv_sec - start->tv_sec) * UINT64_C(1000000000);
  const uint64_t ns_secs = end->tv_nsec - start->tv_nsec;
  return (secs + ns_secs) / UINT64_C(1000);
#endif
}
