// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root of this source tree.

// Optional per-microkernel timing for the xnn_compute_* tile-task
// callbacks. Gated behind XNN_ENABLE_UKERNEL_TIMING (default off) so
// release builds get no extra fields and no overhead.
//
// Looped dispatcher (sum across iterations, one atomic-add at end):
//   XNN_UKERNEL_TIMING_LOCAL;
//   for (...) {
//     XNN_UKERNEL_TIMING_RUN(ctx->ukernel, args, ...);
//   }
//   XNN_UKERNEL_TIMING_COMMIT(ctx->ukernel_elapsed_us);
//
// Single-call dispatcher (atomic-add inline):
//   XNN_UKERNEL_TIMING_RUN_AND_COMMIT(ctx->ukernel_elapsed_us,
//                                     ctx->ukernel, args, ...);
//
// Workers race the per-context slot via __atomic_fetch_add (RELAXED); the
// reader runs after the pthreadpool join, which is the acquire barrier.

#pragma once

#ifndef XNN_ENABLE_UKERNEL_TIMING
#define XNN_ENABLE_UKERNEL_TIMING 0
#endif

#if XNN_ENABLE_UKERNEL_TIMING

#include <stdint.h>

#include "src/xnnpack/subgraph.h"
#include "src/xnnpack/timer.h"

#define XNN_UKERNEL_TIMING_LOCAL uint64_t __ukernel_timing_local = 0

#define XNN_UKERNEL_TIMING_RUN(fn_expr, ...)              \
  do {                                                    \
    const xnn_timestamp _xnn_ut_t0_ = xnn_read_timer();   \
    (fn_expr)(__VA_ARGS__);                               \
    const xnn_timestamp _xnn_ut_t1_ = xnn_read_timer();   \
    __ukernel_timing_local +=                             \
        xnn_get_elapsed_time(&_xnn_ut_t0_, &_xnn_ut_t1_); \
  } while (0)

#define XNN_UKERNEL_TIMING_COMMIT(target)                                      \
  do {                                                                         \
    if (__ukernel_timing_local != 0) {                                         \
      __atomic_fetch_add(&(target), __ukernel_timing_local, __ATOMIC_RELAXED); \
    }                                                                          \
  } while (0)

#define XNN_UKERNEL_TIMING_RUN_AND_COMMIT(target, fn_expr, ...)          \
  do {                                                                   \
    const xnn_timestamp _xnn_ut_t0_ = xnn_read_timer();                  \
    (fn_expr)(__VA_ARGS__);                                              \
    const xnn_timestamp _xnn_ut_t1_ = xnn_read_timer();                  \
    const uint64_t _xnn_ut_elapsed_ =                                    \
        xnn_get_elapsed_time(&_xnn_ut_t0_, &_xnn_ut_t1_);                \
    if (_xnn_ut_elapsed_ != 0) {                                         \
      __atomic_fetch_add(&(target), _xnn_ut_elapsed_, __ATOMIC_RELAXED); \
    }                                                                    \
  } while (0)

#else  // !XNN_ENABLE_UKERNEL_TIMING

#define XNN_UKERNEL_TIMING_LOCAL ((void)0)
#define XNN_UKERNEL_TIMING_RUN(fn_expr, ...) ((fn_expr)(__VA_ARGS__))
#define XNN_UKERNEL_TIMING_COMMIT(target) ((void)0)
#define XNN_UKERNEL_TIMING_RUN_AND_COMMIT(target, fn_expr, ...) \
  ((fn_expr)(__VA_ARGS__))

#endif
