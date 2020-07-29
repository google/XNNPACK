// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__cplusplus) && (__cplusplus >= 201103L)
  #include <climits>
  #include <cstdint>
  #include <cstdbool>
  #include <cassert>
#else
  #include <limits.h>
  #include <stdint.h>
  #include <stdbool.h>
  #include <assert.h>
#endif

#include <fp16.h>

#include <xnnpack/common.h>


#if defined(__clang__)
  #if __clang_major__ == 3 && __clang_minor__ >= 7 || __clang_major__ > 3
    #define XNN_IGNORE_SHIFT_BASE_UB __attribute__((__no_sanitize__("shift-base")))
  #else
    #define XNN_IGNORE_SHIFT_BASE_UB
  #endif
#elif defined(__GNUC__)
  #if __GNUC__ >= 8
    #define XNN_IGNORE_SHIFT_BASE_UB __attribute__((__no_sanitize__("shift-base")))
  #elif __GNUC__ == 4 && __GNUC_MINOR__ >= 9 || __GNUC__ > 4
    // 4.9 <= gcc < 8 support ubsan, but doesn't support no_sanitize attribute
    #define XNN_IGNORE_SHIFT_BASE_UB
    #ifndef XNN_USE_SHIFT_BASE_UB_WORKAROUND
      #define XNN_USE_SHIFT_BASE_UB_WORKAROUND 1
    #endif
  #else
    #define XNN_IGNORE_SHIFT_BASE_UB
  #endif
#else
  #define XNN_IGNORE_SHIFT_BASE_UB
#endif

XNN_IGNORE_SHIFT_BASE_UB
inline static int32_t asr_s32(int32_t x, uint32_t n) {
  #ifdef XNN_USE_SHIFT_BASE_UB_WORKAROUND
    #if XNN_ARCH_X86_64 || XNN_ARCH_ARM64
      return (int32_t) ((uint64_t) (int64_t) x >> n);
    #else
      return x >= 0 ? x >> n : ~(~x >> n);
    #endif
  #else
    return x >> n;
  #endif
}

XNN_IGNORE_SHIFT_BASE_UB
inline static int64_t asr_s64(int64_t x, uint32_t n) {
  #ifdef XNN_USE_SHIFT_BASE_UB_WORKAROUND
    return x >= 0 ? x >> n : ~(~x >> n);
  #else
    return x >> n;
  #endif
}
