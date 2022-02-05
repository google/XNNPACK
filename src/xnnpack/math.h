// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>

#ifdef _MSC_VER
  #include <intrin.h>
#endif

#include <xnnpack/common.h>


// stdlib.h from Windows 10 SDK defines min & max macros.
// Undefine them before defining the corresponding functions.
#ifdef min
  #undef min
#endif
#ifdef max
  #undef max
#endif


inline static size_t min(size_t a, size_t b) {
  return XNN_UNPREDICTABLE(b < a) ? b : a;
}

inline static size_t max(size_t a, size_t b) {
  return XNN_UNPREDICTABLE(b < a) ? a : b;
}

inline static size_t doz(size_t a, size_t b) {
  return XNN_UNPREDICTABLE(b < a) ? a - b : 0;
}

inline static size_t divide_round_up(size_t n, size_t q) {
  return XNN_UNPREDICTABLE(n % q == 0) ? n / q : n / q + 1;
}

inline static size_t round_up(size_t n, size_t q) {
  return divide_round_up(n, q) * q;
}

inline static bool is_po2(size_t n) {
  return (n != 0) && ((n & (n - 1)) == 0);
}
inline static size_t round_down_po2(size_t n, size_t q) {
  assert(is_po2(q));
  return n & -q;
}

inline static size_t round_up_po2(size_t n, size_t q) {
  return round_down_po2(n + q - 1, q);
}

inline static size_t subtract_modulo(size_t a, size_t b, size_t m) {
  assert(a < m);
  assert(b < m);
  return XNN_UNPREDICTABLE(a >= b) ? a - b : a - b + m;
}

inline static int32_t math_min_s32(int32_t a, int32_t b) {
  return XNN_UNPREDICTABLE(a < b) ? a : b;
}

inline static int32_t math_max_s32(int32_t a, int32_t b) {
  return XNN_UNPREDICTABLE(a > b) ? a : b;
}

inline static uint32_t math_min_u32(uint32_t a, uint32_t b) {
  return XNN_UNPREDICTABLE(a < b) ? a : b;
}

inline static uint32_t math_max_u32(uint32_t a, uint32_t b) {
  return XNN_UNPREDICTABLE(a > b) ? a : b;
}

inline static float math_muladd_f32(float x, float y, float acc) {
  #if defined(__GNUC__) && defined(__FP_FAST_FMAF)
    return __builtin_fmaf(x, y, acc);
  #elif defined(__clang__) && defined(__riscv)
    return __builtin_fmaf(x, y, acc);
  #else
    return x * y + acc;
  #endif
}

inline static float math_min_f32(float a, float b) {
  #if defined(__GNUC__) && defined(__ARM_ARCH) && (__ARM_ARCH >= 8)
    return __builtin_fminf(a, b);
  #elif defined(__clang__) && defined(__riscv)
    return __builtin_fminf(a, b);
  #else
    return XNN_UNPREDICTABLE(b < a) ? b : a;
  #endif
}

inline static float math_max_f32(float a, float b) {
  #if defined(__GNUC__) && defined(__ARM_ARCH) && (__ARM_ARCH >= 8)
    return __builtin_fmaxf(a, b);
  #elif defined(__clang__) && defined(__riscv)
    return __builtin_fmaxf(a, b);
  #else
    return XNN_UNPREDICTABLE(b < a) ? a : b;
  #endif
}

inline static float math_nonsign_mask_f32() {
  #if defined(__INTEL_COMPILER)
    // Surprisingly, Intel compiler ignores __builtin_nanf payload
    return _castu32_f32(0x7FFFFFFF);
  #elif defined(__GNUC__)
    return __builtin_nanf("0x7FFFFF");
  #else
    union {
      uint32_t as_word;
      float as_float;
    } f;
    f.as_word = 0x7FFFFFFF;
    return f.as_float;
  #endif
}


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

inline static uint32_t math_ctz_u32(uint32_t x) {
  #ifdef _MSC_VER
    unsigned long index;
    _BitScanForward(&index, (unsigned long) x);
    return (uint32_t) index;
  #else
    return __builtin_ctz(x);
  #endif
}
