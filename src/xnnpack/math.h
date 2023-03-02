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
  #include <stdlib.h> // For _rotl.
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


XNN_INLINE static size_t min(size_t a, size_t b) {
  return XNN_UNPREDICTABLE(b < a) ? b : a;
}

XNN_INLINE static size_t max(size_t a, size_t b) {
  return XNN_UNPREDICTABLE(b < a) ? a : b;
}

XNN_INLINE static size_t doz(size_t a, size_t b) {
  return XNN_UNPREDICTABLE(b < a) ? a - b : 0;
}

XNN_INLINE static size_t divide_round_up(size_t n, size_t q) {
  return XNN_UNPREDICTABLE(n % q == 0) ? n / q : n / q + 1;
}

XNN_INLINE static size_t round_up(size_t n, size_t q) {
  return divide_round_up(n, q) * q;
}

XNN_INLINE static bool is_po2(size_t n) {
  return (n != 0) && ((n & (n - 1)) == 0);
}
XNN_INLINE static size_t round_down_po2(size_t n, size_t q) {
  assert(is_po2(q));
  return n & -q;
}

XNN_INLINE static size_t round_up_po2(size_t n, size_t q) {
  return round_down_po2(n + q - 1, q);
}

XNN_INLINE static size_t mod_po2(size_t n, size_t m) {
  assert(is_po2(m));
  return n & (m - 1);
}

XNN_INLINE static size_t subtract_modulo(size_t a, size_t b, size_t m) {
  assert(a < m);
  assert(b < m);
  return XNN_UNPREDICTABLE(a >= b) ? a - b : a - b + m;
}

XNN_INLINE static float uint32_as_float(uint32_t i) {
  union {
    uint32_t as_uint32;
    float as_float;
  } bits = { i };
  return bits.as_float;
}

XNN_INLINE static uint32_t float_as_uint32(float f) {
  union {
    float as_float;
    uint32_t as_uint32;
  } bits = { f };
  return bits.as_uint32;
}

XNN_INLINE static double uint64_as_double(uint64_t i) {
  union {
    uint64_t as_uint64;
    double as_double;
  } bits = { i };
  return bits.as_double;
}

XNN_INLINE static uint64_t double_as_uint64(double f) {
  union {
    double as_double;
    uint64_t as_uint64;
  } bits = { f };
  return bits.as_uint64;
}

XNN_INLINE static uint32_t math_abs_s32(int32_t n) {
  #if defined(_MSC_VER)
    return (uint32_t) abs((int) n);
  #else
    return XNN_UNPREDICTABLE(n >= 0) ? (uint32_t) n : -(uint32_t) n;
  #endif
}

XNN_INLINE static int32_t math_min_s32(int32_t a, int32_t b) {
  return XNN_UNPREDICTABLE(a < b) ? a : b;
}

XNN_INLINE static int32_t math_max_s32(int32_t a, int32_t b) {
  return XNN_UNPREDICTABLE(a > b) ? a : b;
}

XNN_INLINE static uint32_t math_min_u32(uint32_t a, uint32_t b) {
  return XNN_UNPREDICTABLE(a < b) ? a : b;
}

XNN_INLINE static uint32_t math_max_u32(uint32_t a, uint32_t b) {
  return XNN_UNPREDICTABLE(a > b) ? a : b;
}

XNN_INLINE static uint32_t math_doz_u32(uint32_t a, uint32_t b) {
  return XNN_UNPREDICTABLE(a > b) ? a - b : 0;
}

XNN_INLINE static int64_t math_mulext_s32(int32_t a, int32_t b) {
#if defined(_MSC_VER) && defined(_M_IX86)
  return (int64_t) __emul((int) a, (int) b);
#else
  return (int64_t) a * (int64_t) b;
#endif
}

XNN_INLINE static uint64_t math_mulext_u32(uint32_t a, uint32_t b) {
#if defined(_MSC_VER) && defined(_M_IX86)
  return (uint64_t) __emulu((unsigned int) a, (unsigned int) b);
#else
  return (uint64_t) a * (uint64_t) b;
#endif
}

XNN_INLINE static float math_muladd_f32(float x, float y, float acc) {
  #if defined(__GNUC__) && defined(__FP_FAST_FMAF)
    return __builtin_fmaf(x, y, acc);
  #elif defined(__clang__) && defined(__riscv)
    return __builtin_fmaf(x, y, acc);
  #else
    return x * y + acc;
  #endif
}

XNN_INLINE static float math_pmin_f32(float a, float b) {
  return XNN_UNPREDICTABLE(b < a) ? b : a;
}

XNN_INLINE static float math_pmax_f32(float a, float b) {
  return XNN_UNPREDICTABLE(b < a) ? a : b;
}

XNN_INLINE static double math_pmin_f64(double a, double b) {
  return XNN_UNPREDICTABLE(b < a) ? b : a;
}

XNN_INLINE static double math_pmax_f64(double a, double b) {
  return XNN_UNPREDICTABLE(b < a) ? a : b;
}

XNN_INLINE static float math_min_f32(float a, float b) {
  #if defined(__GNUC__) && defined(__ARM_ARCH) && (__ARM_ARCH >= 8)
    return __builtin_fminf(a, b);
  #elif defined(__clang__) && defined(__riscv)
    return __builtin_fminf(a, b);
  #else
    return XNN_UNPREDICTABLE(b < a) ? b : a;
  #endif
}

XNN_INLINE static float math_max_f32(float a, float b) {
  #if defined(__GNUC__) && defined(__ARM_ARCH) && (__ARM_ARCH >= 8)
    return __builtin_fmaxf(a, b);
  #elif defined(__clang__) && defined(__riscv)
    return __builtin_fmaxf(a, b);
  #else
    return XNN_UNPREDICTABLE(b < a) ? a : b;
  #endif
}

XNN_INLINE static double math_min_f64(double a, double b) {
  #if defined(__GNUC__) && defined(__ARM_ARCH) && (__ARM_ARCH >= 8)
    return __builtin_fmin(a, b);
  #elif defined(__clang__) && defined(__riscv)
    return __builtin_fmin(a, b);
  #else
    return XNN_UNPREDICTABLE(b < a) ? b : a;
  #endif
}

XNN_INLINE static double math_max_f64(double a, double b) {
  #if defined(__GNUC__) && defined(__ARM_ARCH) && (__ARM_ARCH >= 8)
    return __builtin_fmax(a, b);
  #elif defined(__clang__) && defined(__riscv)
    return __builtin_fmax(a, b);
  #else
    return XNN_UNPREDICTABLE(b < a) ? a : b;
  #endif
}

XNN_INLINE static float math_nonsign_mask_f32() {
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
XNN_INLINE static int32_t math_asr_s32(int32_t x, uint32_t n) {
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
XNN_INLINE static int64_t math_asr_s64(int64_t x, uint32_t n) {
  #ifdef XNN_USE_SHIFT_BASE_UB_WORKAROUND
    return x >= 0 ? x >> n : ~(~x >> n);
  #else
    return x >> n;
  #endif
}

XNN_INLINE static uint32_t math_clz_u32(uint32_t x) {
  #if defined(_MSC_VER) && !defined(__clang__)
    unsigned long index;
    if XNN_UNPREDICTABLE(_BitScanReverse(&index, (unsigned long) x) != 0) {
      return (uint32_t) index ^ 31;
    } else {
      return 32;
    }
  #else
    if XNN_UNPREDICTABLE(x == 0) {
      return 32;
    } else {
      return (uint32_t) __builtin_clz((unsigned int) x);
    }
  #endif
}

XNN_INLINE static uint32_t math_clz_nonzero_u32(uint32_t x) {
  assert(x != 0);
  #if defined(_MSC_VER) && !defined(__clang__)
    unsigned long index;
    _BitScanReverse(&index, (unsigned long) x);
    return (uint32_t) index ^ 31;
  #else
    return (uint32_t) __builtin_clz((unsigned int) x);
  #endif
}

XNN_INLINE static uint32_t math_ctz_u32(uint32_t x) {
  #if defined(_MSC_VER) && !defined(__clang__)
    unsigned long index;
    _BitScanForward(&index, (unsigned long) x);
    return (uint32_t) index;
  #else
    return (uint32_t) __builtin_ctz((unsigned int) x);
  #endif
}

XNN_INLINE static uint32_t math_rotl_u32(uint32_t x, int8_t r)
{
  #if XNN_COMPILER_MSVC
    return _rotl((unsigned int) x, (int) r);
  #else
    return (x << r) | (x >> (32 - r));
  #endif
}

#ifndef __cplusplus
XNN_INLINE static uint32_t math_cvt_sat_u32_f64(double x) {
  #if defined(__GNUC__) && defined(__arm__)
    uint32_t i;
    __asm__ ("vcvt.u32.f64 %[i], %P[x]"
      : [i] "=t" (i)
      : [x] "w" (x));
    return i;
  #elif defined(__GNUC__) && defined(__aarch64__)
    uint32_t i;
    __asm__ ("fcvtnu %w[i], %d[x]"
      : [i] "=r" (i)
      : [x] "w" (x));
    return i;
  #elif defined(__GNUC__) && defined(__riscv)
    uint32_t i;
    __asm__ ("fcvt.wu.d %[i], %[x], rne"
      : [i] "=r" (i)
      : [x] "f" (x));
    return i;
  #elif defined(__clang__) && defined(__wasm__) && defined(__wasm_nontrapping_fptoint__)
    return __builtin_wasm_trunc_saturate_u_i32_f64(rint(x));
  #else
    x = math_max_f64(x, 0.0);
    x = math_min_f64(x, 4294967295.0);
    return (uint32_t) double_as_uint64(x + 0x1.0p+52);
  #endif
}
#endif
