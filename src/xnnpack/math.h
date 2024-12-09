// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef _MSC_VER
  #include <intrin.h>
  #include <stdlib.h> // For _rotl.
#endif

#include "xnnpack/common.h"
#include "xnnpack/fp16.h"

// stdlib.h from Windows 10 SDK defines min & max macros.
// Undefine them before defining the corresponding functions.
#ifdef min
  #undef min
#endif
#ifdef max
  #undef max
#endif

#ifdef __cplusplus
extern "C" {
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

XNN_INLINE static uint8_t saturating_cast_s16_u8(int16_t input) {
  if (input > UINT8_MAX) return UINT8_MAX;
  if (input < 0) return 0;
  return input;
}

XNN_INLINE static int16_t saturating_cast_s32_s16(int32_t input) {
  if (input > INT16_MAX) return INT16_MAX;
  if (input < INT16_MIN) return INT16_MIN;
  return input;
}

XNN_INLINE static int32_t saturating_cast_s64_s32(int64_t input) {
  if (input > INT32_MAX) return INT32_MAX;
  if (input < INT32_MIN) return INT32_MIN;
  return input;
}

XNN_INLINE static int16_t saturating_add_s16(int16_t x, int16_t y) {
  return saturating_cast_s32_s16((int32_t)x + (int32_t)y);
}

XNN_INLINE static int32_t math_asr_s32_rounding(int32_t x, int n) {
  if (n == 0) return x;
  int32_t rounding = 1 << (n - 1);
  return ((int64_t)x + rounding) >> n;
}

XNN_INLINE static int32_t saturating_rounding_shift_left_s32(int32_t x,
                                                             int32_t shift) {
  if (shift >= 0) {
    return saturating_cast_s64_s32((int64_t)x << shift);
  } else {
    return math_asr_s32_rounding(x, -shift);
  }
}

XNN_INLINE static uint32_t math_abs_s32(int32_t n) {
  #if defined(_MSC_VER)
    return (uint32_t) abs((int) n);
  #else
    return XNN_UNPREDICTABLE(n >= 0) ? (uint32_t) n : -(uint32_t) n;
  #endif
}

// Flip low 15 bits based on high bit.  Reversible.
XNN_INLINE static int16_t math_signcomplement_f16(uint16_t a) {
  return (a & 0x7FFF) ^ -((int16_t) a < 0);
}

XNN_INLINE static int16_t math_min_s16(int16_t a, int16_t b) {
  return XNN_UNPREDICTABLE(a < b) ? a : b;
}

XNN_INLINE static int16_t math_max_s16(int16_t a, int16_t b) {
  return XNN_UNPREDICTABLE(a > b) ? a : b;
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

XNN_INLINE static uint32_t math_popcount_u32(uint32_t x) {
  #if defined(_MSC_VER) && !defined(__clang__)
    uint32_t result = 0;
    for (int i = 0; i < 32; ++i) {
      result += (x >> i) & 1;
    }
    return result;
  #else
    return (uint32_t) __builtin_popcount((unsigned int) x);
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

extern XNN_INTERNAL const uint16_t xnn_table_vlog[129];

#define LOG_SEGMENTS_LOG2 7
#define LOG_SCALE 65536
#define LOG_SCALE_LOG2 16
#define LOG_COEFF 45426

XNN_INLINE static uint32_t math_u32_log32(uint32_t x, uint32_t out_scale) {
  const uint32_t log2x = math_clz_nonzero_u32(x) ^ 31;
  int32_t frac = x - (UINT32_C(1) << log2x);
  frac <<= math_doz_u32(LOG_SCALE_LOG2, log2x);
  frac >>= math_doz_u32(log2x, LOG_SCALE_LOG2);

  const uint32_t base_seg = frac >> (LOG_SCALE_LOG2 - LOG_SEGMENTS_LOG2);
  const uint32_t seg_unit =
      (UINT32_C(1) << LOG_SCALE_LOG2) >> LOG_SEGMENTS_LOG2;

  const int32_t c0 = xnn_table_vlog[base_seg];
  const int32_t c1 = xnn_table_vlog[base_seg + 1];
  const int32_t seg_base = seg_unit * base_seg;
  const int32_t rel_pos =
      math_asr_s32((c1 - c0) * (frac - seg_base), LOG_SCALE_LOG2);
  const uint32_t fraction = frac + c0 + rel_pos;
  const uint32_t log2 = (log2x << LOG_SCALE_LOG2) + fraction;
  const uint32_t round = LOG_SCALE >> 1;
  const uint32_t loge =
      (math_mulext_u32(log2, LOG_COEFF) + round) >> LOG_SCALE_LOG2;

  const uint32_t loge_scaled = (out_scale * loge + round) >> LOG_SCALE_LOG2;
  return loge_scaled;
}

#ifndef __cplusplus
XNN_INLINE static uint32_t math_cvt_sat_u32_f64(double x) {
  #if defined(__GNUC__) && defined(__arm__) && (__GNUC__ >= 9)
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

XNN_INLINE static float math_cvt_fp32_bf16(uint16_t x) {
   union {
    float as_float;
    uint32_t as_uint32;
  } bits;
  bits.as_uint32 = ((uint32_t) x) << 16;
  return bits.as_float;
}

XNN_INLINE static uint16_t math_cvt_bf16_fp32(float x) {
   union {
    float as_float;
    uint32_t as_uint32;
  } bits;
  bits.as_float = x;

  // TODO Handle fraction rounding
  return bits.as_uint32 >> 16;
}

#ifdef __cplusplus
}  // extern "C"
#endif

// We want to use _Float16 if the compiler supports it fully, but it's
// tricky to do this detection; there are compiler versions that define the
// type in broken ways. We're only going to bother using it if the support is
// known to be at least a robust f16<->f32 conversion, which generally means a
// recent version of Clang or GCC, x86 or ARM or RISC-V architectures, and
// (in some cases) the right architecture flags specified on the command line.

#ifndef XNN_HAVE_FLOAT16

// Some non-GCC compilers define __GNUC__, but we only want to detect the Real
// Thing
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && \
    !defined(__INTEL_LLVM_COMPILER)
#define XNN_GNUC_ACTUAL __GNUC__
#else
#define XNN_GNUC_ACTUAL 0
#endif

#if (defined(__i386__) || defined(__x86_64__)) && defined(__SSE2__) && \
    defined(__FLT16_MAX__) && defined(__F16C__) &&                     \
    ((__clang_major__ >= 15 && !defined(_MSC_VER)) || (XNN_GNUC_ACTUAL >= 12))
#define XNN_HAVE_FLOAT16 1
#endif

#if (defined(__aarch64__) && !defined(_MSC_VER)) &&     \
    ((defined(__clang__) && (__clang_major__ >= 15)) || \
     (XNN_GNUC_ACTUAL >= 13))
#define XNN_HAVE_FLOAT16 1
#endif

#if defined(__riscv) && defined(__riscv_zvfh) && __clang__ >= 1600
#define XNN_HAVE_FLOAT16 1
#endif

#ifndef XNN_HAVE_FLOAT16
#define XNN_HAVE_FLOAT16 0
#endif

#endif  // XNN_HAVE_FLOAT16

#if XNN_HAVE_FLOAT16
typedef _Float16 xnn_float16;
#else
// We want float16s to be a distinct type from uint16_t, to avoid accidental
// reinterpret casts as integers. This type is designed to produce errors when
// using it as an arithmetic type in C, and designed to emulate a native float16
// type in C++.
struct xnn_float16 {
  uint16_t value;

#ifdef __cplusplus
  xnn_float16() = default;
  xnn_float16(float x) : value(fp16_ieee_from_fp32_value(x)) {}

  operator float() const { return fp16_ieee_to_fp32_value(value); }
#endif
};
typedef struct xnn_float16 xnn_float16;
#endif

struct xnn_bfloat16 {
  uint16_t value;

#ifdef __cplusplus
  xnn_bfloat16() = default;
  xnn_bfloat16(float x) : value(math_cvt_bf16_fp32(x)) {}

  operator float() const { return math_cvt_fp32_bf16(value); }
#endif
};
typedef struct xnn_bfloat16 xnn_bfloat16;


#ifdef __cplusplus
extern "C" {
#endif

XNN_INLINE static xnn_float16 xnn_float16_from_float(float f) {
#if XNN_HAVE_FLOAT16
  return (xnn_float16) f;
#else
  struct xnn_float16 result;
  result.value = fp16_ieee_from_fp32_value(f);
  return result;
#endif
}

XNN_INLINE static float xnn_float16_to_float(xnn_float16 fp16) {
#if XNN_HAVE_FLOAT16
  return (float) fp16;
#else
  return fp16_ieee_to_fp32_value(fp16.value);
#endif
}

XNN_INLINE static uint16_t xnn_float16_to_bits(xnn_float16 fp16) {
  uint16_t result;
  memcpy(&result, &fp16, sizeof(result));
  return result;
}

XNN_INLINE static xnn_float16 xnn_float16_from_bits(uint16_t x) {
  xnn_float16 result;
  memcpy(&result, &x, sizeof(result));
  return result;
}

XNN_INLINE static xnn_bfloat16 xnn_bfloat16_from_float(float f) {
  xnn_bfloat16 result;
  result.value = math_cvt_bf16_fp32(f);
  return result;
}

XNN_INLINE static float xnn_bfloat16_to_float(xnn_bfloat16 bf16) {
  return math_cvt_fp32_bf16(bf16.value);
}

XNN_INLINE static uint16_t xnn_bfloat16_to_bits(xnn_bfloat16 fp16) {
  uint16_t result;
  memcpy(&result, &fp16, sizeof(result));
  return result;
}

XNN_INLINE static xnn_bfloat16 xnn_bfloat16_from_bits(uint16_t x) {
  xnn_bfloat16 result;
  memcpy(&result, &x, sizeof(result));
  return result;
}

XNN_INLINE static xnn_float16 xnn_float16_zero() {
#if XNN_HAVE_FLOAT16
  return (xnn_float16) 0.0f;
#else
  struct xnn_float16 result;
  result.value = 0;
  return result;
#endif
}

XNN_INLINE static bool xnn_float16_is_zero(xnn_float16 f) {
  // Check for +/- zero (0x0000/0x8000). uint16 overflow is well defined to wrap around.
  return xnn_float16_to_bits(f) * 2 == 0;
}

XNN_INLINE static bool xnn_bfloat16_is_zero(xnn_bfloat16 f) {
  // Check for +/- zero (0x0000/0x8000). uint16 overflow is well defined to wrap around.
  return xnn_bfloat16_to_bits(f) * 2 == 0;
}

#ifdef __cplusplus
}  // extern "C"
#endif
