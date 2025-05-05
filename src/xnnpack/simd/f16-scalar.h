// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_F16_SCALAR_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_F16_SCALAR_H_

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"

// SIMD vector type for f16 using SCALAR.
typedef xnn_float16 xnn_simd_f16_t;
#define xnn_simd_size_f16 1
#define xnn_simd_log2_size_f16 0
#define xnn_simd_bytes_f16 (xnn_simd_size_f16 * sizeof(xnn_float16))

#define XNN_SIMD_CONST_F16(var, val) static const xnn_simd_f16_t var = val;

#define XNN_SIMD_CONST_F16_FROM_FLOAT(var, val) \
  const xnn_simd_f16_t var = xnn_float16_from_float(val);

#define XNN_SIMD_CONST_U16(var, val) \
  const xnn_simd_f16_t var = xnn_float16_from_bits(val);

// Arithmetic operations.
static XNN_INLINE xnn_simd_f16_t xnn_zero_f16() { return xnn_float16_zero(); }

static XNN_INLINE xnn_simd_f16_t xnn_add_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
#if XNN_HAVE_FLOAT16
  return a + b;
#else
  return xnn_float16_from_float(xnn_float16_to_float(a) +
                                xnn_float16_to_float(b));
#endif  // XNN_HAVE_FLOAT16
}

static XNN_INLINE xnn_simd_f16_t xnn_mul_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
#if XNN_HAVE_FLOAT16
  return a * b;
#else
  return xnn_float16_from_float(xnn_float16_to_float(a) *
                                xnn_float16_to_float(b));
#endif  // XNN_HAVE_FLOAT16
}

// If we're computing the fused ops in `float`, act as if we're going to
// round like native FMA.
#if XNN_HAVE_FLOAT16
#if ((XNN_ARCH_X86 || XNN_ARCH_X86_64) && defined(__FMA__)) || \
    (XNN_ARCH_ARM64 && __ARM_FEATURE_FMA && defined(__ARM_FEATURE_FP16_FML))
#define XNN_SIMD_HAS_NATIVE_FMA 1
#else
#define XNN_SIMD_HAS_NATIVE_FMA 0
#endif  // ((XNN_ARCH_X86 || XNN_ARCH_X86_64) && defined(__FMA__)) ||
        // (XNN_ARCH_ARM64 && __ARM_FEATURE_FMA &&
        // defined(__ARM_FEATURE_FP16_FML))
#else
#define XNN_SIMD_HAS_NATIVE_FMA 1
#endif  // XNN_HAVE_FLOAT16

static XNN_INLINE xnn_simd_f16_t xnn_fmadd_f16(xnn_simd_f16_t a,
                                               xnn_simd_f16_t b,
                                               xnn_simd_f16_t c) {
#if XNN_HAVE_FLOAT16
  return a * b + c;
#else
  return xnn_float16_from_float(
      (xnn_float16_to_float(a) * xnn_float16_to_float(b)) +
      xnn_float16_to_float(c));
#endif  // XNN_HAVE_FLOAT16
}

static XNN_INLINE xnn_simd_f16_t xnn_fnmadd_f16(xnn_simd_f16_t a,
                                                xnn_simd_f16_t b,
                                                xnn_simd_f16_t c) {
#if XNN_HAVE_FLOAT16
  return c - a * b;
#else
  return xnn_float16_from_float(
      xnn_float16_to_float(c) -
      (xnn_float16_to_float(a) * xnn_float16_to_float(b)));
#endif  // XNN_HAVE_FLOAT16
}

static XNN_INLINE xnn_simd_f16_t xnn_fmsub_f16(xnn_simd_f16_t a,
                                               xnn_simd_f16_t b,
                                               xnn_simd_f16_t c) {
#if XNN_HAVE_FLOAT16
  return a * b - c;
#else
  return xnn_float16_from_float(
      (xnn_float16_to_float(a) * xnn_float16_to_float(b)) -
      xnn_float16_to_float(c));
#endif  // XNN_HAVE_FLOAT16
}

static XNN_INLINE xnn_simd_f16_t xnn_sub_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
#if XNN_HAVE_FLOAT16
  return a - b;
#else
  return xnn_float16_from_float(xnn_float16_to_float(a) -
                                xnn_float16_to_float(b));
#endif  // XNN_HAVE_FLOAT16
}

static XNN_INLINE xnn_simd_f16_t xnn_div_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
#if XNN_HAVE_FLOAT16
  return a / b;
#else
  return xnn_float16_from_float(xnn_float16_to_float(a) /
                                xnn_float16_to_float(b));
#endif  // XNN_HAVE_FLOAT16
}

static XNN_INLINE xnn_simd_f16_t xnn_max_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
#if XNN_HAVE_FLOAT16
  return a > b ? a : b;
#else
  return xnn_float16_to_float(a) > xnn_float16_to_float(b) ? a : b;
#endif  // XNN_HAVE_FLOAT16
}

static XNN_INLINE xnn_simd_f16_t xnn_min_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
#if XNN_HAVE_FLOAT16
  return a < b ? a : b;
#else
  return xnn_float16_to_float(a) < xnn_float16_to_float(b) ? a : b;
#endif  // XNN_HAVE_FLOAT16
}

static XNN_INLINE xnn_simd_f16_t xnn_abs_f16(xnn_simd_f16_t a) {
#if XNN_HAVE_FLOAT16
  return fabsf(a);
#else
  return xnn_float16_from_float(fabsf(xnn_float16_to_float(a)));
#endif  // XNN_HAVE_FLOAT16
}

static XNN_INLINE xnn_simd_f16_t xnn_neg_f16(xnn_simd_f16_t a) {
#if XNN_HAVE_FLOAT16
  return -a;
#else
  return xnn_float16_from_float(-xnn_float16_to_float(a));
#endif  // XNN_HAVE_FLOAT16
}

static XNN_INLINE xnn_simd_f16_t xnn_round_f16(xnn_simd_f16_t a) {
#if XNN_HAVE_FLOAT16
  return roundf(a);
#else
  return xnn_float16_from_float(roundf(xnn_float16_to_float(a)));
#endif  // XNN_HAVE_FLOAT16
}

// Logical operations.
static XNN_INLINE xnn_simd_f16_t xnn_and_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return xnn_float16_from_bits(xnn_float16_to_bits(a) & xnn_float16_to_bits(b));
}

static XNN_INLINE xnn_simd_f16_t xnn_or_f16(xnn_simd_f16_t a,
                                            xnn_simd_f16_t b) {
  return xnn_float16_from_bits(xnn_float16_to_bits(a) | xnn_float16_to_bits(b));
}

static XNN_INLINE xnn_simd_f16_t xnn_xor_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return xnn_float16_from_bits(xnn_float16_to_bits(a) ^ xnn_float16_to_bits(b));
}

static XNN_INLINE xnn_simd_f16_t xnn_sll_f16(xnn_simd_f16_t a, uint8_t bits) {
  return xnn_float16_from_bits(xnn_float16_to_bits(a) << bits);
}

static XNN_INLINE xnn_simd_f16_t xnn_srl_f16(xnn_simd_f16_t a, uint8_t bits) {
  return xnn_float16_from_bits(xnn_float16_to_bits(a) >> bits);
}

static XNN_INLINE xnn_simd_f16_t xnn_cmpeq_f16(xnn_simd_f16_t a,
                                               xnn_simd_f16_t b) {
  XNN_SIMD_CONST_U16(ones, 0xFFFF)
#if XNN_HAVE_FLOAT16
  return a == b ? ones : xnn_zero_f16();
#else
  return xnn_float16_to_bits(a) == xnn_float16_to_bits(b) ? ones
                                                          : xnn_zero_f16();
#endif  // XNN_HAVE_FLOAT16
}

// Special functions.
#define XNN_SIMD_HAVE_RCP_F16 0
#define XNN_SIMD_HAVE_RSQRT_F16 0

// Load/store operations.
static XNN_INLINE xnn_simd_f16_t xnn_loadu_f16(const xnn_simd_f16_t *ptr) {
  return *ptr;
}

static XNN_INLINE xnn_simd_f16_t xnn_load_f16(const xnn_simd_f16_t *ptr) {
  return *ptr;
}

static XNN_INLINE void xnn_storeu_f16(xnn_simd_f16_t *ptr, xnn_simd_f16_t v) {
  *ptr = v;
}

static XNN_INLINE void xnn_store_f16(xnn_simd_f16_t *ptr, xnn_simd_f16_t v) {
  *ptr = v;
}

static XNN_INLINE xnn_simd_f16_t xnn_set1_f16(xnn_simd_f16_t v) { return v; }

// Tail load/store operations.
static XNN_INLINE xnn_simd_f16_t xnn_load_tail_f16(const xnn_simd_f16_t *input,
                                                   size_t num_elements) {
  return *input;
}

static XNN_INLINE xnn_simd_f16_t
xnn_load_tail_safe_f16(const xnn_simd_f16_t *input, size_t num_elements) {
  return *input;
}

static XNN_INLINE void xnn_store_tail_f16(xnn_simd_f16_t *output,
                                          xnn_simd_f16_t v,
                                          size_t num_elements) {
  *output = v;
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_F16_SCALAR_H_
