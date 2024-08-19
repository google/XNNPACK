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

#include <fp16/fp16.h>
#include "xnnpack/common.h"

// SIMD vector type for f16 using SCALAR.
typedef uint16_t xnn_simd_f16_t;
#define xnn_simd_size_f16 1
#define xnn_simd_log2_size_f16 0
#define xnn_simd_bytes_f16 (xnn_simd_size_f16 * sizeof(uint16_t))

#define XNN_SIMD_CONST_F16(var, val) static const xnn_simd_f16_t var = val;

#define XNN_SIMD_CONST_U16(var, val) const xnn_simd_f16_t var = val;

// Arithmetic operations.
static XNN_INLINE xnn_simd_f16_t xnn_zero_f16() {
  return fp16_ieee_from_fp32_value(0.0f);
}

static XNN_INLINE xnn_simd_f16_t xnn_add_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return fp16_ieee_from_fp32_value(fp16_ieee_to_fp32_value(a) +
                                   fp16_ieee_to_fp32_value(b));
}

static XNN_INLINE xnn_simd_f16_t xnn_mul_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return fp16_ieee_from_fp32_value(fp16_ieee_to_fp32_value(a) *
                                   fp16_ieee_to_fp32_value(b));
}

static XNN_INLINE xnn_simd_f16_t xnn_fmadd_f16(xnn_simd_f16_t a,
                                               xnn_simd_f16_t b,
                                               xnn_simd_f16_t c) {
  return fp16_ieee_from_fp32_value(
      (fp16_ieee_to_fp32_value(a) * fp16_ieee_to_fp32_value(b)) +
      fp16_ieee_to_fp32_value(c));
}

static XNN_INLINE xnn_simd_f16_t xnn_fnmadd_f16(xnn_simd_f16_t a,
                                                xnn_simd_f16_t b,
                                                xnn_simd_f16_t c) {
  return fp16_ieee_from_fp32_value(
      fp16_ieee_to_fp32_value(c) -
      (fp16_ieee_to_fp32_value(a) * fp16_ieee_to_fp32_value(b)));
}

static XNN_INLINE xnn_simd_f16_t xnn_fmsub_f16(xnn_simd_f16_t a,
                                               xnn_simd_f16_t b,
                                               xnn_simd_f16_t c) {
  return fp16_ieee_from_fp32_value(
      (fp16_ieee_to_fp32_value(a) * fp16_ieee_to_fp32_value(b)) -
      fp16_ieee_to_fp32_value(c));
}

static XNN_INLINE xnn_simd_f16_t xnn_sub_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return fp16_ieee_from_fp32_value(fp16_ieee_to_fp32_value(a) -
                                   fp16_ieee_to_fp32_value(b));
}

static XNN_INLINE xnn_simd_f16_t xnn_div_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return fp16_ieee_from_fp32_value(fp16_ieee_to_fp32_value(a) /
                                   fp16_ieee_to_fp32_value(b));
}

static XNN_INLINE xnn_simd_f16_t xnn_max_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return fp16_ieee_to_fp32_value(a) > fp16_ieee_to_fp32_value(b) ? a : b;
}

static XNN_INLINE xnn_simd_f16_t xnn_min_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return fp16_ieee_to_fp32_value(a) < fp16_ieee_to_fp32_value(b) ? a : b;
}

static XNN_INLINE xnn_simd_f16_t xnn_abs_f16(xnn_simd_f16_t a) {
  return fp16_ieee_from_fp32_value(fabsf(fp16_ieee_to_fp32_value(a)));
}

static XNN_INLINE xnn_simd_f16_t xnn_neg_f16(xnn_simd_f16_t a) {
  return fp16_ieee_from_fp32_value(-fp16_ieee_to_fp32_value(a));
}

// Logical operations.
static XNN_INLINE xnn_simd_f16_t xnn_and_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return a & b;
}

static XNN_INLINE xnn_simd_f16_t xnn_or_f16(xnn_simd_f16_t a,
                                            xnn_simd_f16_t b) {
  return a | b;
}

static XNN_INLINE xnn_simd_f16_t xnn_xor_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return a ^ b;
}

static XNN_INLINE xnn_simd_f16_t xnn_sll_f16(xnn_simd_f16_t a, uint8_t bits) {
  return a << bits;
}

static XNN_INLINE xnn_simd_f16_t xnn_srl_f16(xnn_simd_f16_t a, uint8_t bits) {
  return a >> bits;
}

static XNN_INLINE xnn_simd_f16_t xnn_sra_f16(xnn_simd_f16_t a, uint8_t bits) {
  return a >> bits;
}

static XNN_INLINE xnn_simd_f16_t xnn_cmpeq_f16(xnn_simd_f16_t a,
                                               xnn_simd_f16_t b) {
  XNN_SIMD_CONST_U16(ones, 0xFFFF)
  return fp16_ieee_to_fp32_value(a) == fp16_ieee_to_fp32_value(b) ? ones : 0.0f;
}

// Special functions.
#define XNN_SIMD_HAVE_RCP_F16 0
#define XNN_SIMD_HAVE_RSQRT_F16 0

static XNN_INLINE xnn_simd_f16_t xnn_getexp_f16(xnn_simd_f16_t a) {
  return fp16_ieee_from_fp32_value((float)((a & 0x7c00) >> 10) - 15);
}

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

static XNN_INLINE xnn_simd_f16_t xnn_set1_or_load_f16(xnn_simd_f16_t *v) {
  return *v;
}

// Tail load/store operations.
static XNN_INLINE xnn_simd_f16_t xnn_load_tail_f16(const xnn_simd_f16_t *input,
                                                   size_t num_elements) {
  return *input;
}

static XNN_INLINE void xnn_store_tail_f16(xnn_simd_f16_t *output,
                                          xnn_simd_f16_t v,
                                          size_t num_elements) {
  *output = v;
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_F16_SCALAR_H_
