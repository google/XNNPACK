// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

// Provides generic implementations for some SIMD functions, i.e.
// implementations that don't rely on architecture-specific instructions and
// just use the basic wrapped SIMD intrinsics.

#ifndef THIRD_PARTY_XNNPACK_INCLUDE_SIMD_F32_GENERIC_FUNCTIONS_H_
#define THIRD_PARTY_XNNPACK_INCLUDE_SIMD_F32_GENERIC_FUNCTIONS_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"

// Forward declarations of basic SIMD intrinsics.
static XNN_INLINE xnn_simd_f32_t xnn_sub_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b);
static XNN_INLINE xnn_simd_f32_t xnn_and_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b);
static XNN_INLINE xnn_simd_f32_t xnn_or_f32(xnn_simd_f32_t a, xnn_simd_f32_t b);

#ifndef XNN_SIMD_SHIFTS_ARE_MACROS
static XNN_INLINE xnn_simd_f32_t xnn_srl_f32(xnn_simd_f32_t a, uint8_t b);
#endif  // XNN_SIMD_SHIFTS_ARE_MACROS

// Extracts the exponent of the input `a` as a `float` value.
static XNN_INLINE xnn_simd_f32_t xnn_generic_getexp_f32(xnn_simd_f32_t a) {
  // Some useful constants.
  XNN_SIMD_CONST_F32_FROM_INT32(exp_mask, 0x7f800000);
  XNN_SIMD_CONST_F32(bias_256, 256.0f);
  XNN_SIMD_CONST_F32(bias_383, 383.0f);

  // The bits of IEE754 single-precision floating-point format are:
  //
  //   s | e e e e e e e e | m m m m m m m m m m m m m m m m m m m m m m m
  //
  // We start by masking out the exponent and shifting it 8 bits to the right:
  //
  //   0 | 0 0 0 0 0 0 0 0 | e e e e e e e e 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  //
  // These bits are then `or`-ed with `256.0f`, which has a biased exponent of
  // `135` and all mantissa bit set to zero. This is equivalent to adding the
  // biased integer exponent to `256.0`:
  //
  //   0 | 1 0 0 0 0 1 1 1 | e e e e e e e e 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  //
  // We can re-extract the exponent as a `float` value by subtracting `256.0`
  // plus the exponent bias `127.0`, i.e. `383.0`.

  // Extract the exponent and shift the exponent to the most significant bits of
  // the mantissa.
  const xnn_simd_f32_t exp = xnn_srl_f32(xnn_and_f32(a, exp_mask), 8);

  // Add the shifted exponent to `256.0f` by copying its bits to the mantissa,
  // then subtract out `383.0f`, i.e. the original `256.0f` plus the `127`
  // exponent bias, resulting in the unbiased exponent.
  return xnn_sub_f32(xnn_or_f32(bias_256, exp), bias_383);
}

#endif  // THIRD_PARTY_XNNPACK_INCLUDE_SIMD_F32_GENERIC_FUNCTIONS_H_
