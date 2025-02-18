// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

// Provides generic implementations for some SIMD functions, i.e.
// implementations that don't rely on architecture-specific instructions and
// just use the basic wrapped SIMD intrinsics.

#ifndef THIRD_PARTY_XNNPACK_INCLUDE_SIMD_F16_GENERIC_FUNCTIONS_H_
#define THIRD_PARTY_XNNPACK_INCLUDE_SIMD_F16_GENERIC_FUNCTIONS_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"

// Forward declarations of basic SIMD intrinsics.
static XNN_INLINE xnn_simd_f16_t xnn_sub_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b);
static XNN_INLINE xnn_simd_f16_t xnn_and_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b);
static XNN_INLINE xnn_simd_f16_t xnn_or_f16(xnn_simd_f16_t a, xnn_simd_f16_t b);

#ifndef XNN_SIMD_SHIFTS_ARE_MACROS
static XNN_INLINE xnn_simd_f16_t xnn_srl_f16(xnn_simd_f16_t a, uint8_t b);
#endif  // XNN_SIMD_SHIFTS_ARE_MACROS

// Extracts the exponent of the input `a` as a `float` value.
static XNN_INLINE xnn_simd_f16_t xnn_generic_getexp_f16(xnn_simd_f16_t a) {
  // Some useful constants.
  XNN_SIMD_CONST_F16_FROM_INT16(exp_mask, 0x7c00);
  XNN_SIMD_CONST_F16_FROM_FLOAT(bias_32, 32.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(bias_47, 47.0f);

  // The bits of IEE754 half-precision floating-point format are:
  //
  //   s | e e e e e | m m m m m m m m m m
  //
  // We start by masking out the exponent and shifting it 5 bits to the right:
  //
  //   0 | 0 0 0 0 0 | e e e e e 0 0 0 0 0
  //
  // These bits are then `or`-ed with `32.0f`, which has a biased exponent of
  // `20` and all mantissa bit set to zero. This is equivalent to adding the
  // biased integer exponent to `32.0`:
  //
  //   0 | 1 0 1 0 0 | e e e e e 0 0 0 0 0
  //
  // We can re-extract the exponent as a `float` value by subtracting `256.0`
  // plus the exponent bias `127.0`, i.e. `383.0`.

  // Extract the exponent and shift the exponent to the most significant bits of
  // the mantissa.
  const xnn_simd_f16_t exp = xnn_srl_f16(xnn_and_f16(a, exp_mask), 5);

  // Add the shifted exponent to `32.0f` by copying its bits to the mantissa,
  // then subtract out `47.0f`, i.e. the original `32.0f` plus the `15`
  // exponent bias, resulting in the unbiased exponent.
  return xnn_sub_f16(xnn_or_f16(bias_32, exp), bias_47);
}

#endif  // THIRD_PARTY_XNNPACK_INCLUDE_SIMD_F16_GENERIC_FUNCTIONS_H_
