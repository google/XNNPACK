// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*xnn_f32_unary_math_function)(
  size_t n,
  const float* input,
  float* output);

typedef void (*xnn_f32_ext_unary_math_function)(
  size_t n,
  const float* input,
  float* output_mantissa,
  float* output_exponent);

#define DECLARE_F32_UNARY_MATH_FUNCTION(fn_name) \
  void fn_name(                                  \
    size_t n,                                    \
    const float* input,                          \
    float* output);

#define DECLARE_F32_EXT_UNARY_MATH_FUNCTION(fn_name) \
  void fn_name(                                      \
    size_t n,                                        \
    const float* input,                              \
    float* output_mantissa,                          \
    float* output_exponent);

DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__neonfma_lut64_p2)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__neonfma_p5)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__sse2_p5)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx2_p5)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx2_perm_p3)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx2_perm_p4)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx512f_p5)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx512f_p5_scalef)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx512f_perm_p3)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx512f_perm2_p2)

DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expminus__sse2_p5)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expminus__avx2_p5)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expminus__neonfma_p5)

DECLARE_F32_EXT_UNARY_MATH_FUNCTION(xnn_math_f32_extexp__avx2_p5)
DECLARE_F32_EXT_UNARY_MATH_FUNCTION(xnn_math_f32_extexp__avx512f_p5)

DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_p5_nr2fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__sse2_p5_div)

#ifdef __cplusplus
} /* extern "C" */
#endif
