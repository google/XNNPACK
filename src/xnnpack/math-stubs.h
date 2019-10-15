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

#define DECLARE_F32_UNARY_MATH_FUNCTION(fn_name) \
  void fn_name(                                  \
    size_t n,                                    \
    const float* input,                          \
    float* output);

DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx2_p5)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx2_perm_p3)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx2_perm_p4)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx512f_p5)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx512f_p5_scalef)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx512f_perm_p3)

#ifdef __cplusplus
} /* extern "C" */
#endif
