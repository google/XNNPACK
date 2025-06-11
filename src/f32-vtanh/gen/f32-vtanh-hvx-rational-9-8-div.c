// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vtanh/rational-9-8.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/simd/f32-hvx.h"

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f32_vtanh_ukernel__hvx_rational_9_8_div_u32(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 32);

  // Cap the inputs to this value as `tanh(x)` will always be `+/-1.0f` beyond
  // this point. This value is chosen as the first floating point number as of
  // which the interpolation returns 1.0f.
  #if XNN_SIMD_HAS_NATIVE_FMA
    XNN_SIMD_CONST_F32(vmax_x, 7.9807181358e+00f);
    XNN_SIMD_CONST_F32(vmin_x, -7.9807181358e+00f);
  #else
    XNN_SIMD_CONST_F32(vmax_x, 7.8522667885e+00f);
    XNN_SIMD_CONST_F32(vmin_x, -7.8522667885e+00f);
  #endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  // XNN_SIMD_CONST_F32(valpha_1, 1.0000000000e+00f);
  XNN_SIMD_CONST_F32(valpha_3, 1.3412411511e-01f);
  XNN_SIMD_CONST_F32(valpha_5, 3.5330520477e-03f);
  XNN_SIMD_CONST_F32(valpha_7, 2.1235626264e-05f);
  XNN_SIMD_CONST_F32(valpha_9, 1.4248920266e-08f);


  // The monomial coefficients of the denominator polynomial (even).
  // XNN_SIMD_CONST_F32(vbeta_0, 1.0000000000e+00f);
  XNN_SIMD_CONST_F32(vbeta_2, 4.6745735407e-01f);
  XNN_SIMD_CONST_F32(vbeta_4, 2.6018999517e-02f);
  XNN_SIMD_CONST_F32(vbeta_6, 3.3472978976e-04f);
  XNN_SIMD_CONST_F32(vbeta_8, 8.1365948290e-07f);

  // Some useful constants.
  XNN_SIMD_CONST_F32(vone, 1.0f);

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Clamp the inputs to the interpolation range.
    vx = xnn_min_f32(vmax_x, vx);
    vx = xnn_max_f32(vmin_x, vx);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_9, valpha_7);
    vp = xnn_fmadd_f32(vx2, vp, valpha_5);
    vp = xnn_fmadd_f32(vx2, vp, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, vone);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_8, vbeta_6);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_4);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    // Clamp the inputs to the interpolation range.
    vx = xnn_min_f32(vmax_x, vx);
    vx = xnn_max_f32(vmin_x, vx);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_9, valpha_7);
    vp = xnn_fmadd_f32(vx2, vp, valpha_5);
    vp = xnn_fmadd_f32(vx2, vp, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, vone);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_8, vbeta_6);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_4);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vtanh_ukernel__hvx_rational_9_8_div_u64(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 32);

  // Cap the inputs to this value as `tanh(x)` will always be `+/-1.0f` beyond
  // this point. This value is chosen as the first floating point number as of
  // which the interpolation returns 1.0f.
  #if XNN_SIMD_HAS_NATIVE_FMA
    XNN_SIMD_CONST_F32(vmax_x, 7.9807181358e+00f);
    XNN_SIMD_CONST_F32(vmin_x, -7.9807181358e+00f);
  #else
    XNN_SIMD_CONST_F32(vmax_x, 7.8522667885e+00f);
    XNN_SIMD_CONST_F32(vmin_x, -7.8522667885e+00f);
  #endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  // XNN_SIMD_CONST_F32(valpha_1, 1.0000000000e+00f);
  XNN_SIMD_CONST_F32(valpha_3, 1.3412411511e-01f);
  XNN_SIMD_CONST_F32(valpha_5, 3.5330520477e-03f);
  XNN_SIMD_CONST_F32(valpha_7, 2.1235626264e-05f);
  XNN_SIMD_CONST_F32(valpha_9, 1.4248920266e-08f);


  // The monomial coefficients of the denominator polynomial (even).
  // XNN_SIMD_CONST_F32(vbeta_0, 1.0000000000e+00f);
  XNN_SIMD_CONST_F32(vbeta_2, 4.6745735407e-01f);
  XNN_SIMD_CONST_F32(vbeta_4, 2.6018999517e-02f);
  XNN_SIMD_CONST_F32(vbeta_6, 3.3472978976e-04f);
  XNN_SIMD_CONST_F32(vbeta_8, 8.1365948290e-07f);

  // Some useful constants.
  XNN_SIMD_CONST_F32(vone, 1.0f);

  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 64;

    // Clamp the inputs to the interpolation range.
    vx_0 = xnn_min_f32(vmax_x, vx_0);
    vx_1 = xnn_min_f32(vmax_x, vx_1);
    vx_0 = xnn_max_f32(vmin_x, vx_0);
    vx_1 = xnn_max_f32(vmin_x, vx_1);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2_0 = xnn_mul_f32(vx_0, vx_0);
    const xnn_simd_f32_t vx2_1 = xnn_mul_f32(vx_1, vx_1);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx2_0, valpha_9, valpha_7);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx2_1, valpha_9, valpha_7);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_5);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_5);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_3);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_3);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, vone);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, vone);
    vp_0 = xnn_mul_f32(vx_0, vp_0);
    vp_1 = xnn_mul_f32(vx_1, vp_1);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vx2_0, vbeta_8, vbeta_6);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vx2_1, vbeta_8, vbeta_6);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_4);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_4);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_2);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_2);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vone);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy_0 = xnn_div_f32(vp_0, vq_0);
    const xnn_simd_f32_t vy_1 = xnn_div_f32(vp_1, vq_1);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    output += 64;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Clamp the inputs to the interpolation range.
    vx = xnn_min_f32(vmax_x, vx);
    vx = xnn_max_f32(vmin_x, vx);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_9, valpha_7);
    vp = xnn_fmadd_f32(vx2, vp, valpha_5);
    vp = xnn_fmadd_f32(vx2, vp, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, vone);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_8, vbeta_6);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_4);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    // Clamp the inputs to the interpolation range.
    vx = xnn_min_f32(vmax_x, vx);
    vx = xnn_max_f32(vmin_x, vx);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_9, valpha_7);
    vp = xnn_fmadd_f32(vx2, vp, valpha_5);
    vp = xnn_fmadd_f32(vx2, vp, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, vone);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_8, vbeta_6);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_4);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vtanh_ukernel__hvx_rational_9_8_div_u96(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 32);

  // Cap the inputs to this value as `tanh(x)` will always be `+/-1.0f` beyond
  // this point. This value is chosen as the first floating point number as of
  // which the interpolation returns 1.0f.
  #if XNN_SIMD_HAS_NATIVE_FMA
    XNN_SIMD_CONST_F32(vmax_x, 7.9807181358e+00f);
    XNN_SIMD_CONST_F32(vmin_x, -7.9807181358e+00f);
  #else
    XNN_SIMD_CONST_F32(vmax_x, 7.8522667885e+00f);
    XNN_SIMD_CONST_F32(vmin_x, -7.8522667885e+00f);
  #endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  // XNN_SIMD_CONST_F32(valpha_1, 1.0000000000e+00f);
  XNN_SIMD_CONST_F32(valpha_3, 1.3412411511e-01f);
  XNN_SIMD_CONST_F32(valpha_5, 3.5330520477e-03f);
  XNN_SIMD_CONST_F32(valpha_7, 2.1235626264e-05f);
  XNN_SIMD_CONST_F32(valpha_9, 1.4248920266e-08f);


  // The monomial coefficients of the denominator polynomial (even).
  // XNN_SIMD_CONST_F32(vbeta_0, 1.0000000000e+00f);
  XNN_SIMD_CONST_F32(vbeta_2, 4.6745735407e-01f);
  XNN_SIMD_CONST_F32(vbeta_4, 2.6018999517e-02f);
  XNN_SIMD_CONST_F32(vbeta_6, 3.3472978976e-04f);
  XNN_SIMD_CONST_F32(vbeta_8, 8.1365948290e-07f);

  // Some useful constants.
  XNN_SIMD_CONST_F32(vone, 1.0f);

  for (; batch >= 96 * sizeof(float); batch -= 96 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    input += 96;

    // Clamp the inputs to the interpolation range.
    vx_0 = xnn_min_f32(vmax_x, vx_0);
    vx_1 = xnn_min_f32(vmax_x, vx_1);
    vx_2 = xnn_min_f32(vmax_x, vx_2);
    vx_0 = xnn_max_f32(vmin_x, vx_0);
    vx_1 = xnn_max_f32(vmin_x, vx_1);
    vx_2 = xnn_max_f32(vmin_x, vx_2);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2_0 = xnn_mul_f32(vx_0, vx_0);
    const xnn_simd_f32_t vx2_1 = xnn_mul_f32(vx_1, vx_1);
    const xnn_simd_f32_t vx2_2 = xnn_mul_f32(vx_2, vx_2);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx2_0, valpha_9, valpha_7);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx2_1, valpha_9, valpha_7);
    xnn_simd_f32_t vp_2 = xnn_fmadd_f32(vx2_2, valpha_9, valpha_7);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_5);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_5);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, valpha_5);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_3);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_3);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, valpha_3);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, vone);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, vone);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, vone);
    vp_0 = xnn_mul_f32(vx_0, vp_0);
    vp_1 = xnn_mul_f32(vx_1, vp_1);
    vp_2 = xnn_mul_f32(vx_2, vp_2);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vx2_0, vbeta_8, vbeta_6);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vx2_1, vbeta_8, vbeta_6);
    xnn_simd_f32_t vq_2 = xnn_fmadd_f32(vx2_2, vbeta_8, vbeta_6);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_4);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_4);
    vq_2 = xnn_fmadd_f32(vx2_2, vq_2, vbeta_4);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_2);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_2);
    vq_2 = xnn_fmadd_f32(vx2_2, vq_2, vbeta_2);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vone);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vone);
    vq_2 = xnn_fmadd_f32(vx2_2, vq_2, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy_0 = xnn_div_f32(vp_0, vq_0);
    const xnn_simd_f32_t vy_1 = xnn_div_f32(vp_1, vq_1);
    const xnn_simd_f32_t vy_2 = xnn_div_f32(vp_2, vq_2);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_2);
    output += 96;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Clamp the inputs to the interpolation range.
    vx = xnn_min_f32(vmax_x, vx);
    vx = xnn_max_f32(vmin_x, vx);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_9, valpha_7);
    vp = xnn_fmadd_f32(vx2, vp, valpha_5);
    vp = xnn_fmadd_f32(vx2, vp, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, vone);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_8, vbeta_6);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_4);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    // Clamp the inputs to the interpolation range.
    vx = xnn_min_f32(vmax_x, vx);
    vx = xnn_max_f32(vmin_x, vx);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_9, valpha_7);
    vp = xnn_fmadd_f32(vx2, vp, valpha_5);
    vp = xnn_fmadd_f32(vx2, vp, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, vone);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_8, vbeta_6);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_4);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vtanh_ukernel__hvx_rational_9_8_div_u128(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 32);

  // Cap the inputs to this value as `tanh(x)` will always be `+/-1.0f` beyond
  // this point. This value is chosen as the first floating point number as of
  // which the interpolation returns 1.0f.
  #if XNN_SIMD_HAS_NATIVE_FMA
    XNN_SIMD_CONST_F32(vmax_x, 7.9807181358e+00f);
    XNN_SIMD_CONST_F32(vmin_x, -7.9807181358e+00f);
  #else
    XNN_SIMD_CONST_F32(vmax_x, 7.8522667885e+00f);
    XNN_SIMD_CONST_F32(vmin_x, -7.8522667885e+00f);
  #endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  // XNN_SIMD_CONST_F32(valpha_1, 1.0000000000e+00f);
  XNN_SIMD_CONST_F32(valpha_3, 1.3412411511e-01f);
  XNN_SIMD_CONST_F32(valpha_5, 3.5330520477e-03f);
  XNN_SIMD_CONST_F32(valpha_7, 2.1235626264e-05f);
  XNN_SIMD_CONST_F32(valpha_9, 1.4248920266e-08f);


  // The monomial coefficients of the denominator polynomial (even).
  // XNN_SIMD_CONST_F32(vbeta_0, 1.0000000000e+00f);
  XNN_SIMD_CONST_F32(vbeta_2, 4.6745735407e-01f);
  XNN_SIMD_CONST_F32(vbeta_4, 2.6018999517e-02f);
  XNN_SIMD_CONST_F32(vbeta_6, 3.3472978976e-04f);
  XNN_SIMD_CONST_F32(vbeta_8, 8.1365948290e-07f);

  // Some useful constants.
  XNN_SIMD_CONST_F32(vone, 1.0f);

  for (; batch >= 128 * sizeof(float); batch -= 128 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_3 = xnn_loadu_f32(input + 3 * xnn_simd_size_f32);
    input += 128;

    // Clamp the inputs to the interpolation range.
    vx_0 = xnn_min_f32(vmax_x, vx_0);
    vx_1 = xnn_min_f32(vmax_x, vx_1);
    vx_2 = xnn_min_f32(vmax_x, vx_2);
    vx_3 = xnn_min_f32(vmax_x, vx_3);
    vx_0 = xnn_max_f32(vmin_x, vx_0);
    vx_1 = xnn_max_f32(vmin_x, vx_1);
    vx_2 = xnn_max_f32(vmin_x, vx_2);
    vx_3 = xnn_max_f32(vmin_x, vx_3);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2_0 = xnn_mul_f32(vx_0, vx_0);
    const xnn_simd_f32_t vx2_1 = xnn_mul_f32(vx_1, vx_1);
    const xnn_simd_f32_t vx2_2 = xnn_mul_f32(vx_2, vx_2);
    const xnn_simd_f32_t vx2_3 = xnn_mul_f32(vx_3, vx_3);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx2_0, valpha_9, valpha_7);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx2_1, valpha_9, valpha_7);
    xnn_simd_f32_t vp_2 = xnn_fmadd_f32(vx2_2, valpha_9, valpha_7);
    xnn_simd_f32_t vp_3 = xnn_fmadd_f32(vx2_3, valpha_9, valpha_7);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_5);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_5);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, valpha_5);
    vp_3 = xnn_fmadd_f32(vx2_3, vp_3, valpha_5);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_3);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_3);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, valpha_3);
    vp_3 = xnn_fmadd_f32(vx2_3, vp_3, valpha_3);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, vone);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, vone);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, vone);
    vp_3 = xnn_fmadd_f32(vx2_3, vp_3, vone);
    vp_0 = xnn_mul_f32(vx_0, vp_0);
    vp_1 = xnn_mul_f32(vx_1, vp_1);
    vp_2 = xnn_mul_f32(vx_2, vp_2);
    vp_3 = xnn_mul_f32(vx_3, vp_3);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vx2_0, vbeta_8, vbeta_6);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vx2_1, vbeta_8, vbeta_6);
    xnn_simd_f32_t vq_2 = xnn_fmadd_f32(vx2_2, vbeta_8, vbeta_6);
    xnn_simd_f32_t vq_3 = xnn_fmadd_f32(vx2_3, vbeta_8, vbeta_6);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_4);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_4);
    vq_2 = xnn_fmadd_f32(vx2_2, vq_2, vbeta_4);
    vq_3 = xnn_fmadd_f32(vx2_3, vq_3, vbeta_4);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_2);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_2);
    vq_2 = xnn_fmadd_f32(vx2_2, vq_2, vbeta_2);
    vq_3 = xnn_fmadd_f32(vx2_3, vq_3, vbeta_2);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vone);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vone);
    vq_2 = xnn_fmadd_f32(vx2_2, vq_2, vone);
    vq_3 = xnn_fmadd_f32(vx2_3, vq_3, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy_0 = xnn_div_f32(vp_0, vq_0);
    const xnn_simd_f32_t vy_1 = xnn_div_f32(vp_1, vq_1);
    const xnn_simd_f32_t vy_2 = xnn_div_f32(vp_2, vq_2);
    const xnn_simd_f32_t vy_3 = xnn_div_f32(vp_3, vq_3);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_2);
    xnn_storeu_f32(output + 3 * xnn_simd_size_f32, vy_3);
    output += 128;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Clamp the inputs to the interpolation range.
    vx = xnn_min_f32(vmax_x, vx);
    vx = xnn_max_f32(vmin_x, vx);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_9, valpha_7);
    vp = xnn_fmadd_f32(vx2, vp, valpha_5);
    vp = xnn_fmadd_f32(vx2, vp, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, vone);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_8, vbeta_6);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_4);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    // Clamp the inputs to the interpolation range.
    vx = xnn_min_f32(vmax_x, vx);
    vx = xnn_max_f32(vmin_x, vx);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_9, valpha_7);
    vp = xnn_fmadd_f32(vx2, vp, valpha_5);
    vp = xnn_fmadd_f32(vx2, vp, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, vone);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_8, vbeta_6);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_4);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}
