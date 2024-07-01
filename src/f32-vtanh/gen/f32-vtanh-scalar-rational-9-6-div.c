// Auto-generated file. Do not edit!
//   Template: src/f32-vtanh/rational-9-6.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/simd/f32-scalar.h"

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"


void xnn_f32_vtanh_ukernel__scalar_rational_9_6_div_u1(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  // Cap the inputs to this value as `tanh(x)` will always be `+/-1.0f` beyond
  // this point. This value is chosen as the first floating point number as of
  // which the interpolation returns 1.0f.
  #if XNN_SIMD_HAS_NATIVE_FMA
    XNN_SIMD_CONST_F32(vmax_x, 7.646893501282f);
    XNN_SIMD_CONST_F32(vmin_x, -7.646893501282f);
  #else
    XNN_SIMD_CONST_F32(vmax_x, 7.623543739319f);
    XNN_SIMD_CONST_F32(vmin_x, -7.623543739319f);
  #endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  XNN_SIMD_CONST_F32(valpha_1, -9.022999554873e-03f);
  XNN_SIMD_CONST_F32(valpha_3, -1.146968104877e-03f);
  XNN_SIMD_CONST_F32(valpha_5, -2.432360815874e-05f);
  XNN_SIMD_CONST_F32(valpha_7, -6.458659385089e-08f);
  XNN_SIMD_CONST_F32(valpha_9, 5.535878699892e-11f);

  // The monomial coefficients of the denominator polynomial (even).
  XNN_SIMD_CONST_F32(vbeta_0, -9.023001417518e-03f);
  XNN_SIMD_CONST_F32(vbeta_2, -4.154618829489e-03f);
  XNN_SIMD_CONST_F32(vbeta_4, -2.061512641376e-04f);
  XNN_SIMD_CONST_F32(vbeta_6, -1.774490101525e-06f);


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
    vp = xnn_fmadd_f32(vx2, vp, valpha_1);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_6, vbeta_4);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_0);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
}

void xnn_f32_vtanh_ukernel__scalar_rational_9_6_div_u2(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  // Cap the inputs to this value as `tanh(x)` will always be `+/-1.0f` beyond
  // this point. This value is chosen as the first floating point number as of
  // which the interpolation returns 1.0f.
  #if XNN_SIMD_HAS_NATIVE_FMA
    XNN_SIMD_CONST_F32(vmax_x, 7.646893501282f);
    XNN_SIMD_CONST_F32(vmin_x, -7.646893501282f);
  #else
    XNN_SIMD_CONST_F32(vmax_x, 7.623543739319f);
    XNN_SIMD_CONST_F32(vmin_x, -7.623543739319f);
  #endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  XNN_SIMD_CONST_F32(valpha_1, -9.022999554873e-03f);
  XNN_SIMD_CONST_F32(valpha_3, -1.146968104877e-03f);
  XNN_SIMD_CONST_F32(valpha_5, -2.432360815874e-05f);
  XNN_SIMD_CONST_F32(valpha_7, -6.458659385089e-08f);
  XNN_SIMD_CONST_F32(valpha_9, 5.535878699892e-11f);

  // The monomial coefficients of the denominator polynomial (even).
  XNN_SIMD_CONST_F32(vbeta_0, -9.023001417518e-03f);
  XNN_SIMD_CONST_F32(vbeta_2, -4.154618829489e-03f);
  XNN_SIMD_CONST_F32(vbeta_4, -2.061512641376e-04f);
  XNN_SIMD_CONST_F32(vbeta_6, -1.774490101525e-06f);


  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 2;

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
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_1);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_1);
    vp_0 = xnn_mul_f32(vx_0, vp_0);
    vp_1 = xnn_mul_f32(vx_1, vp_1);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vx2_0, vbeta_6, vbeta_4);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vx2_1, vbeta_6, vbeta_4);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_2);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_2);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_0);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_0);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy_0 = xnn_div_f32(vp_0, vq_0);
    const xnn_simd_f32_t vy_1 = xnn_div_f32(vp_1, vq_1);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    output += 2;
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
    vp = xnn_fmadd_f32(vx2, vp, valpha_1);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_6, vbeta_4);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_0);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
}

void xnn_f32_vtanh_ukernel__scalar_rational_9_6_div_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  // Cap the inputs to this value as `tanh(x)` will always be `+/-1.0f` beyond
  // this point. This value is chosen as the first floating point number as of
  // which the interpolation returns 1.0f.
  #if XNN_SIMD_HAS_NATIVE_FMA
    XNN_SIMD_CONST_F32(vmax_x, 7.646893501282f);
    XNN_SIMD_CONST_F32(vmin_x, -7.646893501282f);
  #else
    XNN_SIMD_CONST_F32(vmax_x, 7.623543739319f);
    XNN_SIMD_CONST_F32(vmin_x, -7.623543739319f);
  #endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  XNN_SIMD_CONST_F32(valpha_1, -9.022999554873e-03f);
  XNN_SIMD_CONST_F32(valpha_3, -1.146968104877e-03f);
  XNN_SIMD_CONST_F32(valpha_5, -2.432360815874e-05f);
  XNN_SIMD_CONST_F32(valpha_7, -6.458659385089e-08f);
  XNN_SIMD_CONST_F32(valpha_9, 5.535878699892e-11f);

  // The monomial coefficients of the denominator polynomial (even).
  XNN_SIMD_CONST_F32(vbeta_0, -9.023001417518e-03f);
  XNN_SIMD_CONST_F32(vbeta_2, -4.154618829489e-03f);
  XNN_SIMD_CONST_F32(vbeta_4, -2.061512641376e-04f);
  XNN_SIMD_CONST_F32(vbeta_6, -1.774490101525e-06f);


  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_3 = xnn_loadu_f32(input + 3 * xnn_simd_size_f32);
    input += 4;

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
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_1);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_1);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, valpha_1);
    vp_3 = xnn_fmadd_f32(vx2_3, vp_3, valpha_1);
    vp_0 = xnn_mul_f32(vx_0, vp_0);
    vp_1 = xnn_mul_f32(vx_1, vp_1);
    vp_2 = xnn_mul_f32(vx_2, vp_2);
    vp_3 = xnn_mul_f32(vx_3, vp_3);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vx2_0, vbeta_6, vbeta_4);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vx2_1, vbeta_6, vbeta_4);
    xnn_simd_f32_t vq_2 = xnn_fmadd_f32(vx2_2, vbeta_6, vbeta_4);
    xnn_simd_f32_t vq_3 = xnn_fmadd_f32(vx2_3, vbeta_6, vbeta_4);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_2);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_2);
    vq_2 = xnn_fmadd_f32(vx2_2, vq_2, vbeta_2);
    vq_3 = xnn_fmadd_f32(vx2_3, vq_3, vbeta_2);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_0);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_0);
    vq_2 = xnn_fmadd_f32(vx2_2, vq_2, vbeta_0);
    vq_3 = xnn_fmadd_f32(vx2_3, vq_3, vbeta_0);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy_0 = xnn_div_f32(vp_0, vq_0);
    const xnn_simd_f32_t vy_1 = xnn_div_f32(vp_1, vq_1);
    const xnn_simd_f32_t vy_2 = xnn_div_f32(vp_2, vq_2);
    const xnn_simd_f32_t vy_3 = xnn_div_f32(vp_3, vq_3);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_2);
    xnn_storeu_f32(output + 3 * xnn_simd_size_f32, vy_3);
    output += 4;
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
    vp = xnn_fmadd_f32(vx2, vp, valpha_1);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_6, vbeta_4);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_0);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
}

void xnn_f32_vtanh_ukernel__scalar_rational_9_6_div_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  // Cap the inputs to this value as `tanh(x)` will always be `+/-1.0f` beyond
  // this point. This value is chosen as the first floating point number as of
  // which the interpolation returns 1.0f.
  #if XNN_SIMD_HAS_NATIVE_FMA
    XNN_SIMD_CONST_F32(vmax_x, 7.646893501282f);
    XNN_SIMD_CONST_F32(vmin_x, -7.646893501282f);
  #else
    XNN_SIMD_CONST_F32(vmax_x, 7.623543739319f);
    XNN_SIMD_CONST_F32(vmin_x, -7.623543739319f);
  #endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  XNN_SIMD_CONST_F32(valpha_1, -9.022999554873e-03f);
  XNN_SIMD_CONST_F32(valpha_3, -1.146968104877e-03f);
  XNN_SIMD_CONST_F32(valpha_5, -2.432360815874e-05f);
  XNN_SIMD_CONST_F32(valpha_7, -6.458659385089e-08f);
  XNN_SIMD_CONST_F32(valpha_9, 5.535878699892e-11f);

  // The monomial coefficients of the denominator polynomial (even).
  XNN_SIMD_CONST_F32(vbeta_0, -9.023001417518e-03f);
  XNN_SIMD_CONST_F32(vbeta_2, -4.154618829489e-03f);
  XNN_SIMD_CONST_F32(vbeta_4, -2.061512641376e-04f);
  XNN_SIMD_CONST_F32(vbeta_6, -1.774490101525e-06f);


  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_3 = xnn_loadu_f32(input + 3 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_4 = xnn_loadu_f32(input + 4 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_5 = xnn_loadu_f32(input + 5 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_6 = xnn_loadu_f32(input + 6 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_7 = xnn_loadu_f32(input + 7 * xnn_simd_size_f32);
    input += 8;

    // Clamp the inputs to the interpolation range.
    vx_0 = xnn_min_f32(vmax_x, vx_0);
    vx_1 = xnn_min_f32(vmax_x, vx_1);
    vx_2 = xnn_min_f32(vmax_x, vx_2);
    vx_3 = xnn_min_f32(vmax_x, vx_3);
    vx_4 = xnn_min_f32(vmax_x, vx_4);
    vx_5 = xnn_min_f32(vmax_x, vx_5);
    vx_6 = xnn_min_f32(vmax_x, vx_6);
    vx_7 = xnn_min_f32(vmax_x, vx_7);
    vx_0 = xnn_max_f32(vmin_x, vx_0);
    vx_1 = xnn_max_f32(vmin_x, vx_1);
    vx_2 = xnn_max_f32(vmin_x, vx_2);
    vx_3 = xnn_max_f32(vmin_x, vx_3);
    vx_4 = xnn_max_f32(vmin_x, vx_4);
    vx_5 = xnn_max_f32(vmin_x, vx_5);
    vx_6 = xnn_max_f32(vmin_x, vx_6);
    vx_7 = xnn_max_f32(vmin_x, vx_7);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2_0 = xnn_mul_f32(vx_0, vx_0);
    const xnn_simd_f32_t vx2_1 = xnn_mul_f32(vx_1, vx_1);
    const xnn_simd_f32_t vx2_2 = xnn_mul_f32(vx_2, vx_2);
    const xnn_simd_f32_t vx2_3 = xnn_mul_f32(vx_3, vx_3);
    const xnn_simd_f32_t vx2_4 = xnn_mul_f32(vx_4, vx_4);
    const xnn_simd_f32_t vx2_5 = xnn_mul_f32(vx_5, vx_5);
    const xnn_simd_f32_t vx2_6 = xnn_mul_f32(vx_6, vx_6);
    const xnn_simd_f32_t vx2_7 = xnn_mul_f32(vx_7, vx_7);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx2_0, valpha_9, valpha_7);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx2_1, valpha_9, valpha_7);
    xnn_simd_f32_t vp_2 = xnn_fmadd_f32(vx2_2, valpha_9, valpha_7);
    xnn_simd_f32_t vp_3 = xnn_fmadd_f32(vx2_3, valpha_9, valpha_7);
    xnn_simd_f32_t vp_4 = xnn_fmadd_f32(vx2_4, valpha_9, valpha_7);
    xnn_simd_f32_t vp_5 = xnn_fmadd_f32(vx2_5, valpha_9, valpha_7);
    xnn_simd_f32_t vp_6 = xnn_fmadd_f32(vx2_6, valpha_9, valpha_7);
    xnn_simd_f32_t vp_7 = xnn_fmadd_f32(vx2_7, valpha_9, valpha_7);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_5);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_5);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, valpha_5);
    vp_3 = xnn_fmadd_f32(vx2_3, vp_3, valpha_5);
    vp_4 = xnn_fmadd_f32(vx2_4, vp_4, valpha_5);
    vp_5 = xnn_fmadd_f32(vx2_5, vp_5, valpha_5);
    vp_6 = xnn_fmadd_f32(vx2_6, vp_6, valpha_5);
    vp_7 = xnn_fmadd_f32(vx2_7, vp_7, valpha_5);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_3);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_3);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, valpha_3);
    vp_3 = xnn_fmadd_f32(vx2_3, vp_3, valpha_3);
    vp_4 = xnn_fmadd_f32(vx2_4, vp_4, valpha_3);
    vp_5 = xnn_fmadd_f32(vx2_5, vp_5, valpha_3);
    vp_6 = xnn_fmadd_f32(vx2_6, vp_6, valpha_3);
    vp_7 = xnn_fmadd_f32(vx2_7, vp_7, valpha_3);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_1);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_1);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, valpha_1);
    vp_3 = xnn_fmadd_f32(vx2_3, vp_3, valpha_1);
    vp_4 = xnn_fmadd_f32(vx2_4, vp_4, valpha_1);
    vp_5 = xnn_fmadd_f32(vx2_5, vp_5, valpha_1);
    vp_6 = xnn_fmadd_f32(vx2_6, vp_6, valpha_1);
    vp_7 = xnn_fmadd_f32(vx2_7, vp_7, valpha_1);
    vp_0 = xnn_mul_f32(vx_0, vp_0);
    vp_1 = xnn_mul_f32(vx_1, vp_1);
    vp_2 = xnn_mul_f32(vx_2, vp_2);
    vp_3 = xnn_mul_f32(vx_3, vp_3);
    vp_4 = xnn_mul_f32(vx_4, vp_4);
    vp_5 = xnn_mul_f32(vx_5, vp_5);
    vp_6 = xnn_mul_f32(vx_6, vp_6);
    vp_7 = xnn_mul_f32(vx_7, vp_7);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vx2_0, vbeta_6, vbeta_4);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vx2_1, vbeta_6, vbeta_4);
    xnn_simd_f32_t vq_2 = xnn_fmadd_f32(vx2_2, vbeta_6, vbeta_4);
    xnn_simd_f32_t vq_3 = xnn_fmadd_f32(vx2_3, vbeta_6, vbeta_4);
    xnn_simd_f32_t vq_4 = xnn_fmadd_f32(vx2_4, vbeta_6, vbeta_4);
    xnn_simd_f32_t vq_5 = xnn_fmadd_f32(vx2_5, vbeta_6, vbeta_4);
    xnn_simd_f32_t vq_6 = xnn_fmadd_f32(vx2_6, vbeta_6, vbeta_4);
    xnn_simd_f32_t vq_7 = xnn_fmadd_f32(vx2_7, vbeta_6, vbeta_4);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_2);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_2);
    vq_2 = xnn_fmadd_f32(vx2_2, vq_2, vbeta_2);
    vq_3 = xnn_fmadd_f32(vx2_3, vq_3, vbeta_2);
    vq_4 = xnn_fmadd_f32(vx2_4, vq_4, vbeta_2);
    vq_5 = xnn_fmadd_f32(vx2_5, vq_5, vbeta_2);
    vq_6 = xnn_fmadd_f32(vx2_6, vq_6, vbeta_2);
    vq_7 = xnn_fmadd_f32(vx2_7, vq_7, vbeta_2);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_0);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_0);
    vq_2 = xnn_fmadd_f32(vx2_2, vq_2, vbeta_0);
    vq_3 = xnn_fmadd_f32(vx2_3, vq_3, vbeta_0);
    vq_4 = xnn_fmadd_f32(vx2_4, vq_4, vbeta_0);
    vq_5 = xnn_fmadd_f32(vx2_5, vq_5, vbeta_0);
    vq_6 = xnn_fmadd_f32(vx2_6, vq_6, vbeta_0);
    vq_7 = xnn_fmadd_f32(vx2_7, vq_7, vbeta_0);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy_0 = xnn_div_f32(vp_0, vq_0);
    const xnn_simd_f32_t vy_1 = xnn_div_f32(vp_1, vq_1);
    const xnn_simd_f32_t vy_2 = xnn_div_f32(vp_2, vq_2);
    const xnn_simd_f32_t vy_3 = xnn_div_f32(vp_3, vq_3);
    const xnn_simd_f32_t vy_4 = xnn_div_f32(vp_4, vq_4);
    const xnn_simd_f32_t vy_5 = xnn_div_f32(vp_5, vq_5);
    const xnn_simd_f32_t vy_6 = xnn_div_f32(vp_6, vq_6);
    const xnn_simd_f32_t vy_7 = xnn_div_f32(vp_7, vq_7);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_2);
    xnn_storeu_f32(output + 3 * xnn_simd_size_f32, vy_3);
    xnn_storeu_f32(output + 4 * xnn_simd_size_f32, vy_4);
    xnn_storeu_f32(output + 5 * xnn_simd_size_f32, vy_5);
    xnn_storeu_f32(output + 6 * xnn_simd_size_f32, vy_6);
    xnn_storeu_f32(output + 7 * xnn_simd_size_f32, vy_7);
    output += 8;
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
    vp = xnn_fmadd_f32(vx2, vp, valpha_1);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_6, vbeta_4);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_0);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
}
