// Auto-generated file. Do not edit!
//   Template: src/f32-vgelu/rational-12-10.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "xnnpack/simd/f32-scalar.h"

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"


void xnn_f32_vgelu_ukernel__scalar_rational_12_10_div_u1(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  // Cap the inputs to this value as `erf(x/sqrt(2))` will always be `+/-1.0f`
  // beyond this point. This value is chosen as the first floating point
  // number as of which the interpolation returns +/-1.0f.
  #if XNN_SIMD_HAS_NATIVE_FMA || (XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR)
    XNN_SIMD_CONST_F32(vmax_abs_x, 5.1638283730e+00f);
  #else
    XNN_SIMD_CONST_F32(vmax_abs_x, 5.1158981323e+00f);
  #endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  XNN_SIMD_CONST_F32(valpha_1, 7.9788452387e-01f);
  XNN_SIMD_CONST_F32(valpha_3, 6.6972173750e-02f);
  XNN_SIMD_CONST_F32(valpha_5, 9.3065137044e-03f);
  XNN_SIMD_CONST_F32(valpha_7, 3.2973114867e-04f);
  XNN_SIMD_CONST_F32(valpha_9, 1.2609783880e-05f);
  XNN_SIMD_CONST_F32(valpha_11, 4.5835321316e-08f);

  // The monomial coefficients of the denominator polynomial (even).
  // XNN_SIMD_CONST_F32(vbeta_0, 1.0f);
  XNN_SIMD_CONST_F32(vbeta_2, 2.5060352683e-01f);
  XNN_SIMD_CONST_F32(vbeta_4, 2.8431978077e-02f);
  XNN_SIMD_CONST_F32(vbeta_6, 1.8622842617e-03f);
  XNN_SIMD_CONST_F32(vbeta_8, 7.2267655923e-05f);
  XNN_SIMD_CONST_F32(vbeta_10, 1.1988805682e-06f);

  XNN_SIMD_CONST_F32(vone, 1.0f);
  XNN_SIMD_CONST_F32(vhalf, 0.5f);

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx_orig = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Clamp the inputs to the interpolation range.
    xnn_simd_f32_t vx = xnn_min_f32(vmax_abs_x, vx_orig);
    vx = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_11, valpha_9);
    vp = xnn_fmadd_f32(vx2, vp, valpha_7);
    vp = xnn_fmadd_f32(vx2, vp, valpha_5);
    vp = xnn_fmadd_f32(vx2, vp, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, valpha_1);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_10, vbeta_8);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_6);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_4);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vone);

    // Divide the numerator by the denominator and add one
    const xnn_simd_f32_t verf =  xnn_div_f32(vp, vq);

    // Add one to the rational interpolant, and multiply by 0.5 times the
    // original input.
    const xnn_simd_f32_t vy = xnn_mul_f32(xnn_mul_f32(vx_orig, vhalf),
                                          xnn_add_f32(verf, vone));

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
}

void xnn_f32_vgelu_ukernel__scalar_rational_12_10_div_u2(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  // Cap the inputs to this value as `erf(x/sqrt(2))` will always be `+/-1.0f`
  // beyond this point. This value is chosen as the first floating point
  // number as of which the interpolation returns +/-1.0f.
  #if XNN_SIMD_HAS_NATIVE_FMA || (XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR)
    XNN_SIMD_CONST_F32(vmax_abs_x, 5.1638283730e+00f);
  #else
    XNN_SIMD_CONST_F32(vmax_abs_x, 5.1158981323e+00f);
  #endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  XNN_SIMD_CONST_F32(valpha_1, 7.9788452387e-01f);
  XNN_SIMD_CONST_F32(valpha_3, 6.6972173750e-02f);
  XNN_SIMD_CONST_F32(valpha_5, 9.3065137044e-03f);
  XNN_SIMD_CONST_F32(valpha_7, 3.2973114867e-04f);
  XNN_SIMD_CONST_F32(valpha_9, 1.2609783880e-05f);
  XNN_SIMD_CONST_F32(valpha_11, 4.5835321316e-08f);

  // The monomial coefficients of the denominator polynomial (even).
  // XNN_SIMD_CONST_F32(vbeta_0, 1.0f);
  XNN_SIMD_CONST_F32(vbeta_2, 2.5060352683e-01f);
  XNN_SIMD_CONST_F32(vbeta_4, 2.8431978077e-02f);
  XNN_SIMD_CONST_F32(vbeta_6, 1.8622842617e-03f);
  XNN_SIMD_CONST_F32(vbeta_8, 7.2267655923e-05f);
  XNN_SIMD_CONST_F32(vbeta_10, 1.1988805682e-06f);

  XNN_SIMD_CONST_F32(vone, 1.0f);
  XNN_SIMD_CONST_F32(vhalf, 0.5f);

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    const xnn_simd_f32_t vx_orig_0 = xnn_loadu_f32(input);
    const xnn_simd_f32_t vx_orig_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 2;

    // Clamp the inputs to the interpolation range.
    xnn_simd_f32_t vx_0 = xnn_min_f32(vmax_abs_x, vx_orig_0);
    xnn_simd_f32_t vx_1 = xnn_min_f32(vmax_abs_x, vx_orig_1);
    vx_0 = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx_0);
    vx_1 = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx_1);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2_0 = xnn_mul_f32(vx_0, vx_0);
    const xnn_simd_f32_t vx2_1 = xnn_mul_f32(vx_1, vx_1);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx2_0, valpha_11, valpha_9);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx2_1, valpha_11, valpha_9);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_7);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_7);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_5);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_5);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_3);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_3);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_1);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_1);
    vp_0 = xnn_mul_f32(vx_0, vp_0);
    vp_1 = xnn_mul_f32(vx_1, vp_1);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vx2_0, vbeta_10, vbeta_8);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vx2_1, vbeta_10, vbeta_8);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_6);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_6);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_4);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_4);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_2);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_2);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vone);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t verf_0 = xnn_div_f32(vp_0, vq_0);
    const xnn_simd_f32_t verf_1 = xnn_div_f32(vp_1, vq_1);

    // Add one to the rational interpolant, and multiply by 0.5 times the
    // original input.
    const xnn_simd_f32_t vy_0 = xnn_mul_f32(xnn_mul_f32(vx_orig_0, vhalf),
                                        xnn_add_f32(verf_0, vone));
    const xnn_simd_f32_t vy_1 = xnn_mul_f32(xnn_mul_f32(vx_orig_1, vhalf),
                                        xnn_add_f32(verf_1, vone));

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    output += 2;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx_orig = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Clamp the inputs to the interpolation range.
    xnn_simd_f32_t vx = xnn_min_f32(vmax_abs_x, vx_orig);
    vx = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_11, valpha_9);
    vp = xnn_fmadd_f32(vx2, vp, valpha_7);
    vp = xnn_fmadd_f32(vx2, vp, valpha_5);
    vp = xnn_fmadd_f32(vx2, vp, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, valpha_1);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_10, vbeta_8);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_6);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_4);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vone);

    // Divide the numerator by the denominator and add one
    const xnn_simd_f32_t verf =  xnn_div_f32(vp, vq);

    // Add one to the rational interpolant, and multiply by 0.5 times the
    // original input.
    const xnn_simd_f32_t vy = xnn_mul_f32(xnn_mul_f32(vx_orig, vhalf),
                                          xnn_add_f32(verf, vone));

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
}

void xnn_f32_vgelu_ukernel__scalar_rational_12_10_div_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  // Cap the inputs to this value as `erf(x/sqrt(2))` will always be `+/-1.0f`
  // beyond this point. This value is chosen as the first floating point
  // number as of which the interpolation returns +/-1.0f.
  #if XNN_SIMD_HAS_NATIVE_FMA || (XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR)
    XNN_SIMD_CONST_F32(vmax_abs_x, 5.1638283730e+00f);
  #else
    XNN_SIMD_CONST_F32(vmax_abs_x, 5.1158981323e+00f);
  #endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  XNN_SIMD_CONST_F32(valpha_1, 7.9788452387e-01f);
  XNN_SIMD_CONST_F32(valpha_3, 6.6972173750e-02f);
  XNN_SIMD_CONST_F32(valpha_5, 9.3065137044e-03f);
  XNN_SIMD_CONST_F32(valpha_7, 3.2973114867e-04f);
  XNN_SIMD_CONST_F32(valpha_9, 1.2609783880e-05f);
  XNN_SIMD_CONST_F32(valpha_11, 4.5835321316e-08f);

  // The monomial coefficients of the denominator polynomial (even).
  // XNN_SIMD_CONST_F32(vbeta_0, 1.0f);
  XNN_SIMD_CONST_F32(vbeta_2, 2.5060352683e-01f);
  XNN_SIMD_CONST_F32(vbeta_4, 2.8431978077e-02f);
  XNN_SIMD_CONST_F32(vbeta_6, 1.8622842617e-03f);
  XNN_SIMD_CONST_F32(vbeta_8, 7.2267655923e-05f);
  XNN_SIMD_CONST_F32(vbeta_10, 1.1988805682e-06f);

  XNN_SIMD_CONST_F32(vone, 1.0f);
  XNN_SIMD_CONST_F32(vhalf, 0.5f);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const xnn_simd_f32_t vx_orig_0 = xnn_loadu_f32(input);
    const xnn_simd_f32_t vx_orig_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    const xnn_simd_f32_t vx_orig_2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    const xnn_simd_f32_t vx_orig_3 = xnn_loadu_f32(input + 3 * xnn_simd_size_f32);
    input += 4;

    // Clamp the inputs to the interpolation range.
    xnn_simd_f32_t vx_0 = xnn_min_f32(vmax_abs_x, vx_orig_0);
    xnn_simd_f32_t vx_1 = xnn_min_f32(vmax_abs_x, vx_orig_1);
    xnn_simd_f32_t vx_2 = xnn_min_f32(vmax_abs_x, vx_orig_2);
    xnn_simd_f32_t vx_3 = xnn_min_f32(vmax_abs_x, vx_orig_3);
    vx_0 = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx_0);
    vx_1 = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx_1);
    vx_2 = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx_2);
    vx_3 = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx_3);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2_0 = xnn_mul_f32(vx_0, vx_0);
    const xnn_simd_f32_t vx2_1 = xnn_mul_f32(vx_1, vx_1);
    const xnn_simd_f32_t vx2_2 = xnn_mul_f32(vx_2, vx_2);
    const xnn_simd_f32_t vx2_3 = xnn_mul_f32(vx_3, vx_3);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx2_0, valpha_11, valpha_9);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx2_1, valpha_11, valpha_9);
    xnn_simd_f32_t vp_2 = xnn_fmadd_f32(vx2_2, valpha_11, valpha_9);
    xnn_simd_f32_t vp_3 = xnn_fmadd_f32(vx2_3, valpha_11, valpha_9);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_7);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_7);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, valpha_7);
    vp_3 = xnn_fmadd_f32(vx2_3, vp_3, valpha_7);
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
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vx2_0, vbeta_10, vbeta_8);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vx2_1, vbeta_10, vbeta_8);
    xnn_simd_f32_t vq_2 = xnn_fmadd_f32(vx2_2, vbeta_10, vbeta_8);
    xnn_simd_f32_t vq_3 = xnn_fmadd_f32(vx2_3, vbeta_10, vbeta_8);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_6);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_6);
    vq_2 = xnn_fmadd_f32(vx2_2, vq_2, vbeta_6);
    vq_3 = xnn_fmadd_f32(vx2_3, vq_3, vbeta_6);
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
    const xnn_simd_f32_t verf_0 = xnn_div_f32(vp_0, vq_0);
    const xnn_simd_f32_t verf_1 = xnn_div_f32(vp_1, vq_1);
    const xnn_simd_f32_t verf_2 = xnn_div_f32(vp_2, vq_2);
    const xnn_simd_f32_t verf_3 = xnn_div_f32(vp_3, vq_3);

    // Add one to the rational interpolant, and multiply by 0.5 times the
    // original input.
    const xnn_simd_f32_t vy_0 = xnn_mul_f32(xnn_mul_f32(vx_orig_0, vhalf),
                                        xnn_add_f32(verf_0, vone));
    const xnn_simd_f32_t vy_1 = xnn_mul_f32(xnn_mul_f32(vx_orig_1, vhalf),
                                        xnn_add_f32(verf_1, vone));
    const xnn_simd_f32_t vy_2 = xnn_mul_f32(xnn_mul_f32(vx_orig_2, vhalf),
                                        xnn_add_f32(verf_2, vone));
    const xnn_simd_f32_t vy_3 = xnn_mul_f32(xnn_mul_f32(vx_orig_3, vhalf),
                                        xnn_add_f32(verf_3, vone));

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_2);
    xnn_storeu_f32(output + 3 * xnn_simd_size_f32, vy_3);
    output += 4;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx_orig = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Clamp the inputs to the interpolation range.
    xnn_simd_f32_t vx = xnn_min_f32(vmax_abs_x, vx_orig);
    vx = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_11, valpha_9);
    vp = xnn_fmadd_f32(vx2, vp, valpha_7);
    vp = xnn_fmadd_f32(vx2, vp, valpha_5);
    vp = xnn_fmadd_f32(vx2, vp, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, valpha_1);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_10, vbeta_8);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_6);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_4);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vone);

    // Divide the numerator by the denominator and add one
    const xnn_simd_f32_t verf =  xnn_div_f32(vp, vq);

    // Add one to the rational interpolant, and multiply by 0.5 times the
    // original input.
    const xnn_simd_f32_t vy = xnn_mul_f32(xnn_mul_f32(vx_orig, vhalf),
                                          xnn_add_f32(verf, vone));

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
}

void xnn_f32_vgelu_ukernel__scalar_rational_12_10_div_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  // Cap the inputs to this value as `erf(x/sqrt(2))` will always be `+/-1.0f`
  // beyond this point. This value is chosen as the first floating point
  // number as of which the interpolation returns +/-1.0f.
  #if XNN_SIMD_HAS_NATIVE_FMA || (XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR)
    XNN_SIMD_CONST_F32(vmax_abs_x, 5.1638283730e+00f);
  #else
    XNN_SIMD_CONST_F32(vmax_abs_x, 5.1158981323e+00f);
  #endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  XNN_SIMD_CONST_F32(valpha_1, 7.9788452387e-01f);
  XNN_SIMD_CONST_F32(valpha_3, 6.6972173750e-02f);
  XNN_SIMD_CONST_F32(valpha_5, 9.3065137044e-03f);
  XNN_SIMD_CONST_F32(valpha_7, 3.2973114867e-04f);
  XNN_SIMD_CONST_F32(valpha_9, 1.2609783880e-05f);
  XNN_SIMD_CONST_F32(valpha_11, 4.5835321316e-08f);

  // The monomial coefficients of the denominator polynomial (even).
  // XNN_SIMD_CONST_F32(vbeta_0, 1.0f);
  XNN_SIMD_CONST_F32(vbeta_2, 2.5060352683e-01f);
  XNN_SIMD_CONST_F32(vbeta_4, 2.8431978077e-02f);
  XNN_SIMD_CONST_F32(vbeta_6, 1.8622842617e-03f);
  XNN_SIMD_CONST_F32(vbeta_8, 7.2267655923e-05f);
  XNN_SIMD_CONST_F32(vbeta_10, 1.1988805682e-06f);

  XNN_SIMD_CONST_F32(vone, 1.0f);
  XNN_SIMD_CONST_F32(vhalf, 0.5f);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const xnn_simd_f32_t vx_orig_0 = xnn_loadu_f32(input);
    const xnn_simd_f32_t vx_orig_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    const xnn_simd_f32_t vx_orig_2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    const xnn_simd_f32_t vx_orig_3 = xnn_loadu_f32(input + 3 * xnn_simd_size_f32);
    const xnn_simd_f32_t vx_orig_4 = xnn_loadu_f32(input + 4 * xnn_simd_size_f32);
    const xnn_simd_f32_t vx_orig_5 = xnn_loadu_f32(input + 5 * xnn_simd_size_f32);
    const xnn_simd_f32_t vx_orig_6 = xnn_loadu_f32(input + 6 * xnn_simd_size_f32);
    const xnn_simd_f32_t vx_orig_7 = xnn_loadu_f32(input + 7 * xnn_simd_size_f32);
    input += 8;

    // Clamp the inputs to the interpolation range.
    xnn_simd_f32_t vx_0 = xnn_min_f32(vmax_abs_x, vx_orig_0);
    xnn_simd_f32_t vx_1 = xnn_min_f32(vmax_abs_x, vx_orig_1);
    xnn_simd_f32_t vx_2 = xnn_min_f32(vmax_abs_x, vx_orig_2);
    xnn_simd_f32_t vx_3 = xnn_min_f32(vmax_abs_x, vx_orig_3);
    xnn_simd_f32_t vx_4 = xnn_min_f32(vmax_abs_x, vx_orig_4);
    xnn_simd_f32_t vx_5 = xnn_min_f32(vmax_abs_x, vx_orig_5);
    xnn_simd_f32_t vx_6 = xnn_min_f32(vmax_abs_x, vx_orig_6);
    xnn_simd_f32_t vx_7 = xnn_min_f32(vmax_abs_x, vx_orig_7);
    vx_0 = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx_0);
    vx_1 = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx_1);
    vx_2 = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx_2);
    vx_3 = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx_3);
    vx_4 = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx_4);
    vx_5 = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx_5);
    vx_6 = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx_6);
    vx_7 = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx_7);

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
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx2_0, valpha_11, valpha_9);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx2_1, valpha_11, valpha_9);
    xnn_simd_f32_t vp_2 = xnn_fmadd_f32(vx2_2, valpha_11, valpha_9);
    xnn_simd_f32_t vp_3 = xnn_fmadd_f32(vx2_3, valpha_11, valpha_9);
    xnn_simd_f32_t vp_4 = xnn_fmadd_f32(vx2_4, valpha_11, valpha_9);
    xnn_simd_f32_t vp_5 = xnn_fmadd_f32(vx2_5, valpha_11, valpha_9);
    xnn_simd_f32_t vp_6 = xnn_fmadd_f32(vx2_6, valpha_11, valpha_9);
    xnn_simd_f32_t vp_7 = xnn_fmadd_f32(vx2_7, valpha_11, valpha_9);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_7);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_7);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, valpha_7);
    vp_3 = xnn_fmadd_f32(vx2_3, vp_3, valpha_7);
    vp_4 = xnn_fmadd_f32(vx2_4, vp_4, valpha_7);
    vp_5 = xnn_fmadd_f32(vx2_5, vp_5, valpha_7);
    vp_6 = xnn_fmadd_f32(vx2_6, vp_6, valpha_7);
    vp_7 = xnn_fmadd_f32(vx2_7, vp_7, valpha_7);
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
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vx2_0, vbeta_10, vbeta_8);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vx2_1, vbeta_10, vbeta_8);
    xnn_simd_f32_t vq_2 = xnn_fmadd_f32(vx2_2, vbeta_10, vbeta_8);
    xnn_simd_f32_t vq_3 = xnn_fmadd_f32(vx2_3, vbeta_10, vbeta_8);
    xnn_simd_f32_t vq_4 = xnn_fmadd_f32(vx2_4, vbeta_10, vbeta_8);
    xnn_simd_f32_t vq_5 = xnn_fmadd_f32(vx2_5, vbeta_10, vbeta_8);
    xnn_simd_f32_t vq_6 = xnn_fmadd_f32(vx2_6, vbeta_10, vbeta_8);
    xnn_simd_f32_t vq_7 = xnn_fmadd_f32(vx2_7, vbeta_10, vbeta_8);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_6);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_6);
    vq_2 = xnn_fmadd_f32(vx2_2, vq_2, vbeta_6);
    vq_3 = xnn_fmadd_f32(vx2_3, vq_3, vbeta_6);
    vq_4 = xnn_fmadd_f32(vx2_4, vq_4, vbeta_6);
    vq_5 = xnn_fmadd_f32(vx2_5, vq_5, vbeta_6);
    vq_6 = xnn_fmadd_f32(vx2_6, vq_6, vbeta_6);
    vq_7 = xnn_fmadd_f32(vx2_7, vq_7, vbeta_6);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_4);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_4);
    vq_2 = xnn_fmadd_f32(vx2_2, vq_2, vbeta_4);
    vq_3 = xnn_fmadd_f32(vx2_3, vq_3, vbeta_4);
    vq_4 = xnn_fmadd_f32(vx2_4, vq_4, vbeta_4);
    vq_5 = xnn_fmadd_f32(vx2_5, vq_5, vbeta_4);
    vq_6 = xnn_fmadd_f32(vx2_6, vq_6, vbeta_4);
    vq_7 = xnn_fmadd_f32(vx2_7, vq_7, vbeta_4);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_2);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_2);
    vq_2 = xnn_fmadd_f32(vx2_2, vq_2, vbeta_2);
    vq_3 = xnn_fmadd_f32(vx2_3, vq_3, vbeta_2);
    vq_4 = xnn_fmadd_f32(vx2_4, vq_4, vbeta_2);
    vq_5 = xnn_fmadd_f32(vx2_5, vq_5, vbeta_2);
    vq_6 = xnn_fmadd_f32(vx2_6, vq_6, vbeta_2);
    vq_7 = xnn_fmadd_f32(vx2_7, vq_7, vbeta_2);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vone);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vone);
    vq_2 = xnn_fmadd_f32(vx2_2, vq_2, vone);
    vq_3 = xnn_fmadd_f32(vx2_3, vq_3, vone);
    vq_4 = xnn_fmadd_f32(vx2_4, vq_4, vone);
    vq_5 = xnn_fmadd_f32(vx2_5, vq_5, vone);
    vq_6 = xnn_fmadd_f32(vx2_6, vq_6, vone);
    vq_7 = xnn_fmadd_f32(vx2_7, vq_7, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t verf_0 = xnn_div_f32(vp_0, vq_0);
    const xnn_simd_f32_t verf_1 = xnn_div_f32(vp_1, vq_1);
    const xnn_simd_f32_t verf_2 = xnn_div_f32(vp_2, vq_2);
    const xnn_simd_f32_t verf_3 = xnn_div_f32(vp_3, vq_3);
    const xnn_simd_f32_t verf_4 = xnn_div_f32(vp_4, vq_4);
    const xnn_simd_f32_t verf_5 = xnn_div_f32(vp_5, vq_5);
    const xnn_simd_f32_t verf_6 = xnn_div_f32(vp_6, vq_6);
    const xnn_simd_f32_t verf_7 = xnn_div_f32(vp_7, vq_7);

    // Add one to the rational interpolant, and multiply by 0.5 times the
    // original input.
    const xnn_simd_f32_t vy_0 = xnn_mul_f32(xnn_mul_f32(vx_orig_0, vhalf),
                                        xnn_add_f32(verf_0, vone));
    const xnn_simd_f32_t vy_1 = xnn_mul_f32(xnn_mul_f32(vx_orig_1, vhalf),
                                        xnn_add_f32(verf_1, vone));
    const xnn_simd_f32_t vy_2 = xnn_mul_f32(xnn_mul_f32(vx_orig_2, vhalf),
                                        xnn_add_f32(verf_2, vone));
    const xnn_simd_f32_t vy_3 = xnn_mul_f32(xnn_mul_f32(vx_orig_3, vhalf),
                                        xnn_add_f32(verf_3, vone));
    const xnn_simd_f32_t vy_4 = xnn_mul_f32(xnn_mul_f32(vx_orig_4, vhalf),
                                        xnn_add_f32(verf_4, vone));
    const xnn_simd_f32_t vy_5 = xnn_mul_f32(xnn_mul_f32(vx_orig_5, vhalf),
                                        xnn_add_f32(verf_5, vone));
    const xnn_simd_f32_t vy_6 = xnn_mul_f32(xnn_mul_f32(vx_orig_6, vhalf),
                                        xnn_add_f32(verf_6, vone));
    const xnn_simd_f32_t vy_7 = xnn_mul_f32(xnn_mul_f32(vx_orig_7, vhalf),
                                        xnn_add_f32(verf_7, vone));

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
    const xnn_simd_f32_t vx_orig = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Clamp the inputs to the interpolation range.
    xnn_simd_f32_t vx = xnn_min_f32(vmax_abs_x, vx_orig);
    vx = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_11, valpha_9);
    vp = xnn_fmadd_f32(vx2, vp, valpha_7);
    vp = xnn_fmadd_f32(vx2, vp, valpha_5);
    vp = xnn_fmadd_f32(vx2, vp, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, valpha_1);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_10, vbeta_8);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_6);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_4);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vone);

    // Divide the numerator by the denominator and add one
    const xnn_simd_f32_t verf =  xnn_div_f32(vp, vq);

    // Add one to the rational interpolant, and multiply by 0.5 times the
    // original input.
    const xnn_simd_f32_t vy = xnn_mul_f32(xnn_mul_f32(vx_orig, vhalf),
                                          xnn_add_f32(verf, vone));

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
}
