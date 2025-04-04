// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vapproxgelu/rational-6-4.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/simd/f16-scalar.h"

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vapproxgelu_ukernel__scalar_rational_6_4_div_u1(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 1);

  // Cap the inputs to this value as `erf(x/sqrt(2))` will always be `+/-1.0f`
  // beyond this point. This value is chosen as the first floating point
  // number as of which the interpolation returns +/-1.0f.
#if XNN_SIMD_HAS_NATIVE_FMA
  XNN_SIMD_CONST_F16_FROM_FLOAT(vmax_abs_x, 3.26367188e+00f);
#else
  XNN_SIMD_CONST_F16_FROM_FLOAT(vmax_abs_x, 3.26953125e+00f);
#endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_1, 7.9763704538e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_3, 1.4811584353e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_5, 4.3507730588e-03f);

  // The monomial coefficients of the denominator polynomial (even).
  // XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_0, 1.0e+00f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_2, 3.5130375624e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_4, 4.0821321309e-02f);

  XNN_SIMD_CONST_F16_FROM_FLOAT(vone, 1.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vhalf, 0.5f);

  for (; batch >= xnn_simd_bytes_f16; batch -= xnn_simd_bytes_f16) {
    const xnn_simd_f16_t vx_orig = xnn_loadu_f16(input);
    input += xnn_simd_size_f16;

    // Clamp the inputs to the interpolation range.
    xnn_simd_f16_t vx = xnn_min_f16(vmax_abs_x, vx_orig);
    vx = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f16_t vx2 = xnn_mul_f16(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f16_t vp = xnn_fmadd_f16(vx2, valpha_5, valpha_3);
    vp = xnn_fmadd_f16(vx2, vp, valpha_1);
    vp = xnn_mul_f16(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f16_t vq = xnn_fmadd_f16(vx2, vbeta_4, vbeta_2);
    vq = xnn_fmadd_f16(vx2, vq, vone);

    // Divide the numerator by the denominator and add one
    const xnn_simd_f16_t verf =  xnn_div_f16(vp, vq);

    // Add one to the rational interpolant, and multiply by 0.5 times the
    // original input.
    const xnn_simd_f16_t vy = xnn_mul_f16(xnn_mul_f16(vx_orig, vhalf),
                                          xnn_add_f16(verf, vone));

    xnn_storeu_f16(output, vy);
    output += xnn_simd_size_f16;
  }
}

void xnn_f16_vapproxgelu_ukernel__scalar_rational_6_4_div_u2(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 1);

  // Cap the inputs to this value as `erf(x/sqrt(2))` will always be `+/-1.0f`
  // beyond this point. This value is chosen as the first floating point
  // number as of which the interpolation returns +/-1.0f.
#if XNN_SIMD_HAS_NATIVE_FMA
  XNN_SIMD_CONST_F16_FROM_FLOAT(vmax_abs_x, 3.26367188e+00f);
#else
  XNN_SIMD_CONST_F16_FROM_FLOAT(vmax_abs_x, 3.26953125e+00f);
#endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_1, 7.9763704538e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_3, 1.4811584353e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_5, 4.3507730588e-03f);

  // The monomial coefficients of the denominator polynomial (even).
  // XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_0, 1.0e+00f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_2, 3.5130375624e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_4, 4.0821321309e-02f);

  XNN_SIMD_CONST_F16_FROM_FLOAT(vone, 1.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vhalf, 0.5f);

  for (; batch >= 2 * sizeof(xnn_float16); batch -= 2 * sizeof(xnn_float16)) {
    const xnn_simd_f16_t vx_orig_0 = xnn_loadu_f16(input);
    const xnn_simd_f16_t vx_orig_1 = xnn_loadu_f16(input + 1 * xnn_simd_size_f16);
    input += 2;

    // Clamp the inputs to the interpolation range.
    xnn_simd_f16_t vx_0 = xnn_min_f16(vmax_abs_x, vx_orig_0);
    xnn_simd_f16_t vx_1 = xnn_min_f16(vmax_abs_x, vx_orig_1);
    vx_0 = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx_0);
    vx_1 = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx_1);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f16_t vx2_0 = xnn_mul_f16(vx_0, vx_0);
    const xnn_simd_f16_t vx2_1 = xnn_mul_f16(vx_1, vx_1);

    // Evaluate the numerator polynomial p.
    xnn_simd_f16_t vp_0 = xnn_fmadd_f16(vx2_0, valpha_5, valpha_3);
    xnn_simd_f16_t vp_1 = xnn_fmadd_f16(vx2_1, valpha_5, valpha_3);
    vp_0 = xnn_fmadd_f16(vx2_0, vp_0, valpha_1);
    vp_1 = xnn_fmadd_f16(vx2_1, vp_1, valpha_1);
    vp_0 = xnn_mul_f16(vx_0, vp_0);
    vp_1 = xnn_mul_f16(vx_1, vp_1);

    // Evaluate the denominator polynomial q.
    xnn_simd_f16_t vq_0 = xnn_fmadd_f16(vx2_0, vbeta_4, vbeta_2);
    xnn_simd_f16_t vq_1 = xnn_fmadd_f16(vx2_1, vbeta_4, vbeta_2);
    vq_0 = xnn_fmadd_f16(vx2_0, vq_0, vone);
    vq_1 = xnn_fmadd_f16(vx2_1, vq_1, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f16_t verf_0 = xnn_div_f16(vp_0, vq_0);
    const xnn_simd_f16_t verf_1 = xnn_div_f16(vp_1, vq_1);

    // Add one to the rational interpolant, and multiply by 0.5 times the
    // original input.
    const xnn_simd_f16_t vy_0 = xnn_mul_f16(xnn_mul_f16(vx_orig_0, vhalf),
                                        xnn_add_f16(verf_0, vone));
    const xnn_simd_f16_t vy_1 = xnn_mul_f16(xnn_mul_f16(vx_orig_1, vhalf),
                                        xnn_add_f16(verf_1, vone));

    xnn_storeu_f16(output, vy_0);
    xnn_storeu_f16(output + 1 * xnn_simd_size_f16, vy_1);
    output += 2;
  }
  for (; batch >= xnn_simd_bytes_f16; batch -= xnn_simd_bytes_f16) {
    const xnn_simd_f16_t vx_orig = xnn_loadu_f16(input);
    input += xnn_simd_size_f16;

    // Clamp the inputs to the interpolation range.
    xnn_simd_f16_t vx = xnn_min_f16(vmax_abs_x, vx_orig);
    vx = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f16_t vx2 = xnn_mul_f16(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f16_t vp = xnn_fmadd_f16(vx2, valpha_5, valpha_3);
    vp = xnn_fmadd_f16(vx2, vp, valpha_1);
    vp = xnn_mul_f16(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f16_t vq = xnn_fmadd_f16(vx2, vbeta_4, vbeta_2);
    vq = xnn_fmadd_f16(vx2, vq, vone);

    // Divide the numerator by the denominator and add one
    const xnn_simd_f16_t verf =  xnn_div_f16(vp, vq);

    // Add one to the rational interpolant, and multiply by 0.5 times the
    // original input.
    const xnn_simd_f16_t vy = xnn_mul_f16(xnn_mul_f16(vx_orig, vhalf),
                                          xnn_add_f16(verf, vone));

    xnn_storeu_f16(output, vy);
    output += xnn_simd_size_f16;
  }
}

void xnn_f16_vapproxgelu_ukernel__scalar_rational_6_4_div_u4(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 1);

  // Cap the inputs to this value as `erf(x/sqrt(2))` will always be `+/-1.0f`
  // beyond this point. This value is chosen as the first floating point
  // number as of which the interpolation returns +/-1.0f.
#if XNN_SIMD_HAS_NATIVE_FMA
  XNN_SIMD_CONST_F16_FROM_FLOAT(vmax_abs_x, 3.26367188e+00f);
#else
  XNN_SIMD_CONST_F16_FROM_FLOAT(vmax_abs_x, 3.26953125e+00f);
#endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_1, 7.9763704538e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_3, 1.4811584353e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_5, 4.3507730588e-03f);

  // The monomial coefficients of the denominator polynomial (even).
  // XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_0, 1.0e+00f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_2, 3.5130375624e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_4, 4.0821321309e-02f);

  XNN_SIMD_CONST_F16_FROM_FLOAT(vone, 1.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vhalf, 0.5f);

  for (; batch >= 4 * sizeof(xnn_float16); batch -= 4 * sizeof(xnn_float16)) {
    const xnn_simd_f16_t vx_orig_0 = xnn_loadu_f16(input);
    const xnn_simd_f16_t vx_orig_1 = xnn_loadu_f16(input + 1 * xnn_simd_size_f16);
    const xnn_simd_f16_t vx_orig_2 = xnn_loadu_f16(input + 2 * xnn_simd_size_f16);
    const xnn_simd_f16_t vx_orig_3 = xnn_loadu_f16(input + 3 * xnn_simd_size_f16);
    input += 4;

    // Clamp the inputs to the interpolation range.
    xnn_simd_f16_t vx_0 = xnn_min_f16(vmax_abs_x, vx_orig_0);
    xnn_simd_f16_t vx_1 = xnn_min_f16(vmax_abs_x, vx_orig_1);
    xnn_simd_f16_t vx_2 = xnn_min_f16(vmax_abs_x, vx_orig_2);
    xnn_simd_f16_t vx_3 = xnn_min_f16(vmax_abs_x, vx_orig_3);
    vx_0 = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx_0);
    vx_1 = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx_1);
    vx_2 = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx_2);
    vx_3 = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx_3);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f16_t vx2_0 = xnn_mul_f16(vx_0, vx_0);
    const xnn_simd_f16_t vx2_1 = xnn_mul_f16(vx_1, vx_1);
    const xnn_simd_f16_t vx2_2 = xnn_mul_f16(vx_2, vx_2);
    const xnn_simd_f16_t vx2_3 = xnn_mul_f16(vx_3, vx_3);

    // Evaluate the numerator polynomial p.
    xnn_simd_f16_t vp_0 = xnn_fmadd_f16(vx2_0, valpha_5, valpha_3);
    xnn_simd_f16_t vp_1 = xnn_fmadd_f16(vx2_1, valpha_5, valpha_3);
    xnn_simd_f16_t vp_2 = xnn_fmadd_f16(vx2_2, valpha_5, valpha_3);
    xnn_simd_f16_t vp_3 = xnn_fmadd_f16(vx2_3, valpha_5, valpha_3);
    vp_0 = xnn_fmadd_f16(vx2_0, vp_0, valpha_1);
    vp_1 = xnn_fmadd_f16(vx2_1, vp_1, valpha_1);
    vp_2 = xnn_fmadd_f16(vx2_2, vp_2, valpha_1);
    vp_3 = xnn_fmadd_f16(vx2_3, vp_3, valpha_1);
    vp_0 = xnn_mul_f16(vx_0, vp_0);
    vp_1 = xnn_mul_f16(vx_1, vp_1);
    vp_2 = xnn_mul_f16(vx_2, vp_2);
    vp_3 = xnn_mul_f16(vx_3, vp_3);

    // Evaluate the denominator polynomial q.
    xnn_simd_f16_t vq_0 = xnn_fmadd_f16(vx2_0, vbeta_4, vbeta_2);
    xnn_simd_f16_t vq_1 = xnn_fmadd_f16(vx2_1, vbeta_4, vbeta_2);
    xnn_simd_f16_t vq_2 = xnn_fmadd_f16(vx2_2, vbeta_4, vbeta_2);
    xnn_simd_f16_t vq_3 = xnn_fmadd_f16(vx2_3, vbeta_4, vbeta_2);
    vq_0 = xnn_fmadd_f16(vx2_0, vq_0, vone);
    vq_1 = xnn_fmadd_f16(vx2_1, vq_1, vone);
    vq_2 = xnn_fmadd_f16(vx2_2, vq_2, vone);
    vq_3 = xnn_fmadd_f16(vx2_3, vq_3, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f16_t verf_0 = xnn_div_f16(vp_0, vq_0);
    const xnn_simd_f16_t verf_1 = xnn_div_f16(vp_1, vq_1);
    const xnn_simd_f16_t verf_2 = xnn_div_f16(vp_2, vq_2);
    const xnn_simd_f16_t verf_3 = xnn_div_f16(vp_3, vq_3);

    // Add one to the rational interpolant, and multiply by 0.5 times the
    // original input.
    const xnn_simd_f16_t vy_0 = xnn_mul_f16(xnn_mul_f16(vx_orig_0, vhalf),
                                        xnn_add_f16(verf_0, vone));
    const xnn_simd_f16_t vy_1 = xnn_mul_f16(xnn_mul_f16(vx_orig_1, vhalf),
                                        xnn_add_f16(verf_1, vone));
    const xnn_simd_f16_t vy_2 = xnn_mul_f16(xnn_mul_f16(vx_orig_2, vhalf),
                                        xnn_add_f16(verf_2, vone));
    const xnn_simd_f16_t vy_3 = xnn_mul_f16(xnn_mul_f16(vx_orig_3, vhalf),
                                        xnn_add_f16(verf_3, vone));

    xnn_storeu_f16(output, vy_0);
    xnn_storeu_f16(output + 1 * xnn_simd_size_f16, vy_1);
    xnn_storeu_f16(output + 2 * xnn_simd_size_f16, vy_2);
    xnn_storeu_f16(output + 3 * xnn_simd_size_f16, vy_3);
    output += 4;
  }
  for (; batch >= xnn_simd_bytes_f16; batch -= xnn_simd_bytes_f16) {
    const xnn_simd_f16_t vx_orig = xnn_loadu_f16(input);
    input += xnn_simd_size_f16;

    // Clamp the inputs to the interpolation range.
    xnn_simd_f16_t vx = xnn_min_f16(vmax_abs_x, vx_orig);
    vx = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f16_t vx2 = xnn_mul_f16(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f16_t vp = xnn_fmadd_f16(vx2, valpha_5, valpha_3);
    vp = xnn_fmadd_f16(vx2, vp, valpha_1);
    vp = xnn_mul_f16(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f16_t vq = xnn_fmadd_f16(vx2, vbeta_4, vbeta_2);
    vq = xnn_fmadd_f16(vx2, vq, vone);

    // Divide the numerator by the denominator and add one
    const xnn_simd_f16_t verf =  xnn_div_f16(vp, vq);

    // Add one to the rational interpolant, and multiply by 0.5 times the
    // original input.
    const xnn_simd_f16_t vy = xnn_mul_f16(xnn_mul_f16(vx_orig, vhalf),
                                          xnn_add_f16(verf, vone));

    xnn_storeu_f16(output, vy);
    output += xnn_simd_size_f16;
  }
}

void xnn_f16_vapproxgelu_ukernel__scalar_rational_6_4_div_u8(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 1);

  // Cap the inputs to this value as `erf(x/sqrt(2))` will always be `+/-1.0f`
  // beyond this point. This value is chosen as the first floating point
  // number as of which the interpolation returns +/-1.0f.
#if XNN_SIMD_HAS_NATIVE_FMA
  XNN_SIMD_CONST_F16_FROM_FLOAT(vmax_abs_x, 3.26367188e+00f);
#else
  XNN_SIMD_CONST_F16_FROM_FLOAT(vmax_abs_x, 3.26953125e+00f);
#endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_1, 7.9763704538e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_3, 1.4811584353e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_5, 4.3507730588e-03f);

  // The monomial coefficients of the denominator polynomial (even).
  // XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_0, 1.0e+00f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_2, 3.5130375624e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_4, 4.0821321309e-02f);

  XNN_SIMD_CONST_F16_FROM_FLOAT(vone, 1.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vhalf, 0.5f);

  for (; batch >= 8 * sizeof(xnn_float16); batch -= 8 * sizeof(xnn_float16)) {
    const xnn_simd_f16_t vx_orig_0 = xnn_loadu_f16(input);
    const xnn_simd_f16_t vx_orig_1 = xnn_loadu_f16(input + 1 * xnn_simd_size_f16);
    const xnn_simd_f16_t vx_orig_2 = xnn_loadu_f16(input + 2 * xnn_simd_size_f16);
    const xnn_simd_f16_t vx_orig_3 = xnn_loadu_f16(input + 3 * xnn_simd_size_f16);
    const xnn_simd_f16_t vx_orig_4 = xnn_loadu_f16(input + 4 * xnn_simd_size_f16);
    const xnn_simd_f16_t vx_orig_5 = xnn_loadu_f16(input + 5 * xnn_simd_size_f16);
    const xnn_simd_f16_t vx_orig_6 = xnn_loadu_f16(input + 6 * xnn_simd_size_f16);
    const xnn_simd_f16_t vx_orig_7 = xnn_loadu_f16(input + 7 * xnn_simd_size_f16);
    input += 8;

    // Clamp the inputs to the interpolation range.
    xnn_simd_f16_t vx_0 = xnn_min_f16(vmax_abs_x, vx_orig_0);
    xnn_simd_f16_t vx_1 = xnn_min_f16(vmax_abs_x, vx_orig_1);
    xnn_simd_f16_t vx_2 = xnn_min_f16(vmax_abs_x, vx_orig_2);
    xnn_simd_f16_t vx_3 = xnn_min_f16(vmax_abs_x, vx_orig_3);
    xnn_simd_f16_t vx_4 = xnn_min_f16(vmax_abs_x, vx_orig_4);
    xnn_simd_f16_t vx_5 = xnn_min_f16(vmax_abs_x, vx_orig_5);
    xnn_simd_f16_t vx_6 = xnn_min_f16(vmax_abs_x, vx_orig_6);
    xnn_simd_f16_t vx_7 = xnn_min_f16(vmax_abs_x, vx_orig_7);
    vx_0 = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx_0);
    vx_1 = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx_1);
    vx_2 = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx_2);
    vx_3 = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx_3);
    vx_4 = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx_4);
    vx_5 = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx_5);
    vx_6 = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx_6);
    vx_7 = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx_7);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f16_t vx2_0 = xnn_mul_f16(vx_0, vx_0);
    const xnn_simd_f16_t vx2_1 = xnn_mul_f16(vx_1, vx_1);
    const xnn_simd_f16_t vx2_2 = xnn_mul_f16(vx_2, vx_2);
    const xnn_simd_f16_t vx2_3 = xnn_mul_f16(vx_3, vx_3);
    const xnn_simd_f16_t vx2_4 = xnn_mul_f16(vx_4, vx_4);
    const xnn_simd_f16_t vx2_5 = xnn_mul_f16(vx_5, vx_5);
    const xnn_simd_f16_t vx2_6 = xnn_mul_f16(vx_6, vx_6);
    const xnn_simd_f16_t vx2_7 = xnn_mul_f16(vx_7, vx_7);

    // Evaluate the numerator polynomial p.
    xnn_simd_f16_t vp_0 = xnn_fmadd_f16(vx2_0, valpha_5, valpha_3);
    xnn_simd_f16_t vp_1 = xnn_fmadd_f16(vx2_1, valpha_5, valpha_3);
    xnn_simd_f16_t vp_2 = xnn_fmadd_f16(vx2_2, valpha_5, valpha_3);
    xnn_simd_f16_t vp_3 = xnn_fmadd_f16(vx2_3, valpha_5, valpha_3);
    xnn_simd_f16_t vp_4 = xnn_fmadd_f16(vx2_4, valpha_5, valpha_3);
    xnn_simd_f16_t vp_5 = xnn_fmadd_f16(vx2_5, valpha_5, valpha_3);
    xnn_simd_f16_t vp_6 = xnn_fmadd_f16(vx2_6, valpha_5, valpha_3);
    xnn_simd_f16_t vp_7 = xnn_fmadd_f16(vx2_7, valpha_5, valpha_3);
    vp_0 = xnn_fmadd_f16(vx2_0, vp_0, valpha_1);
    vp_1 = xnn_fmadd_f16(vx2_1, vp_1, valpha_1);
    vp_2 = xnn_fmadd_f16(vx2_2, vp_2, valpha_1);
    vp_3 = xnn_fmadd_f16(vx2_3, vp_3, valpha_1);
    vp_4 = xnn_fmadd_f16(vx2_4, vp_4, valpha_1);
    vp_5 = xnn_fmadd_f16(vx2_5, vp_5, valpha_1);
    vp_6 = xnn_fmadd_f16(vx2_6, vp_6, valpha_1);
    vp_7 = xnn_fmadd_f16(vx2_7, vp_7, valpha_1);
    vp_0 = xnn_mul_f16(vx_0, vp_0);
    vp_1 = xnn_mul_f16(vx_1, vp_1);
    vp_2 = xnn_mul_f16(vx_2, vp_2);
    vp_3 = xnn_mul_f16(vx_3, vp_3);
    vp_4 = xnn_mul_f16(vx_4, vp_4);
    vp_5 = xnn_mul_f16(vx_5, vp_5);
    vp_6 = xnn_mul_f16(vx_6, vp_6);
    vp_7 = xnn_mul_f16(vx_7, vp_7);

    // Evaluate the denominator polynomial q.
    xnn_simd_f16_t vq_0 = xnn_fmadd_f16(vx2_0, vbeta_4, vbeta_2);
    xnn_simd_f16_t vq_1 = xnn_fmadd_f16(vx2_1, vbeta_4, vbeta_2);
    xnn_simd_f16_t vq_2 = xnn_fmadd_f16(vx2_2, vbeta_4, vbeta_2);
    xnn_simd_f16_t vq_3 = xnn_fmadd_f16(vx2_3, vbeta_4, vbeta_2);
    xnn_simd_f16_t vq_4 = xnn_fmadd_f16(vx2_4, vbeta_4, vbeta_2);
    xnn_simd_f16_t vq_5 = xnn_fmadd_f16(vx2_5, vbeta_4, vbeta_2);
    xnn_simd_f16_t vq_6 = xnn_fmadd_f16(vx2_6, vbeta_4, vbeta_2);
    xnn_simd_f16_t vq_7 = xnn_fmadd_f16(vx2_7, vbeta_4, vbeta_2);
    vq_0 = xnn_fmadd_f16(vx2_0, vq_0, vone);
    vq_1 = xnn_fmadd_f16(vx2_1, vq_1, vone);
    vq_2 = xnn_fmadd_f16(vx2_2, vq_2, vone);
    vq_3 = xnn_fmadd_f16(vx2_3, vq_3, vone);
    vq_4 = xnn_fmadd_f16(vx2_4, vq_4, vone);
    vq_5 = xnn_fmadd_f16(vx2_5, vq_5, vone);
    vq_6 = xnn_fmadd_f16(vx2_6, vq_6, vone);
    vq_7 = xnn_fmadd_f16(vx2_7, vq_7, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f16_t verf_0 = xnn_div_f16(vp_0, vq_0);
    const xnn_simd_f16_t verf_1 = xnn_div_f16(vp_1, vq_1);
    const xnn_simd_f16_t verf_2 = xnn_div_f16(vp_2, vq_2);
    const xnn_simd_f16_t verf_3 = xnn_div_f16(vp_3, vq_3);
    const xnn_simd_f16_t verf_4 = xnn_div_f16(vp_4, vq_4);
    const xnn_simd_f16_t verf_5 = xnn_div_f16(vp_5, vq_5);
    const xnn_simd_f16_t verf_6 = xnn_div_f16(vp_6, vq_6);
    const xnn_simd_f16_t verf_7 = xnn_div_f16(vp_7, vq_7);

    // Add one to the rational interpolant, and multiply by 0.5 times the
    // original input.
    const xnn_simd_f16_t vy_0 = xnn_mul_f16(xnn_mul_f16(vx_orig_0, vhalf),
                                        xnn_add_f16(verf_0, vone));
    const xnn_simd_f16_t vy_1 = xnn_mul_f16(xnn_mul_f16(vx_orig_1, vhalf),
                                        xnn_add_f16(verf_1, vone));
    const xnn_simd_f16_t vy_2 = xnn_mul_f16(xnn_mul_f16(vx_orig_2, vhalf),
                                        xnn_add_f16(verf_2, vone));
    const xnn_simd_f16_t vy_3 = xnn_mul_f16(xnn_mul_f16(vx_orig_3, vhalf),
                                        xnn_add_f16(verf_3, vone));
    const xnn_simd_f16_t vy_4 = xnn_mul_f16(xnn_mul_f16(vx_orig_4, vhalf),
                                        xnn_add_f16(verf_4, vone));
    const xnn_simd_f16_t vy_5 = xnn_mul_f16(xnn_mul_f16(vx_orig_5, vhalf),
                                        xnn_add_f16(verf_5, vone));
    const xnn_simd_f16_t vy_6 = xnn_mul_f16(xnn_mul_f16(vx_orig_6, vhalf),
                                        xnn_add_f16(verf_6, vone));
    const xnn_simd_f16_t vy_7 = xnn_mul_f16(xnn_mul_f16(vx_orig_7, vhalf),
                                        xnn_add_f16(verf_7, vone));

    xnn_storeu_f16(output, vy_0);
    xnn_storeu_f16(output + 1 * xnn_simd_size_f16, vy_1);
    xnn_storeu_f16(output + 2 * xnn_simd_size_f16, vy_2);
    xnn_storeu_f16(output + 3 * xnn_simd_size_f16, vy_3);
    xnn_storeu_f16(output + 4 * xnn_simd_size_f16, vy_4);
    xnn_storeu_f16(output + 5 * xnn_simd_size_f16, vy_5);
    xnn_storeu_f16(output + 6 * xnn_simd_size_f16, vy_6);
    xnn_storeu_f16(output + 7 * xnn_simd_size_f16, vy_7);
    output += 8;
  }
  for (; batch >= xnn_simd_bytes_f16; batch -= xnn_simd_bytes_f16) {
    const xnn_simd_f16_t vx_orig = xnn_loadu_f16(input);
    input += xnn_simd_size_f16;

    // Clamp the inputs to the interpolation range.
    xnn_simd_f16_t vx = xnn_min_f16(vmax_abs_x, vx_orig);
    vx = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f16_t vx2 = xnn_mul_f16(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f16_t vp = xnn_fmadd_f16(vx2, valpha_5, valpha_3);
    vp = xnn_fmadd_f16(vx2, vp, valpha_1);
    vp = xnn_mul_f16(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f16_t vq = xnn_fmadd_f16(vx2, vbeta_4, vbeta_2);
    vq = xnn_fmadd_f16(vx2, vq, vone);

    // Divide the numerator by the denominator and add one
    const xnn_simd_f16_t verf =  xnn_div_f16(vp, vq);

    // Add one to the rational interpolant, and multiply by 0.5 times the
    // original input.
    const xnn_simd_f16_t vy = xnn_mul_f16(xnn_mul_f16(vx_orig, vhalf),
                                          xnn_add_f16(verf, vone));

    xnn_storeu_f16(output, vy);
    output += xnn_simd_size_f16;
  }
}
