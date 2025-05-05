// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vgelu/rational-6-4.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/simd/f16-avx512fp16.h"

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vgelu_ukernel__avx512fp16_rational_6_4_div_u32(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 32);

  // Cap the inputs to this value as `erf(x/sqrt(2))` will always be `+/-1.0f`
  // beyond this point. This value is chosen as the first floating point
  // number as of which the interpolation returns +/-1.0f.
#if XNN_SIMD_HAS_NATIVE_FMA
  XNN_SIMD_CONST_F16_FROM_FLOAT(vmax_abs_x, 3.5e+00);
#else
  XNN_SIMD_CONST_F16_FROM_FLOAT(vmax_abs_x, 3.28906250e+00);
#endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_1, 7.9763203859e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_3, 1.5397867560e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_5, 4.5998394489e-03f);

  // The monomial coefficients of the denominator polynomial (even).
  // XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_0, 1.0e+00f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_2, 3.5756936669e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_4, 4.2862996459e-02f);

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
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f16_t vx_orig = xnn_load_tail_f16(input, batch >> XNN_LOG2_SIZEOF_HALF);

  // See above for comments.
  xnn_simd_f16_t vx = xnn_min_f16(vmax_abs_x, vx_orig);
  vx = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx);
  const xnn_simd_f16_t vx2 = xnn_mul_f16(vx, vx);
  xnn_simd_f16_t vp = xnn_fmadd_f16(vx2, valpha_5, valpha_3);
  vp = xnn_fmadd_f16(vx2, vp, valpha_1);
  vp = xnn_mul_f16(vx, vp);
  xnn_simd_f16_t vq = xnn_fmadd_f16(vx2, vbeta_4, vbeta_2);
  vq = xnn_fmadd_f16(vx2, vq, vone);
  const xnn_simd_f16_t verf =  xnn_div_f16(vp, vq);
  const xnn_simd_f16_t vy = xnn_mul_f16(xnn_mul_f16(vx_orig, vhalf),
                                        xnn_add_f16(verf, vone));

    xnn_store_tail_f16(output, vy, batch >> XNN_LOG2_SIZEOF_HALF);
  }
}

void xnn_f16_vgelu_ukernel__avx512fp16_rational_6_4_div_u64(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 32);

  // Cap the inputs to this value as `erf(x/sqrt(2))` will always be `+/-1.0f`
  // beyond this point. This value is chosen as the first floating point
  // number as of which the interpolation returns +/-1.0f.
#if XNN_SIMD_HAS_NATIVE_FMA
  XNN_SIMD_CONST_F16_FROM_FLOAT(vmax_abs_x, 3.5e+00);
#else
  XNN_SIMD_CONST_F16_FROM_FLOAT(vmax_abs_x, 3.28906250e+00);
#endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_1, 7.9763203859e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_3, 1.5397867560e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_5, 4.5998394489e-03f);

  // The monomial coefficients of the denominator polynomial (even).
  // XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_0, 1.0e+00f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_2, 3.5756936669e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_4, 4.2862996459e-02f);

  XNN_SIMD_CONST_F16_FROM_FLOAT(vone, 1.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vhalf, 0.5f);

  for (; batch >= 64 * sizeof(xnn_float16); batch -= 64 * sizeof(xnn_float16)) {
    const xnn_simd_f16_t vx_orig_0 = xnn_loadu_f16(input + 0 * xnn_simd_size_f16);
    const xnn_simd_f16_t vx_orig_1 = xnn_loadu_f16(input + 1 * xnn_simd_size_f16);
    input += 64;

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

    xnn_storeu_f16(output + 0 * xnn_simd_size_f16, vy_0);
    xnn_storeu_f16(output + 1 * xnn_simd_size_f16, vy_1);
    output += 64;
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
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f16_t vx_orig = xnn_load_tail_f16(input, batch >> XNN_LOG2_SIZEOF_HALF);

  // See above for comments.
  xnn_simd_f16_t vx = xnn_min_f16(vmax_abs_x, vx_orig);
  vx = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx);
  const xnn_simd_f16_t vx2 = xnn_mul_f16(vx, vx);
  xnn_simd_f16_t vp = xnn_fmadd_f16(vx2, valpha_5, valpha_3);
  vp = xnn_fmadd_f16(vx2, vp, valpha_1);
  vp = xnn_mul_f16(vx, vp);
  xnn_simd_f16_t vq = xnn_fmadd_f16(vx2, vbeta_4, vbeta_2);
  vq = xnn_fmadd_f16(vx2, vq, vone);
  const xnn_simd_f16_t verf =  xnn_div_f16(vp, vq);
  const xnn_simd_f16_t vy = xnn_mul_f16(xnn_mul_f16(vx_orig, vhalf),
                                        xnn_add_f16(verf, vone));

    xnn_store_tail_f16(output, vy, batch >> XNN_LOG2_SIZEOF_HALF);
  }
}

void xnn_f16_vgelu_ukernel__avx512fp16_rational_6_4_div_u96(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 32);

  // Cap the inputs to this value as `erf(x/sqrt(2))` will always be `+/-1.0f`
  // beyond this point. This value is chosen as the first floating point
  // number as of which the interpolation returns +/-1.0f.
#if XNN_SIMD_HAS_NATIVE_FMA
  XNN_SIMD_CONST_F16_FROM_FLOAT(vmax_abs_x, 3.5e+00);
#else
  XNN_SIMD_CONST_F16_FROM_FLOAT(vmax_abs_x, 3.28906250e+00);
#endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_1, 7.9763203859e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_3, 1.5397867560e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_5, 4.5998394489e-03f);

  // The monomial coefficients of the denominator polynomial (even).
  // XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_0, 1.0e+00f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_2, 3.5756936669e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_4, 4.2862996459e-02f);

  XNN_SIMD_CONST_F16_FROM_FLOAT(vone, 1.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vhalf, 0.5f);

  for (; batch >= 96 * sizeof(xnn_float16); batch -= 96 * sizeof(xnn_float16)) {
    const xnn_simd_f16_t vx_orig_0 = xnn_loadu_f16(input + 0 * xnn_simd_size_f16);
    const xnn_simd_f16_t vx_orig_1 = xnn_loadu_f16(input + 1 * xnn_simd_size_f16);
    const xnn_simd_f16_t vx_orig_2 = xnn_loadu_f16(input + 2 * xnn_simd_size_f16);
    input += 96;

    // Clamp the inputs to the interpolation range.
    xnn_simd_f16_t vx_0 = xnn_min_f16(vmax_abs_x, vx_orig_0);
    xnn_simd_f16_t vx_1 = xnn_min_f16(vmax_abs_x, vx_orig_1);
    xnn_simd_f16_t vx_2 = xnn_min_f16(vmax_abs_x, vx_orig_2);
    vx_0 = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx_0);
    vx_1 = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx_1);
    vx_2 = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx_2);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f16_t vx2_0 = xnn_mul_f16(vx_0, vx_0);
    const xnn_simd_f16_t vx2_1 = xnn_mul_f16(vx_1, vx_1);
    const xnn_simd_f16_t vx2_2 = xnn_mul_f16(vx_2, vx_2);

    // Evaluate the numerator polynomial p.
    xnn_simd_f16_t vp_0 = xnn_fmadd_f16(vx2_0, valpha_5, valpha_3);
    xnn_simd_f16_t vp_1 = xnn_fmadd_f16(vx2_1, valpha_5, valpha_3);
    xnn_simd_f16_t vp_2 = xnn_fmadd_f16(vx2_2, valpha_5, valpha_3);
    vp_0 = xnn_fmadd_f16(vx2_0, vp_0, valpha_1);
    vp_1 = xnn_fmadd_f16(vx2_1, vp_1, valpha_1);
    vp_2 = xnn_fmadd_f16(vx2_2, vp_2, valpha_1);
    vp_0 = xnn_mul_f16(vx_0, vp_0);
    vp_1 = xnn_mul_f16(vx_1, vp_1);
    vp_2 = xnn_mul_f16(vx_2, vp_2);

    // Evaluate the denominator polynomial q.
    xnn_simd_f16_t vq_0 = xnn_fmadd_f16(vx2_0, vbeta_4, vbeta_2);
    xnn_simd_f16_t vq_1 = xnn_fmadd_f16(vx2_1, vbeta_4, vbeta_2);
    xnn_simd_f16_t vq_2 = xnn_fmadd_f16(vx2_2, vbeta_4, vbeta_2);
    vq_0 = xnn_fmadd_f16(vx2_0, vq_0, vone);
    vq_1 = xnn_fmadd_f16(vx2_1, vq_1, vone);
    vq_2 = xnn_fmadd_f16(vx2_2, vq_2, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f16_t verf_0 = xnn_div_f16(vp_0, vq_0);
    const xnn_simd_f16_t verf_1 = xnn_div_f16(vp_1, vq_1);
    const xnn_simd_f16_t verf_2 = xnn_div_f16(vp_2, vq_2);

    // Add one to the rational interpolant, and multiply by 0.5 times the
    // original input.
    const xnn_simd_f16_t vy_0 = xnn_mul_f16(xnn_mul_f16(vx_orig_0, vhalf),
                                        xnn_add_f16(verf_0, vone));
    const xnn_simd_f16_t vy_1 = xnn_mul_f16(xnn_mul_f16(vx_orig_1, vhalf),
                                        xnn_add_f16(verf_1, vone));
    const xnn_simd_f16_t vy_2 = xnn_mul_f16(xnn_mul_f16(vx_orig_2, vhalf),
                                        xnn_add_f16(verf_2, vone));

    xnn_storeu_f16(output + 0 * xnn_simd_size_f16, vy_0);
    xnn_storeu_f16(output + 1 * xnn_simd_size_f16, vy_1);
    xnn_storeu_f16(output + 2 * xnn_simd_size_f16, vy_2);
    output += 96;
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
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f16_t vx_orig = xnn_load_tail_f16(input, batch >> XNN_LOG2_SIZEOF_HALF);

  // See above for comments.
  xnn_simd_f16_t vx = xnn_min_f16(vmax_abs_x, vx_orig);
  vx = xnn_max_f16(xnn_neg_f16(vmax_abs_x), vx);
  const xnn_simd_f16_t vx2 = xnn_mul_f16(vx, vx);
  xnn_simd_f16_t vp = xnn_fmadd_f16(vx2, valpha_5, valpha_3);
  vp = xnn_fmadd_f16(vx2, vp, valpha_1);
  vp = xnn_mul_f16(vx, vp);
  xnn_simd_f16_t vq = xnn_fmadd_f16(vx2, vbeta_4, vbeta_2);
  vq = xnn_fmadd_f16(vx2, vq, vone);
  const xnn_simd_f16_t verf =  xnn_div_f16(vp, vq);
  const xnn_simd_f16_t vy = xnn_mul_f16(xnn_mul_f16(vx_orig, vhalf),
                                        xnn_add_f16(verf, vone));

    xnn_store_tail_f16(output, vy, batch >> XNN_LOG2_SIZEOF_HALF);
  }
}
