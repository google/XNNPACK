// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vsin/rational-5-4.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/simd/f32-fma3.h"

#include "src/xnnpack/common.h"
// #include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f32_vsin_ukernel__fma3_rational_5_4_div_u8(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 8);

  // Some mathematical constants. We don't use pre-defined macros to ensure
  // that they are rounded exactly as we expect them to be.
  XNN_SIMD_CONST_F32(vpi, 3.1415927f);  // M_PI
  XNN_SIMD_CONST_F32(v2pi_inv, 0.15915494f); // 0.5 / M_PI

  // The following two values sum to 2*Pi with ~33 bits of accuracy. We use
  // them to accurately subtract integer multiples of 2*Pi from large inputs.
  XNN_SIMD_CONST_F32(v2pi_hi, 6.28125f);  // 2.0 * M_PI (first 11 bits of mantissa)
  XNN_SIMD_CONST_F32(v2pi_lo, 1.9353072e-3);  // 2.0 * M_PI (remaining bits)

  // The monomial coefficients of the numerator polynomial (odd,
  // `valpha_1` = `vone`).
  XNN_SIMD_CONST_F32(valpha_3, -1.3314664364e-01f);
  XNN_SIMD_CONST_F32(valpha_5, 3.2340581529e-03f);

  // The monomial coefficients of the denominator polynomial (even,
  // `vbeta_0` = `vone`).
  XNN_SIMD_CONST_F32(vbeta_2, 3.3519912511e-02f);
  XNN_SIMD_CONST_F32(vbeta_4, 4.8770775902e-04f);

  // Some useful constants.
  XNN_SIMD_CONST_F32(vone, 1.0f);

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Map the inputs to the interpolation range.
    xnn_simd_f32_t vx_div_2pi = xnn_mul_f32(vx, v2pi_inv);
    vx_div_2pi = xnn_round_f32(vx_div_2pi);
    vx = xnn_fnmadd_f32(vx_div_2pi, v2pi_hi, vx);
    vx = xnn_fnmadd_f32(vx_div_2pi, v2pi_lo, vx);
    vx = xnn_min_f32(vx, xnn_sub_f32(vpi, vx));
    vx = xnn_max_f32(vx, xnn_sub_f32(xnn_neg_f32(vpi), vx));
    vx = xnn_min_f32(vx, xnn_sub_f32(vpi, vx));

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_5, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, vone);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_4, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    // Map the inputs to the interpolation range.
    xnn_simd_f32_t vx_div_2pi = xnn_mul_f32(vx, v2pi_inv);
    vx_div_2pi = xnn_round_f32(vx_div_2pi);
    vx = xnn_fnmadd_f32(vx_div_2pi, v2pi_hi, vx);
    vx = xnn_fnmadd_f32(vx_div_2pi, v2pi_lo, vx);
    vx = xnn_min_f32(vx, xnn_sub_f32(vpi, vx));
    vx = xnn_max_f32(vx, xnn_sub_f32(xnn_neg_f32(vpi), vx));
    vx = xnn_min_f32(vx, xnn_sub_f32(vpi, vx));

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_5, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, vone);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_4, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vsin_ukernel__fma3_rational_5_4_div_u16(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 8);

  // Some mathematical constants. We don't use pre-defined macros to ensure
  // that they are rounded exactly as we expect them to be.
  XNN_SIMD_CONST_F32(vpi, 3.1415927f);  // M_PI
  XNN_SIMD_CONST_F32(v2pi_inv, 0.15915494f); // 0.5 / M_PI

  // The following two values sum to 2*Pi with ~33 bits of accuracy. We use
  // them to accurately subtract integer multiples of 2*Pi from large inputs.
  XNN_SIMD_CONST_F32(v2pi_hi, 6.28125f);  // 2.0 * M_PI (first 11 bits of mantissa)
  XNN_SIMD_CONST_F32(v2pi_lo, 1.9353072e-3);  // 2.0 * M_PI (remaining bits)

  // The monomial coefficients of the numerator polynomial (odd,
  // `valpha_1` = `vone`).
  XNN_SIMD_CONST_F32(valpha_3, -1.3314664364e-01f);
  XNN_SIMD_CONST_F32(valpha_5, 3.2340581529e-03f);

  // The monomial coefficients of the denominator polynomial (even,
  // `vbeta_0` = `vone`).
  XNN_SIMD_CONST_F32(vbeta_2, 3.3519912511e-02f);
  XNN_SIMD_CONST_F32(vbeta_4, 4.8770775902e-04f);

  // Some useful constants.
  XNN_SIMD_CONST_F32(vone, 1.0f);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 16;

    // Map the inputs to the interpolation range.
    xnn_simd_f32_t vx_div_2pi_0 = xnn_mul_f32(vx_0, v2pi_inv);
    xnn_simd_f32_t vx_div_2pi_1 = xnn_mul_f32(vx_1, v2pi_inv);
    vx_div_2pi_0 = xnn_round_f32(vx_div_2pi_0);
    vx_div_2pi_1 = xnn_round_f32(vx_div_2pi_1);
    vx_0 = xnn_fnmadd_f32(vx_div_2pi_0, v2pi_hi, vx_0);
    vx_1 = xnn_fnmadd_f32(vx_div_2pi_1, v2pi_hi, vx_1);
    vx_0 = xnn_fnmadd_f32(vx_div_2pi_0, v2pi_lo, vx_0);
    vx_1 = xnn_fnmadd_f32(vx_div_2pi_1, v2pi_lo, vx_1);
    vx_0 = xnn_min_f32(vx_0, xnn_sub_f32(vpi, vx_0));
    vx_1 = xnn_min_f32(vx_1, xnn_sub_f32(vpi, vx_1));
    vx_0 = xnn_max_f32(vx_0, xnn_sub_f32(xnn_neg_f32(vpi), vx_0));
    vx_1 = xnn_max_f32(vx_1, xnn_sub_f32(xnn_neg_f32(vpi), vx_1));
    vx_0 = xnn_min_f32(vx_0, xnn_sub_f32(vpi, vx_0));
    vx_1 = xnn_min_f32(vx_1, xnn_sub_f32(vpi, vx_1));

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2_0 = xnn_mul_f32(vx_0, vx_0);
    const xnn_simd_f32_t vx2_1 = xnn_mul_f32(vx_1, vx_1);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx2_0, valpha_5, valpha_3);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx2_1, valpha_5, valpha_3);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, vone);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, vone);
    vp_0 = xnn_mul_f32(vx_0, vp_0);
    vp_1 = xnn_mul_f32(vx_1, vp_1);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vx2_0, vbeta_4, vbeta_2);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vx2_1, vbeta_4, vbeta_2);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vone);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy_0 = xnn_div_f32(vp_0, vq_0);
    const xnn_simd_f32_t vy_1 = xnn_div_f32(vp_1, vq_1);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    output += 16;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Map the inputs to the interpolation range.
    xnn_simd_f32_t vx_div_2pi = xnn_mul_f32(vx, v2pi_inv);
    vx_div_2pi = xnn_round_f32(vx_div_2pi);
    vx = xnn_fnmadd_f32(vx_div_2pi, v2pi_hi, vx);
    vx = xnn_fnmadd_f32(vx_div_2pi, v2pi_lo, vx);
    vx = xnn_min_f32(vx, xnn_sub_f32(vpi, vx));
    vx = xnn_max_f32(vx, xnn_sub_f32(xnn_neg_f32(vpi), vx));
    vx = xnn_min_f32(vx, xnn_sub_f32(vpi, vx));

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_5, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, vone);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_4, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    // Map the inputs to the interpolation range.
    xnn_simd_f32_t vx_div_2pi = xnn_mul_f32(vx, v2pi_inv);
    vx_div_2pi = xnn_round_f32(vx_div_2pi);
    vx = xnn_fnmadd_f32(vx_div_2pi, v2pi_hi, vx);
    vx = xnn_fnmadd_f32(vx_div_2pi, v2pi_lo, vx);
    vx = xnn_min_f32(vx, xnn_sub_f32(vpi, vx));
    vx = xnn_max_f32(vx, xnn_sub_f32(xnn_neg_f32(vpi), vx));
    vx = xnn_min_f32(vx, xnn_sub_f32(vpi, vx));

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_5, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, vone);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_4, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vsin_ukernel__fma3_rational_5_4_div_u24(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 8);

  // Some mathematical constants. We don't use pre-defined macros to ensure
  // that they are rounded exactly as we expect them to be.
  XNN_SIMD_CONST_F32(vpi, 3.1415927f);  // M_PI
  XNN_SIMD_CONST_F32(v2pi_inv, 0.15915494f); // 0.5 / M_PI

  // The following two values sum to 2*Pi with ~33 bits of accuracy. We use
  // them to accurately subtract integer multiples of 2*Pi from large inputs.
  XNN_SIMD_CONST_F32(v2pi_hi, 6.28125f);  // 2.0 * M_PI (first 11 bits of mantissa)
  XNN_SIMD_CONST_F32(v2pi_lo, 1.9353072e-3);  // 2.0 * M_PI (remaining bits)

  // The monomial coefficients of the numerator polynomial (odd,
  // `valpha_1` = `vone`).
  XNN_SIMD_CONST_F32(valpha_3, -1.3314664364e-01f);
  XNN_SIMD_CONST_F32(valpha_5, 3.2340581529e-03f);

  // The monomial coefficients of the denominator polynomial (even,
  // `vbeta_0` = `vone`).
  XNN_SIMD_CONST_F32(vbeta_2, 3.3519912511e-02f);
  XNN_SIMD_CONST_F32(vbeta_4, 4.8770775902e-04f);

  // Some useful constants.
  XNN_SIMD_CONST_F32(vone, 1.0f);

  for (; batch >= 24 * sizeof(float); batch -= 24 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    input += 24;

    // Map the inputs to the interpolation range.
    xnn_simd_f32_t vx_div_2pi_0 = xnn_mul_f32(vx_0, v2pi_inv);
    xnn_simd_f32_t vx_div_2pi_1 = xnn_mul_f32(vx_1, v2pi_inv);
    xnn_simd_f32_t vx_div_2pi_2 = xnn_mul_f32(vx_2, v2pi_inv);
    vx_div_2pi_0 = xnn_round_f32(vx_div_2pi_0);
    vx_div_2pi_1 = xnn_round_f32(vx_div_2pi_1);
    vx_div_2pi_2 = xnn_round_f32(vx_div_2pi_2);
    vx_0 = xnn_fnmadd_f32(vx_div_2pi_0, v2pi_hi, vx_0);
    vx_1 = xnn_fnmadd_f32(vx_div_2pi_1, v2pi_hi, vx_1);
    vx_2 = xnn_fnmadd_f32(vx_div_2pi_2, v2pi_hi, vx_2);
    vx_0 = xnn_fnmadd_f32(vx_div_2pi_0, v2pi_lo, vx_0);
    vx_1 = xnn_fnmadd_f32(vx_div_2pi_1, v2pi_lo, vx_1);
    vx_2 = xnn_fnmadd_f32(vx_div_2pi_2, v2pi_lo, vx_2);
    vx_0 = xnn_min_f32(vx_0, xnn_sub_f32(vpi, vx_0));
    vx_1 = xnn_min_f32(vx_1, xnn_sub_f32(vpi, vx_1));
    vx_2 = xnn_min_f32(vx_2, xnn_sub_f32(vpi, vx_2));
    vx_0 = xnn_max_f32(vx_0, xnn_sub_f32(xnn_neg_f32(vpi), vx_0));
    vx_1 = xnn_max_f32(vx_1, xnn_sub_f32(xnn_neg_f32(vpi), vx_1));
    vx_2 = xnn_max_f32(vx_2, xnn_sub_f32(xnn_neg_f32(vpi), vx_2));
    vx_0 = xnn_min_f32(vx_0, xnn_sub_f32(vpi, vx_0));
    vx_1 = xnn_min_f32(vx_1, xnn_sub_f32(vpi, vx_1));
    vx_2 = xnn_min_f32(vx_2, xnn_sub_f32(vpi, vx_2));

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2_0 = xnn_mul_f32(vx_0, vx_0);
    const xnn_simd_f32_t vx2_1 = xnn_mul_f32(vx_1, vx_1);
    const xnn_simd_f32_t vx2_2 = xnn_mul_f32(vx_2, vx_2);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx2_0, valpha_5, valpha_3);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx2_1, valpha_5, valpha_3);
    xnn_simd_f32_t vp_2 = xnn_fmadd_f32(vx2_2, valpha_5, valpha_3);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, vone);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, vone);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, vone);
    vp_0 = xnn_mul_f32(vx_0, vp_0);
    vp_1 = xnn_mul_f32(vx_1, vp_1);
    vp_2 = xnn_mul_f32(vx_2, vp_2);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vx2_0, vbeta_4, vbeta_2);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vx2_1, vbeta_4, vbeta_2);
    xnn_simd_f32_t vq_2 = xnn_fmadd_f32(vx2_2, vbeta_4, vbeta_2);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vone);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vone);
    vq_2 = xnn_fmadd_f32(vx2_2, vq_2, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy_0 = xnn_div_f32(vp_0, vq_0);
    const xnn_simd_f32_t vy_1 = xnn_div_f32(vp_1, vq_1);
    const xnn_simd_f32_t vy_2 = xnn_div_f32(vp_2, vq_2);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_2);
    output += 24;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Map the inputs to the interpolation range.
    xnn_simd_f32_t vx_div_2pi = xnn_mul_f32(vx, v2pi_inv);
    vx_div_2pi = xnn_round_f32(vx_div_2pi);
    vx = xnn_fnmadd_f32(vx_div_2pi, v2pi_hi, vx);
    vx = xnn_fnmadd_f32(vx_div_2pi, v2pi_lo, vx);
    vx = xnn_min_f32(vx, xnn_sub_f32(vpi, vx));
    vx = xnn_max_f32(vx, xnn_sub_f32(xnn_neg_f32(vpi), vx));
    vx = xnn_min_f32(vx, xnn_sub_f32(vpi, vx));

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_5, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, vone);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_4, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    // Map the inputs to the interpolation range.
    xnn_simd_f32_t vx_div_2pi = xnn_mul_f32(vx, v2pi_inv);
    vx_div_2pi = xnn_round_f32(vx_div_2pi);
    vx = xnn_fnmadd_f32(vx_div_2pi, v2pi_hi, vx);
    vx = xnn_fnmadd_f32(vx_div_2pi, v2pi_lo, vx);
    vx = xnn_min_f32(vx, xnn_sub_f32(vpi, vx));
    vx = xnn_max_f32(vx, xnn_sub_f32(xnn_neg_f32(vpi), vx));
    vx = xnn_min_f32(vx, xnn_sub_f32(vpi, vx));

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_5, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, vone);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_4, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vsin_ukernel__fma3_rational_5_4_div_u32(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 8);

  // Some mathematical constants. We don't use pre-defined macros to ensure
  // that they are rounded exactly as we expect them to be.
  XNN_SIMD_CONST_F32(vpi, 3.1415927f);  // M_PI
  XNN_SIMD_CONST_F32(v2pi_inv, 0.15915494f); // 0.5 / M_PI

  // The following two values sum to 2*Pi with ~33 bits of accuracy. We use
  // them to accurately subtract integer multiples of 2*Pi from large inputs.
  XNN_SIMD_CONST_F32(v2pi_hi, 6.28125f);  // 2.0 * M_PI (first 11 bits of mantissa)
  XNN_SIMD_CONST_F32(v2pi_lo, 1.9353072e-3);  // 2.0 * M_PI (remaining bits)

  // The monomial coefficients of the numerator polynomial (odd,
  // `valpha_1` = `vone`).
  XNN_SIMD_CONST_F32(valpha_3, -1.3314664364e-01f);
  XNN_SIMD_CONST_F32(valpha_5, 3.2340581529e-03f);

  // The monomial coefficients of the denominator polynomial (even,
  // `vbeta_0` = `vone`).
  XNN_SIMD_CONST_F32(vbeta_2, 3.3519912511e-02f);
  XNN_SIMD_CONST_F32(vbeta_4, 4.8770775902e-04f);

  // Some useful constants.
  XNN_SIMD_CONST_F32(vone, 1.0f);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_3 = xnn_loadu_f32(input + 3 * xnn_simd_size_f32);
    input += 32;

    // Map the inputs to the interpolation range.
    xnn_simd_f32_t vx_div_2pi_0 = xnn_mul_f32(vx_0, v2pi_inv);
    xnn_simd_f32_t vx_div_2pi_1 = xnn_mul_f32(vx_1, v2pi_inv);
    xnn_simd_f32_t vx_div_2pi_2 = xnn_mul_f32(vx_2, v2pi_inv);
    xnn_simd_f32_t vx_div_2pi_3 = xnn_mul_f32(vx_3, v2pi_inv);
    vx_div_2pi_0 = xnn_round_f32(vx_div_2pi_0);
    vx_div_2pi_1 = xnn_round_f32(vx_div_2pi_1);
    vx_div_2pi_2 = xnn_round_f32(vx_div_2pi_2);
    vx_div_2pi_3 = xnn_round_f32(vx_div_2pi_3);
    vx_0 = xnn_fnmadd_f32(vx_div_2pi_0, v2pi_hi, vx_0);
    vx_1 = xnn_fnmadd_f32(vx_div_2pi_1, v2pi_hi, vx_1);
    vx_2 = xnn_fnmadd_f32(vx_div_2pi_2, v2pi_hi, vx_2);
    vx_3 = xnn_fnmadd_f32(vx_div_2pi_3, v2pi_hi, vx_3);
    vx_0 = xnn_fnmadd_f32(vx_div_2pi_0, v2pi_lo, vx_0);
    vx_1 = xnn_fnmadd_f32(vx_div_2pi_1, v2pi_lo, vx_1);
    vx_2 = xnn_fnmadd_f32(vx_div_2pi_2, v2pi_lo, vx_2);
    vx_3 = xnn_fnmadd_f32(vx_div_2pi_3, v2pi_lo, vx_3);
    vx_0 = xnn_min_f32(vx_0, xnn_sub_f32(vpi, vx_0));
    vx_1 = xnn_min_f32(vx_1, xnn_sub_f32(vpi, vx_1));
    vx_2 = xnn_min_f32(vx_2, xnn_sub_f32(vpi, vx_2));
    vx_3 = xnn_min_f32(vx_3, xnn_sub_f32(vpi, vx_3));
    vx_0 = xnn_max_f32(vx_0, xnn_sub_f32(xnn_neg_f32(vpi), vx_0));
    vx_1 = xnn_max_f32(vx_1, xnn_sub_f32(xnn_neg_f32(vpi), vx_1));
    vx_2 = xnn_max_f32(vx_2, xnn_sub_f32(xnn_neg_f32(vpi), vx_2));
    vx_3 = xnn_max_f32(vx_3, xnn_sub_f32(xnn_neg_f32(vpi), vx_3));
    vx_0 = xnn_min_f32(vx_0, xnn_sub_f32(vpi, vx_0));
    vx_1 = xnn_min_f32(vx_1, xnn_sub_f32(vpi, vx_1));
    vx_2 = xnn_min_f32(vx_2, xnn_sub_f32(vpi, vx_2));
    vx_3 = xnn_min_f32(vx_3, xnn_sub_f32(vpi, vx_3));

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2_0 = xnn_mul_f32(vx_0, vx_0);
    const xnn_simd_f32_t vx2_1 = xnn_mul_f32(vx_1, vx_1);
    const xnn_simd_f32_t vx2_2 = xnn_mul_f32(vx_2, vx_2);
    const xnn_simd_f32_t vx2_3 = xnn_mul_f32(vx_3, vx_3);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx2_0, valpha_5, valpha_3);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx2_1, valpha_5, valpha_3);
    xnn_simd_f32_t vp_2 = xnn_fmadd_f32(vx2_2, valpha_5, valpha_3);
    xnn_simd_f32_t vp_3 = xnn_fmadd_f32(vx2_3, valpha_5, valpha_3);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, vone);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, vone);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, vone);
    vp_3 = xnn_fmadd_f32(vx2_3, vp_3, vone);
    vp_0 = xnn_mul_f32(vx_0, vp_0);
    vp_1 = xnn_mul_f32(vx_1, vp_1);
    vp_2 = xnn_mul_f32(vx_2, vp_2);
    vp_3 = xnn_mul_f32(vx_3, vp_3);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vx2_0, vbeta_4, vbeta_2);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vx2_1, vbeta_4, vbeta_2);
    xnn_simd_f32_t vq_2 = xnn_fmadd_f32(vx2_2, vbeta_4, vbeta_2);
    xnn_simd_f32_t vq_3 = xnn_fmadd_f32(vx2_3, vbeta_4, vbeta_2);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vone);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vone);
    vq_2 = xnn_fmadd_f32(vx2_2, vq_2, vone);
    vq_3 = xnn_fmadd_f32(vx2_3, vq_3, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy_0 = xnn_div_f32(vp_0, vq_0);
    const xnn_simd_f32_t vy_1 = xnn_div_f32(vp_1, vq_1);
    const xnn_simd_f32_t vy_2 = xnn_div_f32(vp_2, vq_2);
    const xnn_simd_f32_t vy_3 = xnn_div_f32(vp_3, vq_3);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_2);
    xnn_storeu_f32(output + 3 * xnn_simd_size_f32, vy_3);
    output += 32;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Map the inputs to the interpolation range.
    xnn_simd_f32_t vx_div_2pi = xnn_mul_f32(vx, v2pi_inv);
    vx_div_2pi = xnn_round_f32(vx_div_2pi);
    vx = xnn_fnmadd_f32(vx_div_2pi, v2pi_hi, vx);
    vx = xnn_fnmadd_f32(vx_div_2pi, v2pi_lo, vx);
    vx = xnn_min_f32(vx, xnn_sub_f32(vpi, vx));
    vx = xnn_max_f32(vx, xnn_sub_f32(xnn_neg_f32(vpi), vx));
    vx = xnn_min_f32(vx, xnn_sub_f32(vpi, vx));

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_5, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, vone);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_4, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    // Map the inputs to the interpolation range.
    xnn_simd_f32_t vx_div_2pi = xnn_mul_f32(vx, v2pi_inv);
    vx_div_2pi = xnn_round_f32(vx_div_2pi);
    vx = xnn_fnmadd_f32(vx_div_2pi, v2pi_hi, vx);
    vx = xnn_fnmadd_f32(vx_div_2pi, v2pi_lo, vx);
    vx = xnn_min_f32(vx, xnn_sub_f32(vpi, vx));
    vx = xnn_max_f32(vx, xnn_sub_f32(xnn_neg_f32(vpi), vx));
    vx = xnn_min_f32(vx, xnn_sub_f32(vpi, vx));

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_5, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, vone);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_4, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}
