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

#include "src/xnnpack/simd/f32-scalar.h"

#include "src/xnnpack/common.h"
// #include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f32_vcos_ukernel__scalar_rational_5_4_div_u1(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  // Some mathematical constants. We don't use pre-defined macros to ensure
  // that they are rounded exactly as we expect them to be.
  XNN_SIMD_CONST_F32(vpi, 3.1415927f);  // M_PI
  XNN_SIMD_CONST_F32(v2pi_inv, 0.15915494f); // 0.5 / M_PI
  XNN_SIMD_CONST_F32(vpi_half, 1.5707964f);  // M_PI / 2

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
    vx = xnn_sub_f32(vpi_half, vx);
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
}

void xnn_f32_vcos_ukernel__scalar_rational_5_4_div_u2(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  // Some mathematical constants. We don't use pre-defined macros to ensure
  // that they are rounded exactly as we expect them to be.
  XNN_SIMD_CONST_F32(vpi, 3.1415927f);  // M_PI
  XNN_SIMD_CONST_F32(v2pi_inv, 0.15915494f); // 0.5 / M_PI
  XNN_SIMD_CONST_F32(vpi_half, 1.5707964f);  // M_PI / 2

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

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 2;

    // Map the inputs to the interpolation range.
    xnn_simd_f32_t vx_div_2pi_0 = xnn_mul_f32(vx_0, v2pi_inv);
    xnn_simd_f32_t vx_div_2pi_1 = xnn_mul_f32(vx_1, v2pi_inv);
    vx_div_2pi_0 = xnn_round_f32(vx_div_2pi_0);
    vx_div_2pi_1 = xnn_round_f32(vx_div_2pi_1);
    vx_0 = xnn_fnmadd_f32(vx_div_2pi_0, v2pi_hi, vx_0);
    vx_1 = xnn_fnmadd_f32(vx_div_2pi_1, v2pi_hi, vx_1);
    vx_0 = xnn_fnmadd_f32(vx_div_2pi_0, v2pi_lo, vx_0);
    vx_1 = xnn_fnmadd_f32(vx_div_2pi_1, v2pi_lo, vx_1);
    vx_0 = xnn_sub_f32(vpi_half, vx_0);
    vx_1 = xnn_sub_f32(vpi_half, vx_1);
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

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    output += 2;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Map the inputs to the interpolation range.
    xnn_simd_f32_t vx_div_2pi = xnn_mul_f32(vx, v2pi_inv);
    vx_div_2pi = xnn_round_f32(vx_div_2pi);
    vx = xnn_fnmadd_f32(vx_div_2pi, v2pi_hi, vx);
    vx = xnn_fnmadd_f32(vx_div_2pi, v2pi_lo, vx);
    vx = xnn_sub_f32(vpi_half, vx);
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
}

void xnn_f32_vcos_ukernel__scalar_rational_5_4_div_u4(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  // Some mathematical constants. We don't use pre-defined macros to ensure
  // that they are rounded exactly as we expect them to be.
  XNN_SIMD_CONST_F32(vpi, 3.1415927f);  // M_PI
  XNN_SIMD_CONST_F32(v2pi_inv, 0.15915494f); // 0.5 / M_PI
  XNN_SIMD_CONST_F32(vpi_half, 1.5707964f);  // M_PI / 2

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

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_3 = xnn_loadu_f32(input + 3 * xnn_simd_size_f32);
    input += 4;

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
    vx_0 = xnn_sub_f32(vpi_half, vx_0);
    vx_1 = xnn_sub_f32(vpi_half, vx_1);
    vx_2 = xnn_sub_f32(vpi_half, vx_2);
    vx_3 = xnn_sub_f32(vpi_half, vx_3);
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

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_2);
    xnn_storeu_f32(output + 3 * xnn_simd_size_f32, vy_3);
    output += 4;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Map the inputs to the interpolation range.
    xnn_simd_f32_t vx_div_2pi = xnn_mul_f32(vx, v2pi_inv);
    vx_div_2pi = xnn_round_f32(vx_div_2pi);
    vx = xnn_fnmadd_f32(vx_div_2pi, v2pi_hi, vx);
    vx = xnn_fnmadd_f32(vx_div_2pi, v2pi_lo, vx);
    vx = xnn_sub_f32(vpi_half, vx);
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
}

void xnn_f32_vcos_ukernel__scalar_rational_5_4_div_u8(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  // Some mathematical constants. We don't use pre-defined macros to ensure
  // that they are rounded exactly as we expect them to be.
  XNN_SIMD_CONST_F32(vpi, 3.1415927f);  // M_PI
  XNN_SIMD_CONST_F32(v2pi_inv, 0.15915494f); // 0.5 / M_PI
  XNN_SIMD_CONST_F32(vpi_half, 1.5707964f);  // M_PI / 2

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

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_3 = xnn_loadu_f32(input + 3 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_4 = xnn_loadu_f32(input + 4 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_5 = xnn_loadu_f32(input + 5 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_6 = xnn_loadu_f32(input + 6 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_7 = xnn_loadu_f32(input + 7 * xnn_simd_size_f32);
    input += 8;

    // Map the inputs to the interpolation range.
    xnn_simd_f32_t vx_div_2pi_0 = xnn_mul_f32(vx_0, v2pi_inv);
    xnn_simd_f32_t vx_div_2pi_1 = xnn_mul_f32(vx_1, v2pi_inv);
    xnn_simd_f32_t vx_div_2pi_2 = xnn_mul_f32(vx_2, v2pi_inv);
    xnn_simd_f32_t vx_div_2pi_3 = xnn_mul_f32(vx_3, v2pi_inv);
    xnn_simd_f32_t vx_div_2pi_4 = xnn_mul_f32(vx_4, v2pi_inv);
    xnn_simd_f32_t vx_div_2pi_5 = xnn_mul_f32(vx_5, v2pi_inv);
    xnn_simd_f32_t vx_div_2pi_6 = xnn_mul_f32(vx_6, v2pi_inv);
    xnn_simd_f32_t vx_div_2pi_7 = xnn_mul_f32(vx_7, v2pi_inv);
    vx_div_2pi_0 = xnn_round_f32(vx_div_2pi_0);
    vx_div_2pi_1 = xnn_round_f32(vx_div_2pi_1);
    vx_div_2pi_2 = xnn_round_f32(vx_div_2pi_2);
    vx_div_2pi_3 = xnn_round_f32(vx_div_2pi_3);
    vx_div_2pi_4 = xnn_round_f32(vx_div_2pi_4);
    vx_div_2pi_5 = xnn_round_f32(vx_div_2pi_5);
    vx_div_2pi_6 = xnn_round_f32(vx_div_2pi_6);
    vx_div_2pi_7 = xnn_round_f32(vx_div_2pi_7);
    vx_0 = xnn_fnmadd_f32(vx_div_2pi_0, v2pi_hi, vx_0);
    vx_1 = xnn_fnmadd_f32(vx_div_2pi_1, v2pi_hi, vx_1);
    vx_2 = xnn_fnmadd_f32(vx_div_2pi_2, v2pi_hi, vx_2);
    vx_3 = xnn_fnmadd_f32(vx_div_2pi_3, v2pi_hi, vx_3);
    vx_4 = xnn_fnmadd_f32(vx_div_2pi_4, v2pi_hi, vx_4);
    vx_5 = xnn_fnmadd_f32(vx_div_2pi_5, v2pi_hi, vx_5);
    vx_6 = xnn_fnmadd_f32(vx_div_2pi_6, v2pi_hi, vx_6);
    vx_7 = xnn_fnmadd_f32(vx_div_2pi_7, v2pi_hi, vx_7);
    vx_0 = xnn_fnmadd_f32(vx_div_2pi_0, v2pi_lo, vx_0);
    vx_1 = xnn_fnmadd_f32(vx_div_2pi_1, v2pi_lo, vx_1);
    vx_2 = xnn_fnmadd_f32(vx_div_2pi_2, v2pi_lo, vx_2);
    vx_3 = xnn_fnmadd_f32(vx_div_2pi_3, v2pi_lo, vx_3);
    vx_4 = xnn_fnmadd_f32(vx_div_2pi_4, v2pi_lo, vx_4);
    vx_5 = xnn_fnmadd_f32(vx_div_2pi_5, v2pi_lo, vx_5);
    vx_6 = xnn_fnmadd_f32(vx_div_2pi_6, v2pi_lo, vx_6);
    vx_7 = xnn_fnmadd_f32(vx_div_2pi_7, v2pi_lo, vx_7);
    vx_0 = xnn_sub_f32(vpi_half, vx_0);
    vx_1 = xnn_sub_f32(vpi_half, vx_1);
    vx_2 = xnn_sub_f32(vpi_half, vx_2);
    vx_3 = xnn_sub_f32(vpi_half, vx_3);
    vx_4 = xnn_sub_f32(vpi_half, vx_4);
    vx_5 = xnn_sub_f32(vpi_half, vx_5);
    vx_6 = xnn_sub_f32(vpi_half, vx_6);
    vx_7 = xnn_sub_f32(vpi_half, vx_7);
    vx_0 = xnn_min_f32(vx_0, xnn_sub_f32(vpi, vx_0));
    vx_1 = xnn_min_f32(vx_1, xnn_sub_f32(vpi, vx_1));
    vx_2 = xnn_min_f32(vx_2, xnn_sub_f32(vpi, vx_2));
    vx_3 = xnn_min_f32(vx_3, xnn_sub_f32(vpi, vx_3));
    vx_4 = xnn_min_f32(vx_4, xnn_sub_f32(vpi, vx_4));
    vx_5 = xnn_min_f32(vx_5, xnn_sub_f32(vpi, vx_5));
    vx_6 = xnn_min_f32(vx_6, xnn_sub_f32(vpi, vx_6));
    vx_7 = xnn_min_f32(vx_7, xnn_sub_f32(vpi, vx_7));
    vx_0 = xnn_max_f32(vx_0, xnn_sub_f32(xnn_neg_f32(vpi), vx_0));
    vx_1 = xnn_max_f32(vx_1, xnn_sub_f32(xnn_neg_f32(vpi), vx_1));
    vx_2 = xnn_max_f32(vx_2, xnn_sub_f32(xnn_neg_f32(vpi), vx_2));
    vx_3 = xnn_max_f32(vx_3, xnn_sub_f32(xnn_neg_f32(vpi), vx_3));
    vx_4 = xnn_max_f32(vx_4, xnn_sub_f32(xnn_neg_f32(vpi), vx_4));
    vx_5 = xnn_max_f32(vx_5, xnn_sub_f32(xnn_neg_f32(vpi), vx_5));
    vx_6 = xnn_max_f32(vx_6, xnn_sub_f32(xnn_neg_f32(vpi), vx_6));
    vx_7 = xnn_max_f32(vx_7, xnn_sub_f32(xnn_neg_f32(vpi), vx_7));
    vx_0 = xnn_min_f32(vx_0, xnn_sub_f32(vpi, vx_0));
    vx_1 = xnn_min_f32(vx_1, xnn_sub_f32(vpi, vx_1));
    vx_2 = xnn_min_f32(vx_2, xnn_sub_f32(vpi, vx_2));
    vx_3 = xnn_min_f32(vx_3, xnn_sub_f32(vpi, vx_3));
    vx_4 = xnn_min_f32(vx_4, xnn_sub_f32(vpi, vx_4));
    vx_5 = xnn_min_f32(vx_5, xnn_sub_f32(vpi, vx_5));
    vx_6 = xnn_min_f32(vx_6, xnn_sub_f32(vpi, vx_6));
    vx_7 = xnn_min_f32(vx_7, xnn_sub_f32(vpi, vx_7));

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
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx2_0, valpha_5, valpha_3);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx2_1, valpha_5, valpha_3);
    xnn_simd_f32_t vp_2 = xnn_fmadd_f32(vx2_2, valpha_5, valpha_3);
    xnn_simd_f32_t vp_3 = xnn_fmadd_f32(vx2_3, valpha_5, valpha_3);
    xnn_simd_f32_t vp_4 = xnn_fmadd_f32(vx2_4, valpha_5, valpha_3);
    xnn_simd_f32_t vp_5 = xnn_fmadd_f32(vx2_5, valpha_5, valpha_3);
    xnn_simd_f32_t vp_6 = xnn_fmadd_f32(vx2_6, valpha_5, valpha_3);
    xnn_simd_f32_t vp_7 = xnn_fmadd_f32(vx2_7, valpha_5, valpha_3);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, vone);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, vone);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, vone);
    vp_3 = xnn_fmadd_f32(vx2_3, vp_3, vone);
    vp_4 = xnn_fmadd_f32(vx2_4, vp_4, vone);
    vp_5 = xnn_fmadd_f32(vx2_5, vp_5, vone);
    vp_6 = xnn_fmadd_f32(vx2_6, vp_6, vone);
    vp_7 = xnn_fmadd_f32(vx2_7, vp_7, vone);
    vp_0 = xnn_mul_f32(vx_0, vp_0);
    vp_1 = xnn_mul_f32(vx_1, vp_1);
    vp_2 = xnn_mul_f32(vx_2, vp_2);
    vp_3 = xnn_mul_f32(vx_3, vp_3);
    vp_4 = xnn_mul_f32(vx_4, vp_4);
    vp_5 = xnn_mul_f32(vx_5, vp_5);
    vp_6 = xnn_mul_f32(vx_6, vp_6);
    vp_7 = xnn_mul_f32(vx_7, vp_7);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vx2_0, vbeta_4, vbeta_2);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vx2_1, vbeta_4, vbeta_2);
    xnn_simd_f32_t vq_2 = xnn_fmadd_f32(vx2_2, vbeta_4, vbeta_2);
    xnn_simd_f32_t vq_3 = xnn_fmadd_f32(vx2_3, vbeta_4, vbeta_2);
    xnn_simd_f32_t vq_4 = xnn_fmadd_f32(vx2_4, vbeta_4, vbeta_2);
    xnn_simd_f32_t vq_5 = xnn_fmadd_f32(vx2_5, vbeta_4, vbeta_2);
    xnn_simd_f32_t vq_6 = xnn_fmadd_f32(vx2_6, vbeta_4, vbeta_2);
    xnn_simd_f32_t vq_7 = xnn_fmadd_f32(vx2_7, vbeta_4, vbeta_2);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vone);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vone);
    vq_2 = xnn_fmadd_f32(vx2_2, vq_2, vone);
    vq_3 = xnn_fmadd_f32(vx2_3, vq_3, vone);
    vq_4 = xnn_fmadd_f32(vx2_4, vq_4, vone);
    vq_5 = xnn_fmadd_f32(vx2_5, vq_5, vone);
    vq_6 = xnn_fmadd_f32(vx2_6, vq_6, vone);
    vq_7 = xnn_fmadd_f32(vx2_7, vq_7, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy_0 = xnn_div_f32(vp_0, vq_0);
    const xnn_simd_f32_t vy_1 = xnn_div_f32(vp_1, vq_1);
    const xnn_simd_f32_t vy_2 = xnn_div_f32(vp_2, vq_2);
    const xnn_simd_f32_t vy_3 = xnn_div_f32(vp_3, vq_3);
    const xnn_simd_f32_t vy_4 = xnn_div_f32(vp_4, vq_4);
    const xnn_simd_f32_t vy_5 = xnn_div_f32(vp_5, vq_5);
    const xnn_simd_f32_t vy_6 = xnn_div_f32(vp_6, vq_6);
    const xnn_simd_f32_t vy_7 = xnn_div_f32(vp_7, vq_7);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vy_0);
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

    // Map the inputs to the interpolation range.
    xnn_simd_f32_t vx_div_2pi = xnn_mul_f32(vx, v2pi_inv);
    vx_div_2pi = xnn_round_f32(vx_div_2pi);
    vx = xnn_fnmadd_f32(vx_div_2pi, v2pi_hi, vx);
    vx = xnn_fnmadd_f32(vx_div_2pi, v2pi_lo, vx);
    vx = xnn_sub_f32(vpi_half, vx);
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
}
