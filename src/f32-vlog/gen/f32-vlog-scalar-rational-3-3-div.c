// Auto-generated file. Do not edit!
//   Template: src/f32-vlog/rational-3-3.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>

#include "xnnpack/simd/f32-scalar.h"

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"

// Define some mathematical constants in case they are not provided by `math.h`.
#ifndef M_LN2
#define M_LN2 0.69314718055994531
#endif  // M_LN2

// Extracts the exponent of the input `a` as a `float` value.
static XNN_INLINE xnn_simd_f32_t xnn_signed_getexp_f32(xnn_simd_f32_t a) {
  // The bits of IEE754 single-precision floating-point format are:
  //
  //   s | e e e e e e e e | m m m m m m m m m m m m m m m m m m m m m m m
  //
  // We start by masking out the sign and exponent and shifting it 8 bits to the
  // right arithmetically, i.e. extending by the leftmost sign bit:
  //
  //   s | s s s s s s s s | e e e e e e e e 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  //
  // These bits are then `or`-ed with `256.0f`, which has a biased exponent of
  // `135` and all mantissa bit set to zero. This is equivalent to adding the
  // biased integer exponent to `256.0`:
  //
  //   0 | 1 0 0 0 0 1 1 1 | e e e e e e e e 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  //
  // We can re-extract the exponent as a `float` value by subtracting `256.0`
  // plus the exponent bias `127.0`, i.e. `383.0`.
  //
  // Note that if the sign bit is `1`, we end up with the floating point bits:
  //
  //   1 | 1 1 1 1 1 1 1 1 | e e e e e e e e 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  //
  // Which is `-NaN` if the exponent is non-zero, and `-Inf` if the exponent is
  // zero (e.g. the input was `0.0f` or denormal).

  // Some useful constants.
  XNN_SIMD_CONST_F32(sign_mask, -0.0f);
  XNN_SIMD_CONST_U32(sign_and_exp_mask, 0xff800000);
  XNN_SIMD_CONST_F32(bias_256, 256.0f);
  XNN_SIMD_CONST_F32(bias_383, 383.0f);

  // If `a` is `0.0f`, flip its sign bit so that we return `-Inf`.
  a = xnn_or_f32(xnn_and_f32(xnn_cmpeq_f32(a, xnn_zero_f32()), sign_mask), a);

  // Extract the exponent and shift the exponent to the most significant bits of
  // the mantissa.
  const xnn_simd_f32_t exp =
      xnn_sra_f32(xnn_and_f32(a, sign_and_exp_mask), 8);

  // Add the shifted exponent to `256.0f` by copying its bits to the mantissa,
  // then subtract out `383.0f`, i.e. the original `256.0f` plus the `127`
  // exponent bias, resulting in the unbiased exponent.
  return xnn_sub_f32(xnn_or_f32(bias_256, exp), bias_383);
}


void xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u1(
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

  // Some useful constants.
  XNN_SIMD_CONST_F32(vone, 1.0f);
  XNN_SIMD_CONST_F32(vln2, M_LN2);
  XNN_SIMD_CONST_U32(vmantissa_bits_mask, 0x007FFFFFUL);

  // Note that these two values are not _exactly_ `(float)M_SQRT2` and
  // `(float)M_SQRT1_2`, but are instead chosen such that their product is
  // exactly `1.0f` when evaluated in `float` precision.
  XNN_SIMD_CONST_F32(vsqrt2, 1.4142134190e+00);
  XNN_SIMD_CONST_F32(vsqrt1_2, 7.0710688829e-01);

  // The monomial coefficients of the numerator polynomial.
  // XNN_SIMD_CONST_F32(valpha_0, 0.0f);
  // XNN_SIMD_CONST_F32(valpha_1, 1.0f);
  // XNN_SIMD_CONST_F32(valpha_2, 1.0f);
  XNN_SIMD_CONST_F32(valpha_3, 1.824996918440e-01);

  // The monomial coefficients of the denominator polynomial.
  // XNN_SIMD_CONST_F32(vbeta_0, 1.0f);
  XNN_SIMD_CONST_F32(vbeta_1, 1.5f);
  XNN_SIMD_CONST_F32(vbeta_2, 0.599170029163);
  XNN_SIMD_CONST_F32(vbeta_3, 0.049584995955);


  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Scale `x` with `sqrt(2)` so that the exponent is rounded up.
    vx = xnn_mul_f32(vx, vsqrt2);

    // Extract the exponent.
    const xnn_simd_f32_t vexp = xnn_signed_getexp_f32(vx);

    // Normalize `x` to an exponent of zero.
    vx = xnn_or_f32(xnn_and_f32(vx, vmantissa_bits_mask), vone);

    // Scale `x` back with `1/sqrt(2)` to move its range from `[1.0, 2.0)` to
    // `[sqrt(1/2), sqrt(2))`, and further subtract `1.0` so that it is around
    // zero, i.e. `[sqrt(1/2) - 1, sqrt(2) - 1)`, or `[−0.29289, 0.4142136)`.
    vx = xnn_sub_f32(xnn_mul_f32(vx, vsqrt1_2), vone);

    // In the following, we use a 3/2-degree rational polynomial to
    // approximate the (shifted) `log(x + 1)` on the (shifted) interval
    // `[sqrt(1/2) - 1, sqrt(2) - 1)`. The shifted interval is chosen so that
    // `f(0) = 0`.

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx, valpha_3, vone);
    vp = xnn_fmadd_f32(vx, vp, vone);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx, vbeta_3, vbeta_2);
    vq = xnn_fmadd_f32(vx, vq, vbeta_1);
    vq = xnn_fmadd_f32(vx, vq, vone);

    // Divide the numerator by the denominator.
    xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    // Put it all together, i.e. `log(x) = `log(2)*exp + y`.
    vy = xnn_fmadd_f32(vexp, vln2, vy);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
}

void xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u2(
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

  // Some useful constants.
  XNN_SIMD_CONST_F32(vone, 1.0f);
  XNN_SIMD_CONST_F32(vln2, M_LN2);
  XNN_SIMD_CONST_U32(vmantissa_bits_mask, 0x007FFFFFUL);

  // Note that these two values are not _exactly_ `(float)M_SQRT2` and
  // `(float)M_SQRT1_2`, but are instead chosen such that their product is
  // exactly `1.0f` when evaluated in `float` precision.
  XNN_SIMD_CONST_F32(vsqrt2, 1.4142134190e+00);
  XNN_SIMD_CONST_F32(vsqrt1_2, 7.0710688829e-01);

  // The monomial coefficients of the numerator polynomial.
  // XNN_SIMD_CONST_F32(valpha_0, 0.0f);
  // XNN_SIMD_CONST_F32(valpha_1, 1.0f);
  // XNN_SIMD_CONST_F32(valpha_2, 1.0f);
  XNN_SIMD_CONST_F32(valpha_3, 1.824996918440e-01);

  // The monomial coefficients of the denominator polynomial.
  // XNN_SIMD_CONST_F32(vbeta_0, 1.0f);
  XNN_SIMD_CONST_F32(vbeta_1, 1.5f);
  XNN_SIMD_CONST_F32(vbeta_2, 0.599170029163);
  XNN_SIMD_CONST_F32(vbeta_3, 0.049584995955);


  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 2;

    // Scale `x` with `sqrt(2)` so that the exponent is rounded up.
    vx_0 = xnn_mul_f32(vx_0, vsqrt2);
    vx_1 = xnn_mul_f32(vx_1, vsqrt2);

    // Extract the exponent.
    const xnn_simd_f32_t vexp_0 = xnn_signed_getexp_f32(vx_0);
    const xnn_simd_f32_t vexp_1 = xnn_signed_getexp_f32(vx_1);

    // Normalize `x` to an exponent of zero.
    vx_0 = xnn_or_f32(xnn_and_f32(vx_0, vmantissa_bits_mask), vone);
    vx_1 = xnn_or_f32(xnn_and_f32(vx_1, vmantissa_bits_mask), vone);

    // Scale `x` back with `1/sqrt(2)` to move its range from `[1.0, 2.0)` to
    // `[sqrt(1/2), sqrt(2))`, and further subtract `1.0` so that it is around
    // zero, i.e. `[sqrt(1/2) - 1, sqrt(2) - 1)`, or `[−0.29289, 0.4142136)`.
    vx_0 = xnn_sub_f32(xnn_mul_f32(vx_0, vsqrt1_2), vone);
    vx_1 = xnn_sub_f32(xnn_mul_f32(vx_1, vsqrt1_2), vone);

    // In the following, we use a 3/2-degree rational polynomial to
    // approximate the (shifted) `log(x + 1)` on the (shifted) interval
    // `[sqrt(1/2) - 1, sqrt(2) - 1)`. The shifted interval is chosen so that
    // `f(0) = 0`.

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx_0, valpha_3, vone);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx_1, valpha_3, vone);
    vp_0 = xnn_fmadd_f32(vx_0, vp_0, vone);
    vp_1 = xnn_fmadd_f32(vx_1, vp_1, vone);
    vp_0 = xnn_mul_f32(vx_0, vp_0);
    vp_1 = xnn_mul_f32(vx_1, vp_1);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vx_0, vbeta_3, vbeta_2);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vx_1, vbeta_3, vbeta_2);
    vq_0 = xnn_fmadd_f32(vx_0, vq_0, vbeta_1);
    vq_1 = xnn_fmadd_f32(vx_1, vq_1, vbeta_1);
    vq_0 = xnn_fmadd_f32(vx_0, vq_0, vone);
    vq_1 = xnn_fmadd_f32(vx_1, vq_1, vone);

    // Divide the numerator by the denominator.
    xnn_simd_f32_t vy_0 = xnn_div_f32(vp_0, vq_0);
    xnn_simd_f32_t vy_1 = xnn_div_f32(vp_1, vq_1);

    // Put it all together, i.e. `log(x) = `log(2)*exp + y`.
    vy_0 = xnn_fmadd_f32(vexp_0, vln2, vy_0);
    vy_1 = xnn_fmadd_f32(vexp_1, vln2, vy_1);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    output += 2;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Scale `x` with `sqrt(2)` so that the exponent is rounded up.
    vx = xnn_mul_f32(vx, vsqrt2);

    // Extract the exponent.
    const xnn_simd_f32_t vexp = xnn_signed_getexp_f32(vx);

    // Normalize `x` to an exponent of zero.
    vx = xnn_or_f32(xnn_and_f32(vx, vmantissa_bits_mask), vone);

    // Scale `x` back with `1/sqrt(2)` to move its range from `[1.0, 2.0)` to
    // `[sqrt(1/2), sqrt(2))`, and further subtract `1.0` so that it is around
    // zero, i.e. `[sqrt(1/2) - 1, sqrt(2) - 1)`, or `[−0.29289, 0.4142136)`.
    vx = xnn_sub_f32(xnn_mul_f32(vx, vsqrt1_2), vone);

    // In the following, we use a 3/2-degree rational polynomial to
    // approximate the (shifted) `log(x + 1)` on the (shifted) interval
    // `[sqrt(1/2) - 1, sqrt(2) - 1)`. The shifted interval is chosen so that
    // `f(0) = 0`.

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx, valpha_3, vone);
    vp = xnn_fmadd_f32(vx, vp, vone);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx, vbeta_3, vbeta_2);
    vq = xnn_fmadd_f32(vx, vq, vbeta_1);
    vq = xnn_fmadd_f32(vx, vq, vone);

    // Divide the numerator by the denominator.
    xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    // Put it all together, i.e. `log(x) = `log(2)*exp + y`.
    vy = xnn_fmadd_f32(vexp, vln2, vy);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
}

void xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u4(
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

  // Some useful constants.
  XNN_SIMD_CONST_F32(vone, 1.0f);
  XNN_SIMD_CONST_F32(vln2, M_LN2);
  XNN_SIMD_CONST_U32(vmantissa_bits_mask, 0x007FFFFFUL);

  // Note that these two values are not _exactly_ `(float)M_SQRT2` and
  // `(float)M_SQRT1_2`, but are instead chosen such that their product is
  // exactly `1.0f` when evaluated in `float` precision.
  XNN_SIMD_CONST_F32(vsqrt2, 1.4142134190e+00);
  XNN_SIMD_CONST_F32(vsqrt1_2, 7.0710688829e-01);

  // The monomial coefficients of the numerator polynomial.
  // XNN_SIMD_CONST_F32(valpha_0, 0.0f);
  // XNN_SIMD_CONST_F32(valpha_1, 1.0f);
  // XNN_SIMD_CONST_F32(valpha_2, 1.0f);
  XNN_SIMD_CONST_F32(valpha_3, 1.824996918440e-01);

  // The monomial coefficients of the denominator polynomial.
  // XNN_SIMD_CONST_F32(vbeta_0, 1.0f);
  XNN_SIMD_CONST_F32(vbeta_1, 1.5f);
  XNN_SIMD_CONST_F32(vbeta_2, 0.599170029163);
  XNN_SIMD_CONST_F32(vbeta_3, 0.049584995955);


  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_3 = xnn_loadu_f32(input + 3 * xnn_simd_size_f32);
    input += 4;

    // Scale `x` with `sqrt(2)` so that the exponent is rounded up.
    vx_0 = xnn_mul_f32(vx_0, vsqrt2);
    vx_1 = xnn_mul_f32(vx_1, vsqrt2);
    vx_2 = xnn_mul_f32(vx_2, vsqrt2);
    vx_3 = xnn_mul_f32(vx_3, vsqrt2);

    // Extract the exponent.
    const xnn_simd_f32_t vexp_0 = xnn_signed_getexp_f32(vx_0);
    const xnn_simd_f32_t vexp_1 = xnn_signed_getexp_f32(vx_1);
    const xnn_simd_f32_t vexp_2 = xnn_signed_getexp_f32(vx_2);
    const xnn_simd_f32_t vexp_3 = xnn_signed_getexp_f32(vx_3);

    // Normalize `x` to an exponent of zero.
    vx_0 = xnn_or_f32(xnn_and_f32(vx_0, vmantissa_bits_mask), vone);
    vx_1 = xnn_or_f32(xnn_and_f32(vx_1, vmantissa_bits_mask), vone);
    vx_2 = xnn_or_f32(xnn_and_f32(vx_2, vmantissa_bits_mask), vone);
    vx_3 = xnn_or_f32(xnn_and_f32(vx_3, vmantissa_bits_mask), vone);

    // Scale `x` back with `1/sqrt(2)` to move its range from `[1.0, 2.0)` to
    // `[sqrt(1/2), sqrt(2))`, and further subtract `1.0` so that it is around
    // zero, i.e. `[sqrt(1/2) - 1, sqrt(2) - 1)`, or `[−0.29289, 0.4142136)`.
    vx_0 = xnn_sub_f32(xnn_mul_f32(vx_0, vsqrt1_2), vone);
    vx_1 = xnn_sub_f32(xnn_mul_f32(vx_1, vsqrt1_2), vone);
    vx_2 = xnn_sub_f32(xnn_mul_f32(vx_2, vsqrt1_2), vone);
    vx_3 = xnn_sub_f32(xnn_mul_f32(vx_3, vsqrt1_2), vone);

    // In the following, we use a 3/2-degree rational polynomial to
    // approximate the (shifted) `log(x + 1)` on the (shifted) interval
    // `[sqrt(1/2) - 1, sqrt(2) - 1)`. The shifted interval is chosen so that
    // `f(0) = 0`.

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx_0, valpha_3, vone);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx_1, valpha_3, vone);
    xnn_simd_f32_t vp_2 = xnn_fmadd_f32(vx_2, valpha_3, vone);
    xnn_simd_f32_t vp_3 = xnn_fmadd_f32(vx_3, valpha_3, vone);
    vp_0 = xnn_fmadd_f32(vx_0, vp_0, vone);
    vp_1 = xnn_fmadd_f32(vx_1, vp_1, vone);
    vp_2 = xnn_fmadd_f32(vx_2, vp_2, vone);
    vp_3 = xnn_fmadd_f32(vx_3, vp_3, vone);
    vp_0 = xnn_mul_f32(vx_0, vp_0);
    vp_1 = xnn_mul_f32(vx_1, vp_1);
    vp_2 = xnn_mul_f32(vx_2, vp_2);
    vp_3 = xnn_mul_f32(vx_3, vp_3);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vx_0, vbeta_3, vbeta_2);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vx_1, vbeta_3, vbeta_2);
    xnn_simd_f32_t vq_2 = xnn_fmadd_f32(vx_2, vbeta_3, vbeta_2);
    xnn_simd_f32_t vq_3 = xnn_fmadd_f32(vx_3, vbeta_3, vbeta_2);
    vq_0 = xnn_fmadd_f32(vx_0, vq_0, vbeta_1);
    vq_1 = xnn_fmadd_f32(vx_1, vq_1, vbeta_1);
    vq_2 = xnn_fmadd_f32(vx_2, vq_2, vbeta_1);
    vq_3 = xnn_fmadd_f32(vx_3, vq_3, vbeta_1);
    vq_0 = xnn_fmadd_f32(vx_0, vq_0, vone);
    vq_1 = xnn_fmadd_f32(vx_1, vq_1, vone);
    vq_2 = xnn_fmadd_f32(vx_2, vq_2, vone);
    vq_3 = xnn_fmadd_f32(vx_3, vq_3, vone);

    // Divide the numerator by the denominator.
    xnn_simd_f32_t vy_0 = xnn_div_f32(vp_0, vq_0);
    xnn_simd_f32_t vy_1 = xnn_div_f32(vp_1, vq_1);
    xnn_simd_f32_t vy_2 = xnn_div_f32(vp_2, vq_2);
    xnn_simd_f32_t vy_3 = xnn_div_f32(vp_3, vq_3);

    // Put it all together, i.e. `log(x) = `log(2)*exp + y`.
    vy_0 = xnn_fmadd_f32(vexp_0, vln2, vy_0);
    vy_1 = xnn_fmadd_f32(vexp_1, vln2, vy_1);
    vy_2 = xnn_fmadd_f32(vexp_2, vln2, vy_2);
    vy_3 = xnn_fmadd_f32(vexp_3, vln2, vy_3);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_2);
    xnn_storeu_f32(output + 3 * xnn_simd_size_f32, vy_3);
    output += 4;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Scale `x` with `sqrt(2)` so that the exponent is rounded up.
    vx = xnn_mul_f32(vx, vsqrt2);

    // Extract the exponent.
    const xnn_simd_f32_t vexp = xnn_signed_getexp_f32(vx);

    // Normalize `x` to an exponent of zero.
    vx = xnn_or_f32(xnn_and_f32(vx, vmantissa_bits_mask), vone);

    // Scale `x` back with `1/sqrt(2)` to move its range from `[1.0, 2.0)` to
    // `[sqrt(1/2), sqrt(2))`, and further subtract `1.0` so that it is around
    // zero, i.e. `[sqrt(1/2) - 1, sqrt(2) - 1)`, or `[−0.29289, 0.4142136)`.
    vx = xnn_sub_f32(xnn_mul_f32(vx, vsqrt1_2), vone);

    // In the following, we use a 3/2-degree rational polynomial to
    // approximate the (shifted) `log(x + 1)` on the (shifted) interval
    // `[sqrt(1/2) - 1, sqrt(2) - 1)`. The shifted interval is chosen so that
    // `f(0) = 0`.

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx, valpha_3, vone);
    vp = xnn_fmadd_f32(vx, vp, vone);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx, vbeta_3, vbeta_2);
    vq = xnn_fmadd_f32(vx, vq, vbeta_1);
    vq = xnn_fmadd_f32(vx, vq, vone);

    // Divide the numerator by the denominator.
    xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    // Put it all together, i.e. `log(x) = `log(2)*exp + y`.
    vy = xnn_fmadd_f32(vexp, vln2, vy);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
}

void xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u8(
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

  // Some useful constants.
  XNN_SIMD_CONST_F32(vone, 1.0f);
  XNN_SIMD_CONST_F32(vln2, M_LN2);
  XNN_SIMD_CONST_U32(vmantissa_bits_mask, 0x007FFFFFUL);

  // Note that these two values are not _exactly_ `(float)M_SQRT2` and
  // `(float)M_SQRT1_2`, but are instead chosen such that their product is
  // exactly `1.0f` when evaluated in `float` precision.
  XNN_SIMD_CONST_F32(vsqrt2, 1.4142134190e+00);
  XNN_SIMD_CONST_F32(vsqrt1_2, 7.0710688829e-01);

  // The monomial coefficients of the numerator polynomial.
  // XNN_SIMD_CONST_F32(valpha_0, 0.0f);
  // XNN_SIMD_CONST_F32(valpha_1, 1.0f);
  // XNN_SIMD_CONST_F32(valpha_2, 1.0f);
  XNN_SIMD_CONST_F32(valpha_3, 1.824996918440e-01);

  // The monomial coefficients of the denominator polynomial.
  // XNN_SIMD_CONST_F32(vbeta_0, 1.0f);
  XNN_SIMD_CONST_F32(vbeta_1, 1.5f);
  XNN_SIMD_CONST_F32(vbeta_2, 0.599170029163);
  XNN_SIMD_CONST_F32(vbeta_3, 0.049584995955);


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

    // Scale `x` with `sqrt(2)` so that the exponent is rounded up.
    vx_0 = xnn_mul_f32(vx_0, vsqrt2);
    vx_1 = xnn_mul_f32(vx_1, vsqrt2);
    vx_2 = xnn_mul_f32(vx_2, vsqrt2);
    vx_3 = xnn_mul_f32(vx_3, vsqrt2);
    vx_4 = xnn_mul_f32(vx_4, vsqrt2);
    vx_5 = xnn_mul_f32(vx_5, vsqrt2);
    vx_6 = xnn_mul_f32(vx_6, vsqrt2);
    vx_7 = xnn_mul_f32(vx_7, vsqrt2);

    // Extract the exponent.
    const xnn_simd_f32_t vexp_0 = xnn_signed_getexp_f32(vx_0);
    const xnn_simd_f32_t vexp_1 = xnn_signed_getexp_f32(vx_1);
    const xnn_simd_f32_t vexp_2 = xnn_signed_getexp_f32(vx_2);
    const xnn_simd_f32_t vexp_3 = xnn_signed_getexp_f32(vx_3);
    const xnn_simd_f32_t vexp_4 = xnn_signed_getexp_f32(vx_4);
    const xnn_simd_f32_t vexp_5 = xnn_signed_getexp_f32(vx_5);
    const xnn_simd_f32_t vexp_6 = xnn_signed_getexp_f32(vx_6);
    const xnn_simd_f32_t vexp_7 = xnn_signed_getexp_f32(vx_7);

    // Normalize `x` to an exponent of zero.
    vx_0 = xnn_or_f32(xnn_and_f32(vx_0, vmantissa_bits_mask), vone);
    vx_1 = xnn_or_f32(xnn_and_f32(vx_1, vmantissa_bits_mask), vone);
    vx_2 = xnn_or_f32(xnn_and_f32(vx_2, vmantissa_bits_mask), vone);
    vx_3 = xnn_or_f32(xnn_and_f32(vx_3, vmantissa_bits_mask), vone);
    vx_4 = xnn_or_f32(xnn_and_f32(vx_4, vmantissa_bits_mask), vone);
    vx_5 = xnn_or_f32(xnn_and_f32(vx_5, vmantissa_bits_mask), vone);
    vx_6 = xnn_or_f32(xnn_and_f32(vx_6, vmantissa_bits_mask), vone);
    vx_7 = xnn_or_f32(xnn_and_f32(vx_7, vmantissa_bits_mask), vone);

    // Scale `x` back with `1/sqrt(2)` to move its range from `[1.0, 2.0)` to
    // `[sqrt(1/2), sqrt(2))`, and further subtract `1.0` so that it is around
    // zero, i.e. `[sqrt(1/2) - 1, sqrt(2) - 1)`, or `[−0.29289, 0.4142136)`.
    vx_0 = xnn_sub_f32(xnn_mul_f32(vx_0, vsqrt1_2), vone);
    vx_1 = xnn_sub_f32(xnn_mul_f32(vx_1, vsqrt1_2), vone);
    vx_2 = xnn_sub_f32(xnn_mul_f32(vx_2, vsqrt1_2), vone);
    vx_3 = xnn_sub_f32(xnn_mul_f32(vx_3, vsqrt1_2), vone);
    vx_4 = xnn_sub_f32(xnn_mul_f32(vx_4, vsqrt1_2), vone);
    vx_5 = xnn_sub_f32(xnn_mul_f32(vx_5, vsqrt1_2), vone);
    vx_6 = xnn_sub_f32(xnn_mul_f32(vx_6, vsqrt1_2), vone);
    vx_7 = xnn_sub_f32(xnn_mul_f32(vx_7, vsqrt1_2), vone);

    // In the following, we use a 3/2-degree rational polynomial to
    // approximate the (shifted) `log(x + 1)` on the (shifted) interval
    // `[sqrt(1/2) - 1, sqrt(2) - 1)`. The shifted interval is chosen so that
    // `f(0) = 0`.

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx_0, valpha_3, vone);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx_1, valpha_3, vone);
    xnn_simd_f32_t vp_2 = xnn_fmadd_f32(vx_2, valpha_3, vone);
    xnn_simd_f32_t vp_3 = xnn_fmadd_f32(vx_3, valpha_3, vone);
    xnn_simd_f32_t vp_4 = xnn_fmadd_f32(vx_4, valpha_3, vone);
    xnn_simd_f32_t vp_5 = xnn_fmadd_f32(vx_5, valpha_3, vone);
    xnn_simd_f32_t vp_6 = xnn_fmadd_f32(vx_6, valpha_3, vone);
    xnn_simd_f32_t vp_7 = xnn_fmadd_f32(vx_7, valpha_3, vone);
    vp_0 = xnn_fmadd_f32(vx_0, vp_0, vone);
    vp_1 = xnn_fmadd_f32(vx_1, vp_1, vone);
    vp_2 = xnn_fmadd_f32(vx_2, vp_2, vone);
    vp_3 = xnn_fmadd_f32(vx_3, vp_3, vone);
    vp_4 = xnn_fmadd_f32(vx_4, vp_4, vone);
    vp_5 = xnn_fmadd_f32(vx_5, vp_5, vone);
    vp_6 = xnn_fmadd_f32(vx_6, vp_6, vone);
    vp_7 = xnn_fmadd_f32(vx_7, vp_7, vone);
    vp_0 = xnn_mul_f32(vx_0, vp_0);
    vp_1 = xnn_mul_f32(vx_1, vp_1);
    vp_2 = xnn_mul_f32(vx_2, vp_2);
    vp_3 = xnn_mul_f32(vx_3, vp_3);
    vp_4 = xnn_mul_f32(vx_4, vp_4);
    vp_5 = xnn_mul_f32(vx_5, vp_5);
    vp_6 = xnn_mul_f32(vx_6, vp_6);
    vp_7 = xnn_mul_f32(vx_7, vp_7);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vx_0, vbeta_3, vbeta_2);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vx_1, vbeta_3, vbeta_2);
    xnn_simd_f32_t vq_2 = xnn_fmadd_f32(vx_2, vbeta_3, vbeta_2);
    xnn_simd_f32_t vq_3 = xnn_fmadd_f32(vx_3, vbeta_3, vbeta_2);
    xnn_simd_f32_t vq_4 = xnn_fmadd_f32(vx_4, vbeta_3, vbeta_2);
    xnn_simd_f32_t vq_5 = xnn_fmadd_f32(vx_5, vbeta_3, vbeta_2);
    xnn_simd_f32_t vq_6 = xnn_fmadd_f32(vx_6, vbeta_3, vbeta_2);
    xnn_simd_f32_t vq_7 = xnn_fmadd_f32(vx_7, vbeta_3, vbeta_2);
    vq_0 = xnn_fmadd_f32(vx_0, vq_0, vbeta_1);
    vq_1 = xnn_fmadd_f32(vx_1, vq_1, vbeta_1);
    vq_2 = xnn_fmadd_f32(vx_2, vq_2, vbeta_1);
    vq_3 = xnn_fmadd_f32(vx_3, vq_3, vbeta_1);
    vq_4 = xnn_fmadd_f32(vx_4, vq_4, vbeta_1);
    vq_5 = xnn_fmadd_f32(vx_5, vq_5, vbeta_1);
    vq_6 = xnn_fmadd_f32(vx_6, vq_6, vbeta_1);
    vq_7 = xnn_fmadd_f32(vx_7, vq_7, vbeta_1);
    vq_0 = xnn_fmadd_f32(vx_0, vq_0, vone);
    vq_1 = xnn_fmadd_f32(vx_1, vq_1, vone);
    vq_2 = xnn_fmadd_f32(vx_2, vq_2, vone);
    vq_3 = xnn_fmadd_f32(vx_3, vq_3, vone);
    vq_4 = xnn_fmadd_f32(vx_4, vq_4, vone);
    vq_5 = xnn_fmadd_f32(vx_5, vq_5, vone);
    vq_6 = xnn_fmadd_f32(vx_6, vq_6, vone);
    vq_7 = xnn_fmadd_f32(vx_7, vq_7, vone);

    // Divide the numerator by the denominator.
    xnn_simd_f32_t vy_0 = xnn_div_f32(vp_0, vq_0);
    xnn_simd_f32_t vy_1 = xnn_div_f32(vp_1, vq_1);
    xnn_simd_f32_t vy_2 = xnn_div_f32(vp_2, vq_2);
    xnn_simd_f32_t vy_3 = xnn_div_f32(vp_3, vq_3);
    xnn_simd_f32_t vy_4 = xnn_div_f32(vp_4, vq_4);
    xnn_simd_f32_t vy_5 = xnn_div_f32(vp_5, vq_5);
    xnn_simd_f32_t vy_6 = xnn_div_f32(vp_6, vq_6);
    xnn_simd_f32_t vy_7 = xnn_div_f32(vp_7, vq_7);

    // Put it all together, i.e. `log(x) = `log(2)*exp + y`.
    vy_0 = xnn_fmadd_f32(vexp_0, vln2, vy_0);
    vy_1 = xnn_fmadd_f32(vexp_1, vln2, vy_1);
    vy_2 = xnn_fmadd_f32(vexp_2, vln2, vy_2);
    vy_3 = xnn_fmadd_f32(vexp_3, vln2, vy_3);
    vy_4 = xnn_fmadd_f32(vexp_4, vln2, vy_4);
    vy_5 = xnn_fmadd_f32(vexp_5, vln2, vy_5);
    vy_6 = xnn_fmadd_f32(vexp_6, vln2, vy_6);
    vy_7 = xnn_fmadd_f32(vexp_7, vln2, vy_7);

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

    // Scale `x` with `sqrt(2)` so that the exponent is rounded up.
    vx = xnn_mul_f32(vx, vsqrt2);

    // Extract the exponent.
    const xnn_simd_f32_t vexp = xnn_signed_getexp_f32(vx);

    // Normalize `x` to an exponent of zero.
    vx = xnn_or_f32(xnn_and_f32(vx, vmantissa_bits_mask), vone);

    // Scale `x` back with `1/sqrt(2)` to move its range from `[1.0, 2.0)` to
    // `[sqrt(1/2), sqrt(2))`, and further subtract `1.0` so that it is around
    // zero, i.e. `[sqrt(1/2) - 1, sqrt(2) - 1)`, or `[−0.29289, 0.4142136)`.
    vx = xnn_sub_f32(xnn_mul_f32(vx, vsqrt1_2), vone);

    // In the following, we use a 3/2-degree rational polynomial to
    // approximate the (shifted) `log(x + 1)` on the (shifted) interval
    // `[sqrt(1/2) - 1, sqrt(2) - 1)`. The shifted interval is chosen so that
    // `f(0) = 0`.

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx, valpha_3, vone);
    vp = xnn_fmadd_f32(vx, vp, vone);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx, vbeta_3, vbeta_2);
    vq = xnn_fmadd_f32(vx, vq, vbeta_1);
    vq = xnn_fmadd_f32(vx, vq, vone);

    // Divide the numerator by the denominator.
    xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    // Put it all together, i.e. `log(x) = `log(2)*exp + y`.
    vy = xnn_fmadd_f32(vexp, vln2, vy);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
}
