// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vexp/rational-3-2.c.in
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
#include "src/xnnpack/microparams.h"


static XNN_INLINE xnn_simd_f32_t xnn_setexp_f32(xnn_simd_f32_t vx) {
  // If `x` is an floating point value in the range [-127, 128], then
  // `(x + magic) << 23` will generate the floating point value corresponding
  // to `2^round(x)` (2^-127 and 2^128 will flush to zero and infinity,
  // respectively).
  XNN_SIMD_CONST_F32(vmagic, 8388735.0f);  // 2^23 + 127.
  return xnn_sll_f32(xnn_add_f32(vx, vmagic), 23);
}

// Quick-and-dirty round to nearest, only works for floats in the range
// `[2^-22, 2^22)`.
static XNN_INLINE xnn_simd_f32_t xnn_qd_round_f32(xnn_simd_f32_t vx) {
  // If `x` is an floating point value in the range `[2^-22, 2^22)`, then
  // `(x + magic) - magic`` will generate the floating point value corresponding
  // to `round(x)`.
  XNN_SIMD_CONST_F32(vmagic, 12582912.0f);  // 2^23 + 2^22.
  return xnn_sub_f32(xnn_add_f32(vmagic, vx), vmagic);
}

void xnn_f32_vexp_ukernel__scalar_rational_3_2_div_u1(
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

  // The monomial coefficients of the numerator polynomial (`valpha_0` = 1.0).
  XNN_SIMD_CONST_F32(valpha_1, 4.1594290733e-01f);
  XNN_SIMD_CONST_F32(valpha_2, 7.2068706155e-02f);
  XNN_SIMD_CONST_F32(valpha_3, 5.5380910635e-03f);

  // The monomial coefficients of the denominator polynomial (`vbeta_01 = 1.0).
  XNN_SIMD_CONST_F32(vbeta_1, -2.7720427513e-01f);
  XNN_SIMD_CONST_F32(vbeta_2, 2.3986088112e-02f);

  // Some useful constants.
  XNN_SIMD_CONST_F32(vlog2e, 1.44269504089f);
  XNN_SIMD_CONST_F32(v128, 128.0f);
  XNN_SIMD_CONST_F32(vm127, -127.0f);
  XNN_SIMD_CONST_F32(vone, 1.0f);

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    xnn_simd_f32_t vz_prime = xnn_mul_f32(vx, vlog2e);
    vz_prime = xnn_min_f32(xnn_max_f32(vz_prime, vm127), v128);

    // Decompose x * log2e into `z` (integer part) and `r` (remainder).
    const xnn_simd_f32_t vz = xnn_qd_round_f32(vz_prime);
    const xnn_simd_f32_t vr = xnn_sub_f32(vz_prime, vz);

    // Compute 2^z.
    const xnn_simd_f32_t v2z = xnn_setexp_f32(vz);

    // Evaluate the numerator polynomial p(f).
    xnn_simd_f32_t vp = xnn_fmadd_f32(vr, valpha_3, valpha_2);
    vp = xnn_fmadd_f32(vr, vp, valpha_1);
    vp = xnn_fmadd_f32(vr, vp, vone);

    // Evaluate the denominator polynomial q(r).
    xnn_simd_f32_t vq = xnn_fmadd_f32(vr, vbeta_2, vbeta_1);
    vq = xnn_fmadd_f32(vr, vq, vone);

    // Divide the numerator by the denominator, obtaining 2^r.
    const xnn_simd_f32_t v2r =  xnn_div_f32(vp, vq);

    // Compute 2^z * 2^r.
    const xnn_simd_f32_t vy = xnn_mul_f32(v2z, v2r);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
}

void xnn_f32_vexp_ukernel__scalar_rational_3_2_div_u2(
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

  // The monomial coefficients of the numerator polynomial (`valpha_0` = 1.0).
  XNN_SIMD_CONST_F32(valpha_1, 4.1594290733e-01f);
  XNN_SIMD_CONST_F32(valpha_2, 7.2068706155e-02f);
  XNN_SIMD_CONST_F32(valpha_3, 5.5380910635e-03f);

  // The monomial coefficients of the denominator polynomial (`vbeta_01 = 1.0).
  XNN_SIMD_CONST_F32(vbeta_1, -2.7720427513e-01f);
  XNN_SIMD_CONST_F32(vbeta_2, 2.3986088112e-02f);

  // Some useful constants.
  XNN_SIMD_CONST_F32(vlog2e, 1.44269504089f);
  XNN_SIMD_CONST_F32(v128, 128.0f);
  XNN_SIMD_CONST_F32(vm127, -127.0f);
  XNN_SIMD_CONST_F32(vone, 1.0f);

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 2;

    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    xnn_simd_f32_t vz_prime_0 = xnn_mul_f32(vx_0, vlog2e);
    xnn_simd_f32_t vz_prime_1 = xnn_mul_f32(vx_1, vlog2e);
    vz_prime_0 = xnn_min_f32(xnn_max_f32(vz_prime_0, vm127), v128);
    vz_prime_1 = xnn_min_f32(xnn_max_f32(vz_prime_1, vm127), v128);

    // Decompose x * log2e into `z` (integer part) and `r` (remainder).
    const xnn_simd_f32_t vz_0 = xnn_qd_round_f32(vz_prime_0);
    const xnn_simd_f32_t vz_1 = xnn_qd_round_f32(vz_prime_1);
    const xnn_simd_f32_t vr_0 = xnn_sub_f32(vz_prime_0, vz_0);
    const xnn_simd_f32_t vr_1 = xnn_sub_f32(vz_prime_1, vz_1);

    // Compute 2^z.
    const xnn_simd_f32_t v2z_0 = xnn_setexp_f32(vz_0);
    const xnn_simd_f32_t v2z_1 = xnn_setexp_f32(vz_1);

    // Evaluate the numerator polynomial p(f).
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vr_0, valpha_3, valpha_2);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vr_1, valpha_3, valpha_2);
    vp_0 = xnn_fmadd_f32(vr_0, vp_0, valpha_1);
    vp_1 = xnn_fmadd_f32(vr_1, vp_1, valpha_1);
    vp_0 = xnn_fmadd_f32(vr_0, vp_0, vone);
    vp_1 = xnn_fmadd_f32(vr_1, vp_1, vone);

    // Evaluate the denominator polynomial q(r).
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vr_0, vbeta_2, vbeta_1);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vr_1, vbeta_2, vbeta_1);
    vq_0 = xnn_fmadd_f32(vr_0, vq_0, vone);
    vq_1 = xnn_fmadd_f32(vr_1, vq_1, vone);

    // Divide the numerator by the denominator, obtaining 2^r.
    const xnn_simd_f32_t v2r_0 =  xnn_div_f32(vp_0, vq_0);
    const xnn_simd_f32_t v2r_1 =  xnn_div_f32(vp_1, vq_1);

    // Compute 2^z * 2^r.
    const xnn_simd_f32_t vy_0 = xnn_mul_f32(v2z_0, v2r_0);
    const xnn_simd_f32_t vy_1 = xnn_mul_f32(v2z_1, v2r_1);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    output += 2;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    xnn_simd_f32_t vz_prime = xnn_mul_f32(vx, vlog2e);
    vz_prime = xnn_min_f32(xnn_max_f32(vz_prime, vm127), v128);

    // Decompose x * log2e into `z` (integer part) and `r` (remainder).
    const xnn_simd_f32_t vz = xnn_qd_round_f32(vz_prime);
    const xnn_simd_f32_t vr = xnn_sub_f32(vz_prime, vz);

    // Compute 2^z.
    const xnn_simd_f32_t v2z = xnn_setexp_f32(vz);

    // Evaluate the numerator polynomial p(f).
    xnn_simd_f32_t vp = xnn_fmadd_f32(vr, valpha_3, valpha_2);
    vp = xnn_fmadd_f32(vr, vp, valpha_1);
    vp = xnn_fmadd_f32(vr, vp, vone);

    // Evaluate the denominator polynomial q(r).
    xnn_simd_f32_t vq = xnn_fmadd_f32(vr, vbeta_2, vbeta_1);
    vq = xnn_fmadd_f32(vr, vq, vone);

    // Divide the numerator by the denominator, obtaining 2^r.
    const xnn_simd_f32_t v2r =  xnn_div_f32(vp, vq);

    // Compute 2^z * 2^r.
    const xnn_simd_f32_t vy = xnn_mul_f32(v2z, v2r);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
}

void xnn_f32_vexp_ukernel__scalar_rational_3_2_div_u4(
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

  // The monomial coefficients of the numerator polynomial (`valpha_0` = 1.0).
  XNN_SIMD_CONST_F32(valpha_1, 4.1594290733e-01f);
  XNN_SIMD_CONST_F32(valpha_2, 7.2068706155e-02f);
  XNN_SIMD_CONST_F32(valpha_3, 5.5380910635e-03f);

  // The monomial coefficients of the denominator polynomial (`vbeta_01 = 1.0).
  XNN_SIMD_CONST_F32(vbeta_1, -2.7720427513e-01f);
  XNN_SIMD_CONST_F32(vbeta_2, 2.3986088112e-02f);

  // Some useful constants.
  XNN_SIMD_CONST_F32(vlog2e, 1.44269504089f);
  XNN_SIMD_CONST_F32(v128, 128.0f);
  XNN_SIMD_CONST_F32(vm127, -127.0f);
  XNN_SIMD_CONST_F32(vone, 1.0f);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_3 = xnn_loadu_f32(input + 3 * xnn_simd_size_f32);
    input += 4;

    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    xnn_simd_f32_t vz_prime_0 = xnn_mul_f32(vx_0, vlog2e);
    xnn_simd_f32_t vz_prime_1 = xnn_mul_f32(vx_1, vlog2e);
    xnn_simd_f32_t vz_prime_2 = xnn_mul_f32(vx_2, vlog2e);
    xnn_simd_f32_t vz_prime_3 = xnn_mul_f32(vx_3, vlog2e);
    vz_prime_0 = xnn_min_f32(xnn_max_f32(vz_prime_0, vm127), v128);
    vz_prime_1 = xnn_min_f32(xnn_max_f32(vz_prime_1, vm127), v128);
    vz_prime_2 = xnn_min_f32(xnn_max_f32(vz_prime_2, vm127), v128);
    vz_prime_3 = xnn_min_f32(xnn_max_f32(vz_prime_3, vm127), v128);

    // Decompose x * log2e into `z` (integer part) and `r` (remainder).
    const xnn_simd_f32_t vz_0 = xnn_qd_round_f32(vz_prime_0);
    const xnn_simd_f32_t vz_1 = xnn_qd_round_f32(vz_prime_1);
    const xnn_simd_f32_t vz_2 = xnn_qd_round_f32(vz_prime_2);
    const xnn_simd_f32_t vz_3 = xnn_qd_round_f32(vz_prime_3);
    const xnn_simd_f32_t vr_0 = xnn_sub_f32(vz_prime_0, vz_0);
    const xnn_simd_f32_t vr_1 = xnn_sub_f32(vz_prime_1, vz_1);
    const xnn_simd_f32_t vr_2 = xnn_sub_f32(vz_prime_2, vz_2);
    const xnn_simd_f32_t vr_3 = xnn_sub_f32(vz_prime_3, vz_3);

    // Compute 2^z.
    const xnn_simd_f32_t v2z_0 = xnn_setexp_f32(vz_0);
    const xnn_simd_f32_t v2z_1 = xnn_setexp_f32(vz_1);
    const xnn_simd_f32_t v2z_2 = xnn_setexp_f32(vz_2);
    const xnn_simd_f32_t v2z_3 = xnn_setexp_f32(vz_3);

    // Evaluate the numerator polynomial p(f).
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vr_0, valpha_3, valpha_2);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vr_1, valpha_3, valpha_2);
    xnn_simd_f32_t vp_2 = xnn_fmadd_f32(vr_2, valpha_3, valpha_2);
    xnn_simd_f32_t vp_3 = xnn_fmadd_f32(vr_3, valpha_3, valpha_2);
    vp_0 = xnn_fmadd_f32(vr_0, vp_0, valpha_1);
    vp_1 = xnn_fmadd_f32(vr_1, vp_1, valpha_1);
    vp_2 = xnn_fmadd_f32(vr_2, vp_2, valpha_1);
    vp_3 = xnn_fmadd_f32(vr_3, vp_3, valpha_1);
    vp_0 = xnn_fmadd_f32(vr_0, vp_0, vone);
    vp_1 = xnn_fmadd_f32(vr_1, vp_1, vone);
    vp_2 = xnn_fmadd_f32(vr_2, vp_2, vone);
    vp_3 = xnn_fmadd_f32(vr_3, vp_3, vone);

    // Evaluate the denominator polynomial q(r).
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vr_0, vbeta_2, vbeta_1);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vr_1, vbeta_2, vbeta_1);
    xnn_simd_f32_t vq_2 = xnn_fmadd_f32(vr_2, vbeta_2, vbeta_1);
    xnn_simd_f32_t vq_3 = xnn_fmadd_f32(vr_3, vbeta_2, vbeta_1);
    vq_0 = xnn_fmadd_f32(vr_0, vq_0, vone);
    vq_1 = xnn_fmadd_f32(vr_1, vq_1, vone);
    vq_2 = xnn_fmadd_f32(vr_2, vq_2, vone);
    vq_3 = xnn_fmadd_f32(vr_3, vq_3, vone);

    // Divide the numerator by the denominator, obtaining 2^r.
    const xnn_simd_f32_t v2r_0 =  xnn_div_f32(vp_0, vq_0);
    const xnn_simd_f32_t v2r_1 =  xnn_div_f32(vp_1, vq_1);
    const xnn_simd_f32_t v2r_2 =  xnn_div_f32(vp_2, vq_2);
    const xnn_simd_f32_t v2r_3 =  xnn_div_f32(vp_3, vq_3);

    // Compute 2^z * 2^r.
    const xnn_simd_f32_t vy_0 = xnn_mul_f32(v2z_0, v2r_0);
    const xnn_simd_f32_t vy_1 = xnn_mul_f32(v2z_1, v2r_1);
    const xnn_simd_f32_t vy_2 = xnn_mul_f32(v2z_2, v2r_2);
    const xnn_simd_f32_t vy_3 = xnn_mul_f32(v2z_3, v2r_3);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_2);
    xnn_storeu_f32(output + 3 * xnn_simd_size_f32, vy_3);
    output += 4;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    xnn_simd_f32_t vz_prime = xnn_mul_f32(vx, vlog2e);
    vz_prime = xnn_min_f32(xnn_max_f32(vz_prime, vm127), v128);

    // Decompose x * log2e into `z` (integer part) and `r` (remainder).
    const xnn_simd_f32_t vz = xnn_qd_round_f32(vz_prime);
    const xnn_simd_f32_t vr = xnn_sub_f32(vz_prime, vz);

    // Compute 2^z.
    const xnn_simd_f32_t v2z = xnn_setexp_f32(vz);

    // Evaluate the numerator polynomial p(f).
    xnn_simd_f32_t vp = xnn_fmadd_f32(vr, valpha_3, valpha_2);
    vp = xnn_fmadd_f32(vr, vp, valpha_1);
    vp = xnn_fmadd_f32(vr, vp, vone);

    // Evaluate the denominator polynomial q(r).
    xnn_simd_f32_t vq = xnn_fmadd_f32(vr, vbeta_2, vbeta_1);
    vq = xnn_fmadd_f32(vr, vq, vone);

    // Divide the numerator by the denominator, obtaining 2^r.
    const xnn_simd_f32_t v2r =  xnn_div_f32(vp, vq);

    // Compute 2^z * 2^r.
    const xnn_simd_f32_t vy = xnn_mul_f32(v2z, v2r);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
}

void xnn_f32_vexp_ukernel__scalar_rational_3_2_div_u8(
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

  // The monomial coefficients of the numerator polynomial (`valpha_0` = 1.0).
  XNN_SIMD_CONST_F32(valpha_1, 4.1594290733e-01f);
  XNN_SIMD_CONST_F32(valpha_2, 7.2068706155e-02f);
  XNN_SIMD_CONST_F32(valpha_3, 5.5380910635e-03f);

  // The monomial coefficients of the denominator polynomial (`vbeta_01 = 1.0).
  XNN_SIMD_CONST_F32(vbeta_1, -2.7720427513e-01f);
  XNN_SIMD_CONST_F32(vbeta_2, 2.3986088112e-02f);

  // Some useful constants.
  XNN_SIMD_CONST_F32(vlog2e, 1.44269504089f);
  XNN_SIMD_CONST_F32(v128, 128.0f);
  XNN_SIMD_CONST_F32(vm127, -127.0f);
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

    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    xnn_simd_f32_t vz_prime_0 = xnn_mul_f32(vx_0, vlog2e);
    xnn_simd_f32_t vz_prime_1 = xnn_mul_f32(vx_1, vlog2e);
    xnn_simd_f32_t vz_prime_2 = xnn_mul_f32(vx_2, vlog2e);
    xnn_simd_f32_t vz_prime_3 = xnn_mul_f32(vx_3, vlog2e);
    xnn_simd_f32_t vz_prime_4 = xnn_mul_f32(vx_4, vlog2e);
    xnn_simd_f32_t vz_prime_5 = xnn_mul_f32(vx_5, vlog2e);
    xnn_simd_f32_t vz_prime_6 = xnn_mul_f32(vx_6, vlog2e);
    xnn_simd_f32_t vz_prime_7 = xnn_mul_f32(vx_7, vlog2e);
    vz_prime_0 = xnn_min_f32(xnn_max_f32(vz_prime_0, vm127), v128);
    vz_prime_1 = xnn_min_f32(xnn_max_f32(vz_prime_1, vm127), v128);
    vz_prime_2 = xnn_min_f32(xnn_max_f32(vz_prime_2, vm127), v128);
    vz_prime_3 = xnn_min_f32(xnn_max_f32(vz_prime_3, vm127), v128);
    vz_prime_4 = xnn_min_f32(xnn_max_f32(vz_prime_4, vm127), v128);
    vz_prime_5 = xnn_min_f32(xnn_max_f32(vz_prime_5, vm127), v128);
    vz_prime_6 = xnn_min_f32(xnn_max_f32(vz_prime_6, vm127), v128);
    vz_prime_7 = xnn_min_f32(xnn_max_f32(vz_prime_7, vm127), v128);

    // Decompose x * log2e into `z` (integer part) and `r` (remainder).
    const xnn_simd_f32_t vz_0 = xnn_qd_round_f32(vz_prime_0);
    const xnn_simd_f32_t vz_1 = xnn_qd_round_f32(vz_prime_1);
    const xnn_simd_f32_t vz_2 = xnn_qd_round_f32(vz_prime_2);
    const xnn_simd_f32_t vz_3 = xnn_qd_round_f32(vz_prime_3);
    const xnn_simd_f32_t vz_4 = xnn_qd_round_f32(vz_prime_4);
    const xnn_simd_f32_t vz_5 = xnn_qd_round_f32(vz_prime_5);
    const xnn_simd_f32_t vz_6 = xnn_qd_round_f32(vz_prime_6);
    const xnn_simd_f32_t vz_7 = xnn_qd_round_f32(vz_prime_7);
    const xnn_simd_f32_t vr_0 = xnn_sub_f32(vz_prime_0, vz_0);
    const xnn_simd_f32_t vr_1 = xnn_sub_f32(vz_prime_1, vz_1);
    const xnn_simd_f32_t vr_2 = xnn_sub_f32(vz_prime_2, vz_2);
    const xnn_simd_f32_t vr_3 = xnn_sub_f32(vz_prime_3, vz_3);
    const xnn_simd_f32_t vr_4 = xnn_sub_f32(vz_prime_4, vz_4);
    const xnn_simd_f32_t vr_5 = xnn_sub_f32(vz_prime_5, vz_5);
    const xnn_simd_f32_t vr_6 = xnn_sub_f32(vz_prime_6, vz_6);
    const xnn_simd_f32_t vr_7 = xnn_sub_f32(vz_prime_7, vz_7);

    // Compute 2^z.
    const xnn_simd_f32_t v2z_0 = xnn_setexp_f32(vz_0);
    const xnn_simd_f32_t v2z_1 = xnn_setexp_f32(vz_1);
    const xnn_simd_f32_t v2z_2 = xnn_setexp_f32(vz_2);
    const xnn_simd_f32_t v2z_3 = xnn_setexp_f32(vz_3);
    const xnn_simd_f32_t v2z_4 = xnn_setexp_f32(vz_4);
    const xnn_simd_f32_t v2z_5 = xnn_setexp_f32(vz_5);
    const xnn_simd_f32_t v2z_6 = xnn_setexp_f32(vz_6);
    const xnn_simd_f32_t v2z_7 = xnn_setexp_f32(vz_7);

    // Evaluate the numerator polynomial p(f).
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vr_0, valpha_3, valpha_2);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vr_1, valpha_3, valpha_2);
    xnn_simd_f32_t vp_2 = xnn_fmadd_f32(vr_2, valpha_3, valpha_2);
    xnn_simd_f32_t vp_3 = xnn_fmadd_f32(vr_3, valpha_3, valpha_2);
    xnn_simd_f32_t vp_4 = xnn_fmadd_f32(vr_4, valpha_3, valpha_2);
    xnn_simd_f32_t vp_5 = xnn_fmadd_f32(vr_5, valpha_3, valpha_2);
    xnn_simd_f32_t vp_6 = xnn_fmadd_f32(vr_6, valpha_3, valpha_2);
    xnn_simd_f32_t vp_7 = xnn_fmadd_f32(vr_7, valpha_3, valpha_2);
    vp_0 = xnn_fmadd_f32(vr_0, vp_0, valpha_1);
    vp_1 = xnn_fmadd_f32(vr_1, vp_1, valpha_1);
    vp_2 = xnn_fmadd_f32(vr_2, vp_2, valpha_1);
    vp_3 = xnn_fmadd_f32(vr_3, vp_3, valpha_1);
    vp_4 = xnn_fmadd_f32(vr_4, vp_4, valpha_1);
    vp_5 = xnn_fmadd_f32(vr_5, vp_5, valpha_1);
    vp_6 = xnn_fmadd_f32(vr_6, vp_6, valpha_1);
    vp_7 = xnn_fmadd_f32(vr_7, vp_7, valpha_1);
    vp_0 = xnn_fmadd_f32(vr_0, vp_0, vone);
    vp_1 = xnn_fmadd_f32(vr_1, vp_1, vone);
    vp_2 = xnn_fmadd_f32(vr_2, vp_2, vone);
    vp_3 = xnn_fmadd_f32(vr_3, vp_3, vone);
    vp_4 = xnn_fmadd_f32(vr_4, vp_4, vone);
    vp_5 = xnn_fmadd_f32(vr_5, vp_5, vone);
    vp_6 = xnn_fmadd_f32(vr_6, vp_6, vone);
    vp_7 = xnn_fmadd_f32(vr_7, vp_7, vone);

    // Evaluate the denominator polynomial q(r).
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vr_0, vbeta_2, vbeta_1);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vr_1, vbeta_2, vbeta_1);
    xnn_simd_f32_t vq_2 = xnn_fmadd_f32(vr_2, vbeta_2, vbeta_1);
    xnn_simd_f32_t vq_3 = xnn_fmadd_f32(vr_3, vbeta_2, vbeta_1);
    xnn_simd_f32_t vq_4 = xnn_fmadd_f32(vr_4, vbeta_2, vbeta_1);
    xnn_simd_f32_t vq_5 = xnn_fmadd_f32(vr_5, vbeta_2, vbeta_1);
    xnn_simd_f32_t vq_6 = xnn_fmadd_f32(vr_6, vbeta_2, vbeta_1);
    xnn_simd_f32_t vq_7 = xnn_fmadd_f32(vr_7, vbeta_2, vbeta_1);
    vq_0 = xnn_fmadd_f32(vr_0, vq_0, vone);
    vq_1 = xnn_fmadd_f32(vr_1, vq_1, vone);
    vq_2 = xnn_fmadd_f32(vr_2, vq_2, vone);
    vq_3 = xnn_fmadd_f32(vr_3, vq_3, vone);
    vq_4 = xnn_fmadd_f32(vr_4, vq_4, vone);
    vq_5 = xnn_fmadd_f32(vr_5, vq_5, vone);
    vq_6 = xnn_fmadd_f32(vr_6, vq_6, vone);
    vq_7 = xnn_fmadd_f32(vr_7, vq_7, vone);

    // Divide the numerator by the denominator, obtaining 2^r.
    const xnn_simd_f32_t v2r_0 =  xnn_div_f32(vp_0, vq_0);
    const xnn_simd_f32_t v2r_1 =  xnn_div_f32(vp_1, vq_1);
    const xnn_simd_f32_t v2r_2 =  xnn_div_f32(vp_2, vq_2);
    const xnn_simd_f32_t v2r_3 =  xnn_div_f32(vp_3, vq_3);
    const xnn_simd_f32_t v2r_4 =  xnn_div_f32(vp_4, vq_4);
    const xnn_simd_f32_t v2r_5 =  xnn_div_f32(vp_5, vq_5);
    const xnn_simd_f32_t v2r_6 =  xnn_div_f32(vp_6, vq_6);
    const xnn_simd_f32_t v2r_7 =  xnn_div_f32(vp_7, vq_7);

    // Compute 2^z * 2^r.
    const xnn_simd_f32_t vy_0 = xnn_mul_f32(v2z_0, v2r_0);
    const xnn_simd_f32_t vy_1 = xnn_mul_f32(v2z_1, v2r_1);
    const xnn_simd_f32_t vy_2 = xnn_mul_f32(v2z_2, v2r_2);
    const xnn_simd_f32_t vy_3 = xnn_mul_f32(v2z_3, v2r_3);
    const xnn_simd_f32_t vy_4 = xnn_mul_f32(v2z_4, v2r_4);
    const xnn_simd_f32_t vy_5 = xnn_mul_f32(v2z_5, v2r_5);
    const xnn_simd_f32_t vy_6 = xnn_mul_f32(v2z_6, v2r_6);
    const xnn_simd_f32_t vy_7 = xnn_mul_f32(v2z_7, v2r_7);

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

    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    xnn_simd_f32_t vz_prime = xnn_mul_f32(vx, vlog2e);
    vz_prime = xnn_min_f32(xnn_max_f32(vz_prime, vm127), v128);

    // Decompose x * log2e into `z` (integer part) and `r` (remainder).
    const xnn_simd_f32_t vz = xnn_qd_round_f32(vz_prime);
    const xnn_simd_f32_t vr = xnn_sub_f32(vz_prime, vz);

    // Compute 2^z.
    const xnn_simd_f32_t v2z = xnn_setexp_f32(vz);

    // Evaluate the numerator polynomial p(f).
    xnn_simd_f32_t vp = xnn_fmadd_f32(vr, valpha_3, valpha_2);
    vp = xnn_fmadd_f32(vr, vp, valpha_1);
    vp = xnn_fmadd_f32(vr, vp, vone);

    // Evaluate the denominator polynomial q(r).
    xnn_simd_f32_t vq = xnn_fmadd_f32(vr, vbeta_2, vbeta_1);
    vq = xnn_fmadd_f32(vr, vq, vone);

    // Divide the numerator by the denominator, obtaining 2^r.
    const xnn_simd_f32_t v2r =  xnn_div_f32(vp, vq);

    // Compute 2^z * 2^r.
    const xnn_simd_f32_t vy = xnn_mul_f32(v2z, v2r);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
}

