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

#include "src/xnnpack/simd/f32-avx512f.h"

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

void xnn_f32_vexp_ukernel__avx512f_rational_3_2_div_u16(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 16);

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
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

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

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vexp_ukernel__avx512f_rational_3_2_div_u32(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 16);

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

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 32;

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
    output += 32;
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
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

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

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vexp_ukernel__avx512f_rational_3_2_div_u48(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 16);

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

  for (; batch >= 48 * sizeof(float); batch -= 48 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    input += 48;

    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    xnn_simd_f32_t vz_prime_0 = xnn_mul_f32(vx_0, vlog2e);
    xnn_simd_f32_t vz_prime_1 = xnn_mul_f32(vx_1, vlog2e);
    xnn_simd_f32_t vz_prime_2 = xnn_mul_f32(vx_2, vlog2e);
    vz_prime_0 = xnn_min_f32(xnn_max_f32(vz_prime_0, vm127), v128);
    vz_prime_1 = xnn_min_f32(xnn_max_f32(vz_prime_1, vm127), v128);
    vz_prime_2 = xnn_min_f32(xnn_max_f32(vz_prime_2, vm127), v128);

    // Decompose x * log2e into `z` (integer part) and `r` (remainder).
    const xnn_simd_f32_t vz_0 = xnn_qd_round_f32(vz_prime_0);
    const xnn_simd_f32_t vz_1 = xnn_qd_round_f32(vz_prime_1);
    const xnn_simd_f32_t vz_2 = xnn_qd_round_f32(vz_prime_2);
    const xnn_simd_f32_t vr_0 = xnn_sub_f32(vz_prime_0, vz_0);
    const xnn_simd_f32_t vr_1 = xnn_sub_f32(vz_prime_1, vz_1);
    const xnn_simd_f32_t vr_2 = xnn_sub_f32(vz_prime_2, vz_2);

    // Compute 2^z.
    const xnn_simd_f32_t v2z_0 = xnn_setexp_f32(vz_0);
    const xnn_simd_f32_t v2z_1 = xnn_setexp_f32(vz_1);
    const xnn_simd_f32_t v2z_2 = xnn_setexp_f32(vz_2);

    // Evaluate the numerator polynomial p(f).
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vr_0, valpha_3, valpha_2);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vr_1, valpha_3, valpha_2);
    xnn_simd_f32_t vp_2 = xnn_fmadd_f32(vr_2, valpha_3, valpha_2);
    vp_0 = xnn_fmadd_f32(vr_0, vp_0, valpha_1);
    vp_1 = xnn_fmadd_f32(vr_1, vp_1, valpha_1);
    vp_2 = xnn_fmadd_f32(vr_2, vp_2, valpha_1);
    vp_0 = xnn_fmadd_f32(vr_0, vp_0, vone);
    vp_1 = xnn_fmadd_f32(vr_1, vp_1, vone);
    vp_2 = xnn_fmadd_f32(vr_2, vp_2, vone);

    // Evaluate the denominator polynomial q(r).
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vr_0, vbeta_2, vbeta_1);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vr_1, vbeta_2, vbeta_1);
    xnn_simd_f32_t vq_2 = xnn_fmadd_f32(vr_2, vbeta_2, vbeta_1);
    vq_0 = xnn_fmadd_f32(vr_0, vq_0, vone);
    vq_1 = xnn_fmadd_f32(vr_1, vq_1, vone);
    vq_2 = xnn_fmadd_f32(vr_2, vq_2, vone);

    // Divide the numerator by the denominator, obtaining 2^r.
    const xnn_simd_f32_t v2r_0 =  xnn_div_f32(vp_0, vq_0);
    const xnn_simd_f32_t v2r_1 =  xnn_div_f32(vp_1, vq_1);
    const xnn_simd_f32_t v2r_2 =  xnn_div_f32(vp_2, vq_2);

    // Compute 2^z * 2^r.
    const xnn_simd_f32_t vy_0 = xnn_mul_f32(v2z_0, v2r_0);
    const xnn_simd_f32_t vy_1 = xnn_mul_f32(v2z_1, v2r_1);
    const xnn_simd_f32_t vy_2 = xnn_mul_f32(v2z_2, v2r_2);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_2);
    output += 48;
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
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

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

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vexp_ukernel__avx512f_rational_3_2_div_u64(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 16);

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

  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_3 = xnn_loadu_f32(input + 3 * xnn_simd_size_f32);
    input += 64;

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
    output += 64;
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
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

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

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

