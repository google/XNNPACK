// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vexp/poly-3.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/simd/f16-neonfp16arith.h"

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


static XNN_INLINE xnn_simd_f16_t xnn_setexp_f16(xnn_simd_f16_t vx) {
  // If `x` is an xnn_float16ing point value in the range [-15, 16], then
  // `(x + magic) << 10` will generate the floating point value corresponding
  // to `2^round(x)` (2^-15 and 2^16 will flush to zero and infinity,
  // respectively).
  XNN_SIMD_CONST_F16_FROM_FLOAT(vmagic, 1039.0f);  // 2^10 + 15.
  return xnn_sll_f16(xnn_add_f16(vx, vmagic), 10);
}

// Quick-and-dirty round to nearest, only works for xnn_float16s in the range
// `[2^-9, 2^9)`.
static XNN_INLINE xnn_simd_f16_t xnn_qd_round_f16(xnn_simd_f16_t vx) {
  // If `x` is an xnn_float16ing point value in the range `[2^-9, 2^9)`, then
  // `(x + magic) - magic`` will generate the floating point value corresponding
  // to `round(x)`.
  XNN_SIMD_CONST_F16_FROM_FLOAT(vmagic, 1536.0f);  // 2^10 + 2^9.
  return xnn_sub_f16(xnn_add_f16(vmagic, vx), vmagic);
}

void xnn_f16_vexp_ukernel__neonfp16arith_poly_3_u8(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 8);

  // The monomial coefficients of the interpolation polynomial (`valpha_0` = 1).
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_1, 0.6933594f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_2, 0.24255371f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_3, 0.05517578f);
      
  // Some useful constants.
  XNN_SIMD_CONST_F16_FROM_FLOAT(vlog2e, 1.4423828f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(v16, 16.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vm15, -15.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vone, 1.0f);

  for (; batch >= xnn_simd_bytes_f16; batch -= xnn_simd_bytes_f16) {
    xnn_simd_f16_t vx = xnn_loadu_f16(input);
    input += xnn_simd_size_f16;
    
    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    xnn_simd_f16_t vz_prime = xnn_mul_f16(vx, vlog2e);
    vz_prime = xnn_min_f16(xnn_max_f16(vz_prime, vm15), v16);

    // Decompose x * log2e into `z` (integer part) and `r` (remainder).
    const xnn_simd_f16_t vz = xnn_qd_round_f16(vz_prime);
    const xnn_simd_f16_t vr = xnn_sub_f16(vz_prime, vz);
    
    // Compute 2^z.
    const xnn_simd_f16_t v2z = xnn_setexp_f16(vz);

    // Evaluate the interpolation polynomial for `2^r`.
    xnn_simd_f16_t v2r = xnn_fmadd_f16(vr, valpha_3, valpha_2);
    v2r = xnn_fmadd_f16(vr, v2r, valpha_1);
    v2r = xnn_fmadd_f16(vr, v2r, vone);

    // Compute 2^z * 2^r.
    const xnn_simd_f16_t vy = xnn_mul_f16(v2z, v2r);

    xnn_storeu_f16(output, vy);
    output += xnn_simd_size_f16;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f16_t vx = xnn_load_tail_f16(input, batch >> XNN_LOG2_SIZEOF_HALF);

    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    xnn_simd_f16_t vz_prime = xnn_mul_f16(vx, vlog2e);
    vz_prime = xnn_min_f16(xnn_max_f16(vz_prime, vm15), v16);

    // Decompose x * log2e into `z` (integer part) and `r` (remainder).
    const xnn_simd_f16_t vz = xnn_qd_round_f16(vz_prime);
    const xnn_simd_f16_t vr = xnn_sub_f16(vz_prime, vz);
    
    // Compute 2^z.
    const xnn_simd_f16_t v2z = xnn_setexp_f16(vz);

    // Evaluate the interpolation polynomial for `2^r`.
    xnn_simd_f16_t v2r = xnn_fmadd_f16(vr, valpha_3, valpha_2);
    v2r = xnn_fmadd_f16(vr, v2r, valpha_1);
    v2r = xnn_fmadd_f16(vr, v2r, vone);

    // Compute 2^z * 2^r.
    const xnn_simd_f16_t vy = xnn_mul_f16(v2z, v2r);

    xnn_store_tail_f16(output, vy, batch >> XNN_LOG2_SIZEOF_HALF);
  }
}

void xnn_f16_vexp_ukernel__neonfp16arith_poly_3_u16(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 8);

  // The monomial coefficients of the interpolation polynomial (`valpha_0` = 1).
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_1, 0.6933594f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_2, 0.24255371f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_3, 0.05517578f);
      
  // Some useful constants.
  XNN_SIMD_CONST_F16_FROM_FLOAT(vlog2e, 1.4423828f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(v16, 16.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vm15, -15.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vone, 1.0f);

  for (; batch >= 16 * sizeof(xnn_float16); batch -= 16 * sizeof(xnn_float16)) {
    xnn_simd_f16_t vx_0 = xnn_loadu_f16(input + 0 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_1 = xnn_loadu_f16(input + 1 * xnn_simd_size_f16);
    input += 16;
    
    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    xnn_simd_f16_t vz_prime_0 = xnn_mul_f16(vx_0, vlog2e);
    xnn_simd_f16_t vz_prime_1 = xnn_mul_f16(vx_1, vlog2e);
    vz_prime_0 = xnn_min_f16(xnn_max_f16(vz_prime_0, vm15), v16);
    vz_prime_1 = xnn_min_f16(xnn_max_f16(vz_prime_1, vm15), v16);

    // Decompose x * log2e into `z` (integer part) and `r` (remainder).
    const xnn_simd_f16_t vz_0 = xnn_qd_round_f16(vz_prime_0);
    const xnn_simd_f16_t vz_1 = xnn_qd_round_f16(vz_prime_1);
    const xnn_simd_f16_t vr_0 = xnn_sub_f16(vz_prime_0, vz_0);
    const xnn_simd_f16_t vr_1 = xnn_sub_f16(vz_prime_1, vz_1);
    
    // Compute 2^z.
    const xnn_simd_f16_t v2z_0 = xnn_setexp_f16(vz_0);
    const xnn_simd_f16_t v2z_1 = xnn_setexp_f16(vz_1);

    // Evaluate the interpolation polynomial for `2^r`.
    xnn_simd_f16_t v2r_0 = xnn_fmadd_f16(vr_0, valpha_3, valpha_2);
    xnn_simd_f16_t v2r_1 = xnn_fmadd_f16(vr_1, valpha_3, valpha_2);
    v2r_0 = xnn_fmadd_f16(vr_0, v2r_0, valpha_1);
    v2r_1 = xnn_fmadd_f16(vr_1, v2r_1, valpha_1);
    v2r_0 = xnn_fmadd_f16(vr_0, v2r_0, vone);
    v2r_1 = xnn_fmadd_f16(vr_1, v2r_1, vone);

    // Compute 2^z * 2^r.
    const xnn_simd_f16_t vy_0 = xnn_mul_f16(v2z_0, v2r_0);
    const xnn_simd_f16_t vy_1 = xnn_mul_f16(v2z_1, v2r_1);

    xnn_storeu_f16(output + 0 * xnn_simd_size_f16, vy_0);
    xnn_storeu_f16(output + 1 * xnn_simd_size_f16, vy_1);
    output += 16;
  }
  for (; batch >= xnn_simd_bytes_f16; batch -= xnn_simd_bytes_f16) {
    xnn_simd_f16_t vx = xnn_loadu_f16(input);
    input += xnn_simd_size_f16;
    
    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    xnn_simd_f16_t vz_prime = xnn_mul_f16(vx, vlog2e);
    vz_prime = xnn_min_f16(xnn_max_f16(vz_prime, vm15), v16);

    // Decompose x * log2e into `z` (integer part) and `r` (remainder).
    const xnn_simd_f16_t vz = xnn_qd_round_f16(vz_prime);
    const xnn_simd_f16_t vr = xnn_sub_f16(vz_prime, vz);
    
    // Compute 2^z.
    const xnn_simd_f16_t v2z = xnn_setexp_f16(vz);

    // Evaluate the interpolation polynomial for `2^r`.
    xnn_simd_f16_t v2r = xnn_fmadd_f16(vr, valpha_3, valpha_2);
    v2r = xnn_fmadd_f16(vr, v2r, valpha_1);
    v2r = xnn_fmadd_f16(vr, v2r, vone);

    // Compute 2^z * 2^r.
    const xnn_simd_f16_t vy = xnn_mul_f16(v2z, v2r);

    xnn_storeu_f16(output, vy);
    output += xnn_simd_size_f16;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f16_t vx = xnn_load_tail_f16(input, batch >> XNN_LOG2_SIZEOF_HALF);

    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    xnn_simd_f16_t vz_prime = xnn_mul_f16(vx, vlog2e);
    vz_prime = xnn_min_f16(xnn_max_f16(vz_prime, vm15), v16);

    // Decompose x * log2e into `z` (integer part) and `r` (remainder).
    const xnn_simd_f16_t vz = xnn_qd_round_f16(vz_prime);
    const xnn_simd_f16_t vr = xnn_sub_f16(vz_prime, vz);
    
    // Compute 2^z.
    const xnn_simd_f16_t v2z = xnn_setexp_f16(vz);

    // Evaluate the interpolation polynomial for `2^r`.
    xnn_simd_f16_t v2r = xnn_fmadd_f16(vr, valpha_3, valpha_2);
    v2r = xnn_fmadd_f16(vr, v2r, valpha_1);
    v2r = xnn_fmadd_f16(vr, v2r, vone);

    // Compute 2^z * 2^r.
    const xnn_simd_f16_t vy = xnn_mul_f16(v2z, v2r);

    xnn_store_tail_f16(output, vy, batch >> XNN_LOG2_SIZEOF_HALF);
  }
}

void xnn_f16_vexp_ukernel__neonfp16arith_poly_3_u32(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 8);

  // The monomial coefficients of the interpolation polynomial (`valpha_0` = 1).
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_1, 0.6933594f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_2, 0.24255371f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_3, 0.05517578f);
      
  // Some useful constants.
  XNN_SIMD_CONST_F16_FROM_FLOAT(vlog2e, 1.4423828f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(v16, 16.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vm15, -15.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vone, 1.0f);

  for (; batch >= 32 * sizeof(xnn_float16); batch -= 32 * sizeof(xnn_float16)) {
    xnn_simd_f16_t vx_0 = xnn_loadu_f16(input + 0 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_1 = xnn_loadu_f16(input + 1 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_2 = xnn_loadu_f16(input + 2 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_3 = xnn_loadu_f16(input + 3 * xnn_simd_size_f16);
    input += 32;
    
    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    xnn_simd_f16_t vz_prime_0 = xnn_mul_f16(vx_0, vlog2e);
    xnn_simd_f16_t vz_prime_1 = xnn_mul_f16(vx_1, vlog2e);
    xnn_simd_f16_t vz_prime_2 = xnn_mul_f16(vx_2, vlog2e);
    xnn_simd_f16_t vz_prime_3 = xnn_mul_f16(vx_3, vlog2e);
    vz_prime_0 = xnn_min_f16(xnn_max_f16(vz_prime_0, vm15), v16);
    vz_prime_1 = xnn_min_f16(xnn_max_f16(vz_prime_1, vm15), v16);
    vz_prime_2 = xnn_min_f16(xnn_max_f16(vz_prime_2, vm15), v16);
    vz_prime_3 = xnn_min_f16(xnn_max_f16(vz_prime_3, vm15), v16);

    // Decompose x * log2e into `z` (integer part) and `r` (remainder).
    const xnn_simd_f16_t vz_0 = xnn_qd_round_f16(vz_prime_0);
    const xnn_simd_f16_t vz_1 = xnn_qd_round_f16(vz_prime_1);
    const xnn_simd_f16_t vz_2 = xnn_qd_round_f16(vz_prime_2);
    const xnn_simd_f16_t vz_3 = xnn_qd_round_f16(vz_prime_3);
    const xnn_simd_f16_t vr_0 = xnn_sub_f16(vz_prime_0, vz_0);
    const xnn_simd_f16_t vr_1 = xnn_sub_f16(vz_prime_1, vz_1);
    const xnn_simd_f16_t vr_2 = xnn_sub_f16(vz_prime_2, vz_2);
    const xnn_simd_f16_t vr_3 = xnn_sub_f16(vz_prime_3, vz_3);
    
    // Compute 2^z.
    const xnn_simd_f16_t v2z_0 = xnn_setexp_f16(vz_0);
    const xnn_simd_f16_t v2z_1 = xnn_setexp_f16(vz_1);
    const xnn_simd_f16_t v2z_2 = xnn_setexp_f16(vz_2);
    const xnn_simd_f16_t v2z_3 = xnn_setexp_f16(vz_3);

    // Evaluate the interpolation polynomial for `2^r`.
    xnn_simd_f16_t v2r_0 = xnn_fmadd_f16(vr_0, valpha_3, valpha_2);
    xnn_simd_f16_t v2r_1 = xnn_fmadd_f16(vr_1, valpha_3, valpha_2);
    xnn_simd_f16_t v2r_2 = xnn_fmadd_f16(vr_2, valpha_3, valpha_2);
    xnn_simd_f16_t v2r_3 = xnn_fmadd_f16(vr_3, valpha_3, valpha_2);
    v2r_0 = xnn_fmadd_f16(vr_0, v2r_0, valpha_1);
    v2r_1 = xnn_fmadd_f16(vr_1, v2r_1, valpha_1);
    v2r_2 = xnn_fmadd_f16(vr_2, v2r_2, valpha_1);
    v2r_3 = xnn_fmadd_f16(vr_3, v2r_3, valpha_1);
    v2r_0 = xnn_fmadd_f16(vr_0, v2r_0, vone);
    v2r_1 = xnn_fmadd_f16(vr_1, v2r_1, vone);
    v2r_2 = xnn_fmadd_f16(vr_2, v2r_2, vone);
    v2r_3 = xnn_fmadd_f16(vr_3, v2r_3, vone);

    // Compute 2^z * 2^r.
    const xnn_simd_f16_t vy_0 = xnn_mul_f16(v2z_0, v2r_0);
    const xnn_simd_f16_t vy_1 = xnn_mul_f16(v2z_1, v2r_1);
    const xnn_simd_f16_t vy_2 = xnn_mul_f16(v2z_2, v2r_2);
    const xnn_simd_f16_t vy_3 = xnn_mul_f16(v2z_3, v2r_3);

    xnn_storeu_f16(output + 0 * xnn_simd_size_f16, vy_0);
    xnn_storeu_f16(output + 1 * xnn_simd_size_f16, vy_1);
    xnn_storeu_f16(output + 2 * xnn_simd_size_f16, vy_2);
    xnn_storeu_f16(output + 3 * xnn_simd_size_f16, vy_3);
    output += 32;
  }
  for (; batch >= xnn_simd_bytes_f16; batch -= xnn_simd_bytes_f16) {
    xnn_simd_f16_t vx = xnn_loadu_f16(input);
    input += xnn_simd_size_f16;
    
    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    xnn_simd_f16_t vz_prime = xnn_mul_f16(vx, vlog2e);
    vz_prime = xnn_min_f16(xnn_max_f16(vz_prime, vm15), v16);

    // Decompose x * log2e into `z` (integer part) and `r` (remainder).
    const xnn_simd_f16_t vz = xnn_qd_round_f16(vz_prime);
    const xnn_simd_f16_t vr = xnn_sub_f16(vz_prime, vz);
    
    // Compute 2^z.
    const xnn_simd_f16_t v2z = xnn_setexp_f16(vz);

    // Evaluate the interpolation polynomial for `2^r`.
    xnn_simd_f16_t v2r = xnn_fmadd_f16(vr, valpha_3, valpha_2);
    v2r = xnn_fmadd_f16(vr, v2r, valpha_1);
    v2r = xnn_fmadd_f16(vr, v2r, vone);

    // Compute 2^z * 2^r.
    const xnn_simd_f16_t vy = xnn_mul_f16(v2z, v2r);

    xnn_storeu_f16(output, vy);
    output += xnn_simd_size_f16;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f16_t vx = xnn_load_tail_f16(input, batch >> XNN_LOG2_SIZEOF_HALF);

    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    xnn_simd_f16_t vz_prime = xnn_mul_f16(vx, vlog2e);
    vz_prime = xnn_min_f16(xnn_max_f16(vz_prime, vm15), v16);

    // Decompose x * log2e into `z` (integer part) and `r` (remainder).
    const xnn_simd_f16_t vz = xnn_qd_round_f16(vz_prime);
    const xnn_simd_f16_t vr = xnn_sub_f16(vz_prime, vz);
    
    // Compute 2^z.
    const xnn_simd_f16_t v2z = xnn_setexp_f16(vz);

    // Evaluate the interpolation polynomial for `2^r`.
    xnn_simd_f16_t v2r = xnn_fmadd_f16(vr, valpha_3, valpha_2);
    v2r = xnn_fmadd_f16(vr, v2r, valpha_1);
    v2r = xnn_fmadd_f16(vr, v2r, vone);

    // Compute 2^z * 2^r.
    const xnn_simd_f16_t vy = xnn_mul_f16(v2z, v2r);

    xnn_store_tail_f16(output, vy, batch >> XNN_LOG2_SIZEOF_HALF);
  }
}

