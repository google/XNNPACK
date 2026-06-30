// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vexp/f16-f32-poly-3.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

#include "src/xnnpack/simd/f32-scalar.h"

#include "src/xnnpack/simd/f16-scalar.h"


// Helper functions for f16 <-> f32 conversion using xnn_simd_f32_t.
static XNN_INLINE xnn_simd_f32_t xnn_loadu_f16_f32(const xnn_float16* ptr) {
  return xnn_cvt_f32_f16(xnn_loadu_f16(ptr));
}

static XNN_INLINE void xnn_store_f32_f16(xnn_float16* ptr, xnn_simd_f32_t v) {
  xnn_store_tail_f16(ptr, xnn_cvt_f16_f32(v), xnn_simd_size_f32);
}

static XNN_INLINE xnn_simd_f32_t xnn_load_f16_f32(const xnn_float16* ptr, size_t elements) {
  return xnn_cvt_f32_f16(xnn_load_tail_f16(ptr, elements));
}

static XNN_INLINE void xnn_store_f32_f16_tail(xnn_float16* ptr, xnn_simd_f32_t v, size_t elements) {
  xnn_store_tail_f16(ptr, xnn_cvt_f16_f32(v), elements);
}


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

void xnn_f16_f32acc_vexp_ukernel__scalar_poly_3_u1(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  // The monomial coefficients of the interpolation polynomial (`valpha_0` = 1).
  XNN_SIMD_CONST_F32(valpha_1, 0.6933594f);
  XNN_SIMD_CONST_F32(valpha_2, 0.24255371f);
  XNN_SIMD_CONST_F32(valpha_3, 0.05517578f);

  // Some useful constants.
  XNN_SIMD_CONST_F32(vlog2e, 1.4423828f);
  XNN_SIMD_CONST_F32(v16, 16.0f);
  XNN_SIMD_CONST_F32(vm15, -15.0f);
  XNN_SIMD_CONST_F32(vone, 1.0f);

  for (; batch >= 1 * sizeof(xnn_float16); batch -= 1 * sizeof(xnn_float16)) {
    xnn_simd_f32_t vx = xnn_loadu_f16_f32(input);
    input += 1;

    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    xnn_simd_f32_t vz_prime = xnn_mul_f32(vx, vlog2e);
    vz_prime = xnn_min_f32(xnn_max_f32(vz_prime, vm15), v16);

    // Decompose x * log2e into `z` (integer part) and `r` (remainder).
    const xnn_simd_f32_t vz = xnn_qd_round_f32(vz_prime);
    const xnn_simd_f32_t vr = xnn_sub_f32(vz_prime, vz);

    // Compute 2^z.
    const xnn_simd_f32_t v2z = xnn_setexp_f32(vz);

    // Evaluate the interpolation polynomial for `2^r`.
    xnn_simd_f32_t v2r = xnn_fmadd_f32(vr, valpha_3, valpha_2);
    v2r = xnn_fmadd_f32(vr, v2r, valpha_1);
    v2r = xnn_fmadd_f32(vr, v2r, vone);

    // Compute 2^z * 2^r.
    const xnn_simd_f32_t vy = xnn_mul_f32(v2z, v2r);

    xnn_store_f32_f16(output, vy);
    output += 1;
  }
}
void xnn_f16_f32acc_vexp_ukernel__scalar_poly_3_u2(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  // The monomial coefficients of the interpolation polynomial (`valpha_0` = 1).
  XNN_SIMD_CONST_F32(valpha_1, 0.6933594f);
  XNN_SIMD_CONST_F32(valpha_2, 0.24255371f);
  XNN_SIMD_CONST_F32(valpha_3, 0.05517578f);

  // Some useful constants.
  XNN_SIMD_CONST_F32(vlog2e, 1.4423828f);
  XNN_SIMD_CONST_F32(v16, 16.0f);
  XNN_SIMD_CONST_F32(vm15, -15.0f);
  XNN_SIMD_CONST_F32(vone, 1.0f);

  for (; batch >= 2 * sizeof(xnn_float16); batch -= 2 * sizeof(xnn_float16)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f16_f32(input + 0 * 1);
    xnn_simd_f32_t vx_1 = xnn_loadu_f16_f32(input + 1 * 1);
    input += 2;

    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    xnn_simd_f32_t vz_prime_0 = xnn_mul_f32(vx_0, vlog2e);
    xnn_simd_f32_t vz_prime_1 = xnn_mul_f32(vx_1, vlog2e);
    vz_prime_0 = xnn_min_f32(xnn_max_f32(vz_prime_0, vm15), v16);
    vz_prime_1 = xnn_min_f32(xnn_max_f32(vz_prime_1, vm15), v16);

    // Decompose x * log2e into `z` (integer part) and `r` (remainder).
    const xnn_simd_f32_t vz_0 = xnn_qd_round_f32(vz_prime_0);
    const xnn_simd_f32_t vz_1 = xnn_qd_round_f32(vz_prime_1);
    const xnn_simd_f32_t vr_0 = xnn_sub_f32(vz_prime_0, vz_0);
    const xnn_simd_f32_t vr_1 = xnn_sub_f32(vz_prime_1, vz_1);

    // Compute 2^z.
    const xnn_simd_f32_t v2z_0 = xnn_setexp_f32(vz_0);
    const xnn_simd_f32_t v2z_1 = xnn_setexp_f32(vz_1);

    // Evaluate the interpolation polynomial for `2^r`.
    xnn_simd_f32_t v2r_0 = xnn_fmadd_f32(vr_0, valpha_3, valpha_2);
    xnn_simd_f32_t v2r_1 = xnn_fmadd_f32(vr_1, valpha_3, valpha_2);
    v2r_0 = xnn_fmadd_f32(vr_0, v2r_0, valpha_1);
    v2r_1 = xnn_fmadd_f32(vr_1, v2r_1, valpha_1);
    v2r_0 = xnn_fmadd_f32(vr_0, v2r_0, vone);
    v2r_1 = xnn_fmadd_f32(vr_1, v2r_1, vone);

    // Compute 2^z * 2^r.
    const xnn_simd_f32_t vy_0 = xnn_mul_f32(v2z_0, v2r_0);
    const xnn_simd_f32_t vy_1 = xnn_mul_f32(v2z_1, v2r_1);

    xnn_store_f32_f16(output + 0 * 1, vy_0);
    xnn_store_f32_f16(output + 1 * 1, vy_1);
    output += 2;
  }
  for (; batch >= 1 * sizeof(xnn_float16); batch -= 1 * sizeof(xnn_float16)) {
    xnn_simd_f32_t vx = xnn_loadu_f16_f32(input);
    input += 1;

    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    xnn_simd_f32_t vz_prime = xnn_mul_f32(vx, vlog2e);
    vz_prime = xnn_min_f32(xnn_max_f32(vz_prime, vm15), v16);

    // Decompose x * log2e into `z` (integer part) and `r` (remainder).
    const xnn_simd_f32_t vz = xnn_qd_round_f32(vz_prime);
    const xnn_simd_f32_t vr = xnn_sub_f32(vz_prime, vz);

    // Compute 2^z.
    const xnn_simd_f32_t v2z = xnn_setexp_f32(vz);

    // Evaluate the interpolation polynomial for `2^r`.
    xnn_simd_f32_t v2r = xnn_fmadd_f32(vr, valpha_3, valpha_2);
    v2r = xnn_fmadd_f32(vr, v2r, valpha_1);
    v2r = xnn_fmadd_f32(vr, v2r, vone);

    // Compute 2^z * 2^r.
    const xnn_simd_f32_t vy = xnn_mul_f32(v2z, v2r);

    xnn_store_f32_f16(output, vy);
    output += 1;
  }
}
void xnn_f16_f32acc_vexp_ukernel__scalar_poly_3_u4(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  // The monomial coefficients of the interpolation polynomial (`valpha_0` = 1).
  XNN_SIMD_CONST_F32(valpha_1, 0.6933594f);
  XNN_SIMD_CONST_F32(valpha_2, 0.24255371f);
  XNN_SIMD_CONST_F32(valpha_3, 0.05517578f);

  // Some useful constants.
  XNN_SIMD_CONST_F32(vlog2e, 1.4423828f);
  XNN_SIMD_CONST_F32(v16, 16.0f);
  XNN_SIMD_CONST_F32(vm15, -15.0f);
  XNN_SIMD_CONST_F32(vone, 1.0f);

  for (; batch >= 4 * sizeof(xnn_float16); batch -= 4 * sizeof(xnn_float16)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f16_f32(input + 0 * 1);
    xnn_simd_f32_t vx_1 = xnn_loadu_f16_f32(input + 1 * 1);
    xnn_simd_f32_t vx_2 = xnn_loadu_f16_f32(input + 2 * 1);
    xnn_simd_f32_t vx_3 = xnn_loadu_f16_f32(input + 3 * 1);
    input += 4;

    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    xnn_simd_f32_t vz_prime_0 = xnn_mul_f32(vx_0, vlog2e);
    xnn_simd_f32_t vz_prime_1 = xnn_mul_f32(vx_1, vlog2e);
    xnn_simd_f32_t vz_prime_2 = xnn_mul_f32(vx_2, vlog2e);
    xnn_simd_f32_t vz_prime_3 = xnn_mul_f32(vx_3, vlog2e);
    vz_prime_0 = xnn_min_f32(xnn_max_f32(vz_prime_0, vm15), v16);
    vz_prime_1 = xnn_min_f32(xnn_max_f32(vz_prime_1, vm15), v16);
    vz_prime_2 = xnn_min_f32(xnn_max_f32(vz_prime_2, vm15), v16);
    vz_prime_3 = xnn_min_f32(xnn_max_f32(vz_prime_3, vm15), v16);

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

    // Evaluate the interpolation polynomial for `2^r`.
    xnn_simd_f32_t v2r_0 = xnn_fmadd_f32(vr_0, valpha_3, valpha_2);
    xnn_simd_f32_t v2r_1 = xnn_fmadd_f32(vr_1, valpha_3, valpha_2);
    xnn_simd_f32_t v2r_2 = xnn_fmadd_f32(vr_2, valpha_3, valpha_2);
    xnn_simd_f32_t v2r_3 = xnn_fmadd_f32(vr_3, valpha_3, valpha_2);
    v2r_0 = xnn_fmadd_f32(vr_0, v2r_0, valpha_1);
    v2r_1 = xnn_fmadd_f32(vr_1, v2r_1, valpha_1);
    v2r_2 = xnn_fmadd_f32(vr_2, v2r_2, valpha_1);
    v2r_3 = xnn_fmadd_f32(vr_3, v2r_3, valpha_1);
    v2r_0 = xnn_fmadd_f32(vr_0, v2r_0, vone);
    v2r_1 = xnn_fmadd_f32(vr_1, v2r_1, vone);
    v2r_2 = xnn_fmadd_f32(vr_2, v2r_2, vone);
    v2r_3 = xnn_fmadd_f32(vr_3, v2r_3, vone);

    // Compute 2^z * 2^r.
    const xnn_simd_f32_t vy_0 = xnn_mul_f32(v2z_0, v2r_0);
    const xnn_simd_f32_t vy_1 = xnn_mul_f32(v2z_1, v2r_1);
    const xnn_simd_f32_t vy_2 = xnn_mul_f32(v2z_2, v2r_2);
    const xnn_simd_f32_t vy_3 = xnn_mul_f32(v2z_3, v2r_3);

    xnn_store_f32_f16(output + 0 * 1, vy_0);
    xnn_store_f32_f16(output + 1 * 1, vy_1);
    xnn_store_f32_f16(output + 2 * 1, vy_2);
    xnn_store_f32_f16(output + 3 * 1, vy_3);
    output += 4;
  }
  for (; batch >= 1 * sizeof(xnn_float16); batch -= 1 * sizeof(xnn_float16)) {
    xnn_simd_f32_t vx = xnn_loadu_f16_f32(input);
    input += 1;

    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    xnn_simd_f32_t vz_prime = xnn_mul_f32(vx, vlog2e);
    vz_prime = xnn_min_f32(xnn_max_f32(vz_prime, vm15), v16);

    // Decompose x * log2e into `z` (integer part) and `r` (remainder).
    const xnn_simd_f32_t vz = xnn_qd_round_f32(vz_prime);
    const xnn_simd_f32_t vr = xnn_sub_f32(vz_prime, vz);

    // Compute 2^z.
    const xnn_simd_f32_t v2z = xnn_setexp_f32(vz);

    // Evaluate the interpolation polynomial for `2^r`.
    xnn_simd_f32_t v2r = xnn_fmadd_f32(vr, valpha_3, valpha_2);
    v2r = xnn_fmadd_f32(vr, v2r, valpha_1);
    v2r = xnn_fmadd_f32(vr, v2r, vone);

    // Compute 2^z * 2^r.
    const xnn_simd_f32_t vy = xnn_mul_f32(v2z, v2r);

    xnn_store_f32_f16(output, vy);
    output += 1;
  }
}
