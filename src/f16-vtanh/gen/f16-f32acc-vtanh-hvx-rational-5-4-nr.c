// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vtanh/f16-f32acc-rational-5-4.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"

#include "src/xnnpack/simd/f32-hvx.h"



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



void xnn_f16_f32acc_vtanh_ukernel__hvx_rational_5_4_nr_u32(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Cap the inputs to this value as `tanh(x)` will always be `+/-1.0f` beyond
  // this point. This value is chosen as the first floating point number as of
  // which the interpolation returns 1.0f.
  XNN_SIMD_CONST_F32(vmax_x, 4.5f);
  XNN_SIMD_CONST_F32(vmin_x, -4.5f);

  // The monomial coefficients of the numerator polynomial (odd).
  // XNN_SIMD_CONST_F32(valpha_1, 1.0000000000e+00f);
  XNN_SIMD_CONST_F32(valpha_3, 1.0632324219e-01f);
  XNN_SIMD_CONST_F32(valpha_5, 8.1062316895e-04f);

  // The monomial coefficients of the denominator polynomial (even).
  // XNN_SIMD_CONST_F32(vbeta_0, 1.0000000000e+00f);
  XNN_SIMD_CONST_F32(vbeta_2, 4.3945312500e-01f);
  XNN_SIMD_CONST_F32(vbeta_4, 1.4091491699e-02f);

  // Some useful constants.
  XNN_SIMD_CONST_F32(vone, 1.0f);

  for (; batch >= 32 * sizeof(xnn_float16); batch -= 32 * sizeof(xnn_float16)) {
    xnn_simd_f32_t vx = xnn_loadu_f16_f32(input);
    input += 32;

    // Clamp the inputs to the interpolation range.
    vx = xnn_min_f32(vmax_x, vx);
    vx = xnn_max_f32(vmin_x, vx);

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
    const xnn_simd_f32_t vy = xnn_mul_f32(vp, xnn_rcp_f32(vq));

    xnn_store_f32_f16(output, vy);
    output += 32;
  }
  if (batch != 0) {
    xnn_simd_f32_t vx = xnn_load_f16_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT16);

    // Clamp the inputs to the interpolation range.
    vx = xnn_min_f32(vmax_x, vx);
    vx = xnn_max_f32(vmin_x, vx);

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
    const xnn_simd_f32_t vy = xnn_mul_f32(vp, xnn_rcp_f32(vq));

    xnn_store_f32_f16_tail(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT16);
  }
}

void xnn_f16_f32acc_vtanh_ukernel__hvx_rational_5_4_nr_u64(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Cap the inputs to this value as `tanh(x)` will always be `+/-1.0f` beyond
  // this point. This value is chosen as the first floating point number as of
  // which the interpolation returns 1.0f.
  XNN_SIMD_CONST_F32(vmax_x, 4.5f);
  XNN_SIMD_CONST_F32(vmin_x, -4.5f);

  // The monomial coefficients of the numerator polynomial (odd).
  // XNN_SIMD_CONST_F32(valpha_1, 1.0000000000e+00f);
  XNN_SIMD_CONST_F32(valpha_3, 1.0632324219e-01f);
  XNN_SIMD_CONST_F32(valpha_5, 8.1062316895e-04f);

  // The monomial coefficients of the denominator polynomial (even).
  // XNN_SIMD_CONST_F32(vbeta_0, 1.0000000000e+00f);
  XNN_SIMD_CONST_F32(vbeta_2, 4.3945312500e-01f);
  XNN_SIMD_CONST_F32(vbeta_4, 1.4091491699e-02f);

  // Some useful constants.
  XNN_SIMD_CONST_F32(vone, 1.0f);

  for (; batch >= 64 * sizeof(xnn_float16); batch -= 64 * sizeof(xnn_float16)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f16_f32(input + 0 * 32);
    xnn_simd_f32_t vx_1 = xnn_loadu_f16_f32(input + 1 * 32);
    input += 64;

    // Clamp the inputs to the interpolation range.
    vx_0 = xnn_min_f32(vmax_x, vx_0);
    vx_1 = xnn_min_f32(vmax_x, vx_1);
    vx_0 = xnn_max_f32(vmin_x, vx_0);
    vx_1 = xnn_max_f32(vmin_x, vx_1);

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
    const xnn_simd_f32_t vy_0 = xnn_mul_f32(vp_0, xnn_rcp_f32(vq_0));
    const xnn_simd_f32_t vy_1 = xnn_mul_f32(vp_1, xnn_rcp_f32(vq_1));

    xnn_store_f32_f16(output + 0 * 32, vy_0);
    xnn_store_f32_f16(output + 1 * 32, vy_1);
    output += 64;
  }
  for (; batch >= 32 * sizeof(xnn_float16); batch -= 32 * sizeof(xnn_float16)) {
    xnn_simd_f32_t vx = xnn_loadu_f16_f32(input);
    input += 32;

    // Clamp the inputs to the interpolation range.
    vx = xnn_min_f32(vmax_x, vx);
    vx = xnn_max_f32(vmin_x, vx);

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
    const xnn_simd_f32_t vy = xnn_mul_f32(vp, xnn_rcp_f32(vq));

    xnn_store_f32_f16(output, vy);
    output += 32;
  }
  if (batch != 0) {
    xnn_simd_f32_t vx = xnn_load_f16_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT16);

    // Clamp the inputs to the interpolation range.
    vx = xnn_min_f32(vmax_x, vx);
    vx = xnn_max_f32(vmin_x, vx);

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
    const xnn_simd_f32_t vy = xnn_mul_f32(vp, xnn_rcp_f32(vq));

    xnn_store_f32_f16_tail(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT16);
  }
}

void xnn_f16_f32acc_vtanh_ukernel__hvx_rational_5_4_nr_u128(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Cap the inputs to this value as `tanh(x)` will always be `+/-1.0f` beyond
  // this point. This value is chosen as the first floating point number as of
  // which the interpolation returns 1.0f.
  XNN_SIMD_CONST_F32(vmax_x, 4.5f);
  XNN_SIMD_CONST_F32(vmin_x, -4.5f);

  // The monomial coefficients of the numerator polynomial (odd).
  // XNN_SIMD_CONST_F32(valpha_1, 1.0000000000e+00f);
  XNN_SIMD_CONST_F32(valpha_3, 1.0632324219e-01f);
  XNN_SIMD_CONST_F32(valpha_5, 8.1062316895e-04f);

  // The monomial coefficients of the denominator polynomial (even).
  // XNN_SIMD_CONST_F32(vbeta_0, 1.0000000000e+00f);
  XNN_SIMD_CONST_F32(vbeta_2, 4.3945312500e-01f);
  XNN_SIMD_CONST_F32(vbeta_4, 1.4091491699e-02f);

  // Some useful constants.
  XNN_SIMD_CONST_F32(vone, 1.0f);

  for (; batch >= 128 * sizeof(xnn_float16); batch -= 128 * sizeof(xnn_float16)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f16_f32(input + 0 * 32);
    xnn_simd_f32_t vx_1 = xnn_loadu_f16_f32(input + 1 * 32);
    xnn_simd_f32_t vx_2 = xnn_loadu_f16_f32(input + 2 * 32);
    xnn_simd_f32_t vx_3 = xnn_loadu_f16_f32(input + 3 * 32);
    input += 128;

    // Clamp the inputs to the interpolation range.
    vx_0 = xnn_min_f32(vmax_x, vx_0);
    vx_1 = xnn_min_f32(vmax_x, vx_1);
    vx_2 = xnn_min_f32(vmax_x, vx_2);
    vx_3 = xnn_min_f32(vmax_x, vx_3);
    vx_0 = xnn_max_f32(vmin_x, vx_0);
    vx_1 = xnn_max_f32(vmin_x, vx_1);
    vx_2 = xnn_max_f32(vmin_x, vx_2);
    vx_3 = xnn_max_f32(vmin_x, vx_3);

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
    const xnn_simd_f32_t vy_0 = xnn_mul_f32(vp_0, xnn_rcp_f32(vq_0));
    const xnn_simd_f32_t vy_1 = xnn_mul_f32(vp_1, xnn_rcp_f32(vq_1));
    const xnn_simd_f32_t vy_2 = xnn_mul_f32(vp_2, xnn_rcp_f32(vq_2));
    const xnn_simd_f32_t vy_3 = xnn_mul_f32(vp_3, xnn_rcp_f32(vq_3));

    xnn_store_f32_f16(output + 0 * 32, vy_0);
    xnn_store_f32_f16(output + 1 * 32, vy_1);
    xnn_store_f32_f16(output + 2 * 32, vy_2);
    xnn_store_f32_f16(output + 3 * 32, vy_3);
    output += 128;
  }
  for (; batch >= 32 * sizeof(xnn_float16); batch -= 32 * sizeof(xnn_float16)) {
    xnn_simd_f32_t vx = xnn_loadu_f16_f32(input);
    input += 32;

    // Clamp the inputs to the interpolation range.
    vx = xnn_min_f32(vmax_x, vx);
    vx = xnn_max_f32(vmin_x, vx);

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
    const xnn_simd_f32_t vy = xnn_mul_f32(vp, xnn_rcp_f32(vq));

    xnn_store_f32_f16(output, vy);
    output += 32;
  }
  if (batch != 0) {
    xnn_simd_f32_t vx = xnn_load_f16_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT16);

    // Clamp the inputs to the interpolation range.
    vx = xnn_min_f32(vmax_x, vx);
    vx = xnn_max_f32(vmin_x, vx);

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
    const xnn_simd_f32_t vy = xnn_mul_f32(vp, xnn_rcp_f32(vq));

    xnn_store_f32_f16_tail(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT16);
  }
}
