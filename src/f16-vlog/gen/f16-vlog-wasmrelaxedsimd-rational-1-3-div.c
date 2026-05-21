// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vlog/rational-1-3.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/simd/f16-wasmrelaxedsimd.h"
#include "src/xnnpack/vunary.h"

// Define some mathematical constants in case they are not provided by `math.h`.
#ifndef M_LN2
#define M_LN2 0.69314718055994531
#endif  // M_LN2

// Extracts the exponent of the input `a` as a `float16` value.
#ifndef HAVE_XNN_SIGNED_GETEXP_F16
#define HAVE_XNN_SIGNED_GETEXP_F16
static XNN_INLINE xnn_simd_f16_t xnn_signed_getexp_f16(xnn_simd_f16_t a) {
  // See xnn_signed_getexp_f32 for detailed explanation.
  // The bits of IEEE754 half-precision floating-point format are:
  //   s | e e e e e | m m m m m m m m m m
  XNN_SIMD_CONST_F16_FROM_FLOAT(sign_mask, -0.0f);
  XNN_SIMD_CONST_F16_FROM_INT16(sign_and_exp_mask, 0xFC00);
  XNN_SIMD_CONST_F16_FROM_FLOAT(bias_32, 32.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(bias_47, 47.0f);

  // If `a` is `0.0f`, flip its sign bit so that we return `-Inf`.
  a = xnn_or_f16(xnn_and_f16(xnn_cmpeq_f16(a, xnn_zero_f16()), sign_mask), a);

  // Extract the exponent and shift the exponent to the most significant bits of
  // the mantissa.
  const xnn_simd_f16_t exp =
      xnn_sra_f16(xnn_and_f16(a, sign_and_exp_mask), 5);

  // Add the shifted exponent to `32.0f` by copying its bits to the mantissa,
  // then subtract out `47.0f`, i.e. the original `32.0f` plus the `15`
  // exponent bias, resulting in the unbiased exponent.
  return xnn_sub_f16(xnn_or_f16(bias_32, exp), bias_47);
}
#endif  // HAVE_XNN_SIGNED_GETEXP_F16


void xnn_f16_vlog_ukernel__wasmrelaxedsimd_rational_1_3_div_u8(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 8);

  const xnn_float16* i = input;
  xnn_float16* o = output;

  // Some useful constants.
  XNN_SIMD_CONST_F16_FROM_FLOAT(vone, 1.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vln2, 0.69314718f);
  XNN_SIMD_CONST_F16_FROM_INT16(vmantissa_bits_mask, 0x03FF);

  XNN_SIMD_CONST_F16_FROM_FLOAT(vsqrt2, 1.41421356f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vsqrt1_2, 0.70710678f);

  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_1, 4.9951171875e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_2, -8.8439941406e-02f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_3, 4.8828125000e-02f);


  for (; batch >= xnn_simd_bytes_f16; batch -= xnn_simd_bytes_f16) {
    xnn_simd_f16_t vx = xnn_loadu_f16(i);
    i += xnn_simd_size_f16;

    vx = xnn_mul_f16(vx, vsqrt2);
    const xnn_simd_f16_t vexp = xnn_signed_getexp_f16(vx);
    vx = xnn_or_f16(xnn_and_f16(vx, vmantissa_bits_mask), vone);
    vx = xnn_sub_f16(xnn_mul_f16(vx, vsqrt1_2), vone);

    xnn_simd_f16_t vp = vx;

    xnn_simd_f16_t vq = xnn_fmadd_f16(vx, vbeta_3, vbeta_2);
    vq = xnn_fmadd_f16(vx, vq, vbeta_1);
    vq = xnn_fmadd_f16(vx, vq, vone);

    xnn_simd_f16_t vy =  xnn_div_f16(vp, vq);

    vy = xnn_fmadd_f16(vexp, vln2, vy);
    xnn_storeu_f16(o, vy);
    o += xnn_simd_size_f16;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f16_t vx = xnn_load_tail_f16(i, batch >> XNN_LOG2_SIZEOF_FLOAT16);

    vx = xnn_mul_f16(vx, vsqrt2);
    const xnn_simd_f16_t vexp = xnn_signed_getexp_f16(vx);
    vx = xnn_or_f16(xnn_and_f16(vx, vmantissa_bits_mask), vone);
    vx = xnn_sub_f16(xnn_mul_f16(vx, vsqrt1_2), vone);

    xnn_simd_f16_t vp = vx;

    xnn_simd_f16_t vq = xnn_fmadd_f16(vx, vbeta_3, vbeta_2);
    vq = xnn_fmadd_f16(vx, vq, vbeta_1);
    vq = xnn_fmadd_f16(vx, vq, vone);

    xnn_simd_f16_t vy =  xnn_div_f16(vp, vq);

    vy = xnn_fmadd_f16(vexp, vln2, vy);
    xnn_store_tail_f16(o, vy, batch >> XNN_LOG2_SIZEOF_FLOAT16);
  }
}

void xnn_f16_vlog_ukernel__wasmrelaxedsimd_rational_1_3_div_u16(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 8);

  const xnn_float16* i = input;
  xnn_float16* o = output;

  // Some useful constants.
  XNN_SIMD_CONST_F16_FROM_FLOAT(vone, 1.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vln2, 0.69314718f);
  XNN_SIMD_CONST_F16_FROM_INT16(vmantissa_bits_mask, 0x03FF);

  XNN_SIMD_CONST_F16_FROM_FLOAT(vsqrt2, 1.41421356f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vsqrt1_2, 0.70710678f);

  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_1, 4.9951171875e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_2, -8.8439941406e-02f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_3, 4.8828125000e-02f);


  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    xnn_simd_f16_t vx_0 = xnn_loadu_f16(i + 0 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_1 = xnn_loadu_f16(i + 1 * xnn_simd_size_f16);
    i += 16;

    // Scale `x` with `sqrt(2)` so that the exponent is rounded up.
    vx_0 = xnn_mul_f16(vx_0, vsqrt2);
    vx_1 = xnn_mul_f16(vx_1, vsqrt2);

    // Extract the exponent.
    const xnn_simd_f16_t vexp_0 = xnn_signed_getexp_f16(vx_0);
    const xnn_simd_f16_t vexp_1 = xnn_signed_getexp_f16(vx_1);

    // Normalize `x` to an exponent of zero.
    vx_0 = xnn_or_f16(xnn_and_f16(vx_0, vmantissa_bits_mask), vone);
    vx_1 = xnn_or_f16(xnn_and_f16(vx_1, vmantissa_bits_mask), vone);

    // Scale `x` back with `1/sqrt(2)` to move its range from `[1.0, 2.0)` to
    // `[sqrt(1/2), sqrt(2))`, and further subtract `1.0` so that it is around
    // zero, i.e. `[sqrt(1/2) - 1, sqrt(2) - 1)`, or `[−0.29289, 0.4142136)`.
    vx_0 = xnn_sub_f16(xnn_mul_f16(vx_0, vsqrt1_2), vone);
    vx_1 = xnn_sub_f16(xnn_mul_f16(vx_1, vsqrt1_2), vone);

    // Evaluate the numerator polynomial p.
    xnn_simd_f16_t vp_0 = vx_0;
    xnn_simd_f16_t vp_1 = vx_1;

    // Evaluate the denominator polynomial q.
    xnn_simd_f16_t vq_0 = xnn_fmadd_f16(vx_0, vbeta_3, vbeta_2);
    xnn_simd_f16_t vq_1 = xnn_fmadd_f16(vx_1, vbeta_3, vbeta_2);
    vq_0 = xnn_fmadd_f16(vx_0, vq_0, vbeta_1);
    vq_1 = xnn_fmadd_f16(vx_1, vq_1, vbeta_1);
    vq_0 = xnn_fmadd_f16(vx_0, vq_0, vone);
    vq_1 = xnn_fmadd_f16(vx_1, vq_1, vone);

    // Divide the numerator by the denominator.
    xnn_simd_f16_t vy_0 = xnn_div_f16(vp_0, vq_0);
    xnn_simd_f16_t vy_1 = xnn_div_f16(vp_1, vq_1);

    // Put it all together, i.e. `log(x) = `log(2)*exp + y`.
    vy_0 = xnn_fmadd_f16(vexp_0, vln2, vy_0);
    vy_1 = xnn_fmadd_f16(vexp_1, vln2, vy_1);

    xnn_storeu_f16(o + 0 * xnn_simd_size_f16, vy_0);
    xnn_storeu_f16(o + 1 * xnn_simd_size_f16, vy_1);
    o += 16;
  }
  for (; batch >= xnn_simd_bytes_f16; batch -= xnn_simd_bytes_f16) {
    xnn_simd_f16_t vx = xnn_loadu_f16(i);
    i += xnn_simd_size_f16;

    vx = xnn_mul_f16(vx, vsqrt2);
    const xnn_simd_f16_t vexp = xnn_signed_getexp_f16(vx);
    vx = xnn_or_f16(xnn_and_f16(vx, vmantissa_bits_mask), vone);
    vx = xnn_sub_f16(xnn_mul_f16(vx, vsqrt1_2), vone);

    xnn_simd_f16_t vp = vx;

    xnn_simd_f16_t vq = xnn_fmadd_f16(vx, vbeta_3, vbeta_2);
    vq = xnn_fmadd_f16(vx, vq, vbeta_1);
    vq = xnn_fmadd_f16(vx, vq, vone);

    xnn_simd_f16_t vy =  xnn_div_f16(vp, vq);

    vy = xnn_fmadd_f16(vexp, vln2, vy);
    xnn_storeu_f16(o, vy);
    o += xnn_simd_size_f16;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f16_t vx = xnn_load_tail_f16(i, batch >> XNN_LOG2_SIZEOF_FLOAT16);

    vx = xnn_mul_f16(vx, vsqrt2);
    const xnn_simd_f16_t vexp = xnn_signed_getexp_f16(vx);
    vx = xnn_or_f16(xnn_and_f16(vx, vmantissa_bits_mask), vone);
    vx = xnn_sub_f16(xnn_mul_f16(vx, vsqrt1_2), vone);

    xnn_simd_f16_t vp = vx;

    xnn_simd_f16_t vq = xnn_fmadd_f16(vx, vbeta_3, vbeta_2);
    vq = xnn_fmadd_f16(vx, vq, vbeta_1);
    vq = xnn_fmadd_f16(vx, vq, vone);

    xnn_simd_f16_t vy =  xnn_div_f16(vp, vq);

    vy = xnn_fmadd_f16(vexp, vln2, vy);
    xnn_store_tail_f16(o, vy, batch >> XNN_LOG2_SIZEOF_FLOAT16);
  }
}

void xnn_f16_vlog_ukernel__wasmrelaxedsimd_rational_1_3_div_u24(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 8);

  const xnn_float16* i = input;
  xnn_float16* o = output;

  // Some useful constants.
  XNN_SIMD_CONST_F16_FROM_FLOAT(vone, 1.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vln2, 0.69314718f);
  XNN_SIMD_CONST_F16_FROM_INT16(vmantissa_bits_mask, 0x03FF);

  XNN_SIMD_CONST_F16_FROM_FLOAT(vsqrt2, 1.41421356f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vsqrt1_2, 0.70710678f);

  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_1, 4.9951171875e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_2, -8.8439941406e-02f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_3, 4.8828125000e-02f);


  for (; batch >= 24 * sizeof(uint16_t); batch -= 24 * sizeof(uint16_t)) {
    xnn_simd_f16_t vx_0 = xnn_loadu_f16(i + 0 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_1 = xnn_loadu_f16(i + 1 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_2 = xnn_loadu_f16(i + 2 * xnn_simd_size_f16);
    i += 24;

    // Scale `x` with `sqrt(2)` so that the exponent is rounded up.
    vx_0 = xnn_mul_f16(vx_0, vsqrt2);
    vx_1 = xnn_mul_f16(vx_1, vsqrt2);
    vx_2 = xnn_mul_f16(vx_2, vsqrt2);

    // Extract the exponent.
    const xnn_simd_f16_t vexp_0 = xnn_signed_getexp_f16(vx_0);
    const xnn_simd_f16_t vexp_1 = xnn_signed_getexp_f16(vx_1);
    const xnn_simd_f16_t vexp_2 = xnn_signed_getexp_f16(vx_2);

    // Normalize `x` to an exponent of zero.
    vx_0 = xnn_or_f16(xnn_and_f16(vx_0, vmantissa_bits_mask), vone);
    vx_1 = xnn_or_f16(xnn_and_f16(vx_1, vmantissa_bits_mask), vone);
    vx_2 = xnn_or_f16(xnn_and_f16(vx_2, vmantissa_bits_mask), vone);

    // Scale `x` back with `1/sqrt(2)` to move its range from `[1.0, 2.0)` to
    // `[sqrt(1/2), sqrt(2))`, and further subtract `1.0` so that it is around
    // zero, i.e. `[sqrt(1/2) - 1, sqrt(2) - 1)`, or `[−0.29289, 0.4142136)`.
    vx_0 = xnn_sub_f16(xnn_mul_f16(vx_0, vsqrt1_2), vone);
    vx_1 = xnn_sub_f16(xnn_mul_f16(vx_1, vsqrt1_2), vone);
    vx_2 = xnn_sub_f16(xnn_mul_f16(vx_2, vsqrt1_2), vone);

    // Evaluate the numerator polynomial p.
    xnn_simd_f16_t vp_0 = vx_0;
    xnn_simd_f16_t vp_1 = vx_1;
    xnn_simd_f16_t vp_2 = vx_2;

    // Evaluate the denominator polynomial q.
    xnn_simd_f16_t vq_0 = xnn_fmadd_f16(vx_0, vbeta_3, vbeta_2);
    xnn_simd_f16_t vq_1 = xnn_fmadd_f16(vx_1, vbeta_3, vbeta_2);
    xnn_simd_f16_t vq_2 = xnn_fmadd_f16(vx_2, vbeta_3, vbeta_2);
    vq_0 = xnn_fmadd_f16(vx_0, vq_0, vbeta_1);
    vq_1 = xnn_fmadd_f16(vx_1, vq_1, vbeta_1);
    vq_2 = xnn_fmadd_f16(vx_2, vq_2, vbeta_1);
    vq_0 = xnn_fmadd_f16(vx_0, vq_0, vone);
    vq_1 = xnn_fmadd_f16(vx_1, vq_1, vone);
    vq_2 = xnn_fmadd_f16(vx_2, vq_2, vone);

    // Divide the numerator by the denominator.
    xnn_simd_f16_t vy_0 = xnn_div_f16(vp_0, vq_0);
    xnn_simd_f16_t vy_1 = xnn_div_f16(vp_1, vq_1);
    xnn_simd_f16_t vy_2 = xnn_div_f16(vp_2, vq_2);

    // Put it all together, i.e. `log(x) = `log(2)*exp + y`.
    vy_0 = xnn_fmadd_f16(vexp_0, vln2, vy_0);
    vy_1 = xnn_fmadd_f16(vexp_1, vln2, vy_1);
    vy_2 = xnn_fmadd_f16(vexp_2, vln2, vy_2);

    xnn_storeu_f16(o + 0 * xnn_simd_size_f16, vy_0);
    xnn_storeu_f16(o + 1 * xnn_simd_size_f16, vy_1);
    xnn_storeu_f16(o + 2 * xnn_simd_size_f16, vy_2);
    o += 24;
  }
  for (; batch >= xnn_simd_bytes_f16; batch -= xnn_simd_bytes_f16) {
    xnn_simd_f16_t vx = xnn_loadu_f16(i);
    i += xnn_simd_size_f16;

    vx = xnn_mul_f16(vx, vsqrt2);
    const xnn_simd_f16_t vexp = xnn_signed_getexp_f16(vx);
    vx = xnn_or_f16(xnn_and_f16(vx, vmantissa_bits_mask), vone);
    vx = xnn_sub_f16(xnn_mul_f16(vx, vsqrt1_2), vone);

    xnn_simd_f16_t vp = vx;

    xnn_simd_f16_t vq = xnn_fmadd_f16(vx, vbeta_3, vbeta_2);
    vq = xnn_fmadd_f16(vx, vq, vbeta_1);
    vq = xnn_fmadd_f16(vx, vq, vone);

    xnn_simd_f16_t vy =  xnn_div_f16(vp, vq);

    vy = xnn_fmadd_f16(vexp, vln2, vy);
    xnn_storeu_f16(o, vy);
    o += xnn_simd_size_f16;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f16_t vx = xnn_load_tail_f16(i, batch >> XNN_LOG2_SIZEOF_FLOAT16);

    vx = xnn_mul_f16(vx, vsqrt2);
    const xnn_simd_f16_t vexp = xnn_signed_getexp_f16(vx);
    vx = xnn_or_f16(xnn_and_f16(vx, vmantissa_bits_mask), vone);
    vx = xnn_sub_f16(xnn_mul_f16(vx, vsqrt1_2), vone);

    xnn_simd_f16_t vp = vx;

    xnn_simd_f16_t vq = xnn_fmadd_f16(vx, vbeta_3, vbeta_2);
    vq = xnn_fmadd_f16(vx, vq, vbeta_1);
    vq = xnn_fmadd_f16(vx, vq, vone);

    xnn_simd_f16_t vy =  xnn_div_f16(vp, vq);

    vy = xnn_fmadd_f16(vexp, vln2, vy);
    xnn_store_tail_f16(o, vy, batch >> XNN_LOG2_SIZEOF_FLOAT16);
  }
}

void xnn_f16_vlog_ukernel__wasmrelaxedsimd_rational_1_3_div_u32(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 8);

  const xnn_float16* i = input;
  xnn_float16* o = output;

  // Some useful constants.
  XNN_SIMD_CONST_F16_FROM_FLOAT(vone, 1.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vln2, 0.69314718f);
  XNN_SIMD_CONST_F16_FROM_INT16(vmantissa_bits_mask, 0x03FF);

  XNN_SIMD_CONST_F16_FROM_FLOAT(vsqrt2, 1.41421356f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vsqrt1_2, 0.70710678f);

  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_1, 4.9951171875e-01f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_2, -8.8439941406e-02f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vbeta_3, 4.8828125000e-02f);


  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    xnn_simd_f16_t vx_0 = xnn_loadu_f16(i + 0 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_1 = xnn_loadu_f16(i + 1 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_2 = xnn_loadu_f16(i + 2 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_3 = xnn_loadu_f16(i + 3 * xnn_simd_size_f16);
    i += 32;

    // Scale `x` with `sqrt(2)` so that the exponent is rounded up.
    vx_0 = xnn_mul_f16(vx_0, vsqrt2);
    vx_1 = xnn_mul_f16(vx_1, vsqrt2);
    vx_2 = xnn_mul_f16(vx_2, vsqrt2);
    vx_3 = xnn_mul_f16(vx_3, vsqrt2);

    // Extract the exponent.
    const xnn_simd_f16_t vexp_0 = xnn_signed_getexp_f16(vx_0);
    const xnn_simd_f16_t vexp_1 = xnn_signed_getexp_f16(vx_1);
    const xnn_simd_f16_t vexp_2 = xnn_signed_getexp_f16(vx_2);
    const xnn_simd_f16_t vexp_3 = xnn_signed_getexp_f16(vx_3);

    // Normalize `x` to an exponent of zero.
    vx_0 = xnn_or_f16(xnn_and_f16(vx_0, vmantissa_bits_mask), vone);
    vx_1 = xnn_or_f16(xnn_and_f16(vx_1, vmantissa_bits_mask), vone);
    vx_2 = xnn_or_f16(xnn_and_f16(vx_2, vmantissa_bits_mask), vone);
    vx_3 = xnn_or_f16(xnn_and_f16(vx_3, vmantissa_bits_mask), vone);

    // Scale `x` back with `1/sqrt(2)` to move its range from `[1.0, 2.0)` to
    // `[sqrt(1/2), sqrt(2))`, and further subtract `1.0` so that it is around
    // zero, i.e. `[sqrt(1/2) - 1, sqrt(2) - 1)`, or `[−0.29289, 0.4142136)`.
    vx_0 = xnn_sub_f16(xnn_mul_f16(vx_0, vsqrt1_2), vone);
    vx_1 = xnn_sub_f16(xnn_mul_f16(vx_1, vsqrt1_2), vone);
    vx_2 = xnn_sub_f16(xnn_mul_f16(vx_2, vsqrt1_2), vone);
    vx_3 = xnn_sub_f16(xnn_mul_f16(vx_3, vsqrt1_2), vone);

    // Evaluate the numerator polynomial p.
    xnn_simd_f16_t vp_0 = vx_0;
    xnn_simd_f16_t vp_1 = vx_1;
    xnn_simd_f16_t vp_2 = vx_2;
    xnn_simd_f16_t vp_3 = vx_3;

    // Evaluate the denominator polynomial q.
    xnn_simd_f16_t vq_0 = xnn_fmadd_f16(vx_0, vbeta_3, vbeta_2);
    xnn_simd_f16_t vq_1 = xnn_fmadd_f16(vx_1, vbeta_3, vbeta_2);
    xnn_simd_f16_t vq_2 = xnn_fmadd_f16(vx_2, vbeta_3, vbeta_2);
    xnn_simd_f16_t vq_3 = xnn_fmadd_f16(vx_3, vbeta_3, vbeta_2);
    vq_0 = xnn_fmadd_f16(vx_0, vq_0, vbeta_1);
    vq_1 = xnn_fmadd_f16(vx_1, vq_1, vbeta_1);
    vq_2 = xnn_fmadd_f16(vx_2, vq_2, vbeta_1);
    vq_3 = xnn_fmadd_f16(vx_3, vq_3, vbeta_1);
    vq_0 = xnn_fmadd_f16(vx_0, vq_0, vone);
    vq_1 = xnn_fmadd_f16(vx_1, vq_1, vone);
    vq_2 = xnn_fmadd_f16(vx_2, vq_2, vone);
    vq_3 = xnn_fmadd_f16(vx_3, vq_3, vone);

    // Divide the numerator by the denominator.
    xnn_simd_f16_t vy_0 = xnn_div_f16(vp_0, vq_0);
    xnn_simd_f16_t vy_1 = xnn_div_f16(vp_1, vq_1);
    xnn_simd_f16_t vy_2 = xnn_div_f16(vp_2, vq_2);
    xnn_simd_f16_t vy_3 = xnn_div_f16(vp_3, vq_3);

    // Put it all together, i.e. `log(x) = `log(2)*exp + y`.
    vy_0 = xnn_fmadd_f16(vexp_0, vln2, vy_0);
    vy_1 = xnn_fmadd_f16(vexp_1, vln2, vy_1);
    vy_2 = xnn_fmadd_f16(vexp_2, vln2, vy_2);
    vy_3 = xnn_fmadd_f16(vexp_3, vln2, vy_3);

    xnn_storeu_f16(o + 0 * xnn_simd_size_f16, vy_0);
    xnn_storeu_f16(o + 1 * xnn_simd_size_f16, vy_1);
    xnn_storeu_f16(o + 2 * xnn_simd_size_f16, vy_2);
    xnn_storeu_f16(o + 3 * xnn_simd_size_f16, vy_3);
    o += 32;
  }
  for (; batch >= xnn_simd_bytes_f16; batch -= xnn_simd_bytes_f16) {
    xnn_simd_f16_t vx = xnn_loadu_f16(i);
    i += xnn_simd_size_f16;

    vx = xnn_mul_f16(vx, vsqrt2);
    const xnn_simd_f16_t vexp = xnn_signed_getexp_f16(vx);
    vx = xnn_or_f16(xnn_and_f16(vx, vmantissa_bits_mask), vone);
    vx = xnn_sub_f16(xnn_mul_f16(vx, vsqrt1_2), vone);

    xnn_simd_f16_t vp = vx;

    xnn_simd_f16_t vq = xnn_fmadd_f16(vx, vbeta_3, vbeta_2);
    vq = xnn_fmadd_f16(vx, vq, vbeta_1);
    vq = xnn_fmadd_f16(vx, vq, vone);

    xnn_simd_f16_t vy =  xnn_div_f16(vp, vq);

    vy = xnn_fmadd_f16(vexp, vln2, vy);
    xnn_storeu_f16(o, vy);
    o += xnn_simd_size_f16;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f16_t vx = xnn_load_tail_f16(i, batch >> XNN_LOG2_SIZEOF_FLOAT16);

    vx = xnn_mul_f16(vx, vsqrt2);
    const xnn_simd_f16_t vexp = xnn_signed_getexp_f16(vx);
    vx = xnn_or_f16(xnn_and_f16(vx, vmantissa_bits_mask), vone);
    vx = xnn_sub_f16(xnn_mul_f16(vx, vsqrt1_2), vone);

    xnn_simd_f16_t vp = vx;

    xnn_simd_f16_t vq = xnn_fmadd_f16(vx, vbeta_3, vbeta_2);
    vq = xnn_fmadd_f16(vx, vq, vbeta_1);
    vq = xnn_fmadd_f16(vx, vq, vone);

    xnn_simd_f16_t vy =  xnn_div_f16(vp, vq);

    vy = xnn_fmadd_f16(vexp, vln2, vy);
    xnn_store_tail_f16(o, vy, batch >> XNN_LOG2_SIZEOF_FLOAT16);
  }
}
