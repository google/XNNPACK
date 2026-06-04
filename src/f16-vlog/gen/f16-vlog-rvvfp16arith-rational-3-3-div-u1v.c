// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vlog/rvv-rational-3-3.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"

typedef union {
  xnn_float16 f;
  uint16_t u;
  int16_t s;
} xnn_f16_i16_t;

// Define some mathematical constants in case they are not provided by `math.h`.
#ifndef M_LN2
#define M_LN2 0.69314718055994531
#endif  // M_LN2

// Extracts the exponent of the input `a` as a `float16` value.
static XNN_INLINE vfloat16m1_t xnn_signed_getexp_f16(vfloat16m1_t a, size_t vl) {
  // See xnn_signed_getexp_f32 for detailed explanation.
  // The bits of IEEE754 half-precision floating-point format are:
  //   s | e e e e e | m m m m m m m m m m
  const xnn_f16_i16_t sign_mask = {.f = -0.0f};
  const int16_t sign_and_exp_mask = 0xfc00;
  const xnn_f16_i16_t bias_32 = {.f = 32.0f};
  const xnn_float16 bias_47 = 47.0f;

  // If `a` is `0.0f`, flip its sign bit so that we return `-Inf`.
  a =  __riscv_vreinterpret_f16m1(__riscv_vor_mu(__riscv_vmfeq(a, 0.0f, vl), __riscv_vreinterpret_i16m1(a), __riscv_vreinterpret_i16m1(a), sign_mask.s, vl));

  // Extract the exponent and shift the exponent to the most significant bits of
  // the mantissa.
  const vint16m1_t exp =
    __riscv_vsra(__riscv_vand(__riscv_vreinterpret_i16m1(a), sign_and_exp_mask, vl), 5, vl);

  // Add the shifted exponent to `32.0f` by copying its bits to the mantissa,
  // then subtract out `47.0f`, i.e. the original `32.0f` plus the `15`
  // exponent bias, resulting in the unbiased exponent.
  return __riscv_vfsub(__riscv_vreinterpret_f16m1(__riscv_vor(exp, bias_32.s, vl)), bias_47, vl);
}

void xnn_f16_vlog_ukernel__rvvfp16arith_rational_3_3_div_u1v(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT16;

  // Some useful constants.
  const xnn_f16_i16_t vone = {.f = 1.0f};
  const xnn_float16 vln2 = 0.69314718f;
  const int16_t vmantissa_bits_mask = 0x03FF;

  const xnn_float16 vsqrt2 = 1.41421356f;
  const xnn_float16 vsqrt1_2 = 0.70710678f;

  const xnn_float16 valpha_3 = 0.18249969f;
  const xnn_float16 vbeta_1 = 1.5f;
  const xnn_float16 vbeta_2 = 0.59917002f;
  const xnn_float16 vbeta_3 = 0.04958499f;
 
  do {
    size_t vl = __riscv_vsetvl_e16m1(batch); batch -= vl;
    vfloat16m1_t vx = __riscv_vle16_v_f16m1(input, vl); input += vl;

    // Scale `x` with `sqrt(2)` so that the exponent is rounded up.
    vx = __riscv_vfmul(vx, vsqrt2, vl);

    // Extract the exponent.
    const vfloat16m1_t vexp = xnn_signed_getexp_f16(vx, vl);

    // Normalize `x` to an exponent of zero.
    vx = __riscv_vreinterpret_f16m1(__riscv_vor(__riscv_vand(__riscv_vreinterpret_i16m1(vx), vmantissa_bits_mask, vl), vone.s, vl));
    
    // Scale `x` back with `1/sqrt(2)` to move its range from `[1.0, 2.0)` to
    // `[sqrt(1/2), sqrt(2))`, and further subtract `1.0` so that it is around
    // zero, i.e. `[sqrt(1/2) - 1, sqrt(2) - 1)`, or `[−0.29289, 0.4142136)`.

    // TODO: A 3/3 rational polynomial is mathematically overkill for FP16 precision
    // and requires an expensive division. Consider switching to a degree-4 pure
    // minimax polynomial (e.g. Remez) to eliminate the denominator evaluation and
    // the division/NR steps, which will significantly improve throughput.
    vx = __riscv_vfsub(__riscv_vfmul(vx, vsqrt1_2, vl), vone.f, vl);

    // Evaluate the numerator polynomial p.
    vfloat16m1_t vp = __riscv_vfadd(__riscv_vfmul(vx, valpha_3, vl), vone.f, vl);
    vp = __riscv_vfadd(__riscv_vfmul(vx, vp, vl), vone.f, vl);
    vp = __riscv_vfmul(vx, vp, vl);

    // Evaluate the denominator polynomial q.
    vfloat16m1_t vq = __riscv_vfadd(__riscv_vfmul(vx, vbeta_3, vl), vbeta_2, vl);
    vq = __riscv_vfadd(__riscv_vfmul(vx, vq, vl), vbeta_1, vl);
    vq = __riscv_vfadd(__riscv_vfmul(vx, vq, vl), vone.f, vl);

    // Divide the numerator by the denominator.
    vfloat16m1_t vy = __riscv_vfdiv(vp, vq, vl);

    // Put it all together, i.e. `log(x) = `log(2)*exp + y`.
    vy = __riscv_vfmacc(vy, vln2, vexp, vl);

    __riscv_vse16(output, vy, vl); output += vl;

  } while (batch > 0);
}
