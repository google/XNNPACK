// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vlog/rvv-rational-3-3.c.in
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
#include "src/xnnpack/simd/f32-scalar.h" // xnn_f32_i32_t

// Define some mathematical constants in case they are not provided by `math.h`.
#ifndef M_LN2
#define M_LN2 0.69314718055994531
#endif  // M_LN2

// Extracts the exponent of the input `a` as a `float` value.
static XNN_INLINE vfloat32m2_t xnn_signed_getexp_f32(vfloat32m2_t a, size_t vl) {
  // The bits of IEEE754 single-precision floating-point format are:
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
  const xnn_f32_i32_t sign_mask = {.f = -0.0f};
  const int32_t sign_and_exp_mask = 0xff800000;
  const xnn_f32_i32_t bias_256 = {.f = 256.0f};
  const float bias_383 = 383.0f;

  // If `a` is `0.0f`, flip its sign bit so that we return `-Inf`.
  a =  __riscv_vreinterpret_f32m2(__riscv_vor_mu(__riscv_vmfeq(a, 0.0f, vl), __riscv_vreinterpret_i32m2(a), __riscv_vreinterpret_i32m2(a), sign_mask.s, vl));

  // Extract the exponent and shift the exponent to the most significant bits of
  // the mantissa.
  const vint32m2_t exp =
    __riscv_vsra(__riscv_vand(__riscv_vreinterpret_i32m2(a), sign_and_exp_mask, vl), 8, vl);

  // Add the shifted exponent to `256.0f` by copying its bits to the mantissa,
  // then subtract out `383.0f`, i.e. the original `256.0f` plus the `127`
  // exponent bias, resulting in the unbiased exponent.
  return __riscv_vfsub(__riscv_vreinterpret_f32m2(__riscv_vor(exp, bias_256.s, vl)), bias_383, vl);
}


void xnn_f32_vlog_ukernel__rvv_rational_3_3_div_u2v(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT;

  // Some useful constants.
  const xnn_f32_i32_t vone = {.f = 1.0f};
  const float vln2 = M_LN2;
  const int32_t vmantissa_bits_mask = 0x007FFFFFUL;

  // Note that these two values are not _exactly_ `(float)M_SQRT2` and
  // `(float)M_SQRT1_2`, but are instead chosen such that their product is
  // exactly `1.0f` when evaluated in `float` precision.
  const float vsqrt2 = 1.4142134190e+00;
  const float vsqrt1_2 = 7.0710688829e-01;

  // The monomial coefficients of the numerator polynomial.
  // const float valpha_0 = 0.0f;
  // const float valpha_1 = 1.0f;
  // const float valpha_2 = 1.0f;
  const float valpha_3 = 1.824996918440e-01;

  // The monomial coefficients of the denominator polynomial.
  // const float vbeta_0 = 1.0f;
  const float vbeta_1 = 1.5f;
  const float vbeta_2 = 0.599170029163;
  const float vbeta_3 = 0.049584995955;

  do {
    size_t vl = __riscv_vsetvl_e32m2(batch); batch -= vl;
    vfloat32m2_t vx = __riscv_vle32_v_f32m2(input, vl); input += vl;

    // Scale `x` with `sqrt(2)` so that the exponent is rounded up.
    vx = __riscv_vfmul(vx, vsqrt2, vl);

    // Extract the exponent.
    const vfloat32m2_t vexp = xnn_signed_getexp_f32(vx, vl);

    // Normalize `x` to an exponent of zero.
    vx = __riscv_vreinterpret_f32m2(__riscv_vor(__riscv_vand(__riscv_vreinterpret_i32m2(vx), vmantissa_bits_mask, vl), vone.s, vl));

    // Scale `x` back with `1/sqrt(2)` to move its range from `[1.0, 2.0)` to
    // `[sqrt(1/2), sqrt(2))`, and further subtract `1.0` so that it is around
    // zero, i.e. `[sqrt(1/2) - 1, sqrt(2) - 1)`, or `[−0.29289, 0.4142136)`.
    vx = __riscv_vfsub(__riscv_vfmul(vx, vsqrt1_2, vl), vone.f, vl);

    // In the following, we use a 3/2-degree rational polynomial to
    // approximate the (shifted) `log(x + 1)` on the (shifted) interval
    // `[sqrt(1/2) - 1, sqrt(2) - 1)`. The shifted interval is chosen so that
    // `f(0) = 0`.

    // Evaluate the numerator polynomial p.
    vfloat32m2_t vp = __riscv_vfadd(__riscv_vfmul(vx, valpha_3, vl), vone.f, vl);
    vp = __riscv_vfadd(__riscv_vfmul(vx, vp, vl), vone.f, vl);
    vp = __riscv_vfmul(vx, vp, vl);

    // Evaluate the denominator polynomial q.
    vfloat32m2_t vq = __riscv_vfadd(__riscv_vfmul(vx, vbeta_3, vl), vbeta_2, vl);
    vq = __riscv_vfadd(__riscv_vfmul(vx, vq, vl), vbeta_1, vl);
    vq = __riscv_vfadd(__riscv_vfmul(vx, vq, vl), vone.f, vl);

    // Divide the numerator by the denominator.
    vfloat32m2_t vy = __riscv_vfdiv(vp, vq, vl);

    // Put it all together, i.e. `log(x) = `log(2)*exp + y`.
    vy = __riscv_vfmacc(vy, vln2, vexp, vl);

    __riscv_vse32(output, vy, vl); output += vl;

  } while (batch > 0);
}
