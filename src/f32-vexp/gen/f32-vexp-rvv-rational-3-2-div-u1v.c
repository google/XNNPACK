// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vexp/rvv-rational-3-2.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


static XNN_INLINE vfloat32m1_t xnn_setexp_f32(vfloat32m1_t vx, size_t vl) {
  // If `x` is an floating point value in the range [-127, 128], then
  // `(x + magic) << 23` will generate the floating point value corresponding
  // to `2^round(x)` (2^-127 and 2^128 will flush to zero and infinity,
  // respectively).
  const float vmagic = 8388735.0f;  // 2^23 + 127.
  return __riscv_vreinterpret_f32m1(__riscv_vsll(__riscv_vreinterpret_u32m1(__riscv_vfadd(vx, vmagic, vl)), 23, vl));
}

// Quick-and-dirty round to nearest, only works for floats in the range
// `[2^-22, 2^22)`.
static XNN_INLINE vfloat32m1_t xnn_qd_round_f32(vfloat32m1_t vx, size_t vl) {
  // If `x` is an floating point value in the range `[2^-22, 2^22)`, then
  // `(x + magic) - magic`` will generate the floating point value corresponding
  // to `round(x)`.
  const float vmagic = 12582912.0f;  // 2^23 + 2^22.
  return __riscv_vfsub(__riscv_vfadd(vx, vmagic, vl), vmagic, vl);
}

void xnn_f32_vexp_ukernel__rvv_rational_3_2_div_u1v(
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

  // The monomial coefficients of the numerator polynomial (`valpha_0` = 1.0).
  const float valpha_1 = 4.1594290733e-01f;
  const float valpha_2 = 7.2068706155e-02f;
  const float valpha_3 = 5.5380910635e-03f;

  // The monomial coefficients of the denominator polynomial (`vbeta_01 = 1.0).
  const float vbeta_1 = -2.7720427513e-01f;
  const float vbeta_2 = 2.3986088112e-02f;

  // Some useful constants.
  const float vlog2e = 1.44269504089f;
  const float v128 = 128.0f;
  const float vm127 = -127.0f;
  const float vone = 1.0f;
 
  do {
    size_t vl = __riscv_vsetvl_e32m1(batch); batch -= vl;
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(input, vl); input += vl;

    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    vfloat32m1_t vz_prime = __riscv_vfmul(vx, vlog2e, vl);
    vz_prime = __riscv_vfmin(__riscv_vfmax(vz_prime, vm127, vl), v128, vl);

    // Decompose x * log2e into `z` (integer part) and `r` (remainder).
    const vfloat32m1_t vz = xnn_qd_round_f32(vz_prime, vl);
    const vfloat32m1_t vr = __riscv_vfsub(vz_prime, vz, vl);

    // Compute 2^z.
    vfloat32m1_t v2z = xnn_setexp_f32(vz, vl);

    // Evaluate the numerator polynomial p(f).
    vfloat32m1_t vp = __riscv_vfadd(__riscv_vfmul(vr, valpha_3, vl), valpha_2, vl);
    vp = __riscv_vfadd(__riscv_vfmul(vr, vp, vl), valpha_1, vl);
    vp = __riscv_vfadd(__riscv_vfmul(vr, vp, vl), vone, vl);

    // Evaluate the denominator polynomial q(r).
    vfloat32m1_t vq = __riscv_vfadd(__riscv_vfmul(vr, vbeta_2, vl), vbeta_1, vl);
    vq = __riscv_vfadd(__riscv_vfmul(vr, vq, vl), vone, vl);

    // Divide the numerator by the denominator, obtaining 2^r.
    const vfloat32m1_t v2r = __riscv_vfdiv(vp, vq, vl);

    // Compute 2^z * 2^r.
    const vfloat32m1_t vy = __riscv_vfmul(v2z, v2r, vl);

    __riscv_vse32(output, vy, vl); output += vl;

  } while (batch > 0);
}
