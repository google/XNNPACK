// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vlog/rvv-rational-1-3.c.in
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

// Extracts the exponent of the input `a` as a `float16` value.
static XNN_INLINE vfloat16m8_t xnn_signed_getexp_f16(vfloat16m8_t a, size_t vl) {
  const uint16_t sign_mask = 0x8000;
  const uint16_t sign_and_exp_mask = 0xFC00;
  const uint16_t bias_32_bits = 0x5000; // 32.0f in f16 bits
  const xnn_float16 bias_47 = 47.0f;

  // If `a` is `0.0f`, flip its sign bit so that we return `-Inf`.
  vuint16m8_t va = __riscv_vreinterpret_v_f16m8_u16m8(a);
  vbool2_t is_zero = __riscv_vmfeq_vf_f16m8_b2(a, 0.0f, vl);
  va = __riscv_vor_vx_u16m8_mu(is_zero, va, va, sign_mask, vl);

  // Extract the exponent and shift the exponent to the most significant bits of
  // the mantissa.
  const vuint16m8_t exp =
      __riscv_vsrl_vx_u16m8(__riscv_vand_vx_u16m8(va, sign_and_exp_mask, vl), 5, vl);

  // Add the shifted exponent to `32.0f` by copying its bits to the mantissa,
  // then subtract out `47.0f`, i.e. the original `32.0f` plus the `15`
  // exponent bias, resulting in the unbiased exponent.
  return __riscv_vfsub_vf_f16m8(__riscv_vreinterpret_v_u16m8_f16m8(__riscv_vor_vx_u16m8(exp, bias_32_bits, vl)), bias_47, vl);
}


void xnn_f16_vlog_ukernel__rvvfp16arith_rational_1_3_nr_u8v(
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
  const xnn_float16 vone = 1.0f;
  const uint16_t vone_bits = 0x3C00;
  const xnn_float16 vln2 = 0.69314718f;
  const uint16_t vmantissa_bits_mask = 0x03FF;

  const xnn_float16 vsqrt2 = 1.41421356f;
  const xnn_float16 vsqrt1_2 = 0.70710678f;

  const xnn_float16 vbeta_1 = 4.9951171875e-01f;
  const xnn_float16 vbeta_2 = -8.8439941406e-02f;
  const xnn_float16 vbeta_3 = 4.8828125000e-02f;

  do {
    size_t vl = __riscv_vsetvl_e16m8(batch); batch -= vl;
    vfloat16m8_t vx = __riscv_vle16_v_f16m8(input, vl); input += vl;

    // Scale `x` with `sqrt(2)` so that the exponent is rounded up.
    vx = __riscv_vfmul_vf_f16m8(vx, vsqrt2, vl);

    // Extract the exponent.
    const vfloat16m8_t vexp = xnn_signed_getexp_f16(vx, vl);

    // Normalize `x` to an exponent of zero.
    vx = __riscv_vreinterpret_v_u16m8_f16m8(__riscv_vor_vx_u16m8(__riscv_vand_vx_u16m8(__riscv_vreinterpret_v_f16m8_u16m8(vx), vmantissa_bits_mask, vl), vone_bits, vl));

    // Scale `x` back with `1/sqrt(2)` to move its range from `[1.0, 2.0)` to
    // `[sqrt(1/2), sqrt(2))`, and further subtract `1.0` so that it is around
    // zero, i.e. `[sqrt(1/2) - 1, sqrt(2) - 1)`, or `[−0.29289, 0.4142136)`.
    vx = __riscv_vfsub_vf_f16m8(__riscv_vfmul_vf_f16m8(vx, vsqrt1_2, vl), vone, vl);

    // Evaluate the denominator polynomial q.
    vfloat16m8_t vq = __riscv_vfadd_vf_f16m8(__riscv_vfmul_vf_f16m8(vx, vbeta_3, vl), vbeta_2, vl);
    vq = __riscv_vfadd_vv_f16m8(__riscv_vfmul_vv_f16m8(vx, vq, vl), __riscv_vfmv_v_f_f16m8(vbeta_1, vl), vl);
    vq = __riscv_vfadd_vv_f16m8(__riscv_vfmul_vv_f16m8(vx, vq, vl), __riscv_vfmv_v_f_f16m8(vone, vl), vl);

    // Divide the numerator by the denominator.
    // RVV does not have a high-precision reciprocal estimate instruction for f16,
    // so we use vfdiv for both DIV and NR variants.
    vfloat16m8_t vy = __riscv_vfdiv_vv_f16m8(vx, vq, vl);

    // Put it all together, i.e. `log(x) = `log(2)*exp + y`.
    vy = __riscv_vfmacc_vf_f16m8(vy, vln2, vexp, vl);

    __riscv_vse16_v_f16m8(output, vy, vl); output += vl;

  } while (batch > 0);
}
