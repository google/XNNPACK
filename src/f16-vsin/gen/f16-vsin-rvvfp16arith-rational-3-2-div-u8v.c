// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vsin/rvv-rational-3-2.c.in
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
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"

static XNN_INLINE vfloat16m8_t xnn_round_f16(vfloat16m8_t a, size_t vl) {
  // preserve NaN
  vbool2_t nan_bv = __riscv_vmfeq(a, a, vl);
  // magnitude < (1 << FLT16_MANT_DIG)
  vfloat16m8_t mag = __riscv_vfabs(a, vl);
  vbool2_t mag_bv = __riscv_vmflt(mag, (1 << __FLT16_MANT_DIG__), vl);
  vbool2_t mask_bv = __riscv_vmnand(nan_bv, mag_bv, vl);

  vint16m8_t a_rnd = __riscv_vfcvt_x(a, __RISCV_FRM_RNE, vl);
  vfloat16m8_t result = __riscv_vfcvt_f(a_rnd, vl);
  result = __riscv_vmerge(result, a, mask_bv, vl);
  return result;
}


void xnn_f16_vsin_ukernel__rvvfp16arith_rational_3_2_div_u8v(
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

  // Some mathematical constants. We don't use pre-defined macros to ensure
  // that they are rounded exactly as we expect them to be.
  const xnn_float16 vpi = 3.140625f;  // M_PI
  const xnn_float16 v2pi_inv = 0.15917969f; // 0.5 / M_PI

  // The following two values sum to 2*Pi with ~33 bits of accuracy. We use
  // them to accurately subtract integer multiples of 2*Pi from large inputs.
  const xnn_float16 v2pi_hi = 6.25f;  // 2.0 * M_PI (first 5 bits of mantissa)
  const xnn_float16 v2pi_lo = 3.3172607e-2f;  // 2.0 * M_PI (remaining bits)

  // The monomial coefficients of the numerator polynomial (odd,
  // `valpha_1` = `vone`).
  const xnn_float16 valpha_3 = -1.1200523376e-01f;

  // The monomial coefficients of the denominator polynomial (even,
  // `vbeta_0` = `vone`).
  const xnn_float16 vbeta_2 = 5.5543992668e-02f;

  // Some useful constants.
  const xnn_float16 vone = 1.0f;

  do {
    size_t vl = __riscv_vsetvl_e16m8(batch); batch -= vl;
    vfloat16m8_t vx = __riscv_vle16_v_f16m8(input, vl); input += vl;

    // Map the inputs to the interpolation range.
    vfloat16m8_t vx_div_2pi = __riscv_vfmul(vx, v2pi_inv, vl);
    vx_div_2pi = xnn_round_f16(vx_div_2pi, vl);
    vx = __riscv_vfnmsac(vx, v2pi_hi, vx_div_2pi, vl);
    vx = __riscv_vfnmsac(vx, v2pi_lo, vx_div_2pi, vl);
    vx = __riscv_vfmin(vx, __riscv_vfrsub(vx, vpi, vl), vl);
    vx = __riscv_vfmax(vx, __riscv_vfrsub(vx, -vpi, vl), vl);
    vx = __riscv_vfmin(vx, __riscv_vfrsub(vx, vpi, vl), vl);

    // Since the polynomials are odd/even, we need x^2.
    const vfloat16m8_t vx2 = __riscv_vfmul(vx, vx, vl);

    // Evaluate the numerator polynomial p.
    vfloat16m8_t vp = __riscv_vfadd(__riscv_vfmul(vx2, valpha_3, vl), vone, vl);
    vp = __riscv_vfmul(vx, vp, vl);

    // Evaluate the denominator polynomial q.
    vfloat16m8_t vq = __riscv_vfadd(__riscv_vfmul(vx2, vbeta_2, vl), vone, vl);

    // Divide the numerator by the denominator.
    const vfloat16m8_t vy = __riscv_vfdiv(vp, vq, vl);

    __riscv_vse16(output, vy, vl); output += vl;

  } while (batch > 0);
}
