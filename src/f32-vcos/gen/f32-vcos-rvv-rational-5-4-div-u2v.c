// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vsin/rvv-rational-5-4.c.in
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

static XNN_INLINE vfloat32m2_t xnn_round_f32(vfloat32m2_t a, size_t vl) {
  // preserve NaN
  vbool16_t nan_bv = __riscv_vmfeq(a, a, vl);
  // magnitude < (1 << FLT_MANT_DIG)
  vfloat32m2_t mag = __riscv_vfabs(a, vl);
  vbool16_t mag_bv = __riscv_vmflt(mag, (1 << __FLT_MANT_DIG__), vl);
  vbool16_t mask_bv = __riscv_vmnand(nan_bv, mag_bv, vl);

  vint32m2_t a_rnd = __riscv_vfcvt_x(a, __RISCV_FRM_RNE, vl);
  vfloat32m2_t result = __riscv_vfcvt_f(a_rnd, vl);
  result = __riscv_vmerge(result, a, mask_bv, vl);
  return result;
}


void xnn_f32_vcos_ukernel__rvv_rational_5_4_div_u2v(
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

  // Some mathematical constants. We don't use pre-defined macros to ensure
  // that they are rounded exactly as we expect them to be.
  const float vpi = 3.1415927f;  // M_PI
  const float v2pi_inv = 0.15915494f; // 0.5 / M_PI
  const float vpi_half = 1.5707964f;  // M_PI / 2

  // The following two values sum to 2*Pi with ~33 bits of accuracy. We use
  // them to accurately subtract integer multiples of 2*Pi from large inputs.
  const float v2pi_hi = 6.28125f;  // 2.0 * M_PI (first 11 bits of mantissa)
  const float v2pi_lo = 1.9353072e-3;  // 2.0 * M_PI (remaining bits)

  // The monomial coefficients of the numerator polynomial (odd,
  // `valpha_1` = `vone`).
  const float valpha_3 = -1.3314664364e-01f;
  const float valpha_5 = 3.2340581529e-03f;

  // The monomial coefficients of the denominator polynomial (even,
  // `vbeta_0` = `vone`).
  const float vbeta_2 = 3.3519912511e-02f;
  const float vbeta_4 = 4.8770775902e-04f;

  // Some useful constants.
  const float vone = 1.0f;

  do {
    size_t vl = __riscv_vsetvl_e32m2(batch); batch -= vl;
    vfloat32m2_t vx = __riscv_vle32_v_f32m2(input, vl); input += vl;

    // Map the inputs to the interpolation range.
    vfloat32m2_t vx_div_2pi = __riscv_vfmul(vx, v2pi_inv, vl);
    vx_div_2pi = xnn_round_f32(vx_div_2pi, vl);
    vx = __riscv_vfnmsac(vx, v2pi_hi, vx_div_2pi, vl);
    vx = __riscv_vfnmsac(vx, v2pi_lo, vx_div_2pi, vl);
    vx = __riscv_vfrsub(vx, vpi_half, vl);
    vx = __riscv_vfmin(vx, __riscv_vfrsub(vx, vpi, vl), vl);
    vx = __riscv_vfmax(vx, __riscv_vfrsub(vx, -vpi, vl), vl);
    vx = __riscv_vfmin(vx, __riscv_vfrsub(vx, vpi, vl), vl);

    // Since the polynomials are odd/even, we need x^2.
    const vfloat32m2_t vx2 = __riscv_vfmul(vx, vx, vl);

    // Evaluate the numerator polynomial p.
    vfloat32m2_t vp = __riscv_vfadd(__riscv_vfmul(vx2, valpha_5, vl), valpha_3, vl);
    vp = __riscv_vfadd(__riscv_vfmul(vx2, vp, vl), vone, vl);
    vp = __riscv_vfmul(vx, vp, vl);

    // Evaluate the denominator polynomial q.
    vfloat32m2_t vq = __riscv_vfadd(__riscv_vfmul(vx2, vbeta_4, vl), vbeta_2, vl);
    vq = __riscv_vfadd(__riscv_vfmul(vx2, vq, vl), vone, vl);

    // Divide the numerator by the denominator.
    vfloat32m2_t vy = __riscv_vfdiv(vp, vq, vl);

    __riscv_vse32(output, vy, vl); output += vl;

  } while (batch != 0);
}
