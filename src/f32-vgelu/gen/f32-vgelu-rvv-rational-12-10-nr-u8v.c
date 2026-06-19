// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vgelu/rvv-rational-12-10.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/vunary.h"


void xnn_f32_vgelu_ukernel__rvv_rational_12_10_nr_u8v(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Cap the inputs to this value as `erf(x/sqrt(2))` will always be `+/-1.0f`
  // beyond this point. This value is chosen as the first floating point
  // number as of which the interpolation returns +/-1.0f.
  const float vmax_abs_x = 5.1164608002e+00f;

  // The monomial coefficients of the numerator polynomial (odd).
  const float valpha_1 = 7.9788452387e-01f;
  const float valpha_3 = 6.6972173750e-02f;
  const float valpha_5 = 9.3065137044e-03f;
  const float valpha_7 = 3.2973114867e-04f;
  const float valpha_9 = 1.2609783880e-05f;
  const float valpha_11 = 4.5835321316e-08f;

  // The monomial coefficients of the denominator polynomial (even).
  const float vbeta_2 = 2.5060352683e-01f;
  const float vbeta_4 = 2.8431978077e-02f;
  const float vbeta_6 = 1.8622842617e-03f;
  const float vbeta_8 = 7.2267655923e-05f;
  const float vbeta_10 = 1.1988805682e-06f;

  batch >>= XNN_LOG2_SIZEOF_FLOAT;
  do {
    const size_t n = __riscv_vsetvl_e32m8(batch);

    vfloat32m8_t vx_orig = __riscv_vle32_v_f32m8(input, n);
    input += n;

    // Clamp the inputs to the interpolation range.
    vfloat32m8_t vx = __riscv_vfmin_vf_f32m8(vx_orig, vmax_abs_x, n);
    vx = __riscv_vfmax_vf_f32m8(vx, -vmax_abs_x, n);

    // Since the polynomials are odd/even, we need x^2.
    vfloat32m8_t vx2 = __riscv_vfmul_vv_f32m8(vx, vx, n);

    // Evaluate the numerator polynomial p.
    vfloat32m8_t vp = __riscv_vfmv_v_f_f32m8(valpha_9, n);
    vp = __riscv_vfmacc_vf_f32m8(vp, valpha_11, vx2, n);
    vp = __riscv_vfmadd_vv_f32m8(vp, vx2, __riscv_vfmv_v_f_f32m8(valpha_7, n), n);
    vp = __riscv_vfmadd_vv_f32m8(vp, vx2, __riscv_vfmv_v_f_f32m8(valpha_5, n), n);
    vp = __riscv_vfmadd_vv_f32m8(vp, vx2, __riscv_vfmv_v_f_f32m8(valpha_3, n), n);
    vp = __riscv_vfmadd_vv_f32m8(vp, vx2, __riscv_vfmv_v_f_f32m8(valpha_1, n), n);
    vp = __riscv_vfmul_vv_f32m8(vp, vx, n);

    // Evaluate the denominator polynomial q.
    vfloat32m8_t vq = __riscv_vfmv_v_f_f32m8(vbeta_8, n);
    vq = __riscv_vfmacc_vf_f32m8(vq, vbeta_10, vx2, n);
    vq = __riscv_vfmadd_vv_f32m8(vq, vx2, __riscv_vfmv_v_f_f32m8(vbeta_6, n), n);
    vq = __riscv_vfmadd_vv_f32m8(vq, vx2, __riscv_vfmv_v_f_f32m8(vbeta_4, n), n);
    vq = __riscv_vfmadd_vv_f32m8(vq, vx2, __riscv_vfmv_v_f_f32m8(vbeta_2, n), n);
    vq = __riscv_vfmadd_vv_f32m8(vq, vx2, __riscv_vfmv_v_f_f32m8(1.0f, n), n);

    // Divide the numerator by the denominator.
    // Newton-Raphson iteration for reciprocal.
    vfloat32m8_t vrq = __riscv_vfrec7_v_f32m8(vq, n);
    for (size_t iter = 0; iter < 2; iter++) {
      vfloat32m8_t verr = __riscv_vfnmsac_vv_f32m8(
          __riscv_vfmv_v_f_f32m8(2.0f, n), vrq, vq, n);
      vrq = __riscv_vfmul_vv_f32m8(vrq, verr, n);
    }
    vfloat32m8_t verf = __riscv_vfmul_vv_f32m8(vp, vrq, n);

    // Add one to the rational interpolant, and multiply by 0.5 times the
    // original input.
    vfloat32m8_t vy = __riscv_vfadd_vf_f32m8(verf, 1.0f, n);
    vy = __riscv_vfmul_vf_f32m8(vy, 0.5f, n);
    vy = __riscv_vfmul_vv_f32m8(vy, vx_orig, n);

    __riscv_vse32_v_f32m8(output, vy, n);
    output += n;

    batch -= n;
  } while (batch != 0);
}
