// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vapproxgelu/rvv-rational-12-10.c.in
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


void xnn_f32_vapproxgelu_ukernel__rvv_rational_12_10_nr_u2v(
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
  const float vmax_x = 4.79519796e+00f;
  const float vmin_x = -4.84563780e+00f;

  // The monomial coefficients of the numerator polynomial (odd).
  const float valpha_1 = 7.9788458347e-01f;
  const float valpha_3 = 6.0803253204e-02f;
  const float valpha_5 = 7.2898347862e-03f;
  const float valpha_7 = 2.6887017884e-04f;
  const float valpha_9 = 1.4302649106e-05f;
  const float valpha_11 = 4.9544411240e-08f;

  // The monomial coefficients of the denominator polynomial (even).
  const float vbeta_2 = 2.4369759858e-01f;
  const float vbeta_4 = 2.4381054565e-02f;
  const float vbeta_6 = 1.3060354395e-03f;
  const float vbeta_8 = 7.6477612311e-05f;
  const float vbeta_10 = 1.3433452750e-06f;

  batch >>= XNN_LOG2_SIZEOF_FLOAT;
  do {
    const size_t n = __riscv_vsetvl_e32m2(batch);

    vfloat32m2_t vx_orig = __riscv_vle32_v_f32m2(input, n);
    input += n;

    // Clamp the inputs to the interpolation range.
    vfloat32m2_t vx = __riscv_vfmin_vf_f32m2(vx_orig, vmax_x, n);
    vx = __riscv_vfmax_vf_f32m2(vx, vmin_x, n);

    // Since the polynomials are odd/even, we need x^2.
    vfloat32m2_t vx2 = __riscv_vfmul_vv_f32m2(vx, vx, n);

    // Evaluate the numerator polynomial p.
    vfloat32m2_t vp = __riscv_vfmv_v_f_f32m2(valpha_9, n);
    vp = __riscv_vfmacc_vf_f32m2(vp, valpha_11, vx2, n);
    vp = __riscv_vfmadd_vv_f32m2(vp, vx2, __riscv_vfmv_v_f_f32m2(valpha_7, n), n);
    vp = __riscv_vfmadd_vv_f32m2(vp, vx2, __riscv_vfmv_v_f_f32m2(valpha_5, n), n);
    vp = __riscv_vfmadd_vv_f32m2(vp, vx2, __riscv_vfmv_v_f_f32m2(valpha_3, n), n);
    vp = __riscv_vfmadd_vv_f32m2(vp, vx2, __riscv_vfmv_v_f_f32m2(valpha_1, n), n);
    vp = __riscv_vfmul_vv_f32m2(vp, vx, n);

    // Evaluate the denominator polynomial q.
    vfloat32m2_t vq = __riscv_vfmv_v_f_f32m2(vbeta_8, n);
    vq = __riscv_vfmacc_vf_f32m2(vq, vbeta_10, vx2, n);
    vq = __riscv_vfmadd_vv_f32m2(vq, vx2, __riscv_vfmv_v_f_f32m2(vbeta_6, n), n);
    vq = __riscv_vfmadd_vv_f32m2(vq, vx2, __riscv_vfmv_v_f_f32m2(vbeta_4, n), n);
    vq = __riscv_vfmadd_vv_f32m2(vq, vx2, __riscv_vfmv_v_f_f32m2(vbeta_2, n), n);
    vq = __riscv_vfmadd_vv_f32m2(vq, vx2, __riscv_vfmv_v_f_f32m2(1.0f, n), n);

    // Divide the numerator by the denominator.
    // Newton-Raphson iteration for reciprocal.
    vfloat32m2_t vrq = __riscv_vfrec7_v_f32m2(vq, n);
    for (size_t iter = 0; iter < 2; iter++) {
      vfloat32m2_t verr = __riscv_vfnmsac_vv_f32m2(
          __riscv_vfmv_v_f_f32m2(2.0f, n), vrq, vq, n);
      vrq = __riscv_vfmul_vv_f32m2(vrq, verr, n);
    }
    vfloat32m2_t verf = __riscv_vfmul_vv_f32m2(vp, vrq, n);

    // Add one to the rational interpolant, and multiply by 0.5 times the
    // original input.
    vfloat32m2_t vy = __riscv_vfadd_vf_f32m2(verf, 1.0f, n);
    vy = __riscv_vfmul_vf_f32m2(vy, 0.5f, n);
    vy = __riscv_vfmul_vv_f32m2(vy, vx_orig, n);

    __riscv_vse32_v_f32m2(output, vy, n);
    output += n;

    batch -= n;
  } while (batch != 0);
}
