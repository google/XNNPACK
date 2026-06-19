// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vtanh/rvv-rational-9-8.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f32_vtanh_ukernel__rvv_rational_9_8_div_u1v(
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

  // Cap the inputs to this value as `tanh(x)` will always be `+/-1.0f` beyond
  // this point. This value is chosen as the first floating point number as of
  // which the interpolation returns 1.0f.
  const float vmax_x = 7.8522667885e+00f;
  const float vmin_x = -7.8522667885e+00f;

  // The monomial coefficients of the numerator polynomial (odd).
  // const float valpha_1 = 1.0000000000e+00f;
  const float valpha_3 = 1.3412411511e-01f;
  const float valpha_5 = 3.5330520477e-03f;
  const float valpha_7 = 2.1235626264e-05f;
  const float valpha_9 = 1.4248920266e-08f;

  // The monomial coefficients of the denominator polynomial (even).
  // const float vbeta_0 = 1.0000000000e+00f;
  const float vbeta_2 = 4.6745735407e-01f;
  const float vbeta_4 = 2.6018999517e-02f;
  const float vbeta_6 = 3.3472978976e-04f;
  const float vbeta_8 = 8.1365948290e-07f;

  // Some useful constants.
  const float vone = 1.0f;

  do {
    size_t vl = __riscv_vsetvl_e32m1(batch); batch -= vl;
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(input, vl); input += vl;

    // Preserve NaN
    vbool32_t nan_mask = __riscv_vmfne(vx, vx, vl);

    // Clamp the inputs to the interpolation range.
    vx = __riscv_vfmin(vx, vmax_x, vl);
    vx = __riscv_vfmax(vx, vmin_x, vl);

    // Since the polynomials are odd/even, we need x^2.
    const vfloat32m1_t vx2 = __riscv_vfmul(vx, vx, vl);

    // Evaluate the numerator polynomial p.
    vfloat32m1_t vp = __riscv_vfadd(__riscv_vfmul(vx2, valpha_9, vl), valpha_7, vl);
    vp = __riscv_vfadd(__riscv_vfmul(vx2, vp, vl), valpha_5, vl);
    vp = __riscv_vfadd(__riscv_vfmul(vx2, vp, vl), valpha_3, vl);
    vp = __riscv_vfadd(__riscv_vfmul(vx2, vp, vl), vone, vl);
    vp = __riscv_vfmul(vx, vp, vl);

    // Evaluate the denominator polynomial q.
    vfloat32m1_t vq = __riscv_vfadd(__riscv_vfmul(vx2, vbeta_8, vl), vbeta_6, vl);
    vq = __riscv_vfadd(__riscv_vfmul(vx2, vq, vl), vbeta_4, vl);
    vq = __riscv_vfadd(__riscv_vfmul(vx2, vq, vl), vbeta_2, vl);
    vq = __riscv_vfadd(__riscv_vfmul(vx2, vq, vl), vone, vl);

    // Divide the numerator by the denominator.
    vfloat32m1_t vy = __riscv_vfdiv(vp, vq, vl);

    // Propogate NaN
    vy = __riscv_vfmerge(vy, NAN, nan_mask, vl);

    __riscv_vse32(output, vy, vl); output += vl;

  } while (batch > 0);
}
