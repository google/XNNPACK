// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vtanh/rvv-rational-5-4.c.in
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


void xnn_f16_vtanh_ukernel__rvvfp16arith_rational_5_4_div_u2v(
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

  // Cap the inputs to this value as `tanh(x)` will always be `+/-1.0f` beyond
  // this point. This value is chosen as the first floating point number as of
  // which the interpolation returns 1.0f.
  const xnn_float16 vmax_x = 4.5f;
  const xnn_float16 vmin_x = -4.5f;

  // The monomial coefficients of the numerator polynomial (odd).
  // const xnn_float16 valpha_1 = 1.0000000000e+00f;
  const xnn_float16 valpha_3 = 1.0632324219e-01f;
  const xnn_float16 valpha_5 = 8.1062316895e-04f;

  // The monomial coefficients of the denominator polynomial (even).
  // const xnn_float16 vbeta_0 = 1.0000000000e+00f;
  const xnn_float16 vbeta_2 = 4.3945312500e-01f;
  const xnn_float16 vbeta_4 = 1.4091491699e-02f;

  // Some useful constants.
  const xnn_float16 vone = 1.0f;

  do {
    size_t vl = __riscv_vsetvl_e16m2(batch); batch -= vl;
    vfloat16m2_t vx = __riscv_vle16_v_f16m2(input, vl); input += vl;

    // Preserve NaN
    vbool8_t nan_mask = __riscv_vmfne(vx, vx, vl);

    // Clamp the inputs to the interpolation range.
    vx = __riscv_vfmin(vx, vmax_x, vl);
    vx = __riscv_vfmax(vx, vmin_x, vl);

    // Since the polynomials are odd/even, we need x^2.
    const vfloat16m2_t vx2 = __riscv_vfmul(vx, vx, vl);

    // Evaluate the numerator polynomial p.
    vfloat16m2_t vp = __riscv_vfadd(__riscv_vfmul(vx2, valpha_5, vl), valpha_3, vl);
    vp = __riscv_vfadd(__riscv_vfmul(vx2, vp, vl), vone, vl);
    vp = __riscv_vfmul(vx, vp, vl);

    // Evaluate the denominator polynomial q.
    vfloat16m2_t vq = __riscv_vfadd(__riscv_vfmul(vx2, vbeta_4, vl), vbeta_2, vl);
    vq = __riscv_vfadd(__riscv_vfmul(vx2, vq, vl), vone, vl);

    // Divide the numerator by the denominator.
    // RVV doesn't always have a fast NR sequence for f16, so we use vfdiv for both variants.
    vfloat16m2_t vy = __riscv_vfdiv(vp, vq, vl);

    // Propogate NaN
    vy = __riscv_vfmerge(vy, NAN, nan_mask, vl);

    __riscv_vse16(output, vy, vl); output += vl;

  } while (batch > 0);
}
