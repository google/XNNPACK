// Auto-generated file. Do not edit!
//   Template: src/f32-rminmax/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2023 SiFive, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <riscv_vector.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vunary.h"


void xnn_f32_rmax_ukernel__rvv_u2v(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t N = batch >> 2;
  size_t avl;
  size_t vl = __riscv_vsetvl_e32m2(N);

  vfloat32m2_t t0 = __riscv_vle32_v_f32m2(input, vl);
  input += vl;

  for (avl = N - vl; avl; avl -= vl, input += vl) {
    vl = __riscv_vsetvl_e32m2(avl);
    vfloat32m2_t vec = __riscv_vle32_v_f32m2(input, vl);
    t0 = __riscv_vfmax_vv_f32m2_tu(t0, t0, vec, vl);
  }

  vfloat32m1_t fmax = __riscv_vfmv_s_f_f32m1(-INFINITY, 1);
  output[0] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m2_f32m1(t0, fmax, N));
}
