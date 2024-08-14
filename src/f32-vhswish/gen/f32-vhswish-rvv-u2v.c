// Auto-generated file. Do not edit!
//   Template: src/f32-vhswish/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vunary.h"


void xnn_f32_vhswish_ukernel__rvv_u2v(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_hswish_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vsixth = 0x1.555556p-3f;
  const float vthree = 3.0f;
  const float vsix = 6.0f;
  const float vzero = 0.0f;

  batch >>= XNN_LOG2_SIZEOF_FLOAT;
  do {
    const size_t n = __riscv_vsetvl_e32m2(batch);
    vfloat32m2_t vx = __riscv_vle32_v_f32m2(input, n);
    input += n;
    vfloat32m2_t vacc = __riscv_vfadd_vf_f32m2(vx, vthree, n);
    vx = __riscv_vfmul_vf_f32m2(vx, vsixth, n);
    vacc = __riscv_vfmax_vf_f32m2(vacc, vzero, n);
    vacc = __riscv_vfmin_vf_f32m2(vacc, vsix, n);
    vacc = __riscv_vfmul_vv_f32m2(vacc, vx, n);
    __riscv_vse32_v_f32m2(output, vacc, n);
    output += n;

    batch -= n;
  } while (batch != 0);
}
