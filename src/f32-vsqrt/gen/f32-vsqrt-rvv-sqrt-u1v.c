// Auto-generated file. Do not edit!
//   Template: src/f32-vsqrt/rvv-sqrt.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <riscv_vector.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vunary.h"


void xnn_f32_vsqrt_ukernel__rvv_sqrt_u1v(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT;
  do {
    const size_t n = __riscv_vsetvl_e32m1(batch);
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(input, n);
    input += n;
    vfloat32m1_t vacc = __riscv_vfsqrt_v_f32m1(vx, n);
    __riscv_vse32_v_f32m1(output, vacc, n);
    output += n;

    batch -= n;
  } while (batch != 0);
}
