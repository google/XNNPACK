// Auto-generated file. Do not edit!
//   Template: src/f32-vrelu/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies, inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/vunary.h"
#include "xnnpack/common.h"
#include <riscv_vector.h>

void xnn_f32_vrelu_ukernel__rvv_u2v(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  float zero = 0.0f;
  size_t batch_ = batch >> XNN_LOG2_SIZEOF_FLOAT;

  for (; batch_ > 0; ) {
    size_t n = __riscv_vsetvl_e32m2(batch_); batch_ -= n;
    vfloat32m2_t in_f32v = __riscv_vle32_v_f32m2(input, n); input += n;
    vfloat32m2_t out_f32v = __riscv_vfmax_vf_f32m2(in_f32v, zero, n);
    __riscv_vse32_v_f32m2(output, out_f32v, n); output += n;
  }
}
