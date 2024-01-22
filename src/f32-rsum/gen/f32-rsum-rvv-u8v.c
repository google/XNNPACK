// Auto-generated file. Do not edit!
//   Template: src/f32-rsum/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/reduce.h>
#include <riscv_vector.h>

void xnn_f32_rsum_ukernel__rvv_u8v(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_scale_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT;
  int32_t n = __riscv_vsetvl_e32m8(batch);
  vfloat32m1_t acc_f32v = __riscv_vfmv_s_f_f32m1(0.f, n);
  do {
    n = __riscv_vsetvl_e32m8(batch);

    vfloat32m8_t in_f32v = __riscv_vle32_v_f32m8(input, n); input += n;
    acc_f32v = __riscv_vfredosum_vs_f32m8_f32m1(in_f32v, acc_f32v, n);
    batch -= n;
  } while (batch != 0);
  vfloat32m1_t out_f32v = __riscv_vfmul_vf_f32m1(acc_f32v, params->scalar.scale, 1);
  __riscv_vse32_v_f32m1(output, out_f32v, 1);
}
