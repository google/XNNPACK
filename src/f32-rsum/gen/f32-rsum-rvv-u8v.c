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
  size_t n = __riscv_vsetvl_e32m1(batch);
  vfloat32m1_t acc_f32v = __riscv_vfmv_s_f_f32m1(0.f, n);
  n = __riscv_vsetvl_e32m8(batch);
  vfloat32m8_t sum0_f32v = __riscv_vfmv_s_f_f32m8(0.f, n);
  vfloat32m8_t sum1_f32v = __riscv_vfmv_s_f_f32m8(0.f, n);
  if (batch >= n * 2) {
    for (; batch >= n * 2; batch -= n * 2) {
      vfloat32m8_t in0_f32v = __riscv_vle32_v_f32m8(input, n); input += n;
      vfloat32m8_t in1_f32v = __riscv_vle32_v_f32m8(input, n); input += n;
      sum0_f32v = __riscv_vfadd_vv_f32m8(sum0_f32v, in0_f32v, n);
      sum1_f32v = __riscv_vfadd_vv_f32m8(sum1_f32v, in1_f32v, n);
    }
  }
  vfloat32m8_t sum_f32v = __riscv_vfadd_vv_f32m8(sum0_f32v, sum1_f32v, n);
  if (batch > 0) {
    for (; batch > 0;) {
      size_t n1 = __riscv_vsetvl_e32m8(batch);
      vfloat32m8_t in_f32v = __riscv_vle32_v_f32m8(input, n1); input += n1;
      sum_f32v = __riscv_vfadd_vv_f32m8(sum_f32v, in_f32v, n1);
      batch -= n1;
    }
  }
  vfloat32m8_t out_f32v = __riscv_vfmul_vf_f32m8(sum_f32v, params->scalar.scale, n);
  acc_f32v = __riscv_vfredosum_vs_f32m8_f32m1(out_f32v, acc_f32v, n);
  __riscv_vse32_v_f32m1(output, acc_f32v, 1);
}
