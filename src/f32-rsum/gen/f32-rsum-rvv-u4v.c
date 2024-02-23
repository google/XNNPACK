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

void xnn_f32_rsum_ukernel__rvv_u4v(
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
  int32_t n = __riscv_vsetvl_e32m1(batch);
  vfloat32m1_t acc_f32v = __riscv_vfmv_s_f_f32m1(0.f, n);
  n = __riscv_vsetvl_e32m4(batch);
  int32_t n_sum = __riscv_vsetvl_e32m4(batch);
  vfloat32m4_t sum_f32v = __riscv_vfmv_s_f_f32m4(0.f, n_sum);
  if (batch >= n * 4) {
    for (; batch >= n * 4; batch -= n * 4) {
      vfloat32m4_t in0_f32v = __riscv_vle32_v_f32m4(input, n); input += n;
      vfloat32m4_t in1_f32v = __riscv_vle32_v_f32m4(input, n); input += n;
      vfloat32m4_t in2_f32v = __riscv_vle32_v_f32m4(input, n); input += n;
      vfloat32m4_t in3_f32v = __riscv_vle32_v_f32m4(input, n); input += n;
      vfloat32m4_t sum0_f32v = __riscv_vfadd_vv_f32m4(in0_f32v, in1_f32v, n);
      vfloat32m4_t sum1_f32v = __riscv_vfadd_vv_f32m4(in2_f32v, in3_f32v, n);
      sum_f32v = __riscv_vfadd_vv_f32m4(sum_f32v, sum0_f32v, n);
      sum_f32v = __riscv_vfadd_vv_f32m4(sum_f32v, sum1_f32v, n);
    }
  }
  if (batch > 0) {
    for (; batch > 0; batch -= n) {
      n = __riscv_vsetvl_e32m4(batch);
      vfloat32m4_t in_f32v = __riscv_vle32_v_f32m4(input, n); input += n;
      sum_f32v = __riscv_vfadd_vv_f32m4(sum_f32v, in_f32v, n);
    }
  }
  vfloat32m4_t out_f32v = __riscv_vfmul_vf_f32m4(sum_f32v, params->scalar.scale, n_sum);
  acc_f32v = __riscv_vfredosum_vs_f32m4_f32m1(out_f32v, acc_f32v, n_sum);
  __riscv_vse32_v_f32m1(output, acc_f32v, 1);
}
