// Auto-generated file. Do not edit!
//   Template: src/f32-rsum/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>
#include "xnnpack/common.h"
#include "xnnpack/reduce.h"

void xnn_f32_rsum_ukernel__rvv_u2v(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float scale = params->scalar.scale;
  const float min = params->scalar.min;
  const float max = params->scalar.max;

  batch >>= XNN_LOG2_SIZEOF_FLOAT;
  vfloat32m1_t acc_f32v = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvl_e32m1(batch));
  size_t n = __riscv_vsetvl_e32m2(batch);
  vfloat32m2_t sum0_f32v = __riscv_vfmv_v_f_f32m2(0.f, n);
  vfloat32m2_t sum1_f32v = __riscv_vfmv_v_f_f32m2(0.f, n);
  for (; batch >= n * 8; batch -= n * 8) {
    vfloat32m2_t in0_f32v = __riscv_vle32_v_f32m2(input, n); input += n;
    vfloat32m2_t in1_f32v = __riscv_vle32_v_f32m2(input, n); input += n;
    vfloat32m2_t in2_f32v = __riscv_vle32_v_f32m2(input, n); input += n;
    vfloat32m2_t in3_f32v = __riscv_vle32_v_f32m2(input, n); input += n;
    vfloat32m2_t in4_f32v = __riscv_vle32_v_f32m2(input, n); input += n;
    vfloat32m2_t in5_f32v = __riscv_vle32_v_f32m2(input, n); input += n;
    vfloat32m2_t in6_f32v = __riscv_vle32_v_f32m2(input, n); input += n;
    vfloat32m2_t in7_f32v = __riscv_vle32_v_f32m2(input, n); input += n;
    vfloat32m2_t sum01_f32v = __riscv_vfadd_vv_f32m2(in0_f32v, in1_f32v, n);
    vfloat32m2_t sum23_f32v = __riscv_vfadd_vv_f32m2(in2_f32v, in3_f32v, n);
    vfloat32m2_t sum45_f32v = __riscv_vfadd_vv_f32m2(in4_f32v, in5_f32v, n);
    vfloat32m2_t sum67_f32v = __riscv_vfadd_vv_f32m2(in6_f32v, in7_f32v, n);
    vfloat32m2_t sum0123_f32v = __riscv_vfadd_vv_f32m2(sum01_f32v, sum23_f32v, n);
    vfloat32m2_t sum4567_f32v = __riscv_vfadd_vv_f32m2(sum45_f32v, sum67_f32v, n);
    sum0_f32v = __riscv_vfadd_vv_f32m2(sum0_f32v, sum0123_f32v, n);
    sum1_f32v = __riscv_vfadd_vv_f32m2(sum1_f32v, sum4567_f32v, n);
  }
  for (; batch >= n * 4; batch -= n * 4) {
    vfloat32m2_t in0_f32v = __riscv_vle32_v_f32m2(input, n); input += n;
    vfloat32m2_t in1_f32v = __riscv_vle32_v_f32m2(input, n); input += n;
    vfloat32m2_t in2_f32v = __riscv_vle32_v_f32m2(input, n); input += n;
    vfloat32m2_t in3_f32v = __riscv_vle32_v_f32m2(input, n); input += n;
    vfloat32m2_t sum01_f32v = __riscv_vfadd_vv_f32m2(in0_f32v, in1_f32v, n);
    vfloat32m2_t sum23_f32v = __riscv_vfadd_vv_f32m2(in2_f32v, in3_f32v, n);
    sum0_f32v = __riscv_vfadd_vv_f32m2(sum0_f32v, sum01_f32v, n);
    sum1_f32v = __riscv_vfadd_vv_f32m2(sum1_f32v, sum23_f32v, n);
  }
  vfloat32m2_t sum_f32v = __riscv_vfadd_vv_f32m2(sum0_f32v, sum1_f32v, n);
  for (; batch > 0;) {
    size_t n1 = __riscv_vsetvl_e32m2(batch);
    vfloat32m2_t in_f32v = __riscv_vle32_v_f32m2(input, n1); input += n1;
    sum_f32v = __riscv_vfadd_vv_f32m2_tu(sum_f32v, sum_f32v, in_f32v, n1);
    batch -= n1;
  }
  acc_f32v = __riscv_vfredusum_vs_f32m2_f32m1(sum_f32v, acc_f32v, n);
  vfloat32m1_t out_f32v = __riscv_vfmul_vf_f32m1(acc_f32v, scale, 1);
  out_f32v = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(out_f32v, min, n), max, n);
  *output += __riscv_vfmv_f_s_f32m1_f32(out_f32v);
}
