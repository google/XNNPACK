// Copyright 2024 Imagination Technologies, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"
#include <riscv_vector.h>

void xnn_f32_rsum_ukernel__rvv_u1v(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT;
  vfloat32m1_t acc_f32v = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvl_e32m1(batch));
  size_t n = __riscv_vsetvl_e32m1(batch);
  vfloat32m1_t sum0_f32v = __riscv_vfmv_v_f_f32m1(0.f, n);
  vfloat32m1_t sum1_f32v = __riscv_vfmv_v_f_f32m1(0.f, n);
  for (; batch >= n * 8; batch -= n * 8) {
    vfloat32m1_t in0_f32v = __riscv_vle32_v_f32m1(input, n); input += n;
    vfloat32m1_t in1_f32v = __riscv_vle32_v_f32m1(input, n); input += n;
    vfloat32m1_t in2_f32v = __riscv_vle32_v_f32m1(input, n); input += n;
    vfloat32m1_t in3_f32v = __riscv_vle32_v_f32m1(input, n); input += n;
    vfloat32m1_t in4_f32v = __riscv_vle32_v_f32m1(input, n); input += n;
    vfloat32m1_t in5_f32v = __riscv_vle32_v_f32m1(input, n); input += n;
    vfloat32m1_t in6_f32v = __riscv_vle32_v_f32m1(input, n); input += n;
    vfloat32m1_t in7_f32v = __riscv_vle32_v_f32m1(input, n); input += n;
    vfloat32m1_t sum01_f32v = __riscv_vfadd_vv_f32m1(in0_f32v, in1_f32v, n);
    vfloat32m1_t sum23_f32v = __riscv_vfadd_vv_f32m1(in2_f32v, in3_f32v, n);
    vfloat32m1_t sum45_f32v = __riscv_vfadd_vv_f32m1(in4_f32v, in5_f32v, n);
    vfloat32m1_t sum67_f32v = __riscv_vfadd_vv_f32m1(in6_f32v, in7_f32v, n);
    vfloat32m1_t sum0123_f32v = __riscv_vfadd_vv_f32m1(sum01_f32v, sum23_f32v, n);
    vfloat32m1_t sum4567_f32v = __riscv_vfadd_vv_f32m1(sum45_f32v, sum67_f32v, n);
    sum0_f32v = __riscv_vfadd_vv_f32m1(sum0_f32v, sum0123_f32v, n);
    sum1_f32v = __riscv_vfadd_vv_f32m1(sum1_f32v, sum4567_f32v, n);
  }
  for (; batch >= n * 4; batch -= n * 4) {
    vfloat32m1_t in0_f32v = __riscv_vle32_v_f32m1(input, n); input += n;
    vfloat32m1_t in1_f32v = __riscv_vle32_v_f32m1(input, n); input += n;
    vfloat32m1_t in2_f32v = __riscv_vle32_v_f32m1(input, n); input += n;
    vfloat32m1_t in3_f32v = __riscv_vle32_v_f32m1(input, n); input += n;
    vfloat32m1_t sum01_f32v = __riscv_vfadd_vv_f32m1(in0_f32v, in1_f32v, n);
    vfloat32m1_t sum23_f32v = __riscv_vfadd_vv_f32m1(in2_f32v, in3_f32v, n);
    sum0_f32v = __riscv_vfadd_vv_f32m1(sum0_f32v, sum01_f32v, n);
    sum1_f32v = __riscv_vfadd_vv_f32m1(sum1_f32v, sum23_f32v, n);
  }
  vfloat32m1_t sum_f32v = __riscv_vfadd_vv_f32m1(sum0_f32v, sum1_f32v, n);
  for (; batch > 0;) {
    size_t n1 = __riscv_vsetvl_e32m1(batch);
    vfloat32m1_t in_f32v = __riscv_vle32_v_f32m1(input, n1); input += n1;
    sum_f32v = __riscv_vfadd_vv_f32m1_tu(sum_f32v, sum_f32v, in_f32v, n1);
    batch -= n1;
  }
  acc_f32v = __riscv_vfredosum_vs_f32m1_f32m1(sum_f32v, acc_f32v, n);
  vfloat32m1_t out_f32v = __riscv_vfmul_vf_f32m1(acc_f32v, params->scalar.scale, 1);
  out_f32v = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(out_f32v, params->scalar.min, 1), params->scalar.max, 1);
  *output += __riscv_vfmv_f_s_f32m1_f32(out_f32v);
}
