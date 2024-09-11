// Auto-generated file. Do not edit!
//   Template: src/f32-rdsum/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies, inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"


void xnn_f32_rdsum_ukernel_7p7x__rvv_u1v(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* output,
    const struct xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  const float scale = params->scalar.scale;
  const float min = params->scalar.min;
  const float max = params->scalar.max;

  size_t input_increment = 7 * input_stride;
  for (; channels > 0; ) {
    size_t n = __riscv_vsetvl_e32m1(channels); channels -= n;
    const float* i0 = input;
    const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
    const float* i2 = (const float*) ((uintptr_t) i1 + input_stride);
    const float* i3 = (const float*) ((uintptr_t) i2 + input_stride);
    const float* i4 = (const float*) ((uintptr_t) i3 + input_stride);
    const float* i5 = (const float*) ((uintptr_t) i4 + input_stride);
    const float* i6 = (const float*) ((uintptr_t) i5 + input_stride);
    vfloat32m1_t acc_f32v = __riscv_vfmv_v_f_f32m1(0.f, n);

    for (int r = rows; r > 0; r -= 7) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 2) {
        i2 = zero;
      }
      if XNN_UNPREDICTABLE(r < 4) {
        i3 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 4) {
        i4 = zero;
      }
      if XNN_UNPREDICTABLE(r < 6) {
        i5 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 6) {
        i6 = zero;
      }

      vfloat32m1_t in0_f32v = __riscv_vle32_v_f32m1(i0, n);
      acc_f32v = __riscv_vfadd_vv_f32m1(acc_f32v, in0_f32v, n);
      vfloat32m1_t in1_f32v = __riscv_vle32_v_f32m1(i1, n);
      acc_f32v = __riscv_vfadd_vv_f32m1(acc_f32v, in1_f32v, n);
      vfloat32m1_t in2_f32v = __riscv_vle32_v_f32m1(i2, n);
      acc_f32v = __riscv_vfadd_vv_f32m1(acc_f32v, in2_f32v, n);
      vfloat32m1_t in3_f32v = __riscv_vle32_v_f32m1(i3, n);
      acc_f32v = __riscv_vfadd_vv_f32m1(acc_f32v, in3_f32v, n);
      vfloat32m1_t in4_f32v = __riscv_vle32_v_f32m1(i4, n);
      acc_f32v = __riscv_vfadd_vv_f32m1(acc_f32v, in4_f32v, n);
      vfloat32m1_t in5_f32v = __riscv_vle32_v_f32m1(i5, n);
      acc_f32v = __riscv_vfadd_vv_f32m1(acc_f32v, in5_f32v, n);
      vfloat32m1_t in6_f32v = __riscv_vle32_v_f32m1(i6, n);
      acc_f32v = __riscv_vfadd_vv_f32m1(acc_f32v, in6_f32v, n);

      i0 = (const float*) ((uintptr_t) i0 + input_increment);
      i1 = (const float*) ((uintptr_t) i1 + input_increment);
      i2 = (const float*) ((uintptr_t) i2 + input_increment);
      i3 = (const float*) ((uintptr_t) i3 + input_increment);
      i4 = (const float*) ((uintptr_t) i4 + input_increment);
      i5 = (const float*) ((uintptr_t) i5 + input_increment);
      i6 = (const float*) ((uintptr_t) i6 + input_increment);
    }
    acc_f32v = __riscv_vfmul_vf_f32m1(acc_f32v, scale, n);
    acc_f32v = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(acc_f32v, min, n), max, n);
    vfloat32m1_t out_f32v = __riscv_vle32_v_f32m1(output, n);
    out_f32v = __riscv_vfadd_vv_f32m1(out_f32v, acc_f32v, n);
    __riscv_vse32_v_f32m1(output, out_f32v, n); output += n;

    input = (const float*) ((uintptr_t) input + n * sizeof(float));
  }
}
