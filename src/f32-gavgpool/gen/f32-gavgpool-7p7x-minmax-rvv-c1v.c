// Auto-generated file. Do not edit!
//   Template: src/f32-gavgpool/rvv_7p7x.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies, inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include "xnnpack/gavgpool.h"
#include <riscv_vector.h>

void xnn_f32_gavgpool_minmax_ukernel_7p7x__rvv_c1v(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* buffer,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows > 7);
  assert(channels != 0);

  const float* i0 = input;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_stride);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_stride);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_stride);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_stride);
  const size_t input_increment = 7 * input_stride - channels * sizeof(float);

  float* b = buffer;
  for (size_t c = channels; c != 0; ) {
    int32_t n = __riscv_vsetvl_e32m1(c);

    vfloat32m1_t i0_f32v = __riscv_vle32_v_f32m1(i0, n); i0 += n;
    vfloat32m1_t i1_f32v = __riscv_vle32_v_f32m1(i1, n); i1 += n;
    vfloat32m1_t i2_f32v = __riscv_vle32_v_f32m1(i2, n); i2 += n;
    vfloat32m1_t i3_f32v = __riscv_vle32_v_f32m1(i3, n); i3 += n;
    vfloat32m1_t i4_f32v = __riscv_vle32_v_f32m1(i4, n); i4 += n;
    vfloat32m1_t i5_f32v = __riscv_vle32_v_f32m1(i5, n); i5 += n;
    vfloat32m1_t i6_f32v = __riscv_vle32_v_f32m1(i6, n); i6 += n;

    vfloat32m1_t sum01_f32v = __riscv_vfadd_vv_f32m1(i0_f32v, i1_f32v, n);
    vfloat32m1_t sum23_f32v = __riscv_vfadd_vv_f32m1(i2_f32v, i3_f32v, n);
    vfloat32m1_t sum45_f32v = __riscv_vfadd_vv_f32m1(i4_f32v, i5_f32v, n);
    vfloat32m1_t sum016_f32v = __riscv_vfadd_vv_f32m1(sum01_f32v, i6_f32v, n);
    vfloat32m1_t sum2345_f32v = __riscv_vfadd_vv_f32m1(sum23_f32v, sum45_f32v, n);
    vfloat32m1_t sum_f32v = __riscv_vfadd_vv_f32m1(sum2345_f32v, sum016_f32v, n);
    __riscv_vse32_v_f32m1(b, sum_f32v, n); b += n;

    c -= n;
  }

  for (rows -= 7; rows > 7; rows -= 7) {
    b = buffer;

    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    i2 = (const float*) ((uintptr_t) i2 + input_increment);
    i3 = (const float*) ((uintptr_t) i3 + input_increment);
    i4 = (const float*) ((uintptr_t) i4 + input_increment);
    i5 = (const float*) ((uintptr_t) i5 + input_increment);
    i6 = (const float*) ((uintptr_t) i6 + input_increment);

    for (size_t c = channels; c != 0; ) {
      int32_t n = __riscv_vsetvl_e32m1(c);

      vfloat32m1_t i0_f32v = __riscv_vle32_v_f32m1(i0, n); i0 += n;
      vfloat32m1_t i1_f32v = __riscv_vle32_v_f32m1(i1, n); i1 += n;
      vfloat32m1_t i2_f32v = __riscv_vle32_v_f32m1(i2, n); i2 += n;
      vfloat32m1_t i3_f32v = __riscv_vle32_v_f32m1(i3, n); i3 += n;
      vfloat32m1_t i4_f32v = __riscv_vle32_v_f32m1(i4, n); i4 += n;
      vfloat32m1_t i5_f32v = __riscv_vle32_v_f32m1(i5, n); i5 += n;
      vfloat32m1_t i6_f32v = __riscv_vle32_v_f32m1(i6, n); i6 += n;
      vfloat32m1_t vacc_f32v = __riscv_vle32_v_f32m1(b, n);

      vfloat32m1_t sum01_f32v = __riscv_vfadd_vv_f32m1(i0_f32v, i1_f32v, n);
      vfloat32m1_t sum23_f32v = __riscv_vfadd_vv_f32m1(i2_f32v, i3_f32v, n);
      vfloat32m1_t sum45_f32v = __riscv_vfadd_vv_f32m1(i4_f32v, i5_f32v, n);
      vfloat32m1_t sum6a_f32v = __riscv_vfadd_vv_f32m1(i6_f32v, vacc_f32v, n);
      vfloat32m1_t sum0123_f32v = __riscv_vfadd_vv_f32m1(sum01_f32v, sum23_f32v, n);
      vfloat32m1_t sum456a_f32v = __riscv_vfadd_vv_f32m1(sum45_f32v, sum6a_f32v, n);
      vfloat32m1_t sum_f32v = __riscv_vfadd_vv_f32m1(sum0123_f32v, sum456a_f32v, n);
      __riscv_vse32_v_f32m1(b, sum_f32v, n); b += n;

      c -= n;
    }
  }

  i0 = (const float*) ((uintptr_t) i0 + input_increment);
  i1 = (const float*) ((uintptr_t) i1 + input_increment);
  if (rows < 2) {
    i1 = zero;
  }
  i2 = (const float*) ((uintptr_t) i2 + input_increment);
  if (rows <= 2) {
    i2 = zero;
  }
  i3 = (const float*) ((uintptr_t) i3 + input_increment);
  if (rows < 4) {
    i3 = zero;
  }
  i4 = (const float*) ((uintptr_t) i4 + input_increment);
  if (rows <= 4) {
    i4 = zero;
  }
  i5 = (const float*) ((uintptr_t) i5 + input_increment);
  if (rows < 6) {
    i5 = zero;
  }
  i6 = (const float*) ((uintptr_t) i6 + input_increment);
  if (rows <= 6) {
    i6 = zero;
  }
  const float scale = params->scalar.scale;
  const float min = params->scalar.min;
  const float max = params->scalar.max;

  b = buffer;
  for (; channels != 0; ) {
    int32_t n = __riscv_vsetvl_e32m1(channels);

    vfloat32m1_t i0_f32v = __riscv_vle32_v_f32m1(i0, n); i0 += n;
    vfloat32m1_t i1_f32v = __riscv_vle32_v_f32m1(i1, n); i1 += n;
    vfloat32m1_t i2_f32v = __riscv_vle32_v_f32m1(i2, n); i2 += n;
    vfloat32m1_t i3_f32v = __riscv_vle32_v_f32m1(i3, n); i3 += n;
    vfloat32m1_t i4_f32v = __riscv_vle32_v_f32m1(i4, n); i4 += n;
    vfloat32m1_t i5_f32v = __riscv_vle32_v_f32m1(i5, n); i5 += n;
    vfloat32m1_t i6_f32v = __riscv_vle32_v_f32m1(i6, n); i6 += n;
    vfloat32m1_t vacc_f32v = __riscv_vle32_v_f32m1(b, n); b += n;

    vfloat32m1_t sum01_f32v = __riscv_vfadd_vv_f32m1(i0_f32v, i1_f32v, n);
    vfloat32m1_t sum23_f32v = __riscv_vfadd_vv_f32m1(i2_f32v, i3_f32v, n);
    vfloat32m1_t sum45_f32v = __riscv_vfadd_vv_f32m1(i4_f32v, i5_f32v, n);
    vfloat32m1_t sum6a_f32v = __riscv_vfadd_vv_f32m1(i6_f32v, vacc_f32v, n);
    vfloat32m1_t sum0123_f32v = __riscv_vfadd_vv_f32m1(sum01_f32v, sum23_f32v, n);
    vfloat32m1_t sum456a_f32v = __riscv_vfadd_vv_f32m1(sum45_f32v, sum6a_f32v, n);
    vfloat32m1_t sum_f32v = __riscv_vfadd_vv_f32m1(sum0123_f32v, sum456a_f32v, n);
    vfloat32m1_t out_f32v = __riscv_vfmul_vf_f32m1(sum_f32v, scale, n);
    out_f32v = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(out_f32v, min, n), max, n);
    __riscv_vse32_v_f32m1(output, out_f32v, n); output += n;

    channels -= n;
  }
}
