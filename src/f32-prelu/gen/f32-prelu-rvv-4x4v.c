// Auto-generated file. Do not edit!
//   Template: src/f32-prelu/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/math.h"
#include "xnnpack/prelu.h"

void xnn_f32_prelu_ukernel__rvv_4x4v(
    size_t rows,
    size_t channels,
    const float* restrict input,
    size_t input_stride,
    const float* restrict weights,
    float* restrict output,
    size_t output_stride) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const float* i0 = input;
  float* o0 = output;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_stride);
  float* o2 = (float*) ((uintptr_t) o1 + output_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_stride);
  float* o3 = (float*) ((uintptr_t) o2 + output_stride);

  const size_t input_increment = input_stride * 4 - channels;
  const size_t output_increment = output_stride * 4 - channels;

  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(rows <= 2) {
      i2 = i1;
      o2 = o1;
    }
    if XNN_UNPREDICTABLE(rows < 4) {
      i3 = i2;
      o3 = o2;
    }

    const float* w = weights;
    size_t c = channels >> XNN_LOG2_SIZEOF_FLOAT;

    for (; c > 0;) {
      size_t n = __riscv_vsetvl_e32m4(c); c -= n;
      vfloat32m4_t w_f32v = __riscv_vle32_v_f32m4(w, n); w += n;
      vfloat32m4_t in0_f32v = __riscv_vle32_v_f32m4(i0, n); i0 += n;
      vfloat32m4_t in1_f32v = __riscv_vle32_v_f32m4(i1, n); i1 += n;
      vfloat32m4_t in2_f32v = __riscv_vle32_v_f32m4(i2, n); i2 += n;
      vfloat32m4_t in3_f32v = __riscv_vle32_v_f32m4(i3, n); i3 += n;

      vbool8_t mask0_f32v = __riscv_vmflt_vf_f32m4_b8(in0_f32v, 0.0f, n);
      vbool8_t mask1_f32v = __riscv_vmflt_vf_f32m4_b8(in1_f32v, 0.0f, n);
      vbool8_t mask2_f32v = __riscv_vmflt_vf_f32m4_b8(in2_f32v, 0.0f, n);
      vbool8_t mask3_f32v = __riscv_vmflt_vf_f32m4_b8(in3_f32v, 0.0f, n);
      vfloat32m4_t out0_f32v = __riscv_vfmul_vv_f32m4_mu(mask0_f32v, in0_f32v, w_f32v, in0_f32v, n);
      vfloat32m4_t out1_f32v = __riscv_vfmul_vv_f32m4_mu(mask1_f32v, in1_f32v, w_f32v, in1_f32v, n);
      vfloat32m4_t out2_f32v = __riscv_vfmul_vv_f32m4_mu(mask2_f32v, in2_f32v, w_f32v, in2_f32v, n);
      vfloat32m4_t out3_f32v = __riscv_vfmul_vv_f32m4_mu(mask3_f32v, in3_f32v, w_f32v, in3_f32v, n);

      __riscv_vse32_v_f32m4(o0, out0_f32v, n); o0 += n;
      __riscv_vse32_v_f32m4(o1, out1_f32v, n); o1 += n;
      __riscv_vse32_v_f32m4(o2, out2_f32v, n); o2 += n;
      __riscv_vse32_v_f32m4(o3, out3_f32v, n); o3 += n;
    }

    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    i2 = (const float*) ((uintptr_t) i2 + input_increment);
    o2 = (float*) ((uintptr_t) o2 + output_increment);
    i3 = (const float*) ((uintptr_t) i3 + input_increment);
    o3 = (float*) ((uintptr_t) o3 + output_increment);
    rows = doz(rows, 4);
  } while (rows != 0);
}
