// Auto-generated file. Do not edit!
//   Template: src/f32-prelu/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/math.h>
#include <xnnpack/prelu.h>
#include <riscv_vector.h>

void xnn_f32_prelu_ukernel__rvv_1x4v(
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

  channels >>= XNN_LOG2_SIZEOF_FLOAT;
  input_stride >>= XNN_LOG2_SIZEOF_FLOAT;
  output_stride >>= XNN_LOG2_SIZEOF_FLOAT;

  const float* i0 = input;
  float* o0 = output;
  float zero = 0.0f;


  const size_t input_increment = input_stride * 1 - channels;
  const size_t output_increment = output_stride * 1 - channels;

  do {

    const float* w = weights;
    size_t c = channels;

    for (; c > 0;) {
      size_t n = __riscv_vsetvl_e32m4(c); c -= n;
      vfloat32m4_t w_f32v = __riscv_vle32_v_f32m4(w, n); w += n;

      vfloat32m4_t in0_f32v = __riscv_vle32_v_f32m4(i0, n); i0 += n;
      vbool8_t mask0_f32v = __riscv_vmflt_vf_f32m4_b8(in0_f32v, zero, n);
      vfloat32m4_t out0_f32v = __riscv_vfmul_vv_f32m4_mu(mask0_f32v, in0_f32v, w_f32v, in0_f32v, n);
      __riscv_vse32_v_f32m4(o0, out0_f32v, n); o0 += n;
    }

    i0 = i0 + input_increment;
    o0 = o0 + output_increment;
    rows = doz(rows, 1);
  } while (rows != 0);
}
