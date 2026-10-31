// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/unipass-rvv.c.in
//   Generator: tools/xngen
//

// Copyright 2024 Andes Technology Corporation
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.#

#include <assert.h>
#include <riscv_vector.h>
#include "src/xnnpack/dwconv.h"

void xnn_f32_dwconv_minmax_ukernel_9p4vc__rvv(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    size_t input_pixel_stride,
    const float* zero,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  const size_t vlmax = __riscv_vsetvlmax_e32m4();
  const size_t output_pixel_stride = channels * sizeof(float) + output_increment;

  // Two output pixels per iteration: the 9 weight vectors are loaded
  // once and reused for both pixels, so the weight traffic is halved. At
  // CHANNEL_TILE=m8 this needs 4 live vector registers (2 accumulators + weight +
  // input), which is exactly the 32-entry register file; a wider block spills and
  // ends up slower than not blocking at all.
  while (output_width > 1) {
    const float** input1 = (const float**) ((uintptr_t) input + input_stride);
    const float* i0_0 = input[0];
    assert(i0_0 != NULL);
    if XNN_UNPREDICTABLE(i0_0 != zero) {
      i0_0 = (const float*) ((uintptr_t) i0_0 + input_offset);
    }
    const float* i0_1 = input[1];
    assert(i0_1 != NULL);
    if XNN_UNPREDICTABLE(i0_1 != zero) {
      i0_1 = (const float*) ((uintptr_t) i0_1 + input_offset);
    }
    const float* i0_2 = input[2];
    assert(i0_2 != NULL);
    if XNN_UNPREDICTABLE(i0_2 != zero) {
      i0_2 = (const float*) ((uintptr_t) i0_2 + input_offset);
    }
    const float* i0_3 = input[3];
    assert(i0_3 != NULL);
    if XNN_UNPREDICTABLE(i0_3 != zero) {
      i0_3 = (const float*) ((uintptr_t) i0_3 + input_offset);
    }
    const float* i0_4 = input[4];
    assert(i0_4 != NULL);
    if XNN_UNPREDICTABLE(i0_4 != zero) {
      i0_4 = (const float*) ((uintptr_t) i0_4 + input_offset);
    }
    const float* i0_5 = input[5];
    assert(i0_5 != NULL);
    if XNN_UNPREDICTABLE(i0_5 != zero) {
      i0_5 = (const float*) ((uintptr_t) i0_5 + input_offset);
    }
    const float* i0_6 = input[6];
    assert(i0_6 != NULL);
    if XNN_UNPREDICTABLE(i0_6 != zero) {
      i0_6 = (const float*) ((uintptr_t) i0_6 + input_offset);
    }
    const float* i0_7 = input[7];
    assert(i0_7 != NULL);
    if XNN_UNPREDICTABLE(i0_7 != zero) {
      i0_7 = (const float*) ((uintptr_t) i0_7 + input_offset);
    }
    const float* i0_8 = input[8];
    assert(i0_8 != NULL);
    if XNN_UNPREDICTABLE(i0_8 != zero) {
      i0_8 = (const float*) ((uintptr_t) i0_8 + input_offset);
    }
    const float* i1_0 = input1[0];
    assert(i1_0 != NULL);
    if XNN_UNPREDICTABLE(i1_0 != zero) {
      i1_0 = (const float*) ((uintptr_t) i1_0 + input_offset + input_pixel_stride);
    }
    const float* i1_1 = input1[1];
    assert(i1_1 != NULL);
    if XNN_UNPREDICTABLE(i1_1 != zero) {
      i1_1 = (const float*) ((uintptr_t) i1_1 + input_offset + input_pixel_stride);
    }
    const float* i1_2 = input1[2];
    assert(i1_2 != NULL);
    if XNN_UNPREDICTABLE(i1_2 != zero) {
      i1_2 = (const float*) ((uintptr_t) i1_2 + input_offset + input_pixel_stride);
    }
    const float* i1_3 = input1[3];
    assert(i1_3 != NULL);
    if XNN_UNPREDICTABLE(i1_3 != zero) {
      i1_3 = (const float*) ((uintptr_t) i1_3 + input_offset + input_pixel_stride);
    }
    const float* i1_4 = input1[4];
    assert(i1_4 != NULL);
    if XNN_UNPREDICTABLE(i1_4 != zero) {
      i1_4 = (const float*) ((uintptr_t) i1_4 + input_offset + input_pixel_stride);
    }
    const float* i1_5 = input1[5];
    assert(i1_5 != NULL);
    if XNN_UNPREDICTABLE(i1_5 != zero) {
      i1_5 = (const float*) ((uintptr_t) i1_5 + input_offset + input_pixel_stride);
    }
    const float* i1_6 = input1[6];
    assert(i1_6 != NULL);
    if XNN_UNPREDICTABLE(i1_6 != zero) {
      i1_6 = (const float*) ((uintptr_t) i1_6 + input_offset + input_pixel_stride);
    }
    const float* i1_7 = input1[7];
    assert(i1_7 != NULL);
    if XNN_UNPREDICTABLE(i1_7 != zero) {
      i1_7 = (const float*) ((uintptr_t) i1_7 + input_offset + input_pixel_stride);
    }
    const float* i1_8 = input1[8];
    assert(i1_8 != NULL);
    if XNN_UNPREDICTABLE(i1_8 != zero) {
      i1_8 = (const float*) ((uintptr_t) i1_8 + input_offset + input_pixel_stride);
    }
    input = (const float**) ((uintptr_t) input + 2 * input_stride);

    float* o0 = output;
    float* o1 = (float*) ((uintptr_t) output + output_pixel_stride);

    size_t c = channels;
    const float* w = weights;
    do {
      size_t vl = __riscv_vsetvl_e32m4(c);
      // load bias to vAcc0 and vAcc1, both pixels share the same bias
      vfloat32m4_t vAcc0 = __riscv_vundefined_f32m4();
      vAcc0 = __riscv_vle32_v_f32m4_tu(vAcc0, w, vl);
      vfloat32m4_t vAcc1 = vAcc0;
      w += vlmax;

      vfloat32m4_t va = __riscv_vundefined_f32m4();
      vfloat32m4_t vb = __riscv_vundefined_f32m4();
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_0, vl);
      i0_0 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_0, vl);
      i1_0 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_1, vl);
      i0_1 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_1, vl);
      i1_1 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_2, vl);
      i0_2 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_2, vl);
      i1_2 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_3, vl);
      i0_3 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_3, vl);
      i1_3 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_4, vl);
      i0_4 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_4, vl);
      i1_4 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_5, vl);
      i0_5 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_5, vl);
      i1_5 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_6, vl);
      i0_6 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_6, vl);
      i1_6 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_7, vl);
      i0_7 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_7, vl);
      i1_7 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_8, vl);
      i0_8 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_8, vl);
      i1_8 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);

      vAcc0 = __riscv_vfmax_vf_f32m4_tu(vAcc0, vAcc0, vmin, vl);
      vAcc1 = __riscv_vfmax_vf_f32m4_tu(vAcc1, vAcc1, vmin, vl);
      vAcc0 = __riscv_vfmin_vf_f32m4_tu(vAcc0, vAcc0, vmax, vl);
      vAcc1 = __riscv_vfmin_vf_f32m4_tu(vAcc1, vAcc1, vmax, vl);
      __riscv_vse32_v_f32m4(o0, vAcc0, vl);
      __riscv_vse32_v_f32m4(o1, vAcc1, vl);
      o0 += vl;
      o1 += vl;
      c -= vl;
    } while(c != 0);

    input_offset += 2 * input_pixel_stride;
    output = (float*) ((uintptr_t) output + 2 * output_pixel_stride);
    output_width -= 2;
  }

  if (output_width != 0) {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    size_t vl;
    do {
      vl = __riscv_vsetvl_e32m4(c);
      // load bias to vAcc
      vfloat32m4_t vAcc = __riscv_vundefined_f32m4();
      vAcc = __riscv_vle32_v_f32m4_tu(vAcc, w, vl);
      w += vlmax;

      vfloat32m4_t va = __riscv_vundefined_f32m4();
      vfloat32m4_t vb = __riscv_vundefined_f32m4();
      va = __riscv_vle32_v_f32m4_tu(va, i0, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i0 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i1 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i2, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i2 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i3, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i3 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i4, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i4 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i5, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i5 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i6, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i6 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i7, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i7 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i8, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i8 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);

      vAcc = __riscv_vfmax_vf_f32m4_tu(vAcc, vAcc, vmin, vl);
      vAcc = __riscv_vfmin_vf_f32m4_tu(vAcc, vAcc, vmax, vl);
      __riscv_vse32_v_f32m4(output, vAcc, vl);
      output += vl;
      c -= vl;
    } while(c != 0);
  }
}
