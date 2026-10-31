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

void xnn_f32_dwconv_minmax_ukernel_25p4vc__rvv(
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

  // Two output pixels per iteration: the 25 weight vectors are loaded
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
    const float* i0_9 = input[9];
    assert(i0_9 != NULL);
    if XNN_UNPREDICTABLE(i0_9 != zero) {
      i0_9 = (const float*) ((uintptr_t) i0_9 + input_offset);
    }
    const float* i0_10 = input[10];
    assert(i0_10 != NULL);
    if XNN_UNPREDICTABLE(i0_10 != zero) {
      i0_10 = (const float*) ((uintptr_t) i0_10 + input_offset);
    }
    const float* i0_11 = input[11];
    assert(i0_11 != NULL);
    if XNN_UNPREDICTABLE(i0_11 != zero) {
      i0_11 = (const float*) ((uintptr_t) i0_11 + input_offset);
    }
    const float* i0_12 = input[12];
    assert(i0_12 != NULL);
    if XNN_UNPREDICTABLE(i0_12 != zero) {
      i0_12 = (const float*) ((uintptr_t) i0_12 + input_offset);
    }
    const float* i0_13 = input[13];
    assert(i0_13 != NULL);
    if XNN_UNPREDICTABLE(i0_13 != zero) {
      i0_13 = (const float*) ((uintptr_t) i0_13 + input_offset);
    }
    const float* i0_14 = input[14];
    assert(i0_14 != NULL);
    if XNN_UNPREDICTABLE(i0_14 != zero) {
      i0_14 = (const float*) ((uintptr_t) i0_14 + input_offset);
    }
    const float* i0_15 = input[15];
    assert(i0_15 != NULL);
    if XNN_UNPREDICTABLE(i0_15 != zero) {
      i0_15 = (const float*) ((uintptr_t) i0_15 + input_offset);
    }
    const float* i0_16 = input[16];
    assert(i0_16 != NULL);
    if XNN_UNPREDICTABLE(i0_16 != zero) {
      i0_16 = (const float*) ((uintptr_t) i0_16 + input_offset);
    }
    const float* i0_17 = input[17];
    assert(i0_17 != NULL);
    if XNN_UNPREDICTABLE(i0_17 != zero) {
      i0_17 = (const float*) ((uintptr_t) i0_17 + input_offset);
    }
    const float* i0_18 = input[18];
    assert(i0_18 != NULL);
    if XNN_UNPREDICTABLE(i0_18 != zero) {
      i0_18 = (const float*) ((uintptr_t) i0_18 + input_offset);
    }
    const float* i0_19 = input[19];
    assert(i0_19 != NULL);
    if XNN_UNPREDICTABLE(i0_19 != zero) {
      i0_19 = (const float*) ((uintptr_t) i0_19 + input_offset);
    }
    const float* i0_20 = input[20];
    assert(i0_20 != NULL);
    if XNN_UNPREDICTABLE(i0_20 != zero) {
      i0_20 = (const float*) ((uintptr_t) i0_20 + input_offset);
    }
    const float* i0_21 = input[21];
    assert(i0_21 != NULL);
    if XNN_UNPREDICTABLE(i0_21 != zero) {
      i0_21 = (const float*) ((uintptr_t) i0_21 + input_offset);
    }
    const float* i0_22 = input[22];
    assert(i0_22 != NULL);
    if XNN_UNPREDICTABLE(i0_22 != zero) {
      i0_22 = (const float*) ((uintptr_t) i0_22 + input_offset);
    }
    const float* i0_23 = input[23];
    assert(i0_23 != NULL);
    if XNN_UNPREDICTABLE(i0_23 != zero) {
      i0_23 = (const float*) ((uintptr_t) i0_23 + input_offset);
    }
    const float* i0_24 = input[24];
    assert(i0_24 != NULL);
    if XNN_UNPREDICTABLE(i0_24 != zero) {
      i0_24 = (const float*) ((uintptr_t) i0_24 + input_offset);
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
    const float* i1_9 = input1[9];
    assert(i1_9 != NULL);
    if XNN_UNPREDICTABLE(i1_9 != zero) {
      i1_9 = (const float*) ((uintptr_t) i1_9 + input_offset + input_pixel_stride);
    }
    const float* i1_10 = input1[10];
    assert(i1_10 != NULL);
    if XNN_UNPREDICTABLE(i1_10 != zero) {
      i1_10 = (const float*) ((uintptr_t) i1_10 + input_offset + input_pixel_stride);
    }
    const float* i1_11 = input1[11];
    assert(i1_11 != NULL);
    if XNN_UNPREDICTABLE(i1_11 != zero) {
      i1_11 = (const float*) ((uintptr_t) i1_11 + input_offset + input_pixel_stride);
    }
    const float* i1_12 = input1[12];
    assert(i1_12 != NULL);
    if XNN_UNPREDICTABLE(i1_12 != zero) {
      i1_12 = (const float*) ((uintptr_t) i1_12 + input_offset + input_pixel_stride);
    }
    const float* i1_13 = input1[13];
    assert(i1_13 != NULL);
    if XNN_UNPREDICTABLE(i1_13 != zero) {
      i1_13 = (const float*) ((uintptr_t) i1_13 + input_offset + input_pixel_stride);
    }
    const float* i1_14 = input1[14];
    assert(i1_14 != NULL);
    if XNN_UNPREDICTABLE(i1_14 != zero) {
      i1_14 = (const float*) ((uintptr_t) i1_14 + input_offset + input_pixel_stride);
    }
    const float* i1_15 = input1[15];
    assert(i1_15 != NULL);
    if XNN_UNPREDICTABLE(i1_15 != zero) {
      i1_15 = (const float*) ((uintptr_t) i1_15 + input_offset + input_pixel_stride);
    }
    const float* i1_16 = input1[16];
    assert(i1_16 != NULL);
    if XNN_UNPREDICTABLE(i1_16 != zero) {
      i1_16 = (const float*) ((uintptr_t) i1_16 + input_offset + input_pixel_stride);
    }
    const float* i1_17 = input1[17];
    assert(i1_17 != NULL);
    if XNN_UNPREDICTABLE(i1_17 != zero) {
      i1_17 = (const float*) ((uintptr_t) i1_17 + input_offset + input_pixel_stride);
    }
    const float* i1_18 = input1[18];
    assert(i1_18 != NULL);
    if XNN_UNPREDICTABLE(i1_18 != zero) {
      i1_18 = (const float*) ((uintptr_t) i1_18 + input_offset + input_pixel_stride);
    }
    const float* i1_19 = input1[19];
    assert(i1_19 != NULL);
    if XNN_UNPREDICTABLE(i1_19 != zero) {
      i1_19 = (const float*) ((uintptr_t) i1_19 + input_offset + input_pixel_stride);
    }
    const float* i1_20 = input1[20];
    assert(i1_20 != NULL);
    if XNN_UNPREDICTABLE(i1_20 != zero) {
      i1_20 = (const float*) ((uintptr_t) i1_20 + input_offset + input_pixel_stride);
    }
    const float* i1_21 = input1[21];
    assert(i1_21 != NULL);
    if XNN_UNPREDICTABLE(i1_21 != zero) {
      i1_21 = (const float*) ((uintptr_t) i1_21 + input_offset + input_pixel_stride);
    }
    const float* i1_22 = input1[22];
    assert(i1_22 != NULL);
    if XNN_UNPREDICTABLE(i1_22 != zero) {
      i1_22 = (const float*) ((uintptr_t) i1_22 + input_offset + input_pixel_stride);
    }
    const float* i1_23 = input1[23];
    assert(i1_23 != NULL);
    if XNN_UNPREDICTABLE(i1_23 != zero) {
      i1_23 = (const float*) ((uintptr_t) i1_23 + input_offset + input_pixel_stride);
    }
    const float* i1_24 = input1[24];
    assert(i1_24 != NULL);
    if XNN_UNPREDICTABLE(i1_24 != zero) {
      i1_24 = (const float*) ((uintptr_t) i1_24 + input_offset + input_pixel_stride);
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
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_9, vl);
      i0_9 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_9, vl);
      i1_9 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_10, vl);
      i0_10 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_10, vl);
      i1_10 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_11, vl);
      i0_11 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_11, vl);
      i1_11 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_12, vl);
      i0_12 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_12, vl);
      i1_12 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_13, vl);
      i0_13 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_13, vl);
      i1_13 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_14, vl);
      i0_14 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_14, vl);
      i1_14 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_15, vl);
      i0_15 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_15, vl);
      i1_15 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_16, vl);
      i0_16 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_16, vl);
      i1_16 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_17, vl);
      i0_17 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_17, vl);
      i1_17 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_18, vl);
      i0_18 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_18, vl);
      i1_18 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_19, vl);
      i0_19 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_19, vl);
      i1_19 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_20, vl);
      i0_20 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_20, vl);
      i1_20 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_21, vl);
      i0_21 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_21, vl);
      i1_21 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_22, vl);
      i0_22 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_22, vl);
      i1_22 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_23, vl);
      i0_23 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_23, vl);
      i1_23 += vlmax;
      vAcc1 = __riscv_vfmacc_vv_f32m4_tu(vAcc1, va, vb, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      va = __riscv_vle32_v_f32m4_tu(va, i0_24, vl);
      i0_24 += vlmax;
      vAcc0 = __riscv_vfmacc_vv_f32m4_tu(vAcc0, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i1_24, vl);
      i1_24 += vlmax;
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
    const float* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const float*) ((uintptr_t) i9 + input_offset);
    }
    const float* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const float*) ((uintptr_t) i10 + input_offset);
    }
    const float* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const float*) ((uintptr_t) i11 + input_offset);
    }
    const float* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const float*) ((uintptr_t) i12 + input_offset);
    }
    const float* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const float*) ((uintptr_t) i13 + input_offset);
    }
    const float* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const float*) ((uintptr_t) i14 + input_offset);
    }
    const float* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const float*) ((uintptr_t) i15 + input_offset);
    }
    const float* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const float*) ((uintptr_t) i16 + input_offset);
    }
    const float* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const float*) ((uintptr_t) i17 + input_offset);
    }
    const float* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const float*) ((uintptr_t) i18 + input_offset);
    }
    const float* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const float*) ((uintptr_t) i19 + input_offset);
    }
    const float* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const float*) ((uintptr_t) i20 + input_offset);
    }
    const float* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const float*) ((uintptr_t) i21 + input_offset);
    }
    const float* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const float*) ((uintptr_t) i22 + input_offset);
    }
    const float* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const float*) ((uintptr_t) i23 + input_offset);
    }
    const float* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const float*) ((uintptr_t) i24 + input_offset);
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
      va = __riscv_vle32_v_f32m4_tu(va, i9, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i9 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i10, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i10 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i11, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i11 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i12, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i12 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i13, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i13 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i14, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i14 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i15, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i15 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i16, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i16 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i17, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i17 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i18, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i18 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i19, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i19 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i20, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i20 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i21, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i21 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i22, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i22 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i23, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i23 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m4_tu(va, i24, vl);
      vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
      w  += vlmax;
      i24 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);

      vAcc = __riscv_vfmax_vf_f32m4_tu(vAcc, vAcc, vmin, vl);
      vAcc = __riscv_vfmin_vf_f32m4_tu(vAcc, vAcc, vmax, vl);
      __riscv_vse32_v_f32m4(output, vAcc, vl);
      output += vl;
      c -= vl;
    } while(c != 0);
  }
}
