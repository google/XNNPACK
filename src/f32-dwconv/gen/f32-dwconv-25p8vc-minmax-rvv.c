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

void xnn_f32_dwconv_minmax_ukernel_25p8vc__rvv(
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
    const struct xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
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
    const size_t vlmax = __riscv_vsetvlmax_e32m8();
    size_t vl;

    do {
      vl = __riscv_vsetvl_e32m8(c);
      // load bias to vAcc
      vfloat32m8_t vAcc = __riscv_vundefined_f32m8();
      vAcc = __riscv_vle32_v_f32m8_tu(vAcc, w, vl);
      w += vlmax;

      vfloat32m8_t va = __riscv_vundefined_f32m8();
      vfloat32m8_t vb = __riscv_vundefined_f32m8();
      va = __riscv_vle32_v_f32m8_tu(va, i0, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i0 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i1, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i1 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i2, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i2 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i3, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i3 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i4, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i4 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i5, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i5 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i6, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i6 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i7, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i7 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i8, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i8 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i9, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i9 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i10, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i10 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i11, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i11 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i12, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i12 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i13, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i13 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i14, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i14 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i15, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i15 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i16, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i16 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i17, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i17 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i18, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i18 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i19, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i19 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i20, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i20 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i21, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i21 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i22, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i22 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i23, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i23 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      va = __riscv_vle32_v_f32m8_tu(va, i24, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += vlmax;
      i24 += vlmax;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);

      vAcc = __riscv_vfmax_vf_f32m8_tu(vAcc, vAcc, vmin, vl);
      vAcc = __riscv_vfmin_vf_f32m8_tu(vAcc, vAcc, vmax, vl);
      __riscv_vse32_v_f32m8(output, vAcc, vl);
      output += vl;
      c -= vl;
    } while(c != 0);
    input_offset += input_pixel_stride;
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
