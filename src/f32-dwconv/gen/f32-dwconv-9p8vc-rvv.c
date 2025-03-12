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

void xnn_f32_dwconv_ukernel_9p8vc__rvv(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const struct xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

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

      __riscv_vse32_v_f32m8(output, vAcc, vl);
      output += vl;
      c -= vl;
    } while(c != 0);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
