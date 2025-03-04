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
#include "xnnpack/dwconv.h"

void xnn_f32_dwconv_minmax_ukernel_25p8vc__rvv(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const struct xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    const float* i[25];
    i[0] = input[0];
    assert(i[0] != NULL);
    if XNN_UNPREDICTABLE(i[0] != zero) {
      i[0] = (const float*) ((uintptr_t) i[0] + input_offset);
    }
    i[1] = input[1];
    assert(i[1] != NULL);
    if XNN_UNPREDICTABLE(i[1] != zero) {
      i[1] = (const float*) ((uintptr_t) i[1] + input_offset);
    }
    i[2] = input[2];
    assert(i[2] != NULL);
    if XNN_UNPREDICTABLE(i[2] != zero) {
      i[2] = (const float*) ((uintptr_t) i[2] + input_offset);
    }
    i[3] = input[3];
    assert(i[3] != NULL);
    if XNN_UNPREDICTABLE(i[3] != zero) {
      i[3] = (const float*) ((uintptr_t) i[3] + input_offset);
    }
    i[4] = input[4];
    assert(i[4] != NULL);
    if XNN_UNPREDICTABLE(i[4] != zero) {
      i[4] = (const float*) ((uintptr_t) i[4] + input_offset);
    }
    i[5] = input[5];
    assert(i[5] != NULL);
    if XNN_UNPREDICTABLE(i[5] != zero) {
      i[5] = (const float*) ((uintptr_t) i[5] + input_offset);
    }
    i[6] = input[6];
    assert(i[6] != NULL);
    if XNN_UNPREDICTABLE(i[6] != zero) {
      i[6] = (const float*) ((uintptr_t) i[6] + input_offset);
    }
    i[7] = input[7];
    assert(i[7] != NULL);
    if XNN_UNPREDICTABLE(i[7] != zero) {
      i[7] = (const float*) ((uintptr_t) i[7] + input_offset);
    }
    i[8] = input[8];
    assert(i[8] != NULL);
    if XNN_UNPREDICTABLE(i[8] != zero) {
      i[8] = (const float*) ((uintptr_t) i[8] + input_offset);
    }
    i[9] = input[9];
    assert(i[9] != NULL);
    if XNN_UNPREDICTABLE(i[9] != zero) {
      i[9] = (const float*) ((uintptr_t) i[9] + input_offset);
    }
    i[10] = input[10];
    assert(i[10] != NULL);
    if XNN_UNPREDICTABLE(i[10] != zero) {
      i[10] = (const float*) ((uintptr_t) i[10] + input_offset);
    }
    i[11] = input[11];
    assert(i[11] != NULL);
    if XNN_UNPREDICTABLE(i[11] != zero) {
      i[11] = (const float*) ((uintptr_t) i[11] + input_offset);
    }
    i[12] = input[12];
    assert(i[12] != NULL);
    if XNN_UNPREDICTABLE(i[12] != zero) {
      i[12] = (const float*) ((uintptr_t) i[12] + input_offset);
    }
    i[13] = input[13];
    assert(i[13] != NULL);
    if XNN_UNPREDICTABLE(i[13] != zero) {
      i[13] = (const float*) ((uintptr_t) i[13] + input_offset);
    }
    i[14] = input[14];
    assert(i[14] != NULL);
    if XNN_UNPREDICTABLE(i[14] != zero) {
      i[14] = (const float*) ((uintptr_t) i[14] + input_offset);
    }
    i[15] = input[15];
    assert(i[15] != NULL);
    if XNN_UNPREDICTABLE(i[15] != zero) {
      i[15] = (const float*) ((uintptr_t) i[15] + input_offset);
    }
    i[16] = input[16];
    assert(i[16] != NULL);
    if XNN_UNPREDICTABLE(i[16] != zero) {
      i[16] = (const float*) ((uintptr_t) i[16] + input_offset);
    }
    i[17] = input[17];
    assert(i[17] != NULL);
    if XNN_UNPREDICTABLE(i[17] != zero) {
      i[17] = (const float*) ((uintptr_t) i[17] + input_offset);
    }
    i[18] = input[18];
    assert(i[18] != NULL);
    if XNN_UNPREDICTABLE(i[18] != zero) {
      i[18] = (const float*) ((uintptr_t) i[18] + input_offset);
    }
    i[19] = input[19];
    assert(i[19] != NULL);
    if XNN_UNPREDICTABLE(i[19] != zero) {
      i[19] = (const float*) ((uintptr_t) i[19] + input_offset);
    }
    i[20] = input[20];
    assert(i[20] != NULL);
    if XNN_UNPREDICTABLE(i[20] != zero) {
      i[20] = (const float*) ((uintptr_t) i[20] + input_offset);
    }
    i[21] = input[21];
    assert(i[21] != NULL);
    if XNN_UNPREDICTABLE(i[21] != zero) {
      i[21] = (const float*) ((uintptr_t) i[21] + input_offset);
    }
    i[22] = input[22];
    assert(i[22] != NULL);
    if XNN_UNPREDICTABLE(i[22] != zero) {
      i[22] = (const float*) ((uintptr_t) i[22] + input_offset);
    }
    i[23] = input[23];
    assert(i[23] != NULL);
    if XNN_UNPREDICTABLE(i[23] != zero) {
      i[23] = (const float*) ((uintptr_t) i[23] + input_offset);
    }
    i[24] = input[24];
    assert(i[24] != NULL);
    if XNN_UNPREDICTABLE(i[24] != zero) {
      i[24] = (const float*) ((uintptr_t) i[24] + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    const size_t vlmax = __riscv_vsetvlmax_e32m8();
    size_t vl;

    do {
      vl = __riscv_vsetvl_e32m8(c);
      // load bias to vAcc
      vfloat32m8_t vAcc = __riscv_vle32_v_f32m8_tu(vAcc, w, vl);
      w += vlmax;

      vfloat32m8_t va;
      vfloat32m8_t vb;
      for (int k=0; k<25; k++) {
        va = __riscv_vle32_v_f32m8_tu(va, i[k], vl);
        vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
        w  += vlmax;
        i[k] += vlmax;
        vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      }

      vAcc = __riscv_vfmax_vf_f32m8_tu(vAcc, vAcc, vmin, vl);
      vAcc = __riscv_vfmin_vf_f32m8_tu(vAcc, vAcc, vmax, vl);
      __riscv_vse32_v_f32m8(output, vAcc, vl);
      output += vl;
      c -= vl;
    } while(c != 0);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
