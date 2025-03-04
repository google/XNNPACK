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

void xnn_f32_dwconv_ukernel_9p4vc__rvv(
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
    const float* i[9];
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
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    const size_t vlmax = __riscv_vsetvlmax_e32m4();
    size_t vl;

    do {
      vl = __riscv_vsetvl_e32m4(c);
      // load bias to vAcc
      vfloat32m4_t vAcc = __riscv_vle32_v_f32m4_tu(vAcc, w, vl);
      w += vlmax;

      vfloat32m4_t va;
      vfloat32m4_t vb;
      for (int k=0; k<9; k++) {
        va = __riscv_vle32_v_f32m4_tu(va, i[k], vl);
        vb = __riscv_vle32_v_f32m4_tu(vb, w, vl);
        w  += vlmax;
        i[k] += vlmax;
        vAcc = __riscv_vfmacc_vv_f32m4_tu(vAcc, va, vb, vl);
      }

      __riscv_vse32_v_f32m4(output, vAcc, vl);
      output += vl;
      c -= vl;
    } while(c != 0);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
