// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-rdminmax/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/reduce.h"


void xnn_f16_rdmax_ukernel_2p2x__rvvfp16arith_u8v(
    size_t rows,
    size_t channels,
    const xnn_float16* input,
    size_t input_stride,
    const xnn_float16* zero,
    xnn_float16* output,
    const void* params)
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t input_increment = 2 * input_stride;

  do {
    size_t vl = __riscv_vsetvl_e16m8(channels);

    const xnn_float16* i0 = input;
    const xnn_float16* i1 = (const xnn_float16*) ((uintptr_t) input + 1 * input_stride);

    vfloat16m8_t vmax = __riscv_vle16_v_f16m8(output, vl);

    for (int r = rows; r > 0; r -= 2) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = i0;
      }
      vfloat16m8_t vin_0 = __riscv_vle16_v_f16m8(i0, vl);
      vfloat16m8_t vin_1 = __riscv_vle16_v_f16m8(i1, vl);

      vmax = __riscv_vfmax(vmax, vin_0, vl);
      vmax = __riscv_vfmax(vmax, vin_1, vl);

      i0 = (xnn_float16*) ((uintptr_t) i0 + input_increment);
      i1 = (xnn_float16*) ((uintptr_t) i1 + input_increment);
    }

    __riscv_vse16(output, vmax, vl);

    output = (xnn_float16*) ((uintptr_t) output + vl * sizeof(xnn_float16));
    input = (xnn_float16*) ((uintptr_t) input + vl * sizeof(xnn_float16));
    channels -= vl;
  } while (channels > 0);
}
