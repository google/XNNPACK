// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/s8-rdminmax/rvv.c.in
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


void xnn_s8_rdmin_ukernel_2p2x__rvv_u8v(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int8_t* output,
    const void* params)
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t input_increment = 2 * input_stride;

  do {
    size_t vl = __riscv_vsetvl_e8m8(channels);

    const int8_t* i0 = input;
    const int8_t* i1 = (const int8_t*) ((uintptr_t) input + 1 * input_stride);

    vint8m8_t vmin = __riscv_vle8_v_i8m8(output, vl);

    for (int r = rows; r > 0; r -= 2) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = i0;
      }
      vint8m8_t vin_0 = __riscv_vle8_v_i8m8(i0, vl);
      vint8m8_t vin_1 = __riscv_vle8_v_i8m8(i1, vl);

      vmin = __riscv_vmin(vmin, vin_0, vl);
      vmin = __riscv_vmin(vmin, vin_1, vl);

      i0 = (int8_t*) ((uintptr_t) i0 + input_increment);
      i1 = (int8_t*) ((uintptr_t) i1 + input_increment);
    }

    __riscv_vse8(output, vmin, vl);

    output = (int8_t*) ((uintptr_t) output + vl * sizeof(int8_t));
    input = (int8_t*) ((uintptr_t) input + vl * sizeof(int8_t));
    channels -= vl;
  } while (channels > 0);
}
