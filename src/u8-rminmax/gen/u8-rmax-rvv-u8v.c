// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/s8-rminmax/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/reduce.h"


void xnn_u8_rmax_ukernel__rvv_u8v(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t vlmax = __riscv_vsetvl_e8m8(batch); batch -= vlmax;
  vuint8m8_t vmax = __riscv_vle8_v_u8m8(input, vlmax); input += vlmax;

  while (batch > 0) {
    size_t vl = __riscv_vsetvl_e8m8(batch); batch -= vl;
    vuint8m8_t vinput = __riscv_vle8_v_u8m8(input, vl); input += vl;
    vmax = __riscv_vmaxu(vmax, vinput, vl);
  }

  vuint8m1_t max = __riscv_vle8_v_u8m1(output + 0, 1);
  output[0] = __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu(vmax, max, vlmax));
}
