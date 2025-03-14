// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qu8-rsum/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Microchip
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <riscv_vector.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/reduce.h"

void xnn_qu8_rsum_ukernel__rvv_u1v(
    size_t batch,
    const uint8_t* restrict input,
    uint32_t* restrict output,
    const struct xnn_qs8_rsum_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  size_t vlmax = __riscv_vsetvl_e32m4(batch);
  vuint32m4_t vsum = __riscv_vmv_v_x_u32m4(0, vlmax);
  vuint32m1_t vzero = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());

  do {
    size_t vl = __riscv_vsetvl_e8m1(batch); batch -= vl;

    vuint8m1_t vinput = __riscv_vle8_v_u8m1(input, vl); input += vl;
    vuint16m2_t vinput16 = __riscv_vzext_vf2_u16m2(vinput, vl);

    vsum = __riscv_vwaddu_wv_u32m4(vsum, vinput16, vl);

  } while (batch != 0);

  vuint32m1_t vred = __riscv_vredsum(vsum, vzero, vlmax);
  *output += __riscv_vmv_x_s_u32m1_u32(vred);
}
