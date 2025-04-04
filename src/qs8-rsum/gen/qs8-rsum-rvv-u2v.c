// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/rvv.c.in
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

void xnn_qs8_rsum_ukernel__rvv_u2v(
    size_t batch,
    const int8_t* restrict input,
    int32_t* restrict output,
    const struct xnn_qs8_rsum_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  size_t vlmax = __riscv_vsetvl_e32m8(batch);
  vint32m8_t vsum = __riscv_vmv_v_x_i32m8(0, vlmax);
  vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());

  do {
    size_t vl = __riscv_vsetvl_e8m2(batch); batch -= vl;

    vint8m2_t vinput = __riscv_vle8_v_i8m2(input, vl); input += vl;
    vint16m4_t vinput16 = __riscv_vsext_vf2_i16m4(vinput, vl);

    vsum = __riscv_vwadd_wv_i32m8(vsum, vinput16, vl);

  } while (batch != 0);

  vint32m1_t vred = __riscv_vredsum(vsum, vzero, vlmax);
  *output += __riscv_vmv_x_s_i32m1_i32(vred);
}
