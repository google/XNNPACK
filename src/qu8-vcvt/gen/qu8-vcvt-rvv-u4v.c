// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-vcvt/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Microchip
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>
#include "src/xnnpack/vcvt.h"

void xnn_qu8_vcvt_ukernel__rvv_u4v(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const struct xnn_qu8_cvt_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint8_t input_zero_point = params->scalar.input_zero_point;
  const int16_t multiplier = params->scalar.multiplier;
  const int16_t output_zero_point = params->scalar.output_zero_point;

  do {
    size_t vl = __riscv_vsetvl_e8m4(batch); batch -= vl;
    vuint8m4_t in_u8v = __riscv_vle8_v_u8m4(input, vl); input += vl;
    vint16m8_t acc_i16v = __riscv_vreinterpret_i16m8(__riscv_vwsubu_vx(in_u8v, input_zero_point, vl));
    acc_i16v = __riscv_vsll(acc_i16v, 7, vl);
    acc_i16v = __riscv_vsmul(acc_i16v, multiplier, __RISCV_VXRM_RNU, vl);
    acc_i16v = __riscv_vsadd(acc_i16v, output_zero_point, vl);
    acc_i16v = __riscv_vmax(acc_i16v, 0, vl);
    vuint8m4_t out_u8v = __riscv_vnclipu(__riscv_vreinterpret_u16m8(acc_i16v), 0, __RISCV_VXRM_RDN, vl);
    __riscv_vse8(output, out_u8v, vl); output += vl;
  } while (batch != 0);
}
