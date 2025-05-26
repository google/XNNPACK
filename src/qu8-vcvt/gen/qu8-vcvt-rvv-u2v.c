// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-vcvt/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Microchip
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>
#include "src/xnnpack/vcvt.h"

void xnn_qu8_vcvt_ukernel__rvv_u2v(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const struct xnn_qu8_cvt_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int16_t input_zero_point = params->scalar.input_zero_point;
  const int16_t multiplier = params->scalar.multiplier;
  const int16_t output_zero_point = params->scalar.output_zero_point;

  do {
    size_t vl = __riscv_vsetvl_e8m2(batch); batch -= vl;

    vuint8m2_t in_u8v = __riscv_vle8_v_u8m2(input, vl); input += vl;
    vint16m4_t acc_i16v = __riscv_vreinterpret_i16m4(__riscv_vwsubu_vx_u16m4(in_u8v, input_zero_point, vl));
    acc_i16v = __riscv_vsmul_vx_i16m4(acc_i16v, multiplier, __RISCV_VXRM_RNU, vl);
    acc_i16v = __riscv_vsadd_vx_i16m4(acc_i16v, output_zero_point, vl);
    acc_i16v = __riscv_vmax_vx_i16m4(acc_i16v, 0, vl);
    vuint8m2_t out_u8v = __riscv_vnclipu_wx_u8m2(__riscv_vreinterpret_u16m4(acc_i16v), 8, __RISCV_VXRM_RNU, vl);
    __riscv_vse8_v_u8m2(output, out_u8v, vl); output += vl;
  } while (batch != 0);
}
