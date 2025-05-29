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

void xnn_qs8_vcvt_ukernel__rvv_u4v(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_qs8_cvt_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int16_t input_zero_point = params->scalar.input_zero_point;
  const int16_t multiplier = params->scalar.multiplier;
  const int16_t output_zero_point = params->scalar.output_zero_point;

  do {
    size_t vl = __riscv_vsetvl_e8m4(batch); batch -= vl;

    vint8m4_t in_i8v = __riscv_vle8_v_i8m4(input, vl); input += vl;
    vint16m8_t acc_i16v = __riscv_vwsub_vx_i16m8(in_i8v, input_zero_point, vl);
    acc_i16v = __riscv_vsmul_vx_i16m8(acc_i16v, multiplier, __RISCV_VXRM_RNU, vl);
    acc_i16v = __riscv_vsadd_vx_i16m8(acc_i16v, output_zero_point, vl);
    vint8m4_t out_i8v = __riscv_vnclip_wx_i8m4(acc_i16v, 8, __RISCV_VXRM_RNU, vl);
    __riscv_vse8_v_i8m4(output, out_i8v, vl); output += vl;
  } while (batch != 0);
}
