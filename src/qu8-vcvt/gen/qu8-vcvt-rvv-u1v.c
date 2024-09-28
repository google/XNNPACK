// Auto-generated file. Do not edit!
//   Template: src/qs8-vcvt/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies, inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/vcvt.h"


void xnn_qu8_vcvt_ukernel__rvv_u1v(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const struct xnn_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int32_t bias = 
      ((int32_t) params->scalar.output_zero_point << 8) -
      (int32_t) params->scalar.multiplier * (int32_t) params->scalar.input_zero_point + 
      INT32_C(0x80);
  const int32_t multiplier = params->scalar.multiplier;
  int32_t n = __riscv_vsetvl_e8m1(batch);
  vint32m4_t bias_i32v = __riscv_vmv_v_x_i32m4(bias, n);

  do {
    n = __riscv_vsetvl_e8m1(batch); batch -= n;

    vuint8m1_t acc_u8v = __riscv_vle8_v_u8m1(input, n); input += n;
    vuint16m2_t acc_u16v = __riscv_vwcvtu_x_x_v_u16m2(acc_u8v, n);
    vint32m4_t acc_i32v = __riscv_vwmacc_vx_i32m4(bias_i32v, multiplier, __riscv_vreinterpret_v_u16m2_i16m2(acc_u16v), n);
    vint16m2_t out_i16v = __riscv_vnclip_wx_i16m2(acc_i32v, 8, __RISCV_VXRM_RDN, n);
    out_i16v = __riscv_vmax_vx_i16m2(out_i16v, 0, n);
    vuint8m1_t out_u8v = __riscv_vnclipu_wx_u8m1(__riscv_vreinterpret_v_i16m2_u16m2(out_i16v), 0, __RISCV_VXRM_RNU, n);
    __riscv_vse8_v_u8m1(output, out_u8v, n); output += n;
  } while (batch != 0);
}
