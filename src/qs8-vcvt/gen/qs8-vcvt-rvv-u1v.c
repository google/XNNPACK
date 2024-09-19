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


void xnn_qs8_vcvt_ukernel__rvv_u1v(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
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

    vint8m1_t acc_i8v = __riscv_vle8_v_i8m1(input, n); input += n;
    vint16m2_t acc_i16v = __riscv_vwcvt_x_x_v_i16m2(acc_i8v, n);
    vint32m4_t acc_i32v = __riscv_vwmacc_vx_i32m4(bias_i32v, multiplier, acc_i16v, n);
    vint16m2_t out_i16v = __riscv_vnclip_wx_i16m2(acc_i32v, 8, __RISCV_VXRM_RDN, n);
    vint8m1_t out_i8v = __riscv_vnclip_wx_i8m1(out_i16v, 0, __RISCV_VXRM_RNU, n);
    __riscv_vse8_v_i8m1(output, out_i8v, n); output += n;
  } while (batch != 0);
}
