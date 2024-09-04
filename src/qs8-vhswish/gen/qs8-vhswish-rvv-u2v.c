// Auto-generated file. Do not edit!
//   Template: src/qs8-vhswish/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies, inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/vhswish.h"


void xnn_qs8_vhswish_ukernel__rvv_u2v(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_hswish_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint32_t input_zero_point = (uint32_t) params->scalar.input_zero_point;
  const int32_t output_zero_point = params->scalar.output_zero_point;
  const int32_t input_scale_div_mantissa = params->scalar.input_scale_div_mantissa;
  const int32_t input_scale_div_exp = params->scalar.input_scale_div_exp;
  const int32_t scale_ratio = params->scalar.scale_ratio;

  vint8m2_t inzp_i8v = __riscv_vmv_v_x_i8m2(input_zero_point, __riscv_vsetvl_e8m2(batch));

  do {
    int32_t n = __riscv_vsetvl_e8m2(batch); batch -= n;

    vint8m2_t in_i8v = __riscv_vle8_v_i8m2(input, n); input += n;
    vint16m4_t insub_i16v = __riscv_vwsub_vv_i16m4(inzp_i8v, in_i8v, n);

    vint16m4_t sl_i16v = __riscv_vsll_vx_i16m4(insub_i16v, 7, n);
    vint32m8_t acc_i32v = __riscv_vwmul_vx_i32m8(sl_i16v, input_scale_div_mantissa, n);
    vint32m8_t insl_i32v = __riscv_vsra_vx_i32m8(acc_i32v, -input_scale_div_exp, n);
    vint32m8_t insub_i32v = __riscv_vsub_vx_i32m8(insl_i32v, 16384, n);
    vint16m4_t in_i16v = __riscv_vnclip_wx_i16m4(insub_i32v, 0, __RISCV_VXRM_RNU , n);
    in_i16v = __riscv_vmin_vx_i16m4(in_i16v, 0, n);
    vint32m8_t accm_i32v = __riscv_vwmul_vx_i32m8(sl_i16v, scale_ratio, n);
    vint16m4_t sra_i16v = __riscv_vnclip_wx_i16m4(accm_i32v, 15, __RISCV_VXRM_RNU, n);
    vint32m8_t mul_i32v = __riscv_vwmul_vv_i32m8(in_i16v, sra_i16v, n);
    vint16m4_t mulsra_i16v = __riscv_vnclip_wx_i16m4(mul_i32v, 15, __RISCV_VXRM_RNU, n);
    vint16m4_t add_i16v = __riscv_vadd_vx_i16m4(mulsra_i16v, output_zero_point, n);

    vint8m2_t out_i8v = __riscv_vnclip_wx_i8m2(add_i16v, 0, 0, n);
    __riscv_vse8_v_i8m2(output, out_i8v, n); output += n;
  } while (batch != 0);
}
