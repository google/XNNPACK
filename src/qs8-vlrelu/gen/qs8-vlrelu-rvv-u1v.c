// Auto-generated file. Do not edit!
//   Template: src/qs8-vlrelu/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies, inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/vlrelu.h"


void xnn_qs8_vlrelu_ukernel__rvv_u1v(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_qs8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int32_t input_zero_point = params->scalar.input_zero_point;
  const int32_t multiplier_diff = params->scalar.negative_multiplier ^ params->scalar.positive_multiplier;
  const int32_t multiplier_base = params->scalar.positive_multiplier;
  const int32_t bias = (params->scalar.output_zero_point << 8) + 128;
  int32_t n = __riscv_vsetvl_e8m1(batch);
  vint32m4_t bias_i32v = __riscv_vmv_v_x_i32m4(bias, n);

  do {
    n = __riscv_vsetvl_e8m1(batch); batch -= n;

    vint8m1_t in_i8v = __riscv_vle8_v_i8m1(input, n); input += n;
    vint16m2_t acc_i16v = __riscv_vwsub_vx_i16m2(in_i8v, input_zero_point, n);

    vint32m4_t acc_i32v = __riscv_vwcvt_x_x_v_i32m4(acc_i16v, n);
    vint32m4_t sra_i32v = __riscv_vsra_vx_i32m4(acc_i32v, 31, n);
    vint32m4_t and_i32v = __riscv_vand_vx_i32m4(sra_i32v, multiplier_diff, n);
    vint32m4_t mult_i32v = __riscv_vxor_vx_i32m4(and_i32v, multiplier_base, n);
    acc_i32v = __riscv_vmacc_vv_i32m4(bias_i32v, acc_i32v, mult_i32v, n);

    vint16m2_t out_i16v = __riscv_vnclip_wx_i16m2(acc_i32v, 8, __RISCV_VXRM_RDN, n);
    vint8m1_t out_i8v = __riscv_vnclip_wx_i8m1(out_i16v, 0, __RISCV_VXRM_RNU, n);
    __riscv_vse8_v_i8m1(output, out_i8v, n); output += n;
  } while (batch != 0);
}
