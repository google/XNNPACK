// Auto-generated file. Do not edit!
//   Template: src/qs8-vadd/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies, inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/vbinary.h"


void xnn_qs8_vadd_minmax_ukernel__rvv_u1v(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const struct xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t bias = params->scalar.bias;
  const int32_t a_multiplier = params->scalar.a_multiplier;
  const int32_t b_multiplier = params->scalar.b_multiplier;
  const uint32_t shift = params->scalar.shift;
  const int32_t output_min = params->scalar.output_min;
  const int32_t output_max = params->scalar.output_max;
  const int32_t output_zero_point = params->scalar.output_zero_point;

  do {
    int32_t n = __riscv_vsetvl_e8m1(batch); batch -= n;

    vint8m1_t in_a_i8v = __riscv_vle8_v_i8m1(input_a, n); input_a += n;
    vint8m1_t in_b_i8v = __riscv_vle8_v_i8m1(input_b, n); input_b += n;
    vint16m2_t a_i16v = __riscv_vwcvt_x_x_v_i16m2(in_a_i8v, n);
    vint16m2_t b_i16v = __riscv_vwcvt_x_x_v_i16m2(in_b_i8v, n);
    vint32m4_t a_i32v = __riscv_vwcvt_x_x_v_i32m4(a_i16v, n);
    vint32m4_t b_i32v = __riscv_vwcvt_x_x_v_i32m4(b_i16v, n);
    a_i32v = __riscv_vmul_vx_i32m4(a_i32v, a_multiplier, n);
    b_i32v = __riscv_vmul_vx_i32m4(b_i32v, b_multiplier, n);
    vint32m4_t acc_i32v = __riscv_vadd_vx_i32m4(a_i32v, bias, n);
    acc_i32v = __riscv_vadd_vv_i32m4(acc_i32v, b_i32v, n);
    vint32m4_t out_i32v = __riscv_vsra_vx_i32m4(acc_i32v, shift, n);
    out_i32v = __riscv_vadd_vx_i32m4(out_i32v, output_zero_point, n);
    out_i32v = __riscv_vmax_vx_i32m4(out_i32v, output_min, n);
    out_i32v = __riscv_vmin_vx_i32m4(out_i32v, output_max, n);
    vint16m2_t out_i16v = __riscv_vncvt_x_x_w_i16m2(out_i32v, n);
    vint8m1_t out_i8v = __riscv_vncvt_x_x_w_i8m1(out_i16v, n);
    __riscv_vse8_v_i8m1(output, out_i8v, n); output += n;
  } while (batch != 0);
}
