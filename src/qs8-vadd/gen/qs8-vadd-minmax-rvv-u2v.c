// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-vadd/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Imagination Technologies, inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <riscv_vector.h>
#include "src/xnnpack/vbinary.h"


void xnn_qs8_vadd_minmax_ukernel__rvv_u2v(
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
    size_t vl = __riscv_vsetvl_e8m2(batch); batch -= vl;

    vint8m2_t in_a_i8v = __riscv_vle8_v_i8m2(input_a, vl); input_a += vl;
    vint8m2_t in_b_i8v = __riscv_vle8_v_i8m2(input_b, vl); input_b += vl;
    vint32m8_t a_i32v = __riscv_vsext_vf4_i32m8(in_a_i8v, vl);
    vint32m8_t b_i32v = __riscv_vsext_vf4_i32m8(in_b_i8v, vl);

    a_i32v = __riscv_vmul_vx_i32m8(a_i32v, a_multiplier, vl);
    b_i32v = __riscv_vmul_vx_i32m8(b_i32v, b_multiplier, vl);
    vint32m8_t out_i32v = __riscv_vadd_vx_i32m8(a_i32v, bias, vl);
    out_i32v = __riscv_vadd_vv_i32m8(out_i32v, b_i32v, vl);
    out_i32v = __riscv_vsra_vx_i32m8(out_i32v, shift, vl);
    out_i32v = __riscv_vadd_vx_i32m8(out_i32v, output_zero_point, vl);

    out_i32v = __riscv_vmax_vx_i32m8(out_i32v, output_min, vl);
    out_i32v = __riscv_vmin_vx_i32m8(out_i32v, output_max, vl);

    vint16m4_t out_i16v = __riscv_vncvt_x_x_w_i16m4(out_i32v, vl);
    vint8m2_t out_i8v = __riscv_vncvt_x_x_w_i8m2(out_i16v, vl);
    __riscv_vse8_v_i8m2(output, out_i8v, vl); output += vl;
  } while (batch != 0);
}
