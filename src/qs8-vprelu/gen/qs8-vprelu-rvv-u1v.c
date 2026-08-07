// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-vprelu/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Léandre Le Duc
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <riscv_vector.h>
#include <stdint.h>

#include "src/xnnpack/vbinary.h"

void xnn_qs8_vprelu_ukernel__rvv_u1v(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_vprelu_scalar_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t input_zero_point = params->scalar.input_zero_point;
  const int32_t slope_zero_point = params->scalar.slope_zero_point;

  const float positive_multiplier = params->scalar.positive_multiplier;
  const float negative_multiplier = params->scalar.negative_multiplier;

  const int32_t output_zero_point = params->scalar.output_zero_point;
  const float output_min_less_zero_point =
      (int32_t)params->scalar.output_min - output_zero_point;
  const float output_max_less_zero_point =
      (int32_t)params->scalar.output_max - output_zero_point;

  for (size_t vl; batch > 0;
       batch -= vl, input_a += vl, input_b += vl, output += vl) {
    vl = __riscv_vsetvl_e32m1(batch);

    const vint8mf4_t va8 = __riscv_vle8_v_i8mf4(input_a, vl);
    const vint8mf4_t vb8 = __riscv_vle8_v_i8mf4(input_b, vl);
    vint32m1_t va32 = __riscv_vsext_vf4_i32m1(va8, vl);
    vint32m1_t vb32 = __riscv_vsext_vf4_i32m1(vb8, vl);
    va32 = __riscv_vsub_vx_i32m1(va32, input_zero_point, vl);
    vb32 = __riscv_vsub_vx_i32m1(vb32, slope_zero_point, vl);

    const vbool32_t ma =
        __riscv_vmslt_vx_i32m1_b32(va32, 0, vl);

    vint32m1_t vacc = __riscv_vmul_vv_i32m1_mu(ma, va32, va32, vb32, vl);
    const vfloat32m1_t vaccf = __riscv_vfcvt_f_x_v_f32m1(vacc, vl);

    vfloat32m1_t vfpacc =
        __riscv_vfmul_vf_f32m1(vaccf, positive_multiplier, vl);
    vfpacc = __riscv_vfmul_vf_f32m1_mu(
        ma, vfpacc, vaccf, negative_multiplier, vl);

    vfpacc = __riscv_vfmax_vf_f32m1(vfpacc, output_min_less_zero_point, vl);
    vfpacc = __riscv_vfmin_vf_f32m1(vfpacc, output_max_less_zero_point, vl);

    vacc = __riscv_vfcvt_x_f_v_i32m1(vfpacc, vl);
    const vint32m1_t vout32 =
        __riscv_vadd_vx_i32m1(vacc, output_zero_point, vl);

    // Clamped to [output_min, output_max] above, so the narrowing cannot overflow.
    const vint8mf4_t vout8 =
        __riscv_vncvt_x(__riscv_vncvt_x(vout32, vl), vl);
    __riscv_vse8_v_i8mf4(output, vout8, vl);
  }
}
