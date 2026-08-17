// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-vrpreluc/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Léandre Le Duc
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.



#include <assert.h>
#include <riscv_vector.h>

#include "src/xnnpack/vbinary.h"

void xnn_qs8_vrpreluc_ukernel__rvv_u2v(
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

  const int32_t slope = (int32_t) *input_b - params->scalar.slope_zero_point;

  const int32_t output_zero_point = params->scalar.output_zero_point;
  const float output_min_less_zero_point =
      (int32_t) params->scalar.output_min - output_zero_point;
  const float output_max_less_zero_point =
      (int32_t) params->scalar.output_max - output_zero_point;

  // The branch is loop invariant: it depends on the broadcast slope only.
  if (slope >= 0) {
    const size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t vfpacc = __riscv_vfmul_vf_f32m2(
        __riscv_vfcvt_f_x_v_f32m2(
            __riscv_vmv_v_x_i32m2(slope, vlmax), vlmax),
        params->scalar.rprelu_positive_multiplier, vlmax);
    vfpacc =
        __riscv_vfmax_vf_f32m2(vfpacc, output_min_less_zero_point, vlmax);
    vfpacc =
        __riscv_vfmin_vf_f32m2(vfpacc, output_max_less_zero_point, vlmax);

    const vint32m2_t vacc = __riscv_vadd_vx_i32m2(
        __riscv_vfcvt_x_f_v_i32m2(vfpacc, vlmax), output_zero_point,
        vlmax);
    const vint32m2_t vout32 = vacc;
    const vint8mf2_t vout8 =
        __riscv_vncvt_x(__riscv_vncvt_x(vout32, vlmax), vlmax);

    for (size_t vl; batch > 0; batch -= vl, output += vl) {
      vl = __riscv_vsetvl_e8mf2(batch);
      __riscv_vse8_v_i8mf2(output, vout8, vl);
    }
    return;
  }

  const int32_t input_zero_point = params->scalar.input_zero_point;
  const float negative_multiplier = params->scalar.negative_multiplier;

  for (size_t vl; batch > 0; batch -= vl, input_a += vl, output += vl) {
    vl = __riscv_vsetvl_e32m2(batch);

    const vint8mf2_t va8 = __riscv_vle8_v_i8mf2(input_a, vl);
    vint32m2_t va32 = __riscv_vsext_vf4_i32m2(va8, vl);
    va32 = __riscv_vsub_vx_i32m2(va32, input_zero_point, vl);

    vint32m2_t vacc = __riscv_vmul_vx_i32m2(va32, slope, vl);

    vfloat32m2_t vfpacc = __riscv_vfmul_vf_f32m2(
        __riscv_vfcvt_f_x_v_f32m2(vacc, vl), negative_multiplier, vl);

    vfpacc = __riscv_vfmax_vf_f32m2(vfpacc, output_min_less_zero_point, vl);
    vfpacc = __riscv_vfmin_vf_f32m2(vfpacc, output_max_less_zero_point, vl);

    vacc = __riscv_vfcvt_x_f_v_i32m2(vfpacc, vl);
    const vint32m2_t vout32 =
        __riscv_vadd_vx_i32m2(vacc, output_zero_point, vl);

    // Clamped to [output_min, output_max] above, so the narrowing cannot overflow.
    const vint8mf2_t vout8 =
        __riscv_vncvt_x(__riscv_vncvt_x(vout32, vl), vl);
    __riscv_vse8_v_i8mf2(output, vout8, vl);
  }
}
