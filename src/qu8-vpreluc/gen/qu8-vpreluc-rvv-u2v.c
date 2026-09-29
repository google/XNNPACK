// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-vpreluc/rvv.c.in
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

void xnn_qu8_vpreluc_ukernel__rvv_u2v(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qs8_vprelu_scalar_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t input_zero_point = params->scalar.input_zero_point;
  const int32_t slope = (int32_t)*input_b - params->scalar.slope_zero_point;

  const float positive_multiplier = params->scalar.positive_multiplier;
  const float negative_multiplier = params->scalar.negative_multiplier;

  const int32_t output_zero_point = params->scalar.output_zero_point;
  const float output_min_less_zero_point =
      (int32_t)params->scalar.output_min - output_zero_point;
  const float output_max_less_zero_point =
      (int32_t)params->scalar.output_max - output_zero_point;

  for (size_t vl; batch > 0; batch -= vl, input_a += vl, output += vl) {
    vl = __riscv_vsetvl_e32m2(batch);

    const vuint8mf2_t va8 = __riscv_vle8_v_u8mf2(input_a, vl);
    vint32m2_t va32 = __riscv_vreinterpret_v_u32m2_i32m2(
        __riscv_vzext_vf4_u32m2(va8, vl));
    va32 = __riscv_vsub_vx_i32m2(va32, input_zero_point, vl);

    const vbool16_t ma =
        __riscv_vmslt_vx_i32m2_b16(va32, 0, vl);

    vint32m2_t vacc = __riscv_vmul_vx_i32m2_mu(ma, va32, va32, slope, vl);
    const vfloat32m2_t vaccf = __riscv_vfcvt_f_x_v_f32m2(vacc, vl);

    vfloat32m2_t vfpacc =
        __riscv_vfmul_vf_f32m2(vaccf, positive_multiplier, vl);
    vfpacc = __riscv_vfmul_vf_f32m2_mu(
        ma, vfpacc, vaccf, negative_multiplier, vl);

    vfpacc = __riscv_vfmax_vf_f32m2(vfpacc, output_min_less_zero_point, vl);
    vfpacc = __riscv_vfmin_vf_f32m2(vfpacc, output_max_less_zero_point, vl);

    vacc = __riscv_vfcvt_x_f_v_i32m2(vfpacc, vl);
    const vuint32m2_t vout32 = __riscv_vreinterpret_v_i32m2_u32m2(
        __riscv_vadd_vx_i32m2(vacc, output_zero_point, vl));

    // Clamped to [output_min, output_max] above, so the narrowing cannot overflow.
    const vuint8mf2_t vout8 =
        __riscv_vncvt_x(__riscv_vncvt_x(vout32, vl), vl);
    __riscv_vse8_v_u8mf2(output, vout8, vl);
  }
}
