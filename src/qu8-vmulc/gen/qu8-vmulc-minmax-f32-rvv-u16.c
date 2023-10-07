// Auto-generated file. Do not edit!
//   Template: src/qs8-vmulc/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <riscv_vector.h>
#include <xnnpack/vbinary.h>


void xnn_qu8_vmulc_minmax_fp32_ukernel__rvv_u16(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t a_zero_point = params->fp32_scalar.a_zero_point;
  const float scale = params->fp32_scalar.scale;
  const float output_min_less_zero_point = params->fp32_scalar.output_min_less_zero_point;
  const float output_max_less_zero_point = params->fp32_scalar.output_max_less_zero_point;
  const float magic_bias = params->fp32_scalar.magic_bias;
  const int32_t magic_bias_less_output_zero_point = params->fp32_scalar.magic_bias_less_output_zero_point;
  const int32_t vb = (int32_t) *input_b - params->fp32_scalar.b_zero_point;

  do {
    int32_t vl = __riscv_vsetvl_e8m1(batch);

    vuint8m1_t in_a_u8v = __riscv_vle8_v_u8m1(input_a, vl);
    vuint16m2_t a_u16v = __riscv_vwsubu_vx_u16m2(in_a_u8v, a_zero_point, vl);
    vint16m2_t a_i16v = __riscv_vreinterpret_v_u16m2_i16m2(a_u16v);

    vint32m4_t acc_i32v = __riscv_vwmul_vx_i32m4(a_i16v, vb, vl);
    vfloat32m4_t acc_f32v = __riscv_vfcvt_f_x_v_f32m4(acc_i32v, vl);
    acc_f32v = __riscv_vfmul_vf_f32m4(acc_f32v, scale, vl);
    acc_f32v = __riscv_vfmin_vf_f32m4(__riscv_vfmax_vf_f32m4(acc_f32v, output_min_less_zero_point, vl), output_max_less_zero_point, vl);
    acc_f32v = __riscv_vfadd_vf_f32m4(acc_f32v, magic_bias, vl);

    vuint32m4_t out_u32v = __riscv_vfcvt_xu_f_v_u32m4(acc_f32v, vl);
    out_u32v = __riscv_vsub_vx_u32m4(out_u32v, magic_bias_less_output_zero_point, vl);
    vuint16m2_t out_u16v = __riscv_vncvt_x_x_w_u16m2(out_u32v, vl);
    vuint8m1_t out_u8v = __riscv_vncvt_x_x_w_u8m1(out_u16v, vl);
    __riscv_vse8_v_u8m1(output, out_u8v, vl);

    input_a += vl;
    input_b += vl;
    output += vl;
    batch -= vl;
  } while (batch != 0);
}
