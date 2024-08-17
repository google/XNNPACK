// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 SiFive, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/gemm.h"
#include "xnnpack/math.h"

void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;
  kc = round_up_po2(kc, 2);
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e32m4(nc);
    }
    nc = nc - vl;

    vint32m4_t vksum = __riscv_vle32_v_i32m4((const int32_t*)w, vl);
    w = (const int32_t*) w + nr;
    const int32_t vinput_zero_point0 = quantization_params[0].zero_point;
    const int32_t vinput_zero_point1 = quantization_params[1].zero_point;
    const int32_t vinput_zero_point2 = quantization_params[2].zero_point;
    const int32_t vinput_zero_point3 = quantization_params[3].zero_point;
    vint32m4_t vacc0 = __riscv_vmul_vx_i32m4(vksum, vinput_zero_point0, vl);
    vint32m4_t vacc1 = __riscv_vmul_vx_i32m4(vksum, vinput_zero_point1, vl);
    vint32m4_t vacc2 = __riscv_vmul_vx_i32m4(vksum, vinput_zero_point2, vl);
    vint32m4_t vacc3 = __riscv_vmul_vx_i32m4(vksum, vinput_zero_point3, vl);

    size_t k = kc;
    for (; k >= 2 * sizeof(uint8_t); k -= 2 * sizeof(uint8_t)) {
      const int8_t va0c0 = a0[0];
      const int8_t va0c1 = a0[1];
      a0 += 2;
      const int8_t va1c0 = a1[0];
      const int8_t va1c1 = a1[1];
      a1 += 2;
      const int8_t va2c0 = a2[0];
      const int8_t va2c1 = a2[1];
      a2 += 2;
      const int8_t va3c0 = a3[0];
      const int8_t va3c1 = a3[1];
      a3 += 2;
      const vint8m1_t vbi = __riscv_vle8_v_i8m1((const int8_t*) w, vl);
      w = (const uint8_t*) w + nr;
      const vint8m1_t vbc0 = __riscv_vsll_vx_i8m1(vbi, 4, vl);
      const vint8m1_t vbc1 = __riscv_vand_vx_i8m1(vbi, 0xF0, vl);

      vint16m2_t va0bc0 = __riscv_vwmul_vx_i16m2(vbc0, va0c0, vl);
      vacc0 = __riscv_vwadd_wv_i32m4(vacc0, va0bc0, vl);
      vint16m2_t va0bc1 = __riscv_vwmul_vx_i16m2(vbc1, va0c1, vl);
      vacc0 = __riscv_vwadd_wv_i32m4(vacc0, va0bc1, vl);
      vint16m2_t va1bc0 = __riscv_vwmul_vx_i16m2(vbc0, va1c0, vl);
      vacc1 = __riscv_vwadd_wv_i32m4(vacc1, va1bc0, vl);
      vint16m2_t va1bc1 = __riscv_vwmul_vx_i16m2(vbc1, va1c1, vl);
      vacc1 = __riscv_vwadd_wv_i32m4(vacc1, va1bc1, vl);
      vint16m2_t va2bc0 = __riscv_vwmul_vx_i16m2(vbc0, va2c0, vl);
      vacc2 = __riscv_vwadd_wv_i32m4(vacc2, va2bc0, vl);
      vint16m2_t va2bc1 = __riscv_vwmul_vx_i16m2(vbc1, va2c1, vl);
      vacc2 = __riscv_vwadd_wv_i32m4(vacc2, va2bc1, vl);
      vint16m2_t va3bc0 = __riscv_vwmul_vx_i16m2(vbc0, va3c0, vl);
      vacc3 = __riscv_vwadd_wv_i32m4(vacc3, va3bc0, vl);
      vint16m2_t va3bc1 = __riscv_vwmul_vx_i16m2(vbc1, va3c1, vl);
      vacc3 = __riscv_vwadd_wv_i32m4(vacc3, va3bc1, vl);
    }
    vacc0 = __riscv_vsra_vx_i32m4(vacc0, 4, vl);
    vacc1 = __riscv_vsra_vx_i32m4(vacc1, 4, vl);
    vacc2 = __riscv_vsra_vx_i32m4(vacc2, 4, vl);
    vacc3 = __riscv_vsra_vx_i32m4(vacc3, 4, vl);
    // i32 -> f32
    vfloat32m4_t vout0 = __riscv_vfcvt_f_x_v_f32m4(vacc0, vl);
    vfloat32m4_t vout1 = __riscv_vfcvt_f_x_v_f32m4(vacc1, vl);
    vfloat32m4_t vout2 = __riscv_vfcvt_f_x_v_f32m4(vacc2, vl);
    vfloat32m4_t vout3 = __riscv_vfcvt_f_x_v_f32m4(vacc3, vl);

    // vout * input_scale
    const float vinput_scale0 = quantization_params[0].inv_scale;
    const float vinput_scale1 = quantization_params[1].inv_scale;
    const float vinput_scale2 = quantization_params[2].inv_scale;
    const float vinput_scale3 = quantization_params[3].inv_scale;
    vout0 = __riscv_vfmul_vf_f32m4(vout0, vinput_scale0, vl);
    vout1 = __riscv_vfmul_vf_f32m4(vout1, vinput_scale1, vl);
    vout2 = __riscv_vfmul_vf_f32m4(vout2, vinput_scale2, vl);
    vout3 = __riscv_vfmul_vf_f32m4(vout3, vinput_scale3, vl);

    const vfloat32m4_t vfilter_output_scale = __riscv_vle32_v_f32m4((const float*) w, vl);
    w = (const float*) w + nr;
    vout0 = __riscv_vfmul_vv_f32m4(vout0, vfilter_output_scale, vl);
    vout1 = __riscv_vfmul_vv_f32m4(vout1, vfilter_output_scale, vl);
    vout2 = __riscv_vfmul_vv_f32m4(vout2, vfilter_output_scale, vl);
    vout3 = __riscv_vfmul_vv_f32m4(vout3, vfilter_output_scale, vl);

    const vfloat32m4_t vbias =  __riscv_vle32_v_f32m4((const float*) w, vl);
    w = (const float*) w + nr;
    vout0 = __riscv_vfadd_vv_f32m4(vout0, vbias, vl);
    vout1 = __riscv_vfadd_vv_f32m4(vout1, vbias, vl);
    vout2 = __riscv_vfadd_vv_f32m4(vout2, vbias, vl);
    vout3 = __riscv_vfadd_vv_f32m4(vout3, vbias, vl);

    const float vmin = params->scalar.min;
    vout0 = __riscv_vfmax_vf_f32m4(vout0, vmin, vl);
    vout1 = __riscv_vfmax_vf_f32m4(vout1, vmin, vl);
    vout2 = __riscv_vfmax_vf_f32m4(vout2, vmin, vl);
    vout3 = __riscv_vfmax_vf_f32m4(vout3, vmin, vl);
    const float vmax = params->scalar.max;
    vout0 = __riscv_vfmin_vf_f32m4(vout0, vmax, vl);
    vout1 = __riscv_vfmin_vf_f32m4(vout1, vmax, vl);
    vout2 = __riscv_vfmin_vf_f32m4(vout2, vmax, vl);
    vout3 = __riscv_vfmin_vf_f32m4(vout3, vmax, vl);

    // store 4 x vl results to c
    __riscv_vse32_v_f32m4(c0, vout0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);
    __riscv_vse32_v_f32m4(c1, vout1, vl);
    c1 = (float*) ((uintptr_t) c1 + cn_stride);
    __riscv_vse32_v_f32m4(c2, vout2, vl);
    c2 = (float*) ((uintptr_t) c2 + cn_stride);
    __riscv_vse32_v_f32m4(c3, vout3, vl);
    c3 = (float*) ((uintptr_t) c3 + cn_stride);

    a0 = (const int8_t*) ((uintptr_t) a0 - kc);
    a1 = (const int8_t*) ((uintptr_t) a1 - kc);
    a2 = (const int8_t*) ((uintptr_t) a2 - kc);
    a3 = (const int8_t*) ((uintptr_t) a3 - kc);
  } while (nc != 0);
}
