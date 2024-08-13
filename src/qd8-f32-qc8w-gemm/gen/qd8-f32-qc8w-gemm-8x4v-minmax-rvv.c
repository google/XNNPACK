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

void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 8);
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
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const int8_t* a4 = (const int8_t*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const int8_t* a5 = (const int8_t*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const int8_t* a6 = (const int8_t*) ((uintptr_t) a5 + a_stride);
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }
  const int8_t* a7 = (const int8_t*) ((uintptr_t) a6 + a_stride);
  float* c7 = (float*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 8) {
    a7 = a6;
    c7 = c6;
  }

  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;
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
    const int32_t vinput_zero_point4 = quantization_params[4].zero_point;
    const int32_t vinput_zero_point5 = quantization_params[5].zero_point;
    const int32_t vinput_zero_point6 = quantization_params[6].zero_point;
    const int32_t vinput_zero_point7 = quantization_params[7].zero_point;
    vint32m4_t vacc0 = __riscv_vmul_vx_i32m4(vksum, vinput_zero_point0, vl);
    vint32m4_t vacc1 = __riscv_vmul_vx_i32m4(vksum, vinput_zero_point1, vl);
    vint32m4_t vacc2 = __riscv_vmul_vx_i32m4(vksum, vinput_zero_point2, vl);
    vint32m4_t vacc3 = __riscv_vmul_vx_i32m4(vksum, vinput_zero_point3, vl);
    vint32m4_t vacc4 = __riscv_vmul_vx_i32m4(vksum, vinput_zero_point4, vl);
    vint32m4_t vacc5 = __riscv_vmul_vx_i32m4(vksum, vinput_zero_point5, vl);
    vint32m4_t vacc6 = __riscv_vmul_vx_i32m4(vksum, vinput_zero_point6, vl);
    vint32m4_t vacc7 = __riscv_vmul_vx_i32m4(vksum, vinput_zero_point7, vl);

    size_t k = kc;
    do {
      const int8_t va0 = *a0++;
      const int8_t va1 = *a1++;
      const int8_t va2 = *a2++;
      const int8_t va3 = *a3++;
      const int8_t va4 = *a4++;
      const int8_t va5 = *a5++;
      const int8_t va6 = *a6++;
      const int8_t va7 = *a7++;
      const vint8m1_t vb = __riscv_vle8_v_i8m1((const int8_t*) w, vl);
      w = (const int8_t*) w + nr;

      vint16m2_t va0b = __riscv_vwmul_vx_i16m2(vb, va0, vl);
      vacc0 = __riscv_vwadd_wv_i32m4(vacc0, va0b, vl);
      vint16m2_t va1b = __riscv_vwmul_vx_i16m2(vb, va1, vl);
      vacc1 = __riscv_vwadd_wv_i32m4(vacc1, va1b, vl);
      vint16m2_t va2b = __riscv_vwmul_vx_i16m2(vb, va2, vl);
      vacc2 = __riscv_vwadd_wv_i32m4(vacc2, va2b, vl);
      vint16m2_t va3b = __riscv_vwmul_vx_i16m2(vb, va3, vl);
      vacc3 = __riscv_vwadd_wv_i32m4(vacc3, va3b, vl);
      vint16m2_t va4b = __riscv_vwmul_vx_i16m2(vb, va4, vl);
      vacc4 = __riscv_vwadd_wv_i32m4(vacc4, va4b, vl);
      vint16m2_t va5b = __riscv_vwmul_vx_i16m2(vb, va5, vl);
      vacc5 = __riscv_vwadd_wv_i32m4(vacc5, va5b, vl);
      vint16m2_t va6b = __riscv_vwmul_vx_i16m2(vb, va6, vl);
      vacc6 = __riscv_vwadd_wv_i32m4(vacc6, va6b, vl);
      vint16m2_t va7b = __riscv_vwmul_vx_i16m2(vb, va7, vl);
      vacc7 = __riscv_vwadd_wv_i32m4(vacc7, va7b, vl);

      k -= sizeof(int8_t);
    } while (k != 0);
    // i32 -> f32
    vfloat32m4_t vout0 = __riscv_vfcvt_f_x_v_f32m4(vacc0, vl);
    vfloat32m4_t vout1 = __riscv_vfcvt_f_x_v_f32m4(vacc1, vl);
    vfloat32m4_t vout2 = __riscv_vfcvt_f_x_v_f32m4(vacc2, vl);
    vfloat32m4_t vout3 = __riscv_vfcvt_f_x_v_f32m4(vacc3, vl);
    vfloat32m4_t vout4 = __riscv_vfcvt_f_x_v_f32m4(vacc4, vl);
    vfloat32m4_t vout5 = __riscv_vfcvt_f_x_v_f32m4(vacc5, vl);
    vfloat32m4_t vout6 = __riscv_vfcvt_f_x_v_f32m4(vacc6, vl);
    vfloat32m4_t vout7 = __riscv_vfcvt_f_x_v_f32m4(vacc7, vl);

    // vout * input_scale
    const float vinput_scale0 = quantization_params[0].inv_scale;
    const float vinput_scale1 = quantization_params[1].inv_scale;
    const float vinput_scale2 = quantization_params[2].inv_scale;
    const float vinput_scale3 = quantization_params[3].inv_scale;
    const float vinput_scale4 = quantization_params[4].inv_scale;
    const float vinput_scale5 = quantization_params[5].inv_scale;
    const float vinput_scale6 = quantization_params[6].inv_scale;
    const float vinput_scale7 = quantization_params[7].inv_scale;
    vout0 = __riscv_vfmul_vf_f32m4(vout0, vinput_scale0, vl);
    vout1 = __riscv_vfmul_vf_f32m4(vout1, vinput_scale1, vl);
    vout2 = __riscv_vfmul_vf_f32m4(vout2, vinput_scale2, vl);
    vout3 = __riscv_vfmul_vf_f32m4(vout3, vinput_scale3, vl);
    vout4 = __riscv_vfmul_vf_f32m4(vout4, vinput_scale4, vl);
    vout5 = __riscv_vfmul_vf_f32m4(vout5, vinput_scale5, vl);
    vout6 = __riscv_vfmul_vf_f32m4(vout6, vinput_scale6, vl);
    vout7 = __riscv_vfmul_vf_f32m4(vout7, vinput_scale7, vl);

    const vfloat32m4_t vfilter_output_scale = __riscv_vle32_v_f32m4((const float*) w, vl);
    w = (const float*) w + nr;
    vout0 = __riscv_vfmul_vv_f32m4(vout0, vfilter_output_scale, vl);
    vout1 = __riscv_vfmul_vv_f32m4(vout1, vfilter_output_scale, vl);
    vout2 = __riscv_vfmul_vv_f32m4(vout2, vfilter_output_scale, vl);
    vout3 = __riscv_vfmul_vv_f32m4(vout3, vfilter_output_scale, vl);
    vout4 = __riscv_vfmul_vv_f32m4(vout4, vfilter_output_scale, vl);
    vout5 = __riscv_vfmul_vv_f32m4(vout5, vfilter_output_scale, vl);
    vout6 = __riscv_vfmul_vv_f32m4(vout6, vfilter_output_scale, vl);
    vout7 = __riscv_vfmul_vv_f32m4(vout7, vfilter_output_scale, vl);

    const vfloat32m4_t vbias =  __riscv_vle32_v_f32m4((const float*) w, vl);
    w = (const float*) w + nr;
    vout0 = __riscv_vfadd_vv_f32m4(vout0, vbias, vl);
    vout1 = __riscv_vfadd_vv_f32m4(vout1, vbias, vl);
    vout2 = __riscv_vfadd_vv_f32m4(vout2, vbias, vl);
    vout3 = __riscv_vfadd_vv_f32m4(vout3, vbias, vl);
    vout4 = __riscv_vfadd_vv_f32m4(vout4, vbias, vl);
    vout5 = __riscv_vfadd_vv_f32m4(vout5, vbias, vl);
    vout6 = __riscv_vfadd_vv_f32m4(vout6, vbias, vl);
    vout7 = __riscv_vfadd_vv_f32m4(vout7, vbias, vl);

    const float vmin = params->scalar.min;
    vout0 = __riscv_vfmax_vf_f32m4(vout0, vmin, vl);
    vout1 = __riscv_vfmax_vf_f32m4(vout1, vmin, vl);
    vout2 = __riscv_vfmax_vf_f32m4(vout2, vmin, vl);
    vout3 = __riscv_vfmax_vf_f32m4(vout3, vmin, vl);
    vout4 = __riscv_vfmax_vf_f32m4(vout4, vmin, vl);
    vout5 = __riscv_vfmax_vf_f32m4(vout5, vmin, vl);
    vout6 = __riscv_vfmax_vf_f32m4(vout6, vmin, vl);
    vout7 = __riscv_vfmax_vf_f32m4(vout7, vmin, vl);
    const float vmax = params->scalar.max;
    vout0 = __riscv_vfmin_vf_f32m4(vout0, vmax, vl);
    vout1 = __riscv_vfmin_vf_f32m4(vout1, vmax, vl);
    vout2 = __riscv_vfmin_vf_f32m4(vout2, vmax, vl);
    vout3 = __riscv_vfmin_vf_f32m4(vout3, vmax, vl);
    vout4 = __riscv_vfmin_vf_f32m4(vout4, vmax, vl);
    vout5 = __riscv_vfmin_vf_f32m4(vout5, vmax, vl);
    vout6 = __riscv_vfmin_vf_f32m4(vout6, vmax, vl);
    vout7 = __riscv_vfmin_vf_f32m4(vout7, vmax, vl);

    // store 8 x vl results to c
    __riscv_vse32_v_f32m4(c0, vout0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);
    __riscv_vse32_v_f32m4(c1, vout1, vl);
    c1 = (float*) ((uintptr_t) c1 + cn_stride);
    __riscv_vse32_v_f32m4(c2, vout2, vl);
    c2 = (float*) ((uintptr_t) c2 + cn_stride);
    __riscv_vse32_v_f32m4(c3, vout3, vl);
    c3 = (float*) ((uintptr_t) c3 + cn_stride);
    __riscv_vse32_v_f32m4(c4, vout4, vl);
    c4 = (float*) ((uintptr_t) c4 + cn_stride);
    __riscv_vse32_v_f32m4(c5, vout5, vl);
    c5 = (float*) ((uintptr_t) c5 + cn_stride);
    __riscv_vse32_v_f32m4(c6, vout6, vl);
    c6 = (float*) ((uintptr_t) c6 + cn_stride);
    __riscv_vse32_v_f32m4(c7, vout7, vl);
    c7 = (float*) ((uintptr_t) c7 + cn_stride);

    a0 = (const int8_t*) ((uintptr_t) a0 - kc);
    a1 = (const int8_t*) ((uintptr_t) a1 - kc);
    a2 = (const int8_t*) ((uintptr_t) a2 - kc);
    a3 = (const int8_t*) ((uintptr_t) a3 - kc);
    a4 = (const int8_t*) ((uintptr_t) a4 - kc);
    a5 = (const int8_t*) ((uintptr_t) a5 - kc);
    a6 = (const int8_t*) ((uintptr_t) a6 - kc);
    a7 = (const int8_t*) ((uintptr_t) a7 - kc);
  } while (nc != 0);
}
