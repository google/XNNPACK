// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 SiFive, Inc.
// Copyright 2024 Microchip
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 7);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const int8_t* a4 = (const int8_t*) ((uintptr_t) a3 + a_stride);
  int8_t* c4 = (int8_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const int8_t* a5 = (const int8_t*) ((uintptr_t) a4 + a_stride);
  int8_t* c5 = (int8_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const int8_t* a6 = (const int8_t*) ((uintptr_t) a5 + a_stride);
  int8_t* c6 = (int8_t*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }

  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;

  const int32_t output_min_less_zero_point = (int32_t) params->fp32_scalar.output_min - (int32_t) params->fp32_scalar.output_zero_point;
  const int32_t output_max_less_zero_point = (int32_t) params->fp32_scalar.output_max - (int32_t) params->fp32_scalar.output_zero_point;
  const int32_t output_zero_point = params->fp32_scalar.output_zero_point;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e32m4(nc);
    }
    nc = nc - vl;

    vint32m4_t vacc0 = __riscv_vle32_v_i32m4((const int32_t*)w, vl);
    vint32m4_t vacc1 = vacc0;
    vint32m4_t vacc2 = vacc0;
    vint32m4_t vacc3 = vacc0;
    vint32m4_t vacc4 = vacc0;
    vint32m4_t vacc5 = vacc0;
    vint32m4_t vacc6 = vacc0;
 
    w = (const int32_t*) w + nr;

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) *a0++;
      const int32_t va1 = (int32_t) *a1++;
      const int32_t va2 = (int32_t) *a2++;
      const int32_t va3 = (int32_t) *a3++;
      const int32_t va4 = (int32_t) *a4++;
      const int32_t va5 = (int32_t) *a5++;
      const int32_t va6 = (int32_t) *a6++;

      const vint8m1_t vb = __riscv_vle8_v_i8m1((const int8_t*) w, vl);
      const vint16m2_t vb0 = __riscv_vsext_vf2(vb, vl);

      w = (const int8_t*) w + nr;

      vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb0, vl);
      vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, va1, vb0, vl);
      vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, va2, vb0, vl);
      vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, va3, vb0, vl);
      vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, va4, vb0, vl);
      vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, va5, vb0, vl);
      vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, va6, vb0, vl);

      k -= sizeof(int8_t);
    } while (k != 0);
 
    vfloat32m4_t vfacc0 = __riscv_vfcvt_f_x_v_f32m4(vacc0, vl);
    vfloat32m4_t vfacc1 = __riscv_vfcvt_f_x_v_f32m4(vacc1, vl);
    vfloat32m4_t vfacc2 = __riscv_vfcvt_f_x_v_f32m4(vacc2, vl);
    vfloat32m4_t vfacc3 = __riscv_vfcvt_f_x_v_f32m4(vacc3, vl);
    vfloat32m4_t vfacc4 = __riscv_vfcvt_f_x_v_f32m4(vacc4, vl);
    vfloat32m4_t vfacc5 = __riscv_vfcvt_f_x_v_f32m4(vacc5, vl);
    vfloat32m4_t vfacc6 = __riscv_vfcvt_f_x_v_f32m4(vacc6, vl);

    const vfloat32m4_t vscale = __riscv_vle32_v_f32m4((const float*) w, vl);
    vfacc0 = __riscv_vfmul_vv_f32m4(vfacc0, vscale, vl);
    vfacc1 = __riscv_vfmul_vv_f32m4(vfacc1, vscale, vl);
    vfacc2 = __riscv_vfmul_vv_f32m4(vfacc2, vscale, vl);
    vfacc3 = __riscv_vfmul_vv_f32m4(vfacc3, vscale, vl);
    vfacc4 = __riscv_vfmul_vv_f32m4(vfacc4, vscale, vl);
    vfacc5 = __riscv_vfmul_vv_f32m4(vfacc5, vscale, vl);
    vfacc6 = __riscv_vfmul_vv_f32m4(vfacc6, vscale, vl);
    w = (const float*) w + nr;

    vfacc0 = __riscv_vfmax_vf_f32m4(vfacc0, output_min_less_zero_point, vl);
    vfacc1 = __riscv_vfmax_vf_f32m4(vfacc1, output_min_less_zero_point, vl);
    vfacc2 = __riscv_vfmax_vf_f32m4(vfacc2, output_min_less_zero_point, vl);
    vfacc3 = __riscv_vfmax_vf_f32m4(vfacc3, output_min_less_zero_point, vl);
    vfacc4 = __riscv_vfmax_vf_f32m4(vfacc4, output_min_less_zero_point, vl);
    vfacc5 = __riscv_vfmax_vf_f32m4(vfacc5, output_min_less_zero_point, vl);
    vfacc6 = __riscv_vfmax_vf_f32m4(vfacc6, output_min_less_zero_point, vl);
    vfacc0 = __riscv_vfmin_vf_f32m4(vfacc0, output_max_less_zero_point, vl);
    vfacc1 = __riscv_vfmin_vf_f32m4(vfacc1, output_max_less_zero_point, vl);
    vfacc2 = __riscv_vfmin_vf_f32m4(vfacc2, output_max_less_zero_point, vl);
    vfacc3 = __riscv_vfmin_vf_f32m4(vfacc3, output_max_less_zero_point, vl);
    vfacc4 = __riscv_vfmin_vf_f32m4(vfacc4, output_max_less_zero_point, vl);
    vfacc5 = __riscv_vfmin_vf_f32m4(vfacc5, output_max_less_zero_point, vl);
    vfacc6 = __riscv_vfmin_vf_f32m4(vfacc6, output_max_less_zero_point, vl);

    vint16m2_t vout0 = __riscv_vfncvt_x(vfacc0, vl);
    vint16m2_t vout1 = __riscv_vfncvt_x(vfacc1, vl);
    vint16m2_t vout2 = __riscv_vfncvt_x(vfacc2, vl);
    vint16m2_t vout3 = __riscv_vfncvt_x(vfacc3, vl);
    vint16m2_t vout4 = __riscv_vfncvt_x(vfacc4, vl);
    vint16m2_t vout5 = __riscv_vfncvt_x(vfacc5, vl);
    vint16m2_t vout6 = __riscv_vfncvt_x(vfacc6, vl);

    vout0 = __riscv_vadd_vx_i16m2(vout0, (int16_t) output_zero_point, vl);
    vout1 = __riscv_vadd_vx_i16m2(vout1, (int16_t) output_zero_point, vl);
    vout2 = __riscv_vadd_vx_i16m2(vout2, (int16_t) output_zero_point, vl);
    vout3 = __riscv_vadd_vx_i16m2(vout3, (int16_t) output_zero_point, vl);
    vout4 = __riscv_vadd_vx_i16m2(vout4, (int16_t) output_zero_point, vl);
    vout5 = __riscv_vadd_vx_i16m2(vout5, (int16_t) output_zero_point, vl);
    vout6 = __riscv_vadd_vx_i16m2(vout6, (int16_t) output_zero_point, vl);

    vint8m1_t vout80 = __riscv_vncvt_x_x_w_i8m1(vout0, vl);
    vint8m1_t vout81 = __riscv_vncvt_x_x_w_i8m1(vout1, vl);
    vint8m1_t vout82 = __riscv_vncvt_x_x_w_i8m1(vout2, vl);
    vint8m1_t vout83 = __riscv_vncvt_x_x_w_i8m1(vout3, vl);
    vint8m1_t vout84 = __riscv_vncvt_x_x_w_i8m1(vout4, vl);
    vint8m1_t vout85 = __riscv_vncvt_x_x_w_i8m1(vout5, vl);
    vint8m1_t vout86 = __riscv_vncvt_x_x_w_i8m1(vout6, vl);

    __riscv_vse8_v_i8m1(c0, vout80, vl);
    c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
    __riscv_vse8_v_i8m1(c1, vout81, vl);
    c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
    __riscv_vse8_v_i8m1(c2, vout82, vl);
    c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
    __riscv_vse8_v_i8m1(c3, vout83, vl);
    c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
    __riscv_vse8_v_i8m1(c4, vout84, vl);
    c4 = (int8_t*) ((uintptr_t) c4 + cn_stride);
    __riscv_vse8_v_i8m1(c5, vout85, vl);
    c5 = (int8_t*) ((uintptr_t) c5 + cn_stride);
    __riscv_vse8_v_i8m1(c6, vout86, vl);
    c6 = (int8_t*) ((uintptr_t) c6 + cn_stride);

    a0 = (const int8_t*) ((uintptr_t) a0 - kc);
    a1 = (const int8_t*) ((uintptr_t) a1 - kc);
    a2 = (const int8_t*) ((uintptr_t) a2 - kc);
    a3 = (const int8_t*) ((uintptr_t) a3 - kc);
    a4 = (const int8_t*) ((uintptr_t) a4 - kc);
    a5 = (const int8_t*) ((uintptr_t) a5 - kc);
    a6 = (const int8_t*) ((uintptr_t) a6 - kc);

  } while (nc != 0);
}
