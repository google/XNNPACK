// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Microchip, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "src/xnnpack/math.h"
#include "src/xnnpack/igemm.h"

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
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
    w = (const void*) ((const int32_t*) w + nr);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      const int8_t* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      const int8_t* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) *a0++;
        const int32_t va1 = (int32_t) *a1++;
        const int32_t va2 = (int32_t) *a2++;
        const int32_t va3 = (int32_t) *a3++;

        const vint8m1_t vb = __riscv_vle8_v_i8m1((const int8_t*) w, vl);
        const vint16m2_t vb0 = __riscv_vsext_vf2(vb, vl);

        w = (const void*) ((const int8_t*) w + nr);

        vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb0, vl);
        vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, va1, vb0, vl);
        vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, va2, vb0, vl);
        vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, va3, vb0, vl);

        k -= sizeof(int8_t);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);

    vfloat32m4_t vfpacc0 = __riscv_vfcvt_f_x_v_f32m4(vacc0, vl);
    vfloat32m4_t vfpacc1 = __riscv_vfcvt_f_x_v_f32m4(vacc1, vl);
    vfloat32m4_t vfpacc2 = __riscv_vfcvt_f_x_v_f32m4(vacc2, vl);
    vfloat32m4_t vfpacc3 = __riscv_vfcvt_f_x_v_f32m4(vacc3, vl);

    const vfloat32m4_t vscale = __riscv_vle32_v_f32m4((const float*) w, vl);
    vfpacc0 = __riscv_vfmul_vv_f32m4(vfpacc0, vscale, vl);
    vfpacc1 = __riscv_vfmul_vv_f32m4(vfpacc1, vscale, vl);
    vfpacc2 = __riscv_vfmul_vv_f32m4(vfpacc2, vscale, vl);
    vfpacc3 = __riscv_vfmul_vv_f32m4(vfpacc3, vscale, vl);

    w = (const void*) ((const float*) w + nr);

    vfpacc0 = __riscv_vfmax_vf_f32m4(vfpacc0, output_min_less_zero_point, vl);
    vfpacc1 = __riscv_vfmax_vf_f32m4(vfpacc1, output_min_less_zero_point, vl);
    vfpacc2 = __riscv_vfmax_vf_f32m4(vfpacc2, output_min_less_zero_point, vl);
    vfpacc3 = __riscv_vfmax_vf_f32m4(vfpacc3, output_min_less_zero_point, vl);
    vfpacc0 = __riscv_vfmin_vf_f32m4(vfpacc0, output_max_less_zero_point, vl);
    vfpacc1 = __riscv_vfmin_vf_f32m4(vfpacc1, output_max_less_zero_point, vl);
    vfpacc2 = __riscv_vfmin_vf_f32m4(vfpacc2, output_max_less_zero_point, vl);
    vfpacc3 = __riscv_vfmin_vf_f32m4(vfpacc3, output_max_less_zero_point, vl);

    vint16m2_t vout0 = __riscv_vfncvt_x(vfpacc0, vl);
    vint16m2_t vout1 = __riscv_vfncvt_x(vfpacc1, vl);
    vint16m2_t vout2 = __riscv_vfncvt_x(vfpacc2, vl);
    vint16m2_t vout3 = __riscv_vfncvt_x(vfpacc3, vl);

    vout0 = __riscv_vadd_vx_i16m2(vout0, (int16_t) output_zero_point, vl);
    vout1 = __riscv_vadd_vx_i16m2(vout1, (int16_t) output_zero_point, vl);
    vout2 = __riscv_vadd_vx_i16m2(vout2, (int16_t) output_zero_point, vl);
    vout3 = __riscv_vadd_vx_i16m2(vout3, (int16_t) output_zero_point, vl);

    vint8m1_t vout80 = __riscv_vncvt_x_x_w_i8m1(vout0, vl);
    vint8m1_t vout81 = __riscv_vncvt_x_x_w_i8m1(vout1, vl);
    vint8m1_t vout82 = __riscv_vncvt_x_x_w_i8m1(vout2, vl);
    vint8m1_t vout83 = __riscv_vncvt_x_x_w_i8m1(vout3, vl);

    __riscv_vse8_v_i8m1(c0, vout80, vl);
    __riscv_vse8_v_i8m1(c1, vout81, vl);
    __riscv_vse8_v_i8m1(c2, vout82, vl);
    __riscv_vse8_v_i8m1(c3, vout83, vl);

    c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
    c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
    c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
    c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

    a = (const int8_t**restrict) ((uintptr_t) a - ks);

  } while (nc != 0);
}
