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

void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const struct xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)])
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

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;


  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e32m4(nc);
    }
    nc = nc - vl;

    vint32m4_t vksum = __riscv_vle32_v_i32m4((const int32_t*)w, vl);
    const int32_t vinput_zero_point = quantization_params->zero_point;
    vint32m4_t vacc0 = __riscv_vmul_vx_i32m4(vksum, vinput_zero_point, vl);
    vint32m4_t vacc1 = __riscv_vmul_vx_i32m4(vksum, vinput_zero_point, vl);
    vint32m4_t vacc2 = __riscv_vmul_vx_i32m4(vksum, vinput_zero_point, vl);
    vint32m4_t vacc3 = __riscv_vmul_vx_i32m4(vksum, vinput_zero_point, vl);
    w = (const void*) ((const int32_t*) w + nr);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      const int8_t* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      } else {
        a1 = zero_data;
      }
      const int8_t* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      } else {
        a2 = zero_data;
      }
      const int8_t* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      } else {
        a3 = zero_data;
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

    const float vinput_scale = quantization_params->inv_scale;
    vfpacc0 = __riscv_vfmul_vf_f32m4(vfpacc0, vinput_scale, vl);
    vfpacc1 = __riscv_vfmul_vf_f32m4(vfpacc1, vinput_scale, vl);
    vfpacc2 = __riscv_vfmul_vf_f32m4(vfpacc2, vinput_scale, vl);
    vfpacc3 = __riscv_vfmul_vf_f32m4(vfpacc3, vinput_scale, vl);

    const vfloat32m4_t vscale = __riscv_vle32_v_f32m4((const float*) w, vl);
    vfpacc0 = __riscv_vfmul_vv_f32m4(vfpacc0, vscale, vl);
    vfpacc1 = __riscv_vfmul_vv_f32m4(vfpacc1, vscale, vl);
    vfpacc2 = __riscv_vfmul_vv_f32m4(vfpacc2, vscale, vl);
    vfpacc3 = __riscv_vfmul_vv_f32m4(vfpacc3, vscale, vl);

    w = (const void*) ((const float*) w + nr);

    const vfloat32m4_t vbias = __riscv_vle32_v_f32m4((const float*) w, vl);
    vfpacc0 = __riscv_vfadd_vv_f32m4(vfpacc0, vbias, vl);
    vfpacc1 = __riscv_vfadd_vv_f32m4(vfpacc1, vbias, vl);
    vfpacc2 = __riscv_vfadd_vv_f32m4(vfpacc2, vbias, vl);
    vfpacc3 = __riscv_vfadd_vv_f32m4(vfpacc3, vbias, vl);

    w = (const void*) ((const float*) w + nr);

    const float voutput_min = params->scalar.min;
    vfpacc0 = __riscv_vfmax_vf_f32m4(vfpacc0, voutput_min, vl);
    vfpacc1 = __riscv_vfmax_vf_f32m4(vfpacc1, voutput_min, vl);
    vfpacc2 = __riscv_vfmax_vf_f32m4(vfpacc2, voutput_min, vl);
    vfpacc3 = __riscv_vfmax_vf_f32m4(vfpacc3, voutput_min, vl);

    const float voutput_max = params->scalar.max;
    vfpacc0 = __riscv_vfmin_vf_f32m4(vfpacc0, voutput_max, vl);
    vfpacc1 = __riscv_vfmin_vf_f32m4(vfpacc1, voutput_max, vl);
    vfpacc2 = __riscv_vfmin_vf_f32m4(vfpacc2, voutput_max, vl);
    vfpacc3 = __riscv_vfmin_vf_f32m4(vfpacc3, voutput_max, vl);

    __riscv_vse32_v_f32m4(c0, vfpacc0, vl);
    __riscv_vse32_v_f32m4(c1, vfpacc1, vl);
    __riscv_vse32_v_f32m4(c2, vfpacc2, vl);
    __riscv_vse32_v_f32m4(c3, vfpacc3, vl);

    c3 = (float*) ((uintptr_t) c3 + cn_stride);
    c2 = (float*) ((uintptr_t) c2 + cn_stride);
    c1 = (float*) ((uintptr_t) c1 + cn_stride);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);

    a = (const int8_t**restrict) ((uintptr_t) a - ks);

  } while (nc != 0);
}
