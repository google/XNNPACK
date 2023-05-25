// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/MRx2-neon-ld64.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/gemm.h>


void xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    float32x2_t vacc0x01 = vreinterpret_f32_u8(vld1_u8(w)); w = (const float*) w + 2;
    float32x2_t vacc1x01 = vacc0x01;
    float32x2_t vacc2x01 = vacc0x01;
    float32x2_t vacc3x01 = vacc0x01;

    size_t k = kc;
    for (; k >= 2 * sizeof(float); k -= 2 * sizeof(float)) {
      const float32x2_t va0 = vld1_f32(a0); a0 += 2;
      const float32x2_t va1 = vld1_f32(a1); a1 += 2;
      const float32x2_t va2 = vld1_f32(a2); a2 += 2;
      const float32x2_t va3 = vld1_f32(a3); a3 += 2;

      const uint32x2_t vtmpb = vld1_dup_u32(w); w = (const int8_t*) w + 4;
      const int32x4_t vtmpi = vmovl_s16(vget_low_s16(vmovl_s8(vreinterpret_s8_u32(vtmpb))));
      const float32x4_t vb01c01 = vcvtq_f32_s32(vtmpi);
      const float32x2_t vb01c0 = vget_low_f32(vb01c01);
      const float32x2_t vb01c1 = vget_high_f32(vb01c01);
      vacc0x01 = vmla_lane_f32(vacc0x01, vb01c0, va0, 0);
      vacc1x01 = vmla_lane_f32(vacc1x01, vb01c0, va1, 0);
      vacc2x01 = vmla_lane_f32(vacc2x01, vb01c0, va2, 0);
      vacc3x01 = vmla_lane_f32(vacc3x01, vb01c0, va3, 0);
      vacc0x01 = vmla_lane_f32(vacc0x01, vb01c1, va0, 1);
      vacc1x01 = vmla_lane_f32(vacc1x01, vb01c1, va1, 1);
      vacc2x01 = vmla_lane_f32(vacc2x01, vb01c1, va2, 1);
      vacc3x01 = vmla_lane_f32(vacc3x01, vb01c1, va3, 1);
    }
    if XNN_UNLIKELY(k != 0) {
      const float32x2_t va0 = vld1_dup_f32(a0); a0 += 1;
      const float32x2_t va1 = vld1_dup_f32(a1); a1 += 1;
      const float32x2_t va2 = vld1_dup_f32(a2); a2 += 1;
      const float32x2_t va3 = vld1_dup_f32(a3); a3 += 1;

      const uint16x4_t vtmpb = vld1_dup_u16(w); w = (const int8_t*) w + 2;
      const int32x2_t vtmpi = vget_low_s32(vmovl_s16(vget_low_s16(vmovl_s8(vreinterpret_s8_u16(vtmpb)))));
      const float32x2_t vb01 = vcvt_f32_s32(vtmpi);

      vacc0x01 = vmla_f32(vacc0x01, va0, vb01);
      vacc1x01 = vmla_f32(vacc1x01, va1, vb01);
      vacc2x01 = vmla_f32(vacc2x01, va2, vb01);
      vacc3x01 = vmla_f32(vacc3x01, va3, vb01);
    }

    const float32x2_t vscale = vreinterpret_f32_u8(vld1_u8(w)); w = (const float*) w + 2;
    vacc0x01 = vmul_f32(vacc0x01, vscale);
    vacc1x01 = vmul_f32(vacc1x01, vscale);
    vacc2x01 = vmul_f32(vacc2x01, vscale);
    vacc3x01 = vmul_f32(vacc3x01, vscale);
    const float32x2_t vmax = vld1_dup_f32(&params->scalar.max);
    vacc0x01 = vmin_f32(vacc0x01, vmax);
    vacc1x01 = vmin_f32(vacc1x01, vmax);
    vacc2x01 = vmin_f32(vacc2x01, vmax);
    vacc3x01 = vmin_f32(vacc3x01, vmax);
    const float32x2_t vmin = vld1_dup_f32(&params->scalar.min);
    vacc0x01 = vmax_f32(vacc0x01, vmin);
    vacc1x01 = vmax_f32(vacc1x01, vmin);
    vacc2x01 = vmax_f32(vacc2x01, vmin);
    vacc3x01 = vmax_f32(vacc3x01, vmin);

    if XNN_LIKELY(nc >= 2) {
      vst1_f32(c0, vacc0x01);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      vst1_f32(c1, vacc1x01);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1_f32(c2, vacc2x01);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1_f32(c3, vacc3x01);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);

      nc -= 2;
    } else {
      assert(nc == 1);
      vst1_lane_f32(c0, vacc0x01, 0);
      vst1_lane_f32(c1, vacc1x01, 0);
      vst1_lane_f32(c2, vacc2x01, 0);
      vst1_lane_f32(c3, vacc3x01, 0);

      nc = 0;
    }
  } while (nc != 0);
}
