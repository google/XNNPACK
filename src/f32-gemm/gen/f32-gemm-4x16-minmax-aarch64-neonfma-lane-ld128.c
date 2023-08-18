// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/neon-ld128.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/gemm.h>


void xnn_f32_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
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
    float32x4_t vacc0x0123 = vld1q_f32(w); w += 4;
    float32x4_t vacc0x4567 = vld1q_f32(w); w += 4;
    float32x4_t vacc0x89AB = vld1q_f32(w); w += 4;
    float32x4_t vacc0xCDEF = vld1q_f32(w); w += 4;
    float32x4_t vacc1x0123 = vacc0x0123;
    float32x4_t vacc1x4567 = vacc0x4567;
    float32x4_t vacc1x89AB = vacc0x89AB;
    float32x4_t vacc1xCDEF = vacc0xCDEF;
    float32x4_t vacc2x0123 = vacc0x0123;
    float32x4_t vacc2x4567 = vacc0x4567;
    float32x4_t vacc2x89AB = vacc0x89AB;
    float32x4_t vacc2xCDEF = vacc0xCDEF;
    float32x4_t vacc3x0123 = vacc0x0123;
    float32x4_t vacc3x4567 = vacc0x4567;
    float32x4_t vacc3x89AB = vacc0x89AB;
    float32x4_t vacc3xCDEF = vacc0xCDEF;

    size_t k = kc;
    if XNN_LIKELY(k >= 4 * sizeof(float)) {
      do {
        const float32x4_t va0 = vld1q_f32(a0); a0 += 4;
        const float32x4_t va1 = vld1q_f32(a1); a1 += 4;
        const float32x4_t va2 = vld1q_f32(a2); a2 += 4;
        const float32x4_t va3 = vld1q_f32(a3); a3 += 4;


        const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;
        const float32x4_t vb89ABc0 = vld1q_f32(w); w += 4;
        const float32x4_t vbCDEFc0 = vld1q_f32(w); w += 4;

        vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123c0, vget_low_f32(va0), 0);
        vacc1x0123 = vfmaq_lane_f32(vacc1x0123, vb0123c0, vget_low_f32(va1), 0);
        vacc2x0123 = vfmaq_lane_f32(vacc2x0123, vb0123c0, vget_low_f32(va2), 0);
        vacc3x0123 = vfmaq_lane_f32(vacc3x0123, vb0123c0, vget_low_f32(va3), 0);
        vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567c0, vget_low_f32(va0), 0);
        vacc1x4567 = vfmaq_lane_f32(vacc1x4567, vb4567c0, vget_low_f32(va1), 0);
        vacc2x4567 = vfmaq_lane_f32(vacc2x4567, vb4567c0, vget_low_f32(va2), 0);
        vacc3x4567 = vfmaq_lane_f32(vacc3x4567, vb4567c0, vget_low_f32(va3), 0);
        vacc0x89AB = vfmaq_lane_f32(vacc0x89AB, vb89ABc0, vget_low_f32(va0), 0);
        vacc1x89AB = vfmaq_lane_f32(vacc1x89AB, vb89ABc0, vget_low_f32(va1), 0);
        vacc2x89AB = vfmaq_lane_f32(vacc2x89AB, vb89ABc0, vget_low_f32(va2), 0);
        vacc3x89AB = vfmaq_lane_f32(vacc3x89AB, vb89ABc0, vget_low_f32(va3), 0);
        vacc0xCDEF = vfmaq_lane_f32(vacc0xCDEF, vbCDEFc0, vget_low_f32(va0), 0);
        vacc1xCDEF = vfmaq_lane_f32(vacc1xCDEF, vbCDEFc0, vget_low_f32(va1), 0);
        vacc2xCDEF = vfmaq_lane_f32(vacc2xCDEF, vbCDEFc0, vget_low_f32(va2), 0);
        vacc3xCDEF = vfmaq_lane_f32(vacc3xCDEF, vbCDEFc0, vget_low_f32(va3), 0);

        const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;
        const float32x4_t vb89ABc1 = vld1q_f32(w); w += 4;
        const float32x4_t vbCDEFc1 = vld1q_f32(w); w += 4;

        vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123c1, vget_low_f32(va0), 1);
        vacc1x0123 = vfmaq_lane_f32(vacc1x0123, vb0123c1, vget_low_f32(va1), 1);
        vacc2x0123 = vfmaq_lane_f32(vacc2x0123, vb0123c1, vget_low_f32(va2), 1);
        vacc3x0123 = vfmaq_lane_f32(vacc3x0123, vb0123c1, vget_low_f32(va3), 1);
        vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567c1, vget_low_f32(va0), 1);
        vacc1x4567 = vfmaq_lane_f32(vacc1x4567, vb4567c1, vget_low_f32(va1), 1);
        vacc2x4567 = vfmaq_lane_f32(vacc2x4567, vb4567c1, vget_low_f32(va2), 1);
        vacc3x4567 = vfmaq_lane_f32(vacc3x4567, vb4567c1, vget_low_f32(va3), 1);
        vacc0x89AB = vfmaq_lane_f32(vacc0x89AB, vb89ABc1, vget_low_f32(va0), 1);
        vacc1x89AB = vfmaq_lane_f32(vacc1x89AB, vb89ABc1, vget_low_f32(va1), 1);
        vacc2x89AB = vfmaq_lane_f32(vacc2x89AB, vb89ABc1, vget_low_f32(va2), 1);
        vacc3x89AB = vfmaq_lane_f32(vacc3x89AB, vb89ABc1, vget_low_f32(va3), 1);
        vacc0xCDEF = vfmaq_lane_f32(vacc0xCDEF, vbCDEFc1, vget_low_f32(va0), 1);
        vacc1xCDEF = vfmaq_lane_f32(vacc1xCDEF, vbCDEFc1, vget_low_f32(va1), 1);
        vacc2xCDEF = vfmaq_lane_f32(vacc2xCDEF, vbCDEFc1, vget_low_f32(va2), 1);
        vacc3xCDEF = vfmaq_lane_f32(vacc3xCDEF, vbCDEFc1, vget_low_f32(va3), 1);

        const float32x4_t vb0123c2 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c2 = vld1q_f32(w); w += 4;
        const float32x4_t vb89ABc2 = vld1q_f32(w); w += 4;
        const float32x4_t vbCDEFc2 = vld1q_f32(w); w += 4;

        vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123c2, vget_high_f32(va0), 0);
        vacc1x0123 = vfmaq_lane_f32(vacc1x0123, vb0123c2, vget_high_f32(va1), 0);
        vacc2x0123 = vfmaq_lane_f32(vacc2x0123, vb0123c2, vget_high_f32(va2), 0);
        vacc3x0123 = vfmaq_lane_f32(vacc3x0123, vb0123c2, vget_high_f32(va3), 0);
        vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567c2, vget_high_f32(va0), 0);
        vacc1x4567 = vfmaq_lane_f32(vacc1x4567, vb4567c2, vget_high_f32(va1), 0);
        vacc2x4567 = vfmaq_lane_f32(vacc2x4567, vb4567c2, vget_high_f32(va2), 0);
        vacc3x4567 = vfmaq_lane_f32(vacc3x4567, vb4567c2, vget_high_f32(va3), 0);
        vacc0x89AB = vfmaq_lane_f32(vacc0x89AB, vb89ABc2, vget_high_f32(va0), 0);
        vacc1x89AB = vfmaq_lane_f32(vacc1x89AB, vb89ABc2, vget_high_f32(va1), 0);
        vacc2x89AB = vfmaq_lane_f32(vacc2x89AB, vb89ABc2, vget_high_f32(va2), 0);
        vacc3x89AB = vfmaq_lane_f32(vacc3x89AB, vb89ABc2, vget_high_f32(va3), 0);
        vacc0xCDEF = vfmaq_lane_f32(vacc0xCDEF, vbCDEFc2, vget_high_f32(va0), 0);
        vacc1xCDEF = vfmaq_lane_f32(vacc1xCDEF, vbCDEFc2, vget_high_f32(va1), 0);
        vacc2xCDEF = vfmaq_lane_f32(vacc2xCDEF, vbCDEFc2, vget_high_f32(va2), 0);
        vacc3xCDEF = vfmaq_lane_f32(vacc3xCDEF, vbCDEFc2, vget_high_f32(va3), 0);

        const float32x4_t vb0123c3 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c3 = vld1q_f32(w); w += 4;
        const float32x4_t vb89ABc3 = vld1q_f32(w); w += 4;
        const float32x4_t vbCDEFc3 = vld1q_f32(w); w += 4;

        vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123c3, vget_high_f32(va0), 1);
        vacc1x0123 = vfmaq_lane_f32(vacc1x0123, vb0123c3, vget_high_f32(va1), 1);
        vacc2x0123 = vfmaq_lane_f32(vacc2x0123, vb0123c3, vget_high_f32(va2), 1);
        vacc3x0123 = vfmaq_lane_f32(vacc3x0123, vb0123c3, vget_high_f32(va3), 1);
        vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567c3, vget_high_f32(va0), 1);
        vacc1x4567 = vfmaq_lane_f32(vacc1x4567, vb4567c3, vget_high_f32(va1), 1);
        vacc2x4567 = vfmaq_lane_f32(vacc2x4567, vb4567c3, vget_high_f32(va2), 1);
        vacc3x4567 = vfmaq_lane_f32(vacc3x4567, vb4567c3, vget_high_f32(va3), 1);
        vacc0x89AB = vfmaq_lane_f32(vacc0x89AB, vb89ABc3, vget_high_f32(va0), 1);
        vacc1x89AB = vfmaq_lane_f32(vacc1x89AB, vb89ABc3, vget_high_f32(va1), 1);
        vacc2x89AB = vfmaq_lane_f32(vacc2x89AB, vb89ABc3, vget_high_f32(va2), 1);
        vacc3x89AB = vfmaq_lane_f32(vacc3x89AB, vb89ABc3, vget_high_f32(va3), 1);
        vacc0xCDEF = vfmaq_lane_f32(vacc0xCDEF, vbCDEFc3, vget_high_f32(va0), 1);
        vacc1xCDEF = vfmaq_lane_f32(vacc1xCDEF, vbCDEFc3, vget_high_f32(va1), 1);
        vacc2xCDEF = vfmaq_lane_f32(vacc2xCDEF, vbCDEFc3, vget_high_f32(va2), 1);
        vacc3xCDEF = vfmaq_lane_f32(vacc3xCDEF, vbCDEFc3, vget_high_f32(va3), 1);
        k -= 4 * sizeof(float);
      } while (k >= 4 * sizeof(float));
    }

    if XNN_UNLIKELY(k != 0) {
      if XNN_UNLIKELY(k & (2 * sizeof(float))) {
        const float32x2_t va0 = vld1_f32(a0); a0 += 2;
        const float32x2_t va1 = vld1_f32(a1); a1 += 2;
        const float32x2_t va2 = vld1_f32(a2); a2 += 2;
        const float32x2_t va3 = vld1_f32(a3); a3 += 2;


        const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;
        const float32x4_t vb89ABc0 = vld1q_f32(w); w += 4;
        const float32x4_t vbCDEFc0 = vld1q_f32(w); w += 4;

        vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123c0, va0, 0);
        vacc1x0123 = vfmaq_lane_f32(vacc1x0123, vb0123c0, va1, 0);
        vacc2x0123 = vfmaq_lane_f32(vacc2x0123, vb0123c0, va2, 0);
        vacc3x0123 = vfmaq_lane_f32(vacc3x0123, vb0123c0, va3, 0);
        vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567c0, va0, 0);
        vacc1x4567 = vfmaq_lane_f32(vacc1x4567, vb4567c0, va1, 0);
        vacc2x4567 = vfmaq_lane_f32(vacc2x4567, vb4567c0, va2, 0);
        vacc3x4567 = vfmaq_lane_f32(vacc3x4567, vb4567c0, va3, 0);
        vacc0x89AB = vfmaq_lane_f32(vacc0x89AB, vb89ABc0, va0, 0);
        vacc1x89AB = vfmaq_lane_f32(vacc1x89AB, vb89ABc0, va1, 0);
        vacc2x89AB = vfmaq_lane_f32(vacc2x89AB, vb89ABc0, va2, 0);
        vacc3x89AB = vfmaq_lane_f32(vacc3x89AB, vb89ABc0, va3, 0);
        vacc0xCDEF = vfmaq_lane_f32(vacc0xCDEF, vbCDEFc0, va0, 0);
        vacc1xCDEF = vfmaq_lane_f32(vacc1xCDEF, vbCDEFc0, va1, 0);
        vacc2xCDEF = vfmaq_lane_f32(vacc2xCDEF, vbCDEFc0, va2, 0);
        vacc3xCDEF = vfmaq_lane_f32(vacc3xCDEF, vbCDEFc0, va3, 0);

        const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;
        const float32x4_t vb89ABc1 = vld1q_f32(w); w += 4;
        const float32x4_t vbCDEFc1 = vld1q_f32(w); w += 4;

        vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123c1, va0, 1);
        vacc1x0123 = vfmaq_lane_f32(vacc1x0123, vb0123c1, va1, 1);
        vacc2x0123 = vfmaq_lane_f32(vacc2x0123, vb0123c1, va2, 1);
        vacc3x0123 = vfmaq_lane_f32(vacc3x0123, vb0123c1, va3, 1);
        vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567c1, va0, 1);
        vacc1x4567 = vfmaq_lane_f32(vacc1x4567, vb4567c1, va1, 1);
        vacc2x4567 = vfmaq_lane_f32(vacc2x4567, vb4567c1, va2, 1);
        vacc3x4567 = vfmaq_lane_f32(vacc3x4567, vb4567c1, va3, 1);
        vacc0x89AB = vfmaq_lane_f32(vacc0x89AB, vb89ABc1, va0, 1);
        vacc1x89AB = vfmaq_lane_f32(vacc1x89AB, vb89ABc1, va1, 1);
        vacc2x89AB = vfmaq_lane_f32(vacc2x89AB, vb89ABc1, va2, 1);
        vacc3x89AB = vfmaq_lane_f32(vacc3x89AB, vb89ABc1, va3, 1);
        vacc0xCDEF = vfmaq_lane_f32(vacc0xCDEF, vbCDEFc1, va0, 1);
        vacc1xCDEF = vfmaq_lane_f32(vacc1xCDEF, vbCDEFc1, va1, 1);
        vacc2xCDEF = vfmaq_lane_f32(vacc2xCDEF, vbCDEFc1, va2, 1);
        vacc3xCDEF = vfmaq_lane_f32(vacc3xCDEF, vbCDEFc1, va3, 1);
      }
      if XNN_UNLIKELY(k & (1 * sizeof(float))) {
        const float32x4_t va0 = vld1q_dup_f32(a0); a0 += 1;
        const float32x4_t va1 = vld1q_dup_f32(a1); a1 += 1;
        const float32x4_t va2 = vld1q_dup_f32(a2); a2 += 1;
        const float32x4_t va3 = vld1q_dup_f32(a3); a3 += 1;

        const float32x4_t vb0123 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567 = vld1q_f32(w); w += 4;
        const float32x4_t vb89AB = vld1q_f32(w); w += 4;
        const float32x4_t vbCDEF = vld1q_f32(w); w += 4;

        vacc0x0123 = vfmaq_f32(vacc0x0123, va0, vb0123);
        vacc1x0123 = vfmaq_f32(vacc1x0123, va1, vb0123);
        vacc2x0123 = vfmaq_f32(vacc2x0123, va2, vb0123);
        vacc3x0123 = vfmaq_f32(vacc3x0123, va3, vb0123);
        vacc0x4567 = vfmaq_f32(vacc0x4567, va0, vb4567);
        vacc1x4567 = vfmaq_f32(vacc1x4567, va1, vb4567);
        vacc2x4567 = vfmaq_f32(vacc2x4567, va2, vb4567);
        vacc3x4567 = vfmaq_f32(vacc3x4567, va3, vb4567);
        vacc0x89AB = vfmaq_f32(vacc0x89AB, va0, vb89AB);
        vacc1x89AB = vfmaq_f32(vacc1x89AB, va1, vb89AB);
        vacc2x89AB = vfmaq_f32(vacc2x89AB, va2, vb89AB);
        vacc3x89AB = vfmaq_f32(vacc3x89AB, va3, vb89AB);
        vacc0xCDEF = vfmaq_f32(vacc0xCDEF, va0, vbCDEF);
        vacc1xCDEF = vfmaq_f32(vacc1xCDEF, va1, vbCDEF);
        vacc2xCDEF = vfmaq_f32(vacc2xCDEF, va2, vbCDEF);
        vacc3xCDEF = vfmaq_f32(vacc3xCDEF, va3, vbCDEF);
      }
    }
    const float32x4_t vmax = vld1q_dup_f32(&params->scalar_cvt.max);
    vacc0x0123 = vminq_f32(vacc0x0123, vmax);
    vacc1x0123 = vminq_f32(vacc1x0123, vmax);
    vacc2x0123 = vminq_f32(vacc2x0123, vmax);
    vacc3x0123 = vminq_f32(vacc3x0123, vmax);
    vacc0x4567 = vminq_f32(vacc0x4567, vmax);
    vacc1x4567 = vminq_f32(vacc1x4567, vmax);
    vacc2x4567 = vminq_f32(vacc2x4567, vmax);
    vacc3x4567 = vminq_f32(vacc3x4567, vmax);
    vacc0x89AB = vminq_f32(vacc0x89AB, vmax);
    vacc1x89AB = vminq_f32(vacc1x89AB, vmax);
    vacc2x89AB = vminq_f32(vacc2x89AB, vmax);
    vacc3x89AB = vminq_f32(vacc3x89AB, vmax);
    vacc0xCDEF = vminq_f32(vacc0xCDEF, vmax);
    vacc1xCDEF = vminq_f32(vacc1xCDEF, vmax);
    vacc2xCDEF = vminq_f32(vacc2xCDEF, vmax);
    vacc3xCDEF = vminq_f32(vacc3xCDEF, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar_cvt.min);
    vacc0x0123 = vmaxq_f32(vacc0x0123, vmin);
    vacc1x0123 = vmaxq_f32(vacc1x0123, vmin);
    vacc2x0123 = vmaxq_f32(vacc2x0123, vmin);
    vacc3x0123 = vmaxq_f32(vacc3x0123, vmin);
    vacc0x4567 = vmaxq_f32(vacc0x4567, vmin);
    vacc1x4567 = vmaxq_f32(vacc1x4567, vmin);
    vacc2x4567 = vmaxq_f32(vacc2x4567, vmin);
    vacc3x4567 = vmaxq_f32(vacc3x4567, vmin);
    vacc0x89AB = vmaxq_f32(vacc0x89AB, vmin);
    vacc1x89AB = vmaxq_f32(vacc1x89AB, vmin);
    vacc2x89AB = vmaxq_f32(vacc2x89AB, vmin);
    vacc3x89AB = vmaxq_f32(vacc3x89AB, vmin);
    vacc0xCDEF = vmaxq_f32(vacc0xCDEF, vmin);
    vacc1xCDEF = vmaxq_f32(vacc1xCDEF, vmin);
    vacc2xCDEF = vmaxq_f32(vacc2xCDEF, vmin);
    vacc3xCDEF = vmaxq_f32(vacc3xCDEF, vmin);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(c3, vacc3x0123);
      vst1q_f32(c3 + 4, vacc3x4567);
      vst1q_f32(c3 + 8, vacc3x89AB);
      vst1q_f32(c3 + 12, vacc3xCDEF);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      vst1q_f32(c2, vacc2x0123);
      vst1q_f32(c2 + 4, vacc2x4567);
      vst1q_f32(c2 + 8, vacc2x89AB);
      vst1q_f32(c2 + 12, vacc2xCDEF);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1q_f32(c1, vacc1x0123);
      vst1q_f32(c1 + 4, vacc1x4567);
      vst1q_f32(c1 + 8, vacc1x89AB);
      vst1q_f32(c1 + 12, vacc1xCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1q_f32(c0, vacc0x0123);
      vst1q_f32(c0 + 4, vacc0x4567);
      vst1q_f32(c0 + 8, vacc0x89AB);
      vst1q_f32(c0 + 12, vacc0xCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a3 = (const float*) ((uintptr_t) a3 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 16;

    } else {
      if (nc & 8) {
        vst1q_f32(c3, vacc3x0123); c3 += 4;
        vst1q_f32(c2, vacc2x0123); c2 += 4;
        vst1q_f32(c1, vacc1x0123); c1 += 4;
        vst1q_f32(c0, vacc0x0123); c0 += 4;
        vst1q_f32(c3, vacc3x4567); c3 += 4;
        vst1q_f32(c2, vacc2x4567); c2 += 4;
        vst1q_f32(c1, vacc1x4567); c1 += 4;
        vst1q_f32(c0, vacc0x4567); c0 += 4;

        vacc3x0123 = vacc3x89AB;
        vacc3x4567 = vacc3xCDEF;
        vacc2x0123 = vacc2x89AB;
        vacc2x4567 = vacc2xCDEF;
        vacc1x0123 = vacc1x89AB;
        vacc1x4567 = vacc1xCDEF;
        vacc0x0123 = vacc0x89AB;
        vacc0x4567 = vacc0xCDEF;
      }
      if (nc & 4) {
        vst1q_f32(c3, vacc3x0123); c3 += 4;
        vst1q_f32(c2, vacc2x0123); c2 += 4;
        vst1q_f32(c1, vacc1x0123); c1 += 4;
        vst1q_f32(c0, vacc0x0123); c0 += 4;

        vacc3x0123 = vacc3x4567;
        vacc3x4567 = vacc3x89AB;
        vacc3x89AB = vacc3xCDEF;
        vacc2x0123 = vacc2x4567;
        vacc2x4567 = vacc2x89AB;
        vacc2x89AB = vacc2xCDEF;
        vacc1x0123 = vacc1x4567;
        vacc1x4567 = vacc1x89AB;
        vacc1x89AB = vacc1xCDEF;
        vacc0x0123 = vacc0x4567;
        vacc0x4567 = vacc0x89AB;
        vacc0x89AB = vacc0xCDEF;
      }
      float32x2_t vacc3x01 = vget_low_f32(vacc3x0123);
      float32x2_t vacc2x01 = vget_low_f32(vacc2x0123);
      float32x2_t vacc1x01 = vget_low_f32(vacc1x0123);
      float32x2_t vacc0x01 = vget_low_f32(vacc0x0123);
      if (nc & 2) {
        vst1_f32(c3, vacc3x01); c3 += 2;
        vst1_f32(c2, vacc2x01); c2 += 2;
        vst1_f32(c1, vacc1x01); c1 += 2;
        vst1_f32(c0, vacc0x01); c0 += 2;

        vacc3x01 = vget_high_f32(vacc3x0123);
        vacc2x01 = vget_high_f32(vacc2x0123);
        vacc1x01 = vget_high_f32(vacc1x0123);
        vacc0x01 = vget_high_f32(vacc0x0123);
      }
      if (nc & 1) {
        vst1_lane_f32(c3, vacc3x01, 0);
        vst1_lane_f32(c2, vacc2x01, 0);
        vst1_lane_f32(c1, vacc1x01, 0);
        vst1_lane_f32(c0, vacc0x01, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
