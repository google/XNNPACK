// Auto-generated file. Do not edit!
//   Template: src/f16-gemm/neonfp16arith-ld64.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>

#include <xnnpack/gemm.h>


void xnn_f16_gemminc_minmax_ukernel_6x8__neonfp16arith_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* restrict a,
    size_t a_stride,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const void*restrict acc,
    const struct xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(__fp16) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);
  assert(acc != NULL);

  const __fp16* a0 = a;
  __fp16* c0 = c;
  const __fp16* a1 = (const __fp16*) ((uintptr_t) a0 + a_stride);
  __fp16* c1 = (__fp16*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const __fp16* a2 = (const __fp16*) ((uintptr_t) a1 + a_stride);
  __fp16* c2 = (__fp16*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const __fp16* a3 = (const __fp16*) ((uintptr_t) a2 + a_stride);
  __fp16* c3 = (__fp16*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const __fp16* a4 = (const __fp16*) ((uintptr_t) a3 + a_stride);
  __fp16* c4 = (__fp16*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const __fp16* a5 = (const __fp16*) ((uintptr_t) a4 + a_stride);
  __fp16* c5 = (__fp16*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  do {
    float16x8_t vacc0x01234567 = vld1q_f16(acc); acc = (const void*) ((uintptr_t) acc + sizeof(float16x8_t));
    float16x8_t vacc1x01234567 = vld1q_f16(acc); acc = (const void*) ((uintptr_t) acc + sizeof(float16x8_t));
    float16x8_t vacc2x01234567 = vld1q_f16(acc); acc = (const void*) ((uintptr_t) acc + sizeof(float16x8_t));
    float16x8_t vacc3x01234567 = vld1q_f16(acc); acc = (const void*) ((uintptr_t) acc + sizeof(float16x8_t));
    float16x8_t vacc4x01234567 = vld1q_f16(acc); acc = (const void*) ((uintptr_t) acc + sizeof(float16x8_t));
    float16x8_t vacc5x01234567 = vld1q_f16(acc); acc = (const void*) ((uintptr_t) acc + sizeof(float16x8_t));

    size_t k = kc;
    while (k >= 4 * sizeof(__fp16)) {
      const float16x4_t va0 = vld1_f16(a0); a0 += 4;
      const float16x4_t va1 = vld1_f16(a1); a1 += 4;
      const float16x4_t va2 = vld1_f16(a2); a2 += 4;
      const float16x4_t va3 = vld1_f16(a3); a3 += 4;
      const float16x4_t va4 = vld1_f16(a4); a4 += 4;
      const float16x4_t va5 = vld1_f16(a5); a5 += 4;

      const float16x8_t vb01234567c0 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));

      #if XNN_ARCH_ARM64
          vacc0x01234567 = vfmaq_lane_f16(vacc0x01234567, vb01234567c0, va0, 0);
          vacc1x01234567 = vfmaq_lane_f16(vacc1x01234567, vb01234567c0, va1, 0);
          vacc2x01234567 = vfmaq_lane_f16(vacc2x01234567, vb01234567c0, va2, 0);
          vacc3x01234567 = vfmaq_lane_f16(vacc3x01234567, vb01234567c0, va3, 0);
          vacc4x01234567 = vfmaq_lane_f16(vacc4x01234567, vb01234567c0, va4, 0);
          vacc5x01234567 = vfmaq_lane_f16(vacc5x01234567, vb01234567c0, va5, 0);
      #else
        const float16x8_t va0c0 = vdupq_lane_f16(va0, 0);
        const float16x8_t va1c0 = vdupq_lane_f16(va1, 0);
        const float16x8_t va2c0 = vdupq_lane_f16(va2, 0);
        const float16x8_t va3c0 = vdupq_lane_f16(va3, 0);
        const float16x8_t va4c0 = vdupq_lane_f16(va4, 0);
        const float16x8_t va5c0 = vdupq_lane_f16(va5, 0);

        vacc0x01234567 = vfmaq_f16(vacc0x01234567, va0c0, vb01234567c0);
        vacc1x01234567 = vfmaq_f16(vacc1x01234567, va1c0, vb01234567c0);
        vacc2x01234567 = vfmaq_f16(vacc2x01234567, va2c0, vb01234567c0);
        vacc3x01234567 = vfmaq_f16(vacc3x01234567, va3c0, vb01234567c0);
        vacc4x01234567 = vfmaq_f16(vacc4x01234567, va4c0, vb01234567c0);
        vacc5x01234567 = vfmaq_f16(vacc5x01234567, va5c0, vb01234567c0);
      #endif
      const float16x8_t vb01234567c1 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));

      #if XNN_ARCH_ARM64
          vacc0x01234567 = vfmaq_lane_f16(vacc0x01234567, vb01234567c1, va0, 1);
          vacc1x01234567 = vfmaq_lane_f16(vacc1x01234567, vb01234567c1, va1, 1);
          vacc2x01234567 = vfmaq_lane_f16(vacc2x01234567, vb01234567c1, va2, 1);
          vacc3x01234567 = vfmaq_lane_f16(vacc3x01234567, vb01234567c1, va3, 1);
          vacc4x01234567 = vfmaq_lane_f16(vacc4x01234567, vb01234567c1, va4, 1);
          vacc5x01234567 = vfmaq_lane_f16(vacc5x01234567, vb01234567c1, va5, 1);
      #else
        const float16x8_t va0c1 = vdupq_lane_f16(va0, 1);
        const float16x8_t va1c1 = vdupq_lane_f16(va1, 1);
        const float16x8_t va2c1 = vdupq_lane_f16(va2, 1);
        const float16x8_t va3c1 = vdupq_lane_f16(va3, 1);
        const float16x8_t va4c1 = vdupq_lane_f16(va4, 1);
        const float16x8_t va5c1 = vdupq_lane_f16(va5, 1);

        vacc0x01234567 = vfmaq_f16(vacc0x01234567, va0c1, vb01234567c1);
        vacc1x01234567 = vfmaq_f16(vacc1x01234567, va1c1, vb01234567c1);
        vacc2x01234567 = vfmaq_f16(vacc2x01234567, va2c1, vb01234567c1);
        vacc3x01234567 = vfmaq_f16(vacc3x01234567, va3c1, vb01234567c1);
        vacc4x01234567 = vfmaq_f16(vacc4x01234567, va4c1, vb01234567c1);
        vacc5x01234567 = vfmaq_f16(vacc5x01234567, va5c1, vb01234567c1);
      #endif
      const float16x8_t vb01234567c2 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));

      #if XNN_ARCH_ARM64
          vacc0x01234567 = vfmaq_lane_f16(vacc0x01234567, vb01234567c2, va0, 2);
          vacc1x01234567 = vfmaq_lane_f16(vacc1x01234567, vb01234567c2, va1, 2);
          vacc2x01234567 = vfmaq_lane_f16(vacc2x01234567, vb01234567c2, va2, 2);
          vacc3x01234567 = vfmaq_lane_f16(vacc3x01234567, vb01234567c2, va3, 2);
          vacc4x01234567 = vfmaq_lane_f16(vacc4x01234567, vb01234567c2, va4, 2);
          vacc5x01234567 = vfmaq_lane_f16(vacc5x01234567, vb01234567c2, va5, 2);
      #else
        const float16x8_t va0c2 = vdupq_lane_f16(va0, 2);
        const float16x8_t va1c2 = vdupq_lane_f16(va1, 2);
        const float16x8_t va2c2 = vdupq_lane_f16(va2, 2);
        const float16x8_t va3c2 = vdupq_lane_f16(va3, 2);
        const float16x8_t va4c2 = vdupq_lane_f16(va4, 2);
        const float16x8_t va5c2 = vdupq_lane_f16(va5, 2);

        vacc0x01234567 = vfmaq_f16(vacc0x01234567, va0c2, vb01234567c2);
        vacc1x01234567 = vfmaq_f16(vacc1x01234567, va1c2, vb01234567c2);
        vacc2x01234567 = vfmaq_f16(vacc2x01234567, va2c2, vb01234567c2);
        vacc3x01234567 = vfmaq_f16(vacc3x01234567, va3c2, vb01234567c2);
        vacc4x01234567 = vfmaq_f16(vacc4x01234567, va4c2, vb01234567c2);
        vacc5x01234567 = vfmaq_f16(vacc5x01234567, va5c2, vb01234567c2);
      #endif
      const float16x8_t vb01234567c3 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));

      #if XNN_ARCH_ARM64
          vacc0x01234567 = vfmaq_lane_f16(vacc0x01234567, vb01234567c3, va0, 3);
          vacc1x01234567 = vfmaq_lane_f16(vacc1x01234567, vb01234567c3, va1, 3);
          vacc2x01234567 = vfmaq_lane_f16(vacc2x01234567, vb01234567c3, va2, 3);
          vacc3x01234567 = vfmaq_lane_f16(vacc3x01234567, vb01234567c3, va3, 3);
          vacc4x01234567 = vfmaq_lane_f16(vacc4x01234567, vb01234567c3, va4, 3);
          vacc5x01234567 = vfmaq_lane_f16(vacc5x01234567, vb01234567c3, va5, 3);
      #else
        const float16x8_t va0c3 = vdupq_lane_f16(va0, 3);
        const float16x8_t va1c3 = vdupq_lane_f16(va1, 3);
        const float16x8_t va2c3 = vdupq_lane_f16(va2, 3);
        const float16x8_t va3c3 = vdupq_lane_f16(va3, 3);
        const float16x8_t va4c3 = vdupq_lane_f16(va4, 3);
        const float16x8_t va5c3 = vdupq_lane_f16(va5, 3);

        vacc0x01234567 = vfmaq_f16(vacc0x01234567, va0c3, vb01234567c3);
        vacc1x01234567 = vfmaq_f16(vacc1x01234567, va1c3, vb01234567c3);
        vacc2x01234567 = vfmaq_f16(vacc2x01234567, va2c3, vb01234567c3);
        vacc3x01234567 = vfmaq_f16(vacc3x01234567, va3c3, vb01234567c3);
        vacc4x01234567 = vfmaq_f16(vacc4x01234567, va4c3, vb01234567c3);
        vacc5x01234567 = vfmaq_f16(vacc5x01234567, va5c3, vb01234567c3);
      #endif

      k -= 4 * sizeof(__fp16);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const float16x8_t va0 = vld1q_dup_f16(a0); a0 += 1;
        const float16x8_t va1 = vld1q_dup_f16(a1); a1 += 1;
        const float16x8_t va2 = vld1q_dup_f16(a2); a2 += 1;
        const float16x8_t va3 = vld1q_dup_f16(a3); a3 += 1;
        const float16x8_t va4 = vld1q_dup_f16(a4); a4 += 1;
        const float16x8_t va5 = vld1q_dup_f16(a5); a5 += 1;

        const float16x8_t vb01234567 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));

        vacc0x01234567 = vfmaq_f16(vacc0x01234567, va0, vb01234567);
        vacc1x01234567 = vfmaq_f16(vacc1x01234567, va1, vb01234567);
        vacc2x01234567 = vfmaq_f16(vacc2x01234567, va2, vb01234567);
        vacc3x01234567 = vfmaq_f16(vacc3x01234567, va3, vb01234567);
        vacc4x01234567 = vfmaq_f16(vacc4x01234567, va4, vb01234567);
        vacc5x01234567 = vfmaq_f16(vacc5x01234567, va5, vb01234567);

        k -= sizeof(__fp16);
      } while (k != 0);
    }

    const float16x8_t vscale = vld1q_dup_f16((const __fp16*) &params->scale);
    vacc0x01234567 = vmulq_f16(vacc0x01234567, vscale);
    vacc1x01234567 = vmulq_f16(vacc1x01234567, vscale);
    vacc2x01234567 = vmulq_f16(vacc2x01234567, vscale);
    vacc3x01234567 = vmulq_f16(vacc3x01234567, vscale);
    vacc4x01234567 = vmulq_f16(vacc4x01234567, vscale);
    vacc5x01234567 = vmulq_f16(vacc5x01234567, vscale);

    const float16x8_t vmax = vld1q_dup_f16((const __fp16*) &params->max);
    vacc0x01234567 = vminq_f16(vacc0x01234567, vmax);
    vacc1x01234567 = vminq_f16(vacc1x01234567, vmax);
    vacc2x01234567 = vminq_f16(vacc2x01234567, vmax);
    vacc3x01234567 = vminq_f16(vacc3x01234567, vmax);
    vacc4x01234567 = vminq_f16(vacc4x01234567, vmax);
    vacc5x01234567 = vminq_f16(vacc5x01234567, vmax);

    const float16x8_t vmin = vld1q_dup_f16((const __fp16*) &params->min);
    vacc0x01234567 = vmaxq_f16(vacc0x01234567, vmin);
    vacc1x01234567 = vmaxq_f16(vacc1x01234567, vmin);
    vacc2x01234567 = vmaxq_f16(vacc2x01234567, vmin);
    vacc3x01234567 = vmaxq_f16(vacc3x01234567, vmin);
    vacc4x01234567 = vmaxq_f16(vacc4x01234567, vmin);
    vacc5x01234567 = vmaxq_f16(vacc5x01234567, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f16(c0, vacc0x01234567);
      c0 = (__fp16*) ((uintptr_t) c0 + cn_stride);
      vst1q_f16(c1, vacc1x01234567);
      c1 = (__fp16*) ((uintptr_t) c1 + cn_stride);
      vst1q_f16(c2, vacc2x01234567);
      c2 = (__fp16*) ((uintptr_t) c2 + cn_stride);
      vst1q_f16(c3, vacc3x01234567);
      c3 = (__fp16*) ((uintptr_t) c3 + cn_stride);
      vst1q_f16(c4, vacc4x01234567);
      c4 = (__fp16*) ((uintptr_t) c4 + cn_stride);
      vst1q_f16(c5, vacc5x01234567);
      c5 = (__fp16*) ((uintptr_t) c5 + cn_stride);

      a0 = (const __fp16*) ((uintptr_t) a0 - kc);
      a1 = (const __fp16*) ((uintptr_t) a1 - kc);
      a2 = (const __fp16*) ((uintptr_t) a2 - kc);
      a3 = (const __fp16*) ((uintptr_t) a3 - kc);
      a4 = (const __fp16*) ((uintptr_t) a4 - kc);
      a5 = (const __fp16*) ((uintptr_t) a5 - kc);

      nc -= 8;
    } else {
      float16x4_t vacc0x0123 = vget_low_f16(vacc0x01234567);
      float16x4_t vacc1x0123 = vget_low_f16(vacc1x01234567);
      float16x4_t vacc2x0123 = vget_low_f16(vacc2x01234567);
      float16x4_t vacc3x0123 = vget_low_f16(vacc3x01234567);
      float16x4_t vacc4x0123 = vget_low_f16(vacc4x01234567);
      float16x4_t vacc5x0123 = vget_low_f16(vacc5x01234567);
      if (nc & 4) {
        vst1_f16(c0, vacc0x0123); c0 += 4;
        vst1_f16(c1, vacc1x0123); c1 += 4;
        vst1_f16(c2, vacc2x0123); c2 += 4;
        vst1_f16(c3, vacc3x0123); c3 += 4;
        vst1_f16(c4, vacc4x0123); c4 += 4;
        vst1_f16(c5, vacc5x0123); c5 += 4;

        vacc0x0123 = vget_high_f16(vacc0x01234567);
        vacc1x0123 = vget_high_f16(vacc1x01234567);
        vacc2x0123 = vget_high_f16(vacc2x01234567);
        vacc3x0123 = vget_high_f16(vacc3x01234567);
        vacc4x0123 = vget_high_f16(vacc4x01234567);
        vacc5x0123 = vget_high_f16(vacc5x01234567);
      }
      if (nc & 2) {
        vst1_lane_u32(__builtin_assume_aligned(c0, 1), vreinterpret_u32_f16(vacc0x0123), 0); c0 += 2;
        vst1_lane_u32(__builtin_assume_aligned(c1, 1), vreinterpret_u32_f16(vacc1x0123), 0); c1 += 2;
        vst1_lane_u32(__builtin_assume_aligned(c2, 1), vreinterpret_u32_f16(vacc2x0123), 0); c2 += 2;
        vst1_lane_u32(__builtin_assume_aligned(c3, 1), vreinterpret_u32_f16(vacc3x0123), 0); c3 += 2;
        vst1_lane_u32(__builtin_assume_aligned(c4, 1), vreinterpret_u32_f16(vacc4x0123), 0); c4 += 2;
        vst1_lane_u32(__builtin_assume_aligned(c5, 1), vreinterpret_u32_f16(vacc5x0123), 0); c5 += 2;

        vacc0x0123 = vext_f16(vacc0x0123, vacc0x0123, 2);
        vacc1x0123 = vext_f16(vacc1x0123, vacc1x0123, 2);
        vacc2x0123 = vext_f16(vacc2x0123, vacc2x0123, 2);
        vacc3x0123 = vext_f16(vacc3x0123, vacc3x0123, 2);
        vacc4x0123 = vext_f16(vacc4x0123, vacc4x0123, 2);
        vacc5x0123 = vext_f16(vacc5x0123, vacc5x0123, 2);
      }
      if (nc & 1) {
        vst1_lane_f16(c0, vacc0x0123, 0);
        vst1_lane_f16(c1, vacc1x0123, 0);
        vst1_lane_f16(c2, vacc2x0123, 0);
        vst1_lane_f16(c3, vacc3x0123, 0);
        vst1_lane_f16(c4, vacc4x0123, 0);
        vst1_lane_f16(c5, vacc5x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
