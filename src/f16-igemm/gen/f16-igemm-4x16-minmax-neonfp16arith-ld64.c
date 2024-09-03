// Auto-generated file. Do not edit!
//   Template: src/f16-igemm/neonfp16arith-ld64.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/igemm.h"
#include "xnnpack/intrinsics-polyfill.h"


void xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const xnn_float16** restrict a,
    const xnn_float16* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const xnn_float16* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  uint16_t* c0 = (uint16_t*) c;
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  uint16_t* c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    float16x8_t vacc0x0 = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) w)); w = (const xnn_float16*) w + 8;
    float16x8_t vacc0x1 = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) w)); w = (const xnn_float16*) w + 8;
    float16x8_t vacc1x0 = vacc0x0;
    float16x8_t vacc1x1 = vacc0x1;
    float16x8_t vacc2x0 = vacc0x0;
    float16x8_t vacc2x1 = vacc0x1;
    float16x8_t vacc3x0 = vacc0x0;
    float16x8_t vacc3x1 = vacc0x1;

    size_t p = ks;
    do {
      const uint16_t* restrict a0 = (const uint16_t*) a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != (const uint16_t*) zero) {
        a0 = (const uint16_t*) ((uintptr_t) a0 + a_offset);
      }
      const uint16_t* restrict a1 = (const uint16_t*) a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != (const uint16_t*) zero) {
        a1 = (const uint16_t*) ((uintptr_t) a1 + a_offset);
      }
      const uint16_t* restrict a2 = (const uint16_t*) a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != (const uint16_t*) zero) {
        a2 = (const uint16_t*) ((uintptr_t) a2 + a_offset);
      }
      const uint16_t* restrict a3 = (const uint16_t*) a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != (const uint16_t*) zero) {
        a3 = (const uint16_t*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      for (; k >= 4 * sizeof(uint16_t); k -= 4 * sizeof(uint16_t)) {
        const float16x4_t va0 = vreinterpret_f16_u16(vld1_u16(a0)); a0 += 4;
        const float16x4_t va1 = vreinterpret_f16_u16(vld1_u16(a1)); a1 += 4;
        const float16x4_t va2 = vreinterpret_f16_u16(vld1_u16(a2)); a2 += 4;
        const float16x4_t va3 = vreinterpret_f16_u16(vld1_u16(a3)); a3 += 4;

        const float16x8_t vb0c0 = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) w)); w = (const xnn_float16*) w + 8;
        const float16x8_t vb1c0 = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) w)); w = (const xnn_float16*) w + 8;

        #if XNN_ARCH_ARM64
          vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c0, va0, 0);
          vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c0, va1, 0);
          vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c0, va2, 0);
          vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c0, va3, 0);
          vacc0x1 = vfmaq_lane_f16(vacc0x1, vb1c0, va0, 0);
          vacc1x1 = vfmaq_lane_f16(vacc1x1, vb1c0, va1, 0);
          vacc2x1 = vfmaq_lane_f16(vacc2x1, vb1c0, va2, 0);
          vacc3x1 = vfmaq_lane_f16(vacc3x1, vb1c0, va3, 0);
        #else
          vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c0, va0, 0);
          vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c0, va1, 0);
          vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c0, va2, 0);
          vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c0, va3, 0);
          vacc0x1 = vmlaq_lane_f16(vacc0x1, vb1c0, va0, 0);
          vacc1x1 = vmlaq_lane_f16(vacc1x1, vb1c0, va1, 0);
          vacc2x1 = vmlaq_lane_f16(vacc2x1, vb1c0, va2, 0);
          vacc3x1 = vmlaq_lane_f16(vacc3x1, vb1c0, va3, 0);
        #endif
        const float16x8_t vb0c1 = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) w)); w = (const xnn_float16*) w + 8;
        const float16x8_t vb1c1 = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) w)); w = (const xnn_float16*) w + 8;

        #if XNN_ARCH_ARM64
          vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c1, va0, 1);
          vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c1, va1, 1);
          vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c1, va2, 1);
          vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c1, va3, 1);
          vacc0x1 = vfmaq_lane_f16(vacc0x1, vb1c1, va0, 1);
          vacc1x1 = vfmaq_lane_f16(vacc1x1, vb1c1, va1, 1);
          vacc2x1 = vfmaq_lane_f16(vacc2x1, vb1c1, va2, 1);
          vacc3x1 = vfmaq_lane_f16(vacc3x1, vb1c1, va3, 1);
        #else
          vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c1, va0, 1);
          vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c1, va1, 1);
          vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c1, va2, 1);
          vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c1, va3, 1);
          vacc0x1 = vmlaq_lane_f16(vacc0x1, vb1c1, va0, 1);
          vacc1x1 = vmlaq_lane_f16(vacc1x1, vb1c1, va1, 1);
          vacc2x1 = vmlaq_lane_f16(vacc2x1, vb1c1, va2, 1);
          vacc3x1 = vmlaq_lane_f16(vacc3x1, vb1c1, va3, 1);
        #endif
        const float16x8_t vb0c2 = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) w)); w = (const xnn_float16*) w + 8;
        const float16x8_t vb1c2 = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) w)); w = (const xnn_float16*) w + 8;

        #if XNN_ARCH_ARM64
          vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c2, va0, 2);
          vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c2, va1, 2);
          vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c2, va2, 2);
          vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c2, va3, 2);
          vacc0x1 = vfmaq_lane_f16(vacc0x1, vb1c2, va0, 2);
          vacc1x1 = vfmaq_lane_f16(vacc1x1, vb1c2, va1, 2);
          vacc2x1 = vfmaq_lane_f16(vacc2x1, vb1c2, va2, 2);
          vacc3x1 = vfmaq_lane_f16(vacc3x1, vb1c2, va3, 2);
        #else
          vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c2, va0, 2);
          vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c2, va1, 2);
          vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c2, va2, 2);
          vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c2, va3, 2);
          vacc0x1 = vmlaq_lane_f16(vacc0x1, vb1c2, va0, 2);
          vacc1x1 = vmlaq_lane_f16(vacc1x1, vb1c2, va1, 2);
          vacc2x1 = vmlaq_lane_f16(vacc2x1, vb1c2, va2, 2);
          vacc3x1 = vmlaq_lane_f16(vacc3x1, vb1c2, va3, 2);
        #endif
        const float16x8_t vb0c3 = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) w)); w = (const xnn_float16*) w + 8;
        const float16x8_t vb1c3 = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) w)); w = (const xnn_float16*) w + 8;

        #if XNN_ARCH_ARM64
          vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c3, va0, 3);
          vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c3, va1, 3);
          vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c3, va2, 3);
          vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c3, va3, 3);
          vacc0x1 = vfmaq_lane_f16(vacc0x1, vb1c3, va0, 3);
          vacc1x1 = vfmaq_lane_f16(vacc1x1, vb1c3, va1, 3);
          vacc2x1 = vfmaq_lane_f16(vacc2x1, vb1c3, va2, 3);
          vacc3x1 = vfmaq_lane_f16(vacc3x1, vb1c3, va3, 3);
        #else
          vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c3, va0, 3);
          vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c3, va1, 3);
          vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c3, va2, 3);
          vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c3, va3, 3);
          vacc0x1 = vmlaq_lane_f16(vacc0x1, vb1c3, va0, 3);
          vacc1x1 = vmlaq_lane_f16(vacc1x1, vb1c3, va1, 3);
          vacc2x1 = vmlaq_lane_f16(vacc2x1, vb1c3, va2, 3);
          vacc3x1 = vmlaq_lane_f16(vacc3x1, vb1c3, va3, 3);
        #endif
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const float16x8_t va0 = vreinterpretq_f16_u16(vld1q_dup_u16(a0)); a0 += 1;
          const float16x8_t va1 = vreinterpretq_f16_u16(vld1q_dup_u16(a1)); a1 += 1;
          const float16x8_t va2 = vreinterpretq_f16_u16(vld1q_dup_u16(a2)); a2 += 1;
          const float16x8_t va3 = vreinterpretq_f16_u16(vld1q_dup_u16(a3)); a3 += 1;

          const float16x8_t vb0 = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) w)); w = (const xnn_float16*) w + 8;
          const float16x8_t vb1 = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) w)); w = (const xnn_float16*) w + 8;

          vacc0x0 = vfmaq_f16(vacc0x0, va0, vb0);
          vacc1x0 = vfmaq_f16(vacc1x0, va1, vb0);
          vacc2x0 = vfmaq_f16(vacc2x0, va2, vb0);
          vacc3x0 = vfmaq_f16(vacc3x0, va3, vb0);
          vacc0x1 = vfmaq_f16(vacc0x1, va0, vb1);
          vacc1x1 = vfmaq_f16(vacc1x1, va1, vb1);
          vacc2x1 = vfmaq_f16(vacc2x1, va2, vb1);
          vacc3x1 = vfmaq_f16(vacc3x1, va3, vb1);

          k -= sizeof(uint16_t);
        } while (k != 0);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);


    const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16((const uint16_t*) &params->scalar.min));
    vacc0x0 = vmaxq_f16(vacc0x0, vmin);
    vacc1x0 = vmaxq_f16(vacc1x0, vmin);
    vacc2x0 = vmaxq_f16(vacc2x0, vmin);
    vacc3x0 = vmaxq_f16(vacc3x0, vmin);
    vacc0x1 = vmaxq_f16(vacc0x1, vmin);
    vacc1x1 = vmaxq_f16(vacc1x1, vmin);
    vacc2x1 = vmaxq_f16(vacc2x1, vmin);
    vacc3x1 = vmaxq_f16(vacc3x1, vmin);

    const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16((const uint16_t*) &params->scalar.max));
    vacc0x0 = vminq_f16(vacc0x0, vmax);
    vacc1x0 = vminq_f16(vacc1x0, vmax);
    vacc2x0 = vminq_f16(vacc2x0, vmax);
    vacc3x0 = vminq_f16(vacc3x0, vmax);
    vacc0x1 = vminq_f16(vacc0x1, vmax);
    vacc1x1 = vminq_f16(vacc1x1, vmax);
    vacc2x1 = vminq_f16(vacc2x1, vmax);
    vacc3x1 = vminq_f16(vacc3x1, vmax);

    if XNN_LIKELY(nc >= 16) {
      vst1q_u16(c3, vreinterpretq_u16_f16(vacc3x0));
      vst1q_u16(c3 + 8, vreinterpretq_u16_f16(vacc3x1));
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      vst1q_u16(c2, vreinterpretq_u16_f16(vacc2x0));
      vst1q_u16(c2 + 8, vreinterpretq_u16_f16(vacc2x1));
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      vst1q_u16(c1, vreinterpretq_u16_f16(vacc1x0));
      vst1q_u16(c1 + 8, vreinterpretq_u16_f16(vacc1x1));
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      vst1q_u16(c0, vreinterpretq_u16_f16(vacc0x0));
      vst1q_u16(c0 + 8, vreinterpretq_u16_f16(vacc0x1));
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a = (const xnn_float16**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc & 8) {
        vst1q_u16(c3, vreinterpretq_u16_f16(vacc3x0)); c3 += 8;
        vst1q_u16(c2, vreinterpretq_u16_f16(vacc2x0)); c2 += 8;
        vst1q_u16(c1, vreinterpretq_u16_f16(vacc1x0)); c1 += 8;
        vst1q_u16(c0, vreinterpretq_u16_f16(vacc0x0)); c0 += 8;

        vacc3x0 = vacc3x1;
        vacc2x0 = vacc2x1;
        vacc1x0 = vacc1x1;
        vacc0x0 = vacc0x1;
      }
      float16x4_t vacc3 = vget_low_f16(vacc3x0);
      float16x4_t vacc2 = vget_low_f16(vacc2x0);
      float16x4_t vacc1 = vget_low_f16(vacc1x0);
      float16x4_t vacc0 = vget_low_f16(vacc0x0);
      if (nc & 4) {
        vst1_u16(c3, vreinterpret_u16_f16(vacc3)); c3 += 4;
        vst1_u16(c2, vreinterpret_u16_f16(vacc2)); c2 += 4;
        vst1_u16(c1, vreinterpret_u16_f16(vacc1)); c1 += 4;
        vst1_u16(c0, vreinterpret_u16_f16(vacc0)); c0 += 4;

        vacc3 = vget_high_f16(vacc3x0);
        vacc2 = vget_high_f16(vacc2x0);
        vacc1 = vget_high_f16(vacc1x0);
        vacc0 = vget_high_f16(vacc0x0);
      }
      if (nc & 2) {
        vst1_lane_u32((void*) c3, vreinterpret_u32_f16(vacc3), 0); c3 += 2;
        vst1_lane_u32((void*) c2, vreinterpret_u32_f16(vacc2), 0); c2 += 2;
        vst1_lane_u32((void*) c1, vreinterpret_u32_f16(vacc1), 0); c1 += 2;
        vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vacc0), 0); c0 += 2;

        vacc3 = vext_f16(vacc3, vacc3, 2);
        vacc2 = vext_f16(vacc2, vacc2, 2);
        vacc1 = vext_f16(vacc1, vacc1, 2);
        vacc0 = vext_f16(vacc0, vacc0, 2);
      }
      if (nc & 1) {
        vst1_lane_u16(c3, vreinterpret_u16_f16(vacc3), 0);
        vst1_lane_u16(c2, vreinterpret_u16_f16(vacc2), 0);
        vst1_lane_u16(c1, vreinterpret_u16_f16(vacc1), 0);
        vst1_lane_u16(c0, vreinterpret_u16_f16(vacc0), 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
