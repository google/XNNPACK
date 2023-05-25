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
#include <xnnpack/intrinsics-polyfill.h>


void xnn_f16_gemminc_minmax_ukernel_1x16__neonfp16arith_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* restrict a,
    size_t a_stride,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const void* restrict acc,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);
  assert(acc != NULL);

  const uint16_t* a0 = (const uint16_t*) a;
  uint16_t* c0 = (uint16_t*) c;

  do {
    float16x8_t vacc0x01234567 = vreinterpretq_f16_u16(vld1q_u16(acc)); acc = (const void*) ((uintptr_t) acc + sizeof(float16x8_t));
    float16x8_t vacc0x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(acc)); acc = (const void*) ((uintptr_t) acc + sizeof(float16x8_t));

    size_t k = kc;
    while (k >= 4 * sizeof(uint16_t)) {
      const float16x4_t va0 = vreinterpret_f16_u16(vld1_u16(a0)); a0 += 4;

      const float16x8_t vb01234567c0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));
      const float16x8_t vb89ABCDEFc0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));

      #if XNN_ARCH_ARM64
        vacc0x01234567 = vfmaq_lane_f16(vacc0x01234567, vb01234567c0, va0, 0);
        vacc0x89ABCDEF = vfmaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc0, va0, 0);
      #else
        vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567c0, va0, 0);
        vacc0x89ABCDEF = vmlaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc0, va0, 0);
      #endif
      const float16x8_t vb01234567c1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));
      const float16x8_t vb89ABCDEFc1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));

      #if XNN_ARCH_ARM64
        vacc0x01234567 = vfmaq_lane_f16(vacc0x01234567, vb01234567c1, va0, 1);
        vacc0x89ABCDEF = vfmaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc1, va0, 1);
      #else
        vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567c1, va0, 1);
        vacc0x89ABCDEF = vmlaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc1, va0, 1);
      #endif
      const float16x8_t vb01234567c2 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));
      const float16x8_t vb89ABCDEFc2 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));

      #if XNN_ARCH_ARM64
        vacc0x01234567 = vfmaq_lane_f16(vacc0x01234567, vb01234567c2, va0, 2);
        vacc0x89ABCDEF = vfmaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc2, va0, 2);
      #else
        vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567c2, va0, 2);
        vacc0x89ABCDEF = vmlaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc2, va0, 2);
      #endif
      const float16x8_t vb01234567c3 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));
      const float16x8_t vb89ABCDEFc3 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));

      #if XNN_ARCH_ARM64
        vacc0x01234567 = vfmaq_lane_f16(vacc0x01234567, vb01234567c3, va0, 3);
        vacc0x89ABCDEF = vfmaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc3, va0, 3);
      #else
        vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567c3, va0, 3);
        vacc0x89ABCDEF = vmlaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc3, va0, 3);
      #endif

      k -= 4 * sizeof(uint16_t);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const float16x8_t va0 = vreinterpretq_f16_u16(vld1q_dup_u16(a0)); a0 += 1;

        const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));
        const float16x8_t vb89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));

        vacc0x01234567 = vfmaq_f16(vacc0x01234567, va0, vb01234567);
        vacc0x89ABCDEF = vfmaq_f16(vacc0x89ABCDEF, va0, vb89ABCDEF);

        k -= sizeof(uint16_t);
      } while (k != 0);
    }


    const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
    vacc0x01234567 = vmaxq_f16(vacc0x01234567, vmin);
    vacc0x89ABCDEF = vmaxq_f16(vacc0x89ABCDEF, vmin);

    const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
    vacc0x01234567 = vminq_f16(vacc0x01234567, vmax);
    vacc0x89ABCDEF = vminq_f16(vacc0x89ABCDEF, vmax);

    if XNN_LIKELY(nc >= 16) {
      vst1q_u16(c0, vreinterpretq_u16_f16(vacc0x01234567));
      vst1q_u16(c0 + 8, vreinterpretq_u16_f16(vacc0x89ABCDEF));
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const uint16_t*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      if (nc & 8) {
        vst1q_u16(c0, vreinterpretq_u16_f16(vacc0x01234567)); c0 += 8;

        vacc0x01234567 = vacc0x89ABCDEF;
      }
      float16x4_t vacc0x0123 = vget_low_f16(vacc0x01234567);
      if (nc & 4) {
        vst1_u16(c0, vreinterpret_u16_f16(vacc0x0123)); c0 += 4;

        vacc0x0123 = vget_high_f16(vacc0x01234567);
      }
      if (nc & 2) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vacc0x0123), 0); c0 += 2;

        vacc0x0123 = vext_f16(vacc0x0123, vacc0x0123, 2);
      }
      if (nc & 1) {
        vst1_lane_u16(c0, vreinterpret_u16_f16(vacc0x0123), 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
