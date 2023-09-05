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

#include <xnnpack/igemm.h>
#include <xnnpack/intrinsics-polyfill.h>


void xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const void** restrict a,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(ks != 0);
  assert(ks % (6 * sizeof(void*)) == 0);
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
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  uint16_t* c4 = (uint16_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  uint16_t* c5 = (uint16_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    c5 = c4;
  }

  do {
    float16x8_t vacc0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
    float16x8_t vacc0x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
    float16x8_t vacc1x01234567 = vacc0x01234567;
    float16x8_t vacc1x89ABCDEF = vacc0x89ABCDEF;
    float16x8_t vacc2x01234567 = vacc0x01234567;
    float16x8_t vacc2x89ABCDEF = vacc0x89ABCDEF;
    float16x8_t vacc3x01234567 = vacc0x01234567;
    float16x8_t vacc3x89ABCDEF = vacc0x89ABCDEF;
    float16x8_t vacc4x01234567 = vacc0x01234567;
    float16x8_t vacc4x89ABCDEF = vacc0x89ABCDEF;
    float16x8_t vacc5x01234567 = vacc0x01234567;
    float16x8_t vacc5x89ABCDEF = vacc0x89ABCDEF;

    size_t p = ks;
    do {
      const uint16_t* restrict a0 = (const uint16_t*) a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint16_t*) ((uintptr_t) a0 + a_offset);
      }
      const uint16_t* restrict a1 = (const uint16_t*) a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const uint16_t*) ((uintptr_t) a1 + a_offset);
      }
      const uint16_t* restrict a2 = (const uint16_t*) a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const uint16_t*) ((uintptr_t) a2 + a_offset);
      }
      const uint16_t* restrict a3 = (const uint16_t*) a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const uint16_t*) ((uintptr_t) a3 + a_offset);
      }
      const uint16_t* restrict a4 = (const uint16_t*) a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const uint16_t*) ((uintptr_t) a4 + a_offset);
      }
      const uint16_t* restrict a5 = (const uint16_t*) a[5];
      assert(a5 != NULL);
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const uint16_t*) ((uintptr_t) a5 + a_offset);
      }
      a += 6;

      size_t k = kc;
      for (; k >= 4 * sizeof(uint16_t); k -= 4 * sizeof(uint16_t)) {
        const float16x4_t va0 = vreinterpret_f16_u16(vld1_u16(a0)); a0 += 4;
        const float16x4_t va1 = vreinterpret_f16_u16(vld1_u16(a1)); a1 += 4;
        const float16x4_t va2 = vreinterpret_f16_u16(vld1_u16(a2)); a2 += 4;
        const float16x4_t va3 = vreinterpret_f16_u16(vld1_u16(a3)); a3 += 4;
        const float16x4_t va4 = vreinterpret_f16_u16(vld1_u16(a4)); a4 += 4;
        const float16x4_t va5 = vreinterpret_f16_u16(vld1_u16(a5)); a5 += 4;

        const float16x8_t vb01234567c0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
        const float16x8_t vb89ABCDEFc0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        #if XNN_ARCH_ARM64
          vacc0x01234567 = vfmaq_lane_f16(vacc0x01234567, vb01234567c0, va0, 0);
          vacc1x01234567 = vfmaq_lane_f16(vacc1x01234567, vb01234567c0, va1, 0);
          vacc2x01234567 = vfmaq_lane_f16(vacc2x01234567, vb01234567c0, va2, 0);
          vacc3x01234567 = vfmaq_lane_f16(vacc3x01234567, vb01234567c0, va3, 0);
          vacc4x01234567 = vfmaq_lane_f16(vacc4x01234567, vb01234567c0, va4, 0);
          vacc5x01234567 = vfmaq_lane_f16(vacc5x01234567, vb01234567c0, va5, 0);
          vacc0x89ABCDEF = vfmaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc0, va0, 0);
          vacc1x89ABCDEF = vfmaq_lane_f16(vacc1x89ABCDEF, vb89ABCDEFc0, va1, 0);
          vacc2x89ABCDEF = vfmaq_lane_f16(vacc2x89ABCDEF, vb89ABCDEFc0, va2, 0);
          vacc3x89ABCDEF = vfmaq_lane_f16(vacc3x89ABCDEF, vb89ABCDEFc0, va3, 0);
          vacc4x89ABCDEF = vfmaq_lane_f16(vacc4x89ABCDEF, vb89ABCDEFc0, va4, 0);
          vacc5x89ABCDEF = vfmaq_lane_f16(vacc5x89ABCDEF, vb89ABCDEFc0, va5, 0);
        #else
          vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567c0, va0, 0);
          vacc1x01234567 = vmlaq_lane_f16(vacc1x01234567, vb01234567c0, va1, 0);
          vacc2x01234567 = vmlaq_lane_f16(vacc2x01234567, vb01234567c0, va2, 0);
          vacc3x01234567 = vmlaq_lane_f16(vacc3x01234567, vb01234567c0, va3, 0);
          vacc4x01234567 = vmlaq_lane_f16(vacc4x01234567, vb01234567c0, va4, 0);
          vacc5x01234567 = vmlaq_lane_f16(vacc5x01234567, vb01234567c0, va5, 0);
          vacc0x89ABCDEF = vmlaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc0, va0, 0);
          vacc1x89ABCDEF = vmlaq_lane_f16(vacc1x89ABCDEF, vb89ABCDEFc0, va1, 0);
          vacc2x89ABCDEF = vmlaq_lane_f16(vacc2x89ABCDEF, vb89ABCDEFc0, va2, 0);
          vacc3x89ABCDEF = vmlaq_lane_f16(vacc3x89ABCDEF, vb89ABCDEFc0, va3, 0);
          vacc4x89ABCDEF = vmlaq_lane_f16(vacc4x89ABCDEF, vb89ABCDEFc0, va4, 0);
          vacc5x89ABCDEF = vmlaq_lane_f16(vacc5x89ABCDEF, vb89ABCDEFc0, va5, 0);
        #endif
        const float16x8_t vb01234567c1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
        const float16x8_t vb89ABCDEFc1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        #if XNN_ARCH_ARM64
          vacc0x01234567 = vfmaq_lane_f16(vacc0x01234567, vb01234567c1, va0, 1);
          vacc1x01234567 = vfmaq_lane_f16(vacc1x01234567, vb01234567c1, va1, 1);
          vacc2x01234567 = vfmaq_lane_f16(vacc2x01234567, vb01234567c1, va2, 1);
          vacc3x01234567 = vfmaq_lane_f16(vacc3x01234567, vb01234567c1, va3, 1);
          vacc4x01234567 = vfmaq_lane_f16(vacc4x01234567, vb01234567c1, va4, 1);
          vacc5x01234567 = vfmaq_lane_f16(vacc5x01234567, vb01234567c1, va5, 1);
          vacc0x89ABCDEF = vfmaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc1, va0, 1);
          vacc1x89ABCDEF = vfmaq_lane_f16(vacc1x89ABCDEF, vb89ABCDEFc1, va1, 1);
          vacc2x89ABCDEF = vfmaq_lane_f16(vacc2x89ABCDEF, vb89ABCDEFc1, va2, 1);
          vacc3x89ABCDEF = vfmaq_lane_f16(vacc3x89ABCDEF, vb89ABCDEFc1, va3, 1);
          vacc4x89ABCDEF = vfmaq_lane_f16(vacc4x89ABCDEF, vb89ABCDEFc1, va4, 1);
          vacc5x89ABCDEF = vfmaq_lane_f16(vacc5x89ABCDEF, vb89ABCDEFc1, va5, 1);
        #else
          vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567c1, va0, 1);
          vacc1x01234567 = vmlaq_lane_f16(vacc1x01234567, vb01234567c1, va1, 1);
          vacc2x01234567 = vmlaq_lane_f16(vacc2x01234567, vb01234567c1, va2, 1);
          vacc3x01234567 = vmlaq_lane_f16(vacc3x01234567, vb01234567c1, va3, 1);
          vacc4x01234567 = vmlaq_lane_f16(vacc4x01234567, vb01234567c1, va4, 1);
          vacc5x01234567 = vmlaq_lane_f16(vacc5x01234567, vb01234567c1, va5, 1);
          vacc0x89ABCDEF = vmlaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc1, va0, 1);
          vacc1x89ABCDEF = vmlaq_lane_f16(vacc1x89ABCDEF, vb89ABCDEFc1, va1, 1);
          vacc2x89ABCDEF = vmlaq_lane_f16(vacc2x89ABCDEF, vb89ABCDEFc1, va2, 1);
          vacc3x89ABCDEF = vmlaq_lane_f16(vacc3x89ABCDEF, vb89ABCDEFc1, va3, 1);
          vacc4x89ABCDEF = vmlaq_lane_f16(vacc4x89ABCDEF, vb89ABCDEFc1, va4, 1);
          vacc5x89ABCDEF = vmlaq_lane_f16(vacc5x89ABCDEF, vb89ABCDEFc1, va5, 1);
        #endif
        const float16x8_t vb01234567c2 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
        const float16x8_t vb89ABCDEFc2 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        #if XNN_ARCH_ARM64
          vacc0x01234567 = vfmaq_lane_f16(vacc0x01234567, vb01234567c2, va0, 2);
          vacc1x01234567 = vfmaq_lane_f16(vacc1x01234567, vb01234567c2, va1, 2);
          vacc2x01234567 = vfmaq_lane_f16(vacc2x01234567, vb01234567c2, va2, 2);
          vacc3x01234567 = vfmaq_lane_f16(vacc3x01234567, vb01234567c2, va3, 2);
          vacc4x01234567 = vfmaq_lane_f16(vacc4x01234567, vb01234567c2, va4, 2);
          vacc5x01234567 = vfmaq_lane_f16(vacc5x01234567, vb01234567c2, va5, 2);
          vacc0x89ABCDEF = vfmaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc2, va0, 2);
          vacc1x89ABCDEF = vfmaq_lane_f16(vacc1x89ABCDEF, vb89ABCDEFc2, va1, 2);
          vacc2x89ABCDEF = vfmaq_lane_f16(vacc2x89ABCDEF, vb89ABCDEFc2, va2, 2);
          vacc3x89ABCDEF = vfmaq_lane_f16(vacc3x89ABCDEF, vb89ABCDEFc2, va3, 2);
          vacc4x89ABCDEF = vfmaq_lane_f16(vacc4x89ABCDEF, vb89ABCDEFc2, va4, 2);
          vacc5x89ABCDEF = vfmaq_lane_f16(vacc5x89ABCDEF, vb89ABCDEFc2, va5, 2);
        #else
          vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567c2, va0, 2);
          vacc1x01234567 = vmlaq_lane_f16(vacc1x01234567, vb01234567c2, va1, 2);
          vacc2x01234567 = vmlaq_lane_f16(vacc2x01234567, vb01234567c2, va2, 2);
          vacc3x01234567 = vmlaq_lane_f16(vacc3x01234567, vb01234567c2, va3, 2);
          vacc4x01234567 = vmlaq_lane_f16(vacc4x01234567, vb01234567c2, va4, 2);
          vacc5x01234567 = vmlaq_lane_f16(vacc5x01234567, vb01234567c2, va5, 2);
          vacc0x89ABCDEF = vmlaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc2, va0, 2);
          vacc1x89ABCDEF = vmlaq_lane_f16(vacc1x89ABCDEF, vb89ABCDEFc2, va1, 2);
          vacc2x89ABCDEF = vmlaq_lane_f16(vacc2x89ABCDEF, vb89ABCDEFc2, va2, 2);
          vacc3x89ABCDEF = vmlaq_lane_f16(vacc3x89ABCDEF, vb89ABCDEFc2, va3, 2);
          vacc4x89ABCDEF = vmlaq_lane_f16(vacc4x89ABCDEF, vb89ABCDEFc2, va4, 2);
          vacc5x89ABCDEF = vmlaq_lane_f16(vacc5x89ABCDEF, vb89ABCDEFc2, va5, 2);
        #endif
        const float16x8_t vb01234567c3 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
        const float16x8_t vb89ABCDEFc3 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        #if XNN_ARCH_ARM64
          vacc0x01234567 = vfmaq_lane_f16(vacc0x01234567, vb01234567c3, va0, 3);
          vacc1x01234567 = vfmaq_lane_f16(vacc1x01234567, vb01234567c3, va1, 3);
          vacc2x01234567 = vfmaq_lane_f16(vacc2x01234567, vb01234567c3, va2, 3);
          vacc3x01234567 = vfmaq_lane_f16(vacc3x01234567, vb01234567c3, va3, 3);
          vacc4x01234567 = vfmaq_lane_f16(vacc4x01234567, vb01234567c3, va4, 3);
          vacc5x01234567 = vfmaq_lane_f16(vacc5x01234567, vb01234567c3, va5, 3);
          vacc0x89ABCDEF = vfmaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc3, va0, 3);
          vacc1x89ABCDEF = vfmaq_lane_f16(vacc1x89ABCDEF, vb89ABCDEFc3, va1, 3);
          vacc2x89ABCDEF = vfmaq_lane_f16(vacc2x89ABCDEF, vb89ABCDEFc3, va2, 3);
          vacc3x89ABCDEF = vfmaq_lane_f16(vacc3x89ABCDEF, vb89ABCDEFc3, va3, 3);
          vacc4x89ABCDEF = vfmaq_lane_f16(vacc4x89ABCDEF, vb89ABCDEFc3, va4, 3);
          vacc5x89ABCDEF = vfmaq_lane_f16(vacc5x89ABCDEF, vb89ABCDEFc3, va5, 3);
        #else
          vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567c3, va0, 3);
          vacc1x01234567 = vmlaq_lane_f16(vacc1x01234567, vb01234567c3, va1, 3);
          vacc2x01234567 = vmlaq_lane_f16(vacc2x01234567, vb01234567c3, va2, 3);
          vacc3x01234567 = vmlaq_lane_f16(vacc3x01234567, vb01234567c3, va3, 3);
          vacc4x01234567 = vmlaq_lane_f16(vacc4x01234567, vb01234567c3, va4, 3);
          vacc5x01234567 = vmlaq_lane_f16(vacc5x01234567, vb01234567c3, va5, 3);
          vacc0x89ABCDEF = vmlaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc3, va0, 3);
          vacc1x89ABCDEF = vmlaq_lane_f16(vacc1x89ABCDEF, vb89ABCDEFc3, va1, 3);
          vacc2x89ABCDEF = vmlaq_lane_f16(vacc2x89ABCDEF, vb89ABCDEFc3, va2, 3);
          vacc3x89ABCDEF = vmlaq_lane_f16(vacc3x89ABCDEF, vb89ABCDEFc3, va3, 3);
          vacc4x89ABCDEF = vmlaq_lane_f16(vacc4x89ABCDEF, vb89ABCDEFc3, va4, 3);
          vacc5x89ABCDEF = vmlaq_lane_f16(vacc5x89ABCDEF, vb89ABCDEFc3, va5, 3);
        #endif
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const float16x8_t va0 = vreinterpretq_f16_u16(vld1q_dup_u16(a0)); a0 += 1;
          const float16x8_t va1 = vreinterpretq_f16_u16(vld1q_dup_u16(a1)); a1 += 1;
          const float16x8_t va2 = vreinterpretq_f16_u16(vld1q_dup_u16(a2)); a2 += 1;
          const float16x8_t va3 = vreinterpretq_f16_u16(vld1q_dup_u16(a3)); a3 += 1;
          const float16x8_t va4 = vreinterpretq_f16_u16(vld1q_dup_u16(a4)); a4 += 1;
          const float16x8_t va5 = vreinterpretq_f16_u16(vld1q_dup_u16(a5)); a5 += 1;

          const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
          const float16x8_t vb89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

          vacc0x01234567 = vfmaq_f16(vacc0x01234567, va0, vb01234567);
          vacc1x01234567 = vfmaq_f16(vacc1x01234567, va1, vb01234567);
          vacc2x01234567 = vfmaq_f16(vacc2x01234567, va2, vb01234567);
          vacc3x01234567 = vfmaq_f16(vacc3x01234567, va3, vb01234567);
          vacc4x01234567 = vfmaq_f16(vacc4x01234567, va4, vb01234567);
          vacc5x01234567 = vfmaq_f16(vacc5x01234567, va5, vb01234567);
          vacc0x89ABCDEF = vfmaq_f16(vacc0x89ABCDEF, va0, vb89ABCDEF);
          vacc1x89ABCDEF = vfmaq_f16(vacc1x89ABCDEF, va1, vb89ABCDEF);
          vacc2x89ABCDEF = vfmaq_f16(vacc2x89ABCDEF, va2, vb89ABCDEF);
          vacc3x89ABCDEF = vfmaq_f16(vacc3x89ABCDEF, va3, vb89ABCDEF);
          vacc4x89ABCDEF = vfmaq_f16(vacc4x89ABCDEF, va4, vb89ABCDEF);
          vacc5x89ABCDEF = vfmaq_f16(vacc5x89ABCDEF, va5, vb89ABCDEF);

          k -= sizeof(uint16_t);
        } while (k != 0);
      }
      p -= 6 * sizeof(void*);
    } while (p != 0);


    const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
    vacc0x01234567 = vmaxq_f16(vacc0x01234567, vmin);
    vacc1x01234567 = vmaxq_f16(vacc1x01234567, vmin);
    vacc2x01234567 = vmaxq_f16(vacc2x01234567, vmin);
    vacc3x01234567 = vmaxq_f16(vacc3x01234567, vmin);
    vacc4x01234567 = vmaxq_f16(vacc4x01234567, vmin);
    vacc5x01234567 = vmaxq_f16(vacc5x01234567, vmin);
    vacc0x89ABCDEF = vmaxq_f16(vacc0x89ABCDEF, vmin);
    vacc1x89ABCDEF = vmaxq_f16(vacc1x89ABCDEF, vmin);
    vacc2x89ABCDEF = vmaxq_f16(vacc2x89ABCDEF, vmin);
    vacc3x89ABCDEF = vmaxq_f16(vacc3x89ABCDEF, vmin);
    vacc4x89ABCDEF = vmaxq_f16(vacc4x89ABCDEF, vmin);
    vacc5x89ABCDEF = vmaxq_f16(vacc5x89ABCDEF, vmin);

    const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
    vacc0x01234567 = vminq_f16(vacc0x01234567, vmax);
    vacc1x01234567 = vminq_f16(vacc1x01234567, vmax);
    vacc2x01234567 = vminq_f16(vacc2x01234567, vmax);
    vacc3x01234567 = vminq_f16(vacc3x01234567, vmax);
    vacc4x01234567 = vminq_f16(vacc4x01234567, vmax);
    vacc5x01234567 = vminq_f16(vacc5x01234567, vmax);
    vacc0x89ABCDEF = vminq_f16(vacc0x89ABCDEF, vmax);
    vacc1x89ABCDEF = vminq_f16(vacc1x89ABCDEF, vmax);
    vacc2x89ABCDEF = vminq_f16(vacc2x89ABCDEF, vmax);
    vacc3x89ABCDEF = vminq_f16(vacc3x89ABCDEF, vmax);
    vacc4x89ABCDEF = vminq_f16(vacc4x89ABCDEF, vmax);
    vacc5x89ABCDEF = vminq_f16(vacc5x89ABCDEF, vmax);

    if XNN_LIKELY(nc >= 16) {
      vst1q_u16(c5, vreinterpretq_u16_f16(vacc5x01234567));
      vst1q_u16(c5 + 8, vreinterpretq_u16_f16(vacc5x89ABCDEF));
      c5 = (uint16_t*) ((uintptr_t) c5 + cn_stride);
      vst1q_u16(c4, vreinterpretq_u16_f16(vacc4x01234567));
      vst1q_u16(c4 + 8, vreinterpretq_u16_f16(vacc4x89ABCDEF));
      c4 = (uint16_t*) ((uintptr_t) c4 + cn_stride);
      vst1q_u16(c3, vreinterpretq_u16_f16(vacc3x01234567));
      vst1q_u16(c3 + 8, vreinterpretq_u16_f16(vacc3x89ABCDEF));
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      vst1q_u16(c2, vreinterpretq_u16_f16(vacc2x01234567));
      vst1q_u16(c2 + 8, vreinterpretq_u16_f16(vacc2x89ABCDEF));
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      vst1q_u16(c1, vreinterpretq_u16_f16(vacc1x01234567));
      vst1q_u16(c1 + 8, vreinterpretq_u16_f16(vacc1x89ABCDEF));
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      vst1q_u16(c0, vreinterpretq_u16_f16(vacc0x01234567));
      vst1q_u16(c0 + 8, vreinterpretq_u16_f16(vacc0x89ABCDEF));
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a = (const void**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc & 8) {
        vst1q_u16(c5, vreinterpretq_u16_f16(vacc5x01234567)); c5 += 8;
        vst1q_u16(c4, vreinterpretq_u16_f16(vacc4x01234567)); c4 += 8;
        vst1q_u16(c3, vreinterpretq_u16_f16(vacc3x01234567)); c3 += 8;
        vst1q_u16(c2, vreinterpretq_u16_f16(vacc2x01234567)); c2 += 8;
        vst1q_u16(c1, vreinterpretq_u16_f16(vacc1x01234567)); c1 += 8;
        vst1q_u16(c0, vreinterpretq_u16_f16(vacc0x01234567)); c0 += 8;

        vacc5x01234567 = vacc5x89ABCDEF;
        vacc4x01234567 = vacc4x89ABCDEF;
        vacc3x01234567 = vacc3x89ABCDEF;
        vacc2x01234567 = vacc2x89ABCDEF;
        vacc1x01234567 = vacc1x89ABCDEF;
        vacc0x01234567 = vacc0x89ABCDEF;
      }
      float16x4_t vacc5x0123 = vget_low_f16(vacc5x01234567);
      float16x4_t vacc4x0123 = vget_low_f16(vacc4x01234567);
      float16x4_t vacc3x0123 = vget_low_f16(vacc3x01234567);
      float16x4_t vacc2x0123 = vget_low_f16(vacc2x01234567);
      float16x4_t vacc1x0123 = vget_low_f16(vacc1x01234567);
      float16x4_t vacc0x0123 = vget_low_f16(vacc0x01234567);
      if (nc & 4) {
        vst1_u16(c5, vreinterpret_u16_f16(vacc5x0123)); c5 += 4;
        vst1_u16(c4, vreinterpret_u16_f16(vacc4x0123)); c4 += 4;
        vst1_u16(c3, vreinterpret_u16_f16(vacc3x0123)); c3 += 4;
        vst1_u16(c2, vreinterpret_u16_f16(vacc2x0123)); c2 += 4;
        vst1_u16(c1, vreinterpret_u16_f16(vacc1x0123)); c1 += 4;
        vst1_u16(c0, vreinterpret_u16_f16(vacc0x0123)); c0 += 4;

        vacc5x0123 = vget_high_f16(vacc5x01234567);
        vacc4x0123 = vget_high_f16(vacc4x01234567);
        vacc3x0123 = vget_high_f16(vacc3x01234567);
        vacc2x0123 = vget_high_f16(vacc2x01234567);
        vacc1x0123 = vget_high_f16(vacc1x01234567);
        vacc0x0123 = vget_high_f16(vacc0x01234567);
      }
      if (nc & 2) {
        vst1_lane_u32((void*) c5, vreinterpret_u32_f16(vacc5x0123), 0); c5 += 2;
        vst1_lane_u32((void*) c4, vreinterpret_u32_f16(vacc4x0123), 0); c4 += 2;
        vst1_lane_u32((void*) c3, vreinterpret_u32_f16(vacc3x0123), 0); c3 += 2;
        vst1_lane_u32((void*) c2, vreinterpret_u32_f16(vacc2x0123), 0); c2 += 2;
        vst1_lane_u32((void*) c1, vreinterpret_u32_f16(vacc1x0123), 0); c1 += 2;
        vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vacc0x0123), 0); c0 += 2;

        vacc5x0123 = vext_f16(vacc5x0123, vacc5x0123, 2);
        vacc4x0123 = vext_f16(vacc4x0123, vacc4x0123, 2);
        vacc3x0123 = vext_f16(vacc3x0123, vacc3x0123, 2);
        vacc2x0123 = vext_f16(vacc2x0123, vacc2x0123, 2);
        vacc1x0123 = vext_f16(vacc1x0123, vacc1x0123, 2);
        vacc0x0123 = vext_f16(vacc0x0123, vacc0x0123, 2);
      }
      if (nc & 1) {
        vst1_lane_u16(c5, vreinterpret_u16_f16(vacc5x0123), 0);
        vst1_lane_u16(c4, vreinterpret_u16_f16(vacc4x0123), 0);
        vst1_lane_u16(c3, vreinterpret_u16_f16(vacc3x0123), 0);
        vst1_lane_u16(c2, vreinterpret_u16_f16(vacc2x0123), 0);
        vst1_lane_u16(c1, vreinterpret_u16_f16(vacc1x0123), 0);
        vst1_lane_u16(c0, vreinterpret_u16_f16(vacc0x0123), 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
