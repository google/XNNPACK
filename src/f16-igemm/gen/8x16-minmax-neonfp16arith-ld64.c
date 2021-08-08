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


void xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64(
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
    const struct xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 8);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(__fp16) == 0);
  assert(ks != 0);
  assert(ks % (8 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(__fp16) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  __fp16* c0 = (__fp16*) c;
  __fp16* c1 = (__fp16*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  __fp16* c2 = (__fp16*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  __fp16* c3 = (__fp16*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  __fp16* c4 = (__fp16*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  __fp16* c5 = (__fp16*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    c5 = c4;
  }
  __fp16* c6 = (__fp16*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    c6 = c5;
  }
  __fp16* c7 = (__fp16*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 8) {
    c7 = c6;
  }

  do {
    float16x8_t vacc0x01234567 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));
    float16x8_t vacc0x89ABCDEF = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));
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
    float16x8_t vacc6x01234567 = vacc0x01234567;
    float16x8_t vacc6x89ABCDEF = vacc0x89ABCDEF;
    float16x8_t vacc7x01234567 = vacc0x01234567;
    float16x8_t vacc7x89ABCDEF = vacc0x89ABCDEF;

    size_t p = ks;
    do {
      const __fp16* restrict a0 = (const __fp16*) a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const __fp16*) ((uintptr_t) a0 + a_offset);
      }
      const __fp16* restrict a1 = (const __fp16*) a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const __fp16*) ((uintptr_t) a1 + a_offset);
      }
      const __fp16* restrict a2 = (const __fp16*) a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const __fp16*) ((uintptr_t) a2 + a_offset);
      }
      const __fp16* restrict a3 = (const __fp16*) a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const __fp16*) ((uintptr_t) a3 + a_offset);
      }
      const __fp16* restrict a4 = (const __fp16*) a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const __fp16*) ((uintptr_t) a4 + a_offset);
      }
      const __fp16* restrict a5 = (const __fp16*) a[5];
      assert(a5 != NULL);
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const __fp16*) ((uintptr_t) a5 + a_offset);
      }
      const __fp16* restrict a6 = (const __fp16*) a[6];
      assert(a6 != NULL);
      if XNN_UNPREDICTABLE(a6 != zero) {
        a6 = (const __fp16*) ((uintptr_t) a6 + a_offset);
      }
      const __fp16* restrict a7 = (const __fp16*) a[7];
      assert(a7 != NULL);
      if XNN_UNPREDICTABLE(a7 != zero) {
        a7 = (const __fp16*) ((uintptr_t) a7 + a_offset);
      }
      a += 8;

      size_t k = kc;
      for (; k >= 4 * sizeof(__fp16); k -= 4 * sizeof(__fp16)) {
        const float16x4_t va0 = vld1_f16(a0); a0 += 4;
        const float16x4_t va1 = vld1_f16(a1); a1 += 4;
        const float16x4_t va2 = vld1_f16(a2); a2 += 4;
        const float16x4_t va3 = vld1_f16(a3); a3 += 4;
        const float16x4_t va4 = vld1_f16(a4); a4 += 4;
        const float16x4_t va5 = vld1_f16(a5); a5 += 4;
        const float16x4_t va6 = vld1_f16(a6); a6 += 4;
        const float16x4_t va7 = vld1_f16(a7); a7 += 4;

        const float16x8_t vb01234567c0 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof( float16x8_t));
        const float16x8_t vb89ABCDEFc0 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof( float16x8_t));

        #if XNN_ARCH_ARM64
          vacc0x01234567 = vfmaq_lane_f16(vacc0x01234567, vb01234567c0, va0, 0);
          vacc1x01234567 = vfmaq_lane_f16(vacc1x01234567, vb01234567c0, va1, 0);
          vacc2x01234567 = vfmaq_lane_f16(vacc2x01234567, vb01234567c0, va2, 0);
          vacc3x01234567 = vfmaq_lane_f16(vacc3x01234567, vb01234567c0, va3, 0);
          vacc4x01234567 = vfmaq_lane_f16(vacc4x01234567, vb01234567c0, va4, 0);
          vacc5x01234567 = vfmaq_lane_f16(vacc5x01234567, vb01234567c0, va5, 0);
          vacc6x01234567 = vfmaq_lane_f16(vacc6x01234567, vb01234567c0, va6, 0);
          vacc7x01234567 = vfmaq_lane_f16(vacc7x01234567, vb01234567c0, va7, 0);
          vacc0x89ABCDEF = vfmaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc0, va0, 0);
          vacc1x89ABCDEF = vfmaq_lane_f16(vacc1x89ABCDEF, vb89ABCDEFc0, va1, 0);
          vacc2x89ABCDEF = vfmaq_lane_f16(vacc2x89ABCDEF, vb89ABCDEFc0, va2, 0);
          vacc3x89ABCDEF = vfmaq_lane_f16(vacc3x89ABCDEF, vb89ABCDEFc0, va3, 0);
          vacc4x89ABCDEF = vfmaq_lane_f16(vacc4x89ABCDEF, vb89ABCDEFc0, va4, 0);
          vacc5x89ABCDEF = vfmaq_lane_f16(vacc5x89ABCDEF, vb89ABCDEFc0, va5, 0);
          vacc6x89ABCDEF = vfmaq_lane_f16(vacc6x89ABCDEF, vb89ABCDEFc0, va6, 0);
          vacc7x89ABCDEF = vfmaq_lane_f16(vacc7x89ABCDEF, vb89ABCDEFc0, va7, 0);
        #else
          const float16x8_t va0c0 = vdupq_lane_f16(va0, 0);
          const float16x8_t va1c0 = vdupq_lane_f16(va1, 0);
          const float16x8_t va2c0 = vdupq_lane_f16(va2, 0);
          const float16x8_t va3c0 = vdupq_lane_f16(va3, 0);
          const float16x8_t va4c0 = vdupq_lane_f16(va4, 0);
          const float16x8_t va5c0 = vdupq_lane_f16(va5, 0);
          const float16x8_t va6c0 = vdupq_lane_f16(va6, 0);
          const float16x8_t va7c0 = vdupq_lane_f16(va7, 0);

          vacc0x01234567 = vfmaq_f16(vacc0x01234567, va0c0, vb01234567c0);
          vacc1x01234567 = vfmaq_f16(vacc1x01234567, va1c0, vb01234567c0);
          vacc2x01234567 = vfmaq_f16(vacc2x01234567, va2c0, vb01234567c0);
          vacc3x01234567 = vfmaq_f16(vacc3x01234567, va3c0, vb01234567c0);
          vacc4x01234567 = vfmaq_f16(vacc4x01234567, va4c0, vb01234567c0);
          vacc5x01234567 = vfmaq_f16(vacc5x01234567, va5c0, vb01234567c0);
          vacc6x01234567 = vfmaq_f16(vacc6x01234567, va6c0, vb01234567c0);
          vacc7x01234567 = vfmaq_f16(vacc7x01234567, va7c0, vb01234567c0);
          vacc0x89ABCDEF = vfmaq_f16(vacc0x89ABCDEF, va0c0, vb89ABCDEFc0);
          vacc1x89ABCDEF = vfmaq_f16(vacc1x89ABCDEF, va1c0, vb89ABCDEFc0);
          vacc2x89ABCDEF = vfmaq_f16(vacc2x89ABCDEF, va2c0, vb89ABCDEFc0);
          vacc3x89ABCDEF = vfmaq_f16(vacc3x89ABCDEF, va3c0, vb89ABCDEFc0);
          vacc4x89ABCDEF = vfmaq_f16(vacc4x89ABCDEF, va4c0, vb89ABCDEFc0);
          vacc5x89ABCDEF = vfmaq_f16(vacc5x89ABCDEF, va5c0, vb89ABCDEFc0);
          vacc6x89ABCDEF = vfmaq_f16(vacc6x89ABCDEF, va6c0, vb89ABCDEFc0);
          vacc7x89ABCDEF = vfmaq_f16(vacc7x89ABCDEF, va7c0, vb89ABCDEFc0);
        #endif
        const float16x8_t vb01234567c1 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof( float16x8_t));
        const float16x8_t vb89ABCDEFc1 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof( float16x8_t));

        #if XNN_ARCH_ARM64
          vacc0x01234567 = vfmaq_lane_f16(vacc0x01234567, vb01234567c1, va0, 1);
          vacc1x01234567 = vfmaq_lane_f16(vacc1x01234567, vb01234567c1, va1, 1);
          vacc2x01234567 = vfmaq_lane_f16(vacc2x01234567, vb01234567c1, va2, 1);
          vacc3x01234567 = vfmaq_lane_f16(vacc3x01234567, vb01234567c1, va3, 1);
          vacc4x01234567 = vfmaq_lane_f16(vacc4x01234567, vb01234567c1, va4, 1);
          vacc5x01234567 = vfmaq_lane_f16(vacc5x01234567, vb01234567c1, va5, 1);
          vacc6x01234567 = vfmaq_lane_f16(vacc6x01234567, vb01234567c1, va6, 1);
          vacc7x01234567 = vfmaq_lane_f16(vacc7x01234567, vb01234567c1, va7, 1);
          vacc0x89ABCDEF = vfmaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc1, va0, 1);
          vacc1x89ABCDEF = vfmaq_lane_f16(vacc1x89ABCDEF, vb89ABCDEFc1, va1, 1);
          vacc2x89ABCDEF = vfmaq_lane_f16(vacc2x89ABCDEF, vb89ABCDEFc1, va2, 1);
          vacc3x89ABCDEF = vfmaq_lane_f16(vacc3x89ABCDEF, vb89ABCDEFc1, va3, 1);
          vacc4x89ABCDEF = vfmaq_lane_f16(vacc4x89ABCDEF, vb89ABCDEFc1, va4, 1);
          vacc5x89ABCDEF = vfmaq_lane_f16(vacc5x89ABCDEF, vb89ABCDEFc1, va5, 1);
          vacc6x89ABCDEF = vfmaq_lane_f16(vacc6x89ABCDEF, vb89ABCDEFc1, va6, 1);
          vacc7x89ABCDEF = vfmaq_lane_f16(vacc7x89ABCDEF, vb89ABCDEFc1, va7, 1);
        #else
          const float16x8_t va0c1 = vdupq_lane_f16(va0, 1);
          const float16x8_t va1c1 = vdupq_lane_f16(va1, 1);
          const float16x8_t va2c1 = vdupq_lane_f16(va2, 1);
          const float16x8_t va3c1 = vdupq_lane_f16(va3, 1);
          const float16x8_t va4c1 = vdupq_lane_f16(va4, 1);
          const float16x8_t va5c1 = vdupq_lane_f16(va5, 1);
          const float16x8_t va6c1 = vdupq_lane_f16(va6, 1);
          const float16x8_t va7c1 = vdupq_lane_f16(va7, 1);

          vacc0x01234567 = vfmaq_f16(vacc0x01234567, va0c1, vb01234567c1);
          vacc1x01234567 = vfmaq_f16(vacc1x01234567, va1c1, vb01234567c1);
          vacc2x01234567 = vfmaq_f16(vacc2x01234567, va2c1, vb01234567c1);
          vacc3x01234567 = vfmaq_f16(vacc3x01234567, va3c1, vb01234567c1);
          vacc4x01234567 = vfmaq_f16(vacc4x01234567, va4c1, vb01234567c1);
          vacc5x01234567 = vfmaq_f16(vacc5x01234567, va5c1, vb01234567c1);
          vacc6x01234567 = vfmaq_f16(vacc6x01234567, va6c1, vb01234567c1);
          vacc7x01234567 = vfmaq_f16(vacc7x01234567, va7c1, vb01234567c1);
          vacc0x89ABCDEF = vfmaq_f16(vacc0x89ABCDEF, va0c1, vb89ABCDEFc1);
          vacc1x89ABCDEF = vfmaq_f16(vacc1x89ABCDEF, va1c1, vb89ABCDEFc1);
          vacc2x89ABCDEF = vfmaq_f16(vacc2x89ABCDEF, va2c1, vb89ABCDEFc1);
          vacc3x89ABCDEF = vfmaq_f16(vacc3x89ABCDEF, va3c1, vb89ABCDEFc1);
          vacc4x89ABCDEF = vfmaq_f16(vacc4x89ABCDEF, va4c1, vb89ABCDEFc1);
          vacc5x89ABCDEF = vfmaq_f16(vacc5x89ABCDEF, va5c1, vb89ABCDEFc1);
          vacc6x89ABCDEF = vfmaq_f16(vacc6x89ABCDEF, va6c1, vb89ABCDEFc1);
          vacc7x89ABCDEF = vfmaq_f16(vacc7x89ABCDEF, va7c1, vb89ABCDEFc1);
        #endif
        const float16x8_t vb01234567c2 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof( float16x8_t));
        const float16x8_t vb89ABCDEFc2 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof( float16x8_t));

        #if XNN_ARCH_ARM64
          vacc0x01234567 = vfmaq_lane_f16(vacc0x01234567, vb01234567c2, va0, 2);
          vacc1x01234567 = vfmaq_lane_f16(vacc1x01234567, vb01234567c2, va1, 2);
          vacc2x01234567 = vfmaq_lane_f16(vacc2x01234567, vb01234567c2, va2, 2);
          vacc3x01234567 = vfmaq_lane_f16(vacc3x01234567, vb01234567c2, va3, 2);
          vacc4x01234567 = vfmaq_lane_f16(vacc4x01234567, vb01234567c2, va4, 2);
          vacc5x01234567 = vfmaq_lane_f16(vacc5x01234567, vb01234567c2, va5, 2);
          vacc6x01234567 = vfmaq_lane_f16(vacc6x01234567, vb01234567c2, va6, 2);
          vacc7x01234567 = vfmaq_lane_f16(vacc7x01234567, vb01234567c2, va7, 2);
          vacc0x89ABCDEF = vfmaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc2, va0, 2);
          vacc1x89ABCDEF = vfmaq_lane_f16(vacc1x89ABCDEF, vb89ABCDEFc2, va1, 2);
          vacc2x89ABCDEF = vfmaq_lane_f16(vacc2x89ABCDEF, vb89ABCDEFc2, va2, 2);
          vacc3x89ABCDEF = vfmaq_lane_f16(vacc3x89ABCDEF, vb89ABCDEFc2, va3, 2);
          vacc4x89ABCDEF = vfmaq_lane_f16(vacc4x89ABCDEF, vb89ABCDEFc2, va4, 2);
          vacc5x89ABCDEF = vfmaq_lane_f16(vacc5x89ABCDEF, vb89ABCDEFc2, va5, 2);
          vacc6x89ABCDEF = vfmaq_lane_f16(vacc6x89ABCDEF, vb89ABCDEFc2, va6, 2);
          vacc7x89ABCDEF = vfmaq_lane_f16(vacc7x89ABCDEF, vb89ABCDEFc2, va7, 2);
        #else
          const float16x8_t va0c2 = vdupq_lane_f16(va0, 2);
          const float16x8_t va1c2 = vdupq_lane_f16(va1, 2);
          const float16x8_t va2c2 = vdupq_lane_f16(va2, 2);
          const float16x8_t va3c2 = vdupq_lane_f16(va3, 2);
          const float16x8_t va4c2 = vdupq_lane_f16(va4, 2);
          const float16x8_t va5c2 = vdupq_lane_f16(va5, 2);
          const float16x8_t va6c2 = vdupq_lane_f16(va6, 2);
          const float16x8_t va7c2 = vdupq_lane_f16(va7, 2);

          vacc0x01234567 = vfmaq_f16(vacc0x01234567, va0c2, vb01234567c2);
          vacc1x01234567 = vfmaq_f16(vacc1x01234567, va1c2, vb01234567c2);
          vacc2x01234567 = vfmaq_f16(vacc2x01234567, va2c2, vb01234567c2);
          vacc3x01234567 = vfmaq_f16(vacc3x01234567, va3c2, vb01234567c2);
          vacc4x01234567 = vfmaq_f16(vacc4x01234567, va4c2, vb01234567c2);
          vacc5x01234567 = vfmaq_f16(vacc5x01234567, va5c2, vb01234567c2);
          vacc6x01234567 = vfmaq_f16(vacc6x01234567, va6c2, vb01234567c2);
          vacc7x01234567 = vfmaq_f16(vacc7x01234567, va7c2, vb01234567c2);
          vacc0x89ABCDEF = vfmaq_f16(vacc0x89ABCDEF, va0c2, vb89ABCDEFc2);
          vacc1x89ABCDEF = vfmaq_f16(vacc1x89ABCDEF, va1c2, vb89ABCDEFc2);
          vacc2x89ABCDEF = vfmaq_f16(vacc2x89ABCDEF, va2c2, vb89ABCDEFc2);
          vacc3x89ABCDEF = vfmaq_f16(vacc3x89ABCDEF, va3c2, vb89ABCDEFc2);
          vacc4x89ABCDEF = vfmaq_f16(vacc4x89ABCDEF, va4c2, vb89ABCDEFc2);
          vacc5x89ABCDEF = vfmaq_f16(vacc5x89ABCDEF, va5c2, vb89ABCDEFc2);
          vacc6x89ABCDEF = vfmaq_f16(vacc6x89ABCDEF, va6c2, vb89ABCDEFc2);
          vacc7x89ABCDEF = vfmaq_f16(vacc7x89ABCDEF, va7c2, vb89ABCDEFc2);
        #endif
        const float16x8_t vb01234567c3 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof( float16x8_t));
        const float16x8_t vb89ABCDEFc3 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof( float16x8_t));

        #if XNN_ARCH_ARM64
          vacc0x01234567 = vfmaq_lane_f16(vacc0x01234567, vb01234567c3, va0, 3);
          vacc1x01234567 = vfmaq_lane_f16(vacc1x01234567, vb01234567c3, va1, 3);
          vacc2x01234567 = vfmaq_lane_f16(vacc2x01234567, vb01234567c3, va2, 3);
          vacc3x01234567 = vfmaq_lane_f16(vacc3x01234567, vb01234567c3, va3, 3);
          vacc4x01234567 = vfmaq_lane_f16(vacc4x01234567, vb01234567c3, va4, 3);
          vacc5x01234567 = vfmaq_lane_f16(vacc5x01234567, vb01234567c3, va5, 3);
          vacc6x01234567 = vfmaq_lane_f16(vacc6x01234567, vb01234567c3, va6, 3);
          vacc7x01234567 = vfmaq_lane_f16(vacc7x01234567, vb01234567c3, va7, 3);
          vacc0x89ABCDEF = vfmaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc3, va0, 3);
          vacc1x89ABCDEF = vfmaq_lane_f16(vacc1x89ABCDEF, vb89ABCDEFc3, va1, 3);
          vacc2x89ABCDEF = vfmaq_lane_f16(vacc2x89ABCDEF, vb89ABCDEFc3, va2, 3);
          vacc3x89ABCDEF = vfmaq_lane_f16(vacc3x89ABCDEF, vb89ABCDEFc3, va3, 3);
          vacc4x89ABCDEF = vfmaq_lane_f16(vacc4x89ABCDEF, vb89ABCDEFc3, va4, 3);
          vacc5x89ABCDEF = vfmaq_lane_f16(vacc5x89ABCDEF, vb89ABCDEFc3, va5, 3);
          vacc6x89ABCDEF = vfmaq_lane_f16(vacc6x89ABCDEF, vb89ABCDEFc3, va6, 3);
          vacc7x89ABCDEF = vfmaq_lane_f16(vacc7x89ABCDEF, vb89ABCDEFc3, va7, 3);
        #else
          const float16x8_t va0c3 = vdupq_lane_f16(va0, 3);
          const float16x8_t va1c3 = vdupq_lane_f16(va1, 3);
          const float16x8_t va2c3 = vdupq_lane_f16(va2, 3);
          const float16x8_t va3c3 = vdupq_lane_f16(va3, 3);
          const float16x8_t va4c3 = vdupq_lane_f16(va4, 3);
          const float16x8_t va5c3 = vdupq_lane_f16(va5, 3);
          const float16x8_t va6c3 = vdupq_lane_f16(va6, 3);
          const float16x8_t va7c3 = vdupq_lane_f16(va7, 3);

          vacc0x01234567 = vfmaq_f16(vacc0x01234567, va0c3, vb01234567c3);
          vacc1x01234567 = vfmaq_f16(vacc1x01234567, va1c3, vb01234567c3);
          vacc2x01234567 = vfmaq_f16(vacc2x01234567, va2c3, vb01234567c3);
          vacc3x01234567 = vfmaq_f16(vacc3x01234567, va3c3, vb01234567c3);
          vacc4x01234567 = vfmaq_f16(vacc4x01234567, va4c3, vb01234567c3);
          vacc5x01234567 = vfmaq_f16(vacc5x01234567, va5c3, vb01234567c3);
          vacc6x01234567 = vfmaq_f16(vacc6x01234567, va6c3, vb01234567c3);
          vacc7x01234567 = vfmaq_f16(vacc7x01234567, va7c3, vb01234567c3);
          vacc0x89ABCDEF = vfmaq_f16(vacc0x89ABCDEF, va0c3, vb89ABCDEFc3);
          vacc1x89ABCDEF = vfmaq_f16(vacc1x89ABCDEF, va1c3, vb89ABCDEFc3);
          vacc2x89ABCDEF = vfmaq_f16(vacc2x89ABCDEF, va2c3, vb89ABCDEFc3);
          vacc3x89ABCDEF = vfmaq_f16(vacc3x89ABCDEF, va3c3, vb89ABCDEFc3);
          vacc4x89ABCDEF = vfmaq_f16(vacc4x89ABCDEF, va4c3, vb89ABCDEFc3);
          vacc5x89ABCDEF = vfmaq_f16(vacc5x89ABCDEF, va5c3, vb89ABCDEFc3);
          vacc6x89ABCDEF = vfmaq_f16(vacc6x89ABCDEF, va6c3, vb89ABCDEFc3);
          vacc7x89ABCDEF = vfmaq_f16(vacc7x89ABCDEF, va7c3, vb89ABCDEFc3);
        #endif
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const float16x8_t va0 = vld1q_dup_f16(a0); a0 += 1;
          const float16x8_t va1 = vld1q_dup_f16(a1); a1 += 1;
          const float16x8_t va2 = vld1q_dup_f16(a2); a2 += 1;
          const float16x8_t va3 = vld1q_dup_f16(a3); a3 += 1;
          const float16x8_t va4 = vld1q_dup_f16(a4); a4 += 1;
          const float16x8_t va5 = vld1q_dup_f16(a5); a5 += 1;
          const float16x8_t va6 = vld1q_dup_f16(a6); a6 += 1;
          const float16x8_t va7 = vld1q_dup_f16(a7); a7 += 1;

          const float16x8_t vb01234567 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));
          const float16x8_t vb89ABCDEF = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));

          vacc0x01234567 = vfmaq_f16(vacc0x01234567, va0, vb01234567);
          vacc1x01234567 = vfmaq_f16(vacc1x01234567, va1, vb01234567);
          vacc2x01234567 = vfmaq_f16(vacc2x01234567, va2, vb01234567);
          vacc3x01234567 = vfmaq_f16(vacc3x01234567, va3, vb01234567);
          vacc4x01234567 = vfmaq_f16(vacc4x01234567, va4, vb01234567);
          vacc5x01234567 = vfmaq_f16(vacc5x01234567, va5, vb01234567);
          vacc6x01234567 = vfmaq_f16(vacc6x01234567, va6, vb01234567);
          vacc7x01234567 = vfmaq_f16(vacc7x01234567, va7, vb01234567);
          vacc0x89ABCDEF = vfmaq_f16(vacc0x89ABCDEF, va0, vb89ABCDEF);
          vacc1x89ABCDEF = vfmaq_f16(vacc1x89ABCDEF, va1, vb89ABCDEF);
          vacc2x89ABCDEF = vfmaq_f16(vacc2x89ABCDEF, va2, vb89ABCDEF);
          vacc3x89ABCDEF = vfmaq_f16(vacc3x89ABCDEF, va3, vb89ABCDEF);
          vacc4x89ABCDEF = vfmaq_f16(vacc4x89ABCDEF, va4, vb89ABCDEF);
          vacc5x89ABCDEF = vfmaq_f16(vacc5x89ABCDEF, va5, vb89ABCDEF);
          vacc6x89ABCDEF = vfmaq_f16(vacc6x89ABCDEF, va6, vb89ABCDEF);
          vacc7x89ABCDEF = vfmaq_f16(vacc7x89ABCDEF, va7, vb89ABCDEF);

          k -= sizeof(__fp16);
        } while (k != 0);
      }
      p -= 8 * sizeof(void*);
    } while (p != 0);

    const float16x8_t vscale = vreinterpretq_f16_u16(vld1q_dup_u16(&params->scale));
    vacc0x01234567 = vmulq_f16(vacc0x01234567, vscale);
    vacc1x01234567 = vmulq_f16(vacc1x01234567, vscale);
    vacc2x01234567 = vmulq_f16(vacc2x01234567, vscale);
    vacc3x01234567 = vmulq_f16(vacc3x01234567, vscale);
    vacc4x01234567 = vmulq_f16(vacc4x01234567, vscale);
    vacc5x01234567 = vmulq_f16(vacc5x01234567, vscale);
    vacc6x01234567 = vmulq_f16(vacc6x01234567, vscale);
    vacc7x01234567 = vmulq_f16(vacc7x01234567, vscale);
    vacc0x89ABCDEF = vmulq_f16(vacc0x89ABCDEF, vscale);
    vacc1x89ABCDEF = vmulq_f16(vacc1x89ABCDEF, vscale);
    vacc2x89ABCDEF = vmulq_f16(vacc2x89ABCDEF, vscale);
    vacc3x89ABCDEF = vmulq_f16(vacc3x89ABCDEF, vscale);
    vacc4x89ABCDEF = vmulq_f16(vacc4x89ABCDEF, vscale);
    vacc5x89ABCDEF = vmulq_f16(vacc5x89ABCDEF, vscale);
    vacc6x89ABCDEF = vmulq_f16(vacc6x89ABCDEF, vscale);
    vacc7x89ABCDEF = vmulq_f16(vacc7x89ABCDEF, vscale);

    const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->max));
    vacc0x01234567 = vminq_f16(vacc0x01234567, vmax);
    vacc1x01234567 = vminq_f16(vacc1x01234567, vmax);
    vacc2x01234567 = vminq_f16(vacc2x01234567, vmax);
    vacc3x01234567 = vminq_f16(vacc3x01234567, vmax);
    vacc4x01234567 = vminq_f16(vacc4x01234567, vmax);
    vacc5x01234567 = vminq_f16(vacc5x01234567, vmax);
    vacc6x01234567 = vminq_f16(vacc6x01234567, vmax);
    vacc7x01234567 = vminq_f16(vacc7x01234567, vmax);
    vacc0x89ABCDEF = vminq_f16(vacc0x89ABCDEF, vmax);
    vacc1x89ABCDEF = vminq_f16(vacc1x89ABCDEF, vmax);
    vacc2x89ABCDEF = vminq_f16(vacc2x89ABCDEF, vmax);
    vacc3x89ABCDEF = vminq_f16(vacc3x89ABCDEF, vmax);
    vacc4x89ABCDEF = vminq_f16(vacc4x89ABCDEF, vmax);
    vacc5x89ABCDEF = vminq_f16(vacc5x89ABCDEF, vmax);
    vacc6x89ABCDEF = vminq_f16(vacc6x89ABCDEF, vmax);
    vacc7x89ABCDEF = vminq_f16(vacc7x89ABCDEF, vmax);

    const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->min));
    vacc0x01234567 = vmaxq_f16(vacc0x01234567, vmin);
    vacc1x01234567 = vmaxq_f16(vacc1x01234567, vmin);
    vacc2x01234567 = vmaxq_f16(vacc2x01234567, vmin);
    vacc3x01234567 = vmaxq_f16(vacc3x01234567, vmin);
    vacc4x01234567 = vmaxq_f16(vacc4x01234567, vmin);
    vacc5x01234567 = vmaxq_f16(vacc5x01234567, vmin);
    vacc6x01234567 = vmaxq_f16(vacc6x01234567, vmin);
    vacc7x01234567 = vmaxq_f16(vacc7x01234567, vmin);
    vacc0x89ABCDEF = vmaxq_f16(vacc0x89ABCDEF, vmin);
    vacc1x89ABCDEF = vmaxq_f16(vacc1x89ABCDEF, vmin);
    vacc2x89ABCDEF = vmaxq_f16(vacc2x89ABCDEF, vmin);
    vacc3x89ABCDEF = vmaxq_f16(vacc3x89ABCDEF, vmin);
    vacc4x89ABCDEF = vmaxq_f16(vacc4x89ABCDEF, vmin);
    vacc5x89ABCDEF = vmaxq_f16(vacc5x89ABCDEF, vmin);
    vacc6x89ABCDEF = vmaxq_f16(vacc6x89ABCDEF, vmin);
    vacc7x89ABCDEF = vmaxq_f16(vacc7x89ABCDEF, vmin);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f16(c7, vacc7x01234567);
      vst1q_f16(c7 + 8, vacc7x89ABCDEF);
      c7 = (__fp16*) ((uintptr_t) c7 + cn_stride);
      vst1q_f16(c6, vacc6x01234567);
      vst1q_f16(c6 + 8, vacc6x89ABCDEF);
      c6 = (__fp16*) ((uintptr_t) c6 + cn_stride);
      vst1q_f16(c5, vacc5x01234567);
      vst1q_f16(c5 + 8, vacc5x89ABCDEF);
      c5 = (__fp16*) ((uintptr_t) c5 + cn_stride);
      vst1q_f16(c4, vacc4x01234567);
      vst1q_f16(c4 + 8, vacc4x89ABCDEF);
      c4 = (__fp16*) ((uintptr_t) c4 + cn_stride);
      vst1q_f16(c3, vacc3x01234567);
      vst1q_f16(c3 + 8, vacc3x89ABCDEF);
      c3 = (__fp16*) ((uintptr_t) c3 + cn_stride);
      vst1q_f16(c2, vacc2x01234567);
      vst1q_f16(c2 + 8, vacc2x89ABCDEF);
      c2 = (__fp16*) ((uintptr_t) c2 + cn_stride);
      vst1q_f16(c1, vacc1x01234567);
      vst1q_f16(c1 + 8, vacc1x89ABCDEF);
      c1 = (__fp16*) ((uintptr_t) c1 + cn_stride);
      vst1q_f16(c0, vacc0x01234567);
      vst1q_f16(c0 + 8, vacc0x89ABCDEF);
      c0 = (__fp16*) ((uintptr_t) c0 + cn_stride);

      a = (const void**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc & 8) {
        vst1q_f16(c7, vacc7x01234567); c7 += 8;
        vst1q_f16(c6, vacc6x01234567); c6 += 8;
        vst1q_f16(c5, vacc5x01234567); c5 += 8;
        vst1q_f16(c4, vacc4x01234567); c4 += 8;
        vst1q_f16(c3, vacc3x01234567); c3 += 8;
        vst1q_f16(c2, vacc2x01234567); c2 += 8;
        vst1q_f16(c1, vacc1x01234567); c1 += 8;
        vst1q_f16(c0, vacc0x01234567); c0 += 8;

        vacc7x01234567 = vacc7x89ABCDEF;
        vacc6x01234567 = vacc6x89ABCDEF;
        vacc5x01234567 = vacc5x89ABCDEF;
        vacc4x01234567 = vacc4x89ABCDEF;
        vacc3x01234567 = vacc3x89ABCDEF;
        vacc2x01234567 = vacc2x89ABCDEF;
        vacc1x01234567 = vacc1x89ABCDEF;
        vacc0x01234567 = vacc0x89ABCDEF;
      }
      float16x4_t vacc7x0123 = vget_low_f16(vacc7x01234567);
      float16x4_t vacc6x0123 = vget_low_f16(vacc6x01234567);
      float16x4_t vacc5x0123 = vget_low_f16(vacc5x01234567);
      float16x4_t vacc4x0123 = vget_low_f16(vacc4x01234567);
      float16x4_t vacc3x0123 = vget_low_f16(vacc3x01234567);
      float16x4_t vacc2x0123 = vget_low_f16(vacc2x01234567);
      float16x4_t vacc1x0123 = vget_low_f16(vacc1x01234567);
      float16x4_t vacc0x0123 = vget_low_f16(vacc0x01234567);
      if (nc & 4) {
        vst1_f16(c7, vacc7x0123); c7 += 4;
        vst1_f16(c6, vacc6x0123); c6 += 4;
        vst1_f16(c5, vacc5x0123); c5 += 4;
        vst1_f16(c4, vacc4x0123); c4 += 4;
        vst1_f16(c3, vacc3x0123); c3 += 4;
        vst1_f16(c2, vacc2x0123); c2 += 4;
        vst1_f16(c1, vacc1x0123); c1 += 4;
        vst1_f16(c0, vacc0x0123); c0 += 4;

        vacc7x0123 = vget_high_f16(vacc7x01234567);
        vacc6x0123 = vget_high_f16(vacc6x01234567);
        vacc5x0123 = vget_high_f16(vacc5x01234567);
        vacc4x0123 = vget_high_f16(vacc4x01234567);
        vacc3x0123 = vget_high_f16(vacc3x01234567);
        vacc2x0123 = vget_high_f16(vacc2x01234567);
        vacc1x0123 = vget_high_f16(vacc1x01234567);
        vacc0x0123 = vget_high_f16(vacc0x01234567);
      }
      if (nc & 2) {
        vst1_lane_u32(__builtin_assume_aligned(c7, 1), vreinterpret_u32_f16(vacc7x0123), 0); c7 += 2;
        vst1_lane_u32(__builtin_assume_aligned(c6, 1), vreinterpret_u32_f16(vacc6x0123), 0); c6 += 2;
        vst1_lane_u32(__builtin_assume_aligned(c5, 1), vreinterpret_u32_f16(vacc5x0123), 0); c5 += 2;
        vst1_lane_u32(__builtin_assume_aligned(c4, 1), vreinterpret_u32_f16(vacc4x0123), 0); c4 += 2;
        vst1_lane_u32(__builtin_assume_aligned(c3, 1), vreinterpret_u32_f16(vacc3x0123), 0); c3 += 2;
        vst1_lane_u32(__builtin_assume_aligned(c2, 1), vreinterpret_u32_f16(vacc2x0123), 0); c2 += 2;
        vst1_lane_u32(__builtin_assume_aligned(c1, 1), vreinterpret_u32_f16(vacc1x0123), 0); c1 += 2;
        vst1_lane_u32(__builtin_assume_aligned(c0, 1), vreinterpret_u32_f16(vacc0x0123), 0); c0 += 2;

        vacc7x0123 = vext_f16(vacc7x0123, vacc7x0123, 2);
        vacc6x0123 = vext_f16(vacc6x0123, vacc6x0123, 2);
        vacc5x0123 = vext_f16(vacc5x0123, vacc5x0123, 2);
        vacc4x0123 = vext_f16(vacc4x0123, vacc4x0123, 2);
        vacc3x0123 = vext_f16(vacc3x0123, vacc3x0123, 2);
        vacc2x0123 = vext_f16(vacc2x0123, vacc2x0123, 2);
        vacc1x0123 = vext_f16(vacc1x0123, vacc1x0123, 2);
        vacc0x0123 = vext_f16(vacc0x0123, vacc0x0123, 2);
      }
      if (nc & 1) {
        vst1_lane_f16(c7, vacc7x0123, 0);
        vst1_lane_f16(c6, vacc6x0123, 0);
        vst1_lane_f16(c5, vacc5x0123, 0);
        vst1_lane_f16(c4, vacc4x0123, 0);
        vst1_lane_f16(c3, vacc3x0123, 0);
        vst1_lane_f16(c2, vacc2x0123, 0);
        vst1_lane_f16(c1, vacc1x0123, 0);
        vst1_lane_f16(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
