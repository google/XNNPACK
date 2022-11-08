// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/neon-mull-addw-dup.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/gemm.h>


void xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mull_addw_dup(
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
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (2 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    c1 = c0;
  }

  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0x89AB = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0xCDEF = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;
    int32x4_t vacc1x89AB = vacc0x89AB;
    int32x4_t vacc1xCDEF = vacc0xCDEF;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      a += 2;

      size_t k = kc;
      while (k >= 8 * sizeof(int8_t)) {
        const int8x8_t va0 = vld1_s8(a0); a0 += 8;
        const int8x8_t va1 = vld1_s8(a1); a1 += 8;

        const int8x8_t vb01234567c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x01234567c0 = vmull_s8(vb01234567c0, vdup_lane_s8(va0, 0));
        vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c0));
        vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c0));
        const int16x8_t vprod1x01234567c0 = vmull_s8(vb01234567c0, vdup_lane_s8(va1, 0));
        vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c0));
        vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c0));
        const int8x8_t vb89ABCDEFc0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x89ABCDEFc0 = vmull_s8(vb89ABCDEFc0, vdup_lane_s8(va0, 0));
        vacc0x89AB = vaddw_s16(vacc0x89AB, vget_low_s16(vprod0x89ABCDEFc0));
        vacc0xCDEF = vaddw_s16(vacc0xCDEF, vget_high_s16(vprod0x89ABCDEFc0));
        const int16x8_t vprod1x89ABCDEFc0 = vmull_s8(vb89ABCDEFc0, vdup_lane_s8(va1, 0));
        vacc1x89AB = vaddw_s16(vacc1x89AB, vget_low_s16(vprod1x89ABCDEFc0));
        vacc1xCDEF = vaddw_s16(vacc1xCDEF, vget_high_s16(vprod1x89ABCDEFc0));
        const int8x8_t vb01234567c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x01234567c1 = vmull_s8(vb01234567c1, vdup_lane_s8(va0, 1));
        vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c1));
        vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c1));
        const int16x8_t vprod1x01234567c1 = vmull_s8(vb01234567c1, vdup_lane_s8(va1, 1));
        vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c1));
        vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c1));
        const int8x8_t vb89ABCDEFc1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x89ABCDEFc1 = vmull_s8(vb89ABCDEFc1, vdup_lane_s8(va0, 1));
        vacc0x89AB = vaddw_s16(vacc0x89AB, vget_low_s16(vprod0x89ABCDEFc1));
        vacc0xCDEF = vaddw_s16(vacc0xCDEF, vget_high_s16(vprod0x89ABCDEFc1));
        const int16x8_t vprod1x89ABCDEFc1 = vmull_s8(vb89ABCDEFc1, vdup_lane_s8(va1, 1));
        vacc1x89AB = vaddw_s16(vacc1x89AB, vget_low_s16(vprod1x89ABCDEFc1));
        vacc1xCDEF = vaddw_s16(vacc1xCDEF, vget_high_s16(vprod1x89ABCDEFc1));
        const int8x8_t vb01234567c2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x01234567c2 = vmull_s8(vb01234567c2, vdup_lane_s8(va0, 2));
        vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c2));
        vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c2));
        const int16x8_t vprod1x01234567c2 = vmull_s8(vb01234567c2, vdup_lane_s8(va1, 2));
        vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c2));
        vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c2));
        const int8x8_t vb89ABCDEFc2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x89ABCDEFc2 = vmull_s8(vb89ABCDEFc2, vdup_lane_s8(va0, 2));
        vacc0x89AB = vaddw_s16(vacc0x89AB, vget_low_s16(vprod0x89ABCDEFc2));
        vacc0xCDEF = vaddw_s16(vacc0xCDEF, vget_high_s16(vprod0x89ABCDEFc2));
        const int16x8_t vprod1x89ABCDEFc2 = vmull_s8(vb89ABCDEFc2, vdup_lane_s8(va1, 2));
        vacc1x89AB = vaddw_s16(vacc1x89AB, vget_low_s16(vprod1x89ABCDEFc2));
        vacc1xCDEF = vaddw_s16(vacc1xCDEF, vget_high_s16(vprod1x89ABCDEFc2));
        const int8x8_t vb01234567c3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x01234567c3 = vmull_s8(vb01234567c3, vdup_lane_s8(va0, 3));
        vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c3));
        vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c3));
        const int16x8_t vprod1x01234567c3 = vmull_s8(vb01234567c3, vdup_lane_s8(va1, 3));
        vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c3));
        vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c3));
        const int8x8_t vb89ABCDEFc3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x89ABCDEFc3 = vmull_s8(vb89ABCDEFc3, vdup_lane_s8(va0, 3));
        vacc0x89AB = vaddw_s16(vacc0x89AB, vget_low_s16(vprod0x89ABCDEFc3));
        vacc0xCDEF = vaddw_s16(vacc0xCDEF, vget_high_s16(vprod0x89ABCDEFc3));
        const int16x8_t vprod1x89ABCDEFc3 = vmull_s8(vb89ABCDEFc3, vdup_lane_s8(va1, 3));
        vacc1x89AB = vaddw_s16(vacc1x89AB, vget_low_s16(vprod1x89ABCDEFc3));
        vacc1xCDEF = vaddw_s16(vacc1xCDEF, vget_high_s16(vprod1x89ABCDEFc3));
        const int8x8_t vb01234567c4 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x01234567c4 = vmull_s8(vb01234567c4, vdup_lane_s8(va0, 4));
        vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c4));
        vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c4));
        const int16x8_t vprod1x01234567c4 = vmull_s8(vb01234567c4, vdup_lane_s8(va1, 4));
        vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c4));
        vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c4));
        const int8x8_t vb89ABCDEFc4 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x89ABCDEFc4 = vmull_s8(vb89ABCDEFc4, vdup_lane_s8(va0, 4));
        vacc0x89AB = vaddw_s16(vacc0x89AB, vget_low_s16(vprod0x89ABCDEFc4));
        vacc0xCDEF = vaddw_s16(vacc0xCDEF, vget_high_s16(vprod0x89ABCDEFc4));
        const int16x8_t vprod1x89ABCDEFc4 = vmull_s8(vb89ABCDEFc4, vdup_lane_s8(va1, 4));
        vacc1x89AB = vaddw_s16(vacc1x89AB, vget_low_s16(vprod1x89ABCDEFc4));
        vacc1xCDEF = vaddw_s16(vacc1xCDEF, vget_high_s16(vprod1x89ABCDEFc4));
        const int8x8_t vb01234567c5 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x01234567c5 = vmull_s8(vb01234567c5, vdup_lane_s8(va0, 5));
        vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c5));
        vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c5));
        const int16x8_t vprod1x01234567c5 = vmull_s8(vb01234567c5, vdup_lane_s8(va1, 5));
        vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c5));
        vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c5));
        const int8x8_t vb89ABCDEFc5 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x89ABCDEFc5 = vmull_s8(vb89ABCDEFc5, vdup_lane_s8(va0, 5));
        vacc0x89AB = vaddw_s16(vacc0x89AB, vget_low_s16(vprod0x89ABCDEFc5));
        vacc0xCDEF = vaddw_s16(vacc0xCDEF, vget_high_s16(vprod0x89ABCDEFc5));
        const int16x8_t vprod1x89ABCDEFc5 = vmull_s8(vb89ABCDEFc5, vdup_lane_s8(va1, 5));
        vacc1x89AB = vaddw_s16(vacc1x89AB, vget_low_s16(vprod1x89ABCDEFc5));
        vacc1xCDEF = vaddw_s16(vacc1xCDEF, vget_high_s16(vprod1x89ABCDEFc5));
        const int8x8_t vb01234567c6 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x01234567c6 = vmull_s8(vb01234567c6, vdup_lane_s8(va0, 6));
        vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c6));
        vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c6));
        const int16x8_t vprod1x01234567c6 = vmull_s8(vb01234567c6, vdup_lane_s8(va1, 6));
        vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c6));
        vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c6));
        const int8x8_t vb89ABCDEFc6 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x89ABCDEFc6 = vmull_s8(vb89ABCDEFc6, vdup_lane_s8(va0, 6));
        vacc0x89AB = vaddw_s16(vacc0x89AB, vget_low_s16(vprod0x89ABCDEFc6));
        vacc0xCDEF = vaddw_s16(vacc0xCDEF, vget_high_s16(vprod0x89ABCDEFc6));
        const int16x8_t vprod1x89ABCDEFc6 = vmull_s8(vb89ABCDEFc6, vdup_lane_s8(va1, 6));
        vacc1x89AB = vaddw_s16(vacc1x89AB, vget_low_s16(vprod1x89ABCDEFc6));
        vacc1xCDEF = vaddw_s16(vacc1xCDEF, vget_high_s16(vprod1x89ABCDEFc6));
        const int8x8_t vb01234567c7 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x01234567c7 = vmull_s8(vb01234567c7, vdup_lane_s8(va0, 7));
        vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c7));
        vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c7));
        const int16x8_t vprod1x01234567c7 = vmull_s8(vb01234567c7, vdup_lane_s8(va1, 7));
        vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c7));
        vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c7));
        const int8x8_t vb89ABCDEFc7 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x89ABCDEFc7 = vmull_s8(vb89ABCDEFc7, vdup_lane_s8(va0, 7));
        vacc0x89AB = vaddw_s16(vacc0x89AB, vget_low_s16(vprod0x89ABCDEFc7));
        vacc0xCDEF = vaddw_s16(vacc0xCDEF, vget_high_s16(vprod0x89ABCDEFc7));
        const int16x8_t vprod1x89ABCDEFc7 = vmull_s8(vb89ABCDEFc7, vdup_lane_s8(va1, 7));
        vacc1x89AB = vaddw_s16(vacc1x89AB, vget_low_s16(vprod1x89ABCDEFc7));
        vacc1xCDEF = vaddw_s16(vacc1xCDEF, vget_high_s16(vprod1x89ABCDEFc7));

        k -= 8 * sizeof(int8_t);
      }
      if XNN_UNLIKELY(k != 0) {
        const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);
        const int8x8_t va1 = vld1_s8(a1); a1 = (const int8_t*) ((uintptr_t) a1 + k);

        const int8x8_t vb01234567c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb89ABCDEFc0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x01234567c0 = vmull_s8(vb01234567c0, vdup_lane_s8(va0, 0));
        vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c0));
        vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c0));
        const int16x8_t vprod0x89ABCDEFc0 = vmull_s8(vb89ABCDEFc0, vdup_lane_s8(va0, 0));
        vacc0x89AB = vaddw_s16(vacc0x89AB, vget_low_s16(vprod0x89ABCDEFc0));
        vacc0xCDEF = vaddw_s16(vacc0xCDEF, vget_high_s16(vprod0x89ABCDEFc0));
        const int16x8_t vprod1x01234567c0 = vmull_s8(vb01234567c0, vdup_lane_s8(va1, 0));
        vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c0));
        vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c0));
        const int16x8_t vprod1x89ABCDEFc0 = vmull_s8(vb89ABCDEFc0, vdup_lane_s8(va1, 0));
        vacc1x89AB = vaddw_s16(vacc1x89AB, vget_low_s16(vprod1x89ABCDEFc0));
        vacc1xCDEF = vaddw_s16(vacc1xCDEF, vget_high_s16(vprod1x89ABCDEFc0));

        if (k >= 2 * sizeof(int8_t)) {
          const int8x8_t vb01234567c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
          const int8x8_t vb89ABCDEFc1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

          const int16x8_t vprod0x01234567c1 = vmull_s8(vb01234567c1, vdup_lane_s8(va0, 1));
          vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c1));
          vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c1));
          const int16x8_t vprod0x89ABCDEFc1 = vmull_s8(vb89ABCDEFc1, vdup_lane_s8(va0, 1));
          vacc0x89AB = vaddw_s16(vacc0x89AB, vget_low_s16(vprod0x89ABCDEFc1));
          vacc0xCDEF = vaddw_s16(vacc0xCDEF, vget_high_s16(vprod0x89ABCDEFc1));
          const int16x8_t vprod1x01234567c1 = vmull_s8(vb01234567c1, vdup_lane_s8(va1, 1));
          vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c1));
          vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c1));
          const int16x8_t vprod1x89ABCDEFc1 = vmull_s8(vb89ABCDEFc1, vdup_lane_s8(va1, 1));
          vacc1x89AB = vaddw_s16(vacc1x89AB, vget_low_s16(vprod1x89ABCDEFc1));
          vacc1xCDEF = vaddw_s16(vacc1xCDEF, vget_high_s16(vprod1x89ABCDEFc1));

          if (k > 2 * sizeof(int8_t)) {
            const int8x8_t vb01234567c2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
            const int8x8_t vb89ABCDEFc2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

            const int16x8_t vprod0x01234567c2 = vmull_s8(vb01234567c2, vdup_lane_s8(va0, 2));
            vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c2));
            vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c2));
            const int16x8_t vprod0x89ABCDEFc2 = vmull_s8(vb89ABCDEFc2, vdup_lane_s8(va0, 2));
            vacc0x89AB = vaddw_s16(vacc0x89AB, vget_low_s16(vprod0x89ABCDEFc2));
            vacc0xCDEF = vaddw_s16(vacc0xCDEF, vget_high_s16(vprod0x89ABCDEFc2));
            const int16x8_t vprod1x01234567c2 = vmull_s8(vb01234567c2, vdup_lane_s8(va1, 2));
            vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c2));
            vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c2));
            const int16x8_t vprod1x89ABCDEFc2 = vmull_s8(vb89ABCDEFc2, vdup_lane_s8(va1, 2));
            vacc1x89AB = vaddw_s16(vacc1x89AB, vget_low_s16(vprod1x89ABCDEFc2));
            vacc1xCDEF = vaddw_s16(vacc1xCDEF, vget_high_s16(vprod1x89ABCDEFc2));

            if (k >= 4 * sizeof(int8_t)) {
              const int8x8_t vb01234567c3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
              const int8x8_t vb89ABCDEFc3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

              const int16x8_t vprod0x01234567c3 = vmull_s8(vb01234567c3, vdup_lane_s8(va0, 3));
              vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c3));
              vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c3));
              const int16x8_t vprod0x89ABCDEFc3 = vmull_s8(vb89ABCDEFc3, vdup_lane_s8(va0, 3));
              vacc0x89AB = vaddw_s16(vacc0x89AB, vget_low_s16(vprod0x89ABCDEFc3));
              vacc0xCDEF = vaddw_s16(vacc0xCDEF, vget_high_s16(vprod0x89ABCDEFc3));
              const int16x8_t vprod1x01234567c3 = vmull_s8(vb01234567c3, vdup_lane_s8(va1, 3));
              vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c3));
              vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c3));
              const int16x8_t vprod1x89ABCDEFc3 = vmull_s8(vb89ABCDEFc3, vdup_lane_s8(va1, 3));
              vacc1x89AB = vaddw_s16(vacc1x89AB, vget_low_s16(vprod1x89ABCDEFc3));
              vacc1xCDEF = vaddw_s16(vacc1xCDEF, vget_high_s16(vprod1x89ABCDEFc3));

              if (k > 4 * sizeof(int8_t)) {
                const int8x8_t vb01234567c4 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
                const int8x8_t vb89ABCDEFc4 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

                const int16x8_t vprod0x01234567c4 = vmull_s8(vb01234567c4, vdup_lane_s8(va0, 4));
                vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c4));
                vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c4));
                const int16x8_t vprod0x89ABCDEFc4 = vmull_s8(vb89ABCDEFc4, vdup_lane_s8(va0, 4));
                vacc0x89AB = vaddw_s16(vacc0x89AB, vget_low_s16(vprod0x89ABCDEFc4));
                vacc0xCDEF = vaddw_s16(vacc0xCDEF, vget_high_s16(vprod0x89ABCDEFc4));
                const int16x8_t vprod1x01234567c4 = vmull_s8(vb01234567c4, vdup_lane_s8(va1, 4));
                vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c4));
                vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c4));
                const int16x8_t vprod1x89ABCDEFc4 = vmull_s8(vb89ABCDEFc4, vdup_lane_s8(va1, 4));
                vacc1x89AB = vaddw_s16(vacc1x89AB, vget_low_s16(vprod1x89ABCDEFc4));
                vacc1xCDEF = vaddw_s16(vacc1xCDEF, vget_high_s16(vprod1x89ABCDEFc4));

                if (k >= 6 * sizeof(int8_t)) {
                  const int8x8_t vb01234567c5 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
                  const int8x8_t vb89ABCDEFc5 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

                  const int16x8_t vprod0x01234567c5 = vmull_s8(vb01234567c5, vdup_lane_s8(va0, 5));
                  vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c5));
                  vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c5));
                  const int16x8_t vprod0x89ABCDEFc5 = vmull_s8(vb89ABCDEFc5, vdup_lane_s8(va0, 5));
                  vacc0x89AB = vaddw_s16(vacc0x89AB, vget_low_s16(vprod0x89ABCDEFc5));
                  vacc0xCDEF = vaddw_s16(vacc0xCDEF, vget_high_s16(vprod0x89ABCDEFc5));
                  const int16x8_t vprod1x01234567c5 = vmull_s8(vb01234567c5, vdup_lane_s8(va1, 5));
                  vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c5));
                  vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c5));
                  const int16x8_t vprod1x89ABCDEFc5 = vmull_s8(vb89ABCDEFc5, vdup_lane_s8(va1, 5));
                  vacc1x89AB = vaddw_s16(vacc1x89AB, vget_low_s16(vprod1x89ABCDEFc5));
                  vacc1xCDEF = vaddw_s16(vacc1xCDEF, vget_high_s16(vprod1x89ABCDEFc5));

                  if (k > 6 * sizeof(int8_t)) {
                    const int8x8_t vb01234567c6 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
                    const int8x8_t vb89ABCDEFc6 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

                    const int16x8_t vprod0x01234567c6 = vmull_s8(vb01234567c6, vdup_lane_s8(va0, 6));
                    vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c6));
                    vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c6));
                    const int16x8_t vprod0x89ABCDEFc6 = vmull_s8(vb89ABCDEFc6, vdup_lane_s8(va0, 6));
                    vacc0x89AB = vaddw_s16(vacc0x89AB, vget_low_s16(vprod0x89ABCDEFc6));
                    vacc0xCDEF = vaddw_s16(vacc0xCDEF, vget_high_s16(vprod0x89ABCDEFc6));
                    const int16x8_t vprod1x01234567c6 = vmull_s8(vb01234567c6, vdup_lane_s8(va1, 6));
                    vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c6));
                    vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c6));
                    const int16x8_t vprod1x89ABCDEFc6 = vmull_s8(vb89ABCDEFc6, vdup_lane_s8(va1, 6));
                    vacc1x89AB = vaddw_s16(vacc1x89AB, vget_low_s16(vprod1x89ABCDEFc6));
                    vacc1xCDEF = vaddw_s16(vacc1xCDEF, vget_high_s16(vprod1x89ABCDEFc6));
                  }
                }
              }
            }
          }
        }
      }
      p -= 2 * sizeof(void*);
    } while (p != 0);

    const int32x4_t vright_pre_shift = vld1q_dup_s32(&params->rndnu_neon.right_pre_shift);
    const int32x4_t vmultiplier = vld1q_dup_s32(&params->rndnu_neon.multiplier);
    const int32x4_t vright_post_shift = vld1q_dup_s32(&params->rndnu_neon.right_post_shift);

    vacc0x0123 = vqshlq_s32(vacc0x0123, vright_pre_shift);
    vacc0x4567 = vqshlq_s32(vacc0x4567, vright_pre_shift);
    vacc0x89AB = vqshlq_s32(vacc0x89AB, vright_pre_shift);
    vacc0xCDEF = vqshlq_s32(vacc0xCDEF, vright_pre_shift);
    vacc1x0123 = vqshlq_s32(vacc1x0123, vright_pre_shift);
    vacc1x4567 = vqshlq_s32(vacc1x4567, vright_pre_shift);
    vacc1x89AB = vqshlq_s32(vacc1x89AB, vright_pre_shift);
    vacc1xCDEF = vqshlq_s32(vacc1xCDEF, vright_pre_shift);

    vacc0x0123 = vqdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqdmulhq_s32(vacc0x4567, vmultiplier);
    vacc0x89AB = vqdmulhq_s32(vacc0x89AB, vmultiplier);
    vacc0xCDEF = vqdmulhq_s32(vacc0xCDEF, vmultiplier);
    vacc1x0123 = vqdmulhq_s32(vacc1x0123, vmultiplier);
    vacc1x4567 = vqdmulhq_s32(vacc1x4567, vmultiplier);
    vacc1x89AB = vqdmulhq_s32(vacc1x89AB, vmultiplier);
    vacc1xCDEF = vqdmulhq_s32(vacc1xCDEF, vmultiplier);

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_post_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_post_shift);
    vacc0x89AB = vrshlq_s32(vacc0x89AB, vright_post_shift);
    vacc0xCDEF = vrshlq_s32(vacc0xCDEF, vright_post_shift);
    vacc1x0123 = vrshlq_s32(vacc1x0123, vright_post_shift);
    vacc1x4567 = vrshlq_s32(vacc1x4567, vright_post_shift);
    vacc1x89AB = vrshlq_s32(vacc1x89AB, vright_post_shift);
    vacc1xCDEF = vrshlq_s32(vacc1xCDEF, vright_post_shift);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->rndnu_neon.output_zero_point);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
    const int16x8_t vacc1x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x89AB), vacc1xCDEF), voutput_zero_point);

    int8x16_t vout0x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc0x89ABCDEF);
    int8x16_t vout1x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc1x01234567), vacc1x89ABCDEF);
#else
    const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF)), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
    const int16x8_t vacc1x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x89AB), vqmovn_s32(vacc1xCDEF)), voutput_zero_point);

    int8x16_t vout0x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc0x89ABCDEF));
    int8x16_t vout1x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc1x01234567), vqmovn_s16(vacc1x89ABCDEF));
#endif
    const int8x16_t voutput_min = vld1q_dup_s8(&params->rndnu_neon.output_min);
    const int8x16_t voutput_max = vld1q_dup_s8(&params->rndnu_neon.output_max);

    vout1x0123456789ABCDEF = vmaxq_s8(vout1x0123456789ABCDEF, voutput_min);
    vout0x0123456789ABCDEF = vmaxq_s8(vout0x0123456789ABCDEF, voutput_min);

    vout1x0123456789ABCDEF = vminq_s8(vout1x0123456789ABCDEF, voutput_max);
    vout0x0123456789ABCDEF = vminq_s8(vout0x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      vst1q_s8(c1 + 0, vout1x0123456789ABCDEF);
      vst1q_s8(c0 + 0, vout0x0123456789ABCDEF);

      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vget_low_s8(vout0x0123456789ABCDEF), vget_low_s8(vout1x0123456789ABCDEF));
      if (nc & 8) {
        vst1_s8(c1, vget_high_s8(vout0x01234567_1x01234567)); c1 += 8;
        vst1_s8(c0, vget_low_s8(vout0x01234567_1x01234567)); c0 += 8;
        vout0x01234567_1x01234567 = vcombine_s8(vget_high_s8(vout0x0123456789ABCDEF), vget_high_s8(vout1x0123456789ABCDEF));
      }
      if (nc & 4) {
        vst1q_lane_u32((void*) c1, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1q_lane_u32((void*) c0, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 0); c0 += 4;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16((void*) c1, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1q_lane_u16((void*) c0, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 0); c0 += 2;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_s8(c1, vout0x01234567_1x01234567, 8);
        vst1q_lane_s8(c0, vout0x01234567_1x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
