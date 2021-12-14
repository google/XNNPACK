// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/c2-neon-mull-dup.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/gemm.h>
#include <xnnpack/math.h>

void xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r(
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
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 2 * sizeof(int8_t));
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

  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0x89AB = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0xCDEF = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;
    int32x4_t vacc1x89AB = vacc0x89AB;
    int32x4_t vacc1xCDEF = vacc0xCDEF;
    int32x4_t vacc2x0123 = vacc0x0123;
    int32x4_t vacc2x4567 = vacc0x4567;
    int32x4_t vacc2x89AB = vacc0x89AB;
    int32x4_t vacc2xCDEF = vacc0xCDEF;
    int32x4_t vacc3x0123 = vacc0x0123;
    int32x4_t vacc3x4567 = vacc0x4567;
    int32x4_t vacc3x89AB = vacc0x89AB;
    int32x4_t vacc3xCDEF = vacc0xCDEF;

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
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;


      while (k >= 8 * sizeof(int8_t)) {
        const int16x4x2_t va00 = vld2_dup_s16((const void*)a0);
        const int16x4x2_t va01 = vld2_dup_s16((const void*)(a0 + 4)); a0 += 8;
        const int16x4x2_t va10 = vld2_dup_s16((const void*)a1);
        const int16x4x2_t va11 = vld2_dup_s16((const void*)(a1 + 4)); a1 += 8;
        const int16x4x2_t va20 = vld2_dup_s16((const void*)a2);
        const int16x4x2_t va21 = vld2_dup_s16((const void*)(a2 + 4)); a2 += 8;
        const int16x4x2_t va30 = vld2_dup_s16((const void*)a3);
        const int16x4x2_t va31 = vld2_dup_s16((const void*)(a3 + 4)); a3 += 8;

        const int8x8_t vb0123c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb4567c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb89ABc0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vbCDEFc0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb0123c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb4567c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb89ABc1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vbCDEFc1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb0123c2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb4567c2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb89ABc2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vbCDEFc2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb0123c3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb4567c3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb89ABc3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vbCDEFc3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int8x8_t va0c0 = vreinterpret_s8_s16(va00.val[0]);
        const int8x8_t va1c0 = vreinterpret_s8_s16(va10.val[0]);
        const int8x8_t va2c0 = vreinterpret_s8_s16(va20.val[0]);
        const int8x8_t va3c0 = vreinterpret_s8_s16(va30.val[0]);

        const int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0, va0c0);
        const int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0, va1c0);
        const int16x8_t vprod2x0123c0 = vmull_s8(vb0123c0, va2c0);
        const int16x8_t vprod3x0123c0 = vmull_s8(vb0123c0, va3c0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
        vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c0);
        vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c0);
        const int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0, va0c0);
        const int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0, va1c0);
        const int16x8_t vprod2x4567c0 = vmull_s8(vb4567c0, va2c0);
        const int16x8_t vprod3x4567c0 = vmull_s8(vb4567c0, va3c0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
        vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c0);
        vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c0);
        const int16x8_t vprod0x89ABc0 = vmull_s8(vb89ABc0, va0c0);
        const int16x8_t vprod1x89ABc0 = vmull_s8(vb89ABc0, va1c0);
        const int16x8_t vprod2x89ABc0 = vmull_s8(vb89ABc0, va2c0);
        const int16x8_t vprod3x89ABc0 = vmull_s8(vb89ABc0, va3c0);
        vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc0);
        vacc1x89AB = vpadalq_s16(vacc1x89AB, vprod1x89ABc0);
        vacc2x89AB = vpadalq_s16(vacc2x89AB, vprod2x89ABc0);
        vacc3x89AB = vpadalq_s16(vacc3x89AB, vprod3x89ABc0);
        const int16x8_t vprod0xCDEFc0 = vmull_s8(vbCDEFc0, va0c0);
        const int16x8_t vprod1xCDEFc0 = vmull_s8(vbCDEFc0, va1c0);
        const int16x8_t vprod2xCDEFc0 = vmull_s8(vbCDEFc0, va2c0);
        const int16x8_t vprod3xCDEFc0 = vmull_s8(vbCDEFc0, va3c0);
        vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc0);
        vacc1xCDEF = vpadalq_s16(vacc1xCDEF, vprod1xCDEFc0);
        vacc2xCDEF = vpadalq_s16(vacc2xCDEF, vprod2xCDEFc0);
        vacc3xCDEF = vpadalq_s16(vacc3xCDEF, vprod3xCDEFc0);
        const int8x8_t va0c1 = vreinterpret_s8_s16(va00.val[1]);
        const int8x8_t va1c1 = vreinterpret_s8_s16(va10.val[1]);
        const int8x8_t va2c1 = vreinterpret_s8_s16(va20.val[1]);
        const int8x8_t va3c1 = vreinterpret_s8_s16(va30.val[1]);

        const int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1, va0c1);
        const int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1, va1c1);
        const int16x8_t vprod2x0123c1 = vmull_s8(vb0123c1, va2c1);
        const int16x8_t vprod3x0123c1 = vmull_s8(vb0123c1, va3c1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
        vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c1);
        vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c1);
        const int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1, va0c1);
        const int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1, va1c1);
        const int16x8_t vprod2x4567c1 = vmull_s8(vb4567c1, va2c1);
        const int16x8_t vprod3x4567c1 = vmull_s8(vb4567c1, va3c1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
        vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c1);
        vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c1);
        const int16x8_t vprod0x89ABc1 = vmull_s8(vb89ABc1, va0c1);
        const int16x8_t vprod1x89ABc1 = vmull_s8(vb89ABc1, va1c1);
        const int16x8_t vprod2x89ABc1 = vmull_s8(vb89ABc1, va2c1);
        const int16x8_t vprod3x89ABc1 = vmull_s8(vb89ABc1, va3c1);
        vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc1);
        vacc1x89AB = vpadalq_s16(vacc1x89AB, vprod1x89ABc1);
        vacc2x89AB = vpadalq_s16(vacc2x89AB, vprod2x89ABc1);
        vacc3x89AB = vpadalq_s16(vacc3x89AB, vprod3x89ABc1);
        const int16x8_t vprod0xCDEFc1 = vmull_s8(vbCDEFc1, va0c1);
        const int16x8_t vprod1xCDEFc1 = vmull_s8(vbCDEFc1, va1c1);
        const int16x8_t vprod2xCDEFc1 = vmull_s8(vbCDEFc1, va2c1);
        const int16x8_t vprod3xCDEFc1 = vmull_s8(vbCDEFc1, va3c1);
        vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc1);
        vacc1xCDEF = vpadalq_s16(vacc1xCDEF, vprod1xCDEFc1);
        vacc2xCDEF = vpadalq_s16(vacc2xCDEF, vprod2xCDEFc1);
        vacc3xCDEF = vpadalq_s16(vacc3xCDEF, vprod3xCDEFc1);
        const int8x8_t va0c2 = vreinterpret_s8_s16(va01.val[0]);
        const int8x8_t va1c2 = vreinterpret_s8_s16(va11.val[0]);
        const int8x8_t va2c2 = vreinterpret_s8_s16(va21.val[0]);
        const int8x8_t va3c2 = vreinterpret_s8_s16(va31.val[0]);

        const int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2, va0c2);
        const int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2, va1c2);
        const int16x8_t vprod2x0123c2 = vmull_s8(vb0123c2, va2c2);
        const int16x8_t vprod3x0123c2 = vmull_s8(vb0123c2, va3c2);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
        vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c2);
        vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c2);
        const int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2, va0c2);
        const int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2, va1c2);
        const int16x8_t vprod2x4567c2 = vmull_s8(vb4567c2, va2c2);
        const int16x8_t vprod3x4567c2 = vmull_s8(vb4567c2, va3c2);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
        vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c2);
        vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c2);
        const int16x8_t vprod0x89ABc2 = vmull_s8(vb89ABc2, va0c2);
        const int16x8_t vprod1x89ABc2 = vmull_s8(vb89ABc2, va1c2);
        const int16x8_t vprod2x89ABc2 = vmull_s8(vb89ABc2, va2c2);
        const int16x8_t vprod3x89ABc2 = vmull_s8(vb89ABc2, va3c2);
        vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc2);
        vacc1x89AB = vpadalq_s16(vacc1x89AB, vprod1x89ABc2);
        vacc2x89AB = vpadalq_s16(vacc2x89AB, vprod2x89ABc2);
        vacc3x89AB = vpadalq_s16(vacc3x89AB, vprod3x89ABc2);
        const int16x8_t vprod0xCDEFc2 = vmull_s8(vbCDEFc2, va0c2);
        const int16x8_t vprod1xCDEFc2 = vmull_s8(vbCDEFc2, va1c2);
        const int16x8_t vprod2xCDEFc2 = vmull_s8(vbCDEFc2, va2c2);
        const int16x8_t vprod3xCDEFc2 = vmull_s8(vbCDEFc2, va3c2);
        vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc2);
        vacc1xCDEF = vpadalq_s16(vacc1xCDEF, vprod1xCDEFc2);
        vacc2xCDEF = vpadalq_s16(vacc2xCDEF, vprod2xCDEFc2);
        vacc3xCDEF = vpadalq_s16(vacc3xCDEF, vprod3xCDEFc2);
        const int8x8_t va0c3 = vreinterpret_s8_s16(va01.val[1]);
        const int8x8_t va1c3 = vreinterpret_s8_s16(va11.val[1]);
        const int8x8_t va2c3 = vreinterpret_s8_s16(va21.val[1]);
        const int8x8_t va3c3 = vreinterpret_s8_s16(va31.val[1]);

        const int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3, va0c3);
        const int16x8_t vprod1x0123c3 = vmull_s8(vb0123c3, va1c3);
        const int16x8_t vprod2x0123c3 = vmull_s8(vb0123c3, va2c3);
        const int16x8_t vprod3x0123c3 = vmull_s8(vb0123c3, va3c3);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c3);
        vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c3);
        vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c3);
        const int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3, va0c3);
        const int16x8_t vprod1x4567c3 = vmull_s8(vb4567c3, va1c3);
        const int16x8_t vprod2x4567c3 = vmull_s8(vb4567c3, va2c3);
        const int16x8_t vprod3x4567c3 = vmull_s8(vb4567c3, va3c3);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c3);
        vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c3);
        vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c3);
        const int16x8_t vprod0x89ABc3 = vmull_s8(vb89ABc3, va0c3);
        const int16x8_t vprod1x89ABc3 = vmull_s8(vb89ABc3, va1c3);
        const int16x8_t vprod2x89ABc3 = vmull_s8(vb89ABc3, va2c3);
        const int16x8_t vprod3x89ABc3 = vmull_s8(vb89ABc3, va3c3);
        vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc3);
        vacc1x89AB = vpadalq_s16(vacc1x89AB, vprod1x89ABc3);
        vacc2x89AB = vpadalq_s16(vacc2x89AB, vprod2x89ABc3);
        vacc3x89AB = vpadalq_s16(vacc3x89AB, vprod3x89ABc3);
        const int16x8_t vprod0xCDEFc3 = vmull_s8(vbCDEFc3, va0c3);
        const int16x8_t vprod1xCDEFc3 = vmull_s8(vbCDEFc3, va1c3);
        const int16x8_t vprod2xCDEFc3 = vmull_s8(vbCDEFc3, va2c3);
        const int16x8_t vprod3xCDEFc3 = vmull_s8(vbCDEFc3, va3c3);
        vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc3);
        vacc1xCDEF = vpadalq_s16(vacc1xCDEF, vprod1xCDEFc3);
        vacc2xCDEF = vpadalq_s16(vacc2xCDEF, vprod2xCDEFc3);
        vacc3xCDEF = vpadalq_s16(vacc3xCDEF, vprod3xCDEFc3);

        k -= 8 * sizeof(int8_t);
      }

      if XNN_UNLIKELY(k != 0) {
        const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);
        const int8x8_t va1 = vld1_s8(a1); a1 = (const int8_t*) ((uintptr_t) a1 + k);
        const int8x8_t va2 = vld1_s8(a2); a2 = (const int8_t*) ((uintptr_t) a2 + k);
        const int8x8_t va3 = vld1_s8(a3); a3 = (const int8_t*) ((uintptr_t) a3 + k);

        const int8x8_t vb0123c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb4567c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb89ABc0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vbCDEFc0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int8x8_t va0c0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0));
        const int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0, va0c0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
        const int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0, va0c0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
        const int16x8_t vprod0x89ABc0 = vmull_s8(vb89ABc0, va0c0);
        vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc0);
        const int16x8_t vprod0xCDEFc0 = vmull_s8(vbCDEFc0, va0c0);
        vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc0);
        const int8x8_t va1c0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 0));
        const int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0, va1c0);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
        const int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0, va1c0);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
        const int16x8_t vprod1x89ABc0 = vmull_s8(vb89ABc0, va1c0);
        vacc1x89AB = vpadalq_s16(vacc1x89AB, vprod1x89ABc0);
        const int16x8_t vprod1xCDEFc0 = vmull_s8(vbCDEFc0, va1c0);
        vacc1xCDEF = vpadalq_s16(vacc1xCDEF, vprod1xCDEFc0);
        const int8x8_t va2c0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 0));
        const int16x8_t vprod2x0123c0 = vmull_s8(vb0123c0, va2c0);
        vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c0);
        const int16x8_t vprod2x4567c0 = vmull_s8(vb4567c0, va2c0);
        vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c0);
        const int16x8_t vprod2x89ABc0 = vmull_s8(vb89ABc0, va2c0);
        vacc2x89AB = vpadalq_s16(vacc2x89AB, vprod2x89ABc0);
        const int16x8_t vprod2xCDEFc0 = vmull_s8(vbCDEFc0, va2c0);
        vacc2xCDEF = vpadalq_s16(vacc2xCDEF, vprod2xCDEFc0);
        const int8x8_t va3c0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 0));
        const int16x8_t vprod3x0123c0 = vmull_s8(vb0123c0, va3c0);
        vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c0);
        const int16x8_t vprod3x4567c0 = vmull_s8(vb4567c0, va3c0);
        vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c0);
        const int16x8_t vprod3x89ABc0 = vmull_s8(vb89ABc0, va3c0);
        vacc3x89AB = vpadalq_s16(vacc3x89AB, vprod3x89ABc0);
        const int16x8_t vprod3xCDEFc0 = vmull_s8(vbCDEFc0, va3c0);
        vacc3xCDEF = vpadalq_s16(vacc3xCDEF, vprod3xCDEFc0);

        if (k > 2 * sizeof(int8_t)) {
          const int8x8_t vb0123c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
          const int8x8_t vb4567c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
          const int8x8_t vb89ABc1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
          const int8x8_t vbCDEFc1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

          const int8x8_t va0c1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1));
          const int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1, va0c1);
          vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
          const int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1, va0c1);
          vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
          const int16x8_t vprod0x89ABc1 = vmull_s8(vb89ABc1, va0c1);
          vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc1);
          const int16x8_t vprod0xCDEFc1 = vmull_s8(vbCDEFc1, va0c1);
          vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc1);
          const int8x8_t va1c1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 1));
          const int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1, va1c1);
          vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
          const int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1, va1c1);
          vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
          const int16x8_t vprod1x89ABc1 = vmull_s8(vb89ABc1, va1c1);
          vacc1x89AB = vpadalq_s16(vacc1x89AB, vprod1x89ABc1);
          const int16x8_t vprod1xCDEFc1 = vmull_s8(vbCDEFc1, va1c1);
          vacc1xCDEF = vpadalq_s16(vacc1xCDEF, vprod1xCDEFc1);
          const int8x8_t va2c1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 1));
          const int16x8_t vprod2x0123c1 = vmull_s8(vb0123c1, va2c1);
          vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c1);
          const int16x8_t vprod2x4567c1 = vmull_s8(vb4567c1, va2c1);
          vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c1);
          const int16x8_t vprod2x89ABc1 = vmull_s8(vb89ABc1, va2c1);
          vacc2x89AB = vpadalq_s16(vacc2x89AB, vprod2x89ABc1);
          const int16x8_t vprod2xCDEFc1 = vmull_s8(vbCDEFc1, va2c1);
          vacc2xCDEF = vpadalq_s16(vacc2xCDEF, vprod2xCDEFc1);
          const int8x8_t va3c1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 1));
          const int16x8_t vprod3x0123c1 = vmull_s8(vb0123c1, va3c1);
          vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c1);
          const int16x8_t vprod3x4567c1 = vmull_s8(vb4567c1, va3c1);
          vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c1);
          const int16x8_t vprod3x89ABc1 = vmull_s8(vb89ABc1, va3c1);
          vacc3x89AB = vpadalq_s16(vacc3x89AB, vprod3x89ABc1);
          const int16x8_t vprod3xCDEFc1 = vmull_s8(vbCDEFc1, va3c1);
          vacc3xCDEF = vpadalq_s16(vacc3xCDEF, vprod3xCDEFc1);

          if (k > 4 * sizeof(int8_t)) {
            const int8x8_t vb0123c2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
            const int8x8_t vb4567c2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
            const int8x8_t vb89ABc2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
            const int8x8_t vbCDEFc2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

            const int8x8_t va0c2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2));
            const int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2, va0c2);
            vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
            const int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2, va0c2);
            vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
            const int16x8_t vprod0x89ABc2 = vmull_s8(vb89ABc2, va0c2);
            vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc2);
            const int16x8_t vprod0xCDEFc2 = vmull_s8(vbCDEFc2, va0c2);
            vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc2);
            const int8x8_t va1c2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 2));
            const int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2, va1c2);
            vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
            const int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2, va1c2);
            vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
            const int16x8_t vprod1x89ABc2 = vmull_s8(vb89ABc2, va1c2);
            vacc1x89AB = vpadalq_s16(vacc1x89AB, vprod1x89ABc2);
            const int16x8_t vprod1xCDEFc2 = vmull_s8(vbCDEFc2, va1c2);
            vacc1xCDEF = vpadalq_s16(vacc1xCDEF, vprod1xCDEFc2);
            const int8x8_t va2c2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 2));
            const int16x8_t vprod2x0123c2 = vmull_s8(vb0123c2, va2c2);
            vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c2);
            const int16x8_t vprod2x4567c2 = vmull_s8(vb4567c2, va2c2);
            vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c2);
            const int16x8_t vprod2x89ABc2 = vmull_s8(vb89ABc2, va2c2);
            vacc2x89AB = vpadalq_s16(vacc2x89AB, vprod2x89ABc2);
            const int16x8_t vprod2xCDEFc2 = vmull_s8(vbCDEFc2, va2c2);
            vacc2xCDEF = vpadalq_s16(vacc2xCDEF, vprod2xCDEFc2);
            const int8x8_t va3c2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 2));
            const int16x8_t vprod3x0123c2 = vmull_s8(vb0123c2, va3c2);
            vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c2);
            const int16x8_t vprod3x4567c2 = vmull_s8(vb4567c2, va3c2);
            vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c2);
            const int16x8_t vprod3x89ABc2 = vmull_s8(vb89ABc2, va3c2);
            vacc3x89AB = vpadalq_s16(vacc3x89AB, vprod3x89ABc2);
            const int16x8_t vprod3xCDEFc2 = vmull_s8(vbCDEFc2, va3c2);
            vacc3xCDEF = vpadalq_s16(vacc3xCDEF, vprod3xCDEFc2);
          }
        }
      }
      p -= 4 * sizeof(void*);
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
    vacc2x0123 = vqshlq_s32(vacc2x0123, vright_pre_shift);
    vacc2x4567 = vqshlq_s32(vacc2x4567, vright_pre_shift);
    vacc2x89AB = vqshlq_s32(vacc2x89AB, vright_pre_shift);
    vacc2xCDEF = vqshlq_s32(vacc2xCDEF, vright_pre_shift);
    vacc3x0123 = vqshlq_s32(vacc3x0123, vright_pre_shift);
    vacc3x4567 = vqshlq_s32(vacc3x4567, vright_pre_shift);
    vacc3x89AB = vqshlq_s32(vacc3x89AB, vright_pre_shift);
    vacc3xCDEF = vqshlq_s32(vacc3xCDEF, vright_pre_shift);

    vacc0x0123 = vqdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqdmulhq_s32(vacc0x4567, vmultiplier);
    vacc0x89AB = vqdmulhq_s32(vacc0x89AB, vmultiplier);
    vacc0xCDEF = vqdmulhq_s32(vacc0xCDEF, vmultiplier);
    vacc1x0123 = vqdmulhq_s32(vacc1x0123, vmultiplier);
    vacc1x4567 = vqdmulhq_s32(vacc1x4567, vmultiplier);
    vacc1x89AB = vqdmulhq_s32(vacc1x89AB, vmultiplier);
    vacc1xCDEF = vqdmulhq_s32(vacc1xCDEF, vmultiplier);
    vacc2x0123 = vqdmulhq_s32(vacc2x0123, vmultiplier);
    vacc2x4567 = vqdmulhq_s32(vacc2x4567, vmultiplier);
    vacc2x89AB = vqdmulhq_s32(vacc2x89AB, vmultiplier);
    vacc2xCDEF = vqdmulhq_s32(vacc2xCDEF, vmultiplier);
    vacc3x0123 = vqdmulhq_s32(vacc3x0123, vmultiplier);
    vacc3x4567 = vqdmulhq_s32(vacc3x4567, vmultiplier);
    vacc3x89AB = vqdmulhq_s32(vacc3x89AB, vmultiplier);
    vacc3xCDEF = vqdmulhq_s32(vacc3xCDEF, vmultiplier);

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_post_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_post_shift);
    vacc0x89AB = vrshlq_s32(vacc0x89AB, vright_post_shift);
    vacc0xCDEF = vrshlq_s32(vacc0xCDEF, vright_post_shift);
    vacc1x0123 = vrshlq_s32(vacc1x0123, vright_post_shift);
    vacc1x4567 = vrshlq_s32(vacc1x4567, vright_post_shift);
    vacc1x89AB = vrshlq_s32(vacc1x89AB, vright_post_shift);
    vacc1xCDEF = vrshlq_s32(vacc1xCDEF, vright_post_shift);
    vacc2x0123 = vrshlq_s32(vacc2x0123, vright_post_shift);
    vacc2x4567 = vrshlq_s32(vacc2x4567, vright_post_shift);
    vacc2x89AB = vrshlq_s32(vacc2x89AB, vright_post_shift);
    vacc2xCDEF = vrshlq_s32(vacc2xCDEF, vright_post_shift);
    vacc3x0123 = vrshlq_s32(vacc3x0123, vright_post_shift);
    vacc3x4567 = vrshlq_s32(vacc3x4567, vright_post_shift);
    vacc3x89AB = vrshlq_s32(vacc3x89AB, vright_post_shift);
    vacc3xCDEF = vrshlq_s32(vacc3xCDEF, vright_post_shift);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->rndnu_neon.output_zero_point);
#if XNN_ARCH_ARM64
    int16x8_t vacc0x01234567 = vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567);
    int16x8_t vacc0x89ABCDEF = vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF);
    int16x8_t vacc1x01234567 = vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567);
    int16x8_t vacc1x89ABCDEF = vqmovn_high_s32(vqmovn_s32(vacc1x89AB), vacc1xCDEF);
    int16x8_t vacc2x01234567 = vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567);
    int16x8_t vacc2x89ABCDEF = vqmovn_high_s32(vqmovn_s32(vacc2x89AB), vacc2xCDEF);
    int16x8_t vacc3x01234567 = vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567);
    int16x8_t vacc3x89ABCDEF = vqmovn_high_s32(vqmovn_s32(vacc3x89AB), vacc3xCDEF);

    vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);
    vacc0x89ABCDEF = vqaddq_s16(vacc0x89ABCDEF, voutput_zero_point);
    vacc1x01234567 = vqaddq_s16(vacc1x01234567, voutput_zero_point);
    vacc1x89ABCDEF = vqaddq_s16(vacc1x89ABCDEF, voutput_zero_point);
    vacc2x01234567 = vqaddq_s16(vacc2x01234567, voutput_zero_point);
    vacc2x89ABCDEF = vqaddq_s16(vacc2x89ABCDEF, voutput_zero_point);
    vacc3x01234567 = vqaddq_s16(vacc3x01234567, voutput_zero_point);
    vacc3x89ABCDEF = vqaddq_s16(vacc3x89ABCDEF, voutput_zero_point);

    int8x16_t vout0x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc0x89ABCDEF);
    int8x16_t vout1x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc1x01234567), vacc1x89ABCDEF);
    int8x16_t vout2x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc2x01234567), vacc2x89ABCDEF);
    int8x16_t vout3x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc3x01234567), vacc3x89ABCDEF);
#else
    int16x8_t vacc0x01234567 = vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567));
    int16x8_t vacc0x89ABCDEF = vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF));
    int16x8_t vacc1x01234567 = vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567));
    int16x8_t vacc1x89ABCDEF = vcombine_s16(vqmovn_s32(vacc1x89AB), vqmovn_s32(vacc1xCDEF));
    int16x8_t vacc2x01234567 = vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567));
    int16x8_t vacc2x89ABCDEF = vcombine_s16(vqmovn_s32(vacc2x89AB), vqmovn_s32(vacc2xCDEF));
    int16x8_t vacc3x01234567 = vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567));
    int16x8_t vacc3x89ABCDEF = vcombine_s16(vqmovn_s32(vacc3x89AB), vqmovn_s32(vacc3xCDEF));

    vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);
    vacc0x89ABCDEF = vqaddq_s16(vacc0x89ABCDEF, voutput_zero_point);
    vacc1x01234567 = vqaddq_s16(vacc1x01234567, voutput_zero_point);
    vacc1x89ABCDEF = vqaddq_s16(vacc1x89ABCDEF, voutput_zero_point);
    vacc2x01234567 = vqaddq_s16(vacc2x01234567, voutput_zero_point);
    vacc2x89ABCDEF = vqaddq_s16(vacc2x89ABCDEF, voutput_zero_point);
    vacc3x01234567 = vqaddq_s16(vacc3x01234567, voutput_zero_point);
    vacc3x89ABCDEF = vqaddq_s16(vacc3x89ABCDEF, voutput_zero_point);

    int8x16_t vout0x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc0x89ABCDEF));
    int8x16_t vout1x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc1x01234567), vqmovn_s16(vacc1x89ABCDEF));
    int8x16_t vout2x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc2x01234567), vqmovn_s16(vacc2x89ABCDEF));
    int8x16_t vout3x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc3x01234567), vqmovn_s16(vacc3x89ABCDEF));
#endif

    const int8x16_t voutput_min = vld1q_dup_s8(&params->rndnu_neon.output_min);
    vout0x0123456789ABCDEF = vmaxq_s8(vout0x0123456789ABCDEF, voutput_min);
    vout1x0123456789ABCDEF = vmaxq_s8(vout1x0123456789ABCDEF, voutput_min);
    vout2x0123456789ABCDEF = vmaxq_s8(vout2x0123456789ABCDEF, voutput_min);
    vout3x0123456789ABCDEF = vmaxq_s8(vout3x0123456789ABCDEF, voutput_min);

    const int8x16_t voutput_max = vld1q_dup_s8(&params->rndnu_neon.output_max);
    vout0x0123456789ABCDEF = vminq_s8(vout0x0123456789ABCDEF, voutput_max);
    vout1x0123456789ABCDEF = vminq_s8(vout1x0123456789ABCDEF, voutput_max);
    vout2x0123456789ABCDEF = vminq_s8(vout2x0123456789ABCDEF, voutput_max);
    vout3x0123456789ABCDEF = vminq_s8(vout3x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      vst1q_s8(c3 + 0, vout3x0123456789ABCDEF);
      vst1q_s8(c2 + 0, vout2x0123456789ABCDEF);
      vst1q_s8(c1 + 0, vout1x0123456789ABCDEF);
      vst1q_s8(c0 + 0, vout0x0123456789ABCDEF);

      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      int8x16_t vout2x01234567_3x01234567 = vcombine_s8(vget_low_s8(vout2x0123456789ABCDEF), vget_low_s8(vout3x0123456789ABCDEF));
      int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vget_low_s8(vout0x0123456789ABCDEF), vget_low_s8(vout1x0123456789ABCDEF));
      if (nc & 8) {
        vst1_s8(c3, vget_high_s8(vout2x01234567_3x01234567)); c3 += 8;
        vst1_s8(c2, vget_low_s8(vout2x01234567_3x01234567)); c2 += 8;
        vst1_s8(c1, vget_high_s8(vout0x01234567_1x01234567)); c1 += 8;
        vst1_s8(c0, vget_low_s8(vout0x01234567_1x01234567)); c0 += 8;
        vout2x01234567_3x01234567 = vcombine_s8(vget_high_s8(vout2x0123456789ABCDEF), vget_high_s8(vout3x0123456789ABCDEF));
        vout0x01234567_1x01234567 = vcombine_s8(vget_high_s8(vout0x0123456789ABCDEF), vget_high_s8(vout1x0123456789ABCDEF));
      }
      if (nc & 4) {
        vst1q_lane_u32((void*) c3, vreinterpretq_u32_s8(vout2x01234567_3x01234567), 2); c3 += 4;
        vst1q_lane_u32((void*) c2, vreinterpretq_u32_s8(vout2x01234567_3x01234567), 0); c2 += 4;
        vst1q_lane_u32((void*) c1, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1q_lane_u32((void*) c0, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 0); c0 += 4;
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16((void*) c3, vreinterpretq_u16_s8(vout2x01234567_3x01234567), 4); c3 += 2;
        vst1q_lane_u16((void*) c2, vreinterpretq_u16_s8(vout2x01234567_3x01234567), 0); c2 += 2;
        vst1q_lane_u16((void*) c1, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1q_lane_u16((void*) c0, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 0); c0 += 2;
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_s8(c3, vout2x01234567_3x01234567, 8);
        vst1q_lane_s8(c2, vout2x01234567_3x01234567, 0);
        vst1q_lane_s8(c1, vout0x01234567_1x01234567, 8);
        vst1q_lane_s8(c0, vout0x01234567_1x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
