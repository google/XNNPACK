// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c8-neon-mull-addlv.c.in
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


static const int32x4_t combine4(const int32x4_t a, const int32x4_t b, const int32x4_t c, const int32x4_t d) {
#if XNN_ARCH_ARM64
  const int32x4_t ab = vpaddq_s32(a, b);
  const int32x4_t cd = vpaddq_s32(c, d);
  const int32x4_t abcd = vpaddq_s32(ab, cd);
#else
  const int32x2_t ab_low  = vpadd_s32(vget_low_s32(a),  vget_low_s32(b));
  const int32x2_t ab_high = vpadd_s32(vget_high_s32(a), vget_high_s32(b));
  const int32x2_t cd_low  = vpadd_s32(vget_low_s32(c), vget_low_s32(d));
  const int32x2_t cd_high = vpadd_s32(vget_high_s32(c), vget_high_s32(d));
  const int32x2_t ab = vadd_s32(ab_low, ab_high);
  const int32x2_t cd = vadd_s32(cd_low, cd_high);
  const int32x4_t abcd = vcombine_s32(ab, cd);
#endif
  return abcd;
}

void xnn_qs8_gemm_minmax_ukernel_4x16c8__neon_mull_addlv(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_gemm_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
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

    size_t k = kc;
    while (k >= 8 * sizeof(int8_t)) {
      const int8x8_t va0 = vld1_s8(a0); a0 += 8;
      const int8x8_t va1 = vld1_s8(a1); a1 += 8;
      const int8x8_t va2 = vld1_s8(a2); a2 += 8;
      const int8x8_t va3 = vld1_s8(a3); a3 += 8;

      const int8x8_t vbx0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x0 = vmull_s8(vbx0, va0);
      const int32x4_t vacc0x0 = vpaddlq_s16(vprod0x0);
      const int16x8_t vprod1x0 = vmull_s8(vbx0, va1);
      const int32x4_t vacc1x0 = vpaddlq_s16(vprod1x0);
      const int16x8_t vprod2x0 = vmull_s8(vbx0, va2);
      const int32x4_t vacc2x0 = vpaddlq_s16(vprod2x0);
      const int16x8_t vprod3x0 = vmull_s8(vbx0, va3);
      const int32x4_t vacc3x0 = vpaddlq_s16(vprod3x0);
      const int8x8_t vbx1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x1 = vmull_s8(vbx1, va0);
      const int32x4_t vacc0x1 = vpaddlq_s16(vprod0x1);
      const int16x8_t vprod1x1 = vmull_s8(vbx1, va1);
      const int32x4_t vacc1x1 = vpaddlq_s16(vprod1x1);
      const int16x8_t vprod2x1 = vmull_s8(vbx1, va2);
      const int32x4_t vacc2x1 = vpaddlq_s16(vprod2x1);
      const int16x8_t vprod3x1 = vmull_s8(vbx1, va3);
      const int32x4_t vacc3x1 = vpaddlq_s16(vprod3x1);
      const int8x8_t vbx2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x2 = vmull_s8(vbx2, va0);
      const int32x4_t vacc0x2 = vpaddlq_s16(vprod0x2);
      const int16x8_t vprod1x2 = vmull_s8(vbx2, va1);
      const int32x4_t vacc1x2 = vpaddlq_s16(vprod1x2);
      const int16x8_t vprod2x2 = vmull_s8(vbx2, va2);
      const int32x4_t vacc2x2 = vpaddlq_s16(vprod2x2);
      const int16x8_t vprod3x2 = vmull_s8(vbx2, va3);
      const int32x4_t vacc3x2 = vpaddlq_s16(vprod3x2);
      const int8x8_t vbx3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x3 = vmull_s8(vbx3, va0);
      const int32x4_t vacc0x3 = vpaddlq_s16(vprod0x3);
      const int16x8_t vprod1x3 = vmull_s8(vbx3, va1);
      const int32x4_t vacc1x3 = vpaddlq_s16(vprod1x3);
      const int16x8_t vprod2x3 = vmull_s8(vbx3, va2);
      const int32x4_t vacc2x3 = vpaddlq_s16(vprod2x3);
      const int16x8_t vprod3x3 = vmull_s8(vbx3, va3);
      const int32x4_t vacc3x3 = vpaddlq_s16(vprod3x3);
      const int8x8_t vbx4 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x4 = vmull_s8(vbx4, va0);
      const int32x4_t vacc0x4 = vpaddlq_s16(vprod0x4);
      const int16x8_t vprod1x4 = vmull_s8(vbx4, va1);
      const int32x4_t vacc1x4 = vpaddlq_s16(vprod1x4);
      const int16x8_t vprod2x4 = vmull_s8(vbx4, va2);
      const int32x4_t vacc2x4 = vpaddlq_s16(vprod2x4);
      const int16x8_t vprod3x4 = vmull_s8(vbx4, va3);
      const int32x4_t vacc3x4 = vpaddlq_s16(vprod3x4);
      const int8x8_t vbx5 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x5 = vmull_s8(vbx5, va0);
      const int32x4_t vacc0x5 = vpaddlq_s16(vprod0x5);
      const int16x8_t vprod1x5 = vmull_s8(vbx5, va1);
      const int32x4_t vacc1x5 = vpaddlq_s16(vprod1x5);
      const int16x8_t vprod2x5 = vmull_s8(vbx5, va2);
      const int32x4_t vacc2x5 = vpaddlq_s16(vprod2x5);
      const int16x8_t vprod3x5 = vmull_s8(vbx5, va3);
      const int32x4_t vacc3x5 = vpaddlq_s16(vprod3x5);
      const int8x8_t vbx6 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x6 = vmull_s8(vbx6, va0);
      const int32x4_t vacc0x6 = vpaddlq_s16(vprod0x6);
      const int16x8_t vprod1x6 = vmull_s8(vbx6, va1);
      const int32x4_t vacc1x6 = vpaddlq_s16(vprod1x6);
      const int16x8_t vprod2x6 = vmull_s8(vbx6, va2);
      const int32x4_t vacc2x6 = vpaddlq_s16(vprod2x6);
      const int16x8_t vprod3x6 = vmull_s8(vbx6, va3);
      const int32x4_t vacc3x6 = vpaddlq_s16(vprod3x6);
      const int8x8_t vbx7 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x7 = vmull_s8(vbx7, va0);
      const int32x4_t vacc0x7 = vpaddlq_s16(vprod0x7);
      const int16x8_t vprod1x7 = vmull_s8(vbx7, va1);
      const int32x4_t vacc1x7 = vpaddlq_s16(vprod1x7);
      const int16x8_t vprod2x7 = vmull_s8(vbx7, va2);
      const int32x4_t vacc2x7 = vpaddlq_s16(vprod2x7);
      const int16x8_t vprod3x7 = vmull_s8(vbx7, va3);
      const int32x4_t vacc3x7 = vpaddlq_s16(vprod3x7);
      const int8x8_t vbx8 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x8 = vmull_s8(vbx8, va0);
      const int32x4_t vacc0x8 = vpaddlq_s16(vprod0x8);
      const int16x8_t vprod1x8 = vmull_s8(vbx8, va1);
      const int32x4_t vacc1x8 = vpaddlq_s16(vprod1x8);
      const int16x8_t vprod2x8 = vmull_s8(vbx8, va2);
      const int32x4_t vacc2x8 = vpaddlq_s16(vprod2x8);
      const int16x8_t vprod3x8 = vmull_s8(vbx8, va3);
      const int32x4_t vacc3x8 = vpaddlq_s16(vprod3x8);
      const int8x8_t vbx9 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x9 = vmull_s8(vbx9, va0);
      const int32x4_t vacc0x9 = vpaddlq_s16(vprod0x9);
      const int16x8_t vprod1x9 = vmull_s8(vbx9, va1);
      const int32x4_t vacc1x9 = vpaddlq_s16(vprod1x9);
      const int16x8_t vprod2x9 = vmull_s8(vbx9, va2);
      const int32x4_t vacc2x9 = vpaddlq_s16(vprod2x9);
      const int16x8_t vprod3x9 = vmull_s8(vbx9, va3);
      const int32x4_t vacc3x9 = vpaddlq_s16(vprod3x9);
      const int8x8_t vbx10 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x10 = vmull_s8(vbx10, va0);
      const int32x4_t vacc0x10 = vpaddlq_s16(vprod0x10);
      const int16x8_t vprod1x10 = vmull_s8(vbx10, va1);
      const int32x4_t vacc1x10 = vpaddlq_s16(vprod1x10);
      const int16x8_t vprod2x10 = vmull_s8(vbx10, va2);
      const int32x4_t vacc2x10 = vpaddlq_s16(vprod2x10);
      const int16x8_t vprod3x10 = vmull_s8(vbx10, va3);
      const int32x4_t vacc3x10 = vpaddlq_s16(vprod3x10);
      const int8x8_t vbx11 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x11 = vmull_s8(vbx11, va0);
      const int32x4_t vacc0x11 = vpaddlq_s16(vprod0x11);
      const int16x8_t vprod1x11 = vmull_s8(vbx11, va1);
      const int32x4_t vacc1x11 = vpaddlq_s16(vprod1x11);
      const int16x8_t vprod2x11 = vmull_s8(vbx11, va2);
      const int32x4_t vacc2x11 = vpaddlq_s16(vprod2x11);
      const int16x8_t vprod3x11 = vmull_s8(vbx11, va3);
      const int32x4_t vacc3x11 = vpaddlq_s16(vprod3x11);
      const int8x8_t vbx12 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x12 = vmull_s8(vbx12, va0);
      const int32x4_t vacc0x12 = vpaddlq_s16(vprod0x12);
      const int16x8_t vprod1x12 = vmull_s8(vbx12, va1);
      const int32x4_t vacc1x12 = vpaddlq_s16(vprod1x12);
      const int16x8_t vprod2x12 = vmull_s8(vbx12, va2);
      const int32x4_t vacc2x12 = vpaddlq_s16(vprod2x12);
      const int16x8_t vprod3x12 = vmull_s8(vbx12, va3);
      const int32x4_t vacc3x12 = vpaddlq_s16(vprod3x12);
      const int8x8_t vbx13 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x13 = vmull_s8(vbx13, va0);
      const int32x4_t vacc0x13 = vpaddlq_s16(vprod0x13);
      const int16x8_t vprod1x13 = vmull_s8(vbx13, va1);
      const int32x4_t vacc1x13 = vpaddlq_s16(vprod1x13);
      const int16x8_t vprod2x13 = vmull_s8(vbx13, va2);
      const int32x4_t vacc2x13 = vpaddlq_s16(vprod2x13);
      const int16x8_t vprod3x13 = vmull_s8(vbx13, va3);
      const int32x4_t vacc3x13 = vpaddlq_s16(vprod3x13);
      const int8x8_t vbx14 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x14 = vmull_s8(vbx14, va0);
      const int32x4_t vacc0x14 = vpaddlq_s16(vprod0x14);
      const int16x8_t vprod1x14 = vmull_s8(vbx14, va1);
      const int32x4_t vacc1x14 = vpaddlq_s16(vprod1x14);
      const int16x8_t vprod2x14 = vmull_s8(vbx14, va2);
      const int32x4_t vacc2x14 = vpaddlq_s16(vprod2x14);
      const int16x8_t vprod3x14 = vmull_s8(vbx14, va3);
      const int32x4_t vacc3x14 = vpaddlq_s16(vprod3x14);
      const int8x8_t vbx15 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x15 = vmull_s8(vbx15, va0);
      const int32x4_t vacc0x15 = vpaddlq_s16(vprod0x15);
      const int16x8_t vprod1x15 = vmull_s8(vbx15, va1);
      const int32x4_t vacc1x15 = vpaddlq_s16(vprod1x15);
      const int16x8_t vprod2x15 = vmull_s8(vbx15, va2);
      const int32x4_t vacc2x15 = vpaddlq_s16(vprod2x15);
      const int16x8_t vprod3x15 = vmull_s8(vbx15, va3);
      const int32x4_t vacc3x15 = vpaddlq_s16(vprod3x15);

      const int32x4_t vprod0x0123 = combine4(vacc0x0, vacc0x1, vacc0x2, vacc0x3);
      vacc0x0123 = vaddq_s32(vacc0x0123, vprod0x0123);
      const int32x4_t vprod0x4567 = combine4(vacc0x4, vacc0x5, vacc0x6, vacc0x7);
      vacc0x4567 = vaddq_s32(vacc0x4567, vprod0x4567);
      const int32x4_t vprod0x89AB = combine4(vacc0x8, vacc0x9, vacc0x10, vacc0x11);
      vacc0x89AB = vaddq_s32(vacc0x89AB, vprod0x89AB);
      const int32x4_t vprod0xCDEF = combine4(vacc0x12, vacc0x13, vacc0x14, vacc0x15);
      vacc0xCDEF = vaddq_s32(vacc0xCDEF, vprod0xCDEF);
      const int32x4_t vprod1x0123 = combine4(vacc1x0, vacc1x1, vacc1x2, vacc1x3);
      vacc1x0123 = vaddq_s32(vacc1x0123, vprod1x0123);
      const int32x4_t vprod1x4567 = combine4(vacc1x4, vacc1x5, vacc1x6, vacc1x7);
      vacc1x4567 = vaddq_s32(vacc1x4567, vprod1x4567);
      const int32x4_t vprod1x89AB = combine4(vacc1x8, vacc1x9, vacc1x10, vacc1x11);
      vacc1x89AB = vaddq_s32(vacc1x89AB, vprod1x89AB);
      const int32x4_t vprod1xCDEF = combine4(vacc1x12, vacc1x13, vacc1x14, vacc1x15);
      vacc1xCDEF = vaddq_s32(vacc1xCDEF, vprod1xCDEF);
      const int32x4_t vprod2x0123 = combine4(vacc2x0, vacc2x1, vacc2x2, vacc2x3);
      vacc2x0123 = vaddq_s32(vacc2x0123, vprod2x0123);
      const int32x4_t vprod2x4567 = combine4(vacc2x4, vacc2x5, vacc2x6, vacc2x7);
      vacc2x4567 = vaddq_s32(vacc2x4567, vprod2x4567);
      const int32x4_t vprod2x89AB = combine4(vacc2x8, vacc2x9, vacc2x10, vacc2x11);
      vacc2x89AB = vaddq_s32(vacc2x89AB, vprod2x89AB);
      const int32x4_t vprod2xCDEF = combine4(vacc2x12, vacc2x13, vacc2x14, vacc2x15);
      vacc2xCDEF = vaddq_s32(vacc2xCDEF, vprod2xCDEF);
      const int32x4_t vprod3x0123 = combine4(vacc3x0, vacc3x1, vacc3x2, vacc3x3);
      vacc3x0123 = vaddq_s32(vacc3x0123, vprod3x0123);
      const int32x4_t vprod3x4567 = combine4(vacc3x4, vacc3x5, vacc3x6, vacc3x7);
      vacc3x4567 = vaddq_s32(vacc3x4567, vprod3x4567);
      const int32x4_t vprod3x89AB = combine4(vacc3x8, vacc3x9, vacc3x10, vacc3x11);
      vacc3x89AB = vaddq_s32(vacc3x89AB, vprod3x89AB);
      const int32x4_t vprod3xCDEF = combine4(vacc3x12, vacc3x13, vacc3x14, vacc3x15);
      vacc3xCDEF = vaddq_s32(vacc3xCDEF, vprod3xCDEF);

      k -= 8 * sizeof(int8_t);
    }
    if XNN_UNLIKELY(k != 0) {
      const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);
      const int8x8_t va1 = vld1_s8(a1); a1 = (const int8_t*) ((uintptr_t) a1 + k);
      const int8x8_t va2 = vld1_s8(a2); a2 = (const int8_t*) ((uintptr_t) a2 + k);
      const int8x8_t va3 = vld1_s8(a3); a3 = (const int8_t*) ((uintptr_t) a3 + k);

      const int8x8_t vbx0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x0 = vmull_s8(vbx0, va0);
      const int32x4_t vacc0x0 = vpaddlq_s16(vprod0x0);
      const int16x8_t vprod1x0 = vmull_s8(vbx0, va1);
      const int32x4_t vacc1x0 = vpaddlq_s16(vprod1x0);
      const int16x8_t vprod2x0 = vmull_s8(vbx0, va2);
      const int32x4_t vacc2x0 = vpaddlq_s16(vprod2x0);
      const int16x8_t vprod3x0 = vmull_s8(vbx0, va3);
      const int32x4_t vacc3x0 = vpaddlq_s16(vprod3x0);
      const int8x8_t vbx1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x1 = vmull_s8(vbx1, va0);
      const int32x4_t vacc0x1 = vpaddlq_s16(vprod0x1);
      const int16x8_t vprod1x1 = vmull_s8(vbx1, va1);
      const int32x4_t vacc1x1 = vpaddlq_s16(vprod1x1);
      const int16x8_t vprod2x1 = vmull_s8(vbx1, va2);
      const int32x4_t vacc2x1 = vpaddlq_s16(vprod2x1);
      const int16x8_t vprod3x1 = vmull_s8(vbx1, va3);
      const int32x4_t vacc3x1 = vpaddlq_s16(vprod3x1);
      const int8x8_t vbx2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x2 = vmull_s8(vbx2, va0);
      const int32x4_t vacc0x2 = vpaddlq_s16(vprod0x2);
      const int16x8_t vprod1x2 = vmull_s8(vbx2, va1);
      const int32x4_t vacc1x2 = vpaddlq_s16(vprod1x2);
      const int16x8_t vprod2x2 = vmull_s8(vbx2, va2);
      const int32x4_t vacc2x2 = vpaddlq_s16(vprod2x2);
      const int16x8_t vprod3x2 = vmull_s8(vbx2, va3);
      const int32x4_t vacc3x2 = vpaddlq_s16(vprod3x2);
      const int8x8_t vbx3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x3 = vmull_s8(vbx3, va0);
      const int32x4_t vacc0x3 = vpaddlq_s16(vprod0x3);
      const int16x8_t vprod1x3 = vmull_s8(vbx3, va1);
      const int32x4_t vacc1x3 = vpaddlq_s16(vprod1x3);
      const int16x8_t vprod2x3 = vmull_s8(vbx3, va2);
      const int32x4_t vacc2x3 = vpaddlq_s16(vprod2x3);
      const int16x8_t vprod3x3 = vmull_s8(vbx3, va3);
      const int32x4_t vacc3x3 = vpaddlq_s16(vprod3x3);
      const int8x8_t vbx4 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x4 = vmull_s8(vbx4, va0);
      const int32x4_t vacc0x4 = vpaddlq_s16(vprod0x4);
      const int16x8_t vprod1x4 = vmull_s8(vbx4, va1);
      const int32x4_t vacc1x4 = vpaddlq_s16(vprod1x4);
      const int16x8_t vprod2x4 = vmull_s8(vbx4, va2);
      const int32x4_t vacc2x4 = vpaddlq_s16(vprod2x4);
      const int16x8_t vprod3x4 = vmull_s8(vbx4, va3);
      const int32x4_t vacc3x4 = vpaddlq_s16(vprod3x4);
      const int8x8_t vbx5 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x5 = vmull_s8(vbx5, va0);
      const int32x4_t vacc0x5 = vpaddlq_s16(vprod0x5);
      const int16x8_t vprod1x5 = vmull_s8(vbx5, va1);
      const int32x4_t vacc1x5 = vpaddlq_s16(vprod1x5);
      const int16x8_t vprod2x5 = vmull_s8(vbx5, va2);
      const int32x4_t vacc2x5 = vpaddlq_s16(vprod2x5);
      const int16x8_t vprod3x5 = vmull_s8(vbx5, va3);
      const int32x4_t vacc3x5 = vpaddlq_s16(vprod3x5);
      const int8x8_t vbx6 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x6 = vmull_s8(vbx6, va0);
      const int32x4_t vacc0x6 = vpaddlq_s16(vprod0x6);
      const int16x8_t vprod1x6 = vmull_s8(vbx6, va1);
      const int32x4_t vacc1x6 = vpaddlq_s16(vprod1x6);
      const int16x8_t vprod2x6 = vmull_s8(vbx6, va2);
      const int32x4_t vacc2x6 = vpaddlq_s16(vprod2x6);
      const int16x8_t vprod3x6 = vmull_s8(vbx6, va3);
      const int32x4_t vacc3x6 = vpaddlq_s16(vprod3x6);
      const int8x8_t vbx7 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x7 = vmull_s8(vbx7, va0);
      const int32x4_t vacc0x7 = vpaddlq_s16(vprod0x7);
      const int16x8_t vprod1x7 = vmull_s8(vbx7, va1);
      const int32x4_t vacc1x7 = vpaddlq_s16(vprod1x7);
      const int16x8_t vprod2x7 = vmull_s8(vbx7, va2);
      const int32x4_t vacc2x7 = vpaddlq_s16(vprod2x7);
      const int16x8_t vprod3x7 = vmull_s8(vbx7, va3);
      const int32x4_t vacc3x7 = vpaddlq_s16(vprod3x7);
      const int8x8_t vbx8 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x8 = vmull_s8(vbx8, va0);
      const int32x4_t vacc0x8 = vpaddlq_s16(vprod0x8);
      const int16x8_t vprod1x8 = vmull_s8(vbx8, va1);
      const int32x4_t vacc1x8 = vpaddlq_s16(vprod1x8);
      const int16x8_t vprod2x8 = vmull_s8(vbx8, va2);
      const int32x4_t vacc2x8 = vpaddlq_s16(vprod2x8);
      const int16x8_t vprod3x8 = vmull_s8(vbx8, va3);
      const int32x4_t vacc3x8 = vpaddlq_s16(vprod3x8);
      const int8x8_t vbx9 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x9 = vmull_s8(vbx9, va0);
      const int32x4_t vacc0x9 = vpaddlq_s16(vprod0x9);
      const int16x8_t vprod1x9 = vmull_s8(vbx9, va1);
      const int32x4_t vacc1x9 = vpaddlq_s16(vprod1x9);
      const int16x8_t vprod2x9 = vmull_s8(vbx9, va2);
      const int32x4_t vacc2x9 = vpaddlq_s16(vprod2x9);
      const int16x8_t vprod3x9 = vmull_s8(vbx9, va3);
      const int32x4_t vacc3x9 = vpaddlq_s16(vprod3x9);
      const int8x8_t vbx10 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x10 = vmull_s8(vbx10, va0);
      const int32x4_t vacc0x10 = vpaddlq_s16(vprod0x10);
      const int16x8_t vprod1x10 = vmull_s8(vbx10, va1);
      const int32x4_t vacc1x10 = vpaddlq_s16(vprod1x10);
      const int16x8_t vprod2x10 = vmull_s8(vbx10, va2);
      const int32x4_t vacc2x10 = vpaddlq_s16(vprod2x10);
      const int16x8_t vprod3x10 = vmull_s8(vbx10, va3);
      const int32x4_t vacc3x10 = vpaddlq_s16(vprod3x10);
      const int8x8_t vbx11 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x11 = vmull_s8(vbx11, va0);
      const int32x4_t vacc0x11 = vpaddlq_s16(vprod0x11);
      const int16x8_t vprod1x11 = vmull_s8(vbx11, va1);
      const int32x4_t vacc1x11 = vpaddlq_s16(vprod1x11);
      const int16x8_t vprod2x11 = vmull_s8(vbx11, va2);
      const int32x4_t vacc2x11 = vpaddlq_s16(vprod2x11);
      const int16x8_t vprod3x11 = vmull_s8(vbx11, va3);
      const int32x4_t vacc3x11 = vpaddlq_s16(vprod3x11);
      const int8x8_t vbx12 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x12 = vmull_s8(vbx12, va0);
      const int32x4_t vacc0x12 = vpaddlq_s16(vprod0x12);
      const int16x8_t vprod1x12 = vmull_s8(vbx12, va1);
      const int32x4_t vacc1x12 = vpaddlq_s16(vprod1x12);
      const int16x8_t vprod2x12 = vmull_s8(vbx12, va2);
      const int32x4_t vacc2x12 = vpaddlq_s16(vprod2x12);
      const int16x8_t vprod3x12 = vmull_s8(vbx12, va3);
      const int32x4_t vacc3x12 = vpaddlq_s16(vprod3x12);
      const int8x8_t vbx13 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x13 = vmull_s8(vbx13, va0);
      const int32x4_t vacc0x13 = vpaddlq_s16(vprod0x13);
      const int16x8_t vprod1x13 = vmull_s8(vbx13, va1);
      const int32x4_t vacc1x13 = vpaddlq_s16(vprod1x13);
      const int16x8_t vprod2x13 = vmull_s8(vbx13, va2);
      const int32x4_t vacc2x13 = vpaddlq_s16(vprod2x13);
      const int16x8_t vprod3x13 = vmull_s8(vbx13, va3);
      const int32x4_t vacc3x13 = vpaddlq_s16(vprod3x13);
      const int8x8_t vbx14 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x14 = vmull_s8(vbx14, va0);
      const int32x4_t vacc0x14 = vpaddlq_s16(vprod0x14);
      const int16x8_t vprod1x14 = vmull_s8(vbx14, va1);
      const int32x4_t vacc1x14 = vpaddlq_s16(vprod1x14);
      const int16x8_t vprod2x14 = vmull_s8(vbx14, va2);
      const int32x4_t vacc2x14 = vpaddlq_s16(vprod2x14);
      const int16x8_t vprod3x14 = vmull_s8(vbx14, va3);
      const int32x4_t vacc3x14 = vpaddlq_s16(vprod3x14);
      const int8x8_t vbx15 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x15 = vmull_s8(vbx15, va0);
      const int32x4_t vacc0x15 = vpaddlq_s16(vprod0x15);
      const int16x8_t vprod1x15 = vmull_s8(vbx15, va1);
      const int32x4_t vacc1x15 = vpaddlq_s16(vprod1x15);
      const int16x8_t vprod2x15 = vmull_s8(vbx15, va2);
      const int32x4_t vacc2x15 = vpaddlq_s16(vprod2x15);
      const int16x8_t vprod3x15 = vmull_s8(vbx15, va3);
      const int32x4_t vacc3x15 = vpaddlq_s16(vprod3x15);

      const int32x4_t vprod0x0123 = combine4(vacc0x0, vacc0x1, vacc0x2, vacc0x3);
      vacc0x0123 = vaddq_s32(vacc0x0123, vprod0x0123);
      const int32x4_t vprod0x4567 = combine4(vacc0x4, vacc0x5, vacc0x6, vacc0x7);
      vacc0x4567 = vaddq_s32(vacc0x4567, vprod0x4567);
      const int32x4_t vprod0x89AB = combine4(vacc0x8, vacc0x9, vacc0x10, vacc0x11);
      vacc0x89AB = vaddq_s32(vacc0x89AB, vprod0x89AB);
      const int32x4_t vprod0xCDEF = combine4(vacc0x12, vacc0x13, vacc0x14, vacc0x15);
      vacc0xCDEF = vaddq_s32(vacc0xCDEF, vprod0xCDEF);
      const int32x4_t vprod1x0123 = combine4(vacc1x0, vacc1x1, vacc1x2, vacc1x3);
      vacc1x0123 = vaddq_s32(vacc1x0123, vprod1x0123);
      const int32x4_t vprod1x4567 = combine4(vacc1x4, vacc1x5, vacc1x6, vacc1x7);
      vacc1x4567 = vaddq_s32(vacc1x4567, vprod1x4567);
      const int32x4_t vprod1x89AB = combine4(vacc1x8, vacc1x9, vacc1x10, vacc1x11);
      vacc1x89AB = vaddq_s32(vacc1x89AB, vprod1x89AB);
      const int32x4_t vprod1xCDEF = combine4(vacc1x12, vacc1x13, vacc1x14, vacc1x15);
      vacc1xCDEF = vaddq_s32(vacc1xCDEF, vprod1xCDEF);
      const int32x4_t vprod2x0123 = combine4(vacc2x0, vacc2x1, vacc2x2, vacc2x3);
      vacc2x0123 = vaddq_s32(vacc2x0123, vprod2x0123);
      const int32x4_t vprod2x4567 = combine4(vacc2x4, vacc2x5, vacc2x6, vacc2x7);
      vacc2x4567 = vaddq_s32(vacc2x4567, vprod2x4567);
      const int32x4_t vprod2x89AB = combine4(vacc2x8, vacc2x9, vacc2x10, vacc2x11);
      vacc2x89AB = vaddq_s32(vacc2x89AB, vprod2x89AB);
      const int32x4_t vprod2xCDEF = combine4(vacc2x12, vacc2x13, vacc2x14, vacc2x15);
      vacc2xCDEF = vaddq_s32(vacc2xCDEF, vprod2xCDEF);
      const int32x4_t vprod3x0123 = combine4(vacc3x0, vacc3x1, vacc3x2, vacc3x3);
      vacc3x0123 = vaddq_s32(vacc3x0123, vprod3x0123);
      const int32x4_t vprod3x4567 = combine4(vacc3x4, vacc3x5, vacc3x6, vacc3x7);
      vacc3x4567 = vaddq_s32(vacc3x4567, vprod3x4567);
      const int32x4_t vprod3x89AB = combine4(vacc3x8, vacc3x9, vacc3x10, vacc3x11);
      vacc3x89AB = vaddq_s32(vacc3x89AB, vprod3x89AB);
      const int32x4_t vprod3xCDEF = combine4(vacc3x12, vacc3x13, vacc3x14, vacc3x15);
      vacc3xCDEF = vaddq_s32(vacc3xCDEF, vprod3xCDEF);

    }
    const int32x4_t vmultiplier = vld1q_dup_s32(&params->neon.multiplier);
    vacc0x0123 = vqrdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqrdmulhq_s32(vacc0x4567, vmultiplier);
    vacc0x89AB = vqrdmulhq_s32(vacc0x89AB, vmultiplier);
    vacc0xCDEF = vqrdmulhq_s32(vacc0xCDEF, vmultiplier);
    vacc1x0123 = vqrdmulhq_s32(vacc1x0123, vmultiplier);
    vacc1x4567 = vqrdmulhq_s32(vacc1x4567, vmultiplier);
    vacc1x89AB = vqrdmulhq_s32(vacc1x89AB, vmultiplier);
    vacc1xCDEF = vqrdmulhq_s32(vacc1xCDEF, vmultiplier);
    vacc2x0123 = vqrdmulhq_s32(vacc2x0123, vmultiplier);
    vacc2x4567 = vqrdmulhq_s32(vacc2x4567, vmultiplier);
    vacc2x89AB = vqrdmulhq_s32(vacc2x89AB, vmultiplier);
    vacc2xCDEF = vqrdmulhq_s32(vacc2xCDEF, vmultiplier);
    vacc3x0123 = vqrdmulhq_s32(vacc3x0123, vmultiplier);
    vacc3x4567 = vqrdmulhq_s32(vacc3x4567, vmultiplier);
    vacc3x89AB = vqrdmulhq_s32(vacc3x89AB, vmultiplier);
    vacc3xCDEF = vqrdmulhq_s32(vacc3xCDEF, vmultiplier);

    const int32x4_t vright_shift = vld1q_dup_s32(&params->neon.right_shift);
    const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
    vacc0x0123 = vsraq_n_s32(vacc0x0123, vbicq_s32(vacc0x0123, vzero_shift_mask), 31);
    vacc0x4567 = vsraq_n_s32(vacc0x4567, vbicq_s32(vacc0x4567, vzero_shift_mask), 31);
    vacc0x89AB = vsraq_n_s32(vacc0x89AB, vbicq_s32(vacc0x89AB, vzero_shift_mask), 31);
    vacc0xCDEF = vsraq_n_s32(vacc0xCDEF, vbicq_s32(vacc0xCDEF, vzero_shift_mask), 31);
    vacc1x0123 = vsraq_n_s32(vacc1x0123, vbicq_s32(vacc1x0123, vzero_shift_mask), 31);
    vacc1x4567 = vsraq_n_s32(vacc1x4567, vbicq_s32(vacc1x4567, vzero_shift_mask), 31);
    vacc1x89AB = vsraq_n_s32(vacc1x89AB, vbicq_s32(vacc1x89AB, vzero_shift_mask), 31);
    vacc1xCDEF = vsraq_n_s32(vacc1xCDEF, vbicq_s32(vacc1xCDEF, vzero_shift_mask), 31);
    vacc2x0123 = vsraq_n_s32(vacc2x0123, vbicq_s32(vacc2x0123, vzero_shift_mask), 31);
    vacc2x4567 = vsraq_n_s32(vacc2x4567, vbicq_s32(vacc2x4567, vzero_shift_mask), 31);
    vacc2x89AB = vsraq_n_s32(vacc2x89AB, vbicq_s32(vacc2x89AB, vzero_shift_mask), 31);
    vacc2xCDEF = vsraq_n_s32(vacc2xCDEF, vbicq_s32(vacc2xCDEF, vzero_shift_mask), 31);
    vacc3x0123 = vsraq_n_s32(vacc3x0123, vbicq_s32(vacc3x0123, vzero_shift_mask), 31);
    vacc3x4567 = vsraq_n_s32(vacc3x4567, vbicq_s32(vacc3x4567, vzero_shift_mask), 31);
    vacc3x89AB = vsraq_n_s32(vacc3x89AB, vbicq_s32(vacc3x89AB, vzero_shift_mask), 31);
    vacc3xCDEF = vsraq_n_s32(vacc3xCDEF, vbicq_s32(vacc3xCDEF, vzero_shift_mask), 31);

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_shift);
    vacc0x89AB = vrshlq_s32(vacc0x89AB, vright_shift);
    vacc0xCDEF = vrshlq_s32(vacc0xCDEF, vright_shift);
    vacc1x0123 = vrshlq_s32(vacc1x0123, vright_shift);
    vacc1x4567 = vrshlq_s32(vacc1x4567, vright_shift);
    vacc1x89AB = vrshlq_s32(vacc1x89AB, vright_shift);
    vacc1xCDEF = vrshlq_s32(vacc1xCDEF, vright_shift);
    vacc2x0123 = vrshlq_s32(vacc2x0123, vright_shift);
    vacc2x4567 = vrshlq_s32(vacc2x4567, vright_shift);
    vacc2x89AB = vrshlq_s32(vacc2x89AB, vright_shift);
    vacc2xCDEF = vrshlq_s32(vacc2xCDEF, vright_shift);
    vacc3x0123 = vrshlq_s32(vacc3x0123, vright_shift);
    vacc3x4567 = vrshlq_s32(vacc3x4567, vright_shift);
    vacc3x89AB = vrshlq_s32(vacc3x89AB, vright_shift);
    vacc3xCDEF = vrshlq_s32(vacc3xCDEF, vright_shift);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->neon.output_zero_point);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
    const int16x8_t vacc1x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x89AB), vacc1xCDEF), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
    const int16x8_t vacc2x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x89AB), vacc2xCDEF), voutput_zero_point);
    const int16x8_t vacc3x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);
    const int16x8_t vacc3x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x89AB), vacc3xCDEF), voutput_zero_point);
    int8x16_t vout0x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc0x89ABCDEF);
    int8x16_t vout1x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc1x01234567), vacc1x89ABCDEF);
    int8x16_t vout2x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc2x01234567), vacc2x89ABCDEF);
    int8x16_t vout3x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc3x01234567), vacc3x89ABCDEF);
#else
    const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF)), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
    const int16x8_t vacc1x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x89AB), vqmovn_s32(vacc1xCDEF)), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);
    const int16x8_t vacc2x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x89AB), vqmovn_s32(vacc2xCDEF)), voutput_zero_point);
    const int16x8_t vacc3x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)), voutput_zero_point);
    const int16x8_t vacc3x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x89AB), vqmovn_s32(vacc3xCDEF)), voutput_zero_point);

    int8x16_t vout0x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc0x89ABCDEF));
    int8x16_t vout1x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc1x01234567), vqmovn_s16(vacc1x89ABCDEF));
    int8x16_t vout2x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc2x01234567), vqmovn_s16(vacc2x89ABCDEF));
    int8x16_t vout3x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc3x01234567), vqmovn_s16(vacc3x89ABCDEF));
#endif
    const int8x16_t voutput_min = vld1q_dup_s8(&params->neon.output_min);
    const int8x16_t voutput_max = vld1q_dup_s8(&params->neon.output_max);

    vout0x0123456789ABCDEF = vmaxq_s8(vout0x0123456789ABCDEF, voutput_min);
    vout1x0123456789ABCDEF = vmaxq_s8(vout1x0123456789ABCDEF, voutput_min);
    vout2x0123456789ABCDEF = vmaxq_s8(vout2x0123456789ABCDEF, voutput_min);
    vout3x0123456789ABCDEF = vmaxq_s8(vout3x0123456789ABCDEF, voutput_min);

    vout0x0123456789ABCDEF = vminq_s8(vout0x0123456789ABCDEF, voutput_max);
    vout1x0123456789ABCDEF = vminq_s8(vout1x0123456789ABCDEF, voutput_max);
    vout2x0123456789ABCDEF = vminq_s8(vout2x0123456789ABCDEF, voutput_max);
    vout3x0123456789ABCDEF = vminq_s8(vout3x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      vst1q_s8(c0 + 0, vout0x0123456789ABCDEF);
      vst1q_s8(c1 + 0, vout1x0123456789ABCDEF);
      vst1q_s8(c2 + 0, vout2x0123456789ABCDEF);
      vst1q_s8(c3 + 0, vout3x0123456789ABCDEF);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      nc -= 16;
    } else {
      int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vget_low_s8(vout0x0123456789ABCDEF), vget_low_s8(vout1x0123456789ABCDEF));
      int8x16_t vout2x01234567_3x01234567 = vcombine_s8(vget_low_s8(vout2x0123456789ABCDEF), vget_low_s8(vout3x0123456789ABCDEF));
      if (nc & 8) {
        vst1_s8(c0, vget_low_s8(vout0x01234567_1x01234567)); c0 += 8;
        vst1_s8(c1, vget_high_s8(vout0x01234567_1x01234567)); c1 += 8;
        vst1_s8(c2, vget_low_s8(vout2x01234567_3x01234567)); c2 += 8;
        vst1_s8(c3, vget_high_s8(vout2x01234567_3x01234567)); c3 += 8;
        vout0x01234567_1x01234567 = vcombine_s8(vget_high_s8(vout0x0123456789ABCDEF), vget_high_s8(vout1x0123456789ABCDEF));
        vout2x01234567_3x01234567 = vcombine_s8(vget_high_s8(vout2x0123456789ABCDEF), vget_high_s8(vout3x0123456789ABCDEF));
      }
      if (nc & 4) {
        vst1q_lane_u32(__builtin_assume_aligned(c0, 1), vreinterpretq_u32_s8(vout0x01234567_1x01234567), 0); c0 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c1, 1), vreinterpretq_u32_s8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c2, 1), vreinterpretq_u32_s8(vout2x01234567_3x01234567), 0); c2 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c3, 1), vreinterpretq_u32_s8(vout2x01234567_3x01234567), 2); c3 += 4;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16(__builtin_assume_aligned(c0, 1), vreinterpretq_u16_s8(vout0x01234567_1x01234567), 0); c0 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c1, 1), vreinterpretq_u16_s8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c2, 1), vreinterpretq_u16_s8(vout2x01234567_3x01234567), 0); c2 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c3, 1), vreinterpretq_u16_s8(vout2x01234567_3x01234567), 4); c3 += 2;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_s8(c0, vout0x01234567_1x01234567, 0);
        vst1q_lane_s8(c1, vout0x01234567_1x01234567, 8);
        vst1q_lane_s8(c2, vout2x01234567_3x01234567, 0);
        vst1q_lane_s8(c3, vout2x01234567_3x01234567, 8);
      }

      nc = 0;
    }
  } while (nc != 0);
}
