// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c8-neon-mull-padal.c.in
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


void xnn_qs8_gemm_minmax_ukernel_3x16c8__neon_mull_padal(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_gemm_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN XNN_DISABLE_MSAN
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
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

  do {
    int32x4_t vacc0x0 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const void*) ((uintptr_t) w + sizeof(int32_t));
    int32x4_t vacc0x1 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const void*) ((uintptr_t) w + sizeof(int32_t));
    int32x4_t vacc0x2 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const void*) ((uintptr_t) w + sizeof(int32_t));
    int32x4_t vacc0x3 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const void*) ((uintptr_t) w + sizeof(int32_t));
    int32x4_t vacc0x4 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const void*) ((uintptr_t) w + sizeof(int32_t));
    int32x4_t vacc0x5 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const void*) ((uintptr_t) w + sizeof(int32_t));
    int32x4_t vacc0x6 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const void*) ((uintptr_t) w + sizeof(int32_t));
    int32x4_t vacc0x7 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const void*) ((uintptr_t) w + sizeof(int32_t));
    int32x4_t vacc0x8 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const void*) ((uintptr_t) w + sizeof(int32_t));
    int32x4_t vacc0x9 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const void*) ((uintptr_t) w + sizeof(int32_t));
    int32x4_t vacc0x10 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const void*) ((uintptr_t) w + sizeof(int32_t));
    int32x4_t vacc0x11 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const void*) ((uintptr_t) w + sizeof(int32_t));
    int32x4_t vacc0x12 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const void*) ((uintptr_t) w + sizeof(int32_t));
    int32x4_t vacc0x13 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const void*) ((uintptr_t) w + sizeof(int32_t));
    int32x4_t vacc0x14 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const void*) ((uintptr_t) w + sizeof(int32_t));
    int32x4_t vacc0x15 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const void*) ((uintptr_t) w + sizeof(int32_t));
    int32x4_t vacc1x0 = vacc0x0;
    int32x4_t vacc1x1 = vacc0x1;
    int32x4_t vacc1x2 = vacc0x2;
    int32x4_t vacc1x3 = vacc0x3;
    int32x4_t vacc1x4 = vacc0x4;
    int32x4_t vacc1x5 = vacc0x5;
    int32x4_t vacc1x6 = vacc0x6;
    int32x4_t vacc1x7 = vacc0x7;
    int32x4_t vacc1x8 = vacc0x8;
    int32x4_t vacc1x9 = vacc0x9;
    int32x4_t vacc1x10 = vacc0x10;
    int32x4_t vacc1x11 = vacc0x11;
    int32x4_t vacc1x12 = vacc0x12;
    int32x4_t vacc1x13 = vacc0x13;
    int32x4_t vacc1x14 = vacc0x14;
    int32x4_t vacc1x15 = vacc0x15;
    int32x4_t vacc2x0 = vacc0x0;
    int32x4_t vacc2x1 = vacc0x1;
    int32x4_t vacc2x2 = vacc0x2;
    int32x4_t vacc2x3 = vacc0x3;
    int32x4_t vacc2x4 = vacc0x4;
    int32x4_t vacc2x5 = vacc0x5;
    int32x4_t vacc2x6 = vacc0x6;
    int32x4_t vacc2x7 = vacc0x7;
    int32x4_t vacc2x8 = vacc0x8;
    int32x4_t vacc2x9 = vacc0x9;
    int32x4_t vacc2x10 = vacc0x10;
    int32x4_t vacc2x11 = vacc0x11;
    int32x4_t vacc2x12 = vacc0x12;
    int32x4_t vacc2x13 = vacc0x13;
    int32x4_t vacc2x14 = vacc0x14;
    int32x4_t vacc2x15 = vacc0x15;

    size_t k = kc;

    // Handle 8 bytes at a time using MUL.
    while (k > 0) {
      const int8x8_t va0 = vld1_s8(a0); a0 += 8;
      const int8x8_t va1 = vld1_s8(a1); a1 += 8;
      const int8x8_t va2 = vld1_s8(a2); a2 += 8;

      const int8x8_t vb0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x0 = vmull_s8(vb0, va0);
      const int16x8_t vprod1x0 = vmull_s8(vb0, va1);
      const int16x8_t vprod2x0 = vmull_s8(vb0, va2);
      vacc0x0 = vpadalq_s16(vacc0x0, vprod0x0);
      vacc1x0 = vpadalq_s16(vacc1x0, vprod1x0);
      vacc2x0 = vpadalq_s16(vacc2x0, vprod2x0);
      const int8x8_t vb1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x1 = vmull_s8(vb1, va0);
      const int16x8_t vprod1x1 = vmull_s8(vb1, va1);
      const int16x8_t vprod2x1 = vmull_s8(vb1, va2);
      vacc0x1 = vpadalq_s16(vacc0x1, vprod0x1);
      vacc1x1 = vpadalq_s16(vacc1x1, vprod1x1);
      vacc2x1 = vpadalq_s16(vacc2x1, vprod2x1);
      const int8x8_t vb2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x2 = vmull_s8(vb2, va0);
      const int16x8_t vprod1x2 = vmull_s8(vb2, va1);
      const int16x8_t vprod2x2 = vmull_s8(vb2, va2);
      vacc0x2 = vpadalq_s16(vacc0x2, vprod0x2);
      vacc1x2 = vpadalq_s16(vacc1x2, vprod1x2);
      vacc2x2 = vpadalq_s16(vacc2x2, vprod2x2);
      const int8x8_t vb3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x3 = vmull_s8(vb3, va0);
      const int16x8_t vprod1x3 = vmull_s8(vb3, va1);
      const int16x8_t vprod2x3 = vmull_s8(vb3, va2);
      vacc0x3 = vpadalq_s16(vacc0x3, vprod0x3);
      vacc1x3 = vpadalq_s16(vacc1x3, vprod1x3);
      vacc2x3 = vpadalq_s16(vacc2x3, vprod2x3);
      const int8x8_t vb4 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x4 = vmull_s8(vb4, va0);
      const int16x8_t vprod1x4 = vmull_s8(vb4, va1);
      const int16x8_t vprod2x4 = vmull_s8(vb4, va2);
      vacc0x4 = vpadalq_s16(vacc0x4, vprod0x4);
      vacc1x4 = vpadalq_s16(vacc1x4, vprod1x4);
      vacc2x4 = vpadalq_s16(vacc2x4, vprod2x4);
      const int8x8_t vb5 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x5 = vmull_s8(vb5, va0);
      const int16x8_t vprod1x5 = vmull_s8(vb5, va1);
      const int16x8_t vprod2x5 = vmull_s8(vb5, va2);
      vacc0x5 = vpadalq_s16(vacc0x5, vprod0x5);
      vacc1x5 = vpadalq_s16(vacc1x5, vprod1x5);
      vacc2x5 = vpadalq_s16(vacc2x5, vprod2x5);
      const int8x8_t vb6 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x6 = vmull_s8(vb6, va0);
      const int16x8_t vprod1x6 = vmull_s8(vb6, va1);
      const int16x8_t vprod2x6 = vmull_s8(vb6, va2);
      vacc0x6 = vpadalq_s16(vacc0x6, vprod0x6);
      vacc1x6 = vpadalq_s16(vacc1x6, vprod1x6);
      vacc2x6 = vpadalq_s16(vacc2x6, vprod2x6);
      const int8x8_t vb7 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x7 = vmull_s8(vb7, va0);
      const int16x8_t vprod1x7 = vmull_s8(vb7, va1);
      const int16x8_t vprod2x7 = vmull_s8(vb7, va2);
      vacc0x7 = vpadalq_s16(vacc0x7, vprod0x7);
      vacc1x7 = vpadalq_s16(vacc1x7, vprod1x7);
      vacc2x7 = vpadalq_s16(vacc2x7, vprod2x7);
      const int8x8_t vb8 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x8 = vmull_s8(vb8, va0);
      const int16x8_t vprod1x8 = vmull_s8(vb8, va1);
      const int16x8_t vprod2x8 = vmull_s8(vb8, va2);
      vacc0x8 = vpadalq_s16(vacc0x8, vprod0x8);
      vacc1x8 = vpadalq_s16(vacc1x8, vprod1x8);
      vacc2x8 = vpadalq_s16(vacc2x8, vprod2x8);
      const int8x8_t vb9 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x9 = vmull_s8(vb9, va0);
      const int16x8_t vprod1x9 = vmull_s8(vb9, va1);
      const int16x8_t vprod2x9 = vmull_s8(vb9, va2);
      vacc0x9 = vpadalq_s16(vacc0x9, vprod0x9);
      vacc1x9 = vpadalq_s16(vacc1x9, vprod1x9);
      vacc2x9 = vpadalq_s16(vacc2x9, vprod2x9);
      const int8x8_t vb10 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x10 = vmull_s8(vb10, va0);
      const int16x8_t vprod1x10 = vmull_s8(vb10, va1);
      const int16x8_t vprod2x10 = vmull_s8(vb10, va2);
      vacc0x10 = vpadalq_s16(vacc0x10, vprod0x10);
      vacc1x10 = vpadalq_s16(vacc1x10, vprod1x10);
      vacc2x10 = vpadalq_s16(vacc2x10, vprod2x10);
      const int8x8_t vb11 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x11 = vmull_s8(vb11, va0);
      const int16x8_t vprod1x11 = vmull_s8(vb11, va1);
      const int16x8_t vprod2x11 = vmull_s8(vb11, va2);
      vacc0x11 = vpadalq_s16(vacc0x11, vprod0x11);
      vacc1x11 = vpadalq_s16(vacc1x11, vprod1x11);
      vacc2x11 = vpadalq_s16(vacc2x11, vprod2x11);
      const int8x8_t vb12 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x12 = vmull_s8(vb12, va0);
      const int16x8_t vprod1x12 = vmull_s8(vb12, va1);
      const int16x8_t vprod2x12 = vmull_s8(vb12, va2);
      vacc0x12 = vpadalq_s16(vacc0x12, vprod0x12);
      vacc1x12 = vpadalq_s16(vacc1x12, vprod1x12);
      vacc2x12 = vpadalq_s16(vacc2x12, vprod2x12);
      const int8x8_t vb13 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x13 = vmull_s8(vb13, va0);
      const int16x8_t vprod1x13 = vmull_s8(vb13, va1);
      const int16x8_t vprod2x13 = vmull_s8(vb13, va2);
      vacc0x13 = vpadalq_s16(vacc0x13, vprod0x13);
      vacc1x13 = vpadalq_s16(vacc1x13, vprod1x13);
      vacc2x13 = vpadalq_s16(vacc2x13, vprod2x13);
      const int8x8_t vb14 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x14 = vmull_s8(vb14, va0);
      const int16x8_t vprod1x14 = vmull_s8(vb14, va1);
      const int16x8_t vprod2x14 = vmull_s8(vb14, va2);
      vacc0x14 = vpadalq_s16(vacc0x14, vprod0x14);
      vacc1x14 = vpadalq_s16(vacc1x14, vprod1x14);
      vacc2x14 = vpadalq_s16(vacc2x14, vprod2x14);
      const int8x8_t vb15 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x15 = vmull_s8(vb15, va0);
      const int16x8_t vprod1x15 = vmull_s8(vb15, va1);
      const int16x8_t vprod2x15 = vmull_s8(vb15, va2);
      vacc0x15 = vpadalq_s16(vacc0x15, vprod0x15);
      vacc1x15 = vpadalq_s16(vacc1x15, vprod1x15);
      vacc2x15 = vpadalq_s16(vacc2x15, vprod2x15);

      k -= 8 * sizeof(int8_t);
    }

#if XNN_ARCH_ARM64
    const int32x4_t vsum0x01 = vpaddq_s32(vacc0x0, vacc0x1);
    const int32x4_t vsum0x23 = vpaddq_s32(vacc0x2, vacc0x3);
    const int32x4_t vsum0x45 = vpaddq_s32(vacc0x4, vacc0x5);
    const int32x4_t vsum0x67 = vpaddq_s32(vacc0x6, vacc0x7);
    const int32x4_t vsum0x89 = vpaddq_s32(vacc0x8, vacc0x9);
    const int32x4_t vsum0xAB = vpaddq_s32(vacc0x10, vacc0x11);
    const int32x4_t vsum0xCD = vpaddq_s32(vacc0x12, vacc0x13);
    const int32x4_t vsum0xEF = vpaddq_s32(vacc0x14, vacc0x15);
    const int32x4_t vsum1x01 = vpaddq_s32(vacc1x0, vacc1x1);
    const int32x4_t vsum1x23 = vpaddq_s32(vacc1x2, vacc1x3);
    const int32x4_t vsum1x45 = vpaddq_s32(vacc1x4, vacc1x5);
    const int32x4_t vsum1x67 = vpaddq_s32(vacc1x6, vacc1x7);
    const int32x4_t vsum1x89 = vpaddq_s32(vacc1x8, vacc1x9);
    const int32x4_t vsum1xAB = vpaddq_s32(vacc1x10, vacc1x11);
    const int32x4_t vsum1xCD = vpaddq_s32(vacc1x12, vacc1x13);
    const int32x4_t vsum1xEF = vpaddq_s32(vacc1x14, vacc1x15);
    const int32x4_t vsum2x01 = vpaddq_s32(vacc2x0, vacc2x1);
    const int32x4_t vsum2x23 = vpaddq_s32(vacc2x2, vacc2x3);
    const int32x4_t vsum2x45 = vpaddq_s32(vacc2x4, vacc2x5);
    const int32x4_t vsum2x67 = vpaddq_s32(vacc2x6, vacc2x7);
    const int32x4_t vsum2x89 = vpaddq_s32(vacc2x8, vacc2x9);
    const int32x4_t vsum2xAB = vpaddq_s32(vacc2x10, vacc2x11);
    const int32x4_t vsum2xCD = vpaddq_s32(vacc2x12, vacc2x13);
    const int32x4_t vsum2xEF = vpaddq_s32(vacc2x14, vacc2x15);
    int32x4_t vacc0x0123 = vpaddq_s32(vsum0x01, vsum0x23);
    int32x4_t vacc0x4567 = vpaddq_s32(vsum0x45, vsum0x67);
    int32x4_t vacc0x89AB = vpaddq_s32(vsum0x89, vsum0xAB);
    int32x4_t vacc0xCDEF = vpaddq_s32(vsum0xCD, vsum0xEF);
    int32x4_t vacc1x0123 = vpaddq_s32(vsum1x01, vsum1x23);
    int32x4_t vacc1x4567 = vpaddq_s32(vsum1x45, vsum1x67);
    int32x4_t vacc1x89AB = vpaddq_s32(vsum1x89, vsum1xAB);
    int32x4_t vacc1xCDEF = vpaddq_s32(vsum1xCD, vsum1xEF);
    int32x4_t vacc2x0123 = vpaddq_s32(vsum2x01, vsum2x23);
    int32x4_t vacc2x4567 = vpaddq_s32(vsum2x45, vsum2x67);
    int32x4_t vacc2x89AB = vpaddq_s32(vsum2x89, vsum2xAB);
    int32x4_t vacc2xCDEF = vpaddq_s32(vsum2xCD, vsum2xEF);
#else
    const int32x2_t vpsum0x0 = vadd_s32(vget_low_s32(vacc0x0), vget_high_s32(vacc0x0));
    const int32x2_t vpsum0x1 = vadd_s32(vget_low_s32(vacc0x1), vget_high_s32(vacc0x1));
    const int32x2_t vpsum0x2 = vadd_s32(vget_low_s32(vacc0x2), vget_high_s32(vacc0x2));
    const int32x2_t vpsum0x3 = vadd_s32(vget_low_s32(vacc0x3), vget_high_s32(vacc0x3));
    const int32x2_t vsum0x01 = vpadd_s32(vpsum0x0, vpsum0x1);
    const int32x2_t vsum0x23 = vpadd_s32(vpsum0x2, vpsum0x3);
    int32x4_t vacc0x0123 = vcombine_s32(vsum0x01, vsum0x23 );
    const int32x2_t vpsum0x4 = vadd_s32(vget_low_s32(vacc0x4), vget_high_s32(vacc0x4));
    const int32x2_t vpsum0x5 = vadd_s32(vget_low_s32(vacc0x5), vget_high_s32(vacc0x5));
    const int32x2_t vpsum0x6 = vadd_s32(vget_low_s32(vacc0x6), vget_high_s32(vacc0x6));
    const int32x2_t vpsum0x7 = vadd_s32(vget_low_s32(vacc0x7), vget_high_s32(vacc0x7));
    const int32x2_t vsum0x45 = vpadd_s32(vpsum0x4, vpsum0x5);
    const int32x2_t vsum0x67 = vpadd_s32(vpsum0x6, vpsum0x7);
    int32x4_t vacc0x4567 = vcombine_s32(vsum0x45, vsum0x67 );
    const int32x2_t vpsum0x8 = vadd_s32(vget_low_s32(vacc0x8), vget_high_s32(vacc0x8));
    const int32x2_t vpsum0x9 = vadd_s32(vget_low_s32(vacc0x9), vget_high_s32(vacc0x9));
    const int32x2_t vpsum0xA = vadd_s32(vget_low_s32(vacc0x10), vget_high_s32(vacc0x10));
    const int32x2_t vpsum0xB = vadd_s32(vget_low_s32(vacc0x11), vget_high_s32(vacc0x11));
    const int32x2_t vsum0x89 = vpadd_s32(vpsum0x8, vpsum0x9);
    const int32x2_t vsum0xAB = vpadd_s32(vpsum0xA, vpsum0xB);
    int32x4_t vacc0x89AB = vcombine_s32(vsum0x89, vsum0xAB );
    const int32x2_t vpsum0xC = vadd_s32(vget_low_s32(vacc0x12), vget_high_s32(vacc0x12));
    const int32x2_t vpsum0xD = vadd_s32(vget_low_s32(vacc0x13), vget_high_s32(vacc0x13));
    const int32x2_t vpsum0xE = vadd_s32(vget_low_s32(vacc0x14), vget_high_s32(vacc0x14));
    const int32x2_t vpsum0xF = vadd_s32(vget_low_s32(vacc0x15), vget_high_s32(vacc0x15));
    const int32x2_t vsum0xCD = vpadd_s32(vpsum0xC, vpsum0xD);
    const int32x2_t vsum0xEF = vpadd_s32(vpsum0xE, vpsum0xF);
    int32x4_t vacc0xCDEF = vcombine_s32(vsum0xCD, vsum0xEF );
    const int32x2_t vpsum1x0 = vadd_s32(vget_low_s32(vacc1x0), vget_high_s32(vacc1x0));
    const int32x2_t vpsum1x1 = vadd_s32(vget_low_s32(vacc1x1), vget_high_s32(vacc1x1));
    const int32x2_t vpsum1x2 = vadd_s32(vget_low_s32(vacc1x2), vget_high_s32(vacc1x2));
    const int32x2_t vpsum1x3 = vadd_s32(vget_low_s32(vacc1x3), vget_high_s32(vacc1x3));
    const int32x2_t vsum1x01 = vpadd_s32(vpsum1x0, vpsum1x1);
    const int32x2_t vsum1x23 = vpadd_s32(vpsum1x2, vpsum1x3);
    int32x4_t vacc1x0123 = vcombine_s32(vsum1x01, vsum1x23 );
    const int32x2_t vpsum1x4 = vadd_s32(vget_low_s32(vacc1x4), vget_high_s32(vacc1x4));
    const int32x2_t vpsum1x5 = vadd_s32(vget_low_s32(vacc1x5), vget_high_s32(vacc1x5));
    const int32x2_t vpsum1x6 = vadd_s32(vget_low_s32(vacc1x6), vget_high_s32(vacc1x6));
    const int32x2_t vpsum1x7 = vadd_s32(vget_low_s32(vacc1x7), vget_high_s32(vacc1x7));
    const int32x2_t vsum1x45 = vpadd_s32(vpsum1x4, vpsum1x5);
    const int32x2_t vsum1x67 = vpadd_s32(vpsum1x6, vpsum1x7);
    int32x4_t vacc1x4567 = vcombine_s32(vsum1x45, vsum1x67 );
    const int32x2_t vpsum1x8 = vadd_s32(vget_low_s32(vacc1x8), vget_high_s32(vacc1x8));
    const int32x2_t vpsum1x9 = vadd_s32(vget_low_s32(vacc1x9), vget_high_s32(vacc1x9));
    const int32x2_t vpsum1xA = vadd_s32(vget_low_s32(vacc1x10), vget_high_s32(vacc1x10));
    const int32x2_t vpsum1xB = vadd_s32(vget_low_s32(vacc1x11), vget_high_s32(vacc1x11));
    const int32x2_t vsum1x89 = vpadd_s32(vpsum1x8, vpsum1x9);
    const int32x2_t vsum1xAB = vpadd_s32(vpsum1xA, vpsum1xB);
    int32x4_t vacc1x89AB = vcombine_s32(vsum1x89, vsum1xAB );
    const int32x2_t vpsum1xC = vadd_s32(vget_low_s32(vacc1x12), vget_high_s32(vacc1x12));
    const int32x2_t vpsum1xD = vadd_s32(vget_low_s32(vacc1x13), vget_high_s32(vacc1x13));
    const int32x2_t vpsum1xE = vadd_s32(vget_low_s32(vacc1x14), vget_high_s32(vacc1x14));
    const int32x2_t vpsum1xF = vadd_s32(vget_low_s32(vacc1x15), vget_high_s32(vacc1x15));
    const int32x2_t vsum1xCD = vpadd_s32(vpsum1xC, vpsum1xD);
    const int32x2_t vsum1xEF = vpadd_s32(vpsum1xE, vpsum1xF);
    int32x4_t vacc1xCDEF = vcombine_s32(vsum1xCD, vsum1xEF );
    const int32x2_t vpsum2x0 = vadd_s32(vget_low_s32(vacc2x0), vget_high_s32(vacc2x0));
    const int32x2_t vpsum2x1 = vadd_s32(vget_low_s32(vacc2x1), vget_high_s32(vacc2x1));
    const int32x2_t vpsum2x2 = vadd_s32(vget_low_s32(vacc2x2), vget_high_s32(vacc2x2));
    const int32x2_t vpsum2x3 = vadd_s32(vget_low_s32(vacc2x3), vget_high_s32(vacc2x3));
    const int32x2_t vsum2x01 = vpadd_s32(vpsum2x0, vpsum2x1);
    const int32x2_t vsum2x23 = vpadd_s32(vpsum2x2, vpsum2x3);
    int32x4_t vacc2x0123 = vcombine_s32(vsum2x01, vsum2x23 );
    const int32x2_t vpsum2x4 = vadd_s32(vget_low_s32(vacc2x4), vget_high_s32(vacc2x4));
    const int32x2_t vpsum2x5 = vadd_s32(vget_low_s32(vacc2x5), vget_high_s32(vacc2x5));
    const int32x2_t vpsum2x6 = vadd_s32(vget_low_s32(vacc2x6), vget_high_s32(vacc2x6));
    const int32x2_t vpsum2x7 = vadd_s32(vget_low_s32(vacc2x7), vget_high_s32(vacc2x7));
    const int32x2_t vsum2x45 = vpadd_s32(vpsum2x4, vpsum2x5);
    const int32x2_t vsum2x67 = vpadd_s32(vpsum2x6, vpsum2x7);
    int32x4_t vacc2x4567 = vcombine_s32(vsum2x45, vsum2x67 );
    const int32x2_t vpsum2x8 = vadd_s32(vget_low_s32(vacc2x8), vget_high_s32(vacc2x8));
    const int32x2_t vpsum2x9 = vadd_s32(vget_low_s32(vacc2x9), vget_high_s32(vacc2x9));
    const int32x2_t vpsum2xA = vadd_s32(vget_low_s32(vacc2x10), vget_high_s32(vacc2x10));
    const int32x2_t vpsum2xB = vadd_s32(vget_low_s32(vacc2x11), vget_high_s32(vacc2x11));
    const int32x2_t vsum2x89 = vpadd_s32(vpsum2x8, vpsum2x9);
    const int32x2_t vsum2xAB = vpadd_s32(vpsum2xA, vpsum2xB);
    int32x4_t vacc2x89AB = vcombine_s32(vsum2x89, vsum2xAB );
    const int32x2_t vpsum2xC = vadd_s32(vget_low_s32(vacc2x12), vget_high_s32(vacc2x12));
    const int32x2_t vpsum2xD = vadd_s32(vget_low_s32(vacc2x13), vget_high_s32(vacc2x13));
    const int32x2_t vpsum2xE = vadd_s32(vget_low_s32(vacc2x14), vget_high_s32(vacc2x14));
    const int32x2_t vpsum2xF = vadd_s32(vget_low_s32(vacc2x15), vget_high_s32(vacc2x15));
    const int32x2_t vsum2xCD = vpadd_s32(vpsum2xC, vpsum2xD);
    const int32x2_t vsum2xEF = vpadd_s32(vpsum2xE, vpsum2xF);
    int32x4_t vacc2xCDEF = vcombine_s32(vsum2xCD, vsum2xEF );
#endif

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

    const int32x4_t vright_shift = vld1q_dup_s32(&params->neon.right_shift);
    const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
    vacc0x0123 = vcltq_s32(vacc0x0123, vbicq_s32(vacc0x0123, vzero_shift_mask), vmovq_n_s32(0));
    vacc0x4567 = vcltq_s32(vacc0x4567, vbicq_s32(vacc0x4567, vzero_shift_mask), vmovq_n_s32(0));
    vacc0x89AB = vcltq_s32(vacc0x89AB, vbicq_s32(vacc0x89AB, vzero_shift_mask), vmovq_n_s32(0));
    vacc0xCDEF = vcltq_s32(vacc0xCDEF, vbicq_s32(vacc0xCDEF, vzero_shift_mask), vmovq_n_s32(0));
    vacc1x0123 = vcltq_s32(vacc1x0123, vbicq_s32(vacc1x0123, vzero_shift_mask), vmovq_n_s32(0));
    vacc1x4567 = vcltq_s32(vacc1x4567, vbicq_s32(vacc1x4567, vzero_shift_mask), vmovq_n_s32(0));
    vacc1x89AB = vcltq_s32(vacc1x89AB, vbicq_s32(vacc1x89AB, vzero_shift_mask), vmovq_n_s32(0));
    vacc1xCDEF = vcltq_s32(vacc1xCDEF, vbicq_s32(vacc1xCDEF, vzero_shift_mask), vmovq_n_s32(0));
    vacc2x0123 = vcltq_s32(vacc2x0123, vbicq_s32(vacc2x0123, vzero_shift_mask), vmovq_n_s32(0));
    vacc2x4567 = vcltq_s32(vacc2x4567, vbicq_s32(vacc2x4567, vzero_shift_mask), vmovq_n_s32(0));
    vacc2x89AB = vcltq_s32(vacc2x89AB, vbicq_s32(vacc2x89AB, vzero_shift_mask), vmovq_n_s32(0));
    vacc2xCDEF = vcltq_s32(vacc2xCDEF, vbicq_s32(vacc2xCDEF, vzero_shift_mask), vmovq_n_s32(0));

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

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->neon.output_zero_point);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
    const int16x8_t vacc1x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x89AB), vacc1xCDEF), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
    const int16x8_t vacc2x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x89AB), vacc2xCDEF), voutput_zero_point);
    int8x16_t vout0x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc0x89ABCDEF);
    int8x16_t vout1x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc1x01234567), vacc1x89ABCDEF);
    int8x16_t vout2x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc2x01234567), vacc2x89ABCDEF);
#else
    const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF)), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
    const int16x8_t vacc1x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x89AB), vqmovn_s32(vacc1xCDEF)), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);
    const int16x8_t vacc2x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x89AB), vqmovn_s32(vacc2xCDEF)), voutput_zero_point);

    int8x16_t vout0x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc0x89ABCDEF));
    int8x16_t vout1x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc1x01234567), vqmovn_s16(vacc1x89ABCDEF));
    int8x16_t vout2x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc2x01234567), vqmovn_s16(vacc2x89ABCDEF));
#endif
    const int8x16_t voutput_min = vld1q_dup_s8(&params->neon.output_min);
    const int8x16_t voutput_max = vld1q_dup_s8(&params->neon.output_max);

    vout0x0123456789ABCDEF = vmaxq_s8(vout0x0123456789ABCDEF, voutput_min);
    vout1x0123456789ABCDEF = vmaxq_s8(vout1x0123456789ABCDEF, voutput_min);
    vout2x0123456789ABCDEF = vmaxq_s8(vout2x0123456789ABCDEF, voutput_min);

    vout0x0123456789ABCDEF = vminq_s8(vout0x0123456789ABCDEF, voutput_max);
    vout1x0123456789ABCDEF = vminq_s8(vout1x0123456789ABCDEF, voutput_max);
    vout2x0123456789ABCDEF = vminq_s8(vout2x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      vst1q_s8(c0 + 0, vout0x0123456789ABCDEF);
      vst1q_s8(c1 + 0, vout1x0123456789ABCDEF);
      vst1q_s8(c2 + 0, vout2x0123456789ABCDEF);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);

      nc -= 16;
    } else {
      int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vget_low_s8(vout0x0123456789ABCDEF), vget_low_s8(vout1x0123456789ABCDEF));
      int8x8_t vout2x01234567 = vget_low_s8(vout2x0123456789ABCDEF);
      if (nc & 8) {
        vst1_s8(c0, vget_low_s8(vout0x01234567_1x01234567)); c0 += 8;
        vst1_s8(c1, vget_high_s8(vout0x01234567_1x01234567)); c1 += 8;
        vst1_s8(c2, vout2x01234567); c2 += 8;
        vout0x01234567_1x01234567 = vcombine_s8(vget_high_s8(vout0x0123456789ABCDEF), vget_high_s8(vout1x0123456789ABCDEF));
        vout2x01234567 = vget_high_s8(vout2x0123456789ABCDEF);
      }
      if (nc & 4) {
        vst1q_lane_u32(__builtin_assume_aligned(c0, 1), vreinterpretq_u32_s8(vout0x01234567_1x01234567), 0); c0 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c1, 1), vreinterpretq_u32_s8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1_lane_u32(__builtin_assume_aligned(c2, 1), vreinterpret_u32_s8(vout2x01234567), 0); c2 += 4;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
        vout2x01234567 = vext_s8(vout2x01234567, vout2x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16(__builtin_assume_aligned(c0, 1), vreinterpretq_u16_s8(vout0x01234567_1x01234567), 0); c0 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c1, 1), vreinterpretq_u16_s8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1_lane_u16(__builtin_assume_aligned(c2, 1), vreinterpret_u16_s8(vout2x01234567), 0); c2 += 2;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
        vout2x01234567 = vext_s8(vout2x01234567, vout2x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_s8(c0, vout0x01234567_1x01234567, 0);
        vst1q_lane_s8(c1, vout0x01234567_1x01234567, 8);
        vst1_lane_s8(c2, vout2x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
