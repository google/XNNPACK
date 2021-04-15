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


void xnn_qs8_gemm_minmax_ukernel_4x8c8__neon_mull_padal(
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
  assert(mr <= 4);
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
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
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
    int32x4_t vacc1x0 = vacc0x0;
    int32x4_t vacc1x1 = vacc0x1;
    int32x4_t vacc1x2 = vacc0x2;
    int32x4_t vacc1x3 = vacc0x3;
    int32x4_t vacc1x4 = vacc0x4;
    int32x4_t vacc1x5 = vacc0x5;
    int32x4_t vacc1x6 = vacc0x6;
    int32x4_t vacc1x7 = vacc0x7;
    int32x4_t vacc2x0 = vacc0x0;
    int32x4_t vacc2x1 = vacc0x1;
    int32x4_t vacc2x2 = vacc0x2;
    int32x4_t vacc2x3 = vacc0x3;
    int32x4_t vacc2x4 = vacc0x4;
    int32x4_t vacc2x5 = vacc0x5;
    int32x4_t vacc2x6 = vacc0x6;
    int32x4_t vacc2x7 = vacc0x7;
    int32x4_t vacc3x0 = vacc0x0;
    int32x4_t vacc3x1 = vacc0x1;
    int32x4_t vacc3x2 = vacc0x2;
    int32x4_t vacc3x3 = vacc0x3;
    int32x4_t vacc3x4 = vacc0x4;
    int32x4_t vacc3x5 = vacc0x5;
    int32x4_t vacc3x6 = vacc0x6;
    int32x4_t vacc3x7 = vacc0x7;

    size_t k = kc;

    // Handle 8 bytes at a time using MUL.
    while (k > 0) {
      const int8x8_t va0 = vld1_s8(a0); a0 += 8;
      const int8x8_t va1 = vld1_s8(a1); a1 += 8;
      const int8x8_t va2 = vld1_s8(a2); a2 += 8;
      const int8x8_t va3 = vld1_s8(a3); a3 += 8;

      const int8x8_t vb0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x0 = vmull_s8(vb0, va0);
      const int16x8_t vprod1x0 = vmull_s8(vb0, va1);
      const int16x8_t vprod2x0 = vmull_s8(vb0, va2);
      const int16x8_t vprod3x0 = vmull_s8(vb0, va3);
      vacc0x0 = vpadalq_s16(vacc0x0, vprod0x0);
      vacc1x0 = vpadalq_s16(vacc1x0, vprod1x0);
      vacc2x0 = vpadalq_s16(vacc2x0, vprod2x0);
      vacc3x0 = vpadalq_s16(vacc3x0, vprod3x0);
      const int8x8_t vb1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x1 = vmull_s8(vb1, va0);
      const int16x8_t vprod1x1 = vmull_s8(vb1, va1);
      const int16x8_t vprod2x1 = vmull_s8(vb1, va2);
      const int16x8_t vprod3x1 = vmull_s8(vb1, va3);
      vacc0x1 = vpadalq_s16(vacc0x1, vprod0x1);
      vacc1x1 = vpadalq_s16(vacc1x1, vprod1x1);
      vacc2x1 = vpadalq_s16(vacc2x1, vprod2x1);
      vacc3x1 = vpadalq_s16(vacc3x1, vprod3x1);
      const int8x8_t vb2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x2 = vmull_s8(vb2, va0);
      const int16x8_t vprod1x2 = vmull_s8(vb2, va1);
      const int16x8_t vprod2x2 = vmull_s8(vb2, va2);
      const int16x8_t vprod3x2 = vmull_s8(vb2, va3);
      vacc0x2 = vpadalq_s16(vacc0x2, vprod0x2);
      vacc1x2 = vpadalq_s16(vacc1x2, vprod1x2);
      vacc2x2 = vpadalq_s16(vacc2x2, vprod2x2);
      vacc3x2 = vpadalq_s16(vacc3x2, vprod3x2);
      const int8x8_t vb3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x3 = vmull_s8(vb3, va0);
      const int16x8_t vprod1x3 = vmull_s8(vb3, va1);
      const int16x8_t vprod2x3 = vmull_s8(vb3, va2);
      const int16x8_t vprod3x3 = vmull_s8(vb3, va3);
      vacc0x3 = vpadalq_s16(vacc0x3, vprod0x3);
      vacc1x3 = vpadalq_s16(vacc1x3, vprod1x3);
      vacc2x3 = vpadalq_s16(vacc2x3, vprod2x3);
      vacc3x3 = vpadalq_s16(vacc3x3, vprod3x3);
      const int8x8_t vb4 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x4 = vmull_s8(vb4, va0);
      const int16x8_t vprod1x4 = vmull_s8(vb4, va1);
      const int16x8_t vprod2x4 = vmull_s8(vb4, va2);
      const int16x8_t vprod3x4 = vmull_s8(vb4, va3);
      vacc0x4 = vpadalq_s16(vacc0x4, vprod0x4);
      vacc1x4 = vpadalq_s16(vacc1x4, vprod1x4);
      vacc2x4 = vpadalq_s16(vacc2x4, vprod2x4);
      vacc3x4 = vpadalq_s16(vacc3x4, vprod3x4);
      const int8x8_t vb5 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x5 = vmull_s8(vb5, va0);
      const int16x8_t vprod1x5 = vmull_s8(vb5, va1);
      const int16x8_t vprod2x5 = vmull_s8(vb5, va2);
      const int16x8_t vprod3x5 = vmull_s8(vb5, va3);
      vacc0x5 = vpadalq_s16(vacc0x5, vprod0x5);
      vacc1x5 = vpadalq_s16(vacc1x5, vprod1x5);
      vacc2x5 = vpadalq_s16(vacc2x5, vprod2x5);
      vacc3x5 = vpadalq_s16(vacc3x5, vprod3x5);
      const int8x8_t vb6 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x6 = vmull_s8(vb6, va0);
      const int16x8_t vprod1x6 = vmull_s8(vb6, va1);
      const int16x8_t vprod2x6 = vmull_s8(vb6, va2);
      const int16x8_t vprod3x6 = vmull_s8(vb6, va3);
      vacc0x6 = vpadalq_s16(vacc0x6, vprod0x6);
      vacc1x6 = vpadalq_s16(vacc1x6, vprod1x6);
      vacc2x6 = vpadalq_s16(vacc2x6, vprod2x6);
      vacc3x6 = vpadalq_s16(vacc3x6, vprod3x6);
      const int8x8_t vb7 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int16x8_t vprod0x7 = vmull_s8(vb7, va0);
      const int16x8_t vprod1x7 = vmull_s8(vb7, va1);
      const int16x8_t vprod2x7 = vmull_s8(vb7, va2);
      const int16x8_t vprod3x7 = vmull_s8(vb7, va3);
      vacc0x7 = vpadalq_s16(vacc0x7, vprod0x7);
      vacc1x7 = vpadalq_s16(vacc1x7, vprod1x7);
      vacc2x7 = vpadalq_s16(vacc2x7, vprod2x7);
      vacc3x7 = vpadalq_s16(vacc3x7, vprod3x7);

      k -= 8 * sizeof(int8_t);
    }

#if XNN_ARCH_ARM64
    const int32x4_t vsum0x01 = vpaddq_s32(vacc0x0, vacc0x1);
    const int32x4_t vsum0x23 = vpaddq_s32(vacc0x2, vacc0x3);
    const int32x4_t vsum0x45 = vpaddq_s32(vacc0x4, vacc0x5);
    const int32x4_t vsum0x67 = vpaddq_s32(vacc0x6, vacc0x7);
    const int32x4_t vsum1x01 = vpaddq_s32(vacc1x0, vacc1x1);
    const int32x4_t vsum1x23 = vpaddq_s32(vacc1x2, vacc1x3);
    const int32x4_t vsum1x45 = vpaddq_s32(vacc1x4, vacc1x5);
    const int32x4_t vsum1x67 = vpaddq_s32(vacc1x6, vacc1x7);
    const int32x4_t vsum2x01 = vpaddq_s32(vacc2x0, vacc2x1);
    const int32x4_t vsum2x23 = vpaddq_s32(vacc2x2, vacc2x3);
    const int32x4_t vsum2x45 = vpaddq_s32(vacc2x4, vacc2x5);
    const int32x4_t vsum2x67 = vpaddq_s32(vacc2x6, vacc2x7);
    const int32x4_t vsum3x01 = vpaddq_s32(vacc3x0, vacc3x1);
    const int32x4_t vsum3x23 = vpaddq_s32(vacc3x2, vacc3x3);
    const int32x4_t vsum3x45 = vpaddq_s32(vacc3x4, vacc3x5);
    const int32x4_t vsum3x67 = vpaddq_s32(vacc3x6, vacc3x7);
    int32x4_t vacc0x0123 = vpaddq_s32(vsum0x01, vsum0x23);
    int32x4_t vacc0x4567 = vpaddq_s32(vsum0x45, vsum0x67);
    int32x4_t vacc1x0123 = vpaddq_s32(vsum1x01, vsum1x23);
    int32x4_t vacc1x4567 = vpaddq_s32(vsum1x45, vsum1x67);
    int32x4_t vacc2x0123 = vpaddq_s32(vsum2x01, vsum2x23);
    int32x4_t vacc2x4567 = vpaddq_s32(vsum2x45, vsum2x67);
    int32x4_t vacc3x0123 = vpaddq_s32(vsum3x01, vsum3x23);
    int32x4_t vacc3x4567 = vpaddq_s32(vsum3x45, vsum3x67);
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
    const int32x2_t vpsum3x0 = vadd_s32(vget_low_s32(vacc3x0), vget_high_s32(vacc3x0));
    const int32x2_t vpsum3x1 = vadd_s32(vget_low_s32(vacc3x1), vget_high_s32(vacc3x1));
    const int32x2_t vpsum3x2 = vadd_s32(vget_low_s32(vacc3x2), vget_high_s32(vacc3x2));
    const int32x2_t vpsum3x3 = vadd_s32(vget_low_s32(vacc3x3), vget_high_s32(vacc3x3));
    const int32x2_t vsum3x01 = vpadd_s32(vpsum3x0, vpsum3x1);
    const int32x2_t vsum3x23 = vpadd_s32(vpsum3x2, vpsum3x3);
    int32x4_t vacc3x0123 = vcombine_s32(vsum3x01, vsum3x23 );
    const int32x2_t vpsum3x4 = vadd_s32(vget_low_s32(vacc3x4), vget_high_s32(vacc3x4));
    const int32x2_t vpsum3x5 = vadd_s32(vget_low_s32(vacc3x5), vget_high_s32(vacc3x5));
    const int32x2_t vpsum3x6 = vadd_s32(vget_low_s32(vacc3x6), vget_high_s32(vacc3x6));
    const int32x2_t vpsum3x7 = vadd_s32(vget_low_s32(vacc3x7), vget_high_s32(vacc3x7));
    const int32x2_t vsum3x45 = vpadd_s32(vpsum3x4, vpsum3x5);
    const int32x2_t vsum3x67 = vpadd_s32(vpsum3x6, vpsum3x7);
    int32x4_t vacc3x4567 = vcombine_s32(vsum3x45, vsum3x67 );
#endif

    const int32x4_t vmultiplier = vld1q_dup_s32(&params->neon.multiplier);
    vacc0x0123 = vqrdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqrdmulhq_s32(vacc0x4567, vmultiplier);
    vacc1x0123 = vqrdmulhq_s32(vacc1x0123, vmultiplier);
    vacc1x4567 = vqrdmulhq_s32(vacc1x4567, vmultiplier);
    vacc2x0123 = vqrdmulhq_s32(vacc2x0123, vmultiplier);
    vacc2x4567 = vqrdmulhq_s32(vacc2x4567, vmultiplier);
    vacc3x0123 = vqrdmulhq_s32(vacc3x0123, vmultiplier);
    vacc3x4567 = vqrdmulhq_s32(vacc3x4567, vmultiplier);

    const int32x4_t vright_shift = vld1q_dup_s32(&params->neon.right_shift);
    const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
    vacc0x0123 = vcltq_s32(vacc0x0123, vbicq_s32(vacc0x0123, vzero_shift_mask), vmovq_n_s32(0));
    vacc0x4567 = vcltq_s32(vacc0x4567, vbicq_s32(vacc0x4567, vzero_shift_mask), vmovq_n_s32(0));
    vacc1x0123 = vcltq_s32(vacc1x0123, vbicq_s32(vacc1x0123, vzero_shift_mask), vmovq_n_s32(0));
    vacc1x4567 = vcltq_s32(vacc1x4567, vbicq_s32(vacc1x4567, vzero_shift_mask), vmovq_n_s32(0));
    vacc2x0123 = vcltq_s32(vacc2x0123, vbicq_s32(vacc2x0123, vzero_shift_mask), vmovq_n_s32(0));
    vacc2x4567 = vcltq_s32(vacc2x4567, vbicq_s32(vacc2x4567, vzero_shift_mask), vmovq_n_s32(0));
    vacc3x0123 = vcltq_s32(vacc3x0123, vbicq_s32(vacc3x0123, vzero_shift_mask), vmovq_n_s32(0));
    vacc3x4567 = vcltq_s32(vacc3x4567, vbicq_s32(vacc3x4567, vzero_shift_mask), vmovq_n_s32(0));

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_shift);
    vacc1x0123 = vrshlq_s32(vacc1x0123, vright_shift);
    vacc1x4567 = vrshlq_s32(vacc1x4567, vright_shift);
    vacc2x0123 = vrshlq_s32(vacc2x0123, vright_shift);
    vacc2x4567 = vrshlq_s32(vacc2x4567, vright_shift);
    vacc3x0123 = vrshlq_s32(vacc3x0123, vright_shift);
    vacc3x4567 = vrshlq_s32(vacc3x4567, vright_shift);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->neon.output_zero_point);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
    const int16x8_t vacc3x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);
    int8x16_t vout0x01234567_1x01234567 = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc1x01234567);
    int8x16_t vout2x01234567_3x01234567 = vqmovn_high_s16(vqmovn_s16(vacc2x01234567), vacc3x01234567);
#else
    const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);
    const int16x8_t vacc3x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)), voutput_zero_point);

    int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc1x01234567));
    int8x16_t vout2x01234567_3x01234567 = vcombine_s8(vqmovn_s16(vacc2x01234567), vqmovn_s16(vacc3x01234567));
#endif
    const int8x16_t voutput_min = vld1q_dup_s8(&params->neon.output_min);
    const int8x16_t voutput_max = vld1q_dup_s8(&params->neon.output_max);

    vout0x01234567_1x01234567 = vmaxq_s8(vout0x01234567_1x01234567, voutput_min);
    vout2x01234567_3x01234567 = vmaxq_s8(vout2x01234567_3x01234567, voutput_min);

    vout0x01234567_1x01234567 = vminq_s8(vout0x01234567_1x01234567, voutput_max);
    vout2x01234567_3x01234567 = vminq_s8(vout2x01234567_3x01234567, voutput_max);

    if (nc >= 8) {
      vst1_s8(c0 + 0, vget_low_s8(vout0x01234567_1x01234567));
      vst1_s8(c1 + 0, vget_high_s8(vout0x01234567_1x01234567));
      vst1_s8(c2 + 0, vget_low_s8(vout2x01234567_3x01234567));
      vst1_s8(c3 + 0, vget_high_s8(vout2x01234567_3x01234567));

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      nc -= 8;
    } else {
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
