// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c16-neon-mlal.c.in
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


void xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c16__neon_mlal(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 16 * sizeof(int8_t));
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

    // KC loop of 16
    size_t k = kc;
    while (k != 0) {
      const int8x16_t va0 = vld1q_s8(a0); a0 += 16;
      const int8x16_t va1 = vld1q_s8(a1); a1 += 16;
      const int8x16_t va2 = vld1q_s8(a2); a2 += 16;

      const int8x16_t vb0 = vld1q_s8(w); w = (const void*) ((uintptr_t) w + 16 * sizeof(int8_t));
      const int8x16_t vb1 = vld1q_s8(w); w = (const void*) ((uintptr_t) w + 16 * sizeof(int8_t));
      const int8x16_t vb2 = vld1q_s8(w); w = (const void*) ((uintptr_t) w + 16 * sizeof(int8_t));
      const int8x16_t vb3 = vld1q_s8(w); w = (const void*) ((uintptr_t) w + 16 * sizeof(int8_t));
      const int8x16_t vb4 = vld1q_s8(w); w = (const void*) ((uintptr_t) w + 16 * sizeof(int8_t));
      const int8x16_t vb5 = vld1q_s8(w); w = (const void*) ((uintptr_t) w + 16 * sizeof(int8_t));
      const int8x16_t vb6 = vld1q_s8(w); w = (const void*) ((uintptr_t) w + 16 * sizeof(int8_t));
      const int8x16_t vb7 = vld1q_s8(w); w = (const void*) ((uintptr_t) w + 16 * sizeof(int8_t));

      int16x8_t vprod0x0 = vmull_s8(vget_low_s8(vb0), vget_low_s8(va0));
      int16x8_t vprod1x0 = vmull_s8(vget_low_s8(vb0), vget_low_s8(va1));
      int16x8_t vprod2x0 = vmull_s8(vget_low_s8(vb0), vget_low_s8(va2));
      vprod0x0 = vmlal_s8(vprod0x0, vget_high_s8(vb0), vget_high_s8(va0));
      vprod1x0 = vmlal_s8(vprod1x0, vget_high_s8(vb0), vget_high_s8(va1));
      vprod2x0 = vmlal_s8(vprod2x0, vget_high_s8(vb0), vget_high_s8(va2));
      vacc0x0 = vpadalq_s16(vacc0x0, vprod0x0);
      vacc1x0 = vpadalq_s16(vacc1x0, vprod1x0);
      vacc2x0 = vpadalq_s16(vacc2x0, vprod2x0);
      int16x8_t vprod0x1 = vmull_s8(vget_low_s8(vb1), vget_low_s8(va0));
      int16x8_t vprod1x1 = vmull_s8(vget_low_s8(vb1), vget_low_s8(va1));
      int16x8_t vprod2x1 = vmull_s8(vget_low_s8(vb1), vget_low_s8(va2));
      vprod0x1 = vmlal_s8(vprod0x1, vget_high_s8(vb1), vget_high_s8(va0));
      vprod1x1 = vmlal_s8(vprod1x1, vget_high_s8(vb1), vget_high_s8(va1));
      vprod2x1 = vmlal_s8(vprod2x1, vget_high_s8(vb1), vget_high_s8(va2));
      vacc0x1 = vpadalq_s16(vacc0x1, vprod0x1);
      vacc1x1 = vpadalq_s16(vacc1x1, vprod1x1);
      vacc2x1 = vpadalq_s16(vacc2x1, vprod2x1);
      int16x8_t vprod0x2 = vmull_s8(vget_low_s8(vb2), vget_low_s8(va0));
      int16x8_t vprod1x2 = vmull_s8(vget_low_s8(vb2), vget_low_s8(va1));
      int16x8_t vprod2x2 = vmull_s8(vget_low_s8(vb2), vget_low_s8(va2));
      vprod0x2 = vmlal_s8(vprod0x2, vget_high_s8(vb2), vget_high_s8(va0));
      vprod1x2 = vmlal_s8(vprod1x2, vget_high_s8(vb2), vget_high_s8(va1));
      vprod2x2 = vmlal_s8(vprod2x2, vget_high_s8(vb2), vget_high_s8(va2));
      vacc0x2 = vpadalq_s16(vacc0x2, vprod0x2);
      vacc1x2 = vpadalq_s16(vacc1x2, vprod1x2);
      vacc2x2 = vpadalq_s16(vacc2x2, vprod2x2);
      int16x8_t vprod0x3 = vmull_s8(vget_low_s8(vb3), vget_low_s8(va0));
      int16x8_t vprod1x3 = vmull_s8(vget_low_s8(vb3), vget_low_s8(va1));
      int16x8_t vprod2x3 = vmull_s8(vget_low_s8(vb3), vget_low_s8(va2));
      vprod0x3 = vmlal_s8(vprod0x3, vget_high_s8(vb3), vget_high_s8(va0));
      vprod1x3 = vmlal_s8(vprod1x3, vget_high_s8(vb3), vget_high_s8(va1));
      vprod2x3 = vmlal_s8(vprod2x3, vget_high_s8(vb3), vget_high_s8(va2));
      vacc0x3 = vpadalq_s16(vacc0x3, vprod0x3);
      vacc1x3 = vpadalq_s16(vacc1x3, vprod1x3);
      vacc2x3 = vpadalq_s16(vacc2x3, vprod2x3);
      int16x8_t vprod0x4 = vmull_s8(vget_low_s8(vb4), vget_low_s8(va0));
      int16x8_t vprod1x4 = vmull_s8(vget_low_s8(vb4), vget_low_s8(va1));
      int16x8_t vprod2x4 = vmull_s8(vget_low_s8(vb4), vget_low_s8(va2));
      vprod0x4 = vmlal_s8(vprod0x4, vget_high_s8(vb4), vget_high_s8(va0));
      vprod1x4 = vmlal_s8(vprod1x4, vget_high_s8(vb4), vget_high_s8(va1));
      vprod2x4 = vmlal_s8(vprod2x4, vget_high_s8(vb4), vget_high_s8(va2));
      vacc0x4 = vpadalq_s16(vacc0x4, vprod0x4);
      vacc1x4 = vpadalq_s16(vacc1x4, vprod1x4);
      vacc2x4 = vpadalq_s16(vacc2x4, vprod2x4);
      int16x8_t vprod0x5 = vmull_s8(vget_low_s8(vb5), vget_low_s8(va0));
      int16x8_t vprod1x5 = vmull_s8(vget_low_s8(vb5), vget_low_s8(va1));
      int16x8_t vprod2x5 = vmull_s8(vget_low_s8(vb5), vget_low_s8(va2));
      vprod0x5 = vmlal_s8(vprod0x5, vget_high_s8(vb5), vget_high_s8(va0));
      vprod1x5 = vmlal_s8(vprod1x5, vget_high_s8(vb5), vget_high_s8(va1));
      vprod2x5 = vmlal_s8(vprod2x5, vget_high_s8(vb5), vget_high_s8(va2));
      vacc0x5 = vpadalq_s16(vacc0x5, vprod0x5);
      vacc1x5 = vpadalq_s16(vacc1x5, vprod1x5);
      vacc2x5 = vpadalq_s16(vacc2x5, vprod2x5);
      int16x8_t vprod0x6 = vmull_s8(vget_low_s8(vb6), vget_low_s8(va0));
      int16x8_t vprod1x6 = vmull_s8(vget_low_s8(vb6), vget_low_s8(va1));
      int16x8_t vprod2x6 = vmull_s8(vget_low_s8(vb6), vget_low_s8(va2));
      vprod0x6 = vmlal_s8(vprod0x6, vget_high_s8(vb6), vget_high_s8(va0));
      vprod1x6 = vmlal_s8(vprod1x6, vget_high_s8(vb6), vget_high_s8(va1));
      vprod2x6 = vmlal_s8(vprod2x6, vget_high_s8(vb6), vget_high_s8(va2));
      vacc0x6 = vpadalq_s16(vacc0x6, vprod0x6);
      vacc1x6 = vpadalq_s16(vacc1x6, vprod1x6);
      vacc2x6 = vpadalq_s16(vacc2x6, vprod2x6);
      int16x8_t vprod0x7 = vmull_s8(vget_low_s8(vb7), vget_low_s8(va0));
      int16x8_t vprod1x7 = vmull_s8(vget_low_s8(vb7), vget_low_s8(va1));
      int16x8_t vprod2x7 = vmull_s8(vget_low_s8(vb7), vget_low_s8(va2));
      vprod0x7 = vmlal_s8(vprod0x7, vget_high_s8(vb7), vget_high_s8(va0));
      vprod1x7 = vmlal_s8(vprod1x7, vget_high_s8(vb7), vget_high_s8(va1));
      vprod2x7 = vmlal_s8(vprod2x7, vget_high_s8(vb7), vget_high_s8(va2));
      vacc0x7 = vpadalq_s16(vacc0x7, vprod0x7);
      vacc1x7 = vpadalq_s16(vacc1x7, vprod1x7);
      vacc2x7 = vpadalq_s16(vacc2x7, vprod2x7);

      k -= 16 * sizeof(int8_t);
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
    int32x4_t vacc0x0123 = vpaddq_s32(vsum0x01, vsum0x23);
    int32x4_t vacc0x4567 = vpaddq_s32(vsum0x45, vsum0x67);
    int32x4_t vacc1x0123 = vpaddq_s32(vsum1x01, vsum1x23);
    int32x4_t vacc1x4567 = vpaddq_s32(vsum1x45, vsum1x67);
    int32x4_t vacc2x0123 = vpaddq_s32(vsum2x01, vsum2x23);
    int32x4_t vacc2x4567 = vpaddq_s32(vsum2x45, vsum2x67);
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
#endif

    const int32x4_t vright_pre_shift = vld1q_dup_s32(&params->rndnu_neon.right_pre_shift);
    const int32x4_t vmultiplier = vld1q_dup_s32(&params->rndnu_neon.multiplier);
    const int32x4_t vright_post_shift = vld1q_dup_s32(&params->rndnu_neon.right_post_shift);

    vacc0x0123 = vqshlq_s32(vacc0x0123, vright_pre_shift);
    vacc0x4567 = vqshlq_s32(vacc0x4567, vright_pre_shift);
    vacc1x0123 = vqshlq_s32(vacc1x0123, vright_pre_shift);
    vacc1x4567 = vqshlq_s32(vacc1x4567, vright_pre_shift);
    vacc2x0123 = vqshlq_s32(vacc2x0123, vright_pre_shift);
    vacc2x4567 = vqshlq_s32(vacc2x4567, vright_pre_shift);

    vacc0x0123 = vqdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqdmulhq_s32(vacc0x4567, vmultiplier);
    vacc1x0123 = vqdmulhq_s32(vacc1x0123, vmultiplier);
    vacc1x4567 = vqdmulhq_s32(vacc1x4567, vmultiplier);
    vacc2x0123 = vqdmulhq_s32(vacc2x0123, vmultiplier);
    vacc2x4567 = vqdmulhq_s32(vacc2x4567, vmultiplier);

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_post_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_post_shift);
    vacc1x0123 = vrshlq_s32(vacc1x0123, vright_post_shift);
    vacc1x4567 = vrshlq_s32(vacc1x4567, vright_post_shift);
    vacc2x0123 = vrshlq_s32(vacc2x0123, vright_post_shift);
    vacc2x4567 = vrshlq_s32(vacc2x4567, vright_post_shift);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->rndnu_neon.output_zero_point);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
    int8x16_t vout0x01234567_1x01234567 = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc1x01234567);
    int8x8_t vout2x01234567 = vqmovn_s16(vacc2x01234567);
#else
    const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);

    int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc1x01234567));
    int8x8_t vout2x01234567 = vqmovn_s16(vacc2x01234567);
#endif
    const int8x16_t voutput_min = vld1q_dup_s8(&params->rndnu_neon.output_min);
    const int8x16_t voutput_max = vld1q_dup_s8(&params->rndnu_neon.output_max);

    vout0x01234567_1x01234567 = vmaxq_s8(vout0x01234567_1x01234567, voutput_min);
    vout2x01234567 = vmax_s8(vout2x01234567, vget_low_s8(voutput_min));

    vout0x01234567_1x01234567 = vminq_s8(vout0x01234567_1x01234567, voutput_max);
    vout2x01234567 = vmin_s8(vout2x01234567, vget_low_s8(voutput_max));

    if (nc >= 8) {
      vst1_s8(c0 + 0, vget_low_s8(vout0x01234567_1x01234567));
      vst1_s8(c1 + 0, vget_high_s8(vout0x01234567_1x01234567));
      vst1_s8(c2 + 0, vout2x01234567);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);

      nc -= 8;
    } else {
      // Final case where not all of the 8 columns fit in the destination.
      if (nc & 4) {
        vst1q_lane_u32((void*) c0, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 0); c0 += 4;
        vst1q_lane_u32((void*) c1, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1_lane_u32((void*) c2, vreinterpret_u32_s8(vout2x01234567), 0); c2 += 4;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
        vout2x01234567 = vext_s8(vout2x01234567, vout2x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16((void*) c0, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 0); c0 += 2;
        vst1q_lane_u16((void*) c1, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1_lane_u16((void*) c2, vreinterpret_u16_s8(vout2x01234567), 0); c2 += 2;
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
