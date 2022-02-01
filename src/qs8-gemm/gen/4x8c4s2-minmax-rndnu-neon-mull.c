// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c4-neon-mull-shuffle.c.in
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


void xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4s2__neon_mull(
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

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  do {
    int32x4_t vacc0x01 = vreinterpretq_s32_u64(vmovl_u32(vld1_u32(w))); w = (const int32_t*) w + 2;
    int32x4_t vacc0x23 = vreinterpretq_s32_u64(vmovl_u32(vld1_u32(w))); w = (const int32_t*) w + 2;
    int32x4_t vacc0x45 = vreinterpretq_s32_u64(vmovl_u32(vld1_u32(w))); w = (const int32_t*) w + 2;
    int32x4_t vacc0x67 = vreinterpretq_s32_u64(vmovl_u32(vld1_u32(w))); w = (const int32_t*) w + 2;
    int32x4_t vacc1x01 = vacc0x01;
    int32x4_t vacc1x23 = vacc0x23;
    int32x4_t vacc1x45 = vacc0x45;
    int32x4_t vacc1x67 = vacc0x67;
    int32x4_t vacc2x01 = vacc0x01;
    int32x4_t vacc2x23 = vacc0x23;
    int32x4_t vacc2x45 = vacc0x45;
    int32x4_t vacc2x67 = vacc0x67;
    int32x4_t vacc3x01 = vacc0x01;
    int32x4_t vacc3x23 = vacc0x23;
    int32x4_t vacc3x45 = vacc0x45;
    int32x4_t vacc3x67 = vacc0x67;

    size_t k = kc;
    do {
      int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
      int8x8_t va1x0 = vld1_s8(a1); a1 += 8;
      int8x8_t va2x0 = vld1_s8(a2); a2 += 8;
      int8x8_t va3x0 = vld1_s8(a3); a3 += 8;

      const int8x8_t vb01c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb23c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb45c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb67c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb01c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb23c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb45c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb67c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;

      int16x8_t vprod0x01c0 = vmull_s8(vb01c0x0, va0x0);
      int16x8_t vprod1x01c0 = vmull_s8(vb01c0x0, va1x0);
      int16x8_t vprod2x01c0 = vmull_s8(vb01c0x0, va2x0);
      int16x8_t vprod3x01c0 = vmull_s8(vb01c0x0, va3x0);
      vacc0x01 = vpadalq_s16(vacc0x01, vprod0x01c0);
      vacc1x01 = vpadalq_s16(vacc1x01, vprod1x01c0);
      vacc2x01 = vpadalq_s16(vacc2x01, vprod2x01c0);
      vacc3x01 = vpadalq_s16(vacc3x01, vprod3x01c0);
      int16x8_t vprod0x23c0 = vmull_s8(vb23c0x0, va0x0);
      int16x8_t vprod1x23c0 = vmull_s8(vb23c0x0, va1x0);
      int16x8_t vprod2x23c0 = vmull_s8(vb23c0x0, va2x0);
      int16x8_t vprod3x23c0 = vmull_s8(vb23c0x0, va3x0);
      vacc0x23 = vpadalq_s16(vacc0x23, vprod0x23c0);
      vacc1x23 = vpadalq_s16(vacc1x23, vprod1x23c0);
      vacc2x23 = vpadalq_s16(vacc2x23, vprod2x23c0);
      vacc3x23 = vpadalq_s16(vacc3x23, vprod3x23c0);
      int16x8_t vprod0x45c0 = vmull_s8(vb45c0x0, va0x0);
      int16x8_t vprod1x45c0 = vmull_s8(vb45c0x0, va1x0);
      int16x8_t vprod2x45c0 = vmull_s8(vb45c0x0, va2x0);
      int16x8_t vprod3x45c0 = vmull_s8(vb45c0x0, va3x0);
      vacc0x45 = vpadalq_s16(vacc0x45, vprod0x45c0);
      vacc1x45 = vpadalq_s16(vacc1x45, vprod1x45c0);
      vacc2x45 = vpadalq_s16(vacc2x45, vprod2x45c0);
      vacc3x45 = vpadalq_s16(vacc3x45, vprod3x45c0);
      int16x8_t vprod0x67c0 = vmull_s8(vb67c0x0, va0x0);
      int16x8_t vprod1x67c0 = vmull_s8(vb67c0x0, va1x0);
      int16x8_t vprod2x67c0 = vmull_s8(vb67c0x0, va2x0);
      int16x8_t vprod3x67c0 = vmull_s8(vb67c0x0, va3x0);
      vacc0x67 = vpadalq_s16(vacc0x67, vprod0x67c0);
      vacc1x67 = vpadalq_s16(vacc1x67, vprod1x67c0);
      vacc2x67 = vpadalq_s16(vacc2x67, vprod2x67c0);
      vacc3x67 = vpadalq_s16(vacc3x67, vprod3x67c0);
      va0x0 = vext_s8(va0x0, va0x0, 4);
      va1x0 = vext_s8(va1x0, va1x0, 4);
      va2x0 = vext_s8(va2x0, va2x0, 4);
      va3x0 = vext_s8(va3x0, va3x0, 4);
      int16x8_t vprod0x01c1 = vmull_s8(vb01c1x0, va0x0);
      int16x8_t vprod1x01c1 = vmull_s8(vb01c1x0, va1x0);
      int16x8_t vprod2x01c1 = vmull_s8(vb01c1x0, va2x0);
      int16x8_t vprod3x01c1 = vmull_s8(vb01c1x0, va3x0);
      vacc0x01 = vpadalq_s16(vacc0x01, vprod0x01c1);
      vacc1x01 = vpadalq_s16(vacc1x01, vprod1x01c1);
      vacc2x01 = vpadalq_s16(vacc2x01, vprod2x01c1);
      vacc3x01 = vpadalq_s16(vacc3x01, vprod3x01c1);
      int16x8_t vprod0x23c1 = vmull_s8(vb23c1x0, va0x0);
      int16x8_t vprod1x23c1 = vmull_s8(vb23c1x0, va1x0);
      int16x8_t vprod2x23c1 = vmull_s8(vb23c1x0, va2x0);
      int16x8_t vprod3x23c1 = vmull_s8(vb23c1x0, va3x0);
      vacc0x23 = vpadalq_s16(vacc0x23, vprod0x23c1);
      vacc1x23 = vpadalq_s16(vacc1x23, vprod1x23c1);
      vacc2x23 = vpadalq_s16(vacc2x23, vprod2x23c1);
      vacc3x23 = vpadalq_s16(vacc3x23, vprod3x23c1);
      int16x8_t vprod0x45c1 = vmull_s8(vb45c1x0, va0x0);
      int16x8_t vprod1x45c1 = vmull_s8(vb45c1x0, va1x0);
      int16x8_t vprod2x45c1 = vmull_s8(vb45c1x0, va2x0);
      int16x8_t vprod3x45c1 = vmull_s8(vb45c1x0, va3x0);
      vacc0x45 = vpadalq_s16(vacc0x45, vprod0x45c1);
      vacc1x45 = vpadalq_s16(vacc1x45, vprod1x45c1);
      vacc2x45 = vpadalq_s16(vacc2x45, vprod2x45c1);
      vacc3x45 = vpadalq_s16(vacc3x45, vprod3x45c1);
      int16x8_t vprod0x67c1 = vmull_s8(vb67c1x0, va0x0);
      int16x8_t vprod1x67c1 = vmull_s8(vb67c1x0, va1x0);
      int16x8_t vprod2x67c1 = vmull_s8(vb67c1x0, va2x0);
      int16x8_t vprod3x67c1 = vmull_s8(vb67c1x0, va3x0);
      vacc0x67 = vpadalq_s16(vacc0x67, vprod0x67c1);
      vacc1x67 = vpadalq_s16(vacc1x67, vprod1x67c1);
      vacc2x67 = vpadalq_s16(vacc2x67, vprod2x67c1);
      vacc3x67 = vpadalq_s16(vacc3x67, vprod3x67c1);

      k -= 8 * sizeof(int8_t);
    } while (k != 0);

#if XNN_ARCH_ARM64
    int32x4_t vacc0x0123 = vpaddq_s32(vacc0x01, vacc0x23);
    int32x4_t vacc0x4567 = vpaddq_s32(vacc0x45, vacc0x67);
    int32x4_t vacc1x0123 = vpaddq_s32(vacc1x01, vacc1x23);
    int32x4_t vacc1x4567 = vpaddq_s32(vacc1x45, vacc1x67);
    int32x4_t vacc2x0123 = vpaddq_s32(vacc2x01, vacc2x23);
    int32x4_t vacc2x4567 = vpaddq_s32(vacc2x45, vacc2x67);
    int32x4_t vacc3x0123 = vpaddq_s32(vacc3x01, vacc3x23);
    int32x4_t vacc3x4567 = vpaddq_s32(vacc3x45, vacc3x67);
#else
    const int32x2_t vsum0x01 = vpadd_s32(vget_low_s32(vacc0x01), vget_high_s32(vacc0x01));
    const int32x2_t vsum0x23 = vpadd_s32(vget_low_s32(vacc0x23), vget_high_s32(vacc0x23));
    int32x4_t vacc0x0123 = vcombine_s32(vsum0x01, vsum0x23);
    const int32x2_t vsum0x45 = vpadd_s32(vget_low_s32(vacc0x45), vget_high_s32(vacc0x45));
    const int32x2_t vsum0x67 = vpadd_s32(vget_low_s32(vacc0x67), vget_high_s32(vacc0x67));
    int32x4_t vacc0x4567 = vcombine_s32(vsum0x45, vsum0x67);
    const int32x2_t vsum1x01 = vpadd_s32(vget_low_s32(vacc1x01), vget_high_s32(vacc1x01));
    const int32x2_t vsum1x23 = vpadd_s32(vget_low_s32(vacc1x23), vget_high_s32(vacc1x23));
    int32x4_t vacc1x0123 = vcombine_s32(vsum1x01, vsum1x23);
    const int32x2_t vsum1x45 = vpadd_s32(vget_low_s32(vacc1x45), vget_high_s32(vacc1x45));
    const int32x2_t vsum1x67 = vpadd_s32(vget_low_s32(vacc1x67), vget_high_s32(vacc1x67));
    int32x4_t vacc1x4567 = vcombine_s32(vsum1x45, vsum1x67);
    const int32x2_t vsum2x01 = vpadd_s32(vget_low_s32(vacc2x01), vget_high_s32(vacc2x01));
    const int32x2_t vsum2x23 = vpadd_s32(vget_low_s32(vacc2x23), vget_high_s32(vacc2x23));
    int32x4_t vacc2x0123 = vcombine_s32(vsum2x01, vsum2x23);
    const int32x2_t vsum2x45 = vpadd_s32(vget_low_s32(vacc2x45), vget_high_s32(vacc2x45));
    const int32x2_t vsum2x67 = vpadd_s32(vget_low_s32(vacc2x67), vget_high_s32(vacc2x67));
    int32x4_t vacc2x4567 = vcombine_s32(vsum2x45, vsum2x67);
    const int32x2_t vsum3x01 = vpadd_s32(vget_low_s32(vacc3x01), vget_high_s32(vacc3x01));
    const int32x2_t vsum3x23 = vpadd_s32(vget_low_s32(vacc3x23), vget_high_s32(vacc3x23));
    int32x4_t vacc3x0123 = vcombine_s32(vsum3x01, vsum3x23);
    const int32x2_t vsum3x45 = vpadd_s32(vget_low_s32(vacc3x45), vget_high_s32(vacc3x45));
    const int32x2_t vsum3x67 = vpadd_s32(vget_low_s32(vacc3x67), vget_high_s32(vacc3x67));
    int32x4_t vacc3x4567 = vcombine_s32(vsum3x45, vsum3x67);
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
    vacc3x0123 = vqshlq_s32(vacc3x0123, vright_pre_shift);
    vacc3x4567 = vqshlq_s32(vacc3x4567, vright_pre_shift);

    vacc0x0123 = vqdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqdmulhq_s32(vacc0x4567, vmultiplier);
    vacc1x0123 = vqdmulhq_s32(vacc1x0123, vmultiplier);
    vacc1x4567 = vqdmulhq_s32(vacc1x4567, vmultiplier);
    vacc2x0123 = vqdmulhq_s32(vacc2x0123, vmultiplier);
    vacc2x4567 = vqdmulhq_s32(vacc2x4567, vmultiplier);
    vacc3x0123 = vqdmulhq_s32(vacc3x0123, vmultiplier);
    vacc3x4567 = vqdmulhq_s32(vacc3x4567, vmultiplier);

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_post_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_post_shift);
    vacc1x0123 = vrshlq_s32(vacc1x0123, vright_post_shift);
    vacc1x4567 = vrshlq_s32(vacc1x4567, vright_post_shift);
    vacc2x0123 = vrshlq_s32(vacc2x0123, vright_post_shift);
    vacc2x4567 = vrshlq_s32(vacc2x4567, vright_post_shift);
    vacc3x0123 = vrshlq_s32(vacc3x0123, vright_post_shift);
    vacc3x4567 = vrshlq_s32(vacc3x4567, vright_post_shift);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->rndnu_neon.output_zero_point);
#if XNN_ARCH_ARM64
    int16x8_t vacc0x01234567 = vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567);
    int16x8_t vacc1x01234567 = vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567);
    int16x8_t vacc2x01234567 = vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567);
    int16x8_t vacc3x01234567 = vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567);

    vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);
    vacc1x01234567 = vqaddq_s16(vacc1x01234567, voutput_zero_point);
    vacc2x01234567 = vqaddq_s16(vacc2x01234567, voutput_zero_point);
    vacc3x01234567 = vqaddq_s16(vacc3x01234567, voutput_zero_point);

    int8x16_t vout0x01234567_1x01234567 = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc1x01234567);
    int8x16_t vout2x01234567_3x01234567 = vqmovn_high_s16(vqmovn_s16(vacc2x01234567), vacc3x01234567);
#else
    int16x8_t vacc0x01234567 = vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567));
    int16x8_t vacc1x01234567 = vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567));
    int16x8_t vacc2x01234567 = vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567));
    int16x8_t vacc3x01234567 = vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567));

    vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);
    vacc1x01234567 = vqaddq_s16(vacc1x01234567, voutput_zero_point);
    vacc2x01234567 = vqaddq_s16(vacc2x01234567, voutput_zero_point);
    vacc3x01234567 = vqaddq_s16(vacc3x01234567, voutput_zero_point);

    int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc1x01234567));
    int8x16_t vout2x01234567_3x01234567 = vcombine_s8(vqmovn_s16(vacc2x01234567), vqmovn_s16(vacc3x01234567));
#endif

    const int8x16_t voutput_min = vld1q_dup_s8(&params->rndnu_neon.output_min);
    vout0x01234567_1x01234567 = vmaxq_s8(vout0x01234567_1x01234567, voutput_min);
    vout2x01234567_3x01234567 = vmaxq_s8(vout2x01234567_3x01234567, voutput_min);

    const int8x16_t voutput_max = vld1q_dup_s8(&params->rndnu_neon.output_max);
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
      // Final case where not all of the 8 columns fit in the destination.
      if (nc & 4) {
        vst1q_lane_u32((void*) c0, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 0); c0 += 4;
        vst1q_lane_u32((void*) c1, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1q_lane_u32((void*) c2, vreinterpretq_u32_s8(vout2x01234567_3x01234567), 0); c2 += 4;
        vst1q_lane_u32((void*) c3, vreinterpretq_u32_s8(vout2x01234567_3x01234567), 2); c3 += 4;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16((void*) c0, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 0); c0 += 2;
        vst1q_lane_u16((void*) c1, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1q_lane_u16((void*) c2, vreinterpretq_u16_s8(vout2x01234567_3x01234567), 0); c2 += 2;
        vst1q_lane_u16((void*) c3, vreinterpretq_u16_s8(vout2x01234567_3x01234567), 4); c3 += 2;
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
