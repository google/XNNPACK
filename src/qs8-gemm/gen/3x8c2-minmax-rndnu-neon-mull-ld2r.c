// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c2-neon-mull-dup.c.in
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

void xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld2r(
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

  kc = round_up_po2(kc, 2 * sizeof(int8_t));
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
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;
    int32x4_t vacc2x0123 = vacc0x0123;
    int32x4_t vacc2x4567 = vacc0x4567;

    size_t k = kc;


    while (k >= 8 * sizeof(int8_t)) {
      const int16x4x2_t va00 = vld2_dup_s16((const void*)a0);
      const int16x4x2_t va01 = vld2_dup_s16((const void*)(a0 + 4)); a0 += 8;
      const int16x4x2_t va10 = vld2_dup_s16((const void*)a1);
      const int16x4x2_t va11 = vld2_dup_s16((const void*)(a1 + 4)); a1 += 8;
      const int16x4x2_t va20 = vld2_dup_s16((const void*)a2);
      const int16x4x2_t va21 = vld2_dup_s16((const void*)(a2 + 4)); a2 += 8;

      const int8x8_t vb0123c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int8x8_t va0c0 = vreinterpret_s8_s16(va00.val[0]);
      const int8x8_t va1c0 = vreinterpret_s8_s16(va10.val[0]);
      const int8x8_t va2c0 = vreinterpret_s8_s16(va20.val[0]);

      const int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0, va0c0);
      const int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0, va1c0);
      const int16x8_t vprod2x0123c0 = vmull_s8(vb0123c0, va2c0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
      vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c0);
      const int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0, va0c0);
      const int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0, va1c0);
      const int16x8_t vprod2x4567c0 = vmull_s8(vb4567c0, va2c0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
      vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c0);
      const int8x8_t va0c1 = vreinterpret_s8_s16(va00.val[1]);
      const int8x8_t va1c1 = vreinterpret_s8_s16(va10.val[1]);
      const int8x8_t va2c1 = vreinterpret_s8_s16(va20.val[1]);

      const int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1, va0c1);
      const int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1, va1c1);
      const int16x8_t vprod2x0123c1 = vmull_s8(vb0123c1, va2c1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
      vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c1);
      const int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1, va0c1);
      const int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1, va1c1);
      const int16x8_t vprod2x4567c1 = vmull_s8(vb4567c1, va2c1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
      vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c1);
      const int8x8_t va0c2 = vreinterpret_s8_s16(va01.val[0]);
      const int8x8_t va1c2 = vreinterpret_s8_s16(va11.val[0]);
      const int8x8_t va2c2 = vreinterpret_s8_s16(va21.val[0]);

      const int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2, va0c2);
      const int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2, va1c2);
      const int16x8_t vprod2x0123c2 = vmull_s8(vb0123c2, va2c2);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
      vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c2);
      const int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2, va0c2);
      const int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2, va1c2);
      const int16x8_t vprod2x4567c2 = vmull_s8(vb4567c2, va2c2);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
      vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c2);
      const int8x8_t va0c3 = vreinterpret_s8_s16(va01.val[1]);
      const int8x8_t va1c3 = vreinterpret_s8_s16(va11.val[1]);
      const int8x8_t va2c3 = vreinterpret_s8_s16(va21.val[1]);

      const int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3, va0c3);
      const int16x8_t vprod1x0123c3 = vmull_s8(vb0123c3, va1c3);
      const int16x8_t vprod2x0123c3 = vmull_s8(vb0123c3, va2c3);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c3);
      vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c3);
      const int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3, va0c3);
      const int16x8_t vprod1x4567c3 = vmull_s8(vb4567c3, va1c3);
      const int16x8_t vprod2x4567c3 = vmull_s8(vb4567c3, va2c3);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c3);
      vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c3);

      k -= 8 * sizeof(int8_t);
    }

    if XNN_UNLIKELY(k != 0) {
      const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);
      const int8x8_t va1 = vld1_s8(a1); a1 = (const int8_t*) ((uintptr_t) a1 + k);
      const int8x8_t va2 = vld1_s8(a2); a2 = (const int8_t*) ((uintptr_t) a2 + k);

      const int8x8_t vb0123c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int8x8_t va0c0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0));
      const int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0, va0c0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      const int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0, va0c0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      const int8x8_t va1c0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 0));
      const int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0, va1c0);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
      const int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0, va1c0);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
      const int8x8_t va2c0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 0));
      const int16x8_t vprod2x0123c0 = vmull_s8(vb0123c0, va2c0);
      vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c0);
      const int16x8_t vprod2x4567c0 = vmull_s8(vb4567c0, va2c0);
      vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c0);

      if (k > 2 * sizeof(int8_t)) {
        const int8x8_t vb0123c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb4567c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int8x8_t va0c1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1));
        const int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1, va0c1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
        const int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1, va0c1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
        const int8x8_t va1c1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 1));
        const int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1, va1c1);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
        const int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1, va1c1);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
        const int8x8_t va2c1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 1));
        const int16x8_t vprod2x0123c1 = vmull_s8(vb0123c1, va2c1);
        vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c1);
        const int16x8_t vprod2x4567c1 = vmull_s8(vb4567c1, va2c1);
        vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c1);

        if (k > 4 * sizeof(int8_t)) {
          const int8x8_t vb0123c2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
          const int8x8_t vb4567c2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

          const int8x8_t va0c2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2));
          const int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2, va0c2);
          vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
          const int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2, va0c2);
          vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
          const int8x8_t va1c2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 2));
          const int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2, va1c2);
          vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
          const int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2, va1c2);
          vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
          const int8x8_t va2c2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 2));
          const int16x8_t vprod2x0123c2 = vmull_s8(vb0123c2, va2c2);
          vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c2);
          const int16x8_t vprod2x4567c2 = vmull_s8(vb4567c2, va2c2);
          vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c2);
        }
      }
    }

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
    int16x8_t vacc0x01234567 = vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567);
    int16x8_t vacc1x01234567 = vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567);
    int16x8_t vacc2x01234567 = vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567);

    vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);
    vacc1x01234567 = vqaddq_s16(vacc1x01234567, voutput_zero_point);
    vacc2x01234567 = vqaddq_s16(vacc2x01234567, voutput_zero_point);

    int8x16_t vout0x01234567_1x01234567 = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc1x01234567);
    int8x8_t vout2x01234567 = vqmovn_s16(vacc2x01234567);
#else
    int16x8_t vacc0x01234567 = vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567));
    int16x8_t vacc1x01234567 = vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567));
    int16x8_t vacc2x01234567 = vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567));

    vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);
    vacc1x01234567 = vqaddq_s16(vacc1x01234567, voutput_zero_point);
    vacc2x01234567 = vqaddq_s16(vacc2x01234567, voutput_zero_point);

    int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc1x01234567));
    int8x8_t vout2x01234567 = vqmovn_s16(vacc2x01234567);
#endif

    const int8x16_t voutput_min = vld1q_dup_s8(&params->rndnu_neon.output_min);
    vout0x01234567_1x01234567 = vmaxq_s8(vout0x01234567_1x01234567, voutput_min);
    vout2x01234567 = vmax_s8(vout2x01234567, vget_low_s8(voutput_min));

    const int8x16_t voutput_max = vld1q_dup_s8(&params->rndnu_neon.output_max);
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
