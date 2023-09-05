// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/c8-neon-mull.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/igemm.h>
#include <xnnpack/math.h>


void xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neon_mull(
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
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  int8_t* c0 = c;

  do {
    int32x4_t vacc0x0 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc0x1 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc0x2 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc0x3 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc0x4 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc0x5 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc0x6 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc0x7 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc0x8 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc0x9 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc0x10 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc0x11 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc0x12 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc0x13 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc0x14 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc0x15 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;

      // Handle 8 bytes at a time using MUL.
      while (k != 0) {
        const int8x8_t va0 = vld1_s8(a0); a0 += 8;

        const int8x8_t vb0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x0 = vmull_s8(vb0, va0);
        vacc0x0 = vpadalq_s16(vacc0x0, vprod0x0);
        const int8x8_t vb1 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x1 = vmull_s8(vb1, va0);
        vacc0x1 = vpadalq_s16(vacc0x1, vprod0x1);
        const int8x8_t vb2 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x2 = vmull_s8(vb2, va0);
        vacc0x2 = vpadalq_s16(vacc0x2, vprod0x2);
        const int8x8_t vb3 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x3 = vmull_s8(vb3, va0);
        vacc0x3 = vpadalq_s16(vacc0x3, vprod0x3);
        const int8x8_t vb4 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x4 = vmull_s8(vb4, va0);
        vacc0x4 = vpadalq_s16(vacc0x4, vprod0x4);
        const int8x8_t vb5 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x5 = vmull_s8(vb5, va0);
        vacc0x5 = vpadalq_s16(vacc0x5, vprod0x5);
        const int8x8_t vb6 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x6 = vmull_s8(vb6, va0);
        vacc0x6 = vpadalq_s16(vacc0x6, vprod0x6);
        const int8x8_t vb7 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x7 = vmull_s8(vb7, va0);
        vacc0x7 = vpadalq_s16(vacc0x7, vprod0x7);
        const int8x8_t vb8 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x8 = vmull_s8(vb8, va0);
        vacc0x8 = vpadalq_s16(vacc0x8, vprod0x8);
        const int8x8_t vb9 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x9 = vmull_s8(vb9, va0);
        vacc0x9 = vpadalq_s16(vacc0x9, vprod0x9);
        const int8x8_t vb10 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x10 = vmull_s8(vb10, va0);
        vacc0x10 = vpadalq_s16(vacc0x10, vprod0x10);
        const int8x8_t vb11 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x11 = vmull_s8(vb11, va0);
        vacc0x11 = vpadalq_s16(vacc0x11, vprod0x11);
        const int8x8_t vb12 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x12 = vmull_s8(vb12, va0);
        vacc0x12 = vpadalq_s16(vacc0x12, vprod0x12);
        const int8x8_t vb13 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x13 = vmull_s8(vb13, va0);
        vacc0x13 = vpadalq_s16(vacc0x13, vprod0x13);
        const int8x8_t vb14 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x14 = vmull_s8(vb14, va0);
        vacc0x14 = vpadalq_s16(vacc0x14, vprod0x14);
        const int8x8_t vb15 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x15 = vmull_s8(vb15, va0);
        vacc0x15 = vpadalq_s16(vacc0x15, vprod0x15);

        k -= 8 * sizeof(int8_t);
      }

      p -= 1 * sizeof(void*);
    } while (p != 0);

#if XNN_ARCH_ARM64
    const int32x4_t vsum0x01 = vpaddq_s32(vacc0x0, vacc0x1);
    const int32x4_t vsum0x23 = vpaddq_s32(vacc0x2, vacc0x3);
    const int32x4_t vsum0x45 = vpaddq_s32(vacc0x4, vacc0x5);
    const int32x4_t vsum0x67 = vpaddq_s32(vacc0x6, vacc0x7);
    const int32x4_t vsum0x89 = vpaddq_s32(vacc0x8, vacc0x9);
    const int32x4_t vsum0xAB = vpaddq_s32(vacc0x10, vacc0x11);
    const int32x4_t vsum0xCD = vpaddq_s32(vacc0x12, vacc0x13);
    const int32x4_t vsum0xEF = vpaddq_s32(vacc0x14, vacc0x15);

    int32x4_t vacc0x0123 = vpaddq_s32(vsum0x01, vsum0x23);
    int32x4_t vacc0x4567 = vpaddq_s32(vsum0x45, vsum0x67);
    int32x4_t vacc0x89AB = vpaddq_s32(vsum0x89, vsum0xAB);
    int32x4_t vacc0xCDEF = vpaddq_s32(vsum0xCD, vsum0xEF);
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
#endif

    const int32x4_t vright_pre_shift = vld1q_dup_s32(&params->rndnu_neon.right_pre_shift);
    const int32x4_t vmultiplier = vld1q_dup_s32(&params->rndnu_neon.multiplier);
    const int32x4_t vright_post_shift = vld1q_dup_s32(&params->rndnu_neon.right_post_shift);

    vacc0x0123 = vqshlq_s32(vacc0x0123, vright_pre_shift);
    vacc0x4567 = vqshlq_s32(vacc0x4567, vright_pre_shift);
    vacc0x89AB = vqshlq_s32(vacc0x89AB, vright_pre_shift);
    vacc0xCDEF = vqshlq_s32(vacc0xCDEF, vright_pre_shift);

    vacc0x0123 = vqdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqdmulhq_s32(vacc0x4567, vmultiplier);
    vacc0x89AB = vqdmulhq_s32(vacc0x89AB, vmultiplier);
    vacc0xCDEF = vqdmulhq_s32(vacc0xCDEF, vmultiplier);

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_post_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_post_shift);
    vacc0x89AB = vrshlq_s32(vacc0x89AB, vright_post_shift);
    vacc0xCDEF = vrshlq_s32(vacc0xCDEF, vright_post_shift);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->rndnu_neon.output_zero_point);
#if XNN_ARCH_ARM64
    int16x8_t vacc0x01234567 = vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567);
    int16x8_t vacc0x89ABCDEF = vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF);

    vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);
    vacc0x89ABCDEF = vqaddq_s16(vacc0x89ABCDEF, voutput_zero_point);

    int8x16_t vout0x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc0x89ABCDEF);
#else
    int16x8_t vacc0x01234567 = vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567));
    int16x8_t vacc0x89ABCDEF = vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF));

    vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);
    vacc0x89ABCDEF = vqaddq_s16(vacc0x89ABCDEF, voutput_zero_point);

    int8x16_t vout0x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc0x89ABCDEF));
#endif

    const int8x16_t voutput_min = vld1q_dup_s8(&params->rndnu_neon.output_min);
    vout0x0123456789ABCDEF = vmaxq_s8(vout0x0123456789ABCDEF, voutput_min);

    const int8x16_t voutput_max = vld1q_dup_s8(&params->rndnu_neon.output_max);
    vout0x0123456789ABCDEF = vminq_s8(vout0x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      vst1q_s8(c0 + 0, vout0x0123456789ABCDEF);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      int8x8_t vout0x01234567 = vget_low_s8(vout0x0123456789ABCDEF);
      if (nc & 8) {
        vst1_s8(c0, vout0x01234567); c0 += 8;
        vout0x01234567 = vget_high_s8(vout0x0123456789ABCDEF);
      }
      if (nc & 4) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_s8(vout0x01234567), 0); c0 += 4;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 4);
      }
      if (nc & 2) {
        vst1_lane_u16((void*) c0, vreinterpret_u16_s8(vout0x01234567), 0); c0 += 2;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 2);
      }
      if (nc & 1) {
        vst1_lane_s8(c0, vout0x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
