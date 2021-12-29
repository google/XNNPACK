// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c4-neon-mull-dup.c.in
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


void xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld1r(
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
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;

  do {
    int32x4_t vacc0x01 = vreinterpretq_s32_u64(vmovl_u32(vld1_u32(w))); w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t));
    int32x4_t vacc0x23 = vreinterpretq_s32_u64(vmovl_u32(vld1_u32(w))); w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t));
    int32x4_t vacc0x45 = vreinterpretq_s32_u64(vmovl_u32(vld1_u32(w))); w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t));
    int32x4_t vacc0x67 = vreinterpretq_s32_u64(vmovl_u32(vld1_u32(w))); w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t));
    int32x4_t vacc0x89 = vreinterpretq_s32_u64(vmovl_u32(vld1_u32(w))); w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t));
    int32x4_t vacc0xAB = vreinterpretq_s32_u64(vmovl_u32(vld1_u32(w))); w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t));
    int32x4_t vacc0xCD = vreinterpretq_s32_u64(vmovl_u32(vld1_u32(w))); w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t));
    int32x4_t vacc0xEF = vreinterpretq_s32_u64(vmovl_u32(vld1_u32(w))); w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t));

    size_t k = kc;


    while (k >= 8 * sizeof(int8_t)) {
      const int32x2_t va00 = vld1_dup_s32((const void*)a0);
      const int32x2_t va01 = vld1_dup_s32((const void*)(a0 + 4)); a0 += 8;

      const int8x8_t vb01c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb23c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb45c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb67c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb89c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vbABc0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vbCDc0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vbEFc0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb01c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb23c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb45c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb67c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb89c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vbABc1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vbCDc1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vbEFc1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int8x8_t va0c0 = vreinterpret_s8_s32(va00);

      const int16x8_t vprod0x01c0 = vmull_s8(vb01c0, va0c0);
      vacc0x01 = vpadalq_s16(vacc0x01, vprod0x01c0);
      const int16x8_t vprod0x23c0 = vmull_s8(vb23c0, va0c0);
      vacc0x23 = vpadalq_s16(vacc0x23, vprod0x23c0);
      const int16x8_t vprod0x45c0 = vmull_s8(vb45c0, va0c0);
      vacc0x45 = vpadalq_s16(vacc0x45, vprod0x45c0);
      const int16x8_t vprod0x67c0 = vmull_s8(vb67c0, va0c0);
      vacc0x67 = vpadalq_s16(vacc0x67, vprod0x67c0);
      const int16x8_t vprod0x89c0 = vmull_s8(vb89c0, va0c0);
      vacc0x89 = vpadalq_s16(vacc0x89, vprod0x89c0);
      const int16x8_t vprod0xABc0 = vmull_s8(vbABc0, va0c0);
      vacc0xAB = vpadalq_s16(vacc0xAB, vprod0xABc0);
      const int16x8_t vprod0xCDc0 = vmull_s8(vbCDc0, va0c0);
      vacc0xCD = vpadalq_s16(vacc0xCD, vprod0xCDc0);
      const int16x8_t vprod0xEFc0 = vmull_s8(vbEFc0, va0c0);
      vacc0xEF = vpadalq_s16(vacc0xEF, vprod0xEFc0);
      const int8x8_t va0c1 = vreinterpret_s8_s32(va01);

      const int16x8_t vprod0x01c1 = vmull_s8(vb01c1, va0c1);
      vacc0x01 = vpadalq_s16(vacc0x01, vprod0x01c1);
      const int16x8_t vprod0x23c1 = vmull_s8(vb23c1, va0c1);
      vacc0x23 = vpadalq_s16(vacc0x23, vprod0x23c1);
      const int16x8_t vprod0x45c1 = vmull_s8(vb45c1, va0c1);
      vacc0x45 = vpadalq_s16(vacc0x45, vprod0x45c1);
      const int16x8_t vprod0x67c1 = vmull_s8(vb67c1, va0c1);
      vacc0x67 = vpadalq_s16(vacc0x67, vprod0x67c1);
      const int16x8_t vprod0x89c1 = vmull_s8(vb89c1, va0c1);
      vacc0x89 = vpadalq_s16(vacc0x89, vprod0x89c1);
      const int16x8_t vprod0xABc1 = vmull_s8(vbABc1, va0c1);
      vacc0xAB = vpadalq_s16(vacc0xAB, vprod0xABc1);
      const int16x8_t vprod0xCDc1 = vmull_s8(vbCDc1, va0c1);
      vacc0xCD = vpadalq_s16(vacc0xCD, vprod0xCDc1);
      const int16x8_t vprod0xEFc1 = vmull_s8(vbEFc1, va0c1);
      vacc0xEF = vpadalq_s16(vacc0xEF, vprod0xEFc1);

      k -= 8 * sizeof(int8_t);
    }

    if XNN_UNLIKELY(k != 0) {
      const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);

      const int8x8_t vb01c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb23c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb45c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb67c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb89c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vbABc0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vbCDc0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vbEFc0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int8x8_t va0c0 = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(va0), 0));
      const int16x8_t vprod0x01c0 = vmull_s8(vb01c0, va0c0);
      vacc0x01 = vpadalq_s16(vacc0x01, vprod0x01c0);
      const int16x8_t vprod0x23c0 = vmull_s8(vb23c0, va0c0);
      vacc0x23 = vpadalq_s16(vacc0x23, vprod0x23c0);
      const int16x8_t vprod0x45c0 = vmull_s8(vb45c0, va0c0);
      vacc0x45 = vpadalq_s16(vacc0x45, vprod0x45c0);
      const int16x8_t vprod0x67c0 = vmull_s8(vb67c0, va0c0);
      vacc0x67 = vpadalq_s16(vacc0x67, vprod0x67c0);
      const int16x8_t vprod0x89c0 = vmull_s8(vb89c0, va0c0);
      vacc0x89 = vpadalq_s16(vacc0x89, vprod0x89c0);
      const int16x8_t vprod0xABc0 = vmull_s8(vbABc0, va0c0);
      vacc0xAB = vpadalq_s16(vacc0xAB, vprod0xABc0);
      const int16x8_t vprod0xCDc0 = vmull_s8(vbCDc0, va0c0);
      vacc0xCD = vpadalq_s16(vacc0xCD, vprod0xCDc0);
      const int16x8_t vprod0xEFc0 = vmull_s8(vbEFc0, va0c0);
      vacc0xEF = vpadalq_s16(vacc0xEF, vprod0xEFc0);
    }

#if XNN_ARCH_ARM64
    int32x4_t vacc0x0123 = vpaddq_s32(vacc0x01, vacc0x23);
    int32x4_t vacc0x4567 = vpaddq_s32(vacc0x45, vacc0x67);
    int32x4_t vacc0x89AB = vpaddq_s32(vacc0x89, vacc0xAB);
    int32x4_t vacc0xCDEF = vpaddq_s32(vacc0xCD, vacc0xEF);
#else
    const int32x2_t vsum0x01 = vpadd_s32(vget_low_s32(vacc0x01), vget_high_s32(vacc0x01));
    const int32x2_t vsum0x23 = vpadd_s32(vget_low_s32(vacc0x23), vget_high_s32(vacc0x23));
    int32x4_t vacc0x0123 = vcombine_s32(vsum0x01, vsum0x23);
    const int32x2_t vsum0x45 = vpadd_s32(vget_low_s32(vacc0x45), vget_high_s32(vacc0x45));
    const int32x2_t vsum0x67 = vpadd_s32(vget_low_s32(vacc0x67), vget_high_s32(vacc0x67));
    int32x4_t vacc0x4567 = vcombine_s32(vsum0x45, vsum0x67);
    const int32x2_t vsum0x89 = vpadd_s32(vget_low_s32(vacc0x89), vget_high_s32(vacc0x89));
    const int32x2_t vsum0xAB = vpadd_s32(vget_low_s32(vacc0xAB), vget_high_s32(vacc0xAB));
    int32x4_t vacc0x89AB = vcombine_s32(vsum0x89, vsum0xAB);
    const int32x2_t vsum0xCD = vpadd_s32(vget_low_s32(vacc0xCD), vget_high_s32(vacc0xCD));
    const int32x2_t vsum0xEF = vpadd_s32(vget_low_s32(vacc0xEF), vget_high_s32(vacc0xEF));
    int32x4_t vacc0xCDEF = vcombine_s32(vsum0xCD, vsum0xEF);
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

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      // Final case where not all of the 16 columns fit in the destination.
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
