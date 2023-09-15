// Auto-generated file. Do not edit!
//   Template: src/qu8-gemm/c8-neoni8mm.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/gemm.h>
#include <xnnpack/math.h>


void xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c8__neoni8mm(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(uint8_t));
  const uint8_t* a0 = a;
  uint8_t* c0 = c;

  const uint8x16_t vkernel_zero_point = vreinterpretq_u8_u32(vld1q_dup_u32((const void*) params->rndnu_neon.kernel_zero_point));

  // Loop over groups of 8 columns.
  do {
    // Initialize accumulators with bias. 8 bias values are loaded from the
    // weight matrix, at the start of the group of 8 columns.
    #if XNN_ARCH_ARM64
      const uint64x2x2_t vbias01x0123 = vld2q_dup_u64(w); w = (const int32_t*) w + 4;
      const uint64x2x2_t vbias01x4567 = vld2q_dup_u64(w); w = (const int32_t*) w + 4;
    #else
      uint64x2x2_t vbias01x0123;
      vbias01x0123.val[0] = vld1q_dup_u64(w); w = (const int32_t*) w + 2;
      vbias01x0123.val[1] = vld1q_dup_u64(w); w = (const int32_t*) w + 2;
      uint64x2x2_t vbias01x4567;
      vbias01x4567.val[0] = vld1q_dup_u64(w); w = (const int32_t*) w + 2;
      vbias01x4567.val[1] = vld1q_dup_u64(w); w = (const int32_t*) w + 2;
    #endif
    uint32x4_t vpacc01x01 = vreinterpretq_u32_u64(vbias01x0123.val[0]);
    uint32x4_t vpacc01x23 = vreinterpretq_u32_u64(vbias01x0123.val[1]);
    uint32x4_t vpacc01x45 = vreinterpretq_u32_u64(vbias01x4567.val[0]);
    uint32x4_t vpacc01x67 = vreinterpretq_u32_u64(vbias01x4567.val[1]);
    uint32x4_t vnacc01 = vmovq_n_u32(0);

    uint64x2x2_t va01x0123456789ABCDEF;
    va01x0123456789ABCDEF.val[0] = vdupq_n_u64(0);
    va01x0123456789ABCDEF.val[1] = vdupq_n_u64(0);

    // Inner accumulation loop along the 8 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 16 * sizeof(uint8_t)) {
      // Load a 1x16 block of activations.
      #if XNN_ARCH_ARM64
        va01x0123456789ABCDEF = vld2q_lane_u64((const void*) a0, va01x0123456789ABCDEF, 0); a0 += 16;
      #else
        va01x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a0, va01x0123456789ABCDEF.val[0], 0); a0 += 8;
        va01x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a0, va01x0123456789ABCDEF.val[1], 0); a0 += 8;
      #endif

      // Load a 16x8 block of weights.
      const uint8x16_t vb01x01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;
      const uint8x16_t vb23x01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;
      const uint8x16_t vb45x01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;
      const uint8x16_t vb67x01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;
      const uint8x16_t vb01x89ABCDEF = vld1q_u8(w); w = (const uint8_t*) w + 16;
      const uint8x16_t vb23x89ABCDEF = vld1q_u8(w); w = (const uint8_t*) w + 16;
      const uint8x16_t vb45x89ABCDEF = vld1q_u8(w); w = (const uint8_t*) w + 16;
      const uint8x16_t vb67x89ABCDEF = vld1q_u8(w); w = (const uint8_t*) w + 16;

      // Multiply-accumulate: 1x16 * 16x8 --> 1x8.
      vnacc01 = vmmlaq_u32(vnacc01, vkernel_zero_point, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]));
      vpacc01x01 = vmmlaq_u32(vpacc01x01, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]), vb01x01234567);
      vpacc01x23 = vmmlaq_u32(vpacc01x23, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]), vb23x01234567);
      vpacc01x45 = vmmlaq_u32(vpacc01x45, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]), vb45x01234567);
      vpacc01x67 = vmmlaq_u32(vpacc01x67, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]), vb67x01234567);
      vnacc01 = vmmlaq_u32(vnacc01, vkernel_zero_point, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]));
      vpacc01x01 = vmmlaq_u32(vpacc01x01, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]), vb01x89ABCDEF);
      vpacc01x23 = vmmlaq_u32(vpacc01x23, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]), vb23x89ABCDEF);
      vpacc01x45 = vmmlaq_u32(vpacc01x45, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]), vb45x89ABCDEF);
      vpacc01x67 = vmmlaq_u32(vpacc01x67, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]), vb67x89ABCDEF);

      k -= 16 * sizeof(uint8_t);
    }
    // Handle up to 8 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 1x8 block of activations.
      uint64x2_t va01x01234567 = vld1q_dup_u64((const void*) a0); a0 += 8;

      // Load a 16x8 block of weights.
      const uint8x16_t vb01x01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;
      const uint8x16_t vb23x01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;
      const uint8x16_t vb45x01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;
      const uint8x16_t vb67x01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;

      // Multiply-accumulate: 1x8 * 8x8 --> 1x8.
      vnacc01 = vmmlaq_u32(vnacc01, vkernel_zero_point, vreinterpretq_u8_u64(va01x01234567));
      vpacc01x01 = vmmlaq_u32(vpacc01x01, vreinterpretq_u8_u64(va01x01234567), vb01x01234567);
      vpacc01x23 = vmmlaq_u32(vpacc01x23, vreinterpretq_u8_u64(va01x01234567), vb23x01234567);
      vpacc01x45 = vmmlaq_u32(vpacc01x45, vreinterpretq_u8_u64(va01x01234567), vb45x01234567);
      vpacc01x67 = vmmlaq_u32(vpacc01x67, vreinterpretq_u8_u64(va01x01234567), vb67x01234567);
    }

    // Subtract zero point from accumulators.
    #if XNN_ARCH_ARM64
      const uint32x4_t vnacc01x01 = vzip1q_u32(vnacc01, vnacc01);
    #else
      const uint32x4_t vnacc01x01 = vzipq_u32(vnacc01, vnacc01).val[0];
    #endif
    int32x4_t vacc01x01 = vreinterpretq_s32_u32(vsubq_u32(vpacc01x01, vnacc01x01));
    int32x4_t vacc01x23 = vreinterpretq_s32_u32(vsubq_u32(vpacc01x23, vnacc01x01));
    int32x4_t vacc01x45 = vreinterpretq_s32_u32(vsubq_u32(vpacc01x45, vnacc01x01));
    int32x4_t vacc01x67 = vreinterpretq_s32_u32(vsubq_u32(vpacc01x67, vnacc01x01));

    #if XNN_ARCH_ARM64
      int32x4_t vacc0x0123 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x01), vreinterpretq_u64_s32(vacc01x23)));
      int32x4_t vacc0x4567 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x45), vreinterpretq_u64_s32(vacc01x67)));
    #else
      int32x4_t vacc0x0123 = vcombine_s32(vget_low_s32(vacc01x01), vget_low_s32(vacc01x23));
      int32x4_t vacc0x4567 = vcombine_s32(vget_low_s32(vacc01x45), vget_low_s32(vacc01x67));
    #endif
    const int32x4_t vright_pre_shift = vld1q_dup_s32(&params->rndnu_neon.right_pre_shift);
    const int32x4_t vmultiplier = vld1q_dup_s32(&params->rndnu_neon.multiplier);
    const int32x4_t vright_post_shift = vld1q_dup_s32(&params->rndnu_neon.right_post_shift);

    vacc0x0123 = vshlq_s32(vacc0x0123, vright_pre_shift);
    vacc0x4567 = vshlq_s32(vacc0x4567, vright_pre_shift);

    vacc0x0123 = vqdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqdmulhq_s32(vacc0x4567, vmultiplier);

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_post_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_post_shift);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->rndnu_neon.output_zero_point);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);

    uint8x8_t vout0x01234567 = vqmovun_s16(vacc0x01234567);
#else
    const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);

    uint8x8_t vout0x01234567 = vqmovun_s16(vacc0x01234567);
#endif
    const uint8x8_t voutput_min = vld1_dup_u8(&params->rndnu_neon.output_min);
    const uint8x8_t voutput_max = vld1_dup_u8(&params->rndnu_neon.output_max);

    vout0x01234567 = vmax_u8(vout0x01234567, voutput_min);

    vout0x01234567 = vmin_u8(vout0x01234567, voutput_max);

    if (nc >= 8) {
      vst1_u8(c0, vout0x01234567);


      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_u8(vout0x01234567), 0); c0 += 4;
        vout0x01234567 = vext_u8(vout0x01234567, vout0x01234567, 4);
      }
      if (nc & 2) {
        vst1_lane_u16((void*) c0, vreinterpret_u16_u8(vout0x01234567), 0); c0 += 2;
        vout0x01234567 = vext_u8(vout0x01234567, vout0x01234567, 2);
      }
      if (nc & 1) {
        vst1_lane_u8(c0, vout0x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
