// Auto-generated file. Do not edit!
//   Template: src/qu8-igemm/c8-neoni8mm.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/igemm.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>


void xnn_qu8_igemm_minmax_fp32_ukernel_2x16c8__neoni8mm(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const uint8_t** restrict a,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (2 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(uint8_t));
  uint8_t* c0 = c;
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    c1 = c0;
  }

  const uint8x16_t vkernel_zero_point = vreinterpretq_u8_u32(vld1q_dup_u32((const void*) params->fp32_neonv8.kernel_zero_point));

  do {
    // Initialize accumulators with bias. 16 bias values are loaded from the
    // weight matrix, at the start of the group of 16 columns.
    #if XNN_ARCH_ARM64
      const uint64x2x2_t vbias01x0123 = vld2q_dup_u64(w); w = (const int32_t*) w + 4;
      const uint64x2x2_t vbias01x4567 = vld2q_dup_u64(w); w = (const int32_t*) w + 4;
      const uint64x2x2_t vbias01x89AB = vld2q_dup_u64(w); w = (const int32_t*) w + 4;
      const uint64x2x2_t vbias01xCDEF = vld2q_dup_u64(w); w = (const int32_t*) w + 4;
    #else
      uint64x2x2_t vbias01x0123;
      vbias01x0123.val[0] = vld1q_dup_u64(w); w = (const int32_t*) w + 2;
      vbias01x0123.val[1] = vld1q_dup_u64(w); w = (const int32_t*) w + 2;
      uint64x2x2_t vbias01x4567;
      vbias01x4567.val[0] = vld1q_dup_u64(w); w = (const int32_t*) w + 2;
      vbias01x4567.val[1] = vld1q_dup_u64(w); w = (const int32_t*) w + 2;
      uint64x2x2_t vbias01x89AB;
      vbias01x89AB.val[0] = vld1q_dup_u64(w); w = (const int32_t*) w + 2;
      vbias01x89AB.val[1] = vld1q_dup_u64(w); w = (const int32_t*) w + 2;
      uint64x2x2_t vbias01xCDEF;
      vbias01xCDEF.val[0] = vld1q_dup_u64(w); w = (const int32_t*) w + 2;
      vbias01xCDEF.val[1] = vld1q_dup_u64(w); w = (const int32_t*) w + 2;
    #endif
    uint32x4_t vpacc01x01 = vreinterpretq_u32_u64(vbias01x0123.val[0]);
    uint32x4_t vpacc01x23 = vreinterpretq_u32_u64(vbias01x0123.val[1]);
    uint32x4_t vpacc01x45 = vreinterpretq_u32_u64(vbias01x4567.val[0]);
    uint32x4_t vpacc01x67 = vreinterpretq_u32_u64(vbias01x4567.val[1]);
    uint32x4_t vpacc01x89 = vreinterpretq_u32_u64(vbias01x89AB.val[0]);
    uint32x4_t vpacc01xAB = vreinterpretq_u32_u64(vbias01x89AB.val[1]);
    uint32x4_t vpacc01xCD = vreinterpretq_u32_u64(vbias01xCDEF.val[0]);
    uint32x4_t vpacc01xEF = vreinterpretq_u32_u64(vbias01xCDEF.val[1]);
    uint32x4_t vnacc01 = vmovq_n_u32(0);

    size_t p = ks;
    do {
      const uint8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      const uint8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const uint8_t*) ((uintptr_t) a1 + a_offset);
      }
      a += 2;

      uint64x2x2_t va01x0123456789ABCDEF;
      va01x0123456789ABCDEF.val[0] = vdupq_n_u64(0);
      va01x0123456789ABCDEF.val[1] = vdupq_n_u64(0);
      // Inner accumulation loop along the 16 columns.
      size_t k = kc;
      // 2x partial unrolled loop to load 16 bytes at a time.
      while (k >= 16 * sizeof(uint8_t)) {
        // Load a 2x16 block of activations.
        #if XNN_ARCH_ARM64
          va01x0123456789ABCDEF = vld2q_lane_u64((const void*) a0, va01x0123456789ABCDEF, 0); a0 += 16;
          va01x0123456789ABCDEF = vld2q_lane_u64((const void*) a1, va01x0123456789ABCDEF, 1); a1 += 16;
        #else
          va01x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a0, va01x0123456789ABCDEF.val[0], 0); a0 += 8;
          va01x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a0, va01x0123456789ABCDEF.val[1], 0); a0 += 8;
          va01x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a1, va01x0123456789ABCDEF.val[0], 1); a1 += 8;
          va01x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a1, va01x0123456789ABCDEF.val[1], 1); a1 += 8;
        #endif

        // Load a 16x16 block of weights.
        const uint8x16_t vb01x01234567 = vld1q_u8(w); w = (const int8_t*) w + 16;
        const uint8x16_t vb23x01234567 = vld1q_u8(w); w = (const int8_t*) w + 16;
        const uint8x16_t vb45x01234567 = vld1q_u8(w); w = (const int8_t*) w + 16;
        const uint8x16_t vb67x01234567 = vld1q_u8(w); w = (const int8_t*) w + 16;
        const uint8x16_t vb89x01234567 = vld1q_u8(w); w = (const int8_t*) w + 16;
        const uint8x16_t vbABx01234567 = vld1q_u8(w); w = (const int8_t*) w + 16;
        const uint8x16_t vbCDx01234567 = vld1q_u8(w); w = (const int8_t*) w + 16;
        const uint8x16_t vbEFx01234567 = vld1q_u8(w); w = (const int8_t*) w + 16;
        const uint8x16_t vb01x89ABCDEF = vld1q_u8(w); w = (const int8_t*) w + 16;
        const uint8x16_t vb23x89ABCDEF = vld1q_u8(w); w = (const int8_t*) w + 16;
        const uint8x16_t vb45x89ABCDEF = vld1q_u8(w); w = (const int8_t*) w + 16;
        const uint8x16_t vb67x89ABCDEF = vld1q_u8(w); w = (const int8_t*) w + 16;
        const uint8x16_t vb89x89ABCDEF = vld1q_u8(w); w = (const int8_t*) w + 16;
        const uint8x16_t vbABx89ABCDEF = vld1q_u8(w); w = (const int8_t*) w + 16;
        const uint8x16_t vbCDx89ABCDEF = vld1q_u8(w); w = (const int8_t*) w + 16;
        const uint8x16_t vbEFx89ABCDEF = vld1q_u8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 2x16 * 16x16 --> 2x16.
        vnacc01 = vmmlaq_u32(vnacc01, vkernel_zero_point, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]));
        vpacc01x01 = vmmlaq_u32(vpacc01x01, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]), vb01x01234567);
        vpacc01x23 = vmmlaq_u32(vpacc01x23, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]), vb23x01234567);
        vpacc01x45 = vmmlaq_u32(vpacc01x45, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]), vb45x01234567);
        vpacc01x67 = vmmlaq_u32(vpacc01x67, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]), vb67x01234567);
        vpacc01x89 = vmmlaq_u32(vpacc01x89, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]), vb89x01234567);
        vpacc01xAB = vmmlaq_u32(vpacc01xAB, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]), vbABx01234567);
        vpacc01xCD = vmmlaq_u32(vpacc01xCD, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]), vbCDx01234567);
        vpacc01xEF = vmmlaq_u32(vpacc01xEF, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]), vbEFx01234567);
        vnacc01 = vmmlaq_u32(vnacc01, vkernel_zero_point, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]));
        vpacc01x01 = vmmlaq_u32(vpacc01x01, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]), vb01x89ABCDEF);
        vpacc01x23 = vmmlaq_u32(vpacc01x23, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]), vb23x89ABCDEF);
        vpacc01x45 = vmmlaq_u32(vpacc01x45, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]), vb45x89ABCDEF);
        vpacc01x67 = vmmlaq_u32(vpacc01x67, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]), vb67x89ABCDEF);
        vpacc01x89 = vmmlaq_u32(vpacc01x89, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]), vb89x89ABCDEF);
        vpacc01xAB = vmmlaq_u32(vpacc01xAB, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]), vbABx89ABCDEF);
        vpacc01xCD = vmmlaq_u32(vpacc01xCD, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]), vbCDx89ABCDEF);
        vpacc01xEF = vmmlaq_u32(vpacc01xEF, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]), vbEFx89ABCDEF);

        k -= 16 * sizeof(uint8_t);
      }
      // Handle up to 8 final positions of `k`
      if XNN_UNLIKELY(k != 0) {
        // Load a 2x8 block of activations.
        uint64x2_t va01x01234567 = vld1q_dup_u64((const void*) a0); a0 += 8;
        va01x01234567 = vld1q_lane_u64((const void*) a1, va01x01234567, 1); a1 += 8;

        // Load a 16x16 block of weights.
        const uint8x16_t vb01x01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint8x16_t vb23x01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint8x16_t vb45x01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint8x16_t vb67x01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint8x16_t vb89x01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint8x16_t vbABx01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint8x16_t vbCDx01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint8x16_t vbEFx01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;

        // Multiply-accumulate: 2x8 * 8x16 --> 2x16.
        vnacc01 = vmmlaq_u32(vnacc01, vkernel_zero_point, vreinterpretq_u8_u64(va01x01234567));
        vpacc01x01 = vmmlaq_u32(vpacc01x01, vreinterpretq_u8_u64(va01x01234567), vb01x01234567);
        vpacc01x23 = vmmlaq_u32(vpacc01x23, vreinterpretq_u8_u64(va01x01234567), vb23x01234567);
        vpacc01x45 = vmmlaq_u32(vpacc01x45, vreinterpretq_u8_u64(va01x01234567), vb45x01234567);
        vpacc01x67 = vmmlaq_u32(vpacc01x67, vreinterpretq_u8_u64(va01x01234567), vb67x01234567);
        vpacc01x89 = vmmlaq_u32(vpacc01x89, vreinterpretq_u8_u64(va01x01234567), vb89x01234567);
        vpacc01xAB = vmmlaq_u32(vpacc01xAB, vreinterpretq_u8_u64(va01x01234567), vbABx01234567);
        vpacc01xCD = vmmlaq_u32(vpacc01xCD, vreinterpretq_u8_u64(va01x01234567), vbCDx01234567);
        vpacc01xEF = vmmlaq_u32(vpacc01xEF, vreinterpretq_u8_u64(va01x01234567), vbEFx01234567);
      }
      p -= 2 * sizeof(void*);
    } while (p != 0);

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
    int32x4_t vacc01x89 = vreinterpretq_s32_u32(vsubq_u32(vpacc01x89, vnacc01x01));
    int32x4_t vacc01xAB = vreinterpretq_s32_u32(vsubq_u32(vpacc01xAB, vnacc01x01));
    int32x4_t vacc01xCD = vreinterpretq_s32_u32(vsubq_u32(vpacc01xCD, vnacc01x01));
    int32x4_t vacc01xEF = vreinterpretq_s32_u32(vsubq_u32(vpacc01xEF, vnacc01x01));

    #if XNN_ARCH_ARM64
      int32x4_t vacc0x0123 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x01), vreinterpretq_u64_s32(vacc01x23)));
      int32x4_t vacc1x0123 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01x01), vreinterpretq_u64_s32(vacc01x23)));
      int32x4_t vacc0x4567 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x45), vreinterpretq_u64_s32(vacc01x67)));
      int32x4_t vacc1x4567 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01x45), vreinterpretq_u64_s32(vacc01x67)));
      int32x4_t vacc0x89AB = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x89), vreinterpretq_u64_s32(vacc01xAB)));
      int32x4_t vacc1x89AB = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01x89), vreinterpretq_u64_s32(vacc01xAB)));
      int32x4_t vacc0xCDEF = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01xCD), vreinterpretq_u64_s32(vacc01xEF)));
      int32x4_t vacc1xCDEF = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01xCD), vreinterpretq_u64_s32(vacc01xEF)));
    #else
      int32x4_t vacc0x0123 = vcombine_s32(vget_low_s32(vacc01x01), vget_low_s32(vacc01x23));
      int32x4_t vacc1x0123 = vcombine_s32(vget_high_s32(vacc01x01), vget_high_s32(vacc01x23));
      int32x4_t vacc0x4567 = vcombine_s32(vget_low_s32(vacc01x45), vget_low_s32(vacc01x67));
      int32x4_t vacc1x4567 = vcombine_s32(vget_high_s32(vacc01x45), vget_high_s32(vacc01x67));
      int32x4_t vacc0x89AB = vcombine_s32(vget_low_s32(vacc01x89), vget_low_s32(vacc01xAB));
      int32x4_t vacc1x89AB = vcombine_s32(vget_high_s32(vacc01x89), vget_high_s32(vacc01xAB));
      int32x4_t vacc0xCDEF = vcombine_s32(vget_low_s32(vacc01xCD), vget_low_s32(vacc01xEF));
      int32x4_t vacc1xCDEF = vcombine_s32(vget_high_s32(vacc01xCD), vget_high_s32(vacc01xEF));
    #endif
    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vfpacc0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vfpacc0xCDEF = vcvtq_f32_s32(vacc0xCDEF);
    float32x4_t vfpacc1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vfpacc1x4567 = vcvtq_f32_s32(vacc1x4567);
    float32x4_t vfpacc1x89AB = vcvtq_f32_s32(vacc1x89AB);
    float32x4_t vfpacc1xCDEF = vcvtq_f32_s32(vacc1xCDEF);

    const float32x4_t vscale = vld1q_dup_f32(&params->fp32_neonv8.scale);
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale);
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale);
    vfpacc0x89AB = vmulq_f32(vfpacc0x89AB, vscale);
    vfpacc0xCDEF = vmulq_f32(vfpacc0xCDEF, vscale);
    vfpacc1x0123 = vmulq_f32(vfpacc1x0123, vscale);
    vfpacc1x4567 = vmulq_f32(vfpacc1x4567, vscale);
    vfpacc1x89AB = vmulq_f32(vfpacc1x89AB, vscale);
    vfpacc1xCDEF = vmulq_f32(vfpacc1xCDEF, vscale);

    vacc0x0123 = vcvtnq_s32_f32(vfpacc0x0123);
    vacc0x4567 = vcvtnq_s32_f32(vfpacc0x4567);
    vacc0x89AB = vcvtnq_s32_f32(vfpacc0x89AB);
    vacc0xCDEF = vcvtnq_s32_f32(vfpacc0xCDEF);
    vacc1x0123 = vcvtnq_s32_f32(vfpacc1x0123);
    vacc1x4567 = vcvtnq_s32_f32(vfpacc1x4567);
    vacc1x89AB = vcvtnq_s32_f32(vfpacc1x89AB);
    vacc1xCDEF = vcvtnq_s32_f32(vfpacc1xCDEF);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
    const int16x8_t vacc1x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x89AB), vacc1xCDEF), voutput_zero_point);

    uint8x16_t vout0x0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc0x01234567), vacc0x89ABCDEF);
    uint8x16_t vout1x0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc1x01234567), vacc1x89ABCDEF);
#else
    const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF)), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
    const int16x8_t vacc1x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x89AB), vqmovn_s32(vacc1xCDEF)), voutput_zero_point);

    uint8x16_t vout0x0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc0x01234567), vqmovun_s16(vacc0x89ABCDEF));
    uint8x16_t vout1x0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc1x01234567), vqmovun_s16(vacc1x89ABCDEF));
#endif
    const uint8x16_t voutput_min = vld1q_dup_u8(&params->fp32_neonv8.output_min);
    const uint8x16_t voutput_max = vld1q_dup_u8(&params->fp32_neonv8.output_max);

    vout0x0123456789ABCDEF = vmaxq_u8(vout0x0123456789ABCDEF, voutput_min);
    vout1x0123456789ABCDEF = vmaxq_u8(vout1x0123456789ABCDEF, voutput_min);

    vout0x0123456789ABCDEF = vminq_u8(vout0x0123456789ABCDEF, voutput_max);
    vout1x0123456789ABCDEF = vminq_u8(vout1x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      vst1q_u8(c1, vout1x0123456789ABCDEF);
      vst1q_u8(c0, vout0x0123456789ABCDEF);

      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      uint8x16_t vout0x01234567_1x01234567 = vcombine_u8(vget_low_u8(vout0x0123456789ABCDEF), vget_low_u8(vout1x0123456789ABCDEF));
      if (nc & 8) {
        vst1_u8(c1, vget_high_u8(vout0x01234567_1x01234567)); c1 += 8;
        vst1_u8(c0, vget_low_u8(vout0x01234567_1x01234567)); c0 += 8;
        vout0x01234567_1x01234567 = vcombine_u8(vget_high_u8(vout0x0123456789ABCDEF), vget_high_u8(vout1x0123456789ABCDEF));
      }
      if (nc & 4) {
        vst1q_lane_u32((void*) c1, vreinterpretq_u32_u8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1q_lane_u32((void*) c0, vreinterpretq_u32_u8(vout0x01234567_1x01234567), 0); c0 += 4;
        vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16((void*) c1, vreinterpretq_u16_u8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1q_lane_u16((void*) c0, vreinterpretq_u16_u8(vout0x01234567_1x01234567), 0); c0 += 2;
        vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_u8(c1, vout0x01234567_1x01234567, 8);
        vst1q_lane_u8(c0, vout0x01234567_1x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
