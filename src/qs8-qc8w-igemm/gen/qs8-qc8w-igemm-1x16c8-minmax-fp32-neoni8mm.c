// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/c8-neoni8mm.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/igemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"


void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__neoni8mm(
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
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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
    int32x4_t vacc01x01 = vreinterpretq_s32_u64(vbias01x0123.val[0]);
    int32x4_t vacc01x23 = vreinterpretq_s32_u64(vbias01x0123.val[1]);
    int32x4_t vacc01x45 = vreinterpretq_s32_u64(vbias01x4567.val[0]);
    int32x4_t vacc01x67 = vreinterpretq_s32_u64(vbias01x4567.val[1]);
    int32x4_t vacc01x89 = vreinterpretq_s32_u64(vbias01x89AB.val[0]);
    int32x4_t vacc01xAB = vreinterpretq_s32_u64(vbias01x89AB.val[1]);
    int32x4_t vacc01xCD = vreinterpretq_s32_u64(vbias01xCDEF.val[0]);
    int32x4_t vacc01xEF = vreinterpretq_s32_u64(vbias01xCDEF.val[1]);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      uint64x2x2_t va01x0123456789ABCDEF;
      va01x0123456789ABCDEF.val[0] = vdupq_n_u64(0);
      va01x0123456789ABCDEF.val[1] = vdupq_n_u64(0);

      // Inner accumulation loop along the 16 columns.
      size_t k = kc;
      // 2x partial unrolled loop to load 8 bytes at a time.
      while (k >= 16 * sizeof(int8_t)) {
        // Load a 1x16 block of activations.
        #if XNN_ARCH_ARM64
          va01x0123456789ABCDEF = vld2q_lane_u64((const void*) a0, va01x0123456789ABCDEF, 0); a0 += 16;
        #else
          va01x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a0, va01x0123456789ABCDEF.val[0], 0); a0 += 8;
          va01x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a0, va01x0123456789ABCDEF.val[1], 0); a0 += 8;
        #endif

        // Load a 16x16 block of weights.
        const int8x16_t vb01x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb23x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb45x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb67x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb89x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbABx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbCDx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbEFx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb01x89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb23x89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb45x89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb67x89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb89x89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbABx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbCDx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbEFx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 1x16 * 16x16 --> 1x16.
        vacc01x01 = vmmlaq_s32(vacc01x01, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]), vb01x01234567);
        vacc01x23 = vmmlaq_s32(vacc01x23, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]), vb23x01234567);
        vacc01x45 = vmmlaq_s32(vacc01x45, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]), vb45x01234567);
        vacc01x67 = vmmlaq_s32(vacc01x67, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]), vb67x01234567);
        vacc01x89 = vmmlaq_s32(vacc01x89, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]), vb89x01234567);
        vacc01xAB = vmmlaq_s32(vacc01xAB, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]), vbABx01234567);
        vacc01xCD = vmmlaq_s32(vacc01xCD, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]), vbCDx01234567);
        vacc01xEF = vmmlaq_s32(vacc01xEF, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]), vbEFx01234567);
        vacc01x01 = vmmlaq_s32(vacc01x01, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]), vb01x89ABCDEF);
        vacc01x23 = vmmlaq_s32(vacc01x23, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]), vb23x89ABCDEF);
        vacc01x45 = vmmlaq_s32(vacc01x45, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]), vb45x89ABCDEF);
        vacc01x67 = vmmlaq_s32(vacc01x67, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]), vb67x89ABCDEF);
        vacc01x89 = vmmlaq_s32(vacc01x89, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]), vb89x89ABCDEF);
        vacc01xAB = vmmlaq_s32(vacc01xAB, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]), vbABx89ABCDEF);
        vacc01xCD = vmmlaq_s32(vacc01xCD, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]), vbCDx89ABCDEF);
        vacc01xEF = vmmlaq_s32(vacc01xEF, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]), vbEFx89ABCDEF);

        k -= 16 * sizeof(int8_t);
      }
      // Handle up to 8 final positions of `k`
      if XNN_UNLIKELY(k != 0) {
        // Load a 1x8 block of activations.
        uint64x2_t va01x01234567 = vld1q_dup_u64((const void*) a0); a0 += 8;

        // Load a 16x16 block of weights.
        const int8x16_t vb01x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb23x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb45x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb67x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb89x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbABx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbCDx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbEFx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 1x4 * 4x16 --> 1x16.
        vacc01x01 = vmmlaq_s32(vacc01x01, vreinterpretq_s8_u64(va01x01234567), vb01x01234567);
        vacc01x23 = vmmlaq_s32(vacc01x23, vreinterpretq_s8_u64(va01x01234567), vb23x01234567);
        vacc01x45 = vmmlaq_s32(vacc01x45, vreinterpretq_s8_u64(va01x01234567), vb45x01234567);
        vacc01x67 = vmmlaq_s32(vacc01x67, vreinterpretq_s8_u64(va01x01234567), vb67x01234567);
        vacc01x89 = vmmlaq_s32(vacc01x89, vreinterpretq_s8_u64(va01x01234567), vb89x01234567);
        vacc01xAB = vmmlaq_s32(vacc01xAB, vreinterpretq_s8_u64(va01x01234567), vbABx01234567);
        vacc01xCD = vmmlaq_s32(vacc01xCD, vreinterpretq_s8_u64(va01x01234567), vbCDx01234567);
        vacc01xEF = vmmlaq_s32(vacc01xEF, vreinterpretq_s8_u64(va01x01234567), vbEFx01234567);
      }

      p -= 1 * sizeof(void*);
    } while (p != 0);

    int32x4_t vacc0x0123 = vcombine_s32(vget_low_s32(vacc01x01), vget_low_s32(vacc01x23));
    int32x4_t vacc0x4567 = vcombine_s32(vget_low_s32(vacc01x45), vget_low_s32(vacc01x67));
    int32x4_t vacc0x89AB = vcombine_s32(vget_low_s32(vacc01x89), vget_low_s32(vacc01xAB));
    int32x4_t vacc0xCDEF = vcombine_s32(vget_low_s32(vacc01xCD), vget_low_s32(vacc01xEF));

    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vfpacc0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vfpacc0xCDEF = vcvtq_f32_s32(vacc0xCDEF);

    const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale4567);
    const float32x4_t vscale89AB = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x89AB = vmulq_f32(vfpacc0x89AB, vscale89AB);
    const float32x4_t vscaleCDEF = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0xCDEF = vmulq_f32(vfpacc0xCDEF, vscaleCDEF);

    vacc0x0123 = vcvtnq_s32_f32(vfpacc0x0123);
    vacc0x4567 = vcvtnq_s32_f32(vfpacc0x4567);
    vacc0x89AB = vcvtnq_s32_f32(vfpacc0x89AB);
    vacc0xCDEF = vcvtnq_s32_f32(vfpacc0xCDEF);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
    #if XNN_ARCH_ARM64
      const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
      const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF), voutput_zero_point);

      int8x16_t vout0x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc0x89ABCDEF);
    #else
      const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
      const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF)), voutput_zero_point);

      int8x16_t vout0x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc0x89ABCDEF));
    #endif
    const int8x16_t voutput_min = vld1q_dup_s8(&params->fp32_neonv8.output_min);
    const int8x16_t voutput_max = vld1q_dup_s8(&params->fp32_neonv8.output_max);

    vout0x0123456789ABCDEF = vmaxq_s8(vout0x0123456789ABCDEF, voutput_min);

    vout0x0123456789ABCDEF = vminq_s8(vout0x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      vst1q_s8(c0, vout0x0123456789ABCDEF);

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
