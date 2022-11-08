// Auto-generated file. Do not edit!
//   Template: src/qu8-gemm/c4-neondot.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/gemm.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>


void xnn_qu8_gemm_minmax_fp32_ukernel_1x16c4__neondot(
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

  kc = round_up_po2(kc, 4 * sizeof(uint8_t));
  const uint8_t* a0 = a;
  uint8_t* c0 = c;

  const uint8x8_t va_zero_point = vld1_dup_u8(&params->fp32_neonv8.kernel_zero_point[0]);

  // Loop over groups of 16 columns.
  do {
    // Initialize accumulators with bias. 16 bias values are loaded from the
    // weight matrix, at the start of the group of 16 columns.
    uint32x4_t vpacc0x0123 = vld1q_u32(w); w = (const void*) ((const uint32_t*) w + 4);
    uint32x4_t vpacc0x4567 = vld1q_u32(w); w = (const void*) ((const uint32_t*) w + 4);
    uint32x4_t vpacc0x89AB = vld1q_u32(w); w = (const void*) ((const uint32_t*) w + 4);
    uint32x4_t vpacc0xCDEF = vld1q_u32(w); w = (const void*) ((const uint32_t*) w + 4);
    uint32x2_t vnacc0 = vmov_n_u32(0);

    // Inner accumulation loop along the 16 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 8 * sizeof(uint8_t)) {
      // Load a 1x8 block of activations.
      const uint8x8_t va0x01234567 = vld1_u8(a0); a0 += 8;

      // Load a 8x16 block of weights.
      const uint8x16_t vb0123x0123 = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123x4567 = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123x89AB = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123xCDEF = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb4567x0123 = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb4567x4567 = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb4567x89AB = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb4567xCDEF = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);

      // Multiply-accumulate: 1x8 * 8x16 --> 1x16.
      vnacc0 = vdot_u32(vnacc0, va_zero_point, va0x01234567);
      vpacc0x0123 = vdotq_lane_u32(vpacc0x0123, vb0123x0123, va0x01234567, 0);
      vpacc0x4567 = vdotq_lane_u32(vpacc0x4567, vb0123x4567, va0x01234567, 0);
      vpacc0x89AB = vdotq_lane_u32(vpacc0x89AB, vb0123x89AB, va0x01234567, 0);
      vpacc0xCDEF = vdotq_lane_u32(vpacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
      vpacc0x0123 = vdotq_lane_u32(vpacc0x0123, vb4567x0123, va0x01234567, 1);
      vpacc0x4567 = vdotq_lane_u32(vpacc0x4567, vb4567x4567, va0x01234567, 1);
      vpacc0x89AB = vdotq_lane_u32(vpacc0x89AB, vb4567x89AB, va0x01234567, 1);
      vpacc0xCDEF = vdotq_lane_u32(vpacc0xCDEF, vb4567xCDEF, va0x01234567, 1);

      k -= 8 * sizeof(uint8_t);
    }
    // Handle up to 4 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 1x4 block of activations.
      const uint8x8_t va0x01234567 = vreinterpret_u8_u32(vld1_lane_u32((const void*) a0, vmov_n_u32(0), 0)); a0 += 4;

      // Load a 4x16 block of weights.
      const uint8x16_t vb0123x0123 = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123x4567 = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123x89AB = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123xCDEF = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);

      // Multiply-accumulate: 1x4 * 4x16 --> 1x16.
      vnacc0 = vdot_u32(vnacc0, va_zero_point, va0x01234567);
      vpacc0x0123 = vdotq_lane_u32(vpacc0x0123, vb0123x0123, va0x01234567, 0);
      vpacc0x4567 = vdotq_lane_u32(vpacc0x4567, vb0123x4567, va0x01234567, 0);
      vpacc0x89AB = vdotq_lane_u32(vpacc0x89AB, vb0123x89AB, va0x01234567, 0);
      vpacc0xCDEF = vdotq_lane_u32(vpacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
    }

    // Subtract zero point from accumulators.
    vnacc0 = vpadd_u32(vnacc0, vnacc0);
    const uint32x4_t vnacc0x0123 = vcombine_u32(vnacc0, vnacc0);
    int32x4_t vacc0x0123 = vreinterpretq_s32_u32(vsubq_u32(vpacc0x0123, vnacc0x0123));
    int32x4_t vacc0x4567 = vreinterpretq_s32_u32(vsubq_u32(vpacc0x4567, vnacc0x0123));
    int32x4_t vacc0x89AB = vreinterpretq_s32_u32(vsubq_u32(vpacc0x89AB, vnacc0x0123));
    int32x4_t vacc0xCDEF = vreinterpretq_s32_u32(vsubq_u32(vpacc0xCDEF, vnacc0x0123));

    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vfpacc0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vfpacc0xCDEF = vcvtq_f32_s32(vacc0xCDEF);

    const float32x4_t vscale = vld1q_dup_f32(&params->fp32_neonv8.scale);
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale);
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale);
    vfpacc0x89AB = vmulq_f32(vfpacc0x89AB, vscale);
    vfpacc0xCDEF = vmulq_f32(vfpacc0xCDEF, vscale);

    vacc0x0123 = vcvtnq_s32_f32(vfpacc0x0123);
    vacc0x4567 = vcvtnq_s32_f32(vfpacc0x4567);
    vacc0x89AB = vcvtnq_s32_f32(vfpacc0x89AB);
    vacc0xCDEF = vcvtnq_s32_f32(vfpacc0xCDEF);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF), voutput_zero_point);

    uint8x16_t vout0x0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc0x01234567), vacc0x89ABCDEF);
#else
    const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF)), voutput_zero_point);

    uint8x16_t vout0x0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc0x01234567), vqmovun_s16(vacc0x89ABCDEF));
#endif
    const uint8x16_t voutput_min = vld1q_dup_u8(&params->fp32_neonv8.output_min);
    const uint8x16_t voutput_max = vld1q_dup_u8(&params->fp32_neonv8.output_max);

    vout0x0123456789ABCDEF = vmaxq_u8(vout0x0123456789ABCDEF, voutput_min);

    vout0x0123456789ABCDEF = vminq_u8(vout0x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      vst1q_u8(c0 + 0, vout0x0123456789ABCDEF);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      uint8x8_t vout0x01234567 = vget_low_u8(vout0x0123456789ABCDEF);
      if (nc & 8) {
        vst1_u8(c0, vout0x01234567); c0 += 8;
        vout0x01234567 = vget_high_u8(vout0x0123456789ABCDEF);
      }
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
