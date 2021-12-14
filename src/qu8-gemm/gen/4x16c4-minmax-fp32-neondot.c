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


void xnn_qu8_gemm_minmax_fp32_ukernel_4x16c4__neondot(
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
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(uint8_t));
  const uint8_t* a0 = a;
  uint8_t* c0 = c;
  const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const uint8_t* a2 = (const uint8_t*) ((uintptr_t) a1 + a_stride);
  uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const uint8_t* a3 = (const uint8_t*) ((uintptr_t) a2 + a_stride);
  uint8_t* c3 = (uint8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const uint8x8_t va_zero_point = vld1_dup_u8(&params->fp32_neonv8.kernel_zero_point[0]);

  // Loop over groups of 16 columns.
  do {
    // Initialize accumulators with bias. 16 bias values are loaded from the
    // weight matrix, at the start of the group of 16 columns.
    uint32x4_t vpacc0x0123 = vld1q_u32(w); w = (const void*) ((const uint32_t*) w + 4);
    uint32x4_t vpacc0x4567 = vld1q_u32(w); w = (const void*) ((const uint32_t*) w + 4);
    uint32x4_t vpacc0x89AB = vld1q_u32(w); w = (const void*) ((const uint32_t*) w + 4);
    uint32x4_t vpacc0xCDEF = vld1q_u32(w); w = (const void*) ((const uint32_t*) w + 4);
    uint32x4_t vpacc1x0123 = vpacc0x0123;
    uint32x4_t vpacc1x4567 = vpacc0x4567;
    uint32x4_t vpacc1x89AB = vpacc0x89AB;
    uint32x4_t vpacc1xCDEF = vpacc0xCDEF;
    uint32x4_t vpacc2x0123 = vpacc0x0123;
    uint32x4_t vpacc2x4567 = vpacc0x4567;
    uint32x4_t vpacc2x89AB = vpacc0x89AB;
    uint32x4_t vpacc2xCDEF = vpacc0xCDEF;
    uint32x4_t vpacc3x0123 = vpacc0x0123;
    uint32x4_t vpacc3x4567 = vpacc0x4567;
    uint32x4_t vpacc3x89AB = vpacc0x89AB;
    uint32x4_t vpacc3xCDEF = vpacc0xCDEF;
    uint32x2_t vnacc0 = vmov_n_u32(0);
    uint32x2_t vnacc1 = vmov_n_u32(0);
    uint32x2_t vnacc2 = vmov_n_u32(0);
    uint32x2_t vnacc3 = vmov_n_u32(0);

    // Inner accumulation loop along the 16 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 8 * sizeof(uint8_t)) {
      // Load a 4x8 block of activations.
      const uint8x8_t va0x01234567 = vld1_u8(a0); a0 += 8;
      const uint8x8_t va1x01234567 = vld1_u8(a1); a1 += 8;
      const uint8x8_t va2x01234567 = vld1_u8(a2); a2 += 8;
      const uint8x8_t va3x01234567 = vld1_u8(a3); a3 += 8;

      // Load a 8x16 block of weights.
      const uint8x16_t vb0123x0123 = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123x4567 = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123x89AB = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123xCDEF = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb4567x0123 = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb4567x4567 = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb4567x89AB = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb4567xCDEF = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);

      // Multiply-accumulate: 4x8 * 8x16 --> 4x16.
      vnacc0 = vdot_u32(vnacc0, va_zero_point, va0x01234567);
      vpacc0x0123 = vdotq_lane_u32(vpacc0x0123, vb0123x0123, va0x01234567, 0);
      vpacc0x4567 = vdotq_lane_u32(vpacc0x4567, vb0123x4567, va0x01234567, 0);
      vpacc0x89AB = vdotq_lane_u32(vpacc0x89AB, vb0123x89AB, va0x01234567, 0);
      vpacc0xCDEF = vdotq_lane_u32(vpacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
      vpacc0x0123 = vdotq_lane_u32(vpacc0x0123, vb4567x0123, va0x01234567, 1);
      vpacc0x4567 = vdotq_lane_u32(vpacc0x4567, vb4567x4567, va0x01234567, 1);
      vpacc0x89AB = vdotq_lane_u32(vpacc0x89AB, vb4567x89AB, va0x01234567, 1);
      vpacc0xCDEF = vdotq_lane_u32(vpacc0xCDEF, vb4567xCDEF, va0x01234567, 1);
      vnacc1 = vdot_u32(vnacc1, va_zero_point, va1x01234567);
      vpacc1x0123 = vdotq_lane_u32(vpacc1x0123, vb0123x0123, va1x01234567, 0);
      vpacc1x4567 = vdotq_lane_u32(vpacc1x4567, vb0123x4567, va1x01234567, 0);
      vpacc1x89AB = vdotq_lane_u32(vpacc1x89AB, vb0123x89AB, va1x01234567, 0);
      vpacc1xCDEF = vdotq_lane_u32(vpacc1xCDEF, vb0123xCDEF, va1x01234567, 0);
      vpacc1x0123 = vdotq_lane_u32(vpacc1x0123, vb4567x0123, va1x01234567, 1);
      vpacc1x4567 = vdotq_lane_u32(vpacc1x4567, vb4567x4567, va1x01234567, 1);
      vpacc1x89AB = vdotq_lane_u32(vpacc1x89AB, vb4567x89AB, va1x01234567, 1);
      vpacc1xCDEF = vdotq_lane_u32(vpacc1xCDEF, vb4567xCDEF, va1x01234567, 1);
      vnacc2 = vdot_u32(vnacc2, va_zero_point, va2x01234567);
      vpacc2x0123 = vdotq_lane_u32(vpacc2x0123, vb0123x0123, va2x01234567, 0);
      vpacc2x4567 = vdotq_lane_u32(vpacc2x4567, vb0123x4567, va2x01234567, 0);
      vpacc2x89AB = vdotq_lane_u32(vpacc2x89AB, vb0123x89AB, va2x01234567, 0);
      vpacc2xCDEF = vdotq_lane_u32(vpacc2xCDEF, vb0123xCDEF, va2x01234567, 0);
      vpacc2x0123 = vdotq_lane_u32(vpacc2x0123, vb4567x0123, va2x01234567, 1);
      vpacc2x4567 = vdotq_lane_u32(vpacc2x4567, vb4567x4567, va2x01234567, 1);
      vpacc2x89AB = vdotq_lane_u32(vpacc2x89AB, vb4567x89AB, va2x01234567, 1);
      vpacc2xCDEF = vdotq_lane_u32(vpacc2xCDEF, vb4567xCDEF, va2x01234567, 1);
      vnacc3 = vdot_u32(vnacc3, va_zero_point, va3x01234567);
      vpacc3x0123 = vdotq_lane_u32(vpacc3x0123, vb0123x0123, va3x01234567, 0);
      vpacc3x4567 = vdotq_lane_u32(vpacc3x4567, vb0123x4567, va3x01234567, 0);
      vpacc3x89AB = vdotq_lane_u32(vpacc3x89AB, vb0123x89AB, va3x01234567, 0);
      vpacc3xCDEF = vdotq_lane_u32(vpacc3xCDEF, vb0123xCDEF, va3x01234567, 0);
      vpacc3x0123 = vdotq_lane_u32(vpacc3x0123, vb4567x0123, va3x01234567, 1);
      vpacc3x4567 = vdotq_lane_u32(vpacc3x4567, vb4567x4567, va3x01234567, 1);
      vpacc3x89AB = vdotq_lane_u32(vpacc3x89AB, vb4567x89AB, va3x01234567, 1);
      vpacc3xCDEF = vdotq_lane_u32(vpacc3xCDEF, vb4567xCDEF, va3x01234567, 1);

      k -= 8 * sizeof(uint8_t);
    }
    // Handle up to 4 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 4x4 block of activations.
      const uint8x8_t va0x01234567 = vreinterpret_u8_u32(vld1_lane_u32((const void*) a0, vmov_n_u32(0), 0)); a0 += 4;
      const uint8x8_t va1x01234567 = vreinterpret_u8_u32(vld1_lane_u32((const void*) a1, vmov_n_u32(0), 0)); a1 += 4;
      const uint8x8_t va2x01234567 = vreinterpret_u8_u32(vld1_lane_u32((const void*) a2, vmov_n_u32(0), 0)); a2 += 4;
      const uint8x8_t va3x01234567 = vreinterpret_u8_u32(vld1_lane_u32((const void*) a3, vmov_n_u32(0), 0)); a3 += 4;

      // Load a 4x16 block of weights.
      const uint8x16_t vb0123x0123 = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123x4567 = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123x89AB = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123xCDEF = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);

      // Multiply-accumulate: 4x4 * 4x16 --> 4x16.
      vnacc0 = vdot_u32(vnacc0, va_zero_point, va0x01234567);
      vpacc0x0123 = vdotq_lane_u32(vpacc0x0123, vb0123x0123, va0x01234567, 0);
      vpacc0x4567 = vdotq_lane_u32(vpacc0x4567, vb0123x4567, va0x01234567, 0);
      vpacc0x89AB = vdotq_lane_u32(vpacc0x89AB, vb0123x89AB, va0x01234567, 0);
      vpacc0xCDEF = vdotq_lane_u32(vpacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
      vnacc1 = vdot_u32(vnacc1, va_zero_point, va1x01234567);
      vpacc1x0123 = vdotq_lane_u32(vpacc1x0123, vb0123x0123, va1x01234567, 0);
      vpacc1x4567 = vdotq_lane_u32(vpacc1x4567, vb0123x4567, va1x01234567, 0);
      vpacc1x89AB = vdotq_lane_u32(vpacc1x89AB, vb0123x89AB, va1x01234567, 0);
      vpacc1xCDEF = vdotq_lane_u32(vpacc1xCDEF, vb0123xCDEF, va1x01234567, 0);
      vnacc2 = vdot_u32(vnacc2, va_zero_point, va2x01234567);
      vpacc2x0123 = vdotq_lane_u32(vpacc2x0123, vb0123x0123, va2x01234567, 0);
      vpacc2x4567 = vdotq_lane_u32(vpacc2x4567, vb0123x4567, va2x01234567, 0);
      vpacc2x89AB = vdotq_lane_u32(vpacc2x89AB, vb0123x89AB, va2x01234567, 0);
      vpacc2xCDEF = vdotq_lane_u32(vpacc2xCDEF, vb0123xCDEF, va2x01234567, 0);
      vnacc3 = vdot_u32(vnacc3, va_zero_point, va3x01234567);
      vpacc3x0123 = vdotq_lane_u32(vpacc3x0123, vb0123x0123, va3x01234567, 0);
      vpacc3x4567 = vdotq_lane_u32(vpacc3x4567, vb0123x4567, va3x01234567, 0);
      vpacc3x89AB = vdotq_lane_u32(vpacc3x89AB, vb0123x89AB, va3x01234567, 0);
      vpacc3xCDEF = vdotq_lane_u32(vpacc3xCDEF, vb0123xCDEF, va3x01234567, 0);
    }

    // Subtract zero point from accumulators.
    vnacc0 = vpadd_u32(vnacc0, vnacc0);
    const uint32x4_t vnacc0x0123 = vcombine_u32(vnacc0, vnacc0);
    int32x4_t vacc0x0123 = vreinterpretq_s32_u32(vsubq_u32(vpacc0x0123, vnacc0x0123));
    int32x4_t vacc0x4567 = vreinterpretq_s32_u32(vsubq_u32(vpacc0x4567, vnacc0x0123));
    int32x4_t vacc0x89AB = vreinterpretq_s32_u32(vsubq_u32(vpacc0x89AB, vnacc0x0123));
    int32x4_t vacc0xCDEF = vreinterpretq_s32_u32(vsubq_u32(vpacc0xCDEF, vnacc0x0123));
    vnacc1 = vpadd_u32(vnacc1, vnacc1);
    const uint32x4_t vnacc1x0123 = vcombine_u32(vnacc1, vnacc1);
    int32x4_t vacc1x0123 = vreinterpretq_s32_u32(vsubq_u32(vpacc1x0123, vnacc1x0123));
    int32x4_t vacc1x4567 = vreinterpretq_s32_u32(vsubq_u32(vpacc1x4567, vnacc1x0123));
    int32x4_t vacc1x89AB = vreinterpretq_s32_u32(vsubq_u32(vpacc1x89AB, vnacc1x0123));
    int32x4_t vacc1xCDEF = vreinterpretq_s32_u32(vsubq_u32(vpacc1xCDEF, vnacc1x0123));
    vnacc2 = vpadd_u32(vnacc2, vnacc2);
    const uint32x4_t vnacc2x0123 = vcombine_u32(vnacc2, vnacc2);
    int32x4_t vacc2x0123 = vreinterpretq_s32_u32(vsubq_u32(vpacc2x0123, vnacc2x0123));
    int32x4_t vacc2x4567 = vreinterpretq_s32_u32(vsubq_u32(vpacc2x4567, vnacc2x0123));
    int32x4_t vacc2x89AB = vreinterpretq_s32_u32(vsubq_u32(vpacc2x89AB, vnacc2x0123));
    int32x4_t vacc2xCDEF = vreinterpretq_s32_u32(vsubq_u32(vpacc2xCDEF, vnacc2x0123));
    vnacc3 = vpadd_u32(vnacc3, vnacc3);
    const uint32x4_t vnacc3x0123 = vcombine_u32(vnacc3, vnacc3);
    int32x4_t vacc3x0123 = vreinterpretq_s32_u32(vsubq_u32(vpacc3x0123, vnacc3x0123));
    int32x4_t vacc3x4567 = vreinterpretq_s32_u32(vsubq_u32(vpacc3x4567, vnacc3x0123));
    int32x4_t vacc3x89AB = vreinterpretq_s32_u32(vsubq_u32(vpacc3x89AB, vnacc3x0123));
    int32x4_t vacc3xCDEF = vreinterpretq_s32_u32(vsubq_u32(vpacc3xCDEF, vnacc3x0123));

    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vfpacc0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vfpacc0xCDEF = vcvtq_f32_s32(vacc0xCDEF);
    float32x4_t vfpacc1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vfpacc1x4567 = vcvtq_f32_s32(vacc1x4567);
    float32x4_t vfpacc1x89AB = vcvtq_f32_s32(vacc1x89AB);
    float32x4_t vfpacc1xCDEF = vcvtq_f32_s32(vacc1xCDEF);
    float32x4_t vfpacc2x0123 = vcvtq_f32_s32(vacc2x0123);
    float32x4_t vfpacc2x4567 = vcvtq_f32_s32(vacc2x4567);
    float32x4_t vfpacc2x89AB = vcvtq_f32_s32(vacc2x89AB);
    float32x4_t vfpacc2xCDEF = vcvtq_f32_s32(vacc2xCDEF);
    float32x4_t vfpacc3x0123 = vcvtq_f32_s32(vacc3x0123);
    float32x4_t vfpacc3x4567 = vcvtq_f32_s32(vacc3x4567);
    float32x4_t vfpacc3x89AB = vcvtq_f32_s32(vacc3x89AB);
    float32x4_t vfpacc3xCDEF = vcvtq_f32_s32(vacc3xCDEF);

    const float32x4_t vscale = vld1q_dup_f32(&params->fp32_neonv8.scale);
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale);
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale);
    vfpacc0x89AB = vmulq_f32(vfpacc0x89AB, vscale);
    vfpacc0xCDEF = vmulq_f32(vfpacc0xCDEF, vscale);
    vfpacc1x0123 = vmulq_f32(vfpacc1x0123, vscale);
    vfpacc1x4567 = vmulq_f32(vfpacc1x4567, vscale);
    vfpacc1x89AB = vmulq_f32(vfpacc1x89AB, vscale);
    vfpacc1xCDEF = vmulq_f32(vfpacc1xCDEF, vscale);
    vfpacc2x0123 = vmulq_f32(vfpacc2x0123, vscale);
    vfpacc2x4567 = vmulq_f32(vfpacc2x4567, vscale);
    vfpacc2x89AB = vmulq_f32(vfpacc2x89AB, vscale);
    vfpacc2xCDEF = vmulq_f32(vfpacc2xCDEF, vscale);
    vfpacc3x0123 = vmulq_f32(vfpacc3x0123, vscale);
    vfpacc3x4567 = vmulq_f32(vfpacc3x4567, vscale);
    vfpacc3x89AB = vmulq_f32(vfpacc3x89AB, vscale);
    vfpacc3xCDEF = vmulq_f32(vfpacc3xCDEF, vscale);

    vacc0x0123 = vcvtnq_s32_f32(vfpacc0x0123);
    vacc0x4567 = vcvtnq_s32_f32(vfpacc0x4567);
    vacc0x89AB = vcvtnq_s32_f32(vfpacc0x89AB);
    vacc0xCDEF = vcvtnq_s32_f32(vfpacc0xCDEF);
    vacc1x0123 = vcvtnq_s32_f32(vfpacc1x0123);
    vacc1x4567 = vcvtnq_s32_f32(vfpacc1x4567);
    vacc1x89AB = vcvtnq_s32_f32(vfpacc1x89AB);
    vacc1xCDEF = vcvtnq_s32_f32(vfpacc1xCDEF);
    vacc2x0123 = vcvtnq_s32_f32(vfpacc2x0123);
    vacc2x4567 = vcvtnq_s32_f32(vfpacc2x4567);
    vacc2x89AB = vcvtnq_s32_f32(vfpacc2x89AB);
    vacc2xCDEF = vcvtnq_s32_f32(vfpacc2xCDEF);
    vacc3x0123 = vcvtnq_s32_f32(vfpacc3x0123);
    vacc3x4567 = vcvtnq_s32_f32(vfpacc3x4567);
    vacc3x89AB = vcvtnq_s32_f32(vfpacc3x89AB);
    vacc3xCDEF = vcvtnq_s32_f32(vfpacc3xCDEF);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
    const int16x8_t vacc1x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x89AB), vacc1xCDEF), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
    const int16x8_t vacc2x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x89AB), vacc2xCDEF), voutput_zero_point);
    const int16x8_t vacc3x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);
    const int16x8_t vacc3x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x89AB), vacc3xCDEF), voutput_zero_point);

    uint8x16_t vout0x0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc0x01234567), vacc0x89ABCDEF);
    uint8x16_t vout1x0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc1x01234567), vacc1x89ABCDEF);
    uint8x16_t vout2x0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc2x01234567), vacc2x89ABCDEF);
    uint8x16_t vout3x0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc3x01234567), vacc3x89ABCDEF);
#else
    const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF)), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
    const int16x8_t vacc1x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x89AB), vqmovn_s32(vacc1xCDEF)), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);
    const int16x8_t vacc2x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x89AB), vqmovn_s32(vacc2xCDEF)), voutput_zero_point);
    const int16x8_t vacc3x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)), voutput_zero_point);
    const int16x8_t vacc3x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x89AB), vqmovn_s32(vacc3xCDEF)), voutput_zero_point);

    uint8x16_t vout0x0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc0x01234567), vqmovun_s16(vacc0x89ABCDEF));
    uint8x16_t vout1x0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc1x01234567), vqmovun_s16(vacc1x89ABCDEF));
    uint8x16_t vout2x0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc2x01234567), vqmovun_s16(vacc2x89ABCDEF));
    uint8x16_t vout3x0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc3x01234567), vqmovun_s16(vacc3x89ABCDEF));
#endif
    const uint8x16_t voutput_min = vld1q_dup_u8(&params->fp32_neonv8.output_min);
    const uint8x16_t voutput_max = vld1q_dup_u8(&params->fp32_neonv8.output_max);

    vout0x0123456789ABCDEF = vmaxq_u8(vout0x0123456789ABCDEF, voutput_min);
    vout1x0123456789ABCDEF = vmaxq_u8(vout1x0123456789ABCDEF, voutput_min);
    vout2x0123456789ABCDEF = vmaxq_u8(vout2x0123456789ABCDEF, voutput_min);
    vout3x0123456789ABCDEF = vmaxq_u8(vout3x0123456789ABCDEF, voutput_min);

    vout0x0123456789ABCDEF = vminq_u8(vout0x0123456789ABCDEF, voutput_max);
    vout1x0123456789ABCDEF = vminq_u8(vout1x0123456789ABCDEF, voutput_max);
    vout2x0123456789ABCDEF = vminq_u8(vout2x0123456789ABCDEF, voutput_max);
    vout3x0123456789ABCDEF = vminq_u8(vout3x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      vst1q_u8(c0 + 0, vout0x0123456789ABCDEF);
      vst1q_u8(c1 + 0, vout1x0123456789ABCDEF);
      vst1q_u8(c2 + 0, vout2x0123456789ABCDEF);
      vst1q_u8(c3 + 0, vout3x0123456789ABCDEF);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (uint8_t*) ((uintptr_t) c3 + cn_stride);

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint8_t*) ((uintptr_t) a1 - kc);
      a2 = (const uint8_t*) ((uintptr_t) a2 - kc);
      a3 = (const uint8_t*) ((uintptr_t) a3 - kc);

      nc -= 16;
    } else {
      uint8x16_t vout0x01234567_1x01234567 = vcombine_u8(vget_low_u8(vout0x0123456789ABCDEF), vget_low_u8(vout1x0123456789ABCDEF));
      uint8x16_t vout2x01234567_3x01234567 = vcombine_u8(vget_low_u8(vout2x0123456789ABCDEF), vget_low_u8(vout3x0123456789ABCDEF));
      if (nc & 8) {
        vst1_u8(c0, vget_low_u8(vout0x01234567_1x01234567)); c0 += 8;
        vst1_u8(c1, vget_high_u8(vout0x01234567_1x01234567)); c1 += 8;
        vst1_u8(c2, vget_low_u8(vout2x01234567_3x01234567)); c2 += 8;
        vst1_u8(c3, vget_high_u8(vout2x01234567_3x01234567)); c3 += 8;
        vout0x01234567_1x01234567 = vcombine_u8(vget_high_u8(vout0x0123456789ABCDEF), vget_high_u8(vout1x0123456789ABCDEF));
        vout2x01234567_3x01234567 = vcombine_u8(vget_high_u8(vout2x0123456789ABCDEF), vget_high_u8(vout3x0123456789ABCDEF));
      }
      if (nc & 4) {
        vst1q_lane_u32((void*) c0, vreinterpretq_u32_u8(vout0x01234567_1x01234567), 0); c0 += 4;
        vst1q_lane_u32((void*) c1, vreinterpretq_u32_u8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1q_lane_u32((void*) c2, vreinterpretq_u32_u8(vout2x01234567_3x01234567), 0); c2 += 4;
        vst1q_lane_u32((void*) c3, vreinterpretq_u32_u8(vout2x01234567_3x01234567), 2); c3 += 4;
        vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
        vout2x01234567_3x01234567 = vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16((void*) c0, vreinterpretq_u16_u8(vout0x01234567_1x01234567), 0); c0 += 2;
        vst1q_lane_u16((void*) c1, vreinterpretq_u16_u8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1q_lane_u16((void*) c2, vreinterpretq_u16_u8(vout2x01234567_3x01234567), 0); c2 += 2;
        vst1q_lane_u16((void*) c3, vreinterpretq_u16_u8(vout2x01234567_3x01234567), 4); c3 += 2;
        vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
        vout2x01234567_3x01234567 = vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_u8(c0, vout0x01234567_1x01234567, 0);
        vst1q_lane_u8(c1, vout0x01234567_1x01234567, 8);
        vst1q_lane_u8(c2, vout2x01234567_3x01234567, 0);
        vst1q_lane_u8(c3, vout2x01234567_3x01234567, 8);
      }

      nc = 0;
    }
  } while (nc != 0);
}
