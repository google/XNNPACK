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
#include <xnnpack/math.h>


void xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot(
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
  assert(mr <= 3);
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

  const uint8x8_t va_zero_point = vld1_dup_u8(&params->rndnu_neon.kernel_zero_point[0]);

  // Loop over groups of 32 columns.
  do {
    // Initialize accumulators with bias. 32 bias values are loaded from the
    // weight matrix, at the start of the group of 32 columns.
    uint32x4_t vpacc0x0123 = vld1q_u32(w); w = (const void*) ((const uint32_t*) w + 4);
    uint32x4_t vpacc0x4567 = vld1q_u32(w); w = (const void*) ((const uint32_t*) w + 4);
    uint32x4_t vpacc0x89AB = vld1q_u32(w); w = (const void*) ((const uint32_t*) w + 4);
    uint32x4_t vpacc0xCDEF = vld1q_u32(w); w = (const void*) ((const uint32_t*) w + 4);
    uint32x4_t vpacc0xGHIJ = vld1q_u32(w); w = (const void*) ((const uint32_t*) w + 4);
    uint32x4_t vpacc0xKLMN = vld1q_u32(w); w = (const void*) ((const uint32_t*) w + 4);
    uint32x4_t vpacc0xOPQR = vld1q_u32(w); w = (const void*) ((const uint32_t*) w + 4);
    uint32x4_t vpacc0xSTUV = vld1q_u32(w); w = (const void*) ((const uint32_t*) w + 4);
    uint32x4_t vpacc1x0123 = vpacc0x0123;
    uint32x4_t vpacc1x4567 = vpacc0x4567;
    uint32x4_t vpacc1x89AB = vpacc0x89AB;
    uint32x4_t vpacc1xCDEF = vpacc0xCDEF;
    uint32x4_t vpacc1xGHIJ = vpacc0xGHIJ;
    uint32x4_t vpacc1xKLMN = vpacc0xKLMN;
    uint32x4_t vpacc1xOPQR = vpacc0xOPQR;
    uint32x4_t vpacc1xSTUV = vpacc0xSTUV;
    uint32x4_t vpacc2x0123 = vpacc0x0123;
    uint32x4_t vpacc2x4567 = vpacc0x4567;
    uint32x4_t vpacc2x89AB = vpacc0x89AB;
    uint32x4_t vpacc2xCDEF = vpacc0xCDEF;
    uint32x4_t vpacc2xGHIJ = vpacc0xGHIJ;
    uint32x4_t vpacc2xKLMN = vpacc0xKLMN;
    uint32x4_t vpacc2xOPQR = vpacc0xOPQR;
    uint32x4_t vpacc2xSTUV = vpacc0xSTUV;
    uint32x2_t vnacc0 = vmov_n_u32(0);
    uint32x2_t vnacc1 = vmov_n_u32(0);
    uint32x2_t vnacc2 = vmov_n_u32(0);

    // Inner accumulation loop along the 32 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 8 * sizeof(uint8_t)) {
      // Load a 3x8 block of activations.
      const uint8x8_t va0x01234567 = vld1_u8(a0); a0 += 8;
      const uint8x8_t va1x01234567 = vld1_u8(a1); a1 += 8;
      const uint8x8_t va2x01234567 = vld1_u8(a2); a2 += 8;

      // Load a 8x32 block of weights.
      const uint8x16_t vb0123x0123 = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123x4567 = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123x89AB = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123xCDEF = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123xGHIJ = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123xKLMN = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123xOPQR = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123xSTUV = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb4567x0123 = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb4567x4567 = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb4567x89AB = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb4567xCDEF = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb4567xGHIJ = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb4567xKLMN = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb4567xOPQR = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb4567xSTUV = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);

      // Multiply-accumulate: 3x8 * 8x32 --> 3x32.
      vnacc0 = vdot_u32(vnacc0, va_zero_point, va0x01234567);
      vpacc0x0123 = vdotq_lane_u32(vpacc0x0123, vb0123x0123, va0x01234567, 0);
      vpacc0x4567 = vdotq_lane_u32(vpacc0x4567, vb0123x4567, va0x01234567, 0);
      vpacc0x89AB = vdotq_lane_u32(vpacc0x89AB, vb0123x89AB, va0x01234567, 0);
      vpacc0xCDEF = vdotq_lane_u32(vpacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
      vpacc0xGHIJ = vdotq_lane_u32(vpacc0xGHIJ, vb0123xGHIJ, va0x01234567, 0);
      vpacc0xKLMN = vdotq_lane_u32(vpacc0xKLMN, vb0123xKLMN, va0x01234567, 0);
      vpacc0xOPQR = vdotq_lane_u32(vpacc0xOPQR, vb0123xOPQR, va0x01234567, 0);
      vpacc0xSTUV = vdotq_lane_u32(vpacc0xSTUV, vb0123xSTUV, va0x01234567, 0);
      vpacc0x0123 = vdotq_lane_u32(vpacc0x0123, vb4567x0123, va0x01234567, 1);
      vpacc0x4567 = vdotq_lane_u32(vpacc0x4567, vb4567x4567, va0x01234567, 1);
      vpacc0x89AB = vdotq_lane_u32(vpacc0x89AB, vb4567x89AB, va0x01234567, 1);
      vpacc0xCDEF = vdotq_lane_u32(vpacc0xCDEF, vb4567xCDEF, va0x01234567, 1);
      vpacc0xGHIJ = vdotq_lane_u32(vpacc0xGHIJ, vb4567xGHIJ, va0x01234567, 1);
      vpacc0xKLMN = vdotq_lane_u32(vpacc0xKLMN, vb4567xKLMN, va0x01234567, 1);
      vpacc0xOPQR = vdotq_lane_u32(vpacc0xOPQR, vb4567xOPQR, va0x01234567, 1);
      vpacc0xSTUV = vdotq_lane_u32(vpacc0xSTUV, vb4567xSTUV, va0x01234567, 1);
      vnacc1 = vdot_u32(vnacc1, va_zero_point, va1x01234567);
      vpacc1x0123 = vdotq_lane_u32(vpacc1x0123, vb0123x0123, va1x01234567, 0);
      vpacc1x4567 = vdotq_lane_u32(vpacc1x4567, vb0123x4567, va1x01234567, 0);
      vpacc1x89AB = vdotq_lane_u32(vpacc1x89AB, vb0123x89AB, va1x01234567, 0);
      vpacc1xCDEF = vdotq_lane_u32(vpacc1xCDEF, vb0123xCDEF, va1x01234567, 0);
      vpacc1xGHIJ = vdotq_lane_u32(vpacc1xGHIJ, vb0123xGHIJ, va1x01234567, 0);
      vpacc1xKLMN = vdotq_lane_u32(vpacc1xKLMN, vb0123xKLMN, va1x01234567, 0);
      vpacc1xOPQR = vdotq_lane_u32(vpacc1xOPQR, vb0123xOPQR, va1x01234567, 0);
      vpacc1xSTUV = vdotq_lane_u32(vpacc1xSTUV, vb0123xSTUV, va1x01234567, 0);
      vpacc1x0123 = vdotq_lane_u32(vpacc1x0123, vb4567x0123, va1x01234567, 1);
      vpacc1x4567 = vdotq_lane_u32(vpacc1x4567, vb4567x4567, va1x01234567, 1);
      vpacc1x89AB = vdotq_lane_u32(vpacc1x89AB, vb4567x89AB, va1x01234567, 1);
      vpacc1xCDEF = vdotq_lane_u32(vpacc1xCDEF, vb4567xCDEF, va1x01234567, 1);
      vpacc1xGHIJ = vdotq_lane_u32(vpacc1xGHIJ, vb4567xGHIJ, va1x01234567, 1);
      vpacc1xKLMN = vdotq_lane_u32(vpacc1xKLMN, vb4567xKLMN, va1x01234567, 1);
      vpacc1xOPQR = vdotq_lane_u32(vpacc1xOPQR, vb4567xOPQR, va1x01234567, 1);
      vpacc1xSTUV = vdotq_lane_u32(vpacc1xSTUV, vb4567xSTUV, va1x01234567, 1);
      vnacc2 = vdot_u32(vnacc2, va_zero_point, va2x01234567);
      vpacc2x0123 = vdotq_lane_u32(vpacc2x0123, vb0123x0123, va2x01234567, 0);
      vpacc2x4567 = vdotq_lane_u32(vpacc2x4567, vb0123x4567, va2x01234567, 0);
      vpacc2x89AB = vdotq_lane_u32(vpacc2x89AB, vb0123x89AB, va2x01234567, 0);
      vpacc2xCDEF = vdotq_lane_u32(vpacc2xCDEF, vb0123xCDEF, va2x01234567, 0);
      vpacc2xGHIJ = vdotq_lane_u32(vpacc2xGHIJ, vb0123xGHIJ, va2x01234567, 0);
      vpacc2xKLMN = vdotq_lane_u32(vpacc2xKLMN, vb0123xKLMN, va2x01234567, 0);
      vpacc2xOPQR = vdotq_lane_u32(vpacc2xOPQR, vb0123xOPQR, va2x01234567, 0);
      vpacc2xSTUV = vdotq_lane_u32(vpacc2xSTUV, vb0123xSTUV, va2x01234567, 0);
      vpacc2x0123 = vdotq_lane_u32(vpacc2x0123, vb4567x0123, va2x01234567, 1);
      vpacc2x4567 = vdotq_lane_u32(vpacc2x4567, vb4567x4567, va2x01234567, 1);
      vpacc2x89AB = vdotq_lane_u32(vpacc2x89AB, vb4567x89AB, va2x01234567, 1);
      vpacc2xCDEF = vdotq_lane_u32(vpacc2xCDEF, vb4567xCDEF, va2x01234567, 1);
      vpacc2xGHIJ = vdotq_lane_u32(vpacc2xGHIJ, vb4567xGHIJ, va2x01234567, 1);
      vpacc2xKLMN = vdotq_lane_u32(vpacc2xKLMN, vb4567xKLMN, va2x01234567, 1);
      vpacc2xOPQR = vdotq_lane_u32(vpacc2xOPQR, vb4567xOPQR, va2x01234567, 1);
      vpacc2xSTUV = vdotq_lane_u32(vpacc2xSTUV, vb4567xSTUV, va2x01234567, 1);

      k -= 8 * sizeof(uint8_t);
    }
    // Handle up to 4 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 3x4 block of activations.
      const uint8x8_t va0x01234567 = vreinterpret_u8_u32(vld1_lane_u32((const void*) a0, vmov_n_u32(0), 0)); a0 += 4;
      const uint8x8_t va1x01234567 = vreinterpret_u8_u32(vld1_lane_u32((const void*) a1, vmov_n_u32(0), 0)); a1 += 4;
      const uint8x8_t va2x01234567 = vreinterpret_u8_u32(vld1_lane_u32((const void*) a2, vmov_n_u32(0), 0)); a2 += 4;

      // Load a 4x32 block of weights.
      const uint8x16_t vb0123x0123 = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123x4567 = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123x89AB = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123xCDEF = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123xGHIJ = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123xKLMN = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123xOPQR = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);
      const uint8x16_t vb0123xSTUV = vld1q_u8(w); w = (const void*) ((const uint8_t*) w + 16);

      // Multiply-accumulate: 3x4 * 4x32 --> 3x32.
      vnacc0 = vdot_u32(vnacc0, va_zero_point, va0x01234567);
      vpacc0x0123 = vdotq_lane_u32(vpacc0x0123, vb0123x0123, va0x01234567, 0);
      vpacc0x4567 = vdotq_lane_u32(vpacc0x4567, vb0123x4567, va0x01234567, 0);
      vpacc0x89AB = vdotq_lane_u32(vpacc0x89AB, vb0123x89AB, va0x01234567, 0);
      vpacc0xCDEF = vdotq_lane_u32(vpacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
      vpacc0xGHIJ = vdotq_lane_u32(vpacc0xGHIJ, vb0123xGHIJ, va0x01234567, 0);
      vpacc0xKLMN = vdotq_lane_u32(vpacc0xKLMN, vb0123xKLMN, va0x01234567, 0);
      vpacc0xOPQR = vdotq_lane_u32(vpacc0xOPQR, vb0123xOPQR, va0x01234567, 0);
      vpacc0xSTUV = vdotq_lane_u32(vpacc0xSTUV, vb0123xSTUV, va0x01234567, 0);
      vnacc1 = vdot_u32(vnacc1, va_zero_point, va1x01234567);
      vpacc1x0123 = vdotq_lane_u32(vpacc1x0123, vb0123x0123, va1x01234567, 0);
      vpacc1x4567 = vdotq_lane_u32(vpacc1x4567, vb0123x4567, va1x01234567, 0);
      vpacc1x89AB = vdotq_lane_u32(vpacc1x89AB, vb0123x89AB, va1x01234567, 0);
      vpacc1xCDEF = vdotq_lane_u32(vpacc1xCDEF, vb0123xCDEF, va1x01234567, 0);
      vpacc1xGHIJ = vdotq_lane_u32(vpacc1xGHIJ, vb0123xGHIJ, va1x01234567, 0);
      vpacc1xKLMN = vdotq_lane_u32(vpacc1xKLMN, vb0123xKLMN, va1x01234567, 0);
      vpacc1xOPQR = vdotq_lane_u32(vpacc1xOPQR, vb0123xOPQR, va1x01234567, 0);
      vpacc1xSTUV = vdotq_lane_u32(vpacc1xSTUV, vb0123xSTUV, va1x01234567, 0);
      vnacc2 = vdot_u32(vnacc2, va_zero_point, va2x01234567);
      vpacc2x0123 = vdotq_lane_u32(vpacc2x0123, vb0123x0123, va2x01234567, 0);
      vpacc2x4567 = vdotq_lane_u32(vpacc2x4567, vb0123x4567, va2x01234567, 0);
      vpacc2x89AB = vdotq_lane_u32(vpacc2x89AB, vb0123x89AB, va2x01234567, 0);
      vpacc2xCDEF = vdotq_lane_u32(vpacc2xCDEF, vb0123xCDEF, va2x01234567, 0);
      vpacc2xGHIJ = vdotq_lane_u32(vpacc2xGHIJ, vb0123xGHIJ, va2x01234567, 0);
      vpacc2xKLMN = vdotq_lane_u32(vpacc2xKLMN, vb0123xKLMN, va2x01234567, 0);
      vpacc2xOPQR = vdotq_lane_u32(vpacc2xOPQR, vb0123xOPQR, va2x01234567, 0);
      vpacc2xSTUV = vdotq_lane_u32(vpacc2xSTUV, vb0123xSTUV, va2x01234567, 0);
    }

    // Subtract zero point from accumulators.
    vnacc0 = vpadd_u32(vnacc0, vnacc0);
    const uint32x4_t vnacc0x0123 = vcombine_u32(vnacc0, vnacc0);
    int32x4_t vacc0x0123 = vreinterpretq_s32_u32(vsubq_u32(vpacc0x0123, vnacc0x0123));
    int32x4_t vacc0x4567 = vreinterpretq_s32_u32(vsubq_u32(vpacc0x4567, vnacc0x0123));
    int32x4_t vacc0x89AB = vreinterpretq_s32_u32(vsubq_u32(vpacc0x89AB, vnacc0x0123));
    int32x4_t vacc0xCDEF = vreinterpretq_s32_u32(vsubq_u32(vpacc0xCDEF, vnacc0x0123));
    int32x4_t vacc0xGHIJ = vreinterpretq_s32_u32(vsubq_u32(vpacc0xGHIJ, vnacc0x0123));
    int32x4_t vacc0xKLMN = vreinterpretq_s32_u32(vsubq_u32(vpacc0xKLMN, vnacc0x0123));
    int32x4_t vacc0xOPQR = vreinterpretq_s32_u32(vsubq_u32(vpacc0xOPQR, vnacc0x0123));
    int32x4_t vacc0xSTUV = vreinterpretq_s32_u32(vsubq_u32(vpacc0xSTUV, vnacc0x0123));
    vnacc1 = vpadd_u32(vnacc1, vnacc1);
    const uint32x4_t vnacc1x0123 = vcombine_u32(vnacc1, vnacc1);
    int32x4_t vacc1x0123 = vreinterpretq_s32_u32(vsubq_u32(vpacc1x0123, vnacc1x0123));
    int32x4_t vacc1x4567 = vreinterpretq_s32_u32(vsubq_u32(vpacc1x4567, vnacc1x0123));
    int32x4_t vacc1x89AB = vreinterpretq_s32_u32(vsubq_u32(vpacc1x89AB, vnacc1x0123));
    int32x4_t vacc1xCDEF = vreinterpretq_s32_u32(vsubq_u32(vpacc1xCDEF, vnacc1x0123));
    int32x4_t vacc1xGHIJ = vreinterpretq_s32_u32(vsubq_u32(vpacc1xGHIJ, vnacc1x0123));
    int32x4_t vacc1xKLMN = vreinterpretq_s32_u32(vsubq_u32(vpacc1xKLMN, vnacc1x0123));
    int32x4_t vacc1xOPQR = vreinterpretq_s32_u32(vsubq_u32(vpacc1xOPQR, vnacc1x0123));
    int32x4_t vacc1xSTUV = vreinterpretq_s32_u32(vsubq_u32(vpacc1xSTUV, vnacc1x0123));
    vnacc2 = vpadd_u32(vnacc2, vnacc2);
    const uint32x4_t vnacc2x0123 = vcombine_u32(vnacc2, vnacc2);
    int32x4_t vacc2x0123 = vreinterpretq_s32_u32(vsubq_u32(vpacc2x0123, vnacc2x0123));
    int32x4_t vacc2x4567 = vreinterpretq_s32_u32(vsubq_u32(vpacc2x4567, vnacc2x0123));
    int32x4_t vacc2x89AB = vreinterpretq_s32_u32(vsubq_u32(vpacc2x89AB, vnacc2x0123));
    int32x4_t vacc2xCDEF = vreinterpretq_s32_u32(vsubq_u32(vpacc2xCDEF, vnacc2x0123));
    int32x4_t vacc2xGHIJ = vreinterpretq_s32_u32(vsubq_u32(vpacc2xGHIJ, vnacc2x0123));
    int32x4_t vacc2xKLMN = vreinterpretq_s32_u32(vsubq_u32(vpacc2xKLMN, vnacc2x0123));
    int32x4_t vacc2xOPQR = vreinterpretq_s32_u32(vsubq_u32(vpacc2xOPQR, vnacc2x0123));
    int32x4_t vacc2xSTUV = vreinterpretq_s32_u32(vsubq_u32(vpacc2xSTUV, vnacc2x0123));

    const int32x4_t vright_pre_shift = vld1q_dup_s32(&params->rndnu_neon.right_pre_shift);
    const int32x4_t vmultiplier = vld1q_dup_s32(&params->rndnu_neon.multiplier);
    const int32x4_t vright_post_shift = vld1q_dup_s32(&params->rndnu_neon.right_post_shift);

    vacc0x0123 = vshlq_s32(vacc0x0123, vright_pre_shift);
    vacc0x4567 = vshlq_s32(vacc0x4567, vright_pre_shift);
    vacc0x89AB = vshlq_s32(vacc0x89AB, vright_pre_shift);
    vacc0xCDEF = vshlq_s32(vacc0xCDEF, vright_pre_shift);
    vacc0xGHIJ = vshlq_s32(vacc0xGHIJ, vright_pre_shift);
    vacc0xKLMN = vshlq_s32(vacc0xKLMN, vright_pre_shift);
    vacc0xOPQR = vshlq_s32(vacc0xOPQR, vright_pre_shift);
    vacc0xSTUV = vshlq_s32(vacc0xSTUV, vright_pre_shift);
    vacc1x0123 = vshlq_s32(vacc1x0123, vright_pre_shift);
    vacc1x4567 = vshlq_s32(vacc1x4567, vright_pre_shift);
    vacc1x89AB = vshlq_s32(vacc1x89AB, vright_pre_shift);
    vacc1xCDEF = vshlq_s32(vacc1xCDEF, vright_pre_shift);
    vacc1xGHIJ = vshlq_s32(vacc1xGHIJ, vright_pre_shift);
    vacc1xKLMN = vshlq_s32(vacc1xKLMN, vright_pre_shift);
    vacc1xOPQR = vshlq_s32(vacc1xOPQR, vright_pre_shift);
    vacc1xSTUV = vshlq_s32(vacc1xSTUV, vright_pre_shift);
    vacc2x0123 = vshlq_s32(vacc2x0123, vright_pre_shift);
    vacc2x4567 = vshlq_s32(vacc2x4567, vright_pre_shift);
    vacc2x89AB = vshlq_s32(vacc2x89AB, vright_pre_shift);
    vacc2xCDEF = vshlq_s32(vacc2xCDEF, vright_pre_shift);
    vacc2xGHIJ = vshlq_s32(vacc2xGHIJ, vright_pre_shift);
    vacc2xKLMN = vshlq_s32(vacc2xKLMN, vright_pre_shift);
    vacc2xOPQR = vshlq_s32(vacc2xOPQR, vright_pre_shift);
    vacc2xSTUV = vshlq_s32(vacc2xSTUV, vright_pre_shift);

    vacc0x0123 = vqdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqdmulhq_s32(vacc0x4567, vmultiplier);
    vacc0x89AB = vqdmulhq_s32(vacc0x89AB, vmultiplier);
    vacc0xCDEF = vqdmulhq_s32(vacc0xCDEF, vmultiplier);
    vacc0xGHIJ = vqdmulhq_s32(vacc0xGHIJ, vmultiplier);
    vacc0xKLMN = vqdmulhq_s32(vacc0xKLMN, vmultiplier);
    vacc0xOPQR = vqdmulhq_s32(vacc0xOPQR, vmultiplier);
    vacc0xSTUV = vqdmulhq_s32(vacc0xSTUV, vmultiplier);
    vacc1x0123 = vqdmulhq_s32(vacc1x0123, vmultiplier);
    vacc1x4567 = vqdmulhq_s32(vacc1x4567, vmultiplier);
    vacc1x89AB = vqdmulhq_s32(vacc1x89AB, vmultiplier);
    vacc1xCDEF = vqdmulhq_s32(vacc1xCDEF, vmultiplier);
    vacc1xGHIJ = vqdmulhq_s32(vacc1xGHIJ, vmultiplier);
    vacc1xKLMN = vqdmulhq_s32(vacc1xKLMN, vmultiplier);
    vacc1xOPQR = vqdmulhq_s32(vacc1xOPQR, vmultiplier);
    vacc1xSTUV = vqdmulhq_s32(vacc1xSTUV, vmultiplier);
    vacc2x0123 = vqdmulhq_s32(vacc2x0123, vmultiplier);
    vacc2x4567 = vqdmulhq_s32(vacc2x4567, vmultiplier);
    vacc2x89AB = vqdmulhq_s32(vacc2x89AB, vmultiplier);
    vacc2xCDEF = vqdmulhq_s32(vacc2xCDEF, vmultiplier);
    vacc2xGHIJ = vqdmulhq_s32(vacc2xGHIJ, vmultiplier);
    vacc2xKLMN = vqdmulhq_s32(vacc2xKLMN, vmultiplier);
    vacc2xOPQR = vqdmulhq_s32(vacc2xOPQR, vmultiplier);
    vacc2xSTUV = vqdmulhq_s32(vacc2xSTUV, vmultiplier);

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_post_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_post_shift);
    vacc0x89AB = vrshlq_s32(vacc0x89AB, vright_post_shift);
    vacc0xCDEF = vrshlq_s32(vacc0xCDEF, vright_post_shift);
    vacc0xGHIJ = vrshlq_s32(vacc0xGHIJ, vright_post_shift);
    vacc0xKLMN = vrshlq_s32(vacc0xKLMN, vright_post_shift);
    vacc0xOPQR = vrshlq_s32(vacc0xOPQR, vright_post_shift);
    vacc0xSTUV = vrshlq_s32(vacc0xSTUV, vright_post_shift);
    vacc1x0123 = vrshlq_s32(vacc1x0123, vright_post_shift);
    vacc1x4567 = vrshlq_s32(vacc1x4567, vright_post_shift);
    vacc1x89AB = vrshlq_s32(vacc1x89AB, vright_post_shift);
    vacc1xCDEF = vrshlq_s32(vacc1xCDEF, vright_post_shift);
    vacc1xGHIJ = vrshlq_s32(vacc1xGHIJ, vright_post_shift);
    vacc1xKLMN = vrshlq_s32(vacc1xKLMN, vright_post_shift);
    vacc1xOPQR = vrshlq_s32(vacc1xOPQR, vright_post_shift);
    vacc1xSTUV = vrshlq_s32(vacc1xSTUV, vright_post_shift);
    vacc2x0123 = vrshlq_s32(vacc2x0123, vright_post_shift);
    vacc2x4567 = vrshlq_s32(vacc2x4567, vright_post_shift);
    vacc2x89AB = vrshlq_s32(vacc2x89AB, vright_post_shift);
    vacc2xCDEF = vrshlq_s32(vacc2xCDEF, vright_post_shift);
    vacc2xGHIJ = vrshlq_s32(vacc2xGHIJ, vright_post_shift);
    vacc2xKLMN = vrshlq_s32(vacc2xKLMN, vright_post_shift);
    vacc2xOPQR = vrshlq_s32(vacc2xOPQR, vright_post_shift);
    vacc2xSTUV = vrshlq_s32(vacc2xSTUV, vright_post_shift);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->rndnu_neon.output_zero_point);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF), voutput_zero_point);
    const int16x8_t vacc0xGHIJKLMN = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0xGHIJ), vacc0xKLMN), voutput_zero_point);
    const int16x8_t vacc0xOPQRSTUV = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0xOPQR), vacc0xSTUV), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
    const int16x8_t vacc1x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x89AB), vacc1xCDEF), voutput_zero_point);
    const int16x8_t vacc1xGHIJKLMN = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1xGHIJ), vacc1xKLMN), voutput_zero_point);
    const int16x8_t vacc1xOPQRSTUV = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1xOPQR), vacc1xSTUV), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
    const int16x8_t vacc2x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x89AB), vacc2xCDEF), voutput_zero_point);
    const int16x8_t vacc2xGHIJKLMN = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2xGHIJ), vacc2xKLMN), voutput_zero_point);
    const int16x8_t vacc2xOPQRSTUV = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2xOPQR), vacc2xSTUV), voutput_zero_point);

    uint8x16_t vout0x0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc0x01234567), vacc0x89ABCDEF);
    uint8x16_t vout0xGHIJKLMNOPQRSTUV = vqmovun_high_s16(vqmovun_s16(vacc0xGHIJKLMN), vacc0xOPQRSTUV);
    uint8x16_t vout1x0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc1x01234567), vacc1x89ABCDEF);
    uint8x16_t vout1xGHIJKLMNOPQRSTUV = vqmovun_high_s16(vqmovun_s16(vacc1xGHIJKLMN), vacc1xOPQRSTUV);
    uint8x16_t vout2x0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc2x01234567), vacc2x89ABCDEF);
    uint8x16_t vout2xGHIJKLMNOPQRSTUV = vqmovun_high_s16(vqmovun_s16(vacc2xGHIJKLMN), vacc2xOPQRSTUV);
#else
    const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF)), voutput_zero_point);
    const int16x8_t vacc0xGHIJKLMN = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0xGHIJ), vqmovn_s32(vacc0xKLMN)), voutput_zero_point);
    const int16x8_t vacc0xOPQRSTUV = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0xOPQR), vqmovn_s32(vacc0xSTUV)), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
    const int16x8_t vacc1x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x89AB), vqmovn_s32(vacc1xCDEF)), voutput_zero_point);
    const int16x8_t vacc1xGHIJKLMN = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1xGHIJ), vqmovn_s32(vacc1xKLMN)), voutput_zero_point);
    const int16x8_t vacc1xOPQRSTUV = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1xOPQR), vqmovn_s32(vacc1xSTUV)), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);
    const int16x8_t vacc2x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x89AB), vqmovn_s32(vacc2xCDEF)), voutput_zero_point);
    const int16x8_t vacc2xGHIJKLMN = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2xGHIJ), vqmovn_s32(vacc2xKLMN)), voutput_zero_point);
    const int16x8_t vacc2xOPQRSTUV = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2xOPQR), vqmovn_s32(vacc2xSTUV)), voutput_zero_point);

    uint8x16_t vout0x0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc0x01234567), vqmovun_s16(vacc0x89ABCDEF));
    uint8x16_t vout0xGHIJKLMNOPQRSTUV = vcombine_u8(vqmovun_s16(vacc0xGHIJKLMN), vqmovun_s16(vacc0xOPQRSTUV));
    uint8x16_t vout1x0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc1x01234567), vqmovun_s16(vacc1x89ABCDEF));
    uint8x16_t vout1xGHIJKLMNOPQRSTUV = vcombine_u8(vqmovun_s16(vacc1xGHIJKLMN), vqmovun_s16(vacc1xOPQRSTUV));
    uint8x16_t vout2x0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc2x01234567), vqmovun_s16(vacc2x89ABCDEF));
    uint8x16_t vout2xGHIJKLMNOPQRSTUV = vcombine_u8(vqmovun_s16(vacc2xGHIJKLMN), vqmovun_s16(vacc2xOPQRSTUV));
#endif
    const uint8x16_t voutput_min = vld1q_dup_u8(&params->rndnu_neon.output_min);
    const uint8x16_t voutput_max = vld1q_dup_u8(&params->rndnu_neon.output_max);

    vout0x0123456789ABCDEF = vmaxq_u8(vout0x0123456789ABCDEF, voutput_min);
    vout0xGHIJKLMNOPQRSTUV = vmaxq_u8(vout0xGHIJKLMNOPQRSTUV, voutput_min);
    vout1x0123456789ABCDEF = vmaxq_u8(vout1x0123456789ABCDEF, voutput_min);
    vout1xGHIJKLMNOPQRSTUV = vmaxq_u8(vout1xGHIJKLMNOPQRSTUV, voutput_min);
    vout2x0123456789ABCDEF = vmaxq_u8(vout2x0123456789ABCDEF, voutput_min);
    vout2xGHIJKLMNOPQRSTUV = vmaxq_u8(vout2xGHIJKLMNOPQRSTUV, voutput_min);

    vout0x0123456789ABCDEF = vminq_u8(vout0x0123456789ABCDEF, voutput_max);
    vout0xGHIJKLMNOPQRSTUV = vminq_u8(vout0xGHIJKLMNOPQRSTUV, voutput_max);
    vout1x0123456789ABCDEF = vminq_u8(vout1x0123456789ABCDEF, voutput_max);
    vout1xGHIJKLMNOPQRSTUV = vminq_u8(vout1xGHIJKLMNOPQRSTUV, voutput_max);
    vout2x0123456789ABCDEF = vminq_u8(vout2x0123456789ABCDEF, voutput_max);
    vout2xGHIJKLMNOPQRSTUV = vminq_u8(vout2xGHIJKLMNOPQRSTUV, voutput_max);

    if (nc >= 32) {
      vst1q_u8(c0 + 0, vout0x0123456789ABCDEF);
      vst1q_u8(c0 + 16, vout0xGHIJKLMNOPQRSTUV);
      vst1q_u8(c1 + 0, vout1x0123456789ABCDEF);
      vst1q_u8(c1 + 16, vout1xGHIJKLMNOPQRSTUV);
      vst1q_u8(c2 + 0, vout2x0123456789ABCDEF);
      vst1q_u8(c2 + 16, vout2xGHIJKLMNOPQRSTUV);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint8_t*) ((uintptr_t) a1 - kc);
      a2 = (const uint8_t*) ((uintptr_t) a2 - kc);

      nc -= 32;
    } else {
      if (nc & 16) {
        vst1q_u8(c0, vout0x0123456789ABCDEF);  c0 += 16;
        vst1q_u8(c1, vout1x0123456789ABCDEF);  c1 += 16;
        vst1q_u8(c2, vout2x0123456789ABCDEF);  c2 += 16;

        vout0x0123456789ABCDEF = vout0xGHIJKLMNOPQRSTUV;
        vout1x0123456789ABCDEF = vout1xGHIJKLMNOPQRSTUV;
        vout2x0123456789ABCDEF = vout2xGHIJKLMNOPQRSTUV;
      }
      uint8x16_t vout0x01234567_1x01234567 = vcombine_u8(vget_low_u8(vout0x0123456789ABCDEF), vget_low_u8(vout1x0123456789ABCDEF));
      uint8x8_t vout2x01234567 = vget_low_u8(vout2x0123456789ABCDEF);
      if (nc & 8) {
        vst1_u8(c0, vget_low_u8(vout0x01234567_1x01234567)); c0 += 8;
        vst1_u8(c1, vget_high_u8(vout0x01234567_1x01234567)); c1 += 8;
        vst1_u8(c2, vout2x01234567); c2 += 8;
        vout0x01234567_1x01234567 = vcombine_u8(vget_high_u8(vout0x0123456789ABCDEF), vget_high_u8(vout1x0123456789ABCDEF));
        vout2x01234567 = vget_high_u8(vout2x0123456789ABCDEF);
      }
      if (nc & 4) {
        vst1q_lane_u32((void*) c0, vreinterpretq_u32_u8(vout0x01234567_1x01234567), 0); c0 += 4;
        vst1q_lane_u32((void*) c1, vreinterpretq_u32_u8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1_lane_u32((void*) c2, vreinterpret_u32_u8(vout2x01234567), 0); c2 += 4;
        vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
        vout2x01234567 = vext_u8(vout2x01234567, vout2x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16((void*) c0, vreinterpretq_u16_u8(vout0x01234567_1x01234567), 0); c0 += 2;
        vst1q_lane_u16((void*) c1, vreinterpretq_u16_u8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1_lane_u16((void*) c2, vreinterpret_u16_u8(vout2x01234567), 0); c2 += 2;
        vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
        vout2x01234567 = vext_u8(vout2x01234567, vout2x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_u8(c0, vout0x01234567_1x01234567, 0);
        vst1q_lane_u8(c1, vout0x01234567_1x01234567, 8);
        vst1_lane_u8(c2, vout2x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
