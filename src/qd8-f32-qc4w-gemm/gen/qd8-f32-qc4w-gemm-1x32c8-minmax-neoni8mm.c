// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c8-neoni8mm.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/gemm.h"
#include "xnnpack/math.h"


void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x32c8__neoni8mm(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;
  const int8x16_t vmask = vmovq_n_s8(INT8_C(0xF0));

  // Loop over groups of 32 columns.
  do {
    // Initialize accumulators with bias. 32 bias values are loaded from the
    // weight matrix, at the start of the group of 32 columns.
    const int32x4_t vinput_zero_point01 = vld1q_dup_s32(&quantization_params[0].zero_point);
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0x0123 = vmulq_s32(vksum0123, vinput_zero_point01);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0x4567 = vmulq_s32(vksum4567, vinput_zero_point01);
    const int32x4_t vksum89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0x89AB = vmulq_s32(vksum89AB, vinput_zero_point01);
    const int32x4_t vksumCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0xCDEF = vmulq_s32(vksumCDEF, vinput_zero_point01);
    const int32x4_t vksumGHIJ = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0xGHIJ = vmulq_s32(vksumGHIJ, vinput_zero_point01);
    const int32x4_t vksumKLMN = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0xKLMN = vmulq_s32(vksumKLMN, vinput_zero_point01);
    const int32x4_t vksumOPQR = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0xOPQR = vmulq_s32(vksumOPQR, vinput_zero_point01);
    const int32x4_t vksumSTUV = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0xSTUV = vmulq_s32(vksumSTUV, vinput_zero_point01);

    #if XNN_ARCH_ARM64
      int32x4_t vacc01x01 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0x0123), vreinterpretq_u64_s32(vksumzp0x0123)));
      int32x4_t vacc01x23 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0x0123), vreinterpretq_u64_s32(vksumzp0x0123)));
      int32x4_t vacc01x45 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0x4567), vreinterpretq_u64_s32(vksumzp0x4567)));
      int32x4_t vacc01x67 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0x4567), vreinterpretq_u64_s32(vksumzp0x4567)));
      int32x4_t vacc01x89 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0x89AB), vreinterpretq_u64_s32(vksumzp0x89AB)));
      int32x4_t vacc01xAB = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0x89AB), vreinterpretq_u64_s32(vksumzp0x89AB)));
      int32x4_t vacc01xCD = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0xCDEF), vreinterpretq_u64_s32(vksumzp0xCDEF)));
      int32x4_t vacc01xEF = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0xCDEF), vreinterpretq_u64_s32(vksumzp0xCDEF)));
      int32x4_t vacc01xGH = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0xGHIJ), vreinterpretq_u64_s32(vksumzp0xGHIJ)));
      int32x4_t vacc01xIJ = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0xGHIJ), vreinterpretq_u64_s32(vksumzp0xGHIJ)));
      int32x4_t vacc01xKL = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0xKLMN), vreinterpretq_u64_s32(vksumzp0xKLMN)));
      int32x4_t vacc01xMN = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0xKLMN), vreinterpretq_u64_s32(vksumzp0xKLMN)));
      int32x4_t vacc01xOP = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0xOPQR), vreinterpretq_u64_s32(vksumzp0xOPQR)));
      int32x4_t vacc01xQR = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0xOPQR), vreinterpretq_u64_s32(vksumzp0xOPQR)));
      int32x4_t vacc01xST = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0xSTUV), vreinterpretq_u64_s32(vksumzp0xSTUV)));
      int32x4_t vacc01xUV = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0xSTUV), vreinterpretq_u64_s32(vksumzp0xSTUV)));
    #else
      int32x4_t vacc01x01 = vcombine_s32(vget_low_s32(vksumzp0x0123), vget_low_s32(vksumzp0x0123));
      int32x4_t vacc01x23 = vcombine_s32(vget_high_s32(vksumzp0x0123), vget_high_s32(vksumzp0x0123));
      int32x4_t vacc01x45 = vcombine_s32(vget_low_s32(vksumzp0x4567), vget_low_s32(vksumzp0x4567));
      int32x4_t vacc01x67 = vcombine_s32(vget_high_s32(vksumzp0x4567), vget_high_s32(vksumzp0x4567));
      int32x4_t vacc01x89 = vcombine_s32(vget_low_s32(vksumzp0x89AB), vget_low_s32(vksumzp0x89AB));
      int32x4_t vacc01xAB = vcombine_s32(vget_high_s32(vksumzp0x89AB), vget_high_s32(vksumzp0x89AB));
      int32x4_t vacc01xCD = vcombine_s32(vget_low_s32(vksumzp0xCDEF), vget_low_s32(vksumzp0xCDEF));
      int32x4_t vacc01xEF = vcombine_s32(vget_high_s32(vksumzp0xCDEF), vget_high_s32(vksumzp0xCDEF));
      int32x4_t vacc01xGH = vcombine_s32(vget_low_s32(vksumzp0xGHIJ), vget_low_s32(vksumzp0xGHIJ));
      int32x4_t vacc01xIJ = vcombine_s32(vget_high_s32(vksumzp0xGHIJ), vget_high_s32(vksumzp0xGHIJ));
      int32x4_t vacc01xKL = vcombine_s32(vget_low_s32(vksumzp0xKLMN), vget_low_s32(vksumzp0xKLMN));
      int32x4_t vacc01xMN = vcombine_s32(vget_high_s32(vksumzp0xKLMN), vget_high_s32(vksumzp0xKLMN));
      int32x4_t vacc01xOP = vcombine_s32(vget_low_s32(vksumzp0xOPQR), vget_low_s32(vksumzp0xOPQR));
      int32x4_t vacc01xQR = vcombine_s32(vget_high_s32(vksumzp0xOPQR), vget_high_s32(vksumzp0xOPQR));
      int32x4_t vacc01xST = vcombine_s32(vget_low_s32(vksumzp0xSTUV), vget_low_s32(vksumzp0xSTUV));
      int32x4_t vacc01xUV = vcombine_s32(vget_high_s32(vksumzp0xSTUV), vget_high_s32(vksumzp0xSTUV));
    #endif

    // Inner accumulation loop along the 32 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.

    uint64x2x2_t va01x0123456789ABCDEF;
    va01x0123456789ABCDEF.val[0] = vdupq_n_u64(0);
    va01x0123456789ABCDEF.val[1] = vdupq_n_u64(0);

    while (k >= 16 * sizeof(int8_t)) {
      // Load a 1x16 block of activations.
      #if XNN_ARCH_ARM64
        va01x0123456789ABCDEF = vld2q_lane_u64((const void*) a0, va01x0123456789ABCDEF, 0); a0 += 16;
      #else
        va01x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a0, va01x0123456789ABCDEF.val[0], 0); a0 += 8;
        va01x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a0, va01x0123456789ABCDEF.val[1], 0); a0 += 8;
      #endif

      const int8x16_t va01x01234567 = vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]);
      const int8x16_t va01x89ABCDEF = vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]);

      // Load a 16x32 block of weights.
      const int8x16_t vb01x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb23x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb45x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb67x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb89x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbABx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbCDx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbEFx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbGHx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbIJx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbKLx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbMNx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbOPx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbQRx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbSTx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbUVx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01x01234567 = vshlq_n_s8(vb01x0123456789ABCDEF, 4);
      const int8x16_t vb23x01234567 = vshlq_n_s8(vb23x0123456789ABCDEF, 4);
      const int8x16_t vb45x01234567 = vshlq_n_s8(vb45x0123456789ABCDEF, 4);
      const int8x16_t vb67x01234567 = vshlq_n_s8(vb67x0123456789ABCDEF, 4);
      const int8x16_t vb89x01234567 = vshlq_n_s8(vb89x0123456789ABCDEF, 4);
      const int8x16_t vbABx01234567 = vshlq_n_s8(vbABx0123456789ABCDEF, 4);
      const int8x16_t vbCDx01234567 = vshlq_n_s8(vbCDx0123456789ABCDEF, 4);
      const int8x16_t vbEFx01234567 = vshlq_n_s8(vbEFx0123456789ABCDEF, 4);
      const int8x16_t vbGHx01234567 = vshlq_n_s8(vbGHx0123456789ABCDEF, 4);
      const int8x16_t vbIJx01234567 = vshlq_n_s8(vbIJx0123456789ABCDEF, 4);
      const int8x16_t vbKLx01234567 = vshlq_n_s8(vbKLx0123456789ABCDEF, 4);
      const int8x16_t vbMNx01234567 = vshlq_n_s8(vbMNx0123456789ABCDEF, 4);
      const int8x16_t vbOPx01234567 = vshlq_n_s8(vbOPx0123456789ABCDEF, 4);
      const int8x16_t vbQRx01234567 = vshlq_n_s8(vbQRx0123456789ABCDEF, 4);
      const int8x16_t vbSTx01234567 = vshlq_n_s8(vbSTx0123456789ABCDEF, 4);
      const int8x16_t vbUVx01234567 = vshlq_n_s8(vbUVx0123456789ABCDEF, 4);
      const int8x16_t vb01x89ABCDEF = vandq_s8(vb01x0123456789ABCDEF, vmask);
      const int8x16_t vb23x89ABCDEF = vandq_s8(vb23x0123456789ABCDEF, vmask);
      const int8x16_t vb45x89ABCDEF = vandq_s8(vb45x0123456789ABCDEF, vmask);
      const int8x16_t vb67x89ABCDEF = vandq_s8(vb67x0123456789ABCDEF, vmask);
      const int8x16_t vb89x89ABCDEF = vandq_s8(vb89x0123456789ABCDEF, vmask);
      const int8x16_t vbABx89ABCDEF = vandq_s8(vbABx0123456789ABCDEF, vmask);
      const int8x16_t vbCDx89ABCDEF = vandq_s8(vbCDx0123456789ABCDEF, vmask);
      const int8x16_t vbEFx89ABCDEF = vandq_s8(vbEFx0123456789ABCDEF, vmask);
      const int8x16_t vbGHx89ABCDEF = vandq_s8(vbGHx0123456789ABCDEF, vmask);
      const int8x16_t vbIJx89ABCDEF = vandq_s8(vbIJx0123456789ABCDEF, vmask);
      const int8x16_t vbKLx89ABCDEF = vandq_s8(vbKLx0123456789ABCDEF, vmask);
      const int8x16_t vbMNx89ABCDEF = vandq_s8(vbMNx0123456789ABCDEF, vmask);
      const int8x16_t vbOPx89ABCDEF = vandq_s8(vbOPx0123456789ABCDEF, vmask);
      const int8x16_t vbQRx89ABCDEF = vandq_s8(vbQRx0123456789ABCDEF, vmask);
      const int8x16_t vbSTx89ABCDEF = vandq_s8(vbSTx0123456789ABCDEF, vmask);
      const int8x16_t vbUVx89ABCDEF = vandq_s8(vbUVx0123456789ABCDEF, vmask);

      // Multiply-accumulate: 1x8 * 8x32 --> 1x32.
      vacc01x01 = vmmlaq_s32(vacc01x01, va01x01234567, vb01x01234567);
      vacc01x23 = vmmlaq_s32(vacc01x23, va01x01234567, vb23x01234567);
      vacc01x45 = vmmlaq_s32(vacc01x45, va01x01234567, vb45x01234567);
      vacc01x67 = vmmlaq_s32(vacc01x67, va01x01234567, vb67x01234567);
      vacc01x89 = vmmlaq_s32(vacc01x89, va01x01234567, vb89x01234567);
      vacc01xAB = vmmlaq_s32(vacc01xAB, va01x01234567, vbABx01234567);
      vacc01xCD = vmmlaq_s32(vacc01xCD, va01x01234567, vbCDx01234567);
      vacc01xEF = vmmlaq_s32(vacc01xEF, va01x01234567, vbEFx01234567);
      vacc01xGH = vmmlaq_s32(vacc01xGH, va01x01234567, vbGHx01234567);
      vacc01xIJ = vmmlaq_s32(vacc01xIJ, va01x01234567, vbIJx01234567);
      vacc01xKL = vmmlaq_s32(vacc01xKL, va01x01234567, vbKLx01234567);
      vacc01xMN = vmmlaq_s32(vacc01xMN, va01x01234567, vbMNx01234567);
      vacc01xOP = vmmlaq_s32(vacc01xOP, va01x01234567, vbOPx01234567);
      vacc01xQR = vmmlaq_s32(vacc01xQR, va01x01234567, vbQRx01234567);
      vacc01xST = vmmlaq_s32(vacc01xST, va01x01234567, vbSTx01234567);
      vacc01xUV = vmmlaq_s32(vacc01xUV, va01x01234567, vbUVx01234567);
      vacc01x01 = vmmlaq_s32(vacc01x01, va01x89ABCDEF, vb01x89ABCDEF);
      vacc01x23 = vmmlaq_s32(vacc01x23, va01x89ABCDEF, vb23x89ABCDEF);
      vacc01x45 = vmmlaq_s32(vacc01x45, va01x89ABCDEF, vb45x89ABCDEF);
      vacc01x67 = vmmlaq_s32(vacc01x67, va01x89ABCDEF, vb67x89ABCDEF);
      vacc01x89 = vmmlaq_s32(vacc01x89, va01x89ABCDEF, vb89x89ABCDEF);
      vacc01xAB = vmmlaq_s32(vacc01xAB, va01x89ABCDEF, vbABx89ABCDEF);
      vacc01xCD = vmmlaq_s32(vacc01xCD, va01x89ABCDEF, vbCDx89ABCDEF);
      vacc01xEF = vmmlaq_s32(vacc01xEF, va01x89ABCDEF, vbEFx89ABCDEF);
      vacc01xGH = vmmlaq_s32(vacc01xGH, va01x89ABCDEF, vbGHx89ABCDEF);
      vacc01xIJ = vmmlaq_s32(vacc01xIJ, va01x89ABCDEF, vbIJx89ABCDEF);
      vacc01xKL = vmmlaq_s32(vacc01xKL, va01x89ABCDEF, vbKLx89ABCDEF);
      vacc01xMN = vmmlaq_s32(vacc01xMN, va01x89ABCDEF, vbMNx89ABCDEF);
      vacc01xOP = vmmlaq_s32(vacc01xOP, va01x89ABCDEF, vbOPx89ABCDEF);
      vacc01xQR = vmmlaq_s32(vacc01xQR, va01x89ABCDEF, vbQRx89ABCDEF);
      vacc01xST = vmmlaq_s32(vacc01xST, va01x89ABCDEF, vbSTx89ABCDEF);
      vacc01xUV = vmmlaq_s32(vacc01xUV, va01x89ABCDEF, vbUVx89ABCDEF);

      k -= 16 * sizeof(int8_t);
    }
    // Handle up to 8 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 1x8 block of activations.
      uint64x2_t va01x01234567 = vld1q_dup_u64((const void*) a0); a0 += 8;

      // Load a 16x32 block of weights.
      const int8x16_t vb01x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb23x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb45x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb67x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb89x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbABx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbCDx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbEFx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbGHx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbIJx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbKLx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbMNx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbOPx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbQRx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbSTx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbUVx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01x01234567 = vshlq_n_s8(vb01x0123456789ABCDEF, 4);
      const int8x16_t vb23x01234567 = vshlq_n_s8(vb23x0123456789ABCDEF, 4);
      const int8x16_t vb45x01234567 = vshlq_n_s8(vb45x0123456789ABCDEF, 4);
      const int8x16_t vb67x01234567 = vshlq_n_s8(vb67x0123456789ABCDEF, 4);
      const int8x16_t vb89x01234567 = vshlq_n_s8(vb89x0123456789ABCDEF, 4);
      const int8x16_t vbABx01234567 = vshlq_n_s8(vbABx0123456789ABCDEF, 4);
      const int8x16_t vbCDx01234567 = vshlq_n_s8(vbCDx0123456789ABCDEF, 4);
      const int8x16_t vbEFx01234567 = vshlq_n_s8(vbEFx0123456789ABCDEF, 4);
      const int8x16_t vbGHx01234567 = vshlq_n_s8(vbGHx0123456789ABCDEF, 4);
      const int8x16_t vbIJx01234567 = vshlq_n_s8(vbIJx0123456789ABCDEF, 4);
      const int8x16_t vbKLx01234567 = vshlq_n_s8(vbKLx0123456789ABCDEF, 4);
      const int8x16_t vbMNx01234567 = vshlq_n_s8(vbMNx0123456789ABCDEF, 4);
      const int8x16_t vbOPx01234567 = vshlq_n_s8(vbOPx0123456789ABCDEF, 4);
      const int8x16_t vbQRx01234567 = vshlq_n_s8(vbQRx0123456789ABCDEF, 4);
      const int8x16_t vbSTx01234567 = vshlq_n_s8(vbSTx0123456789ABCDEF, 4);
      const int8x16_t vbUVx01234567 = vshlq_n_s8(vbUVx0123456789ABCDEF, 4);

      // Multiply-accumulate: 1x4 * 4x32 --> 1x32.
      vacc01x01 = vmmlaq_s32(vacc01x01, vreinterpretq_s8_u64(va01x01234567), vb01x01234567);
      vacc01x23 = vmmlaq_s32(vacc01x23, vreinterpretq_s8_u64(va01x01234567), vb23x01234567);
      vacc01x45 = vmmlaq_s32(vacc01x45, vreinterpretq_s8_u64(va01x01234567), vb45x01234567);
      vacc01x67 = vmmlaq_s32(vacc01x67, vreinterpretq_s8_u64(va01x01234567), vb67x01234567);
      vacc01x89 = vmmlaq_s32(vacc01x89, vreinterpretq_s8_u64(va01x01234567), vb89x01234567);
      vacc01xAB = vmmlaq_s32(vacc01xAB, vreinterpretq_s8_u64(va01x01234567), vbABx01234567);
      vacc01xCD = vmmlaq_s32(vacc01xCD, vreinterpretq_s8_u64(va01x01234567), vbCDx01234567);
      vacc01xEF = vmmlaq_s32(vacc01xEF, vreinterpretq_s8_u64(va01x01234567), vbEFx01234567);
      vacc01xGH = vmmlaq_s32(vacc01xGH, vreinterpretq_s8_u64(va01x01234567), vbGHx01234567);
      vacc01xIJ = vmmlaq_s32(vacc01xIJ, vreinterpretq_s8_u64(va01x01234567), vbIJx01234567);
      vacc01xKL = vmmlaq_s32(vacc01xKL, vreinterpretq_s8_u64(va01x01234567), vbKLx01234567);
      vacc01xMN = vmmlaq_s32(vacc01xMN, vreinterpretq_s8_u64(va01x01234567), vbMNx01234567);
      vacc01xOP = vmmlaq_s32(vacc01xOP, vreinterpretq_s8_u64(va01x01234567), vbOPx01234567);
      vacc01xQR = vmmlaq_s32(vacc01xQR, vreinterpretq_s8_u64(va01x01234567), vbQRx01234567);
      vacc01xST = vmmlaq_s32(vacc01xST, vreinterpretq_s8_u64(va01x01234567), vbSTx01234567);
      vacc01xUV = vmmlaq_s32(vacc01xUV, vreinterpretq_s8_u64(va01x01234567), vbUVx01234567);
    }

    #if XNN_ARCH_ARM64
      int32x4_t vacc0x0123 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x01), vreinterpretq_u64_s32(vacc01x23)));
      int32x4_t vacc0x4567 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x45), vreinterpretq_u64_s32(vacc01x67)));
      int32x4_t vacc0x89AB = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x89), vreinterpretq_u64_s32(vacc01xAB)));
      int32x4_t vacc0xCDEF = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01xCD), vreinterpretq_u64_s32(vacc01xEF)));
      int32x4_t vacc0xGHIJ = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01xGH), vreinterpretq_u64_s32(vacc01xIJ)));
      int32x4_t vacc0xKLMN = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01xKL), vreinterpretq_u64_s32(vacc01xMN)));
      int32x4_t vacc0xOPQR = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01xOP), vreinterpretq_u64_s32(vacc01xQR)));
      int32x4_t vacc0xSTUV = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01xST), vreinterpretq_u64_s32(vacc01xUV)));
    #else
      int32x4_t vacc0x0123 = vcombine_s32(vget_low_s32(vacc01x01), vget_low_s32(vacc01x23));
      int32x4_t vacc0x4567 = vcombine_s32(vget_low_s32(vacc01x45), vget_low_s32(vacc01x67));
      int32x4_t vacc0x89AB = vcombine_s32(vget_low_s32(vacc01x89), vget_low_s32(vacc01xAB));
      int32x4_t vacc0xCDEF = vcombine_s32(vget_low_s32(vacc01xCD), vget_low_s32(vacc01xEF));
      int32x4_t vacc0xGHIJ = vcombine_s32(vget_low_s32(vacc01xGH), vget_low_s32(vacc01xIJ));
      int32x4_t vacc0xKLMN = vcombine_s32(vget_low_s32(vacc01xKL), vget_low_s32(vacc01xMN));
      int32x4_t vacc0xOPQR = vcombine_s32(vget_low_s32(vacc01xOP), vget_low_s32(vacc01xQR));
      int32x4_t vacc0xSTUV = vcombine_s32(vget_low_s32(vacc01xST), vget_low_s32(vacc01xUV));
    #endif
    float32x4_t vout0x0123 = vcvtq_n_f32_s32(vacc0x0123, 4);
    float32x4_t vout0x4567 = vcvtq_n_f32_s32(vacc0x4567, 4);
    float32x4_t vout0x89AB = vcvtq_n_f32_s32(vacc0x89AB, 4);
    float32x4_t vout0xCDEF = vcvtq_n_f32_s32(vacc0xCDEF, 4);
    float32x4_t vout0xGHIJ = vcvtq_n_f32_s32(vacc0xGHIJ, 4);
    float32x4_t vout0xKLMN = vcvtq_n_f32_s32(vacc0xKLMN, 4);
    float32x4_t vout0xOPQR = vcvtq_n_f32_s32(vacc0xOPQR, 4);
    float32x4_t vout0xSTUV = vcvtq_n_f32_s32(vacc0xSTUV, 4);

    const float32x4_t vinput_scale0 = vld1q_dup_f32(&quantization_params[0].inv_scale);
    vout0x0123 = vmulq_f32(vout0x0123, vinput_scale0);
    vout0x4567 = vmulq_f32(vout0x4567, vinput_scale0);
    vout0x89AB = vmulq_f32(vout0x89AB, vinput_scale0);
    vout0xCDEF = vmulq_f32(vout0xCDEF, vinput_scale0);
    vout0xGHIJ = vmulq_f32(vout0xGHIJ, vinput_scale0);
    vout0xKLMN = vmulq_f32(vout0xKLMN, vinput_scale0);
    vout0xOPQR = vmulq_f32(vout0xOPQR, vinput_scale0);
    vout0xSTUV = vmulq_f32(vout0xSTUV, vinput_scale0);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale89AB = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleCDEF = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleGHIJ = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleKLMN = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleOPQR = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleSTUV = vld1q_f32(w); w = (const float*) w + 4;

    #if XNN_ARCH_ARM64
      const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
      vout0x89AB = vfmaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
      const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
      vout0xCDEF = vfmaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
      const float32x4_t vbiasGHIJ = vld1q_f32(w); w = (const float*) w + 4;
      vout0xGHIJ = vfmaq_f32(vbiasGHIJ, vout0xGHIJ, vfilter_output_scaleGHIJ);
      const float32x4_t vbiasKLMN = vld1q_f32(w); w = (const float*) w + 4;
      vout0xKLMN = vfmaq_f32(vbiasKLMN, vout0xKLMN, vfilter_output_scaleKLMN);
      const float32x4_t vbiasOPQR = vld1q_f32(w); w = (const float*) w + 4;
      vout0xOPQR = vfmaq_f32(vbiasOPQR, vout0xOPQR, vfilter_output_scaleOPQR);
      const float32x4_t vbiasSTUV = vld1q_f32(w); w = (const float*) w + 4;
      vout0xSTUV = vfmaq_f32(vbiasSTUV, vout0xSTUV, vfilter_output_scaleSTUV);
    #else
      const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
      vout0x89AB = vmlaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
      const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
      vout0xCDEF = vmlaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
      const float32x4_t vbiasGHIJ = vld1q_f32(w); w = (const float*) w + 4;
      vout0xGHIJ = vmlaq_f32(vbiasGHIJ, vout0xGHIJ, vfilter_output_scaleGHIJ);
      const float32x4_t vbiasKLMN = vld1q_f32(w); w = (const float*) w + 4;
      vout0xKLMN = vmlaq_f32(vbiasKLMN, vout0xKLMN, vfilter_output_scaleKLMN);
      const float32x4_t vbiasOPQR = vld1q_f32(w); w = (const float*) w + 4;
      vout0xOPQR = vmlaq_f32(vbiasOPQR, vout0xOPQR, vfilter_output_scaleOPQR);
      const float32x4_t vbiasSTUV = vld1q_f32(w); w = (const float*) w + 4;
      vout0xSTUV = vmlaq_f32(vbiasSTUV, vout0xSTUV, vfilter_output_scaleSTUV);
    #endif

    const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, voutput_min);
    vout0x4567 = vmaxq_f32(vout0x4567, voutput_min);
    vout0x89AB = vmaxq_f32(vout0x89AB, voutput_min);
    vout0xCDEF = vmaxq_f32(vout0xCDEF, voutput_min);
    vout0xGHIJ = vmaxq_f32(vout0xGHIJ, voutput_min);
    vout0xKLMN = vmaxq_f32(vout0xKLMN, voutput_min);
    vout0xOPQR = vmaxq_f32(vout0xOPQR, voutput_min);
    vout0xSTUV = vmaxq_f32(vout0xSTUV, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);
    vout0x89AB = vminq_f32(vout0x89AB, voutput_max);
    vout0xCDEF = vminq_f32(vout0xCDEF, voutput_max);
    vout0xGHIJ = vminq_f32(vout0xGHIJ, voutput_max);
    vout0xKLMN = vminq_f32(vout0xKLMN, voutput_max);
    vout0xOPQR = vminq_f32(vout0xOPQR, voutput_max);
    vout0xSTUV = vminq_f32(vout0xSTUV, voutput_max);

    if XNN_LIKELY(nc >= 32) {
      vst1q_f32(c0, vout0x0123);
      vst1q_f32(c0 + 4, vout0x4567);
      vst1q_f32(c0 + 8, vout0x89AB);
      vst1q_f32(c0 + 12, vout0xCDEF);
      vst1q_f32(c0 + 16, vout0xGHIJ);
      vst1q_f32(c0 + 20, vout0xKLMN);
      vst1q_f32(c0 + 24, vout0xOPQR);
      vst1q_f32(c0 + 28, vout0xSTUV);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 32;
    } else {
     if (nc & 16) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0xGHIJ;
       vst1q_f32(c0, vout0x4567); c0 += 4;
       vout0x4567 = vout0xKLMN;
       vst1q_f32(c0, vout0x89AB); c0 += 4;
       vout0x89AB = vout0xOPQR;
       vst1q_f32(c0, vout0xCDEF); c0 += 4;
       vout0xCDEF = vout0xSTUV;
     }
     if (nc & 8) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x89AB;
       vst1q_f32(c0, vout0x4567); c0 += 4;
       vout0x4567 = vout0xCDEF;
     }
     if (nc & 4) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x4567;
     }
     float32x2_t vout0x01 = vget_low_f32(vout0x0123);
     if (nc & 2) {
       vst1_f32(c0, vout0x01); c0 += 2;
       vout0x01 = vget_high_f32(vout0x0123);
     }
     if (nc & 1) {
       vst1_lane_f32(c0, vout0x01, 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}
