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


void xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__neoni8mm(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qb4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 5);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  size_t bl = params->scalar.blocksize;
  assert(bl <= kc);
  assert(bl != 0);
  assert(bl % 32 == 0);
  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const int8_t* a4 = (const int8_t*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const int8x16_t vmask = vmovq_n_s8(INT8_C(0xF0));

  // Loop over groups of 16 columns.
  do {
    // Initialize accumulators with scaled vksum. 16 scaled vksum values are loaded from the
    // weight matrix, at the start of the group of 16 columns.
    const float32x4_t vinput_zero_point01 = vcvtq_f32_s32(vld1q_s32(&quantization_params[0].zero_point));
    const float32x4_t vksum0123 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vout0x0123 = vmulq_lane_f32(vksum0123, vget_low_f32(vinput_zero_point01), 0);
    float32x4_t vout1x0123 = vmulq_lane_f32(vksum0123, vget_high_f32(vinput_zero_point01), 0);
    const float32x4_t vksum4567 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vout0x4567 = vmulq_lane_f32(vksum4567, vget_low_f32(vinput_zero_point01), 0);
    float32x4_t vout1x4567 = vmulq_lane_f32(vksum4567, vget_high_f32(vinput_zero_point01), 0);
    const float32x4_t vksum89AB = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vout0x89AB = vmulq_lane_f32(vksum89AB, vget_low_f32(vinput_zero_point01), 0);
    float32x4_t vout1x89AB = vmulq_lane_f32(vksum89AB, vget_high_f32(vinput_zero_point01), 0);
    const float32x4_t vksumCDEF = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vout0xCDEF = vmulq_lane_f32(vksumCDEF, vget_low_f32(vinput_zero_point01), 0);
    float32x4_t vout1xCDEF = vmulq_lane_f32(vksumCDEF, vget_high_f32(vinput_zero_point01), 0);
    const float32x4_t vinput_zero_point23 = vcvtq_f32_s32(vld1q_s32(&quantization_params[2].zero_point));
    float32x4_t vout2x0123 = vmulq_lane_f32(vksum0123, vget_low_f32(vinput_zero_point23), 0);
    float32x4_t vout3x0123 = vmulq_lane_f32(vksum0123, vget_high_f32(vinput_zero_point23), 0);
    float32x4_t vout2x4567 = vmulq_lane_f32(vksum4567, vget_low_f32(vinput_zero_point23), 0);
    float32x4_t vout3x4567 = vmulq_lane_f32(vksum4567, vget_high_f32(vinput_zero_point23), 0);
    float32x4_t vout2x89AB = vmulq_lane_f32(vksum89AB, vget_low_f32(vinput_zero_point23), 0);
    float32x4_t vout3x89AB = vmulq_lane_f32(vksum89AB, vget_high_f32(vinput_zero_point23), 0);
    float32x4_t vout2xCDEF = vmulq_lane_f32(vksumCDEF, vget_low_f32(vinput_zero_point23), 0);
    float32x4_t vout3xCDEF = vmulq_lane_f32(vksumCDEF, vget_high_f32(vinput_zero_point23), 0);
    const float32x4_t vinput_zero_point45 = vcvtq_f32_s32(vld1q_dup_s32(&quantization_params[4].zero_point));
    float32x4_t vout4x0123 = vmulq_f32(vksum0123, vinput_zero_point45);
    float32x4_t vout4x4567 = vmulq_f32(vksum4567, vinput_zero_point45);
    float32x4_t vout4x89AB = vmulq_f32(vksum89AB, vinput_zero_point45);
    float32x4_t vout4xCDEF = vmulq_f32(vksumCDEF, vinput_zero_point45);


    for (size_t kb=0; kb < kc; kb += bl) {
      int32x4_t vacc01x01 = vdupq_n_s32(0);
      int32x4_t vacc01x23 = vdupq_n_s32(0);
      int32x4_t vacc01x45 = vdupq_n_s32(0);
      int32x4_t vacc01x67 = vdupq_n_s32(0);
      int32x4_t vacc01x89 = vdupq_n_s32(0);
      int32x4_t vacc01xAB = vdupq_n_s32(0);
      int32x4_t vacc01xCD = vdupq_n_s32(0);
      int32x4_t vacc01xEF = vdupq_n_s32(0);
      int32x4_t vacc23x01 = vdupq_n_s32(0);
      int32x4_t vacc23x23 = vdupq_n_s32(0);
      int32x4_t vacc23x45 = vdupq_n_s32(0);
      int32x4_t vacc23x67 = vdupq_n_s32(0);
      int32x4_t vacc23x89 = vdupq_n_s32(0);
      int32x4_t vacc23xAB = vdupq_n_s32(0);
      int32x4_t vacc23xCD = vdupq_n_s32(0);
      int32x4_t vacc23xEF = vdupq_n_s32(0);
      int32x4_t vacc45x01 = vdupq_n_s32(0);
      int32x4_t vacc45x23 = vdupq_n_s32(0);
      int32x4_t vacc45x45 = vdupq_n_s32(0);
      int32x4_t vacc45x67 = vdupq_n_s32(0);
      int32x4_t vacc45x89 = vdupq_n_s32(0);
      int32x4_t vacc45xAB = vdupq_n_s32(0);
      int32x4_t vacc45xCD = vdupq_n_s32(0);
      int32x4_t vacc45xEF = vdupq_n_s32(0);

      size_t k = bl;
      // 2x partial unrolled loop to load 8 bytes at a time.

      uint64x2x2_t va01x0123456789ABCDEF;
      va01x0123456789ABCDEF.val[0] = vdupq_n_u64(0);
      va01x0123456789ABCDEF.val[1] = vdupq_n_u64(0);
      uint64x2x2_t va23x0123456789ABCDEF;
      va23x0123456789ABCDEF.val[0] = vdupq_n_u64(0);
      va23x0123456789ABCDEF.val[1] = vdupq_n_u64(0);
      uint64x2x2_t va45x0123456789ABCDEF;
      va45x0123456789ABCDEF.val[0] = vdupq_n_u64(0);
      va45x0123456789ABCDEF.val[1] = vdupq_n_u64(0);

      while (k >= 16 * sizeof(int8_t)) {
        // Load a 5x16 block of activations.
        #if XNN_ARCH_ARM64
          va01x0123456789ABCDEF = vld2q_lane_u64((const void*) a0, va01x0123456789ABCDEF, 0); a0 += 16;
          va23x0123456789ABCDEF = vld2q_lane_u64((const void*) a2, va23x0123456789ABCDEF, 0); a2 += 16;
          va45x0123456789ABCDEF = vld2q_lane_u64((const void*) a4, va45x0123456789ABCDEF, 0); a4 += 16;
          va01x0123456789ABCDEF = vld2q_lane_u64((const void*) a1, va01x0123456789ABCDEF, 1); a1 += 16;
          va23x0123456789ABCDEF = vld2q_lane_u64((const void*) a3, va23x0123456789ABCDEF, 1); a3 += 16;
        #else
          va01x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a0, va01x0123456789ABCDEF.val[0], 0); a0 += 8;
          va01x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a0, va01x0123456789ABCDEF.val[1], 0); a0 += 8;
          va23x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a2, va23x0123456789ABCDEF.val[0], 0); a2 += 8;
          va23x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a2, va23x0123456789ABCDEF.val[1], 0); a2 += 8;
          va45x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a4, va45x0123456789ABCDEF.val[0], 0); a4 += 8;
          va45x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a4, va45x0123456789ABCDEF.val[1], 0); a4 += 8;
          va01x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a1, va01x0123456789ABCDEF.val[0], 1); a1 += 8;
          va01x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a1, va01x0123456789ABCDEF.val[1], 1); a1 += 8;
          va23x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a3, va23x0123456789ABCDEF.val[0], 1); a3 += 8;
          va23x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a3, va23x0123456789ABCDEF.val[1], 1); a3 += 8;
        #endif

        const int8x16_t va01x01234567 = vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]);
        const int8x16_t va01x89ABCDEF = vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]);
        const int8x16_t va23x01234567 = vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[0]);
        const int8x16_t va23x89ABCDEF = vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[1]);
        const int8x16_t va45x01234567 = vreinterpretq_s8_u64(va45x0123456789ABCDEF.val[0]);
        const int8x16_t va45x89ABCDEF = vreinterpretq_s8_u64(va45x0123456789ABCDEF.val[1]);

        // Load a 16x16 block of weights.
        const int8x16_t vb01x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb23x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb45x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb67x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb89x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbABx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbCDx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbEFx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb01x01234567 = vshlq_n_s8(vb01x0123456789ABCDEF, 4);
        const int8x16_t vb23x01234567 = vshlq_n_s8(vb23x0123456789ABCDEF, 4);
        const int8x16_t vb45x01234567 = vshlq_n_s8(vb45x0123456789ABCDEF, 4);
        const int8x16_t vb67x01234567 = vshlq_n_s8(vb67x0123456789ABCDEF, 4);
        const int8x16_t vb89x01234567 = vshlq_n_s8(vb89x0123456789ABCDEF, 4);
        const int8x16_t vbABx01234567 = vshlq_n_s8(vbABx0123456789ABCDEF, 4);
        const int8x16_t vbCDx01234567 = vshlq_n_s8(vbCDx0123456789ABCDEF, 4);
        const int8x16_t vbEFx01234567 = vshlq_n_s8(vbEFx0123456789ABCDEF, 4);
        const int8x16_t vb01x89ABCDEF = vandq_s8(vb01x0123456789ABCDEF, vmask);
        const int8x16_t vb23x89ABCDEF = vandq_s8(vb23x0123456789ABCDEF, vmask);
        const int8x16_t vb45x89ABCDEF = vandq_s8(vb45x0123456789ABCDEF, vmask);
        const int8x16_t vb67x89ABCDEF = vandq_s8(vb67x0123456789ABCDEF, vmask);
        const int8x16_t vb89x89ABCDEF = vandq_s8(vb89x0123456789ABCDEF, vmask);
        const int8x16_t vbABx89ABCDEF = vandq_s8(vbABx0123456789ABCDEF, vmask);
        const int8x16_t vbCDx89ABCDEF = vandq_s8(vbCDx0123456789ABCDEF, vmask);
        const int8x16_t vbEFx89ABCDEF = vandq_s8(vbEFx0123456789ABCDEF, vmask);

        // Multiply-accumulate: 5x8 * 8x16 --> 5x16.
        vacc01x01 = vmmlaq_s32(vacc01x01, va01x01234567, vb01x01234567);
        vacc01x23 = vmmlaq_s32(vacc01x23, va01x01234567, vb23x01234567);
        vacc01x45 = vmmlaq_s32(vacc01x45, va01x01234567, vb45x01234567);
        vacc01x67 = vmmlaq_s32(vacc01x67, va01x01234567, vb67x01234567);
        vacc01x89 = vmmlaq_s32(vacc01x89, va01x01234567, vb89x01234567);
        vacc01xAB = vmmlaq_s32(vacc01xAB, va01x01234567, vbABx01234567);
        vacc01xCD = vmmlaq_s32(vacc01xCD, va01x01234567, vbCDx01234567);
        vacc01xEF = vmmlaq_s32(vacc01xEF, va01x01234567, vbEFx01234567);
        vacc23x01 = vmmlaq_s32(vacc23x01, va23x01234567, vb01x01234567);
        vacc23x23 = vmmlaq_s32(vacc23x23, va23x01234567, vb23x01234567);
        vacc23x45 = vmmlaq_s32(vacc23x45, va23x01234567, vb45x01234567);
        vacc23x67 = vmmlaq_s32(vacc23x67, va23x01234567, vb67x01234567);
        vacc23x89 = vmmlaq_s32(vacc23x89, va23x01234567, vb89x01234567);
        vacc23xAB = vmmlaq_s32(vacc23xAB, va23x01234567, vbABx01234567);
        vacc23xCD = vmmlaq_s32(vacc23xCD, va23x01234567, vbCDx01234567);
        vacc23xEF = vmmlaq_s32(vacc23xEF, va23x01234567, vbEFx01234567);
        vacc45x01 = vmmlaq_s32(vacc45x01, va45x01234567, vb01x01234567);
        vacc45x23 = vmmlaq_s32(vacc45x23, va45x01234567, vb23x01234567);
        vacc45x45 = vmmlaq_s32(vacc45x45, va45x01234567, vb45x01234567);
        vacc45x67 = vmmlaq_s32(vacc45x67, va45x01234567, vb67x01234567);
        vacc45x89 = vmmlaq_s32(vacc45x89, va45x01234567, vb89x01234567);
        vacc45xAB = vmmlaq_s32(vacc45xAB, va45x01234567, vbABx01234567);
        vacc45xCD = vmmlaq_s32(vacc45xCD, va45x01234567, vbCDx01234567);
        vacc45xEF = vmmlaq_s32(vacc45xEF, va45x01234567, vbEFx01234567);
        vacc01x01 = vmmlaq_s32(vacc01x01, va01x89ABCDEF, vb01x89ABCDEF);
        vacc01x23 = vmmlaq_s32(vacc01x23, va01x89ABCDEF, vb23x89ABCDEF);
        vacc01x45 = vmmlaq_s32(vacc01x45, va01x89ABCDEF, vb45x89ABCDEF);
        vacc01x67 = vmmlaq_s32(vacc01x67, va01x89ABCDEF, vb67x89ABCDEF);
        vacc01x89 = vmmlaq_s32(vacc01x89, va01x89ABCDEF, vb89x89ABCDEF);
        vacc01xAB = vmmlaq_s32(vacc01xAB, va01x89ABCDEF, vbABx89ABCDEF);
        vacc01xCD = vmmlaq_s32(vacc01xCD, va01x89ABCDEF, vbCDx89ABCDEF);
        vacc01xEF = vmmlaq_s32(vacc01xEF, va01x89ABCDEF, vbEFx89ABCDEF);
        vacc23x01 = vmmlaq_s32(vacc23x01, va23x89ABCDEF, vb01x89ABCDEF);
        vacc23x23 = vmmlaq_s32(vacc23x23, va23x89ABCDEF, vb23x89ABCDEF);
        vacc23x45 = vmmlaq_s32(vacc23x45, va23x89ABCDEF, vb45x89ABCDEF);
        vacc23x67 = vmmlaq_s32(vacc23x67, va23x89ABCDEF, vb67x89ABCDEF);
        vacc23x89 = vmmlaq_s32(vacc23x89, va23x89ABCDEF, vb89x89ABCDEF);
        vacc23xAB = vmmlaq_s32(vacc23xAB, va23x89ABCDEF, vbABx89ABCDEF);
        vacc23xCD = vmmlaq_s32(vacc23xCD, va23x89ABCDEF, vbCDx89ABCDEF);
        vacc23xEF = vmmlaq_s32(vacc23xEF, va23x89ABCDEF, vbEFx89ABCDEF);
        vacc45x01 = vmmlaq_s32(vacc45x01, va45x89ABCDEF, vb01x89ABCDEF);
        vacc45x23 = vmmlaq_s32(vacc45x23, va45x89ABCDEF, vb23x89ABCDEF);
        vacc45x45 = vmmlaq_s32(vacc45x45, va45x89ABCDEF, vb45x89ABCDEF);
        vacc45x67 = vmmlaq_s32(vacc45x67, va45x89ABCDEF, vb67x89ABCDEF);
        vacc45x89 = vmmlaq_s32(vacc45x89, va45x89ABCDEF, vb89x89ABCDEF);
        vacc45xAB = vmmlaq_s32(vacc45xAB, va45x89ABCDEF, vbABx89ABCDEF);
        vacc45xCD = vmmlaq_s32(vacc45xCD, va45x89ABCDEF, vbCDx89ABCDEF);
        vacc45xEF = vmmlaq_s32(vacc45xEF, va45x89ABCDEF, vbEFx89ABCDEF);

        k -= 16 * sizeof(int8_t);
      }
      // Handle up to 8 final positions of `k`
      if XNN_UNLIKELY(k != 0) {
        // Load a 5x8 block of activations.
        uint64x2_t va01x01234567 = vld1q_dup_u64((const void*) a0); a0 += 8;
        uint64x2_t va23x01234567 = vld1q_dup_u64((const void*) a2); a2 += 8;
        uint64x2_t va45x01234567 = vld1q_dup_u64((const void*) a4); a4 += 8;
        va01x01234567 = vld1q_lane_u64((const void*) a1, va01x01234567, 1); a1 += 8;
        va23x01234567 = vld1q_lane_u64((const void*) a3, va23x01234567, 1); a3 += 8;

        // Load a 16x16 block of weights.
        const int8x16_t vb01x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb23x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb45x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb67x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb89x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbABx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbCDx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbEFx0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb01x01234567 = vshlq_n_s8(vb01x0123456789ABCDEF, 4);
        const int8x16_t vb23x01234567 = vshlq_n_s8(vb23x0123456789ABCDEF, 4);
        const int8x16_t vb45x01234567 = vshlq_n_s8(vb45x0123456789ABCDEF, 4);
        const int8x16_t vb67x01234567 = vshlq_n_s8(vb67x0123456789ABCDEF, 4);
        const int8x16_t vb89x01234567 = vshlq_n_s8(vb89x0123456789ABCDEF, 4);
        const int8x16_t vbABx01234567 = vshlq_n_s8(vbABx0123456789ABCDEF, 4);
        const int8x16_t vbCDx01234567 = vshlq_n_s8(vbCDx0123456789ABCDEF, 4);
        const int8x16_t vbEFx01234567 = vshlq_n_s8(vbEFx0123456789ABCDEF, 4);

        // Multiply-accumulate: 5x4 * 4x16 --> 5x16.
        vacc01x01 = vmmlaq_s32(vacc01x01, vreinterpretq_s8_u64(va01x01234567), vb01x01234567);
        vacc01x23 = vmmlaq_s32(vacc01x23, vreinterpretq_s8_u64(va01x01234567), vb23x01234567);
        vacc01x45 = vmmlaq_s32(vacc01x45, vreinterpretq_s8_u64(va01x01234567), vb45x01234567);
        vacc01x67 = vmmlaq_s32(vacc01x67, vreinterpretq_s8_u64(va01x01234567), vb67x01234567);
        vacc01x89 = vmmlaq_s32(vacc01x89, vreinterpretq_s8_u64(va01x01234567), vb89x01234567);
        vacc01xAB = vmmlaq_s32(vacc01xAB, vreinterpretq_s8_u64(va01x01234567), vbABx01234567);
        vacc01xCD = vmmlaq_s32(vacc01xCD, vreinterpretq_s8_u64(va01x01234567), vbCDx01234567);
        vacc01xEF = vmmlaq_s32(vacc01xEF, vreinterpretq_s8_u64(va01x01234567), vbEFx01234567);
        vacc23x01 = vmmlaq_s32(vacc23x01, vreinterpretq_s8_u64(va23x01234567), vb01x01234567);
        vacc23x23 = vmmlaq_s32(vacc23x23, vreinterpretq_s8_u64(va23x01234567), vb23x01234567);
        vacc23x45 = vmmlaq_s32(vacc23x45, vreinterpretq_s8_u64(va23x01234567), vb45x01234567);
        vacc23x67 = vmmlaq_s32(vacc23x67, vreinterpretq_s8_u64(va23x01234567), vb67x01234567);
        vacc23x89 = vmmlaq_s32(vacc23x89, vreinterpretq_s8_u64(va23x01234567), vb89x01234567);
        vacc23xAB = vmmlaq_s32(vacc23xAB, vreinterpretq_s8_u64(va23x01234567), vbABx01234567);
        vacc23xCD = vmmlaq_s32(vacc23xCD, vreinterpretq_s8_u64(va23x01234567), vbCDx01234567);
        vacc23xEF = vmmlaq_s32(vacc23xEF, vreinterpretq_s8_u64(va23x01234567), vbEFx01234567);
        vacc45x01 = vmmlaq_s32(vacc45x01, vreinterpretq_s8_u64(va45x01234567), vb01x01234567);
        vacc45x23 = vmmlaq_s32(vacc45x23, vreinterpretq_s8_u64(va45x01234567), vb23x01234567);
        vacc45x45 = vmmlaq_s32(vacc45x45, vreinterpretq_s8_u64(va45x01234567), vb45x01234567);
        vacc45x67 = vmmlaq_s32(vacc45x67, vreinterpretq_s8_u64(va45x01234567), vb67x01234567);
        vacc45x89 = vmmlaq_s32(vacc45x89, vreinterpretq_s8_u64(va45x01234567), vb89x01234567);
        vacc45xAB = vmmlaq_s32(vacc45xAB, vreinterpretq_s8_u64(va45x01234567), vbABx01234567);
        vacc45xCD = vmmlaq_s32(vacc45xCD, vreinterpretq_s8_u64(va45x01234567), vbCDx01234567);
        vacc45xEF = vmmlaq_s32(vacc45xEF, vreinterpretq_s8_u64(va45x01234567), vbEFx01234567);
      }

      #if XNN_ARCH_ARM64
        int32x4_t vacc0x0123 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x01), vreinterpretq_u64_s32(vacc01x23)));
        int32x4_t vacc1x0123 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01x01), vreinterpretq_u64_s32(vacc01x23)));
        int32x4_t vacc0x4567 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x45), vreinterpretq_u64_s32(vacc01x67)));
        int32x4_t vacc1x4567 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01x45), vreinterpretq_u64_s32(vacc01x67)));
        int32x4_t vacc0x89AB = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x89), vreinterpretq_u64_s32(vacc01xAB)));
        int32x4_t vacc1x89AB = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01x89), vreinterpretq_u64_s32(vacc01xAB)));
        int32x4_t vacc0xCDEF = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01xCD), vreinterpretq_u64_s32(vacc01xEF)));
        int32x4_t vacc1xCDEF = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01xCD), vreinterpretq_u64_s32(vacc01xEF)));
        int32x4_t vacc2x0123 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc23x01), vreinterpretq_u64_s32(vacc23x23)));
        int32x4_t vacc3x0123 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc23x01), vreinterpretq_u64_s32(vacc23x23)));
        int32x4_t vacc2x4567 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc23x45), vreinterpretq_u64_s32(vacc23x67)));
        int32x4_t vacc3x4567 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc23x45), vreinterpretq_u64_s32(vacc23x67)));
        int32x4_t vacc2x89AB = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc23x89), vreinterpretq_u64_s32(vacc23xAB)));
        int32x4_t vacc3x89AB = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc23x89), vreinterpretq_u64_s32(vacc23xAB)));
        int32x4_t vacc2xCDEF = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc23xCD), vreinterpretq_u64_s32(vacc23xEF)));
        int32x4_t vacc3xCDEF = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc23xCD), vreinterpretq_u64_s32(vacc23xEF)));
        int32x4_t vacc4x0123 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc45x01), vreinterpretq_u64_s32(vacc45x23)));
        int32x4_t vacc4x4567 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc45x45), vreinterpretq_u64_s32(vacc45x67)));
        int32x4_t vacc4x89AB = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc45x89), vreinterpretq_u64_s32(vacc45xAB)));
        int32x4_t vacc4xCDEF = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc45xCD), vreinterpretq_u64_s32(vacc45xEF)));
      #else
        int32x4_t vacc0x0123 = vcombine_s32(vget_low_s32(vacc01x01), vget_low_s32(vacc01x23));
        int32x4_t vacc1x0123 = vcombine_s32(vget_high_s32(vacc01x01), vget_high_s32(vacc01x23));
        int32x4_t vacc0x4567 = vcombine_s32(vget_low_s32(vacc01x45), vget_low_s32(vacc01x67));
        int32x4_t vacc1x4567 = vcombine_s32(vget_high_s32(vacc01x45), vget_high_s32(vacc01x67));
        int32x4_t vacc0x89AB = vcombine_s32(vget_low_s32(vacc01x89), vget_low_s32(vacc01xAB));
        int32x4_t vacc1x89AB = vcombine_s32(vget_high_s32(vacc01x89), vget_high_s32(vacc01xAB));
        int32x4_t vacc0xCDEF = vcombine_s32(vget_low_s32(vacc01xCD), vget_low_s32(vacc01xEF));
        int32x4_t vacc1xCDEF = vcombine_s32(vget_high_s32(vacc01xCD), vget_high_s32(vacc01xEF));
        int32x4_t vacc2x0123 = vcombine_s32(vget_low_s32(vacc23x01), vget_low_s32(vacc23x23));
        int32x4_t vacc3x0123 = vcombine_s32(vget_high_s32(vacc23x01), vget_high_s32(vacc23x23));
        int32x4_t vacc2x4567 = vcombine_s32(vget_low_s32(vacc23x45), vget_low_s32(vacc23x67));
        int32x4_t vacc3x4567 = vcombine_s32(vget_high_s32(vacc23x45), vget_high_s32(vacc23x67));
        int32x4_t vacc2x89AB = vcombine_s32(vget_low_s32(vacc23x89), vget_low_s32(vacc23xAB));
        int32x4_t vacc3x89AB = vcombine_s32(vget_high_s32(vacc23x89), vget_high_s32(vacc23xAB));
        int32x4_t vacc2xCDEF = vcombine_s32(vget_low_s32(vacc23xCD), vget_low_s32(vacc23xEF));
        int32x4_t vacc3xCDEF = vcombine_s32(vget_high_s32(vacc23xCD), vget_high_s32(vacc23xEF));
        int32x4_t vacc4x0123 = vcombine_s32(vget_low_s32(vacc45x01), vget_low_s32(vacc45x23));
        int32x4_t vacc4x4567 = vcombine_s32(vget_low_s32(vacc45x45), vget_low_s32(vacc45x67));
        int32x4_t vacc4x89AB = vcombine_s32(vget_low_s32(vacc45x89), vget_low_s32(vacc45xAB));
        int32x4_t vacc4xCDEF = vcombine_s32(vget_low_s32(vacc45xCD), vget_low_s32(vacc45xEF));
      #endif
      const float32x4_t vfilter_output_scale0123 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;
      const float32x4_t vfilter_output_scale4567 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;
      const float32x4_t vfilter_output_scale89AB = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;
      const float32x4_t vfilter_output_scaleCDEF = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;
      float32x4_t vf0x0123 = vcvtq_f32_s32(vacc0x0123);
      vout0x0123 = vfmaq_f32(vout0x0123, vf0x0123, vfilter_output_scale0123);
      float32x4_t vf0x4567 = vcvtq_f32_s32(vacc0x4567);
      vout0x4567 = vfmaq_f32(vout0x4567, vf0x4567, vfilter_output_scale4567);
      float32x4_t vf0x89AB = vcvtq_f32_s32(vacc0x89AB);
      vout0x89AB = vfmaq_f32(vout0x89AB, vf0x89AB, vfilter_output_scale89AB);
      float32x4_t vf0xCDEF = vcvtq_f32_s32(vacc0xCDEF);
      vout0xCDEF = vfmaq_f32(vout0xCDEF, vf0xCDEF, vfilter_output_scaleCDEF);
      float32x4_t vf1x0123 = vcvtq_f32_s32(vacc1x0123);
      vout1x0123 = vfmaq_f32(vout1x0123, vf1x0123, vfilter_output_scale0123);
      float32x4_t vf1x4567 = vcvtq_f32_s32(vacc1x4567);
      vout1x4567 = vfmaq_f32(vout1x4567, vf1x4567, vfilter_output_scale4567);
      float32x4_t vf1x89AB = vcvtq_f32_s32(vacc1x89AB);
      vout1x89AB = vfmaq_f32(vout1x89AB, vf1x89AB, vfilter_output_scale89AB);
      float32x4_t vf1xCDEF = vcvtq_f32_s32(vacc1xCDEF);
      vout1xCDEF = vfmaq_f32(vout1xCDEF, vf1xCDEF, vfilter_output_scaleCDEF);
      float32x4_t vf2x0123 = vcvtq_f32_s32(vacc2x0123);
      vout2x0123 = vfmaq_f32(vout2x0123, vf2x0123, vfilter_output_scale0123);
      float32x4_t vf2x4567 = vcvtq_f32_s32(vacc2x4567);
      vout2x4567 = vfmaq_f32(vout2x4567, vf2x4567, vfilter_output_scale4567);
      float32x4_t vf2x89AB = vcvtq_f32_s32(vacc2x89AB);
      vout2x89AB = vfmaq_f32(vout2x89AB, vf2x89AB, vfilter_output_scale89AB);
      float32x4_t vf2xCDEF = vcvtq_f32_s32(vacc2xCDEF);
      vout2xCDEF = vfmaq_f32(vout2xCDEF, vf2xCDEF, vfilter_output_scaleCDEF);
      float32x4_t vf3x0123 = vcvtq_f32_s32(vacc3x0123);
      vout3x0123 = vfmaq_f32(vout3x0123, vf3x0123, vfilter_output_scale0123);
      float32x4_t vf3x4567 = vcvtq_f32_s32(vacc3x4567);
      vout3x4567 = vfmaq_f32(vout3x4567, vf3x4567, vfilter_output_scale4567);
      float32x4_t vf3x89AB = vcvtq_f32_s32(vacc3x89AB);
      vout3x89AB = vfmaq_f32(vout3x89AB, vf3x89AB, vfilter_output_scale89AB);
      float32x4_t vf3xCDEF = vcvtq_f32_s32(vacc3xCDEF);
      vout3xCDEF = vfmaq_f32(vout3xCDEF, vf3xCDEF, vfilter_output_scaleCDEF);
      float32x4_t vf4x0123 = vcvtq_f32_s32(vacc4x0123);
      vout4x0123 = vfmaq_f32(vout4x0123, vf4x0123, vfilter_output_scale0123);
      float32x4_t vf4x4567 = vcvtq_f32_s32(vacc4x4567);
      vout4x4567 = vfmaq_f32(vout4x4567, vf4x4567, vfilter_output_scale4567);
      float32x4_t vf4x89AB = vcvtq_f32_s32(vacc4x89AB);
      vout4x89AB = vfmaq_f32(vout4x89AB, vf4x89AB, vfilter_output_scale89AB);
      float32x4_t vf4xCDEF = vcvtq_f32_s32(vacc4xCDEF);
      vout4xCDEF = vfmaq_f32(vout4xCDEF, vf4xCDEF, vfilter_output_scaleCDEF);
    }

    const float32x4_t vinput_scale01 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[0].zero_point));
    vout0x0123 = vmulq_lane_f32(vout0x0123, vget_low_f32(vinput_scale01), 1);
    vout1x0123 = vmulq_lane_f32(vout1x0123, vget_high_f32(vinput_scale01), 1);
    vout0x4567 = vmulq_lane_f32(vout0x4567, vget_low_f32(vinput_scale01), 1);
    vout1x4567 = vmulq_lane_f32(vout1x4567, vget_high_f32(vinput_scale01), 1);
    vout0x89AB = vmulq_lane_f32(vout0x89AB, vget_low_f32(vinput_scale01), 1);
    vout1x89AB = vmulq_lane_f32(vout1x89AB, vget_high_f32(vinput_scale01), 1);
    vout0xCDEF = vmulq_lane_f32(vout0xCDEF, vget_low_f32(vinput_scale01), 1);
    vout1xCDEF = vmulq_lane_f32(vout1xCDEF, vget_high_f32(vinput_scale01), 1);
    const float32x4_t vinput_scale23 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[2].zero_point));
    vout2x0123 = vmulq_lane_f32(vout2x0123, vget_low_f32(vinput_scale23), 1);
    vout3x0123 = vmulq_lane_f32(vout3x0123, vget_high_f32(vinput_scale23), 1);
    vout2x4567 = vmulq_lane_f32(vout2x4567, vget_low_f32(vinput_scale23), 1);
    vout3x4567 = vmulq_lane_f32(vout3x4567, vget_high_f32(vinput_scale23), 1);
    vout2x89AB = vmulq_lane_f32(vout2x89AB, vget_low_f32(vinput_scale23), 1);
    vout3x89AB = vmulq_lane_f32(vout3x89AB, vget_high_f32(vinput_scale23), 1);
    vout2xCDEF = vmulq_lane_f32(vout2xCDEF, vget_low_f32(vinput_scale23), 1);
    vout3xCDEF = vmulq_lane_f32(vout3xCDEF, vget_high_f32(vinput_scale23), 1);
    const float32x4_t vinput_scale4 = vld1q_dup_f32(&quantization_params[4].inv_scale);
    vout4x0123 = vmulq_f32(vout4x0123, vinput_scale4);
    vout4x4567 = vmulq_f32(vout4x4567, vinput_scale4);
    vout4x89AB = vmulq_f32(vout4x89AB, vinput_scale4);
    vout4xCDEF = vmulq_f32(vout4xCDEF, vinput_scale4);


    #if XNN_ARCH_ARM64
      const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
      vout0x0123 = vaddq_f32(vbias0123, vout0x0123);
      vout1x0123 = vaddq_f32(vbias0123, vout1x0123);
      vout2x0123 = vaddq_f32(vbias0123, vout2x0123);
      vout3x0123 = vaddq_f32(vbias0123, vout3x0123);
      vout4x0123 = vaddq_f32(vbias0123, vout4x0123);
      const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
      vout0x4567 = vaddq_f32(vbias4567, vout0x4567);
      vout1x4567 = vaddq_f32(vbias4567, vout1x4567);
      vout2x4567 = vaddq_f32(vbias4567, vout2x4567);
      vout3x4567 = vaddq_f32(vbias4567, vout3x4567);
      vout4x4567 = vaddq_f32(vbias4567, vout4x4567);
      const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
      vout0x89AB = vaddq_f32(vbias89AB, vout0x89AB);
      vout1x89AB = vaddq_f32(vbias89AB, vout1x89AB);
      vout2x89AB = vaddq_f32(vbias89AB, vout2x89AB);
      vout3x89AB = vaddq_f32(vbias89AB, vout3x89AB);
      vout4x89AB = vaddq_f32(vbias89AB, vout4x89AB);
      const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
      vout0xCDEF = vaddq_f32(vbiasCDEF, vout0xCDEF);
      vout1xCDEF = vaddq_f32(vbiasCDEF, vout1xCDEF);
      vout2xCDEF = vaddq_f32(vbiasCDEF, vout2xCDEF);
      vout3xCDEF = vaddq_f32(vbiasCDEF, vout3xCDEF);
      vout4xCDEF = vaddq_f32(vbiasCDEF, vout4xCDEF);
    #else
      const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
      vout0x0123 = vaddq_f32(vbias0123, vout0x0123);
      vout1x0123 = vaddq_f32(vbias0123, vout1x0123);
      vout2x0123 = vaddq_f32(vbias0123, vout2x0123);
      vout3x0123 = vaddq_f32(vbias0123, vout3x0123);
      vout4x0123 = vaddq_f32(vbias0123, vout4x0123);
      const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
      vout0x4567 = vaddq_f32(vbias4567, vout0x4567);
      vout1x4567 = vaddq_f32(vbias4567, vout1x4567);
      vout2x4567 = vaddq_f32(vbias4567, vout2x4567);
      vout3x4567 = vaddq_f32(vbias4567, vout3x4567);
      vout4x4567 = vaddq_f32(vbias4567, vout4x4567);
      const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
      vout0x89AB = vaddq_f32(vbias89AB, vout0x89AB);
      vout1x89AB = vaddq_f32(vbias89AB, vout1x89AB);
      vout2x89AB = vaddq_f32(vbias89AB, vout2x89AB);
      vout3x89AB = vaddq_f32(vbias89AB, vout3x89AB);
      vout4x89AB = vaddq_f32(vbias89AB, vout4x89AB);
      const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
      vout0xCDEF = vaddq_f32(vbiasCDEF, vout0xCDEF);
      vout1xCDEF = vaddq_f32(vbiasCDEF, vout1xCDEF);
      vout2xCDEF = vaddq_f32(vbiasCDEF, vout2xCDEF);
      vout3xCDEF = vaddq_f32(vbiasCDEF, vout3xCDEF);
      vout4xCDEF = vaddq_f32(vbiasCDEF, vout4xCDEF);
    #endif

    const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, voutput_min);
    vout0x4567 = vmaxq_f32(vout0x4567, voutput_min);
    vout0x89AB = vmaxq_f32(vout0x89AB, voutput_min);
    vout0xCDEF = vmaxq_f32(vout0xCDEF, voutput_min);
    vout1x0123 = vmaxq_f32(vout1x0123, voutput_min);
    vout1x4567 = vmaxq_f32(vout1x4567, voutput_min);
    vout1x89AB = vmaxq_f32(vout1x89AB, voutput_min);
    vout1xCDEF = vmaxq_f32(vout1xCDEF, voutput_min);
    vout2x0123 = vmaxq_f32(vout2x0123, voutput_min);
    vout2x4567 = vmaxq_f32(vout2x4567, voutput_min);
    vout2x89AB = vmaxq_f32(vout2x89AB, voutput_min);
    vout2xCDEF = vmaxq_f32(vout2xCDEF, voutput_min);
    vout3x0123 = vmaxq_f32(vout3x0123, voutput_min);
    vout3x4567 = vmaxq_f32(vout3x4567, voutput_min);
    vout3x89AB = vmaxq_f32(vout3x89AB, voutput_min);
    vout3xCDEF = vmaxq_f32(vout3xCDEF, voutput_min);
    vout4x0123 = vmaxq_f32(vout4x0123, voutput_min);
    vout4x4567 = vmaxq_f32(vout4x4567, voutput_min);
    vout4x89AB = vmaxq_f32(vout4x89AB, voutput_min);
    vout4xCDEF = vmaxq_f32(vout4xCDEF, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);
    vout0x89AB = vminq_f32(vout0x89AB, voutput_max);
    vout0xCDEF = vminq_f32(vout0xCDEF, voutput_max);
    vout1x0123 = vminq_f32(vout1x0123, voutput_max);
    vout1x4567 = vminq_f32(vout1x4567, voutput_max);
    vout1x89AB = vminq_f32(vout1x89AB, voutput_max);
    vout1xCDEF = vminq_f32(vout1xCDEF, voutput_max);
    vout2x0123 = vminq_f32(vout2x0123, voutput_max);
    vout2x4567 = vminq_f32(vout2x4567, voutput_max);
    vout2x89AB = vminq_f32(vout2x89AB, voutput_max);
    vout2xCDEF = vminq_f32(vout2xCDEF, voutput_max);
    vout3x0123 = vminq_f32(vout3x0123, voutput_max);
    vout3x4567 = vminq_f32(vout3x4567, voutput_max);
    vout3x89AB = vminq_f32(vout3x89AB, voutput_max);
    vout3xCDEF = vminq_f32(vout3xCDEF, voutput_max);
    vout4x0123 = vminq_f32(vout4x0123, voutput_max);
    vout4x4567 = vminq_f32(vout4x4567, voutput_max);
    vout4x89AB = vminq_f32(vout4x89AB, voutput_max);
    vout4xCDEF = vminq_f32(vout4xCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(c0, vout0x0123);
      vst1q_f32(c0 + 4, vout0x4567);
      vst1q_f32(c0 + 8, vout0x89AB);
      vst1q_f32(c0 + 12, vout0xCDEF);
      vst1q_f32(c1, vout1x0123);
      vst1q_f32(c1 + 4, vout1x4567);
      vst1q_f32(c1 + 8, vout1x89AB);
      vst1q_f32(c1 + 12, vout1xCDEF);
      vst1q_f32(c2, vout2x0123);
      vst1q_f32(c2 + 4, vout2x4567);
      vst1q_f32(c2 + 8, vout2x89AB);
      vst1q_f32(c2 + 12, vout2xCDEF);
      vst1q_f32(c3, vout3x0123);
      vst1q_f32(c3 + 4, vout3x4567);
      vst1q_f32(c3 + 8, vout3x89AB);
      vst1q_f32(c3 + 12, vout3xCDEF);
      vst1q_f32(c4, vout4x0123);
      vst1q_f32(c4 + 4, vout4x4567);
      vst1q_f32(c4 + 8, vout4x89AB);
      vst1q_f32(c4 + 12, vout4xCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);
      a4 = (const int8_t*) ((uintptr_t) a4 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);

      nc -= 16;
    } else {
     if (nc & 8) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x89AB;
       vst1q_f32(c1, vout1x0123); c1 += 4;
       vout1x0123 = vout1x89AB;
       vst1q_f32(c2, vout2x0123); c2 += 4;
       vout2x0123 = vout2x89AB;
       vst1q_f32(c3, vout3x0123); c3 += 4;
       vout3x0123 = vout3x89AB;
       vst1q_f32(c4, vout4x0123); c4 += 4;
       vout4x0123 = vout4x89AB;
       vst1q_f32(c0, vout0x4567); c0 += 4;
       vout0x4567 = vout0xCDEF;
       vst1q_f32(c1, vout1x4567); c1 += 4;
       vout1x4567 = vout1xCDEF;
       vst1q_f32(c2, vout2x4567); c2 += 4;
       vout2x4567 = vout2xCDEF;
       vst1q_f32(c3, vout3x4567); c3 += 4;
       vout3x4567 = vout3xCDEF;
       vst1q_f32(c4, vout4x4567); c4 += 4;
       vout4x4567 = vout4xCDEF;
     }
     if (nc & 4) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x4567;
       vst1q_f32(c1, vout1x0123); c1 += 4;
       vout1x0123 = vout1x4567;
       vst1q_f32(c2, vout2x0123); c2 += 4;
       vout2x0123 = vout2x4567;
       vst1q_f32(c3, vout3x0123); c3 += 4;
       vout3x0123 = vout3x4567;
       vst1q_f32(c4, vout4x0123); c4 += 4;
       vout4x0123 = vout4x4567;
     }
     float32x2_t vout0x01 = vget_low_f32(vout0x0123);
     float32x2_t vout1x01 = vget_low_f32(vout1x0123);
     float32x2_t vout2x01 = vget_low_f32(vout2x0123);
     float32x2_t vout3x01 = vget_low_f32(vout3x0123);
     float32x2_t vout4x01 = vget_low_f32(vout4x0123);
     if (nc & 2) {
       vst1_f32(c0, vout0x01); c0 += 2;
       vst1_f32(c1, vout1x01); c1 += 2;
       vst1_f32(c2, vout2x01); c2 += 2;
       vst1_f32(c3, vout3x01); c3 += 2;
       vst1_f32(c4, vout4x01); c4 += 2;
       vout0x01 = vget_high_f32(vout0x0123);
       vout1x01 = vget_high_f32(vout1x0123);
       vout2x01 = vget_high_f32(vout2x0123);
       vout3x01 = vget_high_f32(vout3x0123);
       vout4x01 = vget_high_f32(vout4x0123);
     }
     if (nc & 1) {
       vst1_lane_f32(c0, vout0x01, 0);
       vst1_lane_f32(c1, vout1x01, 0);
       vst1_lane_f32(c2, vout2x01, 0);
       vst1_lane_f32(c3, vout3x01, 0);
       vst1_lane_f32(c4, vout4x01, 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}
