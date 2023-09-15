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
#include <xnnpack/math.h>


void xnn_qu8_igemm_minmax_rndnu_ukernel_8x16c8__neoni8mm(
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
  assert(mr <= 8);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (8 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(uint8_t));
  uint8_t* c0 = c;
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  uint8_t* c3 = (uint8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  uint8_t* c4 = (uint8_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  uint8_t* c5 = (uint8_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    c5 = c4;
  }
  uint8_t* c6 = (uint8_t*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    c6 = c5;
  }
  uint8_t* c7 = (uint8_t*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 8) {
    c7 = c6;
  }

  const uint8x16_t vkernel_zero_point = vreinterpretq_u8_u32(vld1q_dup_u32((const void*) params->rndnu_neon.kernel_zero_point));

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
    uint32x4_t vpacc23x01 = vreinterpretq_u32_u64(vbias01x0123.val[0]);
    uint32x4_t vpacc23x23 = vreinterpretq_u32_u64(vbias01x0123.val[1]);
    uint32x4_t vpacc23x45 = vreinterpretq_u32_u64(vbias01x4567.val[0]);
    uint32x4_t vpacc23x67 = vreinterpretq_u32_u64(vbias01x4567.val[1]);
    uint32x4_t vpacc23x89 = vreinterpretq_u32_u64(vbias01x89AB.val[0]);
    uint32x4_t vpacc23xAB = vreinterpretq_u32_u64(vbias01x89AB.val[1]);
    uint32x4_t vpacc23xCD = vreinterpretq_u32_u64(vbias01xCDEF.val[0]);
    uint32x4_t vpacc23xEF = vreinterpretq_u32_u64(vbias01xCDEF.val[1]);
    uint32x4_t vpacc45x01 = vreinterpretq_u32_u64(vbias01x0123.val[0]);
    uint32x4_t vpacc45x23 = vreinterpretq_u32_u64(vbias01x0123.val[1]);
    uint32x4_t vpacc45x45 = vreinterpretq_u32_u64(vbias01x4567.val[0]);
    uint32x4_t vpacc45x67 = vreinterpretq_u32_u64(vbias01x4567.val[1]);
    uint32x4_t vpacc45x89 = vreinterpretq_u32_u64(vbias01x89AB.val[0]);
    uint32x4_t vpacc45xAB = vreinterpretq_u32_u64(vbias01x89AB.val[1]);
    uint32x4_t vpacc45xCD = vreinterpretq_u32_u64(vbias01xCDEF.val[0]);
    uint32x4_t vpacc45xEF = vreinterpretq_u32_u64(vbias01xCDEF.val[1]);
    uint32x4_t vpacc67x01 = vreinterpretq_u32_u64(vbias01x0123.val[0]);
    uint32x4_t vpacc67x23 = vreinterpretq_u32_u64(vbias01x0123.val[1]);
    uint32x4_t vpacc67x45 = vreinterpretq_u32_u64(vbias01x4567.val[0]);
    uint32x4_t vpacc67x67 = vreinterpretq_u32_u64(vbias01x4567.val[1]);
    uint32x4_t vpacc67x89 = vreinterpretq_u32_u64(vbias01x89AB.val[0]);
    uint32x4_t vpacc67xAB = vreinterpretq_u32_u64(vbias01x89AB.val[1]);
    uint32x4_t vpacc67xCD = vreinterpretq_u32_u64(vbias01xCDEF.val[0]);
    uint32x4_t vpacc67xEF = vreinterpretq_u32_u64(vbias01xCDEF.val[1]);
    uint32x4_t vnacc01 = vmovq_n_u32(0);
    uint32x4_t vnacc23 = vmovq_n_u32(0);
    uint32x4_t vnacc45 = vmovq_n_u32(0);
    uint32x4_t vnacc67 = vmovq_n_u32(0);

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
      const uint8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const uint8_t*) ((uintptr_t) a2 + a_offset);
      }
      const uint8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const uint8_t*) ((uintptr_t) a3 + a_offset);
      }
      const uint8_t* restrict a4 = a[4];
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const uint8_t*) ((uintptr_t) a4 + a_offset);
      }
      const uint8_t* restrict a5 = a[5];
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const uint8_t*) ((uintptr_t) a5 + a_offset);
      }
      const uint8_t* restrict a6 = a[6];
      if XNN_UNPREDICTABLE(a6 != zero) {
        a6 = (const uint8_t*) ((uintptr_t) a6 + a_offset);
      }
      const uint8_t* restrict a7 = a[7];
      if XNN_UNPREDICTABLE(a7 != zero) {
        a7 = (const uint8_t*) ((uintptr_t) a7 + a_offset);
      }
      a += 8;

      uint64x2x2_t va01x0123456789ABCDEF;
      va01x0123456789ABCDEF.val[0] = vdupq_n_u64(0);
      va01x0123456789ABCDEF.val[1] = vdupq_n_u64(0);
      uint64x2x2_t va23x0123456789ABCDEF;
      va23x0123456789ABCDEF.val[0] = vdupq_n_u64(0);
      va23x0123456789ABCDEF.val[1] = vdupq_n_u64(0);
      uint64x2x2_t va45x0123456789ABCDEF;
      va45x0123456789ABCDEF.val[0] = vdupq_n_u64(0);
      va45x0123456789ABCDEF.val[1] = vdupq_n_u64(0);
      uint64x2x2_t va67x0123456789ABCDEF;
      va67x0123456789ABCDEF.val[0] = vdupq_n_u64(0);
      va67x0123456789ABCDEF.val[1] = vdupq_n_u64(0);
      // Inner accumulation loop along the 16 columns.
      size_t k = kc;
      // 2x partial unrolled loop to load 16 bytes at a time.
      while (k >= 16 * sizeof(uint8_t)) {
        // Load a 8x16 block of activations.
        #if XNN_ARCH_ARM64
          va01x0123456789ABCDEF = vld2q_lane_u64((const void*) a0, va01x0123456789ABCDEF, 0); a0 += 16;
          va23x0123456789ABCDEF = vld2q_lane_u64((const void*) a2, va23x0123456789ABCDEF, 0); a2 += 16;
          va45x0123456789ABCDEF = vld2q_lane_u64((const void*) a4, va45x0123456789ABCDEF, 0); a4 += 16;
          va67x0123456789ABCDEF = vld2q_lane_u64((const void*) a6, va67x0123456789ABCDEF, 0); a6 += 16;
          va01x0123456789ABCDEF = vld2q_lane_u64((const void*) a1, va01x0123456789ABCDEF, 1); a1 += 16;
          va23x0123456789ABCDEF = vld2q_lane_u64((const void*) a3, va23x0123456789ABCDEF, 1); a3 += 16;
          va45x0123456789ABCDEF = vld2q_lane_u64((const void*) a5, va45x0123456789ABCDEF, 1); a5 += 16;
          va67x0123456789ABCDEF = vld2q_lane_u64((const void*) a7, va67x0123456789ABCDEF, 1); a7 += 16;
        #else
          va01x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a0, va01x0123456789ABCDEF.val[0], 0); a0 += 8;
          va01x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a0, va01x0123456789ABCDEF.val[1], 0); a0 += 8;
          va23x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a2, va23x0123456789ABCDEF.val[0], 0); a2 += 8;
          va23x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a2, va23x0123456789ABCDEF.val[1], 0); a2 += 8;
          va45x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a4, va45x0123456789ABCDEF.val[0], 0); a4 += 8;
          va45x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a4, va45x0123456789ABCDEF.val[1], 0); a4 += 8;
          va67x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a6, va67x0123456789ABCDEF.val[0], 0); a6 += 8;
          va67x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a6, va67x0123456789ABCDEF.val[1], 0); a6 += 8;
          va01x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a1, va01x0123456789ABCDEF.val[0], 1); a1 += 8;
          va01x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a1, va01x0123456789ABCDEF.val[1], 1); a1 += 8;
          va23x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a3, va23x0123456789ABCDEF.val[0], 1); a3 += 8;
          va23x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a3, va23x0123456789ABCDEF.val[1], 1); a3 += 8;
          va45x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a5, va45x0123456789ABCDEF.val[0], 1); a5 += 8;
          va45x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a5, va45x0123456789ABCDEF.val[1], 1); a5 += 8;
          va67x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a7, va67x0123456789ABCDEF.val[0], 1); a7 += 8;
          va67x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a7, va67x0123456789ABCDEF.val[1], 1); a7 += 8;
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

        // Multiply-accumulate: 8x16 * 16x16 --> 8x16.
        vnacc01 = vmmlaq_u32(vnacc01, vkernel_zero_point, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]));
        vpacc01x01 = vmmlaq_u32(vpacc01x01, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]), vb01x01234567);
        vpacc01x23 = vmmlaq_u32(vpacc01x23, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]), vb23x01234567);
        vpacc01x45 = vmmlaq_u32(vpacc01x45, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]), vb45x01234567);
        vpacc01x67 = vmmlaq_u32(vpacc01x67, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]), vb67x01234567);
        vpacc01x89 = vmmlaq_u32(vpacc01x89, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]), vb89x01234567);
        vpacc01xAB = vmmlaq_u32(vpacc01xAB, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]), vbABx01234567);
        vpacc01xCD = vmmlaq_u32(vpacc01xCD, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]), vbCDx01234567);
        vpacc01xEF = vmmlaq_u32(vpacc01xEF, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[0]), vbEFx01234567);
        vnacc23 = vmmlaq_u32(vnacc23, vkernel_zero_point, vreinterpretq_u8_u64(va23x0123456789ABCDEF.val[0]));
        vpacc23x01 = vmmlaq_u32(vpacc23x01, vreinterpretq_u8_u64(va23x0123456789ABCDEF.val[0]), vb01x01234567);
        vpacc23x23 = vmmlaq_u32(vpacc23x23, vreinterpretq_u8_u64(va23x0123456789ABCDEF.val[0]), vb23x01234567);
        vpacc23x45 = vmmlaq_u32(vpacc23x45, vreinterpretq_u8_u64(va23x0123456789ABCDEF.val[0]), vb45x01234567);
        vpacc23x67 = vmmlaq_u32(vpacc23x67, vreinterpretq_u8_u64(va23x0123456789ABCDEF.val[0]), vb67x01234567);
        vpacc23x89 = vmmlaq_u32(vpacc23x89, vreinterpretq_u8_u64(va23x0123456789ABCDEF.val[0]), vb89x01234567);
        vpacc23xAB = vmmlaq_u32(vpacc23xAB, vreinterpretq_u8_u64(va23x0123456789ABCDEF.val[0]), vbABx01234567);
        vpacc23xCD = vmmlaq_u32(vpacc23xCD, vreinterpretq_u8_u64(va23x0123456789ABCDEF.val[0]), vbCDx01234567);
        vpacc23xEF = vmmlaq_u32(vpacc23xEF, vreinterpretq_u8_u64(va23x0123456789ABCDEF.val[0]), vbEFx01234567);
        vnacc45 = vmmlaq_u32(vnacc45, vkernel_zero_point, vreinterpretq_u8_u64(va45x0123456789ABCDEF.val[0]));
        vpacc45x01 = vmmlaq_u32(vpacc45x01, vreinterpretq_u8_u64(va45x0123456789ABCDEF.val[0]), vb01x01234567);
        vpacc45x23 = vmmlaq_u32(vpacc45x23, vreinterpretq_u8_u64(va45x0123456789ABCDEF.val[0]), vb23x01234567);
        vpacc45x45 = vmmlaq_u32(vpacc45x45, vreinterpretq_u8_u64(va45x0123456789ABCDEF.val[0]), vb45x01234567);
        vpacc45x67 = vmmlaq_u32(vpacc45x67, vreinterpretq_u8_u64(va45x0123456789ABCDEF.val[0]), vb67x01234567);
        vpacc45x89 = vmmlaq_u32(vpacc45x89, vreinterpretq_u8_u64(va45x0123456789ABCDEF.val[0]), vb89x01234567);
        vpacc45xAB = vmmlaq_u32(vpacc45xAB, vreinterpretq_u8_u64(va45x0123456789ABCDEF.val[0]), vbABx01234567);
        vpacc45xCD = vmmlaq_u32(vpacc45xCD, vreinterpretq_u8_u64(va45x0123456789ABCDEF.val[0]), vbCDx01234567);
        vpacc45xEF = vmmlaq_u32(vpacc45xEF, vreinterpretq_u8_u64(va45x0123456789ABCDEF.val[0]), vbEFx01234567);
        vnacc67 = vmmlaq_u32(vnacc67, vkernel_zero_point, vreinterpretq_u8_u64(va67x0123456789ABCDEF.val[0]));
        vpacc67x01 = vmmlaq_u32(vpacc67x01, vreinterpretq_u8_u64(va67x0123456789ABCDEF.val[0]), vb01x01234567);
        vpacc67x23 = vmmlaq_u32(vpacc67x23, vreinterpretq_u8_u64(va67x0123456789ABCDEF.val[0]), vb23x01234567);
        vpacc67x45 = vmmlaq_u32(vpacc67x45, vreinterpretq_u8_u64(va67x0123456789ABCDEF.val[0]), vb45x01234567);
        vpacc67x67 = vmmlaq_u32(vpacc67x67, vreinterpretq_u8_u64(va67x0123456789ABCDEF.val[0]), vb67x01234567);
        vpacc67x89 = vmmlaq_u32(vpacc67x89, vreinterpretq_u8_u64(va67x0123456789ABCDEF.val[0]), vb89x01234567);
        vpacc67xAB = vmmlaq_u32(vpacc67xAB, vreinterpretq_u8_u64(va67x0123456789ABCDEF.val[0]), vbABx01234567);
        vpacc67xCD = vmmlaq_u32(vpacc67xCD, vreinterpretq_u8_u64(va67x0123456789ABCDEF.val[0]), vbCDx01234567);
        vpacc67xEF = vmmlaq_u32(vpacc67xEF, vreinterpretq_u8_u64(va67x0123456789ABCDEF.val[0]), vbEFx01234567);
        vnacc01 = vmmlaq_u32(vnacc01, vkernel_zero_point, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]));
        vpacc01x01 = vmmlaq_u32(vpacc01x01, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]), vb01x89ABCDEF);
        vpacc01x23 = vmmlaq_u32(vpacc01x23, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]), vb23x89ABCDEF);
        vpacc01x45 = vmmlaq_u32(vpacc01x45, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]), vb45x89ABCDEF);
        vpacc01x67 = vmmlaq_u32(vpacc01x67, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]), vb67x89ABCDEF);
        vpacc01x89 = vmmlaq_u32(vpacc01x89, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]), vb89x89ABCDEF);
        vpacc01xAB = vmmlaq_u32(vpacc01xAB, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]), vbABx89ABCDEF);
        vpacc01xCD = vmmlaq_u32(vpacc01xCD, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]), vbCDx89ABCDEF);
        vpacc01xEF = vmmlaq_u32(vpacc01xEF, vreinterpretq_u8_u64(va01x0123456789ABCDEF.val[1]), vbEFx89ABCDEF);
        vnacc23 = vmmlaq_u32(vnacc23, vkernel_zero_point, vreinterpretq_u8_u64(va23x0123456789ABCDEF.val[1]));
        vpacc23x01 = vmmlaq_u32(vpacc23x01, vreinterpretq_u8_u64(va23x0123456789ABCDEF.val[1]), vb01x89ABCDEF);
        vpacc23x23 = vmmlaq_u32(vpacc23x23, vreinterpretq_u8_u64(va23x0123456789ABCDEF.val[1]), vb23x89ABCDEF);
        vpacc23x45 = vmmlaq_u32(vpacc23x45, vreinterpretq_u8_u64(va23x0123456789ABCDEF.val[1]), vb45x89ABCDEF);
        vpacc23x67 = vmmlaq_u32(vpacc23x67, vreinterpretq_u8_u64(va23x0123456789ABCDEF.val[1]), vb67x89ABCDEF);
        vpacc23x89 = vmmlaq_u32(vpacc23x89, vreinterpretq_u8_u64(va23x0123456789ABCDEF.val[1]), vb89x89ABCDEF);
        vpacc23xAB = vmmlaq_u32(vpacc23xAB, vreinterpretq_u8_u64(va23x0123456789ABCDEF.val[1]), vbABx89ABCDEF);
        vpacc23xCD = vmmlaq_u32(vpacc23xCD, vreinterpretq_u8_u64(va23x0123456789ABCDEF.val[1]), vbCDx89ABCDEF);
        vpacc23xEF = vmmlaq_u32(vpacc23xEF, vreinterpretq_u8_u64(va23x0123456789ABCDEF.val[1]), vbEFx89ABCDEF);
        vnacc45 = vmmlaq_u32(vnacc45, vkernel_zero_point, vreinterpretq_u8_u64(va45x0123456789ABCDEF.val[1]));
        vpacc45x01 = vmmlaq_u32(vpacc45x01, vreinterpretq_u8_u64(va45x0123456789ABCDEF.val[1]), vb01x89ABCDEF);
        vpacc45x23 = vmmlaq_u32(vpacc45x23, vreinterpretq_u8_u64(va45x0123456789ABCDEF.val[1]), vb23x89ABCDEF);
        vpacc45x45 = vmmlaq_u32(vpacc45x45, vreinterpretq_u8_u64(va45x0123456789ABCDEF.val[1]), vb45x89ABCDEF);
        vpacc45x67 = vmmlaq_u32(vpacc45x67, vreinterpretq_u8_u64(va45x0123456789ABCDEF.val[1]), vb67x89ABCDEF);
        vpacc45x89 = vmmlaq_u32(vpacc45x89, vreinterpretq_u8_u64(va45x0123456789ABCDEF.val[1]), vb89x89ABCDEF);
        vpacc45xAB = vmmlaq_u32(vpacc45xAB, vreinterpretq_u8_u64(va45x0123456789ABCDEF.val[1]), vbABx89ABCDEF);
        vpacc45xCD = vmmlaq_u32(vpacc45xCD, vreinterpretq_u8_u64(va45x0123456789ABCDEF.val[1]), vbCDx89ABCDEF);
        vpacc45xEF = vmmlaq_u32(vpacc45xEF, vreinterpretq_u8_u64(va45x0123456789ABCDEF.val[1]), vbEFx89ABCDEF);
        vnacc67 = vmmlaq_u32(vnacc67, vkernel_zero_point, vreinterpretq_u8_u64(va67x0123456789ABCDEF.val[1]));
        vpacc67x01 = vmmlaq_u32(vpacc67x01, vreinterpretq_u8_u64(va67x0123456789ABCDEF.val[1]), vb01x89ABCDEF);
        vpacc67x23 = vmmlaq_u32(vpacc67x23, vreinterpretq_u8_u64(va67x0123456789ABCDEF.val[1]), vb23x89ABCDEF);
        vpacc67x45 = vmmlaq_u32(vpacc67x45, vreinterpretq_u8_u64(va67x0123456789ABCDEF.val[1]), vb45x89ABCDEF);
        vpacc67x67 = vmmlaq_u32(vpacc67x67, vreinterpretq_u8_u64(va67x0123456789ABCDEF.val[1]), vb67x89ABCDEF);
        vpacc67x89 = vmmlaq_u32(vpacc67x89, vreinterpretq_u8_u64(va67x0123456789ABCDEF.val[1]), vb89x89ABCDEF);
        vpacc67xAB = vmmlaq_u32(vpacc67xAB, vreinterpretq_u8_u64(va67x0123456789ABCDEF.val[1]), vbABx89ABCDEF);
        vpacc67xCD = vmmlaq_u32(vpacc67xCD, vreinterpretq_u8_u64(va67x0123456789ABCDEF.val[1]), vbCDx89ABCDEF);
        vpacc67xEF = vmmlaq_u32(vpacc67xEF, vreinterpretq_u8_u64(va67x0123456789ABCDEF.val[1]), vbEFx89ABCDEF);

        k -= 16 * sizeof(uint8_t);
      }
      // Handle up to 8 final positions of `k`
      if XNN_UNLIKELY(k != 0) {
        // Load a 8x8 block of activations.
        uint64x2_t va01x01234567 = vld1q_dup_u64((const void*) a0); a0 += 8;
        uint64x2_t va23x01234567 = vld1q_dup_u64((const void*) a2); a2 += 8;
        uint64x2_t va45x01234567 = vld1q_dup_u64((const void*) a4); a4 += 8;
        uint64x2_t va67x01234567 = vld1q_dup_u64((const void*) a6); a6 += 8;
        va01x01234567 = vld1q_lane_u64((const void*) a1, va01x01234567, 1); a1 += 8;
        va23x01234567 = vld1q_lane_u64((const void*) a3, va23x01234567, 1); a3 += 8;
        va45x01234567 = vld1q_lane_u64((const void*) a5, va45x01234567, 1); a5 += 8;
        va67x01234567 = vld1q_lane_u64((const void*) a7, va67x01234567, 1); a7 += 8;

        // Load a 16x16 block of weights.
        const uint8x16_t vb01x01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint8x16_t vb23x01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint8x16_t vb45x01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint8x16_t vb67x01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint8x16_t vb89x01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint8x16_t vbABx01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint8x16_t vbCDx01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint8x16_t vbEFx01234567 = vld1q_u8(w); w = (const uint8_t*) w + 16;

        // Multiply-accumulate: 8x8 * 8x16 --> 8x16.
        vnacc01 = vmmlaq_u32(vnacc01, vkernel_zero_point, vreinterpretq_u8_u64(va01x01234567));
        vpacc01x01 = vmmlaq_u32(vpacc01x01, vreinterpretq_u8_u64(va01x01234567), vb01x01234567);
        vpacc01x23 = vmmlaq_u32(vpacc01x23, vreinterpretq_u8_u64(va01x01234567), vb23x01234567);
        vpacc01x45 = vmmlaq_u32(vpacc01x45, vreinterpretq_u8_u64(va01x01234567), vb45x01234567);
        vpacc01x67 = vmmlaq_u32(vpacc01x67, vreinterpretq_u8_u64(va01x01234567), vb67x01234567);
        vpacc01x89 = vmmlaq_u32(vpacc01x89, vreinterpretq_u8_u64(va01x01234567), vb89x01234567);
        vpacc01xAB = vmmlaq_u32(vpacc01xAB, vreinterpretq_u8_u64(va01x01234567), vbABx01234567);
        vpacc01xCD = vmmlaq_u32(vpacc01xCD, vreinterpretq_u8_u64(va01x01234567), vbCDx01234567);
        vpacc01xEF = vmmlaq_u32(vpacc01xEF, vreinterpretq_u8_u64(va01x01234567), vbEFx01234567);
        vnacc23 = vmmlaq_u32(vnacc23, vkernel_zero_point, vreinterpretq_u8_u64(va23x01234567));
        vpacc23x01 = vmmlaq_u32(vpacc23x01, vreinterpretq_u8_u64(va23x01234567), vb01x01234567);
        vpacc23x23 = vmmlaq_u32(vpacc23x23, vreinterpretq_u8_u64(va23x01234567), vb23x01234567);
        vpacc23x45 = vmmlaq_u32(vpacc23x45, vreinterpretq_u8_u64(va23x01234567), vb45x01234567);
        vpacc23x67 = vmmlaq_u32(vpacc23x67, vreinterpretq_u8_u64(va23x01234567), vb67x01234567);
        vpacc23x89 = vmmlaq_u32(vpacc23x89, vreinterpretq_u8_u64(va23x01234567), vb89x01234567);
        vpacc23xAB = vmmlaq_u32(vpacc23xAB, vreinterpretq_u8_u64(va23x01234567), vbABx01234567);
        vpacc23xCD = vmmlaq_u32(vpacc23xCD, vreinterpretq_u8_u64(va23x01234567), vbCDx01234567);
        vpacc23xEF = vmmlaq_u32(vpacc23xEF, vreinterpretq_u8_u64(va23x01234567), vbEFx01234567);
        vnacc45 = vmmlaq_u32(vnacc45, vkernel_zero_point, vreinterpretq_u8_u64(va45x01234567));
        vpacc45x01 = vmmlaq_u32(vpacc45x01, vreinterpretq_u8_u64(va45x01234567), vb01x01234567);
        vpacc45x23 = vmmlaq_u32(vpacc45x23, vreinterpretq_u8_u64(va45x01234567), vb23x01234567);
        vpacc45x45 = vmmlaq_u32(vpacc45x45, vreinterpretq_u8_u64(va45x01234567), vb45x01234567);
        vpacc45x67 = vmmlaq_u32(vpacc45x67, vreinterpretq_u8_u64(va45x01234567), vb67x01234567);
        vpacc45x89 = vmmlaq_u32(vpacc45x89, vreinterpretq_u8_u64(va45x01234567), vb89x01234567);
        vpacc45xAB = vmmlaq_u32(vpacc45xAB, vreinterpretq_u8_u64(va45x01234567), vbABx01234567);
        vpacc45xCD = vmmlaq_u32(vpacc45xCD, vreinterpretq_u8_u64(va45x01234567), vbCDx01234567);
        vpacc45xEF = vmmlaq_u32(vpacc45xEF, vreinterpretq_u8_u64(va45x01234567), vbEFx01234567);
        vnacc67 = vmmlaq_u32(vnacc67, vkernel_zero_point, vreinterpretq_u8_u64(va67x01234567));
        vpacc67x01 = vmmlaq_u32(vpacc67x01, vreinterpretq_u8_u64(va67x01234567), vb01x01234567);
        vpacc67x23 = vmmlaq_u32(vpacc67x23, vreinterpretq_u8_u64(va67x01234567), vb23x01234567);
        vpacc67x45 = vmmlaq_u32(vpacc67x45, vreinterpretq_u8_u64(va67x01234567), vb45x01234567);
        vpacc67x67 = vmmlaq_u32(vpacc67x67, vreinterpretq_u8_u64(va67x01234567), vb67x01234567);
        vpacc67x89 = vmmlaq_u32(vpacc67x89, vreinterpretq_u8_u64(va67x01234567), vb89x01234567);
        vpacc67xAB = vmmlaq_u32(vpacc67xAB, vreinterpretq_u8_u64(va67x01234567), vbABx01234567);
        vpacc67xCD = vmmlaq_u32(vpacc67xCD, vreinterpretq_u8_u64(va67x01234567), vbCDx01234567);
        vpacc67xEF = vmmlaq_u32(vpacc67xEF, vreinterpretq_u8_u64(va67x01234567), vbEFx01234567);
      }
      p -= 8 * sizeof(void*);
    } while (p != 0);

    // Subtract zero point from accumulators.
    #if XNN_ARCH_ARM64
      const uint32x4_t vnacc01x01 = vzip1q_u32(vnacc01, vnacc01);
      const uint32x4_t vnacc23x01 = vzip1q_u32(vnacc23, vnacc23);
      const uint32x4_t vnacc45x01 = vzip1q_u32(vnacc45, vnacc45);
      const uint32x4_t vnacc67x01 = vzip1q_u32(vnacc67, vnacc67);
    #else
      const uint32x4_t vnacc01x01 = vzipq_u32(vnacc01, vnacc01).val[0];
      const uint32x4_t vnacc23x01 = vzipq_u32(vnacc23, vnacc23).val[0];
      const uint32x4_t vnacc45x01 = vzipq_u32(vnacc45, vnacc45).val[0];
      const uint32x4_t vnacc67x01 = vzipq_u32(vnacc67, vnacc67).val[0];
    #endif
    int32x4_t vacc01x01 = vreinterpretq_s32_u32(vsubq_u32(vpacc01x01, vnacc01x01));
    int32x4_t vacc01x23 = vreinterpretq_s32_u32(vsubq_u32(vpacc01x23, vnacc01x01));
    int32x4_t vacc01x45 = vreinterpretq_s32_u32(vsubq_u32(vpacc01x45, vnacc01x01));
    int32x4_t vacc01x67 = vreinterpretq_s32_u32(vsubq_u32(vpacc01x67, vnacc01x01));
    int32x4_t vacc01x89 = vreinterpretq_s32_u32(vsubq_u32(vpacc01x89, vnacc01x01));
    int32x4_t vacc01xAB = vreinterpretq_s32_u32(vsubq_u32(vpacc01xAB, vnacc01x01));
    int32x4_t vacc01xCD = vreinterpretq_s32_u32(vsubq_u32(vpacc01xCD, vnacc01x01));
    int32x4_t vacc01xEF = vreinterpretq_s32_u32(vsubq_u32(vpacc01xEF, vnacc01x01));
    int32x4_t vacc23x01 = vreinterpretq_s32_u32(vsubq_u32(vpacc23x01, vnacc23x01));
    int32x4_t vacc23x23 = vreinterpretq_s32_u32(vsubq_u32(vpacc23x23, vnacc23x01));
    int32x4_t vacc23x45 = vreinterpretq_s32_u32(vsubq_u32(vpacc23x45, vnacc23x01));
    int32x4_t vacc23x67 = vreinterpretq_s32_u32(vsubq_u32(vpacc23x67, vnacc23x01));
    int32x4_t vacc23x89 = vreinterpretq_s32_u32(vsubq_u32(vpacc23x89, vnacc23x01));
    int32x4_t vacc23xAB = vreinterpretq_s32_u32(vsubq_u32(vpacc23xAB, vnacc23x01));
    int32x4_t vacc23xCD = vreinterpretq_s32_u32(vsubq_u32(vpacc23xCD, vnacc23x01));
    int32x4_t vacc23xEF = vreinterpretq_s32_u32(vsubq_u32(vpacc23xEF, vnacc23x01));
    int32x4_t vacc45x01 = vreinterpretq_s32_u32(vsubq_u32(vpacc45x01, vnacc45x01));
    int32x4_t vacc45x23 = vreinterpretq_s32_u32(vsubq_u32(vpacc45x23, vnacc45x01));
    int32x4_t vacc45x45 = vreinterpretq_s32_u32(vsubq_u32(vpacc45x45, vnacc45x01));
    int32x4_t vacc45x67 = vreinterpretq_s32_u32(vsubq_u32(vpacc45x67, vnacc45x01));
    int32x4_t vacc45x89 = vreinterpretq_s32_u32(vsubq_u32(vpacc45x89, vnacc45x01));
    int32x4_t vacc45xAB = vreinterpretq_s32_u32(vsubq_u32(vpacc45xAB, vnacc45x01));
    int32x4_t vacc45xCD = vreinterpretq_s32_u32(vsubq_u32(vpacc45xCD, vnacc45x01));
    int32x4_t vacc45xEF = vreinterpretq_s32_u32(vsubq_u32(vpacc45xEF, vnacc45x01));
    int32x4_t vacc67x01 = vreinterpretq_s32_u32(vsubq_u32(vpacc67x01, vnacc67x01));
    int32x4_t vacc67x23 = vreinterpretq_s32_u32(vsubq_u32(vpacc67x23, vnacc67x01));
    int32x4_t vacc67x45 = vreinterpretq_s32_u32(vsubq_u32(vpacc67x45, vnacc67x01));
    int32x4_t vacc67x67 = vreinterpretq_s32_u32(vsubq_u32(vpacc67x67, vnacc67x01));
    int32x4_t vacc67x89 = vreinterpretq_s32_u32(vsubq_u32(vpacc67x89, vnacc67x01));
    int32x4_t vacc67xAB = vreinterpretq_s32_u32(vsubq_u32(vpacc67xAB, vnacc67x01));
    int32x4_t vacc67xCD = vreinterpretq_s32_u32(vsubq_u32(vpacc67xCD, vnacc67x01));
    int32x4_t vacc67xEF = vreinterpretq_s32_u32(vsubq_u32(vpacc67xEF, vnacc67x01));

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
      int32x4_t vacc5x0123 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc45x01), vreinterpretq_u64_s32(vacc45x23)));
      int32x4_t vacc4x4567 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc45x45), vreinterpretq_u64_s32(vacc45x67)));
      int32x4_t vacc5x4567 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc45x45), vreinterpretq_u64_s32(vacc45x67)));
      int32x4_t vacc4x89AB = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc45x89), vreinterpretq_u64_s32(vacc45xAB)));
      int32x4_t vacc5x89AB = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc45x89), vreinterpretq_u64_s32(vacc45xAB)));
      int32x4_t vacc4xCDEF = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc45xCD), vreinterpretq_u64_s32(vacc45xEF)));
      int32x4_t vacc5xCDEF = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc45xCD), vreinterpretq_u64_s32(vacc45xEF)));
      int32x4_t vacc6x0123 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc67x01), vreinterpretq_u64_s32(vacc67x23)));
      int32x4_t vacc7x0123 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc67x01), vreinterpretq_u64_s32(vacc67x23)));
      int32x4_t vacc6x4567 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc67x45), vreinterpretq_u64_s32(vacc67x67)));
      int32x4_t vacc7x4567 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc67x45), vreinterpretq_u64_s32(vacc67x67)));
      int32x4_t vacc6x89AB = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc67x89), vreinterpretq_u64_s32(vacc67xAB)));
      int32x4_t vacc7x89AB = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc67x89), vreinterpretq_u64_s32(vacc67xAB)));
      int32x4_t vacc6xCDEF = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc67xCD), vreinterpretq_u64_s32(vacc67xEF)));
      int32x4_t vacc7xCDEF = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc67xCD), vreinterpretq_u64_s32(vacc67xEF)));
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
      int32x4_t vacc5x0123 = vcombine_s32(vget_high_s32(vacc45x01), vget_high_s32(vacc45x23));
      int32x4_t vacc4x4567 = vcombine_s32(vget_low_s32(vacc45x45), vget_low_s32(vacc45x67));
      int32x4_t vacc5x4567 = vcombine_s32(vget_high_s32(vacc45x45), vget_high_s32(vacc45x67));
      int32x4_t vacc4x89AB = vcombine_s32(vget_low_s32(vacc45x89), vget_low_s32(vacc45xAB));
      int32x4_t vacc5x89AB = vcombine_s32(vget_high_s32(vacc45x89), vget_high_s32(vacc45xAB));
      int32x4_t vacc4xCDEF = vcombine_s32(vget_low_s32(vacc45xCD), vget_low_s32(vacc45xEF));
      int32x4_t vacc5xCDEF = vcombine_s32(vget_high_s32(vacc45xCD), vget_high_s32(vacc45xEF));
      int32x4_t vacc6x0123 = vcombine_s32(vget_low_s32(vacc67x01), vget_low_s32(vacc67x23));
      int32x4_t vacc7x0123 = vcombine_s32(vget_high_s32(vacc67x01), vget_high_s32(vacc67x23));
      int32x4_t vacc6x4567 = vcombine_s32(vget_low_s32(vacc67x45), vget_low_s32(vacc67x67));
      int32x4_t vacc7x4567 = vcombine_s32(vget_high_s32(vacc67x45), vget_high_s32(vacc67x67));
      int32x4_t vacc6x89AB = vcombine_s32(vget_low_s32(vacc67x89), vget_low_s32(vacc67xAB));
      int32x4_t vacc7x89AB = vcombine_s32(vget_high_s32(vacc67x89), vget_high_s32(vacc67xAB));
      int32x4_t vacc6xCDEF = vcombine_s32(vget_low_s32(vacc67xCD), vget_low_s32(vacc67xEF));
      int32x4_t vacc7xCDEF = vcombine_s32(vget_high_s32(vacc67xCD), vget_high_s32(vacc67xEF));
    #endif
    const int32x4_t vright_pre_shift = vld1q_dup_s32(&params->rndnu_neon.right_pre_shift);
    const int32x4_t vmultiplier = vld1q_dup_s32(&params->rndnu_neon.multiplier);
    const int32x4_t vright_post_shift = vld1q_dup_s32(&params->rndnu_neon.right_post_shift);

    vacc0x0123 = vshlq_s32(vacc0x0123, vright_pre_shift);
    vacc0x4567 = vshlq_s32(vacc0x4567, vright_pre_shift);
    vacc0x89AB = vshlq_s32(vacc0x89AB, vright_pre_shift);
    vacc0xCDEF = vshlq_s32(vacc0xCDEF, vright_pre_shift);
    vacc1x0123 = vshlq_s32(vacc1x0123, vright_pre_shift);
    vacc1x4567 = vshlq_s32(vacc1x4567, vright_pre_shift);
    vacc1x89AB = vshlq_s32(vacc1x89AB, vright_pre_shift);
    vacc1xCDEF = vshlq_s32(vacc1xCDEF, vright_pre_shift);
    vacc2x0123 = vshlq_s32(vacc2x0123, vright_pre_shift);
    vacc2x4567 = vshlq_s32(vacc2x4567, vright_pre_shift);
    vacc2x89AB = vshlq_s32(vacc2x89AB, vright_pre_shift);
    vacc2xCDEF = vshlq_s32(vacc2xCDEF, vright_pre_shift);
    vacc3x0123 = vshlq_s32(vacc3x0123, vright_pre_shift);
    vacc3x4567 = vshlq_s32(vacc3x4567, vright_pre_shift);
    vacc3x89AB = vshlq_s32(vacc3x89AB, vright_pre_shift);
    vacc3xCDEF = vshlq_s32(vacc3xCDEF, vright_pre_shift);
    vacc4x0123 = vshlq_s32(vacc4x0123, vright_pre_shift);
    vacc4x4567 = vshlq_s32(vacc4x4567, vright_pre_shift);
    vacc4x89AB = vshlq_s32(vacc4x89AB, vright_pre_shift);
    vacc4xCDEF = vshlq_s32(vacc4xCDEF, vright_pre_shift);
    vacc5x0123 = vshlq_s32(vacc5x0123, vright_pre_shift);
    vacc5x4567 = vshlq_s32(vacc5x4567, vright_pre_shift);
    vacc5x89AB = vshlq_s32(vacc5x89AB, vright_pre_shift);
    vacc5xCDEF = vshlq_s32(vacc5xCDEF, vright_pre_shift);
    vacc6x0123 = vshlq_s32(vacc6x0123, vright_pre_shift);
    vacc6x4567 = vshlq_s32(vacc6x4567, vright_pre_shift);
    vacc6x89AB = vshlq_s32(vacc6x89AB, vright_pre_shift);
    vacc6xCDEF = vshlq_s32(vacc6xCDEF, vright_pre_shift);
    vacc7x0123 = vshlq_s32(vacc7x0123, vright_pre_shift);
    vacc7x4567 = vshlq_s32(vacc7x4567, vright_pre_shift);
    vacc7x89AB = vshlq_s32(vacc7x89AB, vright_pre_shift);
    vacc7xCDEF = vshlq_s32(vacc7xCDEF, vright_pre_shift);

    vacc0x0123 = vqdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqdmulhq_s32(vacc0x4567, vmultiplier);
    vacc0x89AB = vqdmulhq_s32(vacc0x89AB, vmultiplier);
    vacc0xCDEF = vqdmulhq_s32(vacc0xCDEF, vmultiplier);
    vacc1x0123 = vqdmulhq_s32(vacc1x0123, vmultiplier);
    vacc1x4567 = vqdmulhq_s32(vacc1x4567, vmultiplier);
    vacc1x89AB = vqdmulhq_s32(vacc1x89AB, vmultiplier);
    vacc1xCDEF = vqdmulhq_s32(vacc1xCDEF, vmultiplier);
    vacc2x0123 = vqdmulhq_s32(vacc2x0123, vmultiplier);
    vacc2x4567 = vqdmulhq_s32(vacc2x4567, vmultiplier);
    vacc2x89AB = vqdmulhq_s32(vacc2x89AB, vmultiplier);
    vacc2xCDEF = vqdmulhq_s32(vacc2xCDEF, vmultiplier);
    vacc3x0123 = vqdmulhq_s32(vacc3x0123, vmultiplier);
    vacc3x4567 = vqdmulhq_s32(vacc3x4567, vmultiplier);
    vacc3x89AB = vqdmulhq_s32(vacc3x89AB, vmultiplier);
    vacc3xCDEF = vqdmulhq_s32(vacc3xCDEF, vmultiplier);
    vacc4x0123 = vqdmulhq_s32(vacc4x0123, vmultiplier);
    vacc4x4567 = vqdmulhq_s32(vacc4x4567, vmultiplier);
    vacc4x89AB = vqdmulhq_s32(vacc4x89AB, vmultiplier);
    vacc4xCDEF = vqdmulhq_s32(vacc4xCDEF, vmultiplier);
    vacc5x0123 = vqdmulhq_s32(vacc5x0123, vmultiplier);
    vacc5x4567 = vqdmulhq_s32(vacc5x4567, vmultiplier);
    vacc5x89AB = vqdmulhq_s32(vacc5x89AB, vmultiplier);
    vacc5xCDEF = vqdmulhq_s32(vacc5xCDEF, vmultiplier);
    vacc6x0123 = vqdmulhq_s32(vacc6x0123, vmultiplier);
    vacc6x4567 = vqdmulhq_s32(vacc6x4567, vmultiplier);
    vacc6x89AB = vqdmulhq_s32(vacc6x89AB, vmultiplier);
    vacc6xCDEF = vqdmulhq_s32(vacc6xCDEF, vmultiplier);
    vacc7x0123 = vqdmulhq_s32(vacc7x0123, vmultiplier);
    vacc7x4567 = vqdmulhq_s32(vacc7x4567, vmultiplier);
    vacc7x89AB = vqdmulhq_s32(vacc7x89AB, vmultiplier);
    vacc7xCDEF = vqdmulhq_s32(vacc7xCDEF, vmultiplier);

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_post_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_post_shift);
    vacc0x89AB = vrshlq_s32(vacc0x89AB, vright_post_shift);
    vacc0xCDEF = vrshlq_s32(vacc0xCDEF, vright_post_shift);
    vacc1x0123 = vrshlq_s32(vacc1x0123, vright_post_shift);
    vacc1x4567 = vrshlq_s32(vacc1x4567, vright_post_shift);
    vacc1x89AB = vrshlq_s32(vacc1x89AB, vright_post_shift);
    vacc1xCDEF = vrshlq_s32(vacc1xCDEF, vright_post_shift);
    vacc2x0123 = vrshlq_s32(vacc2x0123, vright_post_shift);
    vacc2x4567 = vrshlq_s32(vacc2x4567, vright_post_shift);
    vacc2x89AB = vrshlq_s32(vacc2x89AB, vright_post_shift);
    vacc2xCDEF = vrshlq_s32(vacc2xCDEF, vright_post_shift);
    vacc3x0123 = vrshlq_s32(vacc3x0123, vright_post_shift);
    vacc3x4567 = vrshlq_s32(vacc3x4567, vright_post_shift);
    vacc3x89AB = vrshlq_s32(vacc3x89AB, vright_post_shift);
    vacc3xCDEF = vrshlq_s32(vacc3xCDEF, vright_post_shift);
    vacc4x0123 = vrshlq_s32(vacc4x0123, vright_post_shift);
    vacc4x4567 = vrshlq_s32(vacc4x4567, vright_post_shift);
    vacc4x89AB = vrshlq_s32(vacc4x89AB, vright_post_shift);
    vacc4xCDEF = vrshlq_s32(vacc4xCDEF, vright_post_shift);
    vacc5x0123 = vrshlq_s32(vacc5x0123, vright_post_shift);
    vacc5x4567 = vrshlq_s32(vacc5x4567, vright_post_shift);
    vacc5x89AB = vrshlq_s32(vacc5x89AB, vright_post_shift);
    vacc5xCDEF = vrshlq_s32(vacc5xCDEF, vright_post_shift);
    vacc6x0123 = vrshlq_s32(vacc6x0123, vright_post_shift);
    vacc6x4567 = vrshlq_s32(vacc6x4567, vright_post_shift);
    vacc6x89AB = vrshlq_s32(vacc6x89AB, vright_post_shift);
    vacc6xCDEF = vrshlq_s32(vacc6xCDEF, vright_post_shift);
    vacc7x0123 = vrshlq_s32(vacc7x0123, vright_post_shift);
    vacc7x4567 = vrshlq_s32(vacc7x4567, vright_post_shift);
    vacc7x89AB = vrshlq_s32(vacc7x89AB, vright_post_shift);
    vacc7xCDEF = vrshlq_s32(vacc7xCDEF, vright_post_shift);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->rndnu_neon.output_zero_point);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
    const int16x8_t vacc1x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x89AB), vacc1xCDEF), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
    const int16x8_t vacc2x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x89AB), vacc2xCDEF), voutput_zero_point);
    const int16x8_t vacc3x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);
    const int16x8_t vacc3x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x89AB), vacc3xCDEF), voutput_zero_point);
    const int16x8_t vacc4x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc4x0123), vacc4x4567), voutput_zero_point);
    const int16x8_t vacc4x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc4x89AB), vacc4xCDEF), voutput_zero_point);
    const int16x8_t vacc5x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc5x0123), vacc5x4567), voutput_zero_point);
    const int16x8_t vacc5x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc5x89AB), vacc5xCDEF), voutput_zero_point);
    const int16x8_t vacc6x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc6x0123), vacc6x4567), voutput_zero_point);
    const int16x8_t vacc6x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc6x89AB), vacc6xCDEF), voutput_zero_point);
    const int16x8_t vacc7x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc7x0123), vacc7x4567), voutput_zero_point);
    const int16x8_t vacc7x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc7x89AB), vacc7xCDEF), voutput_zero_point);

    uint8x16_t vout0x0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc0x01234567), vacc0x89ABCDEF);
    uint8x16_t vout1x0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc1x01234567), vacc1x89ABCDEF);
    uint8x16_t vout2x0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc2x01234567), vacc2x89ABCDEF);
    uint8x16_t vout3x0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc3x01234567), vacc3x89ABCDEF);
    uint8x16_t vout4x0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc4x01234567), vacc4x89ABCDEF);
    uint8x16_t vout5x0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc5x01234567), vacc5x89ABCDEF);
    uint8x16_t vout6x0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc6x01234567), vacc6x89ABCDEF);
    uint8x16_t vout7x0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc7x01234567), vacc7x89ABCDEF);
#else
    const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF)), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
    const int16x8_t vacc1x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x89AB), vqmovn_s32(vacc1xCDEF)), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);
    const int16x8_t vacc2x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x89AB), vqmovn_s32(vacc2xCDEF)), voutput_zero_point);
    const int16x8_t vacc3x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)), voutput_zero_point);
    const int16x8_t vacc3x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x89AB), vqmovn_s32(vacc3xCDEF)), voutput_zero_point);
    const int16x8_t vacc4x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc4x0123), vqmovn_s32(vacc4x4567)), voutput_zero_point);
    const int16x8_t vacc4x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc4x89AB), vqmovn_s32(vacc4xCDEF)), voutput_zero_point);
    const int16x8_t vacc5x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc5x0123), vqmovn_s32(vacc5x4567)), voutput_zero_point);
    const int16x8_t vacc5x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc5x89AB), vqmovn_s32(vacc5xCDEF)), voutput_zero_point);
    const int16x8_t vacc6x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc6x0123), vqmovn_s32(vacc6x4567)), voutput_zero_point);
    const int16x8_t vacc6x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc6x89AB), vqmovn_s32(vacc6xCDEF)), voutput_zero_point);
    const int16x8_t vacc7x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc7x0123), vqmovn_s32(vacc7x4567)), voutput_zero_point);
    const int16x8_t vacc7x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc7x89AB), vqmovn_s32(vacc7xCDEF)), voutput_zero_point);

    uint8x16_t vout0x0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc0x01234567), vqmovun_s16(vacc0x89ABCDEF));
    uint8x16_t vout1x0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc1x01234567), vqmovun_s16(vacc1x89ABCDEF));
    uint8x16_t vout2x0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc2x01234567), vqmovun_s16(vacc2x89ABCDEF));
    uint8x16_t vout3x0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc3x01234567), vqmovun_s16(vacc3x89ABCDEF));
    uint8x16_t vout4x0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc4x01234567), vqmovun_s16(vacc4x89ABCDEF));
    uint8x16_t vout5x0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc5x01234567), vqmovun_s16(vacc5x89ABCDEF));
    uint8x16_t vout6x0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc6x01234567), vqmovun_s16(vacc6x89ABCDEF));
    uint8x16_t vout7x0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc7x01234567), vqmovun_s16(vacc7x89ABCDEF));
#endif
    const uint8x16_t voutput_min = vld1q_dup_u8(&params->rndnu_neon.output_min);
    const uint8x16_t voutput_max = vld1q_dup_u8(&params->rndnu_neon.output_max);

    vout0x0123456789ABCDEF = vmaxq_u8(vout0x0123456789ABCDEF, voutput_min);
    vout1x0123456789ABCDEF = vmaxq_u8(vout1x0123456789ABCDEF, voutput_min);
    vout2x0123456789ABCDEF = vmaxq_u8(vout2x0123456789ABCDEF, voutput_min);
    vout3x0123456789ABCDEF = vmaxq_u8(vout3x0123456789ABCDEF, voutput_min);
    vout4x0123456789ABCDEF = vmaxq_u8(vout4x0123456789ABCDEF, voutput_min);
    vout5x0123456789ABCDEF = vmaxq_u8(vout5x0123456789ABCDEF, voutput_min);
    vout6x0123456789ABCDEF = vmaxq_u8(vout6x0123456789ABCDEF, voutput_min);
    vout7x0123456789ABCDEF = vmaxq_u8(vout7x0123456789ABCDEF, voutput_min);

    vout0x0123456789ABCDEF = vminq_u8(vout0x0123456789ABCDEF, voutput_max);
    vout1x0123456789ABCDEF = vminq_u8(vout1x0123456789ABCDEF, voutput_max);
    vout2x0123456789ABCDEF = vminq_u8(vout2x0123456789ABCDEF, voutput_max);
    vout3x0123456789ABCDEF = vminq_u8(vout3x0123456789ABCDEF, voutput_max);
    vout4x0123456789ABCDEF = vminq_u8(vout4x0123456789ABCDEF, voutput_max);
    vout5x0123456789ABCDEF = vminq_u8(vout5x0123456789ABCDEF, voutput_max);
    vout6x0123456789ABCDEF = vminq_u8(vout6x0123456789ABCDEF, voutput_max);
    vout7x0123456789ABCDEF = vminq_u8(vout7x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      vst1q_u8(c7, vout7x0123456789ABCDEF);
      vst1q_u8(c6, vout6x0123456789ABCDEF);
      vst1q_u8(c5, vout5x0123456789ABCDEF);
      vst1q_u8(c4, vout4x0123456789ABCDEF);
      vst1q_u8(c3, vout3x0123456789ABCDEF);
      vst1q_u8(c2, vout2x0123456789ABCDEF);
      vst1q_u8(c1, vout1x0123456789ABCDEF);
      vst1q_u8(c0, vout0x0123456789ABCDEF);

      c7 = (uint8_t*) ((uintptr_t) c7 + cn_stride);
      c6 = (uint8_t*) ((uintptr_t) c6 + cn_stride);
      c5 = (uint8_t*) ((uintptr_t) c5 + cn_stride);
      c4 = (uint8_t*) ((uintptr_t) c4 + cn_stride);
      c3 = (uint8_t*) ((uintptr_t) c3 + cn_stride);
      c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      uint8x16_t vout6x01234567_7x01234567 = vcombine_u8(vget_low_u8(vout6x0123456789ABCDEF), vget_low_u8(vout7x0123456789ABCDEF));
      uint8x16_t vout4x01234567_5x01234567 = vcombine_u8(vget_low_u8(vout4x0123456789ABCDEF), vget_low_u8(vout5x0123456789ABCDEF));
      uint8x16_t vout2x01234567_3x01234567 = vcombine_u8(vget_low_u8(vout2x0123456789ABCDEF), vget_low_u8(vout3x0123456789ABCDEF));
      uint8x16_t vout0x01234567_1x01234567 = vcombine_u8(vget_low_u8(vout0x0123456789ABCDEF), vget_low_u8(vout1x0123456789ABCDEF));
      if (nc & 8) {
        vst1_u8(c7, vget_high_u8(vout6x01234567_7x01234567)); c7 += 8;
        vst1_u8(c6, vget_low_u8(vout6x01234567_7x01234567)); c6 += 8;
        vst1_u8(c5, vget_high_u8(vout4x01234567_5x01234567)); c5 += 8;
        vst1_u8(c4, vget_low_u8(vout4x01234567_5x01234567)); c4 += 8;
        vst1_u8(c3, vget_high_u8(vout2x01234567_3x01234567)); c3 += 8;
        vst1_u8(c2, vget_low_u8(vout2x01234567_3x01234567)); c2 += 8;
        vst1_u8(c1, vget_high_u8(vout0x01234567_1x01234567)); c1 += 8;
        vst1_u8(c0, vget_low_u8(vout0x01234567_1x01234567)); c0 += 8;
        vout6x01234567_7x01234567 = vcombine_u8(vget_high_u8(vout6x0123456789ABCDEF), vget_high_u8(vout7x0123456789ABCDEF));
        vout4x01234567_5x01234567 = vcombine_u8(vget_high_u8(vout4x0123456789ABCDEF), vget_high_u8(vout5x0123456789ABCDEF));
        vout2x01234567_3x01234567 = vcombine_u8(vget_high_u8(vout2x0123456789ABCDEF), vget_high_u8(vout3x0123456789ABCDEF));
        vout0x01234567_1x01234567 = vcombine_u8(vget_high_u8(vout0x0123456789ABCDEF), vget_high_u8(vout1x0123456789ABCDEF));
      }
      if (nc & 4) {
        vst1q_lane_u32((void*) c7, vreinterpretq_u32_u8(vout6x01234567_7x01234567), 2); c7 += 4;
        vst1q_lane_u32((void*) c6, vreinterpretq_u32_u8(vout6x01234567_7x01234567), 0); c6 += 4;
        vst1q_lane_u32((void*) c5, vreinterpretq_u32_u8(vout4x01234567_5x01234567), 2); c5 += 4;
        vst1q_lane_u32((void*) c4, vreinterpretq_u32_u8(vout4x01234567_5x01234567), 0); c4 += 4;
        vst1q_lane_u32((void*) c3, vreinterpretq_u32_u8(vout2x01234567_3x01234567), 2); c3 += 4;
        vst1q_lane_u32((void*) c2, vreinterpretq_u32_u8(vout2x01234567_3x01234567), 0); c2 += 4;
        vst1q_lane_u32((void*) c1, vreinterpretq_u32_u8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1q_lane_u32((void*) c0, vreinterpretq_u32_u8(vout0x01234567_1x01234567), 0); c0 += 4;
        vout6x01234567_7x01234567 = vextq_u8(vout6x01234567_7x01234567, vout6x01234567_7x01234567, 4);
        vout4x01234567_5x01234567 = vextq_u8(vout4x01234567_5x01234567, vout4x01234567_5x01234567, 4);
        vout2x01234567_3x01234567 = vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
        vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16((void*) c7, vreinterpretq_u16_u8(vout6x01234567_7x01234567), 4); c7 += 2;
        vst1q_lane_u16((void*) c6, vreinterpretq_u16_u8(vout6x01234567_7x01234567), 0); c6 += 2;
        vst1q_lane_u16((void*) c5, vreinterpretq_u16_u8(vout4x01234567_5x01234567), 4); c5 += 2;
        vst1q_lane_u16((void*) c4, vreinterpretq_u16_u8(vout4x01234567_5x01234567), 0); c4 += 2;
        vst1q_lane_u16((void*) c3, vreinterpretq_u16_u8(vout2x01234567_3x01234567), 4); c3 += 2;
        vst1q_lane_u16((void*) c2, vreinterpretq_u16_u8(vout2x01234567_3x01234567), 0); c2 += 2;
        vst1q_lane_u16((void*) c1, vreinterpretq_u16_u8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1q_lane_u16((void*) c0, vreinterpretq_u16_u8(vout0x01234567_1x01234567), 0); c0 += 2;
        vout6x01234567_7x01234567 = vextq_u8(vout6x01234567_7x01234567, vout6x01234567_7x01234567, 2);
        vout4x01234567_5x01234567 = vextq_u8(vout4x01234567_5x01234567, vout4x01234567_5x01234567, 2);
        vout2x01234567_3x01234567 = vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
        vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_u8(c7, vout6x01234567_7x01234567, 8);
        vst1q_lane_u8(c6, vout6x01234567_7x01234567, 0);
        vst1q_lane_u8(c5, vout4x01234567_5x01234567, 8);
        vst1q_lane_u8(c4, vout4x01234567_5x01234567, 0);
        vst1q_lane_u8(c3, vout2x01234567_3x01234567, 8);
        vst1q_lane_u8(c2, vout2x01234567_3x01234567, 0);
        vst1q_lane_u8(c1, vout0x01234567_1x01234567, 8);
        vst1q_lane_u8(c0, vout0x01234567_1x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
