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

#include <xnnpack/gemm.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>


void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c8__neoni8mm(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const int8_t* a4 = (const int8_t*) ((uintptr_t) a3 + a_stride);
  int8_t* c4 = (int8_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const int8_t* a5 = (const int8_t*) ((uintptr_t) a4 + a_stride);
  int8_t* c5 = (int8_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  // Loop over groups of 16 columns.
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
    int32x4_t vacc01x01 = vreinterpretq_s32_u64(vbias01x0123.val[0]);
    int32x4_t vacc01x23 = vreinterpretq_s32_u64(vbias01x0123.val[1]);
    int32x4_t vacc01x45 = vreinterpretq_s32_u64(vbias01x4567.val[0]);
    int32x4_t vacc01x67 = vreinterpretq_s32_u64(vbias01x4567.val[1]);
    int32x4_t vacc01x89 = vreinterpretq_s32_u64(vbias01x89AB.val[0]);
    int32x4_t vacc01xAB = vreinterpretq_s32_u64(vbias01x89AB.val[1]);
    int32x4_t vacc01xCD = vreinterpretq_s32_u64(vbias01xCDEF.val[0]);
    int32x4_t vacc01xEF = vreinterpretq_s32_u64(vbias01xCDEF.val[1]);
    int32x4_t vacc23x01 = vreinterpretq_s32_u64(vbias01x0123.val[0]);
    int32x4_t vacc23x23 = vreinterpretq_s32_u64(vbias01x0123.val[1]);
    int32x4_t vacc23x45 = vreinterpretq_s32_u64(vbias01x4567.val[0]);
    int32x4_t vacc23x67 = vreinterpretq_s32_u64(vbias01x4567.val[1]);
    int32x4_t vacc23x89 = vreinterpretq_s32_u64(vbias01x89AB.val[0]);
    int32x4_t vacc23xAB = vreinterpretq_s32_u64(vbias01x89AB.val[1]);
    int32x4_t vacc23xCD = vreinterpretq_s32_u64(vbias01xCDEF.val[0]);
    int32x4_t vacc23xEF = vreinterpretq_s32_u64(vbias01xCDEF.val[1]);
    int32x4_t vacc45x01 = vreinterpretq_s32_u64(vbias01x0123.val[0]);
    int32x4_t vacc45x23 = vreinterpretq_s32_u64(vbias01x0123.val[1]);
    int32x4_t vacc45x45 = vreinterpretq_s32_u64(vbias01x4567.val[0]);
    int32x4_t vacc45x67 = vreinterpretq_s32_u64(vbias01x4567.val[1]);
    int32x4_t vacc45x89 = vreinterpretq_s32_u64(vbias01x89AB.val[0]);
    int32x4_t vacc45xAB = vreinterpretq_s32_u64(vbias01x89AB.val[1]);
    int32x4_t vacc45xCD = vreinterpretq_s32_u64(vbias01xCDEF.val[0]);
    int32x4_t vacc45xEF = vreinterpretq_s32_u64(vbias01xCDEF.val[1]);

    // Inner accumulation loop along the 16 columns.
    size_t k = kc;
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
      // Load a 6x16 block of activations.
      #if XNN_ARCH_ARM64
        va01x0123456789ABCDEF = vld2q_lane_u64((const void*) a0, va01x0123456789ABCDEF, 0); a0 += 16;
        va23x0123456789ABCDEF = vld2q_lane_u64((const void*) a2, va23x0123456789ABCDEF, 0); a2 += 16;
        va45x0123456789ABCDEF = vld2q_lane_u64((const void*) a4, va45x0123456789ABCDEF, 0); a4 += 16;
        va01x0123456789ABCDEF = vld2q_lane_u64((const void*) a1, va01x0123456789ABCDEF, 1); a1 += 16;
        va23x0123456789ABCDEF = vld2q_lane_u64((const void*) a3, va23x0123456789ABCDEF, 1); a3 += 16;
        va45x0123456789ABCDEF = vld2q_lane_u64((const void*) a5, va45x0123456789ABCDEF, 1); a5 += 16;
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
        va45x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a5, va45x0123456789ABCDEF.val[0], 1); a5 += 8;
        va45x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a5, va45x0123456789ABCDEF.val[1], 1); a5 += 8;
      #endif

      const int8x16_t va01x01234567 = vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]);
      const int8x16_t va01x89ABCDEF = vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]);
      const int8x16_t va23x01234567 = vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[0]);
      const int8x16_t va23x89ABCDEF = vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[1]);
      const int8x16_t va45x01234567 = vreinterpretq_s8_u64(va45x0123456789ABCDEF.val[0]);
      const int8x16_t va45x89ABCDEF = vreinterpretq_s8_u64(va45x0123456789ABCDEF.val[1]);

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

      // Multiply-accumulate: 6x8 * 8x16 --> 6x16.
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
      // Load a 6x8 block of activations.
      uint64x2_t va01x01234567 = vld1q_dup_u64((const void*) a0); a0 += 8;
      uint64x2_t va23x01234567 = vld1q_dup_u64((const void*) a2); a2 += 8;
      uint64x2_t va45x01234567 = vld1q_dup_u64((const void*) a4); a4 += 8;
      va01x01234567 = vld1q_lane_u64((const void*) a1, va01x01234567, 1); a1 += 8;
      va23x01234567 = vld1q_lane_u64((const void*) a3, va23x01234567, 1); a3 += 8;
      va45x01234567 = vld1q_lane_u64((const void*) a5, va45x01234567, 1); a5 += 8;

      // Load a 16x16 block of weights.
      const int8x16_t vb01x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb23x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb45x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb67x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb89x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbABx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbCDx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbEFx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 6x4 * 4x16 --> 6x16.
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
      int32x4_t vacc5x0123 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc45x01), vreinterpretq_u64_s32(vacc45x23)));
      int32x4_t vacc4x4567 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc45x45), vreinterpretq_u64_s32(vacc45x67)));
      int32x4_t vacc5x4567 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc45x45), vreinterpretq_u64_s32(vacc45x67)));
      int32x4_t vacc4x89AB = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc45x89), vreinterpretq_u64_s32(vacc45xAB)));
      int32x4_t vacc5x89AB = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc45x89), vreinterpretq_u64_s32(vacc45xAB)));
      int32x4_t vacc4xCDEF = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc45xCD), vreinterpretq_u64_s32(vacc45xEF)));
      int32x4_t vacc5xCDEF = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc45xCD), vreinterpretq_u64_s32(vacc45xEF)));
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
    #endif

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
    float32x4_t vfpacc4x0123 = vcvtq_f32_s32(vacc4x0123);
    float32x4_t vfpacc4x4567 = vcvtq_f32_s32(vacc4x4567);
    float32x4_t vfpacc4x89AB = vcvtq_f32_s32(vacc4x89AB);
    float32x4_t vfpacc4xCDEF = vcvtq_f32_s32(vacc4xCDEF);
    float32x4_t vfpacc5x0123 = vcvtq_f32_s32(vacc5x0123);
    float32x4_t vfpacc5x4567 = vcvtq_f32_s32(vacc5x4567);
    float32x4_t vfpacc5x89AB = vcvtq_f32_s32(vacc5x89AB);
    float32x4_t vfpacc5xCDEF = vcvtq_f32_s32(vacc5xCDEF);

    const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale0123);
    vfpacc1x0123 = vmulq_f32(vfpacc1x0123, vscale0123);
    vfpacc2x0123 = vmulq_f32(vfpacc2x0123, vscale0123);
    vfpacc3x0123 = vmulq_f32(vfpacc3x0123, vscale0123);
    vfpacc4x0123 = vmulq_f32(vfpacc4x0123, vscale0123);
    vfpacc5x0123 = vmulq_f32(vfpacc5x0123, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale4567);
    vfpacc1x4567 = vmulq_f32(vfpacc1x4567, vscale4567);
    vfpacc2x4567 = vmulq_f32(vfpacc2x4567, vscale4567);
    vfpacc3x4567 = vmulq_f32(vfpacc3x4567, vscale4567);
    vfpacc4x4567 = vmulq_f32(vfpacc4x4567, vscale4567);
    vfpacc5x4567 = vmulq_f32(vfpacc5x4567, vscale4567);
    const float32x4_t vscale89AB = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x89AB = vmulq_f32(vfpacc0x89AB, vscale89AB);
    vfpacc1x89AB = vmulq_f32(vfpacc1x89AB, vscale89AB);
    vfpacc2x89AB = vmulq_f32(vfpacc2x89AB, vscale89AB);
    vfpacc3x89AB = vmulq_f32(vfpacc3x89AB, vscale89AB);
    vfpacc4x89AB = vmulq_f32(vfpacc4x89AB, vscale89AB);
    vfpacc5x89AB = vmulq_f32(vfpacc5x89AB, vscale89AB);
    const float32x4_t vscaleCDEF = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0xCDEF = vmulq_f32(vfpacc0xCDEF, vscaleCDEF);
    vfpacc1xCDEF = vmulq_f32(vfpacc1xCDEF, vscaleCDEF);
    vfpacc2xCDEF = vmulq_f32(vfpacc2xCDEF, vscaleCDEF);
    vfpacc3xCDEF = vmulq_f32(vfpacc3xCDEF, vscaleCDEF);
    vfpacc4xCDEF = vmulq_f32(vfpacc4xCDEF, vscaleCDEF);
    vfpacc5xCDEF = vmulq_f32(vfpacc5xCDEF, vscaleCDEF);

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
    vacc4x0123 = vcvtnq_s32_f32(vfpacc4x0123);
    vacc4x4567 = vcvtnq_s32_f32(vfpacc4x4567);
    vacc4x89AB = vcvtnq_s32_f32(vfpacc4x89AB);
    vacc4xCDEF = vcvtnq_s32_f32(vfpacc4xCDEF);
    vacc5x0123 = vcvtnq_s32_f32(vfpacc5x0123);
    vacc5x4567 = vcvtnq_s32_f32(vfpacc5x4567);
    vacc5x89AB = vcvtnq_s32_f32(vfpacc5x89AB);
    vacc5xCDEF = vcvtnq_s32_f32(vfpacc5xCDEF);

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
      const int16x8_t vacc4x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc4x0123), vacc4x4567), voutput_zero_point);
      const int16x8_t vacc4x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc4x89AB), vacc4xCDEF), voutput_zero_point);
      const int16x8_t vacc5x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc5x0123), vacc5x4567), voutput_zero_point);
      const int16x8_t vacc5x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc5x89AB), vacc5xCDEF), voutput_zero_point);

      int8x16_t vout0x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc0x89ABCDEF);
      int8x16_t vout1x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc1x01234567), vacc1x89ABCDEF);
      int8x16_t vout2x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc2x01234567), vacc2x89ABCDEF);
      int8x16_t vout3x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc3x01234567), vacc3x89ABCDEF);
      int8x16_t vout4x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc4x01234567), vacc4x89ABCDEF);
      int8x16_t vout5x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc5x01234567), vacc5x89ABCDEF);
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

      int8x16_t vout0x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc0x89ABCDEF));
      int8x16_t vout1x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc1x01234567), vqmovn_s16(vacc1x89ABCDEF));
      int8x16_t vout2x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc2x01234567), vqmovn_s16(vacc2x89ABCDEF));
      int8x16_t vout3x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc3x01234567), vqmovn_s16(vacc3x89ABCDEF));
      int8x16_t vout4x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc4x01234567), vqmovn_s16(vacc4x89ABCDEF));
      int8x16_t vout5x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc5x01234567), vqmovn_s16(vacc5x89ABCDEF));
    #endif
    const int8x16_t voutput_min = vld1q_dup_s8(&params->fp32_neonv8.output_min);
    const int8x16_t voutput_max = vld1q_dup_s8(&params->fp32_neonv8.output_max);

    vout0x0123456789ABCDEF = vmaxq_s8(vout0x0123456789ABCDEF, voutput_min);
    vout1x0123456789ABCDEF = vmaxq_s8(vout1x0123456789ABCDEF, voutput_min);
    vout2x0123456789ABCDEF = vmaxq_s8(vout2x0123456789ABCDEF, voutput_min);
    vout3x0123456789ABCDEF = vmaxq_s8(vout3x0123456789ABCDEF, voutput_min);
    vout4x0123456789ABCDEF = vmaxq_s8(vout4x0123456789ABCDEF, voutput_min);
    vout5x0123456789ABCDEF = vmaxq_s8(vout5x0123456789ABCDEF, voutput_min);

    vout0x0123456789ABCDEF = vminq_s8(vout0x0123456789ABCDEF, voutput_max);
    vout1x0123456789ABCDEF = vminq_s8(vout1x0123456789ABCDEF, voutput_max);
    vout2x0123456789ABCDEF = vminq_s8(vout2x0123456789ABCDEF, voutput_max);
    vout3x0123456789ABCDEF = vminq_s8(vout3x0123456789ABCDEF, voutput_max);
    vout4x0123456789ABCDEF = vminq_s8(vout4x0123456789ABCDEF, voutput_max);
    vout5x0123456789ABCDEF = vminq_s8(vout5x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      // Main case where there the 16 columns fit in the destination.
      vst1q_s8(c0 + 0, vout0x0123456789ABCDEF);
      vst1q_s8(c1 + 0, vout1x0123456789ABCDEF);
      vst1q_s8(c2 + 0, vout2x0123456789ABCDEF);
      vst1q_s8(c3 + 0, vout3x0123456789ABCDEF);
      vst1q_s8(c4 + 0, vout4x0123456789ABCDEF);
      vst1q_s8(c5 + 0, vout5x0123456789ABCDEF);

      // Advance to the next 16 columns.
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      c4 = (int8_t*) ((uintptr_t) c4 + cn_stride);
      c5 = (int8_t*) ((uintptr_t) c5 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);
      a4 = (const int8_t*) ((uintptr_t) a4 - kc);
      a5 = (const int8_t*) ((uintptr_t) a5 - kc);

      nc -= 16;
    } else {
      // Final case where not all of the 16 columns fit in the destination.
      int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vget_low_s8(vout0x0123456789ABCDEF), vget_low_s8(vout1x0123456789ABCDEF));
      int8x16_t vout2x01234567_3x01234567 = vcombine_s8(vget_low_s8(vout2x0123456789ABCDEF), vget_low_s8(vout3x0123456789ABCDEF));
      int8x16_t vout4x01234567_5x01234567 = vcombine_s8(vget_low_s8(vout4x0123456789ABCDEF), vget_low_s8(vout5x0123456789ABCDEF));
      if (nc & 8) {
        vst1_s8(c0, vget_low_s8(vout0x01234567_1x01234567)); c0 += 8;
        vst1_s8(c1, vget_high_s8(vout0x01234567_1x01234567)); c1 += 8;
        vst1_s8(c2, vget_low_s8(vout2x01234567_3x01234567)); c2 += 8;
        vst1_s8(c3, vget_high_s8(vout2x01234567_3x01234567)); c3 += 8;
        vst1_s8(c4, vget_low_s8(vout4x01234567_5x01234567)); c4 += 8;
        vst1_s8(c5, vget_high_s8(vout4x01234567_5x01234567)); c5 += 8;
        vout0x01234567_1x01234567 = vcombine_s8(vget_high_s8(vout0x0123456789ABCDEF), vget_high_s8(vout1x0123456789ABCDEF));
        vout2x01234567_3x01234567 = vcombine_s8(vget_high_s8(vout2x0123456789ABCDEF), vget_high_s8(vout3x0123456789ABCDEF));
        vout4x01234567_5x01234567 = vcombine_s8(vget_high_s8(vout4x0123456789ABCDEF), vget_high_s8(vout5x0123456789ABCDEF));
      }
      if (nc & 4) {
        vst1q_lane_u32((void*) c0, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 0); c0 += 4;
        vst1q_lane_u32((void*) c1, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1q_lane_u32((void*) c2, vreinterpretq_u32_s8(vout2x01234567_3x01234567), 0); c2 += 4;
        vst1q_lane_u32((void*) c3, vreinterpretq_u32_s8(vout2x01234567_3x01234567), 2); c3 += 4;
        vst1q_lane_u32((void*) c4, vreinterpretq_u32_s8(vout4x01234567_5x01234567), 0); c4 += 4;
        vst1q_lane_u32((void*) c5, vreinterpretq_u32_s8(vout4x01234567_5x01234567), 2); c5 += 4;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
        vout4x01234567_5x01234567 = vextq_s8(vout4x01234567_5x01234567, vout4x01234567_5x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16((void*) c0, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 0); c0 += 2;
        vst1q_lane_u16((void*) c1, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1q_lane_u16((void*) c2, vreinterpretq_u16_s8(vout2x01234567_3x01234567), 0); c2 += 2;
        vst1q_lane_u16((void*) c3, vreinterpretq_u16_s8(vout2x01234567_3x01234567), 4); c3 += 2;
        vst1q_lane_u16((void*) c4, vreinterpretq_u16_s8(vout4x01234567_5x01234567), 0); c4 += 2;
        vst1q_lane_u16((void*) c5, vreinterpretq_u16_s8(vout4x01234567_5x01234567), 4); c5 += 2;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
        vout4x01234567_5x01234567 = vextq_s8(vout4x01234567_5x01234567, vout4x01234567_5x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_s8(c0, vout0x01234567_1x01234567, 0);
        vst1q_lane_s8(c1, vout0x01234567_1x01234567, 8);
        vst1q_lane_s8(c2, vout2x01234567_3x01234567, 0);
        vst1q_lane_s8(c3, vout2x01234567_3x01234567, 8);
        vst1q_lane_s8(c4, vout4x01234567_5x01234567, 0);
        vst1q_lane_s8(c5, vout4x01234567_5x01234567, 8);
      }

      nc = 0;
    }
  } while (nc != 0);
}
