// Auto-generated file. Do not edit!
//   Template: src/qu8-igemm/c4-neondot.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/igemm.h>
#include <xnnpack/math.h>


void xnn_qu8_igemm_minmax_rndnu_ukernel_8x8c4__neondot(
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

  kc = round_up_po2(kc, 4 * sizeof(uint8_t));
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

  const uint8x8_t va_zero_point = vld1_dup_u8(&params->rndnu_neon.kernel_zero_point[0]);

  do {
    // Initialize accumulators with bias. 8 bias values are loaded from the
    // weight matrix, at the start of the group of 8 columns.
    uint32x4_t vpacc0x0123 = vld1q_u32(w); w = (const uint32_t*) w + 4;
    uint32x4_t vpacc0x4567 = vld1q_u32(w); w = (const uint32_t*) w + 4;
    uint32x4_t vpacc1x0123 = vpacc0x0123;
    uint32x4_t vpacc1x4567 = vpacc0x4567;
    uint32x4_t vpacc2x0123 = vpacc0x0123;
    uint32x4_t vpacc2x4567 = vpacc0x4567;
    uint32x4_t vpacc3x0123 = vpacc0x0123;
    uint32x4_t vpacc3x4567 = vpacc0x4567;
    uint32x4_t vpacc4x0123 = vpacc0x0123;
    uint32x4_t vpacc4x4567 = vpacc0x4567;
    uint32x4_t vpacc5x0123 = vpacc0x0123;
    uint32x4_t vpacc5x4567 = vpacc0x4567;
    uint32x4_t vpacc6x0123 = vpacc0x0123;
    uint32x4_t vpacc6x4567 = vpacc0x4567;
    uint32x4_t vpacc7x0123 = vpacc0x0123;
    uint32x4_t vpacc7x4567 = vpacc0x4567;
    uint32x2_t vnacc0 = vmov_n_u32(0);
    uint32x2_t vnacc1 = vmov_n_u32(0);
    uint32x2_t vnacc2 = vmov_n_u32(0);
    uint32x2_t vnacc3 = vmov_n_u32(0);
    uint32x2_t vnacc4 = vmov_n_u32(0);
    uint32x2_t vnacc5 = vmov_n_u32(0);
    uint32x2_t vnacc6 = vmov_n_u32(0);
    uint32x2_t vnacc7 = vmov_n_u32(0);

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

      // Inner accumulation loop along the 8 columns.
      size_t k = kc;
      // 2x partial unrolled loop to load 8 bytes at a time.
      while (k >= 8 * sizeof(uint8_t)) {
        // Load a 8x8 block of activations.
        const uint8x8_t va0x01234567 = vld1_u8(a0); a0 += 8;
        const uint8x8_t va1x01234567 = vld1_u8(a1); a1 += 8;
        const uint8x8_t va2x01234567 = vld1_u8(a2); a2 += 8;
        const uint8x8_t va3x01234567 = vld1_u8(a3); a3 += 8;
        const uint8x8_t va4x01234567 = vld1_u8(a4); a4 += 8;
        const uint8x8_t va5x01234567 = vld1_u8(a5); a5 += 8;
        const uint8x8_t va6x01234567 = vld1_u8(a6); a6 += 8;
        const uint8x8_t va7x01234567 = vld1_u8(a7); a7 += 8;

        // Load a 8x8 block of weights.
        const uint8x16_t vb0123x0123 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint8x16_t vb0123x4567 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint8x16_t vb4567x0123 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint8x16_t vb4567x4567 = vld1q_u8(w); w = (const uint8_t*) w + 16;

        // Multiply-accumulate: 8x8 * 8x8 --> 8x8.
        vnacc0 = vdot_u32(vnacc0, va_zero_point, va0x01234567);
        vpacc0x0123 = vdotq_lane_u32(vpacc0x0123, vb0123x0123, va0x01234567, 0);
        vpacc0x4567 = vdotq_lane_u32(vpacc0x4567, vb0123x4567, va0x01234567, 0);
        vpacc0x0123 = vdotq_lane_u32(vpacc0x0123, vb4567x0123, va0x01234567, 1);
        vpacc0x4567 = vdotq_lane_u32(vpacc0x4567, vb4567x4567, va0x01234567, 1);
        vnacc1 = vdot_u32(vnacc1, va_zero_point, va1x01234567);
        vpacc1x0123 = vdotq_lane_u32(vpacc1x0123, vb0123x0123, va1x01234567, 0);
        vpacc1x4567 = vdotq_lane_u32(vpacc1x4567, vb0123x4567, va1x01234567, 0);
        vpacc1x0123 = vdotq_lane_u32(vpacc1x0123, vb4567x0123, va1x01234567, 1);
        vpacc1x4567 = vdotq_lane_u32(vpacc1x4567, vb4567x4567, va1x01234567, 1);
        vnacc2 = vdot_u32(vnacc2, va_zero_point, va2x01234567);
        vpacc2x0123 = vdotq_lane_u32(vpacc2x0123, vb0123x0123, va2x01234567, 0);
        vpacc2x4567 = vdotq_lane_u32(vpacc2x4567, vb0123x4567, va2x01234567, 0);
        vpacc2x0123 = vdotq_lane_u32(vpacc2x0123, vb4567x0123, va2x01234567, 1);
        vpacc2x4567 = vdotq_lane_u32(vpacc2x4567, vb4567x4567, va2x01234567, 1);
        vnacc3 = vdot_u32(vnacc3, va_zero_point, va3x01234567);
        vpacc3x0123 = vdotq_lane_u32(vpacc3x0123, vb0123x0123, va3x01234567, 0);
        vpacc3x4567 = vdotq_lane_u32(vpacc3x4567, vb0123x4567, va3x01234567, 0);
        vpacc3x0123 = vdotq_lane_u32(vpacc3x0123, vb4567x0123, va3x01234567, 1);
        vpacc3x4567 = vdotq_lane_u32(vpacc3x4567, vb4567x4567, va3x01234567, 1);
        vnacc4 = vdot_u32(vnacc4, va_zero_point, va4x01234567);
        vpacc4x0123 = vdotq_lane_u32(vpacc4x0123, vb0123x0123, va4x01234567, 0);
        vpacc4x4567 = vdotq_lane_u32(vpacc4x4567, vb0123x4567, va4x01234567, 0);
        vpacc4x0123 = vdotq_lane_u32(vpacc4x0123, vb4567x0123, va4x01234567, 1);
        vpacc4x4567 = vdotq_lane_u32(vpacc4x4567, vb4567x4567, va4x01234567, 1);
        vnacc5 = vdot_u32(vnacc5, va_zero_point, va5x01234567);
        vpacc5x0123 = vdotq_lane_u32(vpacc5x0123, vb0123x0123, va5x01234567, 0);
        vpacc5x4567 = vdotq_lane_u32(vpacc5x4567, vb0123x4567, va5x01234567, 0);
        vpacc5x0123 = vdotq_lane_u32(vpacc5x0123, vb4567x0123, va5x01234567, 1);
        vpacc5x4567 = vdotq_lane_u32(vpacc5x4567, vb4567x4567, va5x01234567, 1);
        vnacc6 = vdot_u32(vnacc6, va_zero_point, va6x01234567);
        vpacc6x0123 = vdotq_lane_u32(vpacc6x0123, vb0123x0123, va6x01234567, 0);
        vpacc6x4567 = vdotq_lane_u32(vpacc6x4567, vb0123x4567, va6x01234567, 0);
        vpacc6x0123 = vdotq_lane_u32(vpacc6x0123, vb4567x0123, va6x01234567, 1);
        vpacc6x4567 = vdotq_lane_u32(vpacc6x4567, vb4567x4567, va6x01234567, 1);
        vnacc7 = vdot_u32(vnacc7, va_zero_point, va7x01234567);
        vpacc7x0123 = vdotq_lane_u32(vpacc7x0123, vb0123x0123, va7x01234567, 0);
        vpacc7x4567 = vdotq_lane_u32(vpacc7x4567, vb0123x4567, va7x01234567, 0);
        vpacc7x0123 = vdotq_lane_u32(vpacc7x0123, vb4567x0123, va7x01234567, 1);
        vpacc7x4567 = vdotq_lane_u32(vpacc7x4567, vb4567x4567, va7x01234567, 1);

        k -= 8 * sizeof(uint8_t);
      }
      // Handle up to 4 final positions of `k`
      if XNN_UNLIKELY(k != 0) {
        // Load a 8x4 block of activations.
        const uint8x8_t va0x01234567 = vreinterpret_u8_u32(vld1_lane_u32((const void*) a0, vmov_n_u32(0), 0)); a0 += 4;
        const uint8x8_t va1x01234567 = vreinterpret_u8_u32(vld1_lane_u32((const void*) a1, vmov_n_u32(0), 0)); a1 += 4;
        const uint8x8_t va2x01234567 = vreinterpret_u8_u32(vld1_lane_u32((const void*) a2, vmov_n_u32(0), 0)); a2 += 4;
        const uint8x8_t va3x01234567 = vreinterpret_u8_u32(vld1_lane_u32((const void*) a3, vmov_n_u32(0), 0)); a3 += 4;
        const uint8x8_t va4x01234567 = vreinterpret_u8_u32(vld1_lane_u32((const void*) a4, vmov_n_u32(0), 0)); a4 += 4;
        const uint8x8_t va5x01234567 = vreinterpret_u8_u32(vld1_lane_u32((const void*) a5, vmov_n_u32(0), 0)); a5 += 4;
        const uint8x8_t va6x01234567 = vreinterpret_u8_u32(vld1_lane_u32((const void*) a6, vmov_n_u32(0), 0)); a6 += 4;
        const uint8x8_t va7x01234567 = vreinterpret_u8_u32(vld1_lane_u32((const void*) a7, vmov_n_u32(0), 0)); a7 += 4;

        // Load a 4x8 block of weights.
        const uint8x16_t vb0123x0123 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint8x16_t vb0123x4567 = vld1q_u8(w); w = (const uint8_t*) w + 16;

        // Multiply-accumulate: 8x4 * 4x8 --> 8x8.
        vnacc0 = vdot_u32(vnacc0, va_zero_point, va0x01234567);
        vpacc0x0123 = vdotq_lane_u32(vpacc0x0123, vb0123x0123, va0x01234567, 0);
        vpacc0x4567 = vdotq_lane_u32(vpacc0x4567, vb0123x4567, va0x01234567, 0);
        vnacc1 = vdot_u32(vnacc1, va_zero_point, va1x01234567);
        vpacc1x0123 = vdotq_lane_u32(vpacc1x0123, vb0123x0123, va1x01234567, 0);
        vpacc1x4567 = vdotq_lane_u32(vpacc1x4567, vb0123x4567, va1x01234567, 0);
        vnacc2 = vdot_u32(vnacc2, va_zero_point, va2x01234567);
        vpacc2x0123 = vdotq_lane_u32(vpacc2x0123, vb0123x0123, va2x01234567, 0);
        vpacc2x4567 = vdotq_lane_u32(vpacc2x4567, vb0123x4567, va2x01234567, 0);
        vnacc3 = vdot_u32(vnacc3, va_zero_point, va3x01234567);
        vpacc3x0123 = vdotq_lane_u32(vpacc3x0123, vb0123x0123, va3x01234567, 0);
        vpacc3x4567 = vdotq_lane_u32(vpacc3x4567, vb0123x4567, va3x01234567, 0);
        vnacc4 = vdot_u32(vnacc4, va_zero_point, va4x01234567);
        vpacc4x0123 = vdotq_lane_u32(vpacc4x0123, vb0123x0123, va4x01234567, 0);
        vpacc4x4567 = vdotq_lane_u32(vpacc4x4567, vb0123x4567, va4x01234567, 0);
        vnacc5 = vdot_u32(vnacc5, va_zero_point, va5x01234567);
        vpacc5x0123 = vdotq_lane_u32(vpacc5x0123, vb0123x0123, va5x01234567, 0);
        vpacc5x4567 = vdotq_lane_u32(vpacc5x4567, vb0123x4567, va5x01234567, 0);
        vnacc6 = vdot_u32(vnacc6, va_zero_point, va6x01234567);
        vpacc6x0123 = vdotq_lane_u32(vpacc6x0123, vb0123x0123, va6x01234567, 0);
        vpacc6x4567 = vdotq_lane_u32(vpacc6x4567, vb0123x4567, va6x01234567, 0);
        vnacc7 = vdot_u32(vnacc7, va_zero_point, va7x01234567);
        vpacc7x0123 = vdotq_lane_u32(vpacc7x0123, vb0123x0123, va7x01234567, 0);
        vpacc7x4567 = vdotq_lane_u32(vpacc7x4567, vb0123x4567, va7x01234567, 0);
      }
      p -= 8 * sizeof(void*);
    } while (p != 0);

    // Subtract zero point from accumulators.
    vnacc0 = vpadd_u32(vnacc0, vnacc0);
    const uint32x4_t vnacc0x0123 = vcombine_u32(vnacc0, vnacc0);
    int32x4_t vacc0x0123 = vreinterpretq_s32_u32(vsubq_u32(vpacc0x0123, vnacc0x0123));
    int32x4_t vacc0x4567 = vreinterpretq_s32_u32(vsubq_u32(vpacc0x4567, vnacc0x0123));
    vnacc1 = vpadd_u32(vnacc1, vnacc1);
    const uint32x4_t vnacc1x0123 = vcombine_u32(vnacc1, vnacc1);
    int32x4_t vacc1x0123 = vreinterpretq_s32_u32(vsubq_u32(vpacc1x0123, vnacc1x0123));
    int32x4_t vacc1x4567 = vreinterpretq_s32_u32(vsubq_u32(vpacc1x4567, vnacc1x0123));
    vnacc2 = vpadd_u32(vnacc2, vnacc2);
    const uint32x4_t vnacc2x0123 = vcombine_u32(vnacc2, vnacc2);
    int32x4_t vacc2x0123 = vreinterpretq_s32_u32(vsubq_u32(vpacc2x0123, vnacc2x0123));
    int32x4_t vacc2x4567 = vreinterpretq_s32_u32(vsubq_u32(vpacc2x4567, vnacc2x0123));
    vnacc3 = vpadd_u32(vnacc3, vnacc3);
    const uint32x4_t vnacc3x0123 = vcombine_u32(vnacc3, vnacc3);
    int32x4_t vacc3x0123 = vreinterpretq_s32_u32(vsubq_u32(vpacc3x0123, vnacc3x0123));
    int32x4_t vacc3x4567 = vreinterpretq_s32_u32(vsubq_u32(vpacc3x4567, vnacc3x0123));
    vnacc4 = vpadd_u32(vnacc4, vnacc4);
    const uint32x4_t vnacc4x0123 = vcombine_u32(vnacc4, vnacc4);
    int32x4_t vacc4x0123 = vreinterpretq_s32_u32(vsubq_u32(vpacc4x0123, vnacc4x0123));
    int32x4_t vacc4x4567 = vreinterpretq_s32_u32(vsubq_u32(vpacc4x4567, vnacc4x0123));
    vnacc5 = vpadd_u32(vnacc5, vnacc5);
    const uint32x4_t vnacc5x0123 = vcombine_u32(vnacc5, vnacc5);
    int32x4_t vacc5x0123 = vreinterpretq_s32_u32(vsubq_u32(vpacc5x0123, vnacc5x0123));
    int32x4_t vacc5x4567 = vreinterpretq_s32_u32(vsubq_u32(vpacc5x4567, vnacc5x0123));
    vnacc6 = vpadd_u32(vnacc6, vnacc6);
    const uint32x4_t vnacc6x0123 = vcombine_u32(vnacc6, vnacc6);
    int32x4_t vacc6x0123 = vreinterpretq_s32_u32(vsubq_u32(vpacc6x0123, vnacc6x0123));
    int32x4_t vacc6x4567 = vreinterpretq_s32_u32(vsubq_u32(vpacc6x4567, vnacc6x0123));
    vnacc7 = vpadd_u32(vnacc7, vnacc7);
    const uint32x4_t vnacc7x0123 = vcombine_u32(vnacc7, vnacc7);
    int32x4_t vacc7x0123 = vreinterpretq_s32_u32(vsubq_u32(vpacc7x0123, vnacc7x0123));
    int32x4_t vacc7x4567 = vreinterpretq_s32_u32(vsubq_u32(vpacc7x4567, vnacc7x0123));

    const int32x4_t vright_pre_shift = vld1q_dup_s32(&params->rndnu_neon.right_pre_shift);
    const int32x4_t vmultiplier = vld1q_dup_s32(&params->rndnu_neon.multiplier);
    const int32x4_t vright_post_shift = vld1q_dup_s32(&params->rndnu_neon.right_post_shift);

    vacc0x0123 = vshlq_s32(vacc0x0123, vright_pre_shift);
    vacc0x4567 = vshlq_s32(vacc0x4567, vright_pre_shift);
    vacc1x0123 = vshlq_s32(vacc1x0123, vright_pre_shift);
    vacc1x4567 = vshlq_s32(vacc1x4567, vright_pre_shift);
    vacc2x0123 = vshlq_s32(vacc2x0123, vright_pre_shift);
    vacc2x4567 = vshlq_s32(vacc2x4567, vright_pre_shift);
    vacc3x0123 = vshlq_s32(vacc3x0123, vright_pre_shift);
    vacc3x4567 = vshlq_s32(vacc3x4567, vright_pre_shift);
    vacc4x0123 = vshlq_s32(vacc4x0123, vright_pre_shift);
    vacc4x4567 = vshlq_s32(vacc4x4567, vright_pre_shift);
    vacc5x0123 = vshlq_s32(vacc5x0123, vright_pre_shift);
    vacc5x4567 = vshlq_s32(vacc5x4567, vright_pre_shift);
    vacc6x0123 = vshlq_s32(vacc6x0123, vright_pre_shift);
    vacc6x4567 = vshlq_s32(vacc6x4567, vright_pre_shift);
    vacc7x0123 = vshlq_s32(vacc7x0123, vright_pre_shift);
    vacc7x4567 = vshlq_s32(vacc7x4567, vright_pre_shift);

    vacc0x0123 = vqdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqdmulhq_s32(vacc0x4567, vmultiplier);
    vacc1x0123 = vqdmulhq_s32(vacc1x0123, vmultiplier);
    vacc1x4567 = vqdmulhq_s32(vacc1x4567, vmultiplier);
    vacc2x0123 = vqdmulhq_s32(vacc2x0123, vmultiplier);
    vacc2x4567 = vqdmulhq_s32(vacc2x4567, vmultiplier);
    vacc3x0123 = vqdmulhq_s32(vacc3x0123, vmultiplier);
    vacc3x4567 = vqdmulhq_s32(vacc3x4567, vmultiplier);
    vacc4x0123 = vqdmulhq_s32(vacc4x0123, vmultiplier);
    vacc4x4567 = vqdmulhq_s32(vacc4x4567, vmultiplier);
    vacc5x0123 = vqdmulhq_s32(vacc5x0123, vmultiplier);
    vacc5x4567 = vqdmulhq_s32(vacc5x4567, vmultiplier);
    vacc6x0123 = vqdmulhq_s32(vacc6x0123, vmultiplier);
    vacc6x4567 = vqdmulhq_s32(vacc6x4567, vmultiplier);
    vacc7x0123 = vqdmulhq_s32(vacc7x0123, vmultiplier);
    vacc7x4567 = vqdmulhq_s32(vacc7x4567, vmultiplier);

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_post_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_post_shift);
    vacc1x0123 = vrshlq_s32(vacc1x0123, vright_post_shift);
    vacc1x4567 = vrshlq_s32(vacc1x4567, vright_post_shift);
    vacc2x0123 = vrshlq_s32(vacc2x0123, vright_post_shift);
    vacc2x4567 = vrshlq_s32(vacc2x4567, vright_post_shift);
    vacc3x0123 = vrshlq_s32(vacc3x0123, vright_post_shift);
    vacc3x4567 = vrshlq_s32(vacc3x4567, vright_post_shift);
    vacc4x0123 = vrshlq_s32(vacc4x0123, vright_post_shift);
    vacc4x4567 = vrshlq_s32(vacc4x4567, vright_post_shift);
    vacc5x0123 = vrshlq_s32(vacc5x0123, vright_post_shift);
    vacc5x4567 = vrshlq_s32(vacc5x4567, vright_post_shift);
    vacc6x0123 = vrshlq_s32(vacc6x0123, vright_post_shift);
    vacc6x4567 = vrshlq_s32(vacc6x4567, vright_post_shift);
    vacc7x0123 = vrshlq_s32(vacc7x0123, vright_post_shift);
    vacc7x4567 = vrshlq_s32(vacc7x4567, vright_post_shift);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->rndnu_neon.output_zero_point);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
    const int16x8_t vacc3x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);
    const int16x8_t vacc4x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc4x0123), vacc4x4567), voutput_zero_point);
    const int16x8_t vacc5x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc5x0123), vacc5x4567), voutput_zero_point);
    const int16x8_t vacc6x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc6x0123), vacc6x4567), voutput_zero_point);
    const int16x8_t vacc7x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc7x0123), vacc7x4567), voutput_zero_point);

    uint8x16_t vout0x01234567_1x01234567 = vqmovun_high_s16(vqmovun_s16(vacc0x01234567), vacc1x01234567);
    uint8x16_t vout2x01234567_3x01234567 = vqmovun_high_s16(vqmovun_s16(vacc2x01234567), vacc3x01234567);
    uint8x16_t vout4x01234567_5x01234567 = vqmovun_high_s16(vqmovun_s16(vacc4x01234567), vacc5x01234567);
    uint8x16_t vout6x01234567_7x01234567 = vqmovun_high_s16(vqmovun_s16(vacc6x01234567), vacc7x01234567);
#else
    const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);
    const int16x8_t vacc3x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)), voutput_zero_point);
    const int16x8_t vacc4x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc4x0123), vqmovn_s32(vacc4x4567)), voutput_zero_point);
    const int16x8_t vacc5x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc5x0123), vqmovn_s32(vacc5x4567)), voutput_zero_point);
    const int16x8_t vacc6x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc6x0123), vqmovn_s32(vacc6x4567)), voutput_zero_point);
    const int16x8_t vacc7x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc7x0123), vqmovn_s32(vacc7x4567)), voutput_zero_point);

    uint8x16_t vout0x01234567_1x01234567 = vcombine_u8(vqmovun_s16(vacc0x01234567), vqmovun_s16(vacc1x01234567));
    uint8x16_t vout2x01234567_3x01234567 = vcombine_u8(vqmovun_s16(vacc2x01234567), vqmovun_s16(vacc3x01234567));
    uint8x16_t vout4x01234567_5x01234567 = vcombine_u8(vqmovun_s16(vacc4x01234567), vqmovun_s16(vacc5x01234567));
    uint8x16_t vout6x01234567_7x01234567 = vcombine_u8(vqmovun_s16(vacc6x01234567), vqmovun_s16(vacc7x01234567));
#endif
    const uint8x16_t voutput_min = vld1q_dup_u8(&params->rndnu_neon.output_min);
    const uint8x16_t voutput_max = vld1q_dup_u8(&params->rndnu_neon.output_max);

    vout0x01234567_1x01234567 = vmaxq_u8(vout0x01234567_1x01234567, voutput_min);
    vout2x01234567_3x01234567 = vmaxq_u8(vout2x01234567_3x01234567, voutput_min);
    vout4x01234567_5x01234567 = vmaxq_u8(vout4x01234567_5x01234567, voutput_min);
    vout6x01234567_7x01234567 = vmaxq_u8(vout6x01234567_7x01234567, voutput_min);

    vout0x01234567_1x01234567 = vminq_u8(vout0x01234567_1x01234567, voutput_max);
    vout2x01234567_3x01234567 = vminq_u8(vout2x01234567_3x01234567, voutput_max);
    vout4x01234567_5x01234567 = vminq_u8(vout4x01234567_5x01234567, voutput_max);
    vout6x01234567_7x01234567 = vminq_u8(vout6x01234567_7x01234567, voutput_max);

    if (nc >= 8) {
      vst1_u8(c7 + 0, vget_high_u8(vout6x01234567_7x01234567));
      vst1_u8(c6 + 0, vget_low_u8(vout6x01234567_7x01234567));
      vst1_u8(c5 + 0, vget_high_u8(vout4x01234567_5x01234567));
      vst1_u8(c4 + 0, vget_low_u8(vout4x01234567_5x01234567));
      vst1_u8(c3 + 0, vget_high_u8(vout2x01234567_3x01234567));
      vst1_u8(c2 + 0, vget_low_u8(vout2x01234567_3x01234567));
      vst1_u8(c1 + 0, vget_high_u8(vout0x01234567_1x01234567));
      vst1_u8(c0 + 0, vget_low_u8(vout0x01234567_1x01234567));

      c7 = (uint8_t*) ((uintptr_t) c7 + cn_stride);
      c6 = (uint8_t*) ((uintptr_t) c6 + cn_stride);
      c5 = (uint8_t*) ((uintptr_t) c5 + cn_stride);
      c4 = (uint8_t*) ((uintptr_t) c4 + cn_stride);
      c3 = (uint8_t*) ((uintptr_t) c3 + cn_stride);
      c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);

      nc -= 8;
    } else {
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
