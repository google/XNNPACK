// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRxNRc4-neondot.c.in
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


void xnn_qs8_gemm_minmax_ukernel_8x16c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_gemm_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN XNN_DISABLE_MSAN
{
  assert(mr != 0);
  assert(mr <= 8);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4);
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
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const int8_t* a6 = (const int8_t*) ((uintptr_t) a5 + a_stride);
  int8_t* c6 = (int8_t*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }
  const int8_t* a7 = (const int8_t*) ((uintptr_t) a6 + a_stride);
  int8_t* c7 = (int8_t*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 8) {
    a7 = a6;
    c7 = c6;
  }

  // Loop over groups of 16 columns.
  do {
    // Initialize accumulators with bias. 16 bias values are loaded from the
    // weight matrix, at the start of the group of 16 columns.
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0x89AB = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0xCDEF = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;
    int32x4_t vacc1x89AB = vacc0x89AB;
    int32x4_t vacc1xCDEF = vacc0xCDEF;
    int32x4_t vacc2x0123 = vacc0x0123;
    int32x4_t vacc2x4567 = vacc0x4567;
    int32x4_t vacc2x89AB = vacc0x89AB;
    int32x4_t vacc2xCDEF = vacc0xCDEF;
    int32x4_t vacc3x0123 = vacc0x0123;
    int32x4_t vacc3x4567 = vacc0x4567;
    int32x4_t vacc3x89AB = vacc0x89AB;
    int32x4_t vacc3xCDEF = vacc0xCDEF;
    int32x4_t vacc4x0123 = vacc0x0123;
    int32x4_t vacc4x4567 = vacc0x4567;
    int32x4_t vacc4x89AB = vacc0x89AB;
    int32x4_t vacc4xCDEF = vacc0xCDEF;
    int32x4_t vacc5x0123 = vacc0x0123;
    int32x4_t vacc5x4567 = vacc0x4567;
    int32x4_t vacc5x89AB = vacc0x89AB;
    int32x4_t vacc5xCDEF = vacc0xCDEF;
    int32x4_t vacc6x0123 = vacc0x0123;
    int32x4_t vacc6x4567 = vacc0x4567;
    int32x4_t vacc6x89AB = vacc0x89AB;
    int32x4_t vacc6xCDEF = vacc0xCDEF;
    int32x4_t vacc7x0123 = vacc0x0123;
    int32x4_t vacc7x4567 = vacc0x4567;
    int32x4_t vacc7x89AB = vacc0x89AB;
    int32x4_t vacc7xCDEF = vacc0xCDEF;

    // Inner accumulation loop along the 16 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 8 * sizeof(int8_t)) {
      // Load a 8x8 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;
      const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 8;
      const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 8;
      const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 8;
      const int8x8_t va4x01234567 = vld1_s8(a4); a4 += 8;
      const int8x8_t va5x01234567 = vld1_s8(a5); a5 += 8;
      const int8x8_t va6x01234567 = vld1_s8(a6); a6 += 8;
      const int8x8_t va7x01234567 = vld1_s8(a7); a7 += 8;

      // Load a 8x16 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const void*) ((const int8_t*) w + 16);
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const void*) ((const int8_t*) w + 16);
      const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const void*) ((const int8_t*) w + 16);
      const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const void*) ((const int8_t*) w + 16);
      const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const void*) ((const int8_t*) w + 16);
      const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const void*) ((const int8_t*) w + 16);
      const int8x16_t vb4567x89AB = vld1q_s8(w); w = (const void*) ((const int8_t*) w + 16);
      const int8x16_t vb4567xCDEF = vld1q_s8(w); w = (const void*) ((const int8_t*) w + 16);

      // Multiply-accumulate: 8x8 * 8x16 --> 8x16.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
      vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb0123x89AB, va1x01234567, 0);
      vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb0123xCDEF, va1x01234567, 0);
      vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
      vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
      vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb0123x89AB, va2x01234567, 0);
      vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb0123xCDEF, va2x01234567, 0);
      vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
      vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
      vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb0123x89AB, va3x01234567, 0);
      vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb0123xCDEF, va3x01234567, 0);
      vacc4x0123 = vdotq_lane_s32(vacc4x0123, vb0123x0123, va4x01234567, 0);
      vacc4x4567 = vdotq_lane_s32(vacc4x4567, vb0123x4567, va4x01234567, 0);
      vacc4x89AB = vdotq_lane_s32(vacc4x89AB, vb0123x89AB, va4x01234567, 0);
      vacc4xCDEF = vdotq_lane_s32(vacc4xCDEF, vb0123xCDEF, va4x01234567, 0);
      vacc5x0123 = vdotq_lane_s32(vacc5x0123, vb0123x0123, va5x01234567, 0);
      vacc5x4567 = vdotq_lane_s32(vacc5x4567, vb0123x4567, va5x01234567, 0);
      vacc5x89AB = vdotq_lane_s32(vacc5x89AB, vb0123x89AB, va5x01234567, 0);
      vacc5xCDEF = vdotq_lane_s32(vacc5xCDEF, vb0123xCDEF, va5x01234567, 0);
      vacc6x0123 = vdotq_lane_s32(vacc6x0123, vb0123x0123, va6x01234567, 0);
      vacc6x4567 = vdotq_lane_s32(vacc6x4567, vb0123x4567, va6x01234567, 0);
      vacc6x89AB = vdotq_lane_s32(vacc6x89AB, vb0123x89AB, va6x01234567, 0);
      vacc6xCDEF = vdotq_lane_s32(vacc6xCDEF, vb0123xCDEF, va6x01234567, 0);
      vacc7x0123 = vdotq_lane_s32(vacc7x0123, vb0123x0123, va7x01234567, 0);
      vacc7x4567 = vdotq_lane_s32(vacc7x4567, vb0123x4567, va7x01234567, 0);
      vacc7x89AB = vdotq_lane_s32(vacc7x89AB, vb0123x89AB, va7x01234567, 0);
      vacc7xCDEF = vdotq_lane_s32(vacc7xCDEF, vb0123xCDEF, va7x01234567, 0);
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb4567x89AB, va0x01234567, 1);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb4567xCDEF, va0x01234567, 1);
      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb4567x0123, va1x01234567, 1);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb4567x4567, va1x01234567, 1);
      vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb4567x89AB, va1x01234567, 1);
      vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb4567xCDEF, va1x01234567, 1);
      vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb4567x0123, va2x01234567, 1);
      vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb4567x4567, va2x01234567, 1);
      vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb4567x89AB, va2x01234567, 1);
      vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb4567xCDEF, va2x01234567, 1);
      vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb4567x0123, va3x01234567, 1);
      vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb4567x4567, va3x01234567, 1);
      vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb4567x89AB, va3x01234567, 1);
      vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb4567xCDEF, va3x01234567, 1);
      vacc4x0123 = vdotq_lane_s32(vacc4x0123, vb4567x0123, va4x01234567, 1);
      vacc4x4567 = vdotq_lane_s32(vacc4x4567, vb4567x4567, va4x01234567, 1);
      vacc4x89AB = vdotq_lane_s32(vacc4x89AB, vb4567x89AB, va4x01234567, 1);
      vacc4xCDEF = vdotq_lane_s32(vacc4xCDEF, vb4567xCDEF, va4x01234567, 1);
      vacc5x0123 = vdotq_lane_s32(vacc5x0123, vb4567x0123, va5x01234567, 1);
      vacc5x4567 = vdotq_lane_s32(vacc5x4567, vb4567x4567, va5x01234567, 1);
      vacc5x89AB = vdotq_lane_s32(vacc5x89AB, vb4567x89AB, va5x01234567, 1);
      vacc5xCDEF = vdotq_lane_s32(vacc5xCDEF, vb4567xCDEF, va5x01234567, 1);
      vacc6x0123 = vdotq_lane_s32(vacc6x0123, vb4567x0123, va6x01234567, 1);
      vacc6x4567 = vdotq_lane_s32(vacc6x4567, vb4567x4567, va6x01234567, 1);
      vacc6x89AB = vdotq_lane_s32(vacc6x89AB, vb4567x89AB, va6x01234567, 1);
      vacc6xCDEF = vdotq_lane_s32(vacc6xCDEF, vb4567xCDEF, va6x01234567, 1);
      vacc7x0123 = vdotq_lane_s32(vacc7x0123, vb4567x0123, va7x01234567, 1);
      vacc7x4567 = vdotq_lane_s32(vacc7x4567, vb4567x4567, va7x01234567, 1);
      vacc7x89AB = vdotq_lane_s32(vacc7x89AB, vb4567x89AB, va7x01234567, 1);
      vacc7xCDEF = vdotq_lane_s32(vacc7xCDEF, vb4567xCDEF, va7x01234567, 1);

      k -= 8 * sizeof(int8_t);
    }
    // Handle up to 4 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 8x4 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 4;
      const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 4;
      const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 4;
      const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 4;
      const int8x8_t va4x01234567 = vld1_s8(a4); a4 += 4;
      const int8x8_t va5x01234567 = vld1_s8(a5); a5 += 4;
      const int8x8_t va6x01234567 = vld1_s8(a6); a6 += 4;
      const int8x8_t va7x01234567 = vld1_s8(a7); a7 += 4;

      // Load a 4x16 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const void*) ((const int8_t*) w + 16);
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const void*) ((const int8_t*) w + 16);
      const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const void*) ((const int8_t*) w + 16);
      const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const void*) ((const int8_t*) w + 16);

      // Multiply-accumulate: 8x4 * 4x16 --> 8x16.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
      vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb0123x89AB, va1x01234567, 0);
      vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb0123xCDEF, va1x01234567, 0);
      vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
      vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
      vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb0123x89AB, va2x01234567, 0);
      vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb0123xCDEF, va2x01234567, 0);
      vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
      vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
      vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb0123x89AB, va3x01234567, 0);
      vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb0123xCDEF, va3x01234567, 0);
      vacc4x0123 = vdotq_lane_s32(vacc4x0123, vb0123x0123, va4x01234567, 0);
      vacc4x4567 = vdotq_lane_s32(vacc4x4567, vb0123x4567, va4x01234567, 0);
      vacc4x89AB = vdotq_lane_s32(vacc4x89AB, vb0123x89AB, va4x01234567, 0);
      vacc4xCDEF = vdotq_lane_s32(vacc4xCDEF, vb0123xCDEF, va4x01234567, 0);
      vacc5x0123 = vdotq_lane_s32(vacc5x0123, vb0123x0123, va5x01234567, 0);
      vacc5x4567 = vdotq_lane_s32(vacc5x4567, vb0123x4567, va5x01234567, 0);
      vacc5x89AB = vdotq_lane_s32(vacc5x89AB, vb0123x89AB, va5x01234567, 0);
      vacc5xCDEF = vdotq_lane_s32(vacc5xCDEF, vb0123xCDEF, va5x01234567, 0);
      vacc6x0123 = vdotq_lane_s32(vacc6x0123, vb0123x0123, va6x01234567, 0);
      vacc6x4567 = vdotq_lane_s32(vacc6x4567, vb0123x4567, va6x01234567, 0);
      vacc6x89AB = vdotq_lane_s32(vacc6x89AB, vb0123x89AB, va6x01234567, 0);
      vacc6xCDEF = vdotq_lane_s32(vacc6xCDEF, vb0123xCDEF, va6x01234567, 0);
      vacc7x0123 = vdotq_lane_s32(vacc7x0123, vb0123x0123, va7x01234567, 0);
      vacc7x4567 = vdotq_lane_s32(vacc7x4567, vb0123x4567, va7x01234567, 0);
      vacc7x89AB = vdotq_lane_s32(vacc7x89AB, vb0123x89AB, va7x01234567, 0);
      vacc7xCDEF = vdotq_lane_s32(vacc7xCDEF, vb0123xCDEF, va7x01234567, 0);
    }

    // Post-accumulation work
    const int32x4_t vright_shift = vld1q_dup_s32(&params->neon.right_shift);
    const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));

    const int32x4_t vproduct0x0123 = vqrdmulhq_n_s32(vacc0x0123, params->neon.multiplier);
    const int32x4_t vproduct0x4567 = vqrdmulhq_n_s32(vacc0x4567, params->neon.multiplier);
    const int32x4_t vproduct0x89AB = vqrdmulhq_n_s32(vacc0x89AB, params->neon.multiplier);
    const int32x4_t vproduct0xCDEF = vqrdmulhq_n_s32(vacc0xCDEF, params->neon.multiplier);
    const int32x4_t vproduct1x0123 = vqrdmulhq_n_s32(vacc1x0123, params->neon.multiplier);
    const int32x4_t vproduct1x4567 = vqrdmulhq_n_s32(vacc1x4567, params->neon.multiplier);
    const int32x4_t vproduct1x89AB = vqrdmulhq_n_s32(vacc1x89AB, params->neon.multiplier);
    const int32x4_t vproduct1xCDEF = vqrdmulhq_n_s32(vacc1xCDEF, params->neon.multiplier);
    const int32x4_t vproduct2x0123 = vqrdmulhq_n_s32(vacc2x0123, params->neon.multiplier);
    const int32x4_t vproduct2x4567 = vqrdmulhq_n_s32(vacc2x4567, params->neon.multiplier);
    const int32x4_t vproduct2x89AB = vqrdmulhq_n_s32(vacc2x89AB, params->neon.multiplier);
    const int32x4_t vproduct2xCDEF = vqrdmulhq_n_s32(vacc2xCDEF, params->neon.multiplier);
    const int32x4_t vproduct3x0123 = vqrdmulhq_n_s32(vacc3x0123, params->neon.multiplier);
    const int32x4_t vproduct3x4567 = vqrdmulhq_n_s32(vacc3x4567, params->neon.multiplier);
    const int32x4_t vproduct3x89AB = vqrdmulhq_n_s32(vacc3x89AB, params->neon.multiplier);
    const int32x4_t vproduct3xCDEF = vqrdmulhq_n_s32(vacc3xCDEF, params->neon.multiplier);
    const int32x4_t vproduct4x0123 = vqrdmulhq_n_s32(vacc4x0123, params->neon.multiplier);
    const int32x4_t vproduct4x4567 = vqrdmulhq_n_s32(vacc4x4567, params->neon.multiplier);
    const int32x4_t vproduct4x89AB = vqrdmulhq_n_s32(vacc4x89AB, params->neon.multiplier);
    const int32x4_t vproduct4xCDEF = vqrdmulhq_n_s32(vacc4xCDEF, params->neon.multiplier);
    const int32x4_t vproduct5x0123 = vqrdmulhq_n_s32(vacc5x0123, params->neon.multiplier);
    const int32x4_t vproduct5x4567 = vqrdmulhq_n_s32(vacc5x4567, params->neon.multiplier);
    const int32x4_t vproduct5x89AB = vqrdmulhq_n_s32(vacc5x89AB, params->neon.multiplier);
    const int32x4_t vproduct5xCDEF = vqrdmulhq_n_s32(vacc5xCDEF, params->neon.multiplier);
    const int32x4_t vproduct6x0123 = vqrdmulhq_n_s32(vacc6x0123, params->neon.multiplier);
    const int32x4_t vproduct6x4567 = vqrdmulhq_n_s32(vacc6x4567, params->neon.multiplier);
    const int32x4_t vproduct6x89AB = vqrdmulhq_n_s32(vacc6x89AB, params->neon.multiplier);
    const int32x4_t vproduct6xCDEF = vqrdmulhq_n_s32(vacc6xCDEF, params->neon.multiplier);
    const int32x4_t vproduct7x0123 = vqrdmulhq_n_s32(vacc7x0123, params->neon.multiplier);
    const int32x4_t vproduct7x4567 = vqrdmulhq_n_s32(vacc7x4567, params->neon.multiplier);
    const int32x4_t vproduct7x89AB = vqrdmulhq_n_s32(vacc7x89AB, params->neon.multiplier);
    const int32x4_t vproduct7xCDEF = vqrdmulhq_n_s32(vacc7xCDEF, params->neon.multiplier);

    vacc0x0123 = vcltq_s32(vproduct0x0123, vbicq_s32(vacc0x0123, vzero_shift_mask), vmovq_n_s32(0));
    vacc0x4567 = vcltq_s32(vproduct0x4567, vbicq_s32(vacc0x4567, vzero_shift_mask), vmovq_n_s32(0));
    vacc0x89AB = vcltq_s32(vproduct0x89AB, vbicq_s32(vacc0x89AB, vzero_shift_mask), vmovq_n_s32(0));
    vacc0xCDEF = vcltq_s32(vproduct0xCDEF, vbicq_s32(vacc0xCDEF, vzero_shift_mask), vmovq_n_s32(0));
    vacc1x0123 = vcltq_s32(vproduct1x0123, vbicq_s32(vacc1x0123, vzero_shift_mask), vmovq_n_s32(0));
    vacc1x4567 = vcltq_s32(vproduct1x4567, vbicq_s32(vacc1x4567, vzero_shift_mask), vmovq_n_s32(0));
    vacc1x89AB = vcltq_s32(vproduct1x89AB, vbicq_s32(vacc1x89AB, vzero_shift_mask), vmovq_n_s32(0));
    vacc1xCDEF = vcltq_s32(vproduct1xCDEF, vbicq_s32(vacc1xCDEF, vzero_shift_mask), vmovq_n_s32(0));
    vacc2x0123 = vcltq_s32(vproduct2x0123, vbicq_s32(vacc2x0123, vzero_shift_mask), vmovq_n_s32(0));
    vacc2x4567 = vcltq_s32(vproduct2x4567, vbicq_s32(vacc2x4567, vzero_shift_mask), vmovq_n_s32(0));
    vacc2x89AB = vcltq_s32(vproduct2x89AB, vbicq_s32(vacc2x89AB, vzero_shift_mask), vmovq_n_s32(0));
    vacc2xCDEF = vcltq_s32(vproduct2xCDEF, vbicq_s32(vacc2xCDEF, vzero_shift_mask), vmovq_n_s32(0));
    vacc3x0123 = vcltq_s32(vproduct3x0123, vbicq_s32(vacc3x0123, vzero_shift_mask), vmovq_n_s32(0));
    vacc3x4567 = vcltq_s32(vproduct3x4567, vbicq_s32(vacc3x4567, vzero_shift_mask), vmovq_n_s32(0));
    vacc3x89AB = vcltq_s32(vproduct3x89AB, vbicq_s32(vacc3x89AB, vzero_shift_mask), vmovq_n_s32(0));
    vacc3xCDEF = vcltq_s32(vproduct3xCDEF, vbicq_s32(vacc3xCDEF, vzero_shift_mask), vmovq_n_s32(0));
    vacc4x0123 = vcltq_s32(vproduct4x0123, vbicq_s32(vacc4x0123, vzero_shift_mask), vmovq_n_s32(0));
    vacc4x4567 = vcltq_s32(vproduct4x4567, vbicq_s32(vacc4x4567, vzero_shift_mask), vmovq_n_s32(0));
    vacc4x89AB = vcltq_s32(vproduct4x89AB, vbicq_s32(vacc4x89AB, vzero_shift_mask), vmovq_n_s32(0));
    vacc4xCDEF = vcltq_s32(vproduct4xCDEF, vbicq_s32(vacc4xCDEF, vzero_shift_mask), vmovq_n_s32(0));
    vacc5x0123 = vcltq_s32(vproduct5x0123, vbicq_s32(vacc5x0123, vzero_shift_mask), vmovq_n_s32(0));
    vacc5x4567 = vcltq_s32(vproduct5x4567, vbicq_s32(vacc5x4567, vzero_shift_mask), vmovq_n_s32(0));
    vacc5x89AB = vcltq_s32(vproduct5x89AB, vbicq_s32(vacc5x89AB, vzero_shift_mask), vmovq_n_s32(0));
    vacc5xCDEF = vcltq_s32(vproduct5xCDEF, vbicq_s32(vacc5xCDEF, vzero_shift_mask), vmovq_n_s32(0));
    vacc6x0123 = vcltq_s32(vproduct6x0123, vbicq_s32(vacc6x0123, vzero_shift_mask), vmovq_n_s32(0));
    vacc6x4567 = vcltq_s32(vproduct6x4567, vbicq_s32(vacc6x4567, vzero_shift_mask), vmovq_n_s32(0));
    vacc6x89AB = vcltq_s32(vproduct6x89AB, vbicq_s32(vacc6x89AB, vzero_shift_mask), vmovq_n_s32(0));
    vacc6xCDEF = vcltq_s32(vproduct6xCDEF, vbicq_s32(vacc6xCDEF, vzero_shift_mask), vmovq_n_s32(0));
    vacc7x0123 = vcltq_s32(vproduct7x0123, vbicq_s32(vacc7x0123, vzero_shift_mask), vmovq_n_s32(0));
    vacc7x4567 = vcltq_s32(vproduct7x4567, vbicq_s32(vacc7x4567, vzero_shift_mask), vmovq_n_s32(0));
    vacc7x89AB = vcltq_s32(vproduct7x89AB, vbicq_s32(vacc7x89AB, vzero_shift_mask), vmovq_n_s32(0));
    vacc7xCDEF = vcltq_s32(vproduct7xCDEF, vbicq_s32(vacc7xCDEF, vzero_shift_mask), vmovq_n_s32(0));

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_shift);
    vacc0x89AB = vrshlq_s32(vacc0x89AB, vright_shift);
    vacc0xCDEF = vrshlq_s32(vacc0xCDEF, vright_shift);
    vacc1x0123 = vrshlq_s32(vacc1x0123, vright_shift);
    vacc1x4567 = vrshlq_s32(vacc1x4567, vright_shift);
    vacc1x89AB = vrshlq_s32(vacc1x89AB, vright_shift);
    vacc1xCDEF = vrshlq_s32(vacc1xCDEF, vright_shift);
    vacc2x0123 = vrshlq_s32(vacc2x0123, vright_shift);
    vacc2x4567 = vrshlq_s32(vacc2x4567, vright_shift);
    vacc2x89AB = vrshlq_s32(vacc2x89AB, vright_shift);
    vacc2xCDEF = vrshlq_s32(vacc2xCDEF, vright_shift);
    vacc3x0123 = vrshlq_s32(vacc3x0123, vright_shift);
    vacc3x4567 = vrshlq_s32(vacc3x4567, vright_shift);
    vacc3x89AB = vrshlq_s32(vacc3x89AB, vright_shift);
    vacc3xCDEF = vrshlq_s32(vacc3xCDEF, vright_shift);
    vacc4x0123 = vrshlq_s32(vacc4x0123, vright_shift);
    vacc4x4567 = vrshlq_s32(vacc4x4567, vright_shift);
    vacc4x89AB = vrshlq_s32(vacc4x89AB, vright_shift);
    vacc4xCDEF = vrshlq_s32(vacc4xCDEF, vright_shift);
    vacc5x0123 = vrshlq_s32(vacc5x0123, vright_shift);
    vacc5x4567 = vrshlq_s32(vacc5x4567, vright_shift);
    vacc5x89AB = vrshlq_s32(vacc5x89AB, vright_shift);
    vacc5xCDEF = vrshlq_s32(vacc5xCDEF, vright_shift);
    vacc6x0123 = vrshlq_s32(vacc6x0123, vright_shift);
    vacc6x4567 = vrshlq_s32(vacc6x4567, vright_shift);
    vacc6x89AB = vrshlq_s32(vacc6x89AB, vright_shift);
    vacc6xCDEF = vrshlq_s32(vacc6xCDEF, vright_shift);
    vacc7x0123 = vrshlq_s32(vacc7x0123, vright_shift);
    vacc7x4567 = vrshlq_s32(vacc7x4567, vright_shift);
    vacc7x89AB = vrshlq_s32(vacc7x89AB, vright_shift);
    vacc7xCDEF = vrshlq_s32(vacc7xCDEF, vright_shift);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->neon.output_zero_point);
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

    int8x16_t vout0x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc0x89ABCDEF);
    int8x16_t vout1x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc1x01234567), vacc1x89ABCDEF);
    int8x16_t vout2x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc2x01234567), vacc2x89ABCDEF);
    int8x16_t vout3x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc3x01234567), vacc3x89ABCDEF);
    int8x16_t vout4x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc4x01234567), vacc4x89ABCDEF);
    int8x16_t vout5x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc5x01234567), vacc5x89ABCDEF);
    int8x16_t vout6x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc6x01234567), vacc6x89ABCDEF);
    int8x16_t vout7x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc7x01234567), vacc7x89ABCDEF);
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

    int8x16_t vout0x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc0x89ABCDEF));
    int8x16_t vout1x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc1x01234567), vqmovn_s16(vacc1x89ABCDEF));
    int8x16_t vout2x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc2x01234567), vqmovn_s16(vacc2x89ABCDEF));
    int8x16_t vout3x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc3x01234567), vqmovn_s16(vacc3x89ABCDEF));
    int8x16_t vout4x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc4x01234567), vqmovn_s16(vacc4x89ABCDEF));
    int8x16_t vout5x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc5x01234567), vqmovn_s16(vacc5x89ABCDEF));
    int8x16_t vout6x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc6x01234567), vqmovn_s16(vacc6x89ABCDEF));
    int8x16_t vout7x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc7x01234567), vqmovn_s16(vacc7x89ABCDEF));
#endif
    const int8x16_t voutput_min = vld1q_dup_s8(&params->neon.output_min);
    const int8x16_t voutput_max = vld1q_dup_s8(&params->neon.output_max);

    vout0x0123456789ABCDEF = vmaxq_s8(vout0x0123456789ABCDEF, voutput_min);
    vout1x0123456789ABCDEF = vmaxq_s8(vout1x0123456789ABCDEF, voutput_min);
    vout2x0123456789ABCDEF = vmaxq_s8(vout2x0123456789ABCDEF, voutput_min);
    vout3x0123456789ABCDEF = vmaxq_s8(vout3x0123456789ABCDEF, voutput_min);
    vout4x0123456789ABCDEF = vmaxq_s8(vout4x0123456789ABCDEF, voutput_min);
    vout5x0123456789ABCDEF = vmaxq_s8(vout5x0123456789ABCDEF, voutput_min);
    vout6x0123456789ABCDEF = vmaxq_s8(vout6x0123456789ABCDEF, voutput_min);
    vout7x0123456789ABCDEF = vmaxq_s8(vout7x0123456789ABCDEF, voutput_min);

    vout0x0123456789ABCDEF = vminq_s8(vout0x0123456789ABCDEF, voutput_max);
    vout1x0123456789ABCDEF = vminq_s8(vout1x0123456789ABCDEF, voutput_max);
    vout2x0123456789ABCDEF = vminq_s8(vout2x0123456789ABCDEF, voutput_max);
    vout3x0123456789ABCDEF = vminq_s8(vout3x0123456789ABCDEF, voutput_max);
    vout4x0123456789ABCDEF = vminq_s8(vout4x0123456789ABCDEF, voutput_max);
    vout5x0123456789ABCDEF = vminq_s8(vout5x0123456789ABCDEF, voutput_max);
    vout6x0123456789ABCDEF = vminq_s8(vout6x0123456789ABCDEF, voutput_max);
    vout7x0123456789ABCDEF = vminq_s8(vout7x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      // Main case where there the 16 columns fit in the destination.
      vst1q_s8(c0 + 0, vout0x0123456789ABCDEF);
      vst1q_s8(c1 + 0, vout1x0123456789ABCDEF);
      vst1q_s8(c2 + 0, vout2x0123456789ABCDEF);
      vst1q_s8(c3 + 0, vout3x0123456789ABCDEF);
      vst1q_s8(c4 + 0, vout4x0123456789ABCDEF);
      vst1q_s8(c5 + 0, vout5x0123456789ABCDEF);
      vst1q_s8(c6 + 0, vout6x0123456789ABCDEF);
      vst1q_s8(c7 + 0, vout7x0123456789ABCDEF);

      // Advance to the next 16 columns.
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      c4 = (int8_t*) ((uintptr_t) c4 + cn_stride);
      c5 = (int8_t*) ((uintptr_t) c5 + cn_stride);
      c6 = (int8_t*) ((uintptr_t) c6 + cn_stride);
      c7 = (int8_t*) ((uintptr_t) c7 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);
      a4 = (const int8_t*) ((uintptr_t) a4 - kc);
      a5 = (const int8_t*) ((uintptr_t) a5 - kc);
      a6 = (const int8_t*) ((uintptr_t) a6 - kc);
      a7 = (const int8_t*) ((uintptr_t) a7 - kc);

      nc -= 16;
    } else {
      // Final case where not all of the 16 columns fit in the destination.
      int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vget_low_s8(vout0x0123456789ABCDEF), vget_low_s8(vout1x0123456789ABCDEF));
      int8x16_t vout2x01234567_3x01234567 = vcombine_s8(vget_low_s8(vout2x0123456789ABCDEF), vget_low_s8(vout3x0123456789ABCDEF));
      int8x16_t vout4x01234567_5x01234567 = vcombine_s8(vget_low_s8(vout4x0123456789ABCDEF), vget_low_s8(vout5x0123456789ABCDEF));
      int8x16_t vout6x01234567_7x01234567 = vcombine_s8(vget_low_s8(vout6x0123456789ABCDEF), vget_low_s8(vout7x0123456789ABCDEF));
      if (nc & 8) {
        vst1_s8(c0, vget_low_s8(vout0x01234567_1x01234567)); c0 += 8;
        vst1_s8(c1, vget_high_s8(vout0x01234567_1x01234567)); c1 += 8;
        vst1_s8(c2, vget_low_s8(vout2x01234567_3x01234567)); c2 += 8;
        vst1_s8(c3, vget_high_s8(vout2x01234567_3x01234567)); c3 += 8;
        vst1_s8(c4, vget_low_s8(vout4x01234567_5x01234567)); c4 += 8;
        vst1_s8(c5, vget_high_s8(vout4x01234567_5x01234567)); c5 += 8;
        vst1_s8(c6, vget_low_s8(vout6x01234567_7x01234567)); c6 += 8;
        vst1_s8(c7, vget_high_s8(vout6x01234567_7x01234567)); c7 += 8;
        vout0x01234567_1x01234567 = vcombine_s8(vget_high_s8(vout0x0123456789ABCDEF), vget_high_s8(vout1x0123456789ABCDEF));
        vout2x01234567_3x01234567 = vcombine_s8(vget_high_s8(vout2x0123456789ABCDEF), vget_high_s8(vout3x0123456789ABCDEF));
        vout4x01234567_5x01234567 = vcombine_s8(vget_high_s8(vout4x0123456789ABCDEF), vget_high_s8(vout5x0123456789ABCDEF));
        vout6x01234567_7x01234567 = vcombine_s8(vget_high_s8(vout6x0123456789ABCDEF), vget_high_s8(vout7x0123456789ABCDEF));
      }
      if (nc & 4) {
        vst1q_lane_u32(__builtin_assume_aligned(c0, 1), vreinterpretq_u32_s8(vout0x01234567_1x01234567), 0); c0 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c1, 1), vreinterpretq_u32_s8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c2, 1), vreinterpretq_u32_s8(vout2x01234567_3x01234567), 0); c2 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c3, 1), vreinterpretq_u32_s8(vout2x01234567_3x01234567), 2); c3 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c4, 1), vreinterpretq_u32_s8(vout4x01234567_5x01234567), 0); c4 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c5, 1), vreinterpretq_u32_s8(vout4x01234567_5x01234567), 2); c5 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c6, 1), vreinterpretq_u32_s8(vout6x01234567_7x01234567), 0); c6 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c7, 1), vreinterpretq_u32_s8(vout6x01234567_7x01234567), 2); c7 += 4;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
        vout4x01234567_5x01234567 = vextq_s8(vout4x01234567_5x01234567, vout4x01234567_5x01234567, 4);
        vout6x01234567_7x01234567 = vextq_s8(vout6x01234567_7x01234567, vout6x01234567_7x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16(__builtin_assume_aligned(c0, 1), vreinterpretq_u16_s8(vout0x01234567_1x01234567), 0); c0 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c1, 1), vreinterpretq_u16_s8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c2, 1), vreinterpretq_u16_s8(vout2x01234567_3x01234567), 0); c2 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c3, 1), vreinterpretq_u16_s8(vout2x01234567_3x01234567), 4); c3 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c4, 1), vreinterpretq_u16_s8(vout4x01234567_5x01234567), 0); c4 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c5, 1), vreinterpretq_u16_s8(vout4x01234567_5x01234567), 4); c5 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c6, 1), vreinterpretq_u16_s8(vout6x01234567_7x01234567), 0); c6 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c7, 1), vreinterpretq_u16_s8(vout6x01234567_7x01234567), 4); c7 += 2;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
        vout4x01234567_5x01234567 = vextq_s8(vout4x01234567_5x01234567, vout4x01234567_5x01234567, 2);
        vout6x01234567_7x01234567 = vextq_s8(vout6x01234567_7x01234567, vout6x01234567_7x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_s8(c0, vout0x01234567_1x01234567, 0);
        vst1q_lane_s8(c1, vout0x01234567_1x01234567, 8);
        vst1q_lane_s8(c2, vout2x01234567_3x01234567, 0);
        vst1q_lane_s8(c3, vout2x01234567_3x01234567, 8);
        vst1q_lane_s8(c4, vout4x01234567_5x01234567, 0);
        vst1q_lane_s8(c5, vout4x01234567_5x01234567, 8);
        vst1q_lane_s8(c6, vout6x01234567_7x01234567, 0);
        vst1q_lane_s8(c7, vout6x01234567_7x01234567, 8);
      }

      nc = 0;
    }
  } while (nc != 0);
}
