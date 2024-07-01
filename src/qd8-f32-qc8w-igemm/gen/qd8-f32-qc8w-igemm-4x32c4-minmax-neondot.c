// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/c4-neondot.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/igemm.h"
#include "xnnpack/math.h"


void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x32c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    const int32x4_t vinput_zero_point = vld1q_dup_s32(&quantization_params->zero_point);
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x0123 = vmulq_s32(vksum0123, vinput_zero_point);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vmulq_s32(vksum4567, vinput_zero_point);
    const int32x4_t vksum89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x89AB = vmulq_s32(vksum89AB, vinput_zero_point);
    const int32x4_t vksumCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xCDEF = vmulq_s32(vksumCDEF, vinput_zero_point);
    const int32x4_t vksumGHIJ = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xGHIJ = vmulq_s32(vksumGHIJ, vinput_zero_point);
    const int32x4_t vksumKLMN = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xKLMN = vmulq_s32(vksumKLMN, vinput_zero_point);
    const int32x4_t vksumOPQR = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xOPQR = vmulq_s32(vksumOPQR, vinput_zero_point);
    const int32x4_t vksumSTUV = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xSTUV = vmulq_s32(vksumSTUV, vinput_zero_point);
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;
    int32x4_t vacc1x89AB = vacc0x89AB;
    int32x4_t vacc1xCDEF = vacc0xCDEF;
    int32x4_t vacc1xGHIJ = vacc0xGHIJ;
    int32x4_t vacc1xKLMN = vacc0xKLMN;
    int32x4_t vacc1xOPQR = vacc0xOPQR;
    int32x4_t vacc1xSTUV = vacc0xSTUV;
    int32x4_t vacc2x0123 = vacc0x0123;
    int32x4_t vacc2x4567 = vacc0x4567;
    int32x4_t vacc2x89AB = vacc0x89AB;
    int32x4_t vacc2xCDEF = vacc0xCDEF;
    int32x4_t vacc2xGHIJ = vacc0xGHIJ;
    int32x4_t vacc2xKLMN = vacc0xKLMN;
    int32x4_t vacc2xOPQR = vacc0xOPQR;
    int32x4_t vacc2xSTUV = vacc0xSTUV;
    int32x4_t vacc3x0123 = vacc0x0123;
    int32x4_t vacc3x4567 = vacc0x4567;
    int32x4_t vacc3x89AB = vacc0x89AB;
    int32x4_t vacc3xCDEF = vacc0xCDEF;
    int32x4_t vacc3xGHIJ = vacc0xGHIJ;
    int32x4_t vacc3xKLMN = vacc0xKLMN;
    int32x4_t vacc3xOPQR = vacc0xOPQR;
    int32x4_t vacc3xSTUV = vacc0xSTUV;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      } else {
        a1 = zero_data;
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      } else {
        a2 = zero_data;
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      } else {
        a3 = zero_data;
      }
      a += 4;

      // Inner accumulation loop along the 32 columns.
      size_t k = kc;
      // 2x partial unrolled loop to load 8 bytes at a time.
      while (k >= 8 * sizeof(int8_t)) {
        // Load a 4x8 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;
        const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 8;
        const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 8;
        const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 8;

        // Load a 8x32 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xGHIJ = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xKLMN = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xOPQR = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xSTUV = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567xGHIJ = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567xKLMN = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567xOPQR = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb4567xSTUV = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 4x8 * 8x32 --> 4x32.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
        vacc0xGHIJ = vdotq_lane_s32(vacc0xGHIJ, vb0123xGHIJ, va0x01234567, 0);
        vacc0xKLMN = vdotq_lane_s32(vacc0xKLMN, vb0123xKLMN, va0x01234567, 0);
        vacc0xOPQR = vdotq_lane_s32(vacc0xOPQR, vb0123xOPQR, va0x01234567, 0);
        vacc0xSTUV = vdotq_lane_s32(vacc0xSTUV, vb0123xSTUV, va0x01234567, 0);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
        vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb0123x89AB, va1x01234567, 0);
        vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb0123xCDEF, va1x01234567, 0);
        vacc1xGHIJ = vdotq_lane_s32(vacc1xGHIJ, vb0123xGHIJ, va1x01234567, 0);
        vacc1xKLMN = vdotq_lane_s32(vacc1xKLMN, vb0123xKLMN, va1x01234567, 0);
        vacc1xOPQR = vdotq_lane_s32(vacc1xOPQR, vb0123xOPQR, va1x01234567, 0);
        vacc1xSTUV = vdotq_lane_s32(vacc1xSTUV, vb0123xSTUV, va1x01234567, 0);
        vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
        vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
        vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb0123x89AB, va2x01234567, 0);
        vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb0123xCDEF, va2x01234567, 0);
        vacc2xGHIJ = vdotq_lane_s32(vacc2xGHIJ, vb0123xGHIJ, va2x01234567, 0);
        vacc2xKLMN = vdotq_lane_s32(vacc2xKLMN, vb0123xKLMN, va2x01234567, 0);
        vacc2xOPQR = vdotq_lane_s32(vacc2xOPQR, vb0123xOPQR, va2x01234567, 0);
        vacc2xSTUV = vdotq_lane_s32(vacc2xSTUV, vb0123xSTUV, va2x01234567, 0);
        vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
        vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
        vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb0123x89AB, va3x01234567, 0);
        vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb0123xCDEF, va3x01234567, 0);
        vacc3xGHIJ = vdotq_lane_s32(vacc3xGHIJ, vb0123xGHIJ, va3x01234567, 0);
        vacc3xKLMN = vdotq_lane_s32(vacc3xKLMN, vb0123xKLMN, va3x01234567, 0);
        vacc3xOPQR = vdotq_lane_s32(vacc3xOPQR, vb0123xOPQR, va3x01234567, 0);
        vacc3xSTUV = vdotq_lane_s32(vacc3xSTUV, vb0123xSTUV, va3x01234567, 0);
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb4567x89AB, va0x01234567, 1);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb4567xCDEF, va0x01234567, 1);
        vacc0xGHIJ = vdotq_lane_s32(vacc0xGHIJ, vb4567xGHIJ, va0x01234567, 1);
        vacc0xKLMN = vdotq_lane_s32(vacc0xKLMN, vb4567xKLMN, va0x01234567, 1);
        vacc0xOPQR = vdotq_lane_s32(vacc0xOPQR, vb4567xOPQR, va0x01234567, 1);
        vacc0xSTUV = vdotq_lane_s32(vacc0xSTUV, vb4567xSTUV, va0x01234567, 1);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb4567x0123, va1x01234567, 1);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb4567x4567, va1x01234567, 1);
        vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb4567x89AB, va1x01234567, 1);
        vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb4567xCDEF, va1x01234567, 1);
        vacc1xGHIJ = vdotq_lane_s32(vacc1xGHIJ, vb4567xGHIJ, va1x01234567, 1);
        vacc1xKLMN = vdotq_lane_s32(vacc1xKLMN, vb4567xKLMN, va1x01234567, 1);
        vacc1xOPQR = vdotq_lane_s32(vacc1xOPQR, vb4567xOPQR, va1x01234567, 1);
        vacc1xSTUV = vdotq_lane_s32(vacc1xSTUV, vb4567xSTUV, va1x01234567, 1);
        vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb4567x0123, va2x01234567, 1);
        vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb4567x4567, va2x01234567, 1);
        vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb4567x89AB, va2x01234567, 1);
        vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb4567xCDEF, va2x01234567, 1);
        vacc2xGHIJ = vdotq_lane_s32(vacc2xGHIJ, vb4567xGHIJ, va2x01234567, 1);
        vacc2xKLMN = vdotq_lane_s32(vacc2xKLMN, vb4567xKLMN, va2x01234567, 1);
        vacc2xOPQR = vdotq_lane_s32(vacc2xOPQR, vb4567xOPQR, va2x01234567, 1);
        vacc2xSTUV = vdotq_lane_s32(vacc2xSTUV, vb4567xSTUV, va2x01234567, 1);
        vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb4567x0123, va3x01234567, 1);
        vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb4567x4567, va3x01234567, 1);
        vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb4567x89AB, va3x01234567, 1);
        vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb4567xCDEF, va3x01234567, 1);
        vacc3xGHIJ = vdotq_lane_s32(vacc3xGHIJ, vb4567xGHIJ, va3x01234567, 1);
        vacc3xKLMN = vdotq_lane_s32(vacc3xKLMN, vb4567xKLMN, va3x01234567, 1);
        vacc3xOPQR = vdotq_lane_s32(vacc3xOPQR, vb4567xOPQR, va3x01234567, 1);
        vacc3xSTUV = vdotq_lane_s32(vacc3xSTUV, vb4567xSTUV, va3x01234567, 1);

        k -= 8 * sizeof(int8_t);
      }
      // Handle up to 4 final positions of `k`
      if XNN_UNLIKELY(k != 0) {
        // Load a 4x4 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0);
        const int8x8_t va1x01234567 = vld1_s8(a1);
        const int8x8_t va2x01234567 = vld1_s8(a2);
        const int8x8_t va3x01234567 = vld1_s8(a3);

        // Load a 4x32 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xGHIJ = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xKLMN = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xOPQR = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xSTUV = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 4x4 * 4x32 --> 4x32.
        vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
        vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
        vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
        vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
        vacc0xGHIJ = vdotq_lane_s32(vacc0xGHIJ, vb0123xGHIJ, va0x01234567, 0);
        vacc0xKLMN = vdotq_lane_s32(vacc0xKLMN, vb0123xKLMN, va0x01234567, 0);
        vacc0xOPQR = vdotq_lane_s32(vacc0xOPQR, vb0123xOPQR, va0x01234567, 0);
        vacc0xSTUV = vdotq_lane_s32(vacc0xSTUV, vb0123xSTUV, va0x01234567, 0);
        vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
        vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
        vacc1x89AB = vdotq_lane_s32(vacc1x89AB, vb0123x89AB, va1x01234567, 0);
        vacc1xCDEF = vdotq_lane_s32(vacc1xCDEF, vb0123xCDEF, va1x01234567, 0);
        vacc1xGHIJ = vdotq_lane_s32(vacc1xGHIJ, vb0123xGHIJ, va1x01234567, 0);
        vacc1xKLMN = vdotq_lane_s32(vacc1xKLMN, vb0123xKLMN, va1x01234567, 0);
        vacc1xOPQR = vdotq_lane_s32(vacc1xOPQR, vb0123xOPQR, va1x01234567, 0);
        vacc1xSTUV = vdotq_lane_s32(vacc1xSTUV, vb0123xSTUV, va1x01234567, 0);
        vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
        vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
        vacc2x89AB = vdotq_lane_s32(vacc2x89AB, vb0123x89AB, va2x01234567, 0);
        vacc2xCDEF = vdotq_lane_s32(vacc2xCDEF, vb0123xCDEF, va2x01234567, 0);
        vacc2xGHIJ = vdotq_lane_s32(vacc2xGHIJ, vb0123xGHIJ, va2x01234567, 0);
        vacc2xKLMN = vdotq_lane_s32(vacc2xKLMN, vb0123xKLMN, va2x01234567, 0);
        vacc2xOPQR = vdotq_lane_s32(vacc2xOPQR, vb0123xOPQR, va2x01234567, 0);
        vacc2xSTUV = vdotq_lane_s32(vacc2xSTUV, vb0123xSTUV, va2x01234567, 0);
        vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
        vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
        vacc3x89AB = vdotq_lane_s32(vacc3x89AB, vb0123x89AB, va3x01234567, 0);
        vacc3xCDEF = vdotq_lane_s32(vacc3xCDEF, vb0123xCDEF, va3x01234567, 0);
        vacc3xGHIJ = vdotq_lane_s32(vacc3xGHIJ, vb0123xGHIJ, va3x01234567, 0);
        vacc3xKLMN = vdotq_lane_s32(vacc3xKLMN, vb0123xKLMN, va3x01234567, 0);
        vacc3xOPQR = vdotq_lane_s32(vacc3xOPQR, vb0123xOPQR, va3x01234567, 0);
        vacc3xSTUV = vdotq_lane_s32(vacc3xSTUV, vb0123xSTUV, va3x01234567, 0);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vout0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vout0xCDEF = vcvtq_f32_s32(vacc0xCDEF);
    float32x4_t vout0xGHIJ = vcvtq_f32_s32(vacc0xGHIJ);
    float32x4_t vout0xKLMN = vcvtq_f32_s32(vacc0xKLMN);
    float32x4_t vout0xOPQR = vcvtq_f32_s32(vacc0xOPQR);
    float32x4_t vout0xSTUV = vcvtq_f32_s32(vacc0xSTUV);
    float32x4_t vout1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vout1x4567 = vcvtq_f32_s32(vacc1x4567);
    float32x4_t vout1x89AB = vcvtq_f32_s32(vacc1x89AB);
    float32x4_t vout1xCDEF = vcvtq_f32_s32(vacc1xCDEF);
    float32x4_t vout1xGHIJ = vcvtq_f32_s32(vacc1xGHIJ);
    float32x4_t vout1xKLMN = vcvtq_f32_s32(vacc1xKLMN);
    float32x4_t vout1xOPQR = vcvtq_f32_s32(vacc1xOPQR);
    float32x4_t vout1xSTUV = vcvtq_f32_s32(vacc1xSTUV);
    float32x4_t vout2x0123 = vcvtq_f32_s32(vacc2x0123);
    float32x4_t vout2x4567 = vcvtq_f32_s32(vacc2x4567);
    float32x4_t vout2x89AB = vcvtq_f32_s32(vacc2x89AB);
    float32x4_t vout2xCDEF = vcvtq_f32_s32(vacc2xCDEF);
    float32x4_t vout2xGHIJ = vcvtq_f32_s32(vacc2xGHIJ);
    float32x4_t vout2xKLMN = vcvtq_f32_s32(vacc2xKLMN);
    float32x4_t vout2xOPQR = vcvtq_f32_s32(vacc2xOPQR);
    float32x4_t vout2xSTUV = vcvtq_f32_s32(vacc2xSTUV);
    float32x4_t vout3x0123 = vcvtq_f32_s32(vacc3x0123);
    float32x4_t vout3x4567 = vcvtq_f32_s32(vacc3x4567);
    float32x4_t vout3x89AB = vcvtq_f32_s32(vacc3x89AB);
    float32x4_t vout3xCDEF = vcvtq_f32_s32(vacc3xCDEF);
    float32x4_t vout3xGHIJ = vcvtq_f32_s32(vacc3xGHIJ);
    float32x4_t vout3xKLMN = vcvtq_f32_s32(vacc3xKLMN);
    float32x4_t vout3xOPQR = vcvtq_f32_s32(vacc3xOPQR);
    float32x4_t vout3xSTUV = vcvtq_f32_s32(vacc3xSTUV);
    const float32x4_t vinput_scale = vld1q_dup_f32(&quantization_params->inv_scale);
    vout0x0123 = vmulq_f32(vout0x0123, vinput_scale);
    vout0x4567 = vmulq_f32(vout0x4567, vinput_scale);
    vout0x89AB = vmulq_f32(vout0x89AB, vinput_scale);
    vout0xCDEF = vmulq_f32(vout0xCDEF, vinput_scale);
    vout0xGHIJ = vmulq_f32(vout0xGHIJ, vinput_scale);
    vout0xKLMN = vmulq_f32(vout0xKLMN, vinput_scale);
    vout0xOPQR = vmulq_f32(vout0xOPQR, vinput_scale);
    vout0xSTUV = vmulq_f32(vout0xSTUV, vinput_scale);
    vout1x0123 = vmulq_f32(vout1x0123, vinput_scale);
    vout1x4567 = vmulq_f32(vout1x4567, vinput_scale);
    vout1x89AB = vmulq_f32(vout1x89AB, vinput_scale);
    vout1xCDEF = vmulq_f32(vout1xCDEF, vinput_scale);
    vout1xGHIJ = vmulq_f32(vout1xGHIJ, vinput_scale);
    vout1xKLMN = vmulq_f32(vout1xKLMN, vinput_scale);
    vout1xOPQR = vmulq_f32(vout1xOPQR, vinput_scale);
    vout1xSTUV = vmulq_f32(vout1xSTUV, vinput_scale);
    vout2x0123 = vmulq_f32(vout2x0123, vinput_scale);
    vout2x4567 = vmulq_f32(vout2x4567, vinput_scale);
    vout2x89AB = vmulq_f32(vout2x89AB, vinput_scale);
    vout2xCDEF = vmulq_f32(vout2xCDEF, vinput_scale);
    vout2xGHIJ = vmulq_f32(vout2xGHIJ, vinput_scale);
    vout2xKLMN = vmulq_f32(vout2xKLMN, vinput_scale);
    vout2xOPQR = vmulq_f32(vout2xOPQR, vinput_scale);
    vout2xSTUV = vmulq_f32(vout2xSTUV, vinput_scale);
    vout3x0123 = vmulq_f32(vout3x0123, vinput_scale);
    vout3x4567 = vmulq_f32(vout3x4567, vinput_scale);
    vout3x89AB = vmulq_f32(vout3x89AB, vinput_scale);
    vout3xCDEF = vmulq_f32(vout3xCDEF, vinput_scale);
    vout3xGHIJ = vmulq_f32(vout3xGHIJ, vinput_scale);
    vout3xKLMN = vmulq_f32(vout3xKLMN, vinput_scale);
    vout3xOPQR = vmulq_f32(vout3xOPQR, vinput_scale);
    vout3xSTUV = vmulq_f32(vout3xSTUV, vinput_scale);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale89AB = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleCDEF = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleGHIJ = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleKLMN = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleOPQR = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleSTUV = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vfmaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vfmaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vfmaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
    #else
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vmlaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vmlaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vmlaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vfmaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vfmaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vfmaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
    #else
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vmlaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vmlaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vmlaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
    #endif
    const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x89AB = vfmaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
      vout1x89AB = vfmaq_f32(vbias89AB, vout1x89AB, vfilter_output_scale89AB);
      vout2x89AB = vfmaq_f32(vbias89AB, vout2x89AB, vfilter_output_scale89AB);
      vout3x89AB = vfmaq_f32(vbias89AB, vout3x89AB, vfilter_output_scale89AB);
    #else
      vout0x89AB = vmlaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
      vout1x89AB = vmlaq_f32(vbias89AB, vout1x89AB, vfilter_output_scale89AB);
      vout2x89AB = vmlaq_f32(vbias89AB, vout2x89AB, vfilter_output_scale89AB);
      vout3x89AB = vmlaq_f32(vbias89AB, vout3x89AB, vfilter_output_scale89AB);
    #endif
    const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xCDEF = vfmaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
      vout1xCDEF = vfmaq_f32(vbiasCDEF, vout1xCDEF, vfilter_output_scaleCDEF);
      vout2xCDEF = vfmaq_f32(vbiasCDEF, vout2xCDEF, vfilter_output_scaleCDEF);
      vout3xCDEF = vfmaq_f32(vbiasCDEF, vout3xCDEF, vfilter_output_scaleCDEF);
    #else
      vout0xCDEF = vmlaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
      vout1xCDEF = vmlaq_f32(vbiasCDEF, vout1xCDEF, vfilter_output_scaleCDEF);
      vout2xCDEF = vmlaq_f32(vbiasCDEF, vout2xCDEF, vfilter_output_scaleCDEF);
      vout3xCDEF = vmlaq_f32(vbiasCDEF, vout3xCDEF, vfilter_output_scaleCDEF);
    #endif
    const float32x4_t vbiasGHIJ = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xGHIJ = vfmaq_f32(vbiasGHIJ, vout0xGHIJ, vfilter_output_scaleGHIJ);
      vout1xGHIJ = vfmaq_f32(vbiasGHIJ, vout1xGHIJ, vfilter_output_scaleGHIJ);
      vout2xGHIJ = vfmaq_f32(vbiasGHIJ, vout2xGHIJ, vfilter_output_scaleGHIJ);
      vout3xGHIJ = vfmaq_f32(vbiasGHIJ, vout3xGHIJ, vfilter_output_scaleGHIJ);
    #else
      vout0xGHIJ = vmlaq_f32(vbiasGHIJ, vout0xGHIJ, vfilter_output_scaleGHIJ);
      vout1xGHIJ = vmlaq_f32(vbiasGHIJ, vout1xGHIJ, vfilter_output_scaleGHIJ);
      vout2xGHIJ = vmlaq_f32(vbiasGHIJ, vout2xGHIJ, vfilter_output_scaleGHIJ);
      vout3xGHIJ = vmlaq_f32(vbiasGHIJ, vout3xGHIJ, vfilter_output_scaleGHIJ);
    #endif
    const float32x4_t vbiasKLMN = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xKLMN = vfmaq_f32(vbiasKLMN, vout0xKLMN, vfilter_output_scaleKLMN);
      vout1xKLMN = vfmaq_f32(vbiasKLMN, vout1xKLMN, vfilter_output_scaleKLMN);
      vout2xKLMN = vfmaq_f32(vbiasKLMN, vout2xKLMN, vfilter_output_scaleKLMN);
      vout3xKLMN = vfmaq_f32(vbiasKLMN, vout3xKLMN, vfilter_output_scaleKLMN);
    #else
      vout0xKLMN = vmlaq_f32(vbiasKLMN, vout0xKLMN, vfilter_output_scaleKLMN);
      vout1xKLMN = vmlaq_f32(vbiasKLMN, vout1xKLMN, vfilter_output_scaleKLMN);
      vout2xKLMN = vmlaq_f32(vbiasKLMN, vout2xKLMN, vfilter_output_scaleKLMN);
      vout3xKLMN = vmlaq_f32(vbiasKLMN, vout3xKLMN, vfilter_output_scaleKLMN);
    #endif
    const float32x4_t vbiasOPQR = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xOPQR = vfmaq_f32(vbiasOPQR, vout0xOPQR, vfilter_output_scaleOPQR);
      vout1xOPQR = vfmaq_f32(vbiasOPQR, vout1xOPQR, vfilter_output_scaleOPQR);
      vout2xOPQR = vfmaq_f32(vbiasOPQR, vout2xOPQR, vfilter_output_scaleOPQR);
      vout3xOPQR = vfmaq_f32(vbiasOPQR, vout3xOPQR, vfilter_output_scaleOPQR);
    #else
      vout0xOPQR = vmlaq_f32(vbiasOPQR, vout0xOPQR, vfilter_output_scaleOPQR);
      vout1xOPQR = vmlaq_f32(vbiasOPQR, vout1xOPQR, vfilter_output_scaleOPQR);
      vout2xOPQR = vmlaq_f32(vbiasOPQR, vout2xOPQR, vfilter_output_scaleOPQR);
      vout3xOPQR = vmlaq_f32(vbiasOPQR, vout3xOPQR, vfilter_output_scaleOPQR);
    #endif
    const float32x4_t vbiasSTUV = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xSTUV = vfmaq_f32(vbiasSTUV, vout0xSTUV, vfilter_output_scaleSTUV);
      vout1xSTUV = vfmaq_f32(vbiasSTUV, vout1xSTUV, vfilter_output_scaleSTUV);
      vout2xSTUV = vfmaq_f32(vbiasSTUV, vout2xSTUV, vfilter_output_scaleSTUV);
      vout3xSTUV = vfmaq_f32(vbiasSTUV, vout3xSTUV, vfilter_output_scaleSTUV);
    #else
      vout0xSTUV = vmlaq_f32(vbiasSTUV, vout0xSTUV, vfilter_output_scaleSTUV);
      vout1xSTUV = vmlaq_f32(vbiasSTUV, vout1xSTUV, vfilter_output_scaleSTUV);
      vout2xSTUV = vmlaq_f32(vbiasSTUV, vout2xSTUV, vfilter_output_scaleSTUV);
      vout3xSTUV = vmlaq_f32(vbiasSTUV, vout3xSTUV, vfilter_output_scaleSTUV);
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
    vout1x0123 = vmaxq_f32(vout1x0123, voutput_min);
    vout1x4567 = vmaxq_f32(vout1x4567, voutput_min);
    vout1x89AB = vmaxq_f32(vout1x89AB, voutput_min);
    vout1xCDEF = vmaxq_f32(vout1xCDEF, voutput_min);
    vout1xGHIJ = vmaxq_f32(vout1xGHIJ, voutput_min);
    vout1xKLMN = vmaxq_f32(vout1xKLMN, voutput_min);
    vout1xOPQR = vmaxq_f32(vout1xOPQR, voutput_min);
    vout1xSTUV = vmaxq_f32(vout1xSTUV, voutput_min);
    vout2x0123 = vmaxq_f32(vout2x0123, voutput_min);
    vout2x4567 = vmaxq_f32(vout2x4567, voutput_min);
    vout2x89AB = vmaxq_f32(vout2x89AB, voutput_min);
    vout2xCDEF = vmaxq_f32(vout2xCDEF, voutput_min);
    vout2xGHIJ = vmaxq_f32(vout2xGHIJ, voutput_min);
    vout2xKLMN = vmaxq_f32(vout2xKLMN, voutput_min);
    vout2xOPQR = vmaxq_f32(vout2xOPQR, voutput_min);
    vout2xSTUV = vmaxq_f32(vout2xSTUV, voutput_min);
    vout3x0123 = vmaxq_f32(vout3x0123, voutput_min);
    vout3x4567 = vmaxq_f32(vout3x4567, voutput_min);
    vout3x89AB = vmaxq_f32(vout3x89AB, voutput_min);
    vout3xCDEF = vmaxq_f32(vout3xCDEF, voutput_min);
    vout3xGHIJ = vmaxq_f32(vout3xGHIJ, voutput_min);
    vout3xKLMN = vmaxq_f32(vout3xKLMN, voutput_min);
    vout3xOPQR = vmaxq_f32(vout3xOPQR, voutput_min);
    vout3xSTUV = vmaxq_f32(vout3xSTUV, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);
    vout0x89AB = vminq_f32(vout0x89AB, voutput_max);
    vout0xCDEF = vminq_f32(vout0xCDEF, voutput_max);
    vout0xGHIJ = vminq_f32(vout0xGHIJ, voutput_max);
    vout0xKLMN = vminq_f32(vout0xKLMN, voutput_max);
    vout0xOPQR = vminq_f32(vout0xOPQR, voutput_max);
    vout0xSTUV = vminq_f32(vout0xSTUV, voutput_max);
    vout1x0123 = vminq_f32(vout1x0123, voutput_max);
    vout1x4567 = vminq_f32(vout1x4567, voutput_max);
    vout1x89AB = vminq_f32(vout1x89AB, voutput_max);
    vout1xCDEF = vminq_f32(vout1xCDEF, voutput_max);
    vout1xGHIJ = vminq_f32(vout1xGHIJ, voutput_max);
    vout1xKLMN = vminq_f32(vout1xKLMN, voutput_max);
    vout1xOPQR = vminq_f32(vout1xOPQR, voutput_max);
    vout1xSTUV = vminq_f32(vout1xSTUV, voutput_max);
    vout2x0123 = vminq_f32(vout2x0123, voutput_max);
    vout2x4567 = vminq_f32(vout2x4567, voutput_max);
    vout2x89AB = vminq_f32(vout2x89AB, voutput_max);
    vout2xCDEF = vminq_f32(vout2xCDEF, voutput_max);
    vout2xGHIJ = vminq_f32(vout2xGHIJ, voutput_max);
    vout2xKLMN = vminq_f32(vout2xKLMN, voutput_max);
    vout2xOPQR = vminq_f32(vout2xOPQR, voutput_max);
    vout2xSTUV = vminq_f32(vout2xSTUV, voutput_max);
    vout3x0123 = vminq_f32(vout3x0123, voutput_max);
    vout3x4567 = vminq_f32(vout3x4567, voutput_max);
    vout3x89AB = vminq_f32(vout3x89AB, voutput_max);
    vout3xCDEF = vminq_f32(vout3xCDEF, voutput_max);
    vout3xGHIJ = vminq_f32(vout3xGHIJ, voutput_max);
    vout3xKLMN = vminq_f32(vout3xKLMN, voutput_max);
    vout3xOPQR = vminq_f32(vout3xOPQR, voutput_max);
    vout3xSTUV = vminq_f32(vout3xSTUV, voutput_max);

    if XNN_LIKELY(nc >= 32) {
      vst1q_f32(c3, vout3x0123);
      vst1q_f32(c3 + 4, vout3x4567);
      vst1q_f32(c3 + 8, vout3x89AB);
      vst1q_f32(c3 + 12, vout3xCDEF);
      vst1q_f32(c3 + 16, vout3xGHIJ);
      vst1q_f32(c3 + 20, vout3xKLMN);
      vst1q_f32(c3 + 24, vout3xOPQR);
      vst1q_f32(c3 + 28, vout3xSTUV);
      vst1q_f32(c2, vout2x0123);
      vst1q_f32(c2 + 4, vout2x4567);
      vst1q_f32(c2 + 8, vout2x89AB);
      vst1q_f32(c2 + 12, vout2xCDEF);
      vst1q_f32(c2 + 16, vout2xGHIJ);
      vst1q_f32(c2 + 20, vout2xKLMN);
      vst1q_f32(c2 + 24, vout2xOPQR);
      vst1q_f32(c2 + 28, vout2xSTUV);
      vst1q_f32(c1, vout1x0123);
      vst1q_f32(c1 + 4, vout1x4567);
      vst1q_f32(c1 + 8, vout1x89AB);
      vst1q_f32(c1 + 12, vout1xCDEF);
      vst1q_f32(c1 + 16, vout1xGHIJ);
      vst1q_f32(c1 + 20, vout1xKLMN);
      vst1q_f32(c1 + 24, vout1xOPQR);
      vst1q_f32(c1 + 28, vout1xSTUV);
      vst1q_f32(c0, vout0x0123);
      vst1q_f32(c0 + 4, vout0x4567);
      vst1q_f32(c0 + 8, vout0x89AB);
      vst1q_f32(c0 + 12, vout0xCDEF);
      vst1q_f32(c0 + 16, vout0xGHIJ);
      vst1q_f32(c0 + 20, vout0xKLMN);
      vst1q_f32(c0 + 24, vout0xOPQR);
      vst1q_f32(c0 + 28, vout0xSTUV);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      nc -= 32;
    } else {
     if (nc & 16) {
       vst1q_f32(c3, vout3x0123); c3 += 4;
       vout3x0123 = vout3xGHIJ;
       vst1q_f32(c2, vout2x0123); c2 += 4;
       vout2x0123 = vout2xGHIJ;
       vst1q_f32(c1, vout1x0123); c1 += 4;
       vout1x0123 = vout1xGHIJ;
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0xGHIJ;
       vst1q_f32(c3, vout3x4567); c3 += 4;
       vout3x4567 = vout3xKLMN;
       vst1q_f32(c2, vout2x4567); c2 += 4;
       vout2x4567 = vout2xKLMN;
       vst1q_f32(c1, vout1x4567); c1 += 4;
       vout1x4567 = vout1xKLMN;
       vst1q_f32(c0, vout0x4567); c0 += 4;
       vout0x4567 = vout0xKLMN;
       vst1q_f32(c3, vout3x89AB); c3 += 4;
       vout3x89AB = vout3xOPQR;
       vst1q_f32(c2, vout2x89AB); c2 += 4;
       vout2x89AB = vout2xOPQR;
       vst1q_f32(c1, vout1x89AB); c1 += 4;
       vout1x89AB = vout1xOPQR;
       vst1q_f32(c0, vout0x89AB); c0 += 4;
       vout0x89AB = vout0xOPQR;
       vst1q_f32(c3, vout3xCDEF); c3 += 4;
       vout3xCDEF = vout3xSTUV;
       vst1q_f32(c2, vout2xCDEF); c2 += 4;
       vout2xCDEF = vout2xSTUV;
       vst1q_f32(c1, vout1xCDEF); c1 += 4;
       vout1xCDEF = vout1xSTUV;
       vst1q_f32(c0, vout0xCDEF); c0 += 4;
       vout0xCDEF = vout0xSTUV;
     }
     if (nc & 8) {
       vst1q_f32(c3, vout3x0123); c3 += 4;
       vout3x0123 = vout3x89AB;
       vst1q_f32(c2, vout2x0123); c2 += 4;
       vout2x0123 = vout2x89AB;
       vst1q_f32(c1, vout1x0123); c1 += 4;
       vout1x0123 = vout1x89AB;
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x89AB;
       vst1q_f32(c3, vout3x4567); c3 += 4;
       vout3x4567 = vout3xCDEF;
       vst1q_f32(c2, vout2x4567); c2 += 4;
       vout2x4567 = vout2xCDEF;
       vst1q_f32(c1, vout1x4567); c1 += 4;
       vout1x4567 = vout1xCDEF;
       vst1q_f32(c0, vout0x4567); c0 += 4;
       vout0x4567 = vout0xCDEF;
     }
     if (nc & 4) {
       vst1q_f32(c3, vout3x0123); c3 += 4;
       vout3x0123 = vout3x4567;
       vst1q_f32(c2, vout2x0123); c2 += 4;
       vout2x0123 = vout2x4567;
       vst1q_f32(c1, vout1x0123); c1 += 4;
       vout1x0123 = vout1x4567;
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x4567;
     }
     float32x2_t vout3x01 = vget_low_f32(vout3x0123);
     float32x2_t vout2x01 = vget_low_f32(vout2x0123);
     float32x2_t vout1x01 = vget_low_f32(vout1x0123);
     float32x2_t vout0x01 = vget_low_f32(vout0x0123);
     if (nc & 2) {
       vst1_f32(c3, vout3x01); c3 += 2;
       vst1_f32(c2, vout2x01); c2 += 2;
       vst1_f32(c1, vout1x01); c1 += 2;
       vst1_f32(c0, vout0x01); c0 += 2;
       vout3x01 = vget_high_f32(vout3x0123);
       vout2x01 = vget_high_f32(vout2x0123);
       vout1x01 = vget_high_f32(vout1x0123);
       vout0x01 = vget_high_f32(vout0x0123);
     }
     if (nc & 1) {
       vst1_lane_f32(c3, vout3x01, 0);
       vst1_lane_f32(c2, vout2x01, 0);
       vst1_lane_f32(c1, vout1x01, 0);
       vst1_lane_f32(c0, vout0x01, 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}
