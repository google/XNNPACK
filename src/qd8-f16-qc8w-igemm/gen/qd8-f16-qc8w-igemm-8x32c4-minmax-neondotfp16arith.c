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


void xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x32c4__neondotfp16arith(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 8);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (8 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  uint16_t* c0 = (uint16_t*) c;
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  uint16_t* c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  uint16_t* c4 = (uint16_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  uint16_t* c5 = (uint16_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    c5 = c4;
  }
  uint16_t* c6 = (uint16_t*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    c6 = c5;
  }
  uint16_t* c7 = (uint16_t*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 8) {
    c7 = c6;
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
    int32x4_t vacc4x0123 = vacc0x0123;
    int32x4_t vacc4x4567 = vacc0x4567;
    int32x4_t vacc4x89AB = vacc0x89AB;
    int32x4_t vacc4xCDEF = vacc0xCDEF;
    int32x4_t vacc4xGHIJ = vacc0xGHIJ;
    int32x4_t vacc4xKLMN = vacc0xKLMN;
    int32x4_t vacc4xOPQR = vacc0xOPQR;
    int32x4_t vacc4xSTUV = vacc0xSTUV;
    int32x4_t vacc5x0123 = vacc0x0123;
    int32x4_t vacc5x4567 = vacc0x4567;
    int32x4_t vacc5x89AB = vacc0x89AB;
    int32x4_t vacc5xCDEF = vacc0xCDEF;
    int32x4_t vacc5xGHIJ = vacc0xGHIJ;
    int32x4_t vacc5xKLMN = vacc0xKLMN;
    int32x4_t vacc5xOPQR = vacc0xOPQR;
    int32x4_t vacc5xSTUV = vacc0xSTUV;
    int32x4_t vacc6x0123 = vacc0x0123;
    int32x4_t vacc6x4567 = vacc0x4567;
    int32x4_t vacc6x89AB = vacc0x89AB;
    int32x4_t vacc6xCDEF = vacc0xCDEF;
    int32x4_t vacc6xGHIJ = vacc0xGHIJ;
    int32x4_t vacc6xKLMN = vacc0xKLMN;
    int32x4_t vacc6xOPQR = vacc0xOPQR;
    int32x4_t vacc6xSTUV = vacc0xSTUV;
    int32x4_t vacc7x0123 = vacc0x0123;
    int32x4_t vacc7x4567 = vacc0x4567;
    int32x4_t vacc7x89AB = vacc0x89AB;
    int32x4_t vacc7xCDEF = vacc0xCDEF;
    int32x4_t vacc7xGHIJ = vacc0xGHIJ;
    int32x4_t vacc7xKLMN = vacc0xKLMN;
    int32x4_t vacc7xOPQR = vacc0xOPQR;
    int32x4_t vacc7xSTUV = vacc0xSTUV;

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
      const int8_t* restrict a4 = a[4];
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const int8_t*) ((uintptr_t) a4 + a_offset);
      } else {
        a4 = zero_data;
      }
      const int8_t* restrict a5 = a[5];
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const int8_t*) ((uintptr_t) a5 + a_offset);
      } else {
        a5 = zero_data;
      }
      const int8_t* restrict a6 = a[6];
      if XNN_UNPREDICTABLE(a6 != zero) {
        a6 = (const int8_t*) ((uintptr_t) a6 + a_offset);
      } else {
        a6 = zero_data;
      }
      const int8_t* restrict a7 = a[7];
      if XNN_UNPREDICTABLE(a7 != zero) {
        a7 = (const int8_t*) ((uintptr_t) a7 + a_offset);
      } else {
        a7 = zero_data;
      }
      a += 8;

      // Inner accumulation loop along the 32 columns.
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

        // Multiply-accumulate: 8x8 * 8x32 --> 8x32.
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
        vacc4x0123 = vdotq_lane_s32(vacc4x0123, vb0123x0123, va4x01234567, 0);
        vacc4x4567 = vdotq_lane_s32(vacc4x4567, vb0123x4567, va4x01234567, 0);
        vacc4x89AB = vdotq_lane_s32(vacc4x89AB, vb0123x89AB, va4x01234567, 0);
        vacc4xCDEF = vdotq_lane_s32(vacc4xCDEF, vb0123xCDEF, va4x01234567, 0);
        vacc4xGHIJ = vdotq_lane_s32(vacc4xGHIJ, vb0123xGHIJ, va4x01234567, 0);
        vacc4xKLMN = vdotq_lane_s32(vacc4xKLMN, vb0123xKLMN, va4x01234567, 0);
        vacc4xOPQR = vdotq_lane_s32(vacc4xOPQR, vb0123xOPQR, va4x01234567, 0);
        vacc4xSTUV = vdotq_lane_s32(vacc4xSTUV, vb0123xSTUV, va4x01234567, 0);
        vacc5x0123 = vdotq_lane_s32(vacc5x0123, vb0123x0123, va5x01234567, 0);
        vacc5x4567 = vdotq_lane_s32(vacc5x4567, vb0123x4567, va5x01234567, 0);
        vacc5x89AB = vdotq_lane_s32(vacc5x89AB, vb0123x89AB, va5x01234567, 0);
        vacc5xCDEF = vdotq_lane_s32(vacc5xCDEF, vb0123xCDEF, va5x01234567, 0);
        vacc5xGHIJ = vdotq_lane_s32(vacc5xGHIJ, vb0123xGHIJ, va5x01234567, 0);
        vacc5xKLMN = vdotq_lane_s32(vacc5xKLMN, vb0123xKLMN, va5x01234567, 0);
        vacc5xOPQR = vdotq_lane_s32(vacc5xOPQR, vb0123xOPQR, va5x01234567, 0);
        vacc5xSTUV = vdotq_lane_s32(vacc5xSTUV, vb0123xSTUV, va5x01234567, 0);
        vacc6x0123 = vdotq_lane_s32(vacc6x0123, vb0123x0123, va6x01234567, 0);
        vacc6x4567 = vdotq_lane_s32(vacc6x4567, vb0123x4567, va6x01234567, 0);
        vacc6x89AB = vdotq_lane_s32(vacc6x89AB, vb0123x89AB, va6x01234567, 0);
        vacc6xCDEF = vdotq_lane_s32(vacc6xCDEF, vb0123xCDEF, va6x01234567, 0);
        vacc6xGHIJ = vdotq_lane_s32(vacc6xGHIJ, vb0123xGHIJ, va6x01234567, 0);
        vacc6xKLMN = vdotq_lane_s32(vacc6xKLMN, vb0123xKLMN, va6x01234567, 0);
        vacc6xOPQR = vdotq_lane_s32(vacc6xOPQR, vb0123xOPQR, va6x01234567, 0);
        vacc6xSTUV = vdotq_lane_s32(vacc6xSTUV, vb0123xSTUV, va6x01234567, 0);
        vacc7x0123 = vdotq_lane_s32(vacc7x0123, vb0123x0123, va7x01234567, 0);
        vacc7x4567 = vdotq_lane_s32(vacc7x4567, vb0123x4567, va7x01234567, 0);
        vacc7x89AB = vdotq_lane_s32(vacc7x89AB, vb0123x89AB, va7x01234567, 0);
        vacc7xCDEF = vdotq_lane_s32(vacc7xCDEF, vb0123xCDEF, va7x01234567, 0);
        vacc7xGHIJ = vdotq_lane_s32(vacc7xGHIJ, vb0123xGHIJ, va7x01234567, 0);
        vacc7xKLMN = vdotq_lane_s32(vacc7xKLMN, vb0123xKLMN, va7x01234567, 0);
        vacc7xOPQR = vdotq_lane_s32(vacc7xOPQR, vb0123xOPQR, va7x01234567, 0);
        vacc7xSTUV = vdotq_lane_s32(vacc7xSTUV, vb0123xSTUV, va7x01234567, 0);
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
        vacc4x0123 = vdotq_lane_s32(vacc4x0123, vb4567x0123, va4x01234567, 1);
        vacc4x4567 = vdotq_lane_s32(vacc4x4567, vb4567x4567, va4x01234567, 1);
        vacc4x89AB = vdotq_lane_s32(vacc4x89AB, vb4567x89AB, va4x01234567, 1);
        vacc4xCDEF = vdotq_lane_s32(vacc4xCDEF, vb4567xCDEF, va4x01234567, 1);
        vacc4xGHIJ = vdotq_lane_s32(vacc4xGHIJ, vb4567xGHIJ, va4x01234567, 1);
        vacc4xKLMN = vdotq_lane_s32(vacc4xKLMN, vb4567xKLMN, va4x01234567, 1);
        vacc4xOPQR = vdotq_lane_s32(vacc4xOPQR, vb4567xOPQR, va4x01234567, 1);
        vacc4xSTUV = vdotq_lane_s32(vacc4xSTUV, vb4567xSTUV, va4x01234567, 1);
        vacc5x0123 = vdotq_lane_s32(vacc5x0123, vb4567x0123, va5x01234567, 1);
        vacc5x4567 = vdotq_lane_s32(vacc5x4567, vb4567x4567, va5x01234567, 1);
        vacc5x89AB = vdotq_lane_s32(vacc5x89AB, vb4567x89AB, va5x01234567, 1);
        vacc5xCDEF = vdotq_lane_s32(vacc5xCDEF, vb4567xCDEF, va5x01234567, 1);
        vacc5xGHIJ = vdotq_lane_s32(vacc5xGHIJ, vb4567xGHIJ, va5x01234567, 1);
        vacc5xKLMN = vdotq_lane_s32(vacc5xKLMN, vb4567xKLMN, va5x01234567, 1);
        vacc5xOPQR = vdotq_lane_s32(vacc5xOPQR, vb4567xOPQR, va5x01234567, 1);
        vacc5xSTUV = vdotq_lane_s32(vacc5xSTUV, vb4567xSTUV, va5x01234567, 1);
        vacc6x0123 = vdotq_lane_s32(vacc6x0123, vb4567x0123, va6x01234567, 1);
        vacc6x4567 = vdotq_lane_s32(vacc6x4567, vb4567x4567, va6x01234567, 1);
        vacc6x89AB = vdotq_lane_s32(vacc6x89AB, vb4567x89AB, va6x01234567, 1);
        vacc6xCDEF = vdotq_lane_s32(vacc6xCDEF, vb4567xCDEF, va6x01234567, 1);
        vacc6xGHIJ = vdotq_lane_s32(vacc6xGHIJ, vb4567xGHIJ, va6x01234567, 1);
        vacc6xKLMN = vdotq_lane_s32(vacc6xKLMN, vb4567xKLMN, va6x01234567, 1);
        vacc6xOPQR = vdotq_lane_s32(vacc6xOPQR, vb4567xOPQR, va6x01234567, 1);
        vacc6xSTUV = vdotq_lane_s32(vacc6xSTUV, vb4567xSTUV, va6x01234567, 1);
        vacc7x0123 = vdotq_lane_s32(vacc7x0123, vb4567x0123, va7x01234567, 1);
        vacc7x4567 = vdotq_lane_s32(vacc7x4567, vb4567x4567, va7x01234567, 1);
        vacc7x89AB = vdotq_lane_s32(vacc7x89AB, vb4567x89AB, va7x01234567, 1);
        vacc7xCDEF = vdotq_lane_s32(vacc7xCDEF, vb4567xCDEF, va7x01234567, 1);
        vacc7xGHIJ = vdotq_lane_s32(vacc7xGHIJ, vb4567xGHIJ, va7x01234567, 1);
        vacc7xKLMN = vdotq_lane_s32(vacc7xKLMN, vb4567xKLMN, va7x01234567, 1);
        vacc7xOPQR = vdotq_lane_s32(vacc7xOPQR, vb4567xOPQR, va7x01234567, 1);
        vacc7xSTUV = vdotq_lane_s32(vacc7xSTUV, vb4567xSTUV, va7x01234567, 1);

        k -= 8 * sizeof(int8_t);
      }
      // Handle up to 4 final positions of `k`
      if XNN_UNLIKELY(k != 0) {
        // Load a 8x4 block of activations.
        const int8x8_t va0x01234567 = vld1_s8(a0);
        const int8x8_t va1x01234567 = vld1_s8(a1);
        const int8x8_t va2x01234567 = vld1_s8(a2);
        const int8x8_t va3x01234567 = vld1_s8(a3);
        const int8x8_t va4x01234567 = vld1_s8(a4);
        const int8x8_t va5x01234567 = vld1_s8(a5);
        const int8x8_t va6x01234567 = vld1_s8(a6);
        const int8x8_t va7x01234567 = vld1_s8(a7);

        // Load a 4x32 block of weights.
        const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xGHIJ = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xKLMN = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xOPQR = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb0123xSTUV = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 8x4 * 4x32 --> 8x32.
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
        vacc4x0123 = vdotq_lane_s32(vacc4x0123, vb0123x0123, va4x01234567, 0);
        vacc4x4567 = vdotq_lane_s32(vacc4x4567, vb0123x4567, va4x01234567, 0);
        vacc4x89AB = vdotq_lane_s32(vacc4x89AB, vb0123x89AB, va4x01234567, 0);
        vacc4xCDEF = vdotq_lane_s32(vacc4xCDEF, vb0123xCDEF, va4x01234567, 0);
        vacc4xGHIJ = vdotq_lane_s32(vacc4xGHIJ, vb0123xGHIJ, va4x01234567, 0);
        vacc4xKLMN = vdotq_lane_s32(vacc4xKLMN, vb0123xKLMN, va4x01234567, 0);
        vacc4xOPQR = vdotq_lane_s32(vacc4xOPQR, vb0123xOPQR, va4x01234567, 0);
        vacc4xSTUV = vdotq_lane_s32(vacc4xSTUV, vb0123xSTUV, va4x01234567, 0);
        vacc5x0123 = vdotq_lane_s32(vacc5x0123, vb0123x0123, va5x01234567, 0);
        vacc5x4567 = vdotq_lane_s32(vacc5x4567, vb0123x4567, va5x01234567, 0);
        vacc5x89AB = vdotq_lane_s32(vacc5x89AB, vb0123x89AB, va5x01234567, 0);
        vacc5xCDEF = vdotq_lane_s32(vacc5xCDEF, vb0123xCDEF, va5x01234567, 0);
        vacc5xGHIJ = vdotq_lane_s32(vacc5xGHIJ, vb0123xGHIJ, va5x01234567, 0);
        vacc5xKLMN = vdotq_lane_s32(vacc5xKLMN, vb0123xKLMN, va5x01234567, 0);
        vacc5xOPQR = vdotq_lane_s32(vacc5xOPQR, vb0123xOPQR, va5x01234567, 0);
        vacc5xSTUV = vdotq_lane_s32(vacc5xSTUV, vb0123xSTUV, va5x01234567, 0);
        vacc6x0123 = vdotq_lane_s32(vacc6x0123, vb0123x0123, va6x01234567, 0);
        vacc6x4567 = vdotq_lane_s32(vacc6x4567, vb0123x4567, va6x01234567, 0);
        vacc6x89AB = vdotq_lane_s32(vacc6x89AB, vb0123x89AB, va6x01234567, 0);
        vacc6xCDEF = vdotq_lane_s32(vacc6xCDEF, vb0123xCDEF, va6x01234567, 0);
        vacc6xGHIJ = vdotq_lane_s32(vacc6xGHIJ, vb0123xGHIJ, va6x01234567, 0);
        vacc6xKLMN = vdotq_lane_s32(vacc6xKLMN, vb0123xKLMN, va6x01234567, 0);
        vacc6xOPQR = vdotq_lane_s32(vacc6xOPQR, vb0123xOPQR, va6x01234567, 0);
        vacc6xSTUV = vdotq_lane_s32(vacc6xSTUV, vb0123xSTUV, va6x01234567, 0);
        vacc7x0123 = vdotq_lane_s32(vacc7x0123, vb0123x0123, va7x01234567, 0);
        vacc7x4567 = vdotq_lane_s32(vacc7x4567, vb0123x4567, va7x01234567, 0);
        vacc7x89AB = vdotq_lane_s32(vacc7x89AB, vb0123x89AB, va7x01234567, 0);
        vacc7xCDEF = vdotq_lane_s32(vacc7xCDEF, vb0123xCDEF, va7x01234567, 0);
        vacc7xGHIJ = vdotq_lane_s32(vacc7xGHIJ, vb0123xGHIJ, va7x01234567, 0);
        vacc7xKLMN = vdotq_lane_s32(vacc7xKLMN, vb0123xKLMN, va7x01234567, 0);
        vacc7xOPQR = vdotq_lane_s32(vacc7xOPQR, vb0123xOPQR, va7x01234567, 0);
        vacc7xSTUV = vdotq_lane_s32(vacc7xSTUV, vb0123xSTUV, va7x01234567, 0);
      }
      p -= 8 * sizeof(void*);
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
    float32x4_t vout4x0123 = vcvtq_f32_s32(vacc4x0123);
    float32x4_t vout4x4567 = vcvtq_f32_s32(vacc4x4567);
    float32x4_t vout4x89AB = vcvtq_f32_s32(vacc4x89AB);
    float32x4_t vout4xCDEF = vcvtq_f32_s32(vacc4xCDEF);
    float32x4_t vout4xGHIJ = vcvtq_f32_s32(vacc4xGHIJ);
    float32x4_t vout4xKLMN = vcvtq_f32_s32(vacc4xKLMN);
    float32x4_t vout4xOPQR = vcvtq_f32_s32(vacc4xOPQR);
    float32x4_t vout4xSTUV = vcvtq_f32_s32(vacc4xSTUV);
    float32x4_t vout5x0123 = vcvtq_f32_s32(vacc5x0123);
    float32x4_t vout5x4567 = vcvtq_f32_s32(vacc5x4567);
    float32x4_t vout5x89AB = vcvtq_f32_s32(vacc5x89AB);
    float32x4_t vout5xCDEF = vcvtq_f32_s32(vacc5xCDEF);
    float32x4_t vout5xGHIJ = vcvtq_f32_s32(vacc5xGHIJ);
    float32x4_t vout5xKLMN = vcvtq_f32_s32(vacc5xKLMN);
    float32x4_t vout5xOPQR = vcvtq_f32_s32(vacc5xOPQR);
    float32x4_t vout5xSTUV = vcvtq_f32_s32(vacc5xSTUV);
    float32x4_t vout6x0123 = vcvtq_f32_s32(vacc6x0123);
    float32x4_t vout6x4567 = vcvtq_f32_s32(vacc6x4567);
    float32x4_t vout6x89AB = vcvtq_f32_s32(vacc6x89AB);
    float32x4_t vout6xCDEF = vcvtq_f32_s32(vacc6xCDEF);
    float32x4_t vout6xGHIJ = vcvtq_f32_s32(vacc6xGHIJ);
    float32x4_t vout6xKLMN = vcvtq_f32_s32(vacc6xKLMN);
    float32x4_t vout6xOPQR = vcvtq_f32_s32(vacc6xOPQR);
    float32x4_t vout6xSTUV = vcvtq_f32_s32(vacc6xSTUV);
    float32x4_t vout7x0123 = vcvtq_f32_s32(vacc7x0123);
    float32x4_t vout7x4567 = vcvtq_f32_s32(vacc7x4567);
    float32x4_t vout7x89AB = vcvtq_f32_s32(vacc7x89AB);
    float32x4_t vout7xCDEF = vcvtq_f32_s32(vacc7xCDEF);
    float32x4_t vout7xGHIJ = vcvtq_f32_s32(vacc7xGHIJ);
    float32x4_t vout7xKLMN = vcvtq_f32_s32(vacc7xKLMN);
    float32x4_t vout7xOPQR = vcvtq_f32_s32(vacc7xOPQR);
    float32x4_t vout7xSTUV = vcvtq_f32_s32(vacc7xSTUV);
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
    vout4x0123 = vmulq_f32(vout4x0123, vinput_scale);
    vout4x4567 = vmulq_f32(vout4x4567, vinput_scale);
    vout4x89AB = vmulq_f32(vout4x89AB, vinput_scale);
    vout4xCDEF = vmulq_f32(vout4xCDEF, vinput_scale);
    vout4xGHIJ = vmulq_f32(vout4xGHIJ, vinput_scale);
    vout4xKLMN = vmulq_f32(vout4xKLMN, vinput_scale);
    vout4xOPQR = vmulq_f32(vout4xOPQR, vinput_scale);
    vout4xSTUV = vmulq_f32(vout4xSTUV, vinput_scale);
    vout5x0123 = vmulq_f32(vout5x0123, vinput_scale);
    vout5x4567 = vmulq_f32(vout5x4567, vinput_scale);
    vout5x89AB = vmulq_f32(vout5x89AB, vinput_scale);
    vout5xCDEF = vmulq_f32(vout5xCDEF, vinput_scale);
    vout5xGHIJ = vmulq_f32(vout5xGHIJ, vinput_scale);
    vout5xKLMN = vmulq_f32(vout5xKLMN, vinput_scale);
    vout5xOPQR = vmulq_f32(vout5xOPQR, vinput_scale);
    vout5xSTUV = vmulq_f32(vout5xSTUV, vinput_scale);
    vout6x0123 = vmulq_f32(vout6x0123, vinput_scale);
    vout6x4567 = vmulq_f32(vout6x4567, vinput_scale);
    vout6x89AB = vmulq_f32(vout6x89AB, vinput_scale);
    vout6xCDEF = vmulq_f32(vout6xCDEF, vinput_scale);
    vout6xGHIJ = vmulq_f32(vout6xGHIJ, vinput_scale);
    vout6xKLMN = vmulq_f32(vout6xKLMN, vinput_scale);
    vout6xOPQR = vmulq_f32(vout6xOPQR, vinput_scale);
    vout6xSTUV = vmulq_f32(vout6xSTUV, vinput_scale);
    vout7x0123 = vmulq_f32(vout7x0123, vinput_scale);
    vout7x4567 = vmulq_f32(vout7x4567, vinput_scale);
    vout7x89AB = vmulq_f32(vout7x89AB, vinput_scale);
    vout7xCDEF = vmulq_f32(vout7xCDEF, vinput_scale);
    vout7xGHIJ = vmulq_f32(vout7xGHIJ, vinput_scale);
    vout7xKLMN = vmulq_f32(vout7xKLMN, vinput_scale);
    vout7xOPQR = vmulq_f32(vout7xOPQR, vinput_scale);
    vout7xSTUV = vmulq_f32(vout7xSTUV, vinput_scale);

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
      vout4x0123 = vfmaq_f32(vbias0123, vout4x0123, vfilter_output_scale0123);
      vout5x0123 = vfmaq_f32(vbias0123, vout5x0123, vfilter_output_scale0123);
      vout6x0123 = vfmaq_f32(vbias0123, vout6x0123, vfilter_output_scale0123);
      vout7x0123 = vfmaq_f32(vbias0123, vout7x0123, vfilter_output_scale0123);
    #else
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vmlaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vmlaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vmlaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
      vout4x0123 = vmlaq_f32(vbias0123, vout4x0123, vfilter_output_scale0123);
      vout5x0123 = vmlaq_f32(vbias0123, vout5x0123, vfilter_output_scale0123);
      vout6x0123 = vmlaq_f32(vbias0123, vout6x0123, vfilter_output_scale0123);
      vout7x0123 = vmlaq_f32(vbias0123, vout7x0123, vfilter_output_scale0123);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vfmaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vfmaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vfmaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
      vout4x4567 = vfmaq_f32(vbias4567, vout4x4567, vfilter_output_scale4567);
      vout5x4567 = vfmaq_f32(vbias4567, vout5x4567, vfilter_output_scale4567);
      vout6x4567 = vfmaq_f32(vbias4567, vout6x4567, vfilter_output_scale4567);
      vout7x4567 = vfmaq_f32(vbias4567, vout7x4567, vfilter_output_scale4567);
    #else
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vmlaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vmlaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vmlaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
      vout4x4567 = vmlaq_f32(vbias4567, vout4x4567, vfilter_output_scale4567);
      vout5x4567 = vmlaq_f32(vbias4567, vout5x4567, vfilter_output_scale4567);
      vout6x4567 = vmlaq_f32(vbias4567, vout6x4567, vfilter_output_scale4567);
      vout7x4567 = vmlaq_f32(vbias4567, vout7x4567, vfilter_output_scale4567);
    #endif
    const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x89AB = vfmaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
      vout1x89AB = vfmaq_f32(vbias89AB, vout1x89AB, vfilter_output_scale89AB);
      vout2x89AB = vfmaq_f32(vbias89AB, vout2x89AB, vfilter_output_scale89AB);
      vout3x89AB = vfmaq_f32(vbias89AB, vout3x89AB, vfilter_output_scale89AB);
      vout4x89AB = vfmaq_f32(vbias89AB, vout4x89AB, vfilter_output_scale89AB);
      vout5x89AB = vfmaq_f32(vbias89AB, vout5x89AB, vfilter_output_scale89AB);
      vout6x89AB = vfmaq_f32(vbias89AB, vout6x89AB, vfilter_output_scale89AB);
      vout7x89AB = vfmaq_f32(vbias89AB, vout7x89AB, vfilter_output_scale89AB);
    #else
      vout0x89AB = vmlaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
      vout1x89AB = vmlaq_f32(vbias89AB, vout1x89AB, vfilter_output_scale89AB);
      vout2x89AB = vmlaq_f32(vbias89AB, vout2x89AB, vfilter_output_scale89AB);
      vout3x89AB = vmlaq_f32(vbias89AB, vout3x89AB, vfilter_output_scale89AB);
      vout4x89AB = vmlaq_f32(vbias89AB, vout4x89AB, vfilter_output_scale89AB);
      vout5x89AB = vmlaq_f32(vbias89AB, vout5x89AB, vfilter_output_scale89AB);
      vout6x89AB = vmlaq_f32(vbias89AB, vout6x89AB, vfilter_output_scale89AB);
      vout7x89AB = vmlaq_f32(vbias89AB, vout7x89AB, vfilter_output_scale89AB);
    #endif
    const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xCDEF = vfmaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
      vout1xCDEF = vfmaq_f32(vbiasCDEF, vout1xCDEF, vfilter_output_scaleCDEF);
      vout2xCDEF = vfmaq_f32(vbiasCDEF, vout2xCDEF, vfilter_output_scaleCDEF);
      vout3xCDEF = vfmaq_f32(vbiasCDEF, vout3xCDEF, vfilter_output_scaleCDEF);
      vout4xCDEF = vfmaq_f32(vbiasCDEF, vout4xCDEF, vfilter_output_scaleCDEF);
      vout5xCDEF = vfmaq_f32(vbiasCDEF, vout5xCDEF, vfilter_output_scaleCDEF);
      vout6xCDEF = vfmaq_f32(vbiasCDEF, vout6xCDEF, vfilter_output_scaleCDEF);
      vout7xCDEF = vfmaq_f32(vbiasCDEF, vout7xCDEF, vfilter_output_scaleCDEF);
    #else
      vout0xCDEF = vmlaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
      vout1xCDEF = vmlaq_f32(vbiasCDEF, vout1xCDEF, vfilter_output_scaleCDEF);
      vout2xCDEF = vmlaq_f32(vbiasCDEF, vout2xCDEF, vfilter_output_scaleCDEF);
      vout3xCDEF = vmlaq_f32(vbiasCDEF, vout3xCDEF, vfilter_output_scaleCDEF);
      vout4xCDEF = vmlaq_f32(vbiasCDEF, vout4xCDEF, vfilter_output_scaleCDEF);
      vout5xCDEF = vmlaq_f32(vbiasCDEF, vout5xCDEF, vfilter_output_scaleCDEF);
      vout6xCDEF = vmlaq_f32(vbiasCDEF, vout6xCDEF, vfilter_output_scaleCDEF);
      vout7xCDEF = vmlaq_f32(vbiasCDEF, vout7xCDEF, vfilter_output_scaleCDEF);
    #endif
    const float32x4_t vbiasGHIJ = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xGHIJ = vfmaq_f32(vbiasGHIJ, vout0xGHIJ, vfilter_output_scaleGHIJ);
      vout1xGHIJ = vfmaq_f32(vbiasGHIJ, vout1xGHIJ, vfilter_output_scaleGHIJ);
      vout2xGHIJ = vfmaq_f32(vbiasGHIJ, vout2xGHIJ, vfilter_output_scaleGHIJ);
      vout3xGHIJ = vfmaq_f32(vbiasGHIJ, vout3xGHIJ, vfilter_output_scaleGHIJ);
      vout4xGHIJ = vfmaq_f32(vbiasGHIJ, vout4xGHIJ, vfilter_output_scaleGHIJ);
      vout5xGHIJ = vfmaq_f32(vbiasGHIJ, vout5xGHIJ, vfilter_output_scaleGHIJ);
      vout6xGHIJ = vfmaq_f32(vbiasGHIJ, vout6xGHIJ, vfilter_output_scaleGHIJ);
      vout7xGHIJ = vfmaq_f32(vbiasGHIJ, vout7xGHIJ, vfilter_output_scaleGHIJ);
    #else
      vout0xGHIJ = vmlaq_f32(vbiasGHIJ, vout0xGHIJ, vfilter_output_scaleGHIJ);
      vout1xGHIJ = vmlaq_f32(vbiasGHIJ, vout1xGHIJ, vfilter_output_scaleGHIJ);
      vout2xGHIJ = vmlaq_f32(vbiasGHIJ, vout2xGHIJ, vfilter_output_scaleGHIJ);
      vout3xGHIJ = vmlaq_f32(vbiasGHIJ, vout3xGHIJ, vfilter_output_scaleGHIJ);
      vout4xGHIJ = vmlaq_f32(vbiasGHIJ, vout4xGHIJ, vfilter_output_scaleGHIJ);
      vout5xGHIJ = vmlaq_f32(vbiasGHIJ, vout5xGHIJ, vfilter_output_scaleGHIJ);
      vout6xGHIJ = vmlaq_f32(vbiasGHIJ, vout6xGHIJ, vfilter_output_scaleGHIJ);
      vout7xGHIJ = vmlaq_f32(vbiasGHIJ, vout7xGHIJ, vfilter_output_scaleGHIJ);
    #endif
    const float32x4_t vbiasKLMN = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xKLMN = vfmaq_f32(vbiasKLMN, vout0xKLMN, vfilter_output_scaleKLMN);
      vout1xKLMN = vfmaq_f32(vbiasKLMN, vout1xKLMN, vfilter_output_scaleKLMN);
      vout2xKLMN = vfmaq_f32(vbiasKLMN, vout2xKLMN, vfilter_output_scaleKLMN);
      vout3xKLMN = vfmaq_f32(vbiasKLMN, vout3xKLMN, vfilter_output_scaleKLMN);
      vout4xKLMN = vfmaq_f32(vbiasKLMN, vout4xKLMN, vfilter_output_scaleKLMN);
      vout5xKLMN = vfmaq_f32(vbiasKLMN, vout5xKLMN, vfilter_output_scaleKLMN);
      vout6xKLMN = vfmaq_f32(vbiasKLMN, vout6xKLMN, vfilter_output_scaleKLMN);
      vout7xKLMN = vfmaq_f32(vbiasKLMN, vout7xKLMN, vfilter_output_scaleKLMN);
    #else
      vout0xKLMN = vmlaq_f32(vbiasKLMN, vout0xKLMN, vfilter_output_scaleKLMN);
      vout1xKLMN = vmlaq_f32(vbiasKLMN, vout1xKLMN, vfilter_output_scaleKLMN);
      vout2xKLMN = vmlaq_f32(vbiasKLMN, vout2xKLMN, vfilter_output_scaleKLMN);
      vout3xKLMN = vmlaq_f32(vbiasKLMN, vout3xKLMN, vfilter_output_scaleKLMN);
      vout4xKLMN = vmlaq_f32(vbiasKLMN, vout4xKLMN, vfilter_output_scaleKLMN);
      vout5xKLMN = vmlaq_f32(vbiasKLMN, vout5xKLMN, vfilter_output_scaleKLMN);
      vout6xKLMN = vmlaq_f32(vbiasKLMN, vout6xKLMN, vfilter_output_scaleKLMN);
      vout7xKLMN = vmlaq_f32(vbiasKLMN, vout7xKLMN, vfilter_output_scaleKLMN);
    #endif
    const float32x4_t vbiasOPQR = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xOPQR = vfmaq_f32(vbiasOPQR, vout0xOPQR, vfilter_output_scaleOPQR);
      vout1xOPQR = vfmaq_f32(vbiasOPQR, vout1xOPQR, vfilter_output_scaleOPQR);
      vout2xOPQR = vfmaq_f32(vbiasOPQR, vout2xOPQR, vfilter_output_scaleOPQR);
      vout3xOPQR = vfmaq_f32(vbiasOPQR, vout3xOPQR, vfilter_output_scaleOPQR);
      vout4xOPQR = vfmaq_f32(vbiasOPQR, vout4xOPQR, vfilter_output_scaleOPQR);
      vout5xOPQR = vfmaq_f32(vbiasOPQR, vout5xOPQR, vfilter_output_scaleOPQR);
      vout6xOPQR = vfmaq_f32(vbiasOPQR, vout6xOPQR, vfilter_output_scaleOPQR);
      vout7xOPQR = vfmaq_f32(vbiasOPQR, vout7xOPQR, vfilter_output_scaleOPQR);
    #else
      vout0xOPQR = vmlaq_f32(vbiasOPQR, vout0xOPQR, vfilter_output_scaleOPQR);
      vout1xOPQR = vmlaq_f32(vbiasOPQR, vout1xOPQR, vfilter_output_scaleOPQR);
      vout2xOPQR = vmlaq_f32(vbiasOPQR, vout2xOPQR, vfilter_output_scaleOPQR);
      vout3xOPQR = vmlaq_f32(vbiasOPQR, vout3xOPQR, vfilter_output_scaleOPQR);
      vout4xOPQR = vmlaq_f32(vbiasOPQR, vout4xOPQR, vfilter_output_scaleOPQR);
      vout5xOPQR = vmlaq_f32(vbiasOPQR, vout5xOPQR, vfilter_output_scaleOPQR);
      vout6xOPQR = vmlaq_f32(vbiasOPQR, vout6xOPQR, vfilter_output_scaleOPQR);
      vout7xOPQR = vmlaq_f32(vbiasOPQR, vout7xOPQR, vfilter_output_scaleOPQR);
    #endif
    const float32x4_t vbiasSTUV = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xSTUV = vfmaq_f32(vbiasSTUV, vout0xSTUV, vfilter_output_scaleSTUV);
      vout1xSTUV = vfmaq_f32(vbiasSTUV, vout1xSTUV, vfilter_output_scaleSTUV);
      vout2xSTUV = vfmaq_f32(vbiasSTUV, vout2xSTUV, vfilter_output_scaleSTUV);
      vout3xSTUV = vfmaq_f32(vbiasSTUV, vout3xSTUV, vfilter_output_scaleSTUV);
      vout4xSTUV = vfmaq_f32(vbiasSTUV, vout4xSTUV, vfilter_output_scaleSTUV);
      vout5xSTUV = vfmaq_f32(vbiasSTUV, vout5xSTUV, vfilter_output_scaleSTUV);
      vout6xSTUV = vfmaq_f32(vbiasSTUV, vout6xSTUV, vfilter_output_scaleSTUV);
      vout7xSTUV = vfmaq_f32(vbiasSTUV, vout7xSTUV, vfilter_output_scaleSTUV);
    #else
      vout0xSTUV = vmlaq_f32(vbiasSTUV, vout0xSTUV, vfilter_output_scaleSTUV);
      vout1xSTUV = vmlaq_f32(vbiasSTUV, vout1xSTUV, vfilter_output_scaleSTUV);
      vout2xSTUV = vmlaq_f32(vbiasSTUV, vout2xSTUV, vfilter_output_scaleSTUV);
      vout3xSTUV = vmlaq_f32(vbiasSTUV, vout3xSTUV, vfilter_output_scaleSTUV);
      vout4xSTUV = vmlaq_f32(vbiasSTUV, vout4xSTUV, vfilter_output_scaleSTUV);
      vout5xSTUV = vmlaq_f32(vbiasSTUV, vout5xSTUV, vfilter_output_scaleSTUV);
      vout6xSTUV = vmlaq_f32(vbiasSTUV, vout6xSTUV, vfilter_output_scaleSTUV);
      vout7xSTUV = vmlaq_f32(vbiasSTUV, vout7xSTUV, vfilter_output_scaleSTUV);
    #endif

    float16x8_t vfp16out0x01234567 = vcombine_f16(vcvt_f16_f32(vout0x0123), vcvt_f16_f32(vout0x4567));
    float16x8_t vfp16out0x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout0x89AB), vcvt_f16_f32(vout0xCDEF));
    float16x8_t vfp16out0xGHIJKLMN = vcombine_f16(vcvt_f16_f32(vout0xGHIJ), vcvt_f16_f32(vout0xKLMN));
    float16x8_t vfp16out0xOPQRSTUV = vcombine_f16(vcvt_f16_f32(vout0xOPQR), vcvt_f16_f32(vout0xSTUV));
    float16x8_t vfp16out1x01234567 = vcombine_f16(vcvt_f16_f32(vout1x0123), vcvt_f16_f32(vout1x4567));
    float16x8_t vfp16out1x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout1x89AB), vcvt_f16_f32(vout1xCDEF));
    float16x8_t vfp16out1xGHIJKLMN = vcombine_f16(vcvt_f16_f32(vout1xGHIJ), vcvt_f16_f32(vout1xKLMN));
    float16x8_t vfp16out1xOPQRSTUV = vcombine_f16(vcvt_f16_f32(vout1xOPQR), vcvt_f16_f32(vout1xSTUV));
    float16x8_t vfp16out2x01234567 = vcombine_f16(vcvt_f16_f32(vout2x0123), vcvt_f16_f32(vout2x4567));
    float16x8_t vfp16out2x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout2x89AB), vcvt_f16_f32(vout2xCDEF));
    float16x8_t vfp16out2xGHIJKLMN = vcombine_f16(vcvt_f16_f32(vout2xGHIJ), vcvt_f16_f32(vout2xKLMN));
    float16x8_t vfp16out2xOPQRSTUV = vcombine_f16(vcvt_f16_f32(vout2xOPQR), vcvt_f16_f32(vout2xSTUV));
    float16x8_t vfp16out3x01234567 = vcombine_f16(vcvt_f16_f32(vout3x0123), vcvt_f16_f32(vout3x4567));
    float16x8_t vfp16out3x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout3x89AB), vcvt_f16_f32(vout3xCDEF));
    float16x8_t vfp16out3xGHIJKLMN = vcombine_f16(vcvt_f16_f32(vout3xGHIJ), vcvt_f16_f32(vout3xKLMN));
    float16x8_t vfp16out3xOPQRSTUV = vcombine_f16(vcvt_f16_f32(vout3xOPQR), vcvt_f16_f32(vout3xSTUV));
    float16x8_t vfp16out4x01234567 = vcombine_f16(vcvt_f16_f32(vout4x0123), vcvt_f16_f32(vout4x4567));
    float16x8_t vfp16out4x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout4x89AB), vcvt_f16_f32(vout4xCDEF));
    float16x8_t vfp16out4xGHIJKLMN = vcombine_f16(vcvt_f16_f32(vout4xGHIJ), vcvt_f16_f32(vout4xKLMN));
    float16x8_t vfp16out4xOPQRSTUV = vcombine_f16(vcvt_f16_f32(vout4xOPQR), vcvt_f16_f32(vout4xSTUV));
    float16x8_t vfp16out5x01234567 = vcombine_f16(vcvt_f16_f32(vout5x0123), vcvt_f16_f32(vout5x4567));
    float16x8_t vfp16out5x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout5x89AB), vcvt_f16_f32(vout5xCDEF));
    float16x8_t vfp16out5xGHIJKLMN = vcombine_f16(vcvt_f16_f32(vout5xGHIJ), vcvt_f16_f32(vout5xKLMN));
    float16x8_t vfp16out5xOPQRSTUV = vcombine_f16(vcvt_f16_f32(vout5xOPQR), vcvt_f16_f32(vout5xSTUV));
    float16x8_t vfp16out6x01234567 = vcombine_f16(vcvt_f16_f32(vout6x0123), vcvt_f16_f32(vout6x4567));
    float16x8_t vfp16out6x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout6x89AB), vcvt_f16_f32(vout6xCDEF));
    float16x8_t vfp16out6xGHIJKLMN = vcombine_f16(vcvt_f16_f32(vout6xGHIJ), vcvt_f16_f32(vout6xKLMN));
    float16x8_t vfp16out6xOPQRSTUV = vcombine_f16(vcvt_f16_f32(vout6xOPQR), vcvt_f16_f32(vout6xSTUV));
    float16x8_t vfp16out7x01234567 = vcombine_f16(vcvt_f16_f32(vout7x0123), vcvt_f16_f32(vout7x4567));
    float16x8_t vfp16out7x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout7x89AB), vcvt_f16_f32(vout7xCDEF));
    float16x8_t vfp16out7xGHIJKLMN = vcombine_f16(vcvt_f16_f32(vout7xGHIJ), vcvt_f16_f32(vout7xKLMN));
    float16x8_t vfp16out7xOPQRSTUV = vcombine_f16(vcvt_f16_f32(vout7xOPQR), vcvt_f16_f32(vout7xSTUV));

    const float16x8_t voutput_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
    vfp16out0x01234567 = vmaxq_f16(vfp16out0x01234567, voutput_min);
    vfp16out0x89ABCDEF = vmaxq_f16(vfp16out0x89ABCDEF, voutput_min);
    vfp16out0xGHIJKLMN = vmaxq_f16(vfp16out0xGHIJKLMN, voutput_min);
    vfp16out0xOPQRSTUV = vmaxq_f16(vfp16out0xOPQRSTUV, voutput_min);
    vfp16out1x01234567 = vmaxq_f16(vfp16out1x01234567, voutput_min);
    vfp16out1x89ABCDEF = vmaxq_f16(vfp16out1x89ABCDEF, voutput_min);
    vfp16out1xGHIJKLMN = vmaxq_f16(vfp16out1xGHIJKLMN, voutput_min);
    vfp16out1xOPQRSTUV = vmaxq_f16(vfp16out1xOPQRSTUV, voutput_min);
    vfp16out2x01234567 = vmaxq_f16(vfp16out2x01234567, voutput_min);
    vfp16out2x89ABCDEF = vmaxq_f16(vfp16out2x89ABCDEF, voutput_min);
    vfp16out2xGHIJKLMN = vmaxq_f16(vfp16out2xGHIJKLMN, voutput_min);
    vfp16out2xOPQRSTUV = vmaxq_f16(vfp16out2xOPQRSTUV, voutput_min);
    vfp16out3x01234567 = vmaxq_f16(vfp16out3x01234567, voutput_min);
    vfp16out3x89ABCDEF = vmaxq_f16(vfp16out3x89ABCDEF, voutput_min);
    vfp16out3xGHIJKLMN = vmaxq_f16(vfp16out3xGHIJKLMN, voutput_min);
    vfp16out3xOPQRSTUV = vmaxq_f16(vfp16out3xOPQRSTUV, voutput_min);
    vfp16out4x01234567 = vmaxq_f16(vfp16out4x01234567, voutput_min);
    vfp16out4x89ABCDEF = vmaxq_f16(vfp16out4x89ABCDEF, voutput_min);
    vfp16out4xGHIJKLMN = vmaxq_f16(vfp16out4xGHIJKLMN, voutput_min);
    vfp16out4xOPQRSTUV = vmaxq_f16(vfp16out4xOPQRSTUV, voutput_min);
    vfp16out5x01234567 = vmaxq_f16(vfp16out5x01234567, voutput_min);
    vfp16out5x89ABCDEF = vmaxq_f16(vfp16out5x89ABCDEF, voutput_min);
    vfp16out5xGHIJKLMN = vmaxq_f16(vfp16out5xGHIJKLMN, voutput_min);
    vfp16out5xOPQRSTUV = vmaxq_f16(vfp16out5xOPQRSTUV, voutput_min);
    vfp16out6x01234567 = vmaxq_f16(vfp16out6x01234567, voutput_min);
    vfp16out6x89ABCDEF = vmaxq_f16(vfp16out6x89ABCDEF, voutput_min);
    vfp16out6xGHIJKLMN = vmaxq_f16(vfp16out6xGHIJKLMN, voutput_min);
    vfp16out6xOPQRSTUV = vmaxq_f16(vfp16out6xOPQRSTUV, voutput_min);
    vfp16out7x01234567 = vmaxq_f16(vfp16out7x01234567, voutput_min);
    vfp16out7x89ABCDEF = vmaxq_f16(vfp16out7x89ABCDEF, voutput_min);
    vfp16out7xGHIJKLMN = vmaxq_f16(vfp16out7xGHIJKLMN, voutput_min);
    vfp16out7xOPQRSTUV = vmaxq_f16(vfp16out7xOPQRSTUV, voutput_min);
    const float16x8_t voutput_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
    vfp16out0x01234567 = vminq_f16(vfp16out0x01234567, voutput_max);
    vfp16out0x89ABCDEF = vminq_f16(vfp16out0x89ABCDEF, voutput_max);
    vfp16out0xGHIJKLMN = vminq_f16(vfp16out0xGHIJKLMN, voutput_max);
    vfp16out0xOPQRSTUV = vminq_f16(vfp16out0xOPQRSTUV, voutput_max);
    vfp16out1x01234567 = vminq_f16(vfp16out1x01234567, voutput_max);
    vfp16out1x89ABCDEF = vminq_f16(vfp16out1x89ABCDEF, voutput_max);
    vfp16out1xGHIJKLMN = vminq_f16(vfp16out1xGHIJKLMN, voutput_max);
    vfp16out1xOPQRSTUV = vminq_f16(vfp16out1xOPQRSTUV, voutput_max);
    vfp16out2x01234567 = vminq_f16(vfp16out2x01234567, voutput_max);
    vfp16out2x89ABCDEF = vminq_f16(vfp16out2x89ABCDEF, voutput_max);
    vfp16out2xGHIJKLMN = vminq_f16(vfp16out2xGHIJKLMN, voutput_max);
    vfp16out2xOPQRSTUV = vminq_f16(vfp16out2xOPQRSTUV, voutput_max);
    vfp16out3x01234567 = vminq_f16(vfp16out3x01234567, voutput_max);
    vfp16out3x89ABCDEF = vminq_f16(vfp16out3x89ABCDEF, voutput_max);
    vfp16out3xGHIJKLMN = vminq_f16(vfp16out3xGHIJKLMN, voutput_max);
    vfp16out3xOPQRSTUV = vminq_f16(vfp16out3xOPQRSTUV, voutput_max);
    vfp16out4x01234567 = vminq_f16(vfp16out4x01234567, voutput_max);
    vfp16out4x89ABCDEF = vminq_f16(vfp16out4x89ABCDEF, voutput_max);
    vfp16out4xGHIJKLMN = vminq_f16(vfp16out4xGHIJKLMN, voutput_max);
    vfp16out4xOPQRSTUV = vminq_f16(vfp16out4xOPQRSTUV, voutput_max);
    vfp16out5x01234567 = vminq_f16(vfp16out5x01234567, voutput_max);
    vfp16out5x89ABCDEF = vminq_f16(vfp16out5x89ABCDEF, voutput_max);
    vfp16out5xGHIJKLMN = vminq_f16(vfp16out5xGHIJKLMN, voutput_max);
    vfp16out5xOPQRSTUV = vminq_f16(vfp16out5xOPQRSTUV, voutput_max);
    vfp16out6x01234567 = vminq_f16(vfp16out6x01234567, voutput_max);
    vfp16out6x89ABCDEF = vminq_f16(vfp16out6x89ABCDEF, voutput_max);
    vfp16out6xGHIJKLMN = vminq_f16(vfp16out6xGHIJKLMN, voutput_max);
    vfp16out6xOPQRSTUV = vminq_f16(vfp16out6xOPQRSTUV, voutput_max);
    vfp16out7x01234567 = vminq_f16(vfp16out7x01234567, voutput_max);
    vfp16out7x89ABCDEF = vminq_f16(vfp16out7x89ABCDEF, voutput_max);
    vfp16out7xGHIJKLMN = vminq_f16(vfp16out7xGHIJKLMN, voutput_max);
    vfp16out7xOPQRSTUV = vminq_f16(vfp16out7xOPQRSTUV, voutput_max);
    if XNN_LIKELY(nc >= 32) {
      vst1q_u16(c7, vreinterpretq_u16_f16(vfp16out7x01234567));
      vst1q_u16(c7 + 8, vreinterpretq_u16_f16(vfp16out7x89ABCDEF));
      vst1q_u16(c7 + 16, vreinterpretq_u16_f16(vfp16out7xGHIJKLMN));
      vst1q_u16(c7 + 24, vreinterpretq_u16_f16(vfp16out7xOPQRSTUV));
      vst1q_u16(c6, vreinterpretq_u16_f16(vfp16out6x01234567));
      vst1q_u16(c6 + 8, vreinterpretq_u16_f16(vfp16out6x89ABCDEF));
      vst1q_u16(c6 + 16, vreinterpretq_u16_f16(vfp16out6xGHIJKLMN));
      vst1q_u16(c6 + 24, vreinterpretq_u16_f16(vfp16out6xOPQRSTUV));
      vst1q_u16(c5, vreinterpretq_u16_f16(vfp16out5x01234567));
      vst1q_u16(c5 + 8, vreinterpretq_u16_f16(vfp16out5x89ABCDEF));
      vst1q_u16(c5 + 16, vreinterpretq_u16_f16(vfp16out5xGHIJKLMN));
      vst1q_u16(c5 + 24, vreinterpretq_u16_f16(vfp16out5xOPQRSTUV));
      vst1q_u16(c4, vreinterpretq_u16_f16(vfp16out4x01234567));
      vst1q_u16(c4 + 8, vreinterpretq_u16_f16(vfp16out4x89ABCDEF));
      vst1q_u16(c4 + 16, vreinterpretq_u16_f16(vfp16out4xGHIJKLMN));
      vst1q_u16(c4 + 24, vreinterpretq_u16_f16(vfp16out4xOPQRSTUV));
      vst1q_u16(c3, vreinterpretq_u16_f16(vfp16out3x01234567));
      vst1q_u16(c3 + 8, vreinterpretq_u16_f16(vfp16out3x89ABCDEF));
      vst1q_u16(c3 + 16, vreinterpretq_u16_f16(vfp16out3xGHIJKLMN));
      vst1q_u16(c3 + 24, vreinterpretq_u16_f16(vfp16out3xOPQRSTUV));
      vst1q_u16(c2, vreinterpretq_u16_f16(vfp16out2x01234567));
      vst1q_u16(c2 + 8, vreinterpretq_u16_f16(vfp16out2x89ABCDEF));
      vst1q_u16(c2 + 16, vreinterpretq_u16_f16(vfp16out2xGHIJKLMN));
      vst1q_u16(c2 + 24, vreinterpretq_u16_f16(vfp16out2xOPQRSTUV));
      vst1q_u16(c1, vreinterpretq_u16_f16(vfp16out1x01234567));
      vst1q_u16(c1 + 8, vreinterpretq_u16_f16(vfp16out1x89ABCDEF));
      vst1q_u16(c1 + 16, vreinterpretq_u16_f16(vfp16out1xGHIJKLMN));
      vst1q_u16(c1 + 24, vreinterpretq_u16_f16(vfp16out1xOPQRSTUV));
      vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567));
      vst1q_u16(c0 + 8, vreinterpretq_u16_f16(vfp16out0x89ABCDEF));
      vst1q_u16(c0 + 16, vreinterpretq_u16_f16(vfp16out0xGHIJKLMN));
      vst1q_u16(c0 + 24, vreinterpretq_u16_f16(vfp16out0xOPQRSTUV));

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      c4 = (uint16_t*) ((uintptr_t) c4 + cn_stride);
      c5 = (uint16_t*) ((uintptr_t) c5 + cn_stride);
      c6 = (uint16_t*) ((uintptr_t) c6 + cn_stride);
      c7 = (uint16_t*) ((uintptr_t) c7 + cn_stride);

      nc -= 32;
    } else {
     if (nc & 16) {
       vst1q_u16(c7, vreinterpretq_u16_f16(vfp16out7x01234567)); c7 += 8;
       vfp16out7x01234567 = vfp16out7xGHIJKLMN;
       vst1q_u16(c6, vreinterpretq_u16_f16(vfp16out6x01234567)); c6 += 8;
       vfp16out6x01234567 = vfp16out6xGHIJKLMN;
       vst1q_u16(c5, vreinterpretq_u16_f16(vfp16out5x01234567)); c5 += 8;
       vfp16out5x01234567 = vfp16out5xGHIJKLMN;
       vst1q_u16(c4, vreinterpretq_u16_f16(vfp16out4x01234567)); c4 += 8;
       vfp16out4x01234567 = vfp16out4xGHIJKLMN;
       vst1q_u16(c3, vreinterpretq_u16_f16(vfp16out3x01234567)); c3 += 8;
       vfp16out3x01234567 = vfp16out3xGHIJKLMN;
       vst1q_u16(c2, vreinterpretq_u16_f16(vfp16out2x01234567)); c2 += 8;
       vfp16out2x01234567 = vfp16out2xGHIJKLMN;
       vst1q_u16(c1, vreinterpretq_u16_f16(vfp16out1x01234567)); c1 += 8;
       vfp16out1x01234567 = vfp16out1xGHIJKLMN;
       vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567)); c0 += 8;
       vfp16out0x01234567 = vfp16out0xGHIJKLMN;
       vst1q_u16(c7, vreinterpretq_u16_f16(vfp16out7x89ABCDEF)); c7 += 8;
       vfp16out7x89ABCDEF = vfp16out7xOPQRSTUV;
       vst1q_u16(c6, vreinterpretq_u16_f16(vfp16out6x89ABCDEF)); c6 += 8;
       vfp16out6x89ABCDEF = vfp16out6xOPQRSTUV;
       vst1q_u16(c5, vreinterpretq_u16_f16(vfp16out5x89ABCDEF)); c5 += 8;
       vfp16out5x89ABCDEF = vfp16out5xOPQRSTUV;
       vst1q_u16(c4, vreinterpretq_u16_f16(vfp16out4x89ABCDEF)); c4 += 8;
       vfp16out4x89ABCDEF = vfp16out4xOPQRSTUV;
       vst1q_u16(c3, vreinterpretq_u16_f16(vfp16out3x89ABCDEF)); c3 += 8;
       vfp16out3x89ABCDEF = vfp16out3xOPQRSTUV;
       vst1q_u16(c2, vreinterpretq_u16_f16(vfp16out2x89ABCDEF)); c2 += 8;
       vfp16out2x89ABCDEF = vfp16out2xOPQRSTUV;
       vst1q_u16(c1, vreinterpretq_u16_f16(vfp16out1x89ABCDEF)); c1 += 8;
       vfp16out1x89ABCDEF = vfp16out1xOPQRSTUV;
       vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x89ABCDEF)); c0 += 8;
       vfp16out0x89ABCDEF = vfp16out0xOPQRSTUV;
     }
     if (nc & 8) {
       vst1q_u16(c7, vreinterpretq_u16_f16(vfp16out7x01234567)); c7 += 8;
       vfp16out7x01234567 = vfp16out7x89ABCDEF;
       vst1q_u16(c6, vreinterpretq_u16_f16(vfp16out6x01234567)); c6 += 8;
       vfp16out6x01234567 = vfp16out6x89ABCDEF;
       vst1q_u16(c5, vreinterpretq_u16_f16(vfp16out5x01234567)); c5 += 8;
       vfp16out5x01234567 = vfp16out5x89ABCDEF;
       vst1q_u16(c4, vreinterpretq_u16_f16(vfp16out4x01234567)); c4 += 8;
       vfp16out4x01234567 = vfp16out4x89ABCDEF;
       vst1q_u16(c3, vreinterpretq_u16_f16(vfp16out3x01234567)); c3 += 8;
       vfp16out3x01234567 = vfp16out3x89ABCDEF;
       vst1q_u16(c2, vreinterpretq_u16_f16(vfp16out2x01234567)); c2 += 8;
       vfp16out2x01234567 = vfp16out2x89ABCDEF;
       vst1q_u16(c1, vreinterpretq_u16_f16(vfp16out1x01234567)); c1 += 8;
       vfp16out1x01234567 = vfp16out1x89ABCDEF;
       vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567)); c0 += 8;
       vfp16out0x01234567 = vfp16out0x89ABCDEF;
     }
     float16x4_t vfp16out7x0123 = vget_low_f16(vfp16out7x01234567);
     float16x4_t vfp16out6x0123 = vget_low_f16(vfp16out6x01234567);
     float16x4_t vfp16out5x0123 = vget_low_f16(vfp16out5x01234567);
     float16x4_t vfp16out4x0123 = vget_low_f16(vfp16out4x01234567);
     float16x4_t vfp16out3x0123 = vget_low_f16(vfp16out3x01234567);
     float16x4_t vfp16out2x0123 = vget_low_f16(vfp16out2x01234567);
     float16x4_t vfp16out1x0123 = vget_low_f16(vfp16out1x01234567);
     float16x4_t vfp16out0x0123 = vget_low_f16(vfp16out0x01234567);
     if (nc & 4) {
       vst1_u16(c7, vreinterpret_u16_f16(vfp16out7x0123)); c7 += 4;
       vst1_u16(c6, vreinterpret_u16_f16(vfp16out6x0123)); c6 += 4;
       vst1_u16(c5, vreinterpret_u16_f16(vfp16out5x0123)); c5 += 4;
       vst1_u16(c4, vreinterpret_u16_f16(vfp16out4x0123)); c4 += 4;
       vst1_u16(c3, vreinterpret_u16_f16(vfp16out3x0123)); c3 += 4;
       vst1_u16(c2, vreinterpret_u16_f16(vfp16out2x0123)); c2 += 4;
       vst1_u16(c1, vreinterpret_u16_f16(vfp16out1x0123)); c1 += 4;
       vst1_u16(c0, vreinterpret_u16_f16(vfp16out0x0123)); c0 += 4;
       vfp16out7x0123 = vget_high_f16(vfp16out7x01234567);
       vfp16out6x0123 = vget_high_f16(vfp16out6x01234567);
       vfp16out5x0123 = vget_high_f16(vfp16out5x01234567);
       vfp16out4x0123 = vget_high_f16(vfp16out4x01234567);
       vfp16out3x0123 = vget_high_f16(vfp16out3x01234567);
       vfp16out2x0123 = vget_high_f16(vfp16out2x01234567);
       vfp16out1x0123 = vget_high_f16(vfp16out1x01234567);
       vfp16out0x0123 = vget_high_f16(vfp16out0x01234567);
     }
     if (nc & 2) {
       vst1_lane_u32((void*) c7, vreinterpret_u32_f16(vfp16out7x0123), 0); c7 += 2;
       vst1_lane_u32((void*) c6, vreinterpret_u32_f16(vfp16out6x0123), 0); c6 += 2;
       vst1_lane_u32((void*) c5, vreinterpret_u32_f16(vfp16out5x0123), 0); c5 += 2;
       vst1_lane_u32((void*) c4, vreinterpret_u32_f16(vfp16out4x0123), 0); c4 += 2;
       vst1_lane_u32((void*) c3, vreinterpret_u32_f16(vfp16out3x0123), 0); c3 += 2;
       vst1_lane_u32((void*) c2, vreinterpret_u32_f16(vfp16out2x0123), 0); c2 += 2;
       vst1_lane_u32((void*) c1, vreinterpret_u32_f16(vfp16out1x0123), 0); c1 += 2;
       vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vfp16out0x0123), 0); c0 += 2;
       vfp16out7x0123 = vext_f16(vfp16out7x0123, vfp16out7x0123, 2);
       vfp16out6x0123 = vext_f16(vfp16out6x0123, vfp16out6x0123, 2);
       vfp16out5x0123 = vext_f16(vfp16out5x0123, vfp16out5x0123, 2);
       vfp16out4x0123 = vext_f16(vfp16out4x0123, vfp16out4x0123, 2);
       vfp16out3x0123 = vext_f16(vfp16out3x0123, vfp16out3x0123, 2);
       vfp16out2x0123 = vext_f16(vfp16out2x0123, vfp16out2x0123, 2);
       vfp16out1x0123 = vext_f16(vfp16out1x0123, vfp16out1x0123, 2);
       vfp16out0x0123 = vext_f16(vfp16out0x0123, vfp16out0x0123, 2);
     }
     if (nc & 1) {
       vst1_lane_u16(c7, vreinterpret_u16_f16(vfp16out7x0123), 0);
       vst1_lane_u16(c6, vreinterpret_u16_f16(vfp16out6x0123), 0);
       vst1_lane_u16(c5, vreinterpret_u16_f16(vfp16out5x0123), 0);
       vst1_lane_u16(c4, vreinterpret_u16_f16(vfp16out4x0123), 0);
       vst1_lane_u16(c3, vreinterpret_u16_f16(vfp16out3x0123), 0);
       vst1_lane_u16(c2, vreinterpret_u16_f16(vfp16out2x0123), 0);
       vst1_lane_u16(c1, vreinterpret_u16_f16(vfp16out1x0123), 0);
       vst1_lane_u16(c0, vreinterpret_u16_f16(vfp16out0x0123), 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}
