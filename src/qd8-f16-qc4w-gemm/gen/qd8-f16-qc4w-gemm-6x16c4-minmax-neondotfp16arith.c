// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c4-neondot.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/gemm.h"
#include "xnnpack/math.h"


void xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const int8_t* a0 = a;
  uint16_t* c0 = (uint16_t*) c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  uint16_t* c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const int8_t* a4 = (const int8_t*) ((uintptr_t) a3 + a_stride);
  uint16_t* c4 = (uint16_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const int8_t* a5 = (const int8_t*) ((uintptr_t) a4 + a_stride);
  uint16_t* c5 = (uint16_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  const int8x16_t vmask = vmovq_n_s8(INT8_C(0xF0));
  // Loop over groups of 16 columns.
  do {
    // Initialize accumulators with bias. 16 bias values are loaded from the
    // weight matrix, at the start of the group of 16 columns.
    const int32x4_t vinput_zero_point01 = vld1q_s32(&quantization_params[0].zero_point);
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x0123 = vmulq_lane_s32(vksum0123, vget_low_s32(vinput_zero_point01), 0);
    int32x4_t vacc1x0123 = vmulq_lane_s32(vksum0123, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vmulq_lane_s32(vksum4567, vget_low_s32(vinput_zero_point01), 0);
    int32x4_t vacc1x4567 = vmulq_lane_s32(vksum4567, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksum89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x89AB = vmulq_lane_s32(vksum89AB, vget_low_s32(vinput_zero_point01), 0);
    int32x4_t vacc1x89AB = vmulq_lane_s32(vksum89AB, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksumCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xCDEF = vmulq_lane_s32(vksumCDEF, vget_low_s32(vinput_zero_point01), 0);
    int32x4_t vacc1xCDEF = vmulq_lane_s32(vksumCDEF, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vinput_zero_point23 = vld1q_s32(&quantization_params[2].zero_point);
    int32x4_t vacc2x0123 = vmulq_lane_s32(vksum0123, vget_low_s32(vinput_zero_point23), 0);
    int32x4_t vacc3x0123 = vmulq_lane_s32(vksum0123, vget_high_s32(vinput_zero_point23), 0);
    int32x4_t vacc2x4567 = vmulq_lane_s32(vksum4567, vget_low_s32(vinput_zero_point23), 0);
    int32x4_t vacc3x4567 = vmulq_lane_s32(vksum4567, vget_high_s32(vinput_zero_point23), 0);
    int32x4_t vacc2x89AB = vmulq_lane_s32(vksum89AB, vget_low_s32(vinput_zero_point23), 0);
    int32x4_t vacc3x89AB = vmulq_lane_s32(vksum89AB, vget_high_s32(vinput_zero_point23), 0);
    int32x4_t vacc2xCDEF = vmulq_lane_s32(vksumCDEF, vget_low_s32(vinput_zero_point23), 0);
    int32x4_t vacc3xCDEF = vmulq_lane_s32(vksumCDEF, vget_high_s32(vinput_zero_point23), 0);
    const int32x4_t vinput_zero_point45 = vld1q_s32(&quantization_params[4].zero_point);
    int32x4_t vacc4x0123 = vmulq_lane_s32(vksum0123, vget_low_s32(vinput_zero_point45), 0);
    int32x4_t vacc5x0123 = vmulq_lane_s32(vksum0123, vget_high_s32(vinput_zero_point45), 0);
    int32x4_t vacc4x4567 = vmulq_lane_s32(vksum4567, vget_low_s32(vinput_zero_point45), 0);
    int32x4_t vacc5x4567 = vmulq_lane_s32(vksum4567, vget_high_s32(vinput_zero_point45), 0);
    int32x4_t vacc4x89AB = vmulq_lane_s32(vksum89AB, vget_low_s32(vinput_zero_point45), 0);
    int32x4_t vacc5x89AB = vmulq_lane_s32(vksum89AB, vget_high_s32(vinput_zero_point45), 0);
    int32x4_t vacc4xCDEF = vmulq_lane_s32(vksumCDEF, vget_low_s32(vinput_zero_point45), 0);
    int32x4_t vacc5xCDEF = vmulq_lane_s32(vksumCDEF, vget_high_s32(vinput_zero_point45), 0);

    // Inner accumulation loop along the 16 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 8 * sizeof(int8_t)) {
      // Load a 6x8 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;
      const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 8;
      const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 8;
      const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 8;
      const int8x8_t va4x01234567 = vld1_s8(a4); a4 += 8;
      const int8x8_t va5x01234567 = vld1_s8(a5); a5 += 8;

      // Load a 8x16 block of weights.
      const int8x16_t vb01234567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01234567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01234567x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01234567xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x0123 = vshlq_n_s8(vb01234567x0123, 4);
      const int8x16_t vb0123x4567 = vshlq_n_s8(vb01234567x4567, 4);
      const int8x16_t vb0123x89AB = vshlq_n_s8(vb01234567x89AB, 4);
      const int8x16_t vb0123xCDEF = vshlq_n_s8(vb01234567xCDEF, 4);
      const int8x16_t vb4567x0123 = vandq_s8(vb01234567x0123, vmask);
      const int8x16_t vb4567x4567 = vandq_s8(vb01234567x4567, vmask);
      const int8x16_t vb4567x89AB = vandq_s8(vb01234567x89AB, vmask);
      const int8x16_t vb4567xCDEF = vandq_s8(vb01234567xCDEF, vmask);

      // Multiply-accumulate: 6x8 * 8x16 --> 6x16.
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

      k -= 8 * sizeof(int8_t);
    }
    // Handle up to 4 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 6x4 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 4;
      const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 4;
      const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 4;
      const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 4;
      const int8x8_t va4x01234567 = vld1_s8(a4); a4 += 4;
      const int8x8_t va5x01234567 = vld1_s8(a5); a5 += 4;

      // Load a 4x16 block of weights.
      const int8x16_t vb01234567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01234567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01234567x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01234567xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x0123 = vshlq_n_s8(vb01234567x0123, 4);
      const int8x16_t vb0123x4567 = vshlq_n_s8(vb01234567x4567, 4);
      const int8x16_t vb0123x89AB = vshlq_n_s8(vb01234567x89AB, 4);
      const int8x16_t vb0123xCDEF = vshlq_n_s8(vb01234567xCDEF, 4);

      // Multiply-accumulate: 6x4 * 4x16 --> 6x16.
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
    }

    float32x4_t vout0x0123 = vcvtq_n_f32_s32(vacc0x0123, 4);
    float32x4_t vout0x4567 = vcvtq_n_f32_s32(vacc0x4567, 4);
    float32x4_t vout0x89AB = vcvtq_n_f32_s32(vacc0x89AB, 4);
    float32x4_t vout0xCDEF = vcvtq_n_f32_s32(vacc0xCDEF, 4);
    float32x4_t vout1x0123 = vcvtq_n_f32_s32(vacc1x0123, 4);
    float32x4_t vout1x4567 = vcvtq_n_f32_s32(vacc1x4567, 4);
    float32x4_t vout1x89AB = vcvtq_n_f32_s32(vacc1x89AB, 4);
    float32x4_t vout1xCDEF = vcvtq_n_f32_s32(vacc1xCDEF, 4);
    float32x4_t vout2x0123 = vcvtq_n_f32_s32(vacc2x0123, 4);
    float32x4_t vout2x4567 = vcvtq_n_f32_s32(vacc2x4567, 4);
    float32x4_t vout2x89AB = vcvtq_n_f32_s32(vacc2x89AB, 4);
    float32x4_t vout2xCDEF = vcvtq_n_f32_s32(vacc2xCDEF, 4);
    float32x4_t vout3x0123 = vcvtq_n_f32_s32(vacc3x0123, 4);
    float32x4_t vout3x4567 = vcvtq_n_f32_s32(vacc3x4567, 4);
    float32x4_t vout3x89AB = vcvtq_n_f32_s32(vacc3x89AB, 4);
    float32x4_t vout3xCDEF = vcvtq_n_f32_s32(vacc3xCDEF, 4);
    float32x4_t vout4x0123 = vcvtq_n_f32_s32(vacc4x0123, 4);
    float32x4_t vout4x4567 = vcvtq_n_f32_s32(vacc4x4567, 4);
    float32x4_t vout4x89AB = vcvtq_n_f32_s32(vacc4x89AB, 4);
    float32x4_t vout4xCDEF = vcvtq_n_f32_s32(vacc4xCDEF, 4);
    float32x4_t vout5x0123 = vcvtq_n_f32_s32(vacc5x0123, 4);
    float32x4_t vout5x4567 = vcvtq_n_f32_s32(vacc5x4567, 4);
    float32x4_t vout5x89AB = vcvtq_n_f32_s32(vacc5x89AB, 4);
    float32x4_t vout5xCDEF = vcvtq_n_f32_s32(vacc5xCDEF, 4);
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
    const float32x4_t vinput_scale45 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[4].zero_point));
    vout4x0123 = vmulq_lane_f32(vout4x0123, vget_low_f32(vinput_scale45), 1);
    vout5x0123 = vmulq_lane_f32(vout5x0123, vget_high_f32(vinput_scale45), 1);
    vout4x4567 = vmulq_lane_f32(vout4x4567, vget_low_f32(vinput_scale45), 1);
    vout5x4567 = vmulq_lane_f32(vout5x4567, vget_high_f32(vinput_scale45), 1);
    vout4x89AB = vmulq_lane_f32(vout4x89AB, vget_low_f32(vinput_scale45), 1);
    vout5x89AB = vmulq_lane_f32(vout5x89AB, vget_high_f32(vinput_scale45), 1);
    vout4xCDEF = vmulq_lane_f32(vout4xCDEF, vget_low_f32(vinput_scale45), 1);
    vout5xCDEF = vmulq_lane_f32(vout5xCDEF, vget_high_f32(vinput_scale45), 1);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale89AB = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleCDEF = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vfmaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vfmaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vfmaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
      vout4x0123 = vfmaq_f32(vbias0123, vout4x0123, vfilter_output_scale0123);
      vout5x0123 = vfmaq_f32(vbias0123, vout5x0123, vfilter_output_scale0123);
    #else
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vmlaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vmlaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vmlaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
      vout4x0123 = vmlaq_f32(vbias0123, vout4x0123, vfilter_output_scale0123);
      vout5x0123 = vmlaq_f32(vbias0123, vout5x0123, vfilter_output_scale0123);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vfmaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vfmaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vfmaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
      vout4x4567 = vfmaq_f32(vbias4567, vout4x4567, vfilter_output_scale4567);
      vout5x4567 = vfmaq_f32(vbias4567, vout5x4567, vfilter_output_scale4567);
    #else
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vmlaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vmlaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vmlaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
      vout4x4567 = vmlaq_f32(vbias4567, vout4x4567, vfilter_output_scale4567);
      vout5x4567 = vmlaq_f32(vbias4567, vout5x4567, vfilter_output_scale4567);
    #endif
    const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x89AB = vfmaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
      vout1x89AB = vfmaq_f32(vbias89AB, vout1x89AB, vfilter_output_scale89AB);
      vout2x89AB = vfmaq_f32(vbias89AB, vout2x89AB, vfilter_output_scale89AB);
      vout3x89AB = vfmaq_f32(vbias89AB, vout3x89AB, vfilter_output_scale89AB);
      vout4x89AB = vfmaq_f32(vbias89AB, vout4x89AB, vfilter_output_scale89AB);
      vout5x89AB = vfmaq_f32(vbias89AB, vout5x89AB, vfilter_output_scale89AB);
    #else
      vout0x89AB = vmlaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
      vout1x89AB = vmlaq_f32(vbias89AB, vout1x89AB, vfilter_output_scale89AB);
      vout2x89AB = vmlaq_f32(vbias89AB, vout2x89AB, vfilter_output_scale89AB);
      vout3x89AB = vmlaq_f32(vbias89AB, vout3x89AB, vfilter_output_scale89AB);
      vout4x89AB = vmlaq_f32(vbias89AB, vout4x89AB, vfilter_output_scale89AB);
      vout5x89AB = vmlaq_f32(vbias89AB, vout5x89AB, vfilter_output_scale89AB);
    #endif
    const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xCDEF = vfmaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
      vout1xCDEF = vfmaq_f32(vbiasCDEF, vout1xCDEF, vfilter_output_scaleCDEF);
      vout2xCDEF = vfmaq_f32(vbiasCDEF, vout2xCDEF, vfilter_output_scaleCDEF);
      vout3xCDEF = vfmaq_f32(vbiasCDEF, vout3xCDEF, vfilter_output_scaleCDEF);
      vout4xCDEF = vfmaq_f32(vbiasCDEF, vout4xCDEF, vfilter_output_scaleCDEF);
      vout5xCDEF = vfmaq_f32(vbiasCDEF, vout5xCDEF, vfilter_output_scaleCDEF);
    #else
      vout0xCDEF = vmlaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
      vout1xCDEF = vmlaq_f32(vbiasCDEF, vout1xCDEF, vfilter_output_scaleCDEF);
      vout2xCDEF = vmlaq_f32(vbiasCDEF, vout2xCDEF, vfilter_output_scaleCDEF);
      vout3xCDEF = vmlaq_f32(vbiasCDEF, vout3xCDEF, vfilter_output_scaleCDEF);
      vout4xCDEF = vmlaq_f32(vbiasCDEF, vout4xCDEF, vfilter_output_scaleCDEF);
      vout5xCDEF = vmlaq_f32(vbiasCDEF, vout5xCDEF, vfilter_output_scaleCDEF);
    #endif

    float16x8_t vfp16out0x01234567 = vcombine_f16(vcvt_f16_f32(vout0x0123), vcvt_f16_f32(vout0x4567));
    float16x8_t vfp16out0x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout0x89AB), vcvt_f16_f32(vout0xCDEF));
    float16x8_t vfp16out1x01234567 = vcombine_f16(vcvt_f16_f32(vout1x0123), vcvt_f16_f32(vout1x4567));
    float16x8_t vfp16out1x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout1x89AB), vcvt_f16_f32(vout1xCDEF));
    float16x8_t vfp16out2x01234567 = vcombine_f16(vcvt_f16_f32(vout2x0123), vcvt_f16_f32(vout2x4567));
    float16x8_t vfp16out2x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout2x89AB), vcvt_f16_f32(vout2xCDEF));
    float16x8_t vfp16out3x01234567 = vcombine_f16(vcvt_f16_f32(vout3x0123), vcvt_f16_f32(vout3x4567));
    float16x8_t vfp16out3x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout3x89AB), vcvt_f16_f32(vout3xCDEF));
    float16x8_t vfp16out4x01234567 = vcombine_f16(vcvt_f16_f32(vout4x0123), vcvt_f16_f32(vout4x4567));
    float16x8_t vfp16out4x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout4x89AB), vcvt_f16_f32(vout4xCDEF));
    float16x8_t vfp16out5x01234567 = vcombine_f16(vcvt_f16_f32(vout5x0123), vcvt_f16_f32(vout5x4567));
    float16x8_t vfp16out5x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout5x89AB), vcvt_f16_f32(vout5xCDEF));

    const float16x8_t voutput_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
    vfp16out0x01234567 = vmaxq_f16(vfp16out0x01234567, voutput_min);
    vfp16out0x89ABCDEF = vmaxq_f16(vfp16out0x89ABCDEF, voutput_min);
    vfp16out1x01234567 = vmaxq_f16(vfp16out1x01234567, voutput_min);
    vfp16out1x89ABCDEF = vmaxq_f16(vfp16out1x89ABCDEF, voutput_min);
    vfp16out2x01234567 = vmaxq_f16(vfp16out2x01234567, voutput_min);
    vfp16out2x89ABCDEF = vmaxq_f16(vfp16out2x89ABCDEF, voutput_min);
    vfp16out3x01234567 = vmaxq_f16(vfp16out3x01234567, voutput_min);
    vfp16out3x89ABCDEF = vmaxq_f16(vfp16out3x89ABCDEF, voutput_min);
    vfp16out4x01234567 = vmaxq_f16(vfp16out4x01234567, voutput_min);
    vfp16out4x89ABCDEF = vmaxq_f16(vfp16out4x89ABCDEF, voutput_min);
    vfp16out5x01234567 = vmaxq_f16(vfp16out5x01234567, voutput_min);
    vfp16out5x89ABCDEF = vmaxq_f16(vfp16out5x89ABCDEF, voutput_min);
    const float16x8_t voutput_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
    vfp16out0x01234567 = vminq_f16(vfp16out0x01234567, voutput_max);
    vfp16out0x89ABCDEF = vminq_f16(vfp16out0x89ABCDEF, voutput_max);
    vfp16out1x01234567 = vminq_f16(vfp16out1x01234567, voutput_max);
    vfp16out1x89ABCDEF = vminq_f16(vfp16out1x89ABCDEF, voutput_max);
    vfp16out2x01234567 = vminq_f16(vfp16out2x01234567, voutput_max);
    vfp16out2x89ABCDEF = vminq_f16(vfp16out2x89ABCDEF, voutput_max);
    vfp16out3x01234567 = vminq_f16(vfp16out3x01234567, voutput_max);
    vfp16out3x89ABCDEF = vminq_f16(vfp16out3x89ABCDEF, voutput_max);
    vfp16out4x01234567 = vminq_f16(vfp16out4x01234567, voutput_max);
    vfp16out4x89ABCDEF = vminq_f16(vfp16out4x89ABCDEF, voutput_max);
    vfp16out5x01234567 = vminq_f16(vfp16out5x01234567, voutput_max);
    vfp16out5x89ABCDEF = vminq_f16(vfp16out5x89ABCDEF, voutput_max);
    if XNN_LIKELY(nc >= 16) {
      vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567));
      vst1q_u16(c0 + 8, vreinterpretq_u16_f16(vfp16out0x89ABCDEF));
      vst1q_u16(c1, vreinterpretq_u16_f16(vfp16out1x01234567));
      vst1q_u16(c1 + 8, vreinterpretq_u16_f16(vfp16out1x89ABCDEF));
      vst1q_u16(c2, vreinterpretq_u16_f16(vfp16out2x01234567));
      vst1q_u16(c2 + 8, vreinterpretq_u16_f16(vfp16out2x89ABCDEF));
      vst1q_u16(c3, vreinterpretq_u16_f16(vfp16out3x01234567));
      vst1q_u16(c3 + 8, vreinterpretq_u16_f16(vfp16out3x89ABCDEF));
      vst1q_u16(c4, vreinterpretq_u16_f16(vfp16out4x01234567));
      vst1q_u16(c4 + 8, vreinterpretq_u16_f16(vfp16out4x89ABCDEF));
      vst1q_u16(c5, vreinterpretq_u16_f16(vfp16out5x01234567));
      vst1q_u16(c5 + 8, vreinterpretq_u16_f16(vfp16out5x89ABCDEF));

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);
      a4 = (const int8_t*) ((uintptr_t) a4 - kc);
      a5 = (const int8_t*) ((uintptr_t) a5 - kc);

      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      c4 = (uint16_t*) ((uintptr_t) c4 + cn_stride);
      c5 = (uint16_t*) ((uintptr_t) c5 + cn_stride);

      nc -= 16;
    } else {
     if (nc & 8) {
       vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567)); c0 += 8;
       vfp16out0x01234567 = vfp16out0x89ABCDEF;
       vst1q_u16(c1, vreinterpretq_u16_f16(vfp16out1x01234567)); c1 += 8;
       vfp16out1x01234567 = vfp16out1x89ABCDEF;
       vst1q_u16(c2, vreinterpretq_u16_f16(vfp16out2x01234567)); c2 += 8;
       vfp16out2x01234567 = vfp16out2x89ABCDEF;
       vst1q_u16(c3, vreinterpretq_u16_f16(vfp16out3x01234567)); c3 += 8;
       vfp16out3x01234567 = vfp16out3x89ABCDEF;
       vst1q_u16(c4, vreinterpretq_u16_f16(vfp16out4x01234567)); c4 += 8;
       vfp16out4x01234567 = vfp16out4x89ABCDEF;
       vst1q_u16(c5, vreinterpretq_u16_f16(vfp16out5x01234567)); c5 += 8;
       vfp16out5x01234567 = vfp16out5x89ABCDEF;
     }
     float16x4_t vfp16out0x0123 = vget_low_f16(vfp16out0x01234567);
     float16x4_t vfp16out1x0123 = vget_low_f16(vfp16out1x01234567);
     float16x4_t vfp16out2x0123 = vget_low_f16(vfp16out2x01234567);
     float16x4_t vfp16out3x0123 = vget_low_f16(vfp16out3x01234567);
     float16x4_t vfp16out4x0123 = vget_low_f16(vfp16out4x01234567);
     float16x4_t vfp16out5x0123 = vget_low_f16(vfp16out5x01234567);
     if (nc & 4) {
       vst1_u16(c0, vreinterpret_u16_f16(vfp16out0x0123)); c0 += 4;
       vst1_u16(c1, vreinterpret_u16_f16(vfp16out1x0123)); c1 += 4;
       vst1_u16(c2, vreinterpret_u16_f16(vfp16out2x0123)); c2 += 4;
       vst1_u16(c3, vreinterpret_u16_f16(vfp16out3x0123)); c3 += 4;
       vst1_u16(c4, vreinterpret_u16_f16(vfp16out4x0123)); c4 += 4;
       vst1_u16(c5, vreinterpret_u16_f16(vfp16out5x0123)); c5 += 4;
       vfp16out0x0123 = vget_high_f16(vfp16out0x01234567);
       vfp16out1x0123 = vget_high_f16(vfp16out1x01234567);
       vfp16out2x0123 = vget_high_f16(vfp16out2x01234567);
       vfp16out3x0123 = vget_high_f16(vfp16out3x01234567);
       vfp16out4x0123 = vget_high_f16(vfp16out4x01234567);
       vfp16out5x0123 = vget_high_f16(vfp16out5x01234567);
     }
     if (nc & 2) {
       vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vfp16out0x0123), 0); c0 += 2;
       vst1_lane_u32((void*) c1, vreinterpret_u32_f16(vfp16out1x0123), 0); c1 += 2;
       vst1_lane_u32((void*) c2, vreinterpret_u32_f16(vfp16out2x0123), 0); c2 += 2;
       vst1_lane_u32((void*) c3, vreinterpret_u32_f16(vfp16out3x0123), 0); c3 += 2;
       vst1_lane_u32((void*) c4, vreinterpret_u32_f16(vfp16out4x0123), 0); c4 += 2;
       vst1_lane_u32((void*) c5, vreinterpret_u32_f16(vfp16out5x0123), 0); c5 += 2;
       vfp16out0x0123 = vext_f16(vfp16out0x0123, vfp16out0x0123, 2);
       vfp16out1x0123 = vext_f16(vfp16out1x0123, vfp16out1x0123, 2);
       vfp16out2x0123 = vext_f16(vfp16out2x0123, vfp16out2x0123, 2);
       vfp16out3x0123 = vext_f16(vfp16out3x0123, vfp16out3x0123, 2);
       vfp16out4x0123 = vext_f16(vfp16out4x0123, vfp16out4x0123, 2);
       vfp16out5x0123 = vext_f16(vfp16out5x0123, vfp16out5x0123, 2);
     }
     if (nc & 1) {
       vst1_lane_u16(c0, vreinterpret_u16_f16(vfp16out0x0123), 0);
       vst1_lane_u16(c1, vreinterpret_u16_f16(vfp16out1x0123), 0);
       vst1_lane_u16(c2, vreinterpret_u16_f16(vfp16out2x0123), 0);
       vst1_lane_u16(c3, vreinterpret_u16_f16(vfp16out3x0123), 0);
       vst1_lane_u16(c4, vreinterpret_u16_f16(vfp16out4x0123), 0);
       vst1_lane_u16(c5, vreinterpret_u16_f16(vfp16out5x0123), 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}
