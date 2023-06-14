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

#include <xnnpack/gemm.h>
#include <xnnpack/math.h>


void xnn_qd8_f32_qs8w_gemm_minmax_ukernel_1x16c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
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

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;

  const int32x4_t vzp0 = vdupq_n_s32(quantization_params[0].zero_point);
  // Loop over groups of 16 columns.
  do {
    // Initialize accumulators with bias. 16 bias values are loaded from the
    // weight matrix, at the start of the group of 16 columns.
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
    vacc0x0123 = vmulq_s32(vacc0x0123, vzp0);
    vacc0x4567 = vmulq_s32(vacc0x4567, vzp0);
    vacc0x89AB = vmulq_s32(vacc0x89AB, vzp0);
    vacc0xCDEF = vmulq_s32(vacc0xCDEF, vzp0);

    // Inner accumulation loop along the 16 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 8 * sizeof(int8_t)) {
      // Load a 1x8 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;

      // Load a 8x16 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 1x8 * 8x16 --> 1x16.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb4567x89AB, va0x01234567, 1);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb4567xCDEF, va0x01234567, 1);

      k -= 8 * sizeof(int8_t);
    }
    // Handle up to 4 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 1x4 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 4;

      // Load a 4x16 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 1x4 * 4x16 --> 1x16.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc0x89AB = vdotq_lane_s32(vacc0x89AB, vb0123x89AB, va0x01234567, 0);
      vacc0xCDEF = vdotq_lane_s32(vacc0xCDEF, vb0123xCDEF, va0x01234567, 0);
    }

    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vout0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vout0xCDEF = vcvtq_f32_s32(vacc0xCDEF);
    const float32x4_t vscale0 = vdupq_n_f32(quantization_params[0].scale);
    vout0x0123 = vmulq_f32(vout0x0123, vscale0);
    vout0x4567 = vmulq_f32(vout0x4567, vscale0);
    vout0x89AB = vmulq_f32(vout0x89AB, vscale0);
    vout0xCDEF = vmulq_f32(vout0xCDEF, vscale0);

    const float32x4_t vbias0 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vbias4 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vbias8 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vbias12 = vld1q_f32(w); w = (const float*) w + 4;
    vout0x0123 = vaddq_f32(vout0x0123, vbias0);
    vout0x4567 = vaddq_f32(vout0x4567, vbias4);
    vout0x89AB = vaddq_f32(vout0x89AB, vbias8);
    vout0xCDEF = vaddq_f32(vout0xCDEF, vbias12);
    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, vmin);
    vout0x0123 = vminq_f32(vout0x0123, vmax);
    vout0x4567 = vmaxq_f32(vout0x4567, vmin);
    vout0x4567 = vminq_f32(vout0x4567, vmax);
    vout0x89AB = vmaxq_f32(vout0x89AB, vmin);
    vout0x89AB = vminq_f32(vout0x89AB, vmax);
    vout0xCDEF = vmaxq_f32(vout0xCDEF, vmin);
    vout0xCDEF = vminq_f32(vout0xCDEF, vmax);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(&c0[0], vout0x0123);
      vst1q_f32(&c0[4], vout0x4567);
      vst1q_f32(&c0[8], vout0x89AB);
      vst1q_f32(&c0[12], vout0xCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 16;
    } else {
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
