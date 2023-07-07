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


void xnn_qd8_f32_qs8w_gemm_minmax_ukernel_4x16c4__neondot(
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
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
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
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const int32x4_t vzp0 = vdupq_n_s32(quantization_params[0].zero_point);
  const int32x4_t vzp1 = vdupq_n_s32(quantization_params[1].zero_point);
  const int32x4_t vzp2 = vdupq_n_s32(quantization_params[2].zero_point);
  const int32x4_t vzp3 = vdupq_n_s32(quantization_params[3].zero_point);
  // Loop over groups of 16 columns.
  do {
    // Initialize accumulators with bias. 16 bias values are loaded from the
    // weight matrix, at the start of the group of 16 columns.
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
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
    vacc0x0123 = vmulq_s32(vacc0x0123, vzp0);
    vacc0x4567 = vmulq_s32(vacc0x4567, vzp0);
    vacc0x89AB = vmulq_s32(vacc0x89AB, vzp0);
    vacc0xCDEF = vmulq_s32(vacc0xCDEF, vzp0);
    vacc1x0123 = vmulq_s32(vacc1x0123, vzp1);
    vacc1x4567 = vmulq_s32(vacc1x4567, vzp1);
    vacc1x89AB = vmulq_s32(vacc1x89AB, vzp1);
    vacc1xCDEF = vmulq_s32(vacc1xCDEF, vzp1);
    vacc2x0123 = vmulq_s32(vacc2x0123, vzp2);
    vacc2x4567 = vmulq_s32(vacc2x4567, vzp2);
    vacc2x89AB = vmulq_s32(vacc2x89AB, vzp2);
    vacc2xCDEF = vmulq_s32(vacc2xCDEF, vzp2);
    vacc3x0123 = vmulq_s32(vacc3x0123, vzp3);
    vacc3x4567 = vmulq_s32(vacc3x4567, vzp3);
    vacc3x89AB = vmulq_s32(vacc3x89AB, vzp3);
    vacc3xCDEF = vmulq_s32(vacc3xCDEF, vzp3);

    // Inner accumulation loop along the 16 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 8 * sizeof(int8_t)) {
      // Load a 4x8 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;
      const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 8;
      const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 8;
      const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 8;

      // Load a 8x16 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 4x8 * 8x16 --> 4x16.
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

      k -= 8 * sizeof(int8_t);
    }
    // Handle up to 4 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 4x4 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 4;
      const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 4;
      const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 4;
      const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 4;

      // Load a 4x16 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x89AB = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123xCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 4x4 * 4x16 --> 4x16.
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
    }

    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vout0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vout0xCDEF = vcvtq_f32_s32(vacc0xCDEF);
    float32x4_t vout1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vout1x4567 = vcvtq_f32_s32(vacc1x4567);
    float32x4_t vout1x89AB = vcvtq_f32_s32(vacc1x89AB);
    float32x4_t vout1xCDEF = vcvtq_f32_s32(vacc1xCDEF);
    float32x4_t vout2x0123 = vcvtq_f32_s32(vacc2x0123);
    float32x4_t vout2x4567 = vcvtq_f32_s32(vacc2x4567);
    float32x4_t vout2x89AB = vcvtq_f32_s32(vacc2x89AB);
    float32x4_t vout2xCDEF = vcvtq_f32_s32(vacc2xCDEF);
    float32x4_t vout3x0123 = vcvtq_f32_s32(vacc3x0123);
    float32x4_t vout3x4567 = vcvtq_f32_s32(vacc3x4567);
    float32x4_t vout3x89AB = vcvtq_f32_s32(vacc3x89AB);
    float32x4_t vout3xCDEF = vcvtq_f32_s32(vacc3xCDEF);
    const float32x4_t vscale0 = vdupq_n_f32(quantization_params[0].scale);
    vout0x0123 = vmulq_f32(vout0x0123, vscale0);
    vout0x4567 = vmulq_f32(vout0x4567, vscale0);
    vout0x89AB = vmulq_f32(vout0x89AB, vscale0);
    vout0xCDEF = vmulq_f32(vout0xCDEF, vscale0);
    const float32x4_t vscale1 = vdupq_n_f32(quantization_params[1].scale);
    vout1x0123 = vmulq_f32(vout1x0123, vscale1);
    vout1x4567 = vmulq_f32(vout1x4567, vscale1);
    vout1x89AB = vmulq_f32(vout1x89AB, vscale1);
    vout1xCDEF = vmulq_f32(vout1xCDEF, vscale1);
    const float32x4_t vscale2 = vdupq_n_f32(quantization_params[2].scale);
    vout2x0123 = vmulq_f32(vout2x0123, vscale2);
    vout2x4567 = vmulq_f32(vout2x4567, vscale2);
    vout2x89AB = vmulq_f32(vout2x89AB, vscale2);
    vout2xCDEF = vmulq_f32(vout2xCDEF, vscale2);
    const float32x4_t vscale3 = vdupq_n_f32(quantization_params[3].scale);
    vout3x0123 = vmulq_f32(vout3x0123, vscale3);
    vout3x4567 = vmulq_f32(vout3x4567, vscale3);
    vout3x89AB = vmulq_f32(vout3x89AB, vscale3);
    vout3xCDEF = vmulq_f32(vout3xCDEF, vscale3);

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
    vout0x0123 = vaddq_f32(vout0x0123, vbias0123);
    vout0x4567 = vaddq_f32(vout0x4567, vbias4567);
    vout0x89AB = vaddq_f32(vout0x89AB, vbias89AB);
    vout0xCDEF = vaddq_f32(vout0xCDEF, vbiasCDEF);
    vout1x0123 = vaddq_f32(vout1x0123, vbias0123);
    vout1x4567 = vaddq_f32(vout1x4567, vbias4567);
    vout1x89AB = vaddq_f32(vout1x89AB, vbias89AB);
    vout1xCDEF = vaddq_f32(vout1xCDEF, vbiasCDEF);
    vout2x0123 = vaddq_f32(vout2x0123, vbias0123);
    vout2x4567 = vaddq_f32(vout2x4567, vbias4567);
    vout2x89AB = vaddq_f32(vout2x89AB, vbias89AB);
    vout2xCDEF = vaddq_f32(vout2xCDEF, vbiasCDEF);
    vout3x0123 = vaddq_f32(vout3x0123, vbias0123);
    vout3x4567 = vaddq_f32(vout3x4567, vbias4567);
    vout3x89AB = vaddq_f32(vout3x89AB, vbias89AB);
    vout3xCDEF = vaddq_f32(vout3xCDEF, vbiasCDEF);
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
    vout1x0123 = vmaxq_f32(vout1x0123, vmin);
    vout1x0123 = vminq_f32(vout1x0123, vmax);
    vout1x4567 = vmaxq_f32(vout1x4567, vmin);
    vout1x4567 = vminq_f32(vout1x4567, vmax);
    vout1x89AB = vmaxq_f32(vout1x89AB, vmin);
    vout1x89AB = vminq_f32(vout1x89AB, vmax);
    vout1xCDEF = vmaxq_f32(vout1xCDEF, vmin);
    vout1xCDEF = vminq_f32(vout1xCDEF, vmax);
    vout2x0123 = vmaxq_f32(vout2x0123, vmin);
    vout2x0123 = vminq_f32(vout2x0123, vmax);
    vout2x4567 = vmaxq_f32(vout2x4567, vmin);
    vout2x4567 = vminq_f32(vout2x4567, vmax);
    vout2x89AB = vmaxq_f32(vout2x89AB, vmin);
    vout2x89AB = vminq_f32(vout2x89AB, vmax);
    vout2xCDEF = vmaxq_f32(vout2xCDEF, vmin);
    vout2xCDEF = vminq_f32(vout2xCDEF, vmax);
    vout3x0123 = vmaxq_f32(vout3x0123, vmin);
    vout3x0123 = vminq_f32(vout3x0123, vmax);
    vout3x4567 = vmaxq_f32(vout3x4567, vmin);
    vout3x4567 = vminq_f32(vout3x4567, vmax);
    vout3x89AB = vmaxq_f32(vout3x89AB, vmin);
    vout3x89AB = vminq_f32(vout3x89AB, vmax);
    vout3xCDEF = vmaxq_f32(vout3xCDEF, vmin);
    vout3xCDEF = vminq_f32(vout3xCDEF, vmax);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(&c3[0], vout3x0123);
      vst1q_f32(&c3[4], vout3x4567);
      vst1q_f32(&c3[8], vout3x89AB);
      vst1q_f32(&c3[12], vout3xCDEF);
      vst1q_f32(&c2[0], vout2x0123);
      vst1q_f32(&c2[4], vout2x4567);
      vst1q_f32(&c2[8], vout2x89AB);
      vst1q_f32(&c2[12], vout2xCDEF);
      vst1q_f32(&c1[0], vout1x0123);
      vst1q_f32(&c1[4], vout1x4567);
      vst1q_f32(&c1[8], vout1x89AB);
      vst1q_f32(&c1[12], vout1xCDEF);
      vst1q_f32(&c0[0], vout0x0123);
      vst1q_f32(&c0[4], vout0x4567);
      vst1q_f32(&c0[8], vout0x89AB);
      vst1q_f32(&c0[12], vout0xCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      nc -= 16;
    } else {
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
