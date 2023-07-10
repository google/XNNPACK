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


void xnn_qd8_f32_qs8w_gemm_minmax_ukernel_4x8c4__neondot(
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
  // Loop over groups of 8 columns.
  do {
    // Initialize accumulators with bias. 8 bias values are loaded from the
    // weight matrix, at the start of the group of 8 columns.
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;
    int32x4_t vacc2x0123 = vacc0x0123;
    int32x4_t vacc2x4567 = vacc0x4567;
    int32x4_t vacc3x0123 = vacc0x0123;
    int32x4_t vacc3x4567 = vacc0x4567;
    vacc0x0123 = vmulq_s32(vacc0x0123, vzp0);
    vacc0x4567 = vmulq_s32(vacc0x4567, vzp0);
    vacc1x0123 = vmulq_s32(vacc1x0123, vzp1);
    vacc1x4567 = vmulq_s32(vacc1x4567, vzp1);
    vacc2x0123 = vmulq_s32(vacc2x0123, vzp2);
    vacc2x4567 = vmulq_s32(vacc2x4567, vzp2);
    vacc3x0123 = vmulq_s32(vacc3x0123, vzp3);
    vacc3x4567 = vmulq_s32(vacc3x4567, vzp3);

    // Inner accumulation loop along the 8 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 8 * sizeof(int8_t)) {
      // Load a 4x8 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;
      const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 8;
      const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 8;
      const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 8;

      // Load a 8x8 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 4x8 * 8x8 --> 4x8.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
      vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
      vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
      vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
      vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);
      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb4567x0123, va1x01234567, 1);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb4567x4567, va1x01234567, 1);
      vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb4567x0123, va2x01234567, 1);
      vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb4567x4567, va2x01234567, 1);
      vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb4567x0123, va3x01234567, 1);
      vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb4567x4567, va3x01234567, 1);

      k -= 8 * sizeof(int8_t);
    }
    // Handle up to 4 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 4x4 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 4;
      const int8x8_t va1x01234567 = vld1_s8(a1); a1 += 4;
      const int8x8_t va2x01234567 = vld1_s8(a2); a2 += 4;
      const int8x8_t va3x01234567 = vld1_s8(a3); a3 += 4;

      // Load a 4x8 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 4x4 * 4x8 --> 4x8.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc1x0123 = vdotq_lane_s32(vacc1x0123, vb0123x0123, va1x01234567, 0);
      vacc1x4567 = vdotq_lane_s32(vacc1x4567, vb0123x4567, va1x01234567, 0);
      vacc2x0123 = vdotq_lane_s32(vacc2x0123, vb0123x0123, va2x01234567, 0);
      vacc2x4567 = vdotq_lane_s32(vacc2x4567, vb0123x4567, va2x01234567, 0);
      vacc3x0123 = vdotq_lane_s32(vacc3x0123, vb0123x0123, va3x01234567, 0);
      vacc3x4567 = vdotq_lane_s32(vacc3x4567, vb0123x4567, va3x01234567, 0);
    }

    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vout1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vout1x4567 = vcvtq_f32_s32(vacc1x4567);
    float32x4_t vout2x0123 = vcvtq_f32_s32(vacc2x0123);
    float32x4_t vout2x4567 = vcvtq_f32_s32(vacc2x4567);
    float32x4_t vout3x0123 = vcvtq_f32_s32(vacc3x0123);
    float32x4_t vout3x4567 = vcvtq_f32_s32(vacc3x4567);
    const float32x4_t vscale0 = vdupq_n_f32(quantization_params[0].scale);
    vout0x0123 = vmulq_f32(vout0x0123, vscale0);
    vout0x4567 = vmulq_f32(vout0x4567, vscale0);
    const float32x4_t vscale1 = vdupq_n_f32(quantization_params[1].scale);
    vout1x0123 = vmulq_f32(vout1x0123, vscale1);
    vout1x4567 = vmulq_f32(vout1x4567, vscale1);
    const float32x4_t vscale2 = vdupq_n_f32(quantization_params[2].scale);
    vout2x0123 = vmulq_f32(vout2x0123, vscale2);
    vout2x4567 = vmulq_f32(vout2x4567, vscale2);
    const float32x4_t vscale3 = vdupq_n_f32(quantization_params[3].scale);
    vout3x0123 = vmulq_f32(vout3x0123, vscale3);
    vout3x4567 = vmulq_f32(vout3x4567, vscale3);

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    vout0x0123 = vaddq_f32(vout0x0123, vbias0123);
    vout0x4567 = vaddq_f32(vout0x4567, vbias4567);
    vout1x0123 = vaddq_f32(vout1x0123, vbias0123);
    vout1x4567 = vaddq_f32(vout1x4567, vbias4567);
    vout2x0123 = vaddq_f32(vout2x0123, vbias0123);
    vout2x4567 = vaddq_f32(vout2x4567, vbias4567);
    vout3x0123 = vaddq_f32(vout3x0123, vbias0123);
    vout3x4567 = vaddq_f32(vout3x4567, vbias4567);
    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, vmin);
    vout0x0123 = vminq_f32(vout0x0123, vmax);
    vout0x4567 = vmaxq_f32(vout0x4567, vmin);
    vout0x4567 = vminq_f32(vout0x4567, vmax);
    vout1x0123 = vmaxq_f32(vout1x0123, vmin);
    vout1x0123 = vminq_f32(vout1x0123, vmax);
    vout1x4567 = vmaxq_f32(vout1x4567, vmin);
    vout1x4567 = vminq_f32(vout1x4567, vmax);
    vout2x0123 = vmaxq_f32(vout2x0123, vmin);
    vout2x0123 = vminq_f32(vout2x0123, vmax);
    vout2x4567 = vmaxq_f32(vout2x4567, vmin);
    vout2x4567 = vminq_f32(vout2x4567, vmax);
    vout3x0123 = vmaxq_f32(vout3x0123, vmin);
    vout3x0123 = vminq_f32(vout3x0123, vmax);
    vout3x4567 = vmaxq_f32(vout3x4567, vmin);
    vout3x4567 = vminq_f32(vout3x4567, vmax);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(&c3[0], vout3x0123);
      vst1q_f32(&c3[4], vout3x4567);
      vst1q_f32(&c2[0], vout2x0123);
      vst1q_f32(&c2[4], vout2x4567);
      vst1q_f32(&c1[0], vout1x0123);
      vst1q_f32(&c1[4], vout1x4567);
      vst1q_f32(&c0[0], vout0x0123);
      vst1q_f32(&c0[4], vout0x4567);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      nc -= 8;
    } else {
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
