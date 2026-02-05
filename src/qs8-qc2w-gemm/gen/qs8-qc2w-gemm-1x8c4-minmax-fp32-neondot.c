// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c4-neondot.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_1x8c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params* restrict params) XNN_OOB_READS
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
  int8_t* c0 = c;

  // Loop over groups of 8 columns.
  do {
    // Initialize accumulators with bias. 8 bias values are loaded from the
    // weight matrix, at the start of the group of 8 columns.
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const int32_t*) w + 4;

    // Inner accumulation loop along the 8 columns.
    size_t k = kc;
    // 4x partial unrolled loop to load 16 bytes at a time.
    while (k >= 16 * sizeof(int8_t)) {
      // Load a 1x16 block of activations.
      const int8x16_t va_0x16 = vld1q_s8(a0); a0 += 16;

      // Load a 16x8 block of weights.
      const int8x16_t vb0123x16 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb4567x16 = vld1q_s8(w); w = (const int8_t*) w + 16;
      // First crumb.
      const int8x16_t vb0123x0123 = vshrq_n_s8(vshlq_n_s8(vb0123x16, 6), 6);
      const int8x16_t vb4567x0123 = vshrq_n_s8(vshlq_n_s8(vb4567x16, 6), 6);
      // Second crumb.
      const int8x16_t vb0123x4567 = vshrq_n_s8(vshlq_n_s8(vb0123x16, 4), 6);
      const int8x16_t vb4567x4567 = vshrq_n_s8(vshlq_n_s8(vb4567x16, 4), 6);
      // Third crumb.
      const int8x16_t vb0123x89AB = vshrq_n_s8(vshlq_n_s8(vb0123x16, 2), 6);
      const int8x16_t vb4567x89AB = vshrq_n_s8(vshlq_n_s8(vb4567x16, 2), 6);
      // Fourth crumb.
      const int8x16_t vb0123xCDEF = vshrq_n_s8(vb0123x16, 6);
      const int8x16_t vb4567xCDEF = vshrq_n_s8(vb4567x16, 6);

      // Multiply-accumulate: 1x16 * 16x8 --> 1x8.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, vget_low_s8(va_0x16), 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x0123, vget_low_s8(va_0x16), 0);

      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x4567, vget_low_s8(va_0x16), 1);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, vget_low_s8(va_0x16), 1);

      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x89AB, vget_high_s8(va_0x16), 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x89AB, vget_high_s8(va_0x16), 0);

      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123xCDEF, vget_high_s8(va_0x16), 1);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567xCDEF, vget_high_s8(va_0x16), 1);

      k -= 16 * sizeof(int8_t);
    }
    // Handle up to 8 final positions of `k`.
    if XNN_UNLIKELY(k > 0) {
      int8x16_t vb01234567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      int8x16_t vb01234567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 8 * sizeof(int8_t)) {
      // Load a 1x8 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;

      // Load a 8x8 block of weights.
      const int8x16_t vb0123x0123 = vshrq_n_s8(vshlq_n_s8(vb01234567x0123, 6), 6);
      const int8x16_t vb0123x4567 = vshrq_n_s8(vshlq_n_s8(vb01234567x4567, 6), 6);
      const int8x16_t vb4567x0123 = vshrq_n_s8(vshlq_n_s8(vb01234567x0123, 4), 6);
      const int8x16_t vb4567x4567 = vshrq_n_s8(vshlq_n_s8(vb01234567x4567, 4), 6);

      // Multiply-accumulate: 1x8 * 8x8 --> 1x8.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);

      k -= 8 * sizeof(int8_t);
      vb01234567x0123 = vshrq_n_s8(vb01234567x0123, 4);
      vb01234567x4567 = vshrq_n_s8(vb01234567x4567, 4);
    }
    // Handle up to 4 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 1x4 block of activations.
      const int8x8_t va0x0123 = vreinterpret_s8_s32(vld1_dup_s32((const int32_t*)a0)); a0 += 4;

      const int8x16_t vb0123x0123 = vshrq_n_s8(vshlq_n_s8(vb01234567x0123, 6), 6);
      const int8x16_t vb0123x4567 = vshrq_n_s8(vshlq_n_s8(vb01234567x4567, 6), 6);

      // Multiply-accumulate: 1x4 * 4x8 --> 1x8.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x0123, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x0123, 0);
    }
    }

    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);

    const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale4567);

    vacc0x0123 = vcvtnq_s32_f32(vfpacc0x0123);
    vacc0x4567 = vcvtnq_s32_f32(vfpacc0x4567);

    const int16x8_t voutput_zero_point = vdupq_n_s16(params->fp32_neonv8.output_zero_point);
    #if XNN_ARCH_ARM64
      const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);

      int8x8_t vout0x01234567 = vqmovn_s16(vacc0x01234567);
    #else
      const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);

      int8x8_t vout0x01234567 = vqmovn_s16(vacc0x01234567);
    #endif
    const int8x8_t voutput_min = vdup_n_s8(params->fp32_neonv8.output_min);
    const int8x8_t voutput_max = vdup_n_s8(params->fp32_neonv8.output_max);

    vout0x01234567 = vmax_s8(vout0x01234567, voutput_min);

    vout0x01234567 = vmin_s8(vout0x01234567, voutput_max);

    if (nc >= 8) {
      // Main case where there the 8 columns fit in the destination.
      vst1_s8(c0 + 0, vout0x01234567);

      // Advance to the next 8 columns.
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      // Final case where not all of the 8 columns fit in the destination.
      if (nc & 4) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_s8(vout0x01234567), 0); c0 += 4;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 4);
      }
      if (nc & 2) {
        vst1_lane_u16((void*) c0, vreinterpret_u16_s8(vout0x01234567), 0); c0 += 2;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 2);
      }
      if (nc & 1) {
        vst1_lane_s8(c0, vout0x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
