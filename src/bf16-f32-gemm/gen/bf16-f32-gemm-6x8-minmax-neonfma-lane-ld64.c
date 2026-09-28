// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-f32-gemm/neonfma-lane.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


// Packed weights layout, per block of 8 output channels:
//   float bias[8];
//   for each input channel: uint16_t kernel[8];  // bf16
//
// Every bf16 value is widened to fp32 with a 16-bit shift, so this only needs
// NEON FMA (e.g. Apple M1, which has no BF16 instructions).
void xnn_bf16_f32_gemm_minmax_ukernel_6x8__neonfma_lane_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint16_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(xnn_bfloat16) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const uint16_t* a0 = a;
  float* c0 = c;
  const uint16_t* a1 = (const uint16_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const uint16_t* a2 = (const uint16_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const uint16_t* a3 = (const uint16_t*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const uint16_t* a4 = (const uint16_t*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const uint16_t* a5 = (const uint16_t*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  const float32x4_t vmin = vdupq_n_f32(params->scalar.min);
  const float32x4_t vmax = vdupq_n_f32(params->scalar.max);
  do {
    float32x4_t vacc0x0 = vld1q_f32((const float*) w + 0);
    float32x4_t vacc0x1 = vld1q_f32((const float*) w + 4);
    float32x4_t vacc1x0 = vacc0x0;
    float32x4_t vacc1x1 = vacc0x1;
    float32x4_t vacc2x0 = vacc0x0;
    float32x4_t vacc2x1 = vacc0x1;
    float32x4_t vacc3x0 = vacc0x0;
    float32x4_t vacc3x1 = vacc0x1;
    float32x4_t vacc4x0 = vacc0x0;
    float32x4_t vacc4x1 = vacc0x1;
    float32x4_t vacc5x0 = vacc0x0;
    float32x4_t vacc5x1 = vacc0x1;
    const uint16_t* wk = (const uint16_t*) ((const float*) w + 8);

    size_t k = kc;
    for (; k >= 4 * sizeof(xnn_bfloat16); k -= 4 * sizeof(xnn_bfloat16)) {
      const float32x4_t va0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(a0), 16));
      a0 += 4;
      const float32x4_t va1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(a1), 16));
      a1 += 4;
      const float32x4_t va2 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(a2), 16));
      a2 += 4;
      const float32x4_t va3 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(a3), 16));
      a3 += 4;
      const float32x4_t va4 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(a4), 16));
      a4 += 4;
      const float32x4_t va5 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(a5), 16));
      a5 += 4;

      const uint16x8_t vb0c0 = vld1q_u16(wk + 0);
      const float32x4_t vb0x0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vb0c0), 16));
      const float32x4_t vb1x0 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vb0c0), 16));
      vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0x0, vget_low_f32(va0), 0);
      vacc0x1 = vfmaq_lane_f32(vacc0x1, vb1x0, vget_low_f32(va0), 0);
      vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0x0, vget_low_f32(va1), 0);
      vacc1x1 = vfmaq_lane_f32(vacc1x1, vb1x0, vget_low_f32(va1), 0);
      vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0x0, vget_low_f32(va2), 0);
      vacc2x1 = vfmaq_lane_f32(vacc2x1, vb1x0, vget_low_f32(va2), 0);
      vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0x0, vget_low_f32(va3), 0);
      vacc3x1 = vfmaq_lane_f32(vacc3x1, vb1x0, vget_low_f32(va3), 0);
      vacc4x0 = vfmaq_lane_f32(vacc4x0, vb0x0, vget_low_f32(va4), 0);
      vacc4x1 = vfmaq_lane_f32(vacc4x1, vb1x0, vget_low_f32(va4), 0);
      vacc5x0 = vfmaq_lane_f32(vacc5x0, vb0x0, vget_low_f32(va5), 0);
      vacc5x1 = vfmaq_lane_f32(vacc5x1, vb1x0, vget_low_f32(va5), 0);
      const uint16x8_t vb0c1 = vld1q_u16(wk + 8);
      const float32x4_t vb0x1 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vb0c1), 16));
      const float32x4_t vb1x1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vb0c1), 16));
      vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0x1, vget_low_f32(va0), 1);
      vacc0x1 = vfmaq_lane_f32(vacc0x1, vb1x1, vget_low_f32(va0), 1);
      vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0x1, vget_low_f32(va1), 1);
      vacc1x1 = vfmaq_lane_f32(vacc1x1, vb1x1, vget_low_f32(va1), 1);
      vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0x1, vget_low_f32(va2), 1);
      vacc2x1 = vfmaq_lane_f32(vacc2x1, vb1x1, vget_low_f32(va2), 1);
      vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0x1, vget_low_f32(va3), 1);
      vacc3x1 = vfmaq_lane_f32(vacc3x1, vb1x1, vget_low_f32(va3), 1);
      vacc4x0 = vfmaq_lane_f32(vacc4x0, vb0x1, vget_low_f32(va4), 1);
      vacc4x1 = vfmaq_lane_f32(vacc4x1, vb1x1, vget_low_f32(va4), 1);
      vacc5x0 = vfmaq_lane_f32(vacc5x0, vb0x1, vget_low_f32(va5), 1);
      vacc5x1 = vfmaq_lane_f32(vacc5x1, vb1x1, vget_low_f32(va5), 1);
      const uint16x8_t vb0c2 = vld1q_u16(wk + 16);
      const float32x4_t vb0x2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vb0c2), 16));
      const float32x4_t vb1x2 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vb0c2), 16));
      vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0x2, vget_high_f32(va0), 0);
      vacc0x1 = vfmaq_lane_f32(vacc0x1, vb1x2, vget_high_f32(va0), 0);
      vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0x2, vget_high_f32(va1), 0);
      vacc1x1 = vfmaq_lane_f32(vacc1x1, vb1x2, vget_high_f32(va1), 0);
      vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0x2, vget_high_f32(va2), 0);
      vacc2x1 = vfmaq_lane_f32(vacc2x1, vb1x2, vget_high_f32(va2), 0);
      vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0x2, vget_high_f32(va3), 0);
      vacc3x1 = vfmaq_lane_f32(vacc3x1, vb1x2, vget_high_f32(va3), 0);
      vacc4x0 = vfmaq_lane_f32(vacc4x0, vb0x2, vget_high_f32(va4), 0);
      vacc4x1 = vfmaq_lane_f32(vacc4x1, vb1x2, vget_high_f32(va4), 0);
      vacc5x0 = vfmaq_lane_f32(vacc5x0, vb0x2, vget_high_f32(va5), 0);
      vacc5x1 = vfmaq_lane_f32(vacc5x1, vb1x2, vget_high_f32(va5), 0);
      const uint16x8_t vb0c3 = vld1q_u16(wk + 24);
      const float32x4_t vb0x3 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vb0c3), 16));
      const float32x4_t vb1x3 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vb0c3), 16));
      vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0x3, vget_high_f32(va0), 1);
      vacc0x1 = vfmaq_lane_f32(vacc0x1, vb1x3, vget_high_f32(va0), 1);
      vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0x3, vget_high_f32(va1), 1);
      vacc1x1 = vfmaq_lane_f32(vacc1x1, vb1x3, vget_high_f32(va1), 1);
      vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0x3, vget_high_f32(va2), 1);
      vacc2x1 = vfmaq_lane_f32(vacc2x1, vb1x3, vget_high_f32(va2), 1);
      vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0x3, vget_high_f32(va3), 1);
      vacc3x1 = vfmaq_lane_f32(vacc3x1, vb1x3, vget_high_f32(va3), 1);
      vacc4x0 = vfmaq_lane_f32(vacc4x0, vb0x3, vget_high_f32(va4), 1);
      vacc4x1 = vfmaq_lane_f32(vacc4x1, vb1x3, vget_high_f32(va4), 1);
      vacc5x0 = vfmaq_lane_f32(vacc5x0, vb0x3, vget_high_f32(va5), 1);
      vacc5x1 = vfmaq_lane_f32(vacc5x1, vb1x3, vget_high_f32(va5), 1);
      wk += 32;
    }
    for (; k != 0; k -= sizeof(xnn_bfloat16)) {
      const float32x4_t va0 = vdupq_n_f32(math_cvt_fp32_bf16(*a0++));
      const float32x4_t va1 = vdupq_n_f32(math_cvt_fp32_bf16(*a1++));
      const float32x4_t va2 = vdupq_n_f32(math_cvt_fp32_bf16(*a2++));
      const float32x4_t va3 = vdupq_n_f32(math_cvt_fp32_bf16(*a3++));
      const float32x4_t va4 = vdupq_n_f32(math_cvt_fp32_bf16(*a4++));
      const float32x4_t va5 = vdupq_n_f32(math_cvt_fp32_bf16(*a5++));

      const uint16x8_t vb0c = vld1q_u16(wk + 0);
      const float32x4_t vb0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vb0c), 16));
      const float32x4_t vb1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vb0c), 16));
      wk += 8;

      vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0);
      vacc0x1 = vfmaq_f32(vacc0x1, va0, vb1);
      vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0);
      vacc1x1 = vfmaq_f32(vacc1x1, va1, vb1);
      vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0);
      vacc2x1 = vfmaq_f32(vacc2x1, va2, vb1);
      vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0);
      vacc3x1 = vfmaq_f32(vacc3x1, va3, vb1);
      vacc4x0 = vfmaq_f32(vacc4x0, va4, vb0);
      vacc4x1 = vfmaq_f32(vacc4x1, va4, vb1);
      vacc5x0 = vfmaq_f32(vacc5x0, va5, vb0);
      vacc5x1 = vfmaq_f32(vacc5x1, va5, vb1);
    }
    w = (const void*) wk;

    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);
    vacc1x0 = vmaxq_f32(vacc1x0, vmin);
    vacc1x1 = vmaxq_f32(vacc1x1, vmin);
    vacc2x0 = vmaxq_f32(vacc2x0, vmin);
    vacc2x1 = vmaxq_f32(vacc2x1, vmin);
    vacc3x0 = vmaxq_f32(vacc3x0, vmin);
    vacc3x1 = vmaxq_f32(vacc3x1, vmin);
    vacc4x0 = vmaxq_f32(vacc4x0, vmin);
    vacc4x1 = vmaxq_f32(vacc4x1, vmin);
    vacc5x0 = vmaxq_f32(vacc5x0, vmin);
    vacc5x1 = vmaxq_f32(vacc5x1, vmin);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);
    vacc1x0 = vminq_f32(vacc1x0, vmax);
    vacc1x1 = vminq_f32(vacc1x1, vmax);
    vacc2x0 = vminq_f32(vacc2x0, vmax);
    vacc2x1 = vminq_f32(vacc2x1, vmax);
    vacc3x0 = vminq_f32(vacc3x0, vmax);
    vacc3x1 = vminq_f32(vacc3x1, vmax);
    vacc4x0 = vminq_f32(vacc4x0, vmax);
    vacc4x1 = vminq_f32(vacc4x1, vmax);
    vacc5x0 = vminq_f32(vacc5x0, vmax);
    vacc5x1 = vminq_f32(vacc5x1, vmax);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0 + 0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      vst1q_f32(c1 + 0, vacc1x0);
      vst1q_f32(c1 + 4, vacc1x1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1q_f32(c2 + 0, vacc2x0);
      vst1q_f32(c2 + 4, vacc2x1);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1q_f32(c3 + 0, vacc3x0);
      vst1q_f32(c3 + 4, vacc3x1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      vst1q_f32(c4 + 0, vacc4x0);
      vst1q_f32(c4 + 4, vacc4x1);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      vst1q_f32(c5 + 0, vacc5x0);
      vst1q_f32(c5 + 4, vacc5x1);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);

      a0 = (const uint16_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint16_t*) ((uintptr_t) a1 - kc);
      a2 = (const uint16_t*) ((uintptr_t) a2 - kc);
      a3 = (const uint16_t*) ((uintptr_t) a3 - kc);
      a4 = (const uint16_t*) ((uintptr_t) a4 - kc);
      a5 = (const uint16_t*) ((uintptr_t) a5 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        vst1q_f32(c0, vacc0x0);
        vacc0x0 = vacc0x1;
        c0 += 4;
        vst1q_f32(c1, vacc1x0);
        vacc1x0 = vacc1x1;
        c1 += 4;
        vst1q_f32(c2, vacc2x0);
        vacc2x0 = vacc2x1;
        c2 += 4;
        vst1q_f32(c3, vacc3x0);
        vacc3x0 = vacc3x1;
        c3 += 4;
        vst1q_f32(c4, vacc4x0);
        vacc4x0 = vacc4x1;
        c4 += 4;
        vst1q_f32(c5, vacc5x0);
        vacc5x0 = vacc5x1;
        c5 += 4;
      }
      float32x2_t vacc0_lo = vget_low_f32(vacc0x0);
      float32x2_t vacc1_lo = vget_low_f32(vacc1x0);
      float32x2_t vacc2_lo = vget_low_f32(vacc2x0);
      float32x2_t vacc3_lo = vget_low_f32(vacc3x0);
      float32x2_t vacc4_lo = vget_low_f32(vacc4x0);
      float32x2_t vacc5_lo = vget_low_f32(vacc5x0);
      if (nc & 2) {
        vst1_f32(c0, vacc0_lo);
        vacc0_lo = vget_high_f32(vacc0x0);
        c0 += 2;
        vst1_f32(c1, vacc1_lo);
        vacc1_lo = vget_high_f32(vacc1x0);
        c1 += 2;
        vst1_f32(c2, vacc2_lo);
        vacc2_lo = vget_high_f32(vacc2x0);
        c2 += 2;
        vst1_f32(c3, vacc3_lo);
        vacc3_lo = vget_high_f32(vacc3x0);
        c3 += 2;
        vst1_f32(c4, vacc4_lo);
        vacc4_lo = vget_high_f32(vacc4x0);
        c4 += 2;
        vst1_f32(c5, vacc5_lo);
        vacc5_lo = vget_high_f32(vacc5x0);
        c5 += 2;
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vacc0_lo, 0);
        vst1_lane_f32(c1, vacc1_lo, 0);
        vst1_lane_f32(c2, vacc2_lo, 0);
        vst1_lane_f32(c3, vacc3_lo, 0);
        vst1_lane_f32(c4, vacc4_lo, 0);
        vst1_lane_f32(c5, vacc5_lo, 0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
