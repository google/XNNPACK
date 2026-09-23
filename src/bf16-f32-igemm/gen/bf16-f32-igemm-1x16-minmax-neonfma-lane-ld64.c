// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-f32-igemm/neonfma-lane.c.in
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
#include "src/xnnpack/igemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


// Packed weights layout, per block of 16 output channels:
//   float bias[16];
//   for each kernel element, for each input channel:
//     uint16_t kernel[16];  // bf16
//
// Every bf16 value is widened to fp32 with a 16-bit shift, so this only needs
// NEON FMA (e.g. Apple M1, which has no BF16 instructions).
void xnn_bf16_f32_igemm_minmax_ukernel_1x16__neonfma_lane_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const xnn_bfloat16** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const xnn_bfloat16* zero,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(xnn_bfloat16) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(xnn_bfloat16) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  const float32x4_t vmin = vdupq_n_f32(params->scalar.min);
  const float32x4_t vmax = vdupq_n_f32(params->scalar.max);
  do {
    float32x4_t vacc0x0 = vld1q_f32((const float*) w + 0);
    float32x4_t vacc0x1 = vld1q_f32((const float*) w + 4);
    float32x4_t vacc0x2 = vld1q_f32((const float*) w + 8);
    float32x4_t vacc0x3 = vld1q_f32((const float*) w + 12);
    const uint16_t* wk = (const uint16_t*) ((const float*) w + 16);

    size_t p = ks;
    do {
      const uint16_t* restrict a0 = (const uint16_t*) a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != (const uint16_t*) zero) {
        a0 = (const uint16_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      for (; k >= 4 * sizeof(xnn_bfloat16); k -= 4 * sizeof(xnn_bfloat16)) {
        const float32x4_t va0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(a0), 16));
        a0 += 4;

        const uint16x8_t vb0c0 = vld1q_u16(wk + 0);
        const float32x4_t vb0x0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vb0c0), 16));
        const float32x4_t vb1x0 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vb0c0), 16));
        const uint16x8_t vb2c0 = vld1q_u16(wk + 8);
        const float32x4_t vb2x0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vb2c0), 16));
        const float32x4_t vb3x0 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vb2c0), 16));
        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0x0, vget_low_f32(va0), 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb1x0, vget_low_f32(va0), 0);
        vacc0x2 = vfmaq_lane_f32(vacc0x2, vb2x0, vget_low_f32(va0), 0);
        vacc0x3 = vfmaq_lane_f32(vacc0x3, vb3x0, vget_low_f32(va0), 0);
        const uint16x8_t vb0c1 = vld1q_u16(wk + 16);
        const float32x4_t vb0x1 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vb0c1), 16));
        const float32x4_t vb1x1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vb0c1), 16));
        const uint16x8_t vb2c1 = vld1q_u16(wk + 24);
        const float32x4_t vb2x1 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vb2c1), 16));
        const float32x4_t vb3x1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vb2c1), 16));
        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0x1, vget_low_f32(va0), 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb1x1, vget_low_f32(va0), 1);
        vacc0x2 = vfmaq_lane_f32(vacc0x2, vb2x1, vget_low_f32(va0), 1);
        vacc0x3 = vfmaq_lane_f32(vacc0x3, vb3x1, vget_low_f32(va0), 1);
        const uint16x8_t vb0c2 = vld1q_u16(wk + 32);
        const float32x4_t vb0x2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vb0c2), 16));
        const float32x4_t vb1x2 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vb0c2), 16));
        const uint16x8_t vb2c2 = vld1q_u16(wk + 40);
        const float32x4_t vb2x2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vb2c2), 16));
        const float32x4_t vb3x2 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vb2c2), 16));
        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0x2, vget_high_f32(va0), 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb1x2, vget_high_f32(va0), 0);
        vacc0x2 = vfmaq_lane_f32(vacc0x2, vb2x2, vget_high_f32(va0), 0);
        vacc0x3 = vfmaq_lane_f32(vacc0x3, vb3x2, vget_high_f32(va0), 0);
        const uint16x8_t vb0c3 = vld1q_u16(wk + 48);
        const float32x4_t vb0x3 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vb0c3), 16));
        const float32x4_t vb1x3 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vb0c3), 16));
        const uint16x8_t vb2c3 = vld1q_u16(wk + 56);
        const float32x4_t vb2x3 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vb2c3), 16));
        const float32x4_t vb3x3 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vb2c3), 16));
        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0x3, vget_high_f32(va0), 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb1x3, vget_high_f32(va0), 1);
        vacc0x2 = vfmaq_lane_f32(vacc0x2, vb2x3, vget_high_f32(va0), 1);
        vacc0x3 = vfmaq_lane_f32(vacc0x3, vb3x3, vget_high_f32(va0), 1);
        wk += 64;
      }
      for (; k != 0; k -= sizeof(xnn_bfloat16)) {
        const float32x4_t va0 = vdupq_n_f32(math_cvt_fp32_bf16(*a0++));

        const uint16x8_t vb0c = vld1q_u16(wk + 0);
        const float32x4_t vb0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vb0c), 16));
        const float32x4_t vb1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vb0c), 16));
        const uint16x8_t vb2c = vld1q_u16(wk + 8);
        const float32x4_t vb2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vb2c), 16));
        const float32x4_t vb3 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vb2c), 16));
        wk += 16;

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb1);
        vacc0x2 = vfmaq_f32(vacc0x2, va0, vb2);
        vacc0x3 = vfmaq_f32(vacc0x3, va0, vb3);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);
    w = (const void*) wk;

    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);
    vacc0x2 = vmaxq_f32(vacc0x2, vmin);
    vacc0x3 = vmaxq_f32(vacc0x3, vmin);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);
    vacc0x2 = vminq_f32(vacc0x2, vmax);
    vacc0x3 = vminq_f32(vacc0x3, vmax);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(c0 + 0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      vst1q_f32(c0 + 8, vacc0x2);
      vst1q_f32(c0 + 12, vacc0x3);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const xnn_bfloat16**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc & 8) {
        vst1q_f32(c0, vacc0x0);
        vst1q_f32(c0 + 4, vacc0x1);
        vacc0x0 = vacc0x2;
        vacc0x1 = vacc0x3;
        c0 += 8;
      }
      if (nc & 4) {
        vst1q_f32(c0, vacc0x0);
        vacc0x0 = vacc0x1;
        c0 += 4;
      }
      float32x2_t vacc0_lo = vget_low_f32(vacc0x0);
      if (nc & 2) {
        vst1_f32(c0, vacc0_lo);
        vacc0_lo = vget_high_f32(vacc0x0);
        c0 += 2;
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vacc0_lo, 0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
