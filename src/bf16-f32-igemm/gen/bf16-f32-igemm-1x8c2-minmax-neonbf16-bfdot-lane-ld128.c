// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-f32-igemm/c2-neonbf16-bfdot-lane-ld128.c.in
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
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/unaligned.h"


// Packed weights layout, per block of 8 output channels:
//   float bias[8];
//   for each kernel element, for each pair of input channels:
//     uint16_t kernel[8][2];  // bf16
//
// Each pair of input channels is multiplied with the pairs of weights using
// BFDOT, so this needs the Arm BF16 extension (e.g. Apple M2 and later).
void xnn_bf16_f32_igemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128(
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
    const uint16_t* wk = (const uint16_t*) ((const float*) w + 8);

    size_t p = ks;
    do {
      const uint16_t* restrict a0 = (const uint16_t*) a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != (const uint16_t*) zero) {
        a0 = (const uint16_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      for (; k >= 8 * sizeof(xnn_bfloat16); k -= 8 * sizeof(xnn_bfloat16)) {
        const bfloat16x8_t va0 = vreinterpretq_bf16_u16(vld1q_u16(a0));
        a0 += 8;

        const bfloat16x8_t vb0c0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 0));
        const bfloat16x8_t vb1c0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 8));
        vacc0x0 = vbfdotq_laneq_f32(vacc0x0, vb0c0, va0, 0);
        vacc0x1 = vbfdotq_laneq_f32(vacc0x1, vb1c0, va0, 0);
        const bfloat16x8_t vb0c1 = vreinterpretq_bf16_u16(vld1q_u16(wk + 16));
        const bfloat16x8_t vb1c1 = vreinterpretq_bf16_u16(vld1q_u16(wk + 24));
        vacc0x0 = vbfdotq_laneq_f32(vacc0x0, vb0c1, va0, 1);
        vacc0x1 = vbfdotq_laneq_f32(vacc0x1, vb1c1, va0, 1);
        const bfloat16x8_t vb0c2 = vreinterpretq_bf16_u16(vld1q_u16(wk + 32));
        const bfloat16x8_t vb1c2 = vreinterpretq_bf16_u16(vld1q_u16(wk + 40));
        vacc0x0 = vbfdotq_laneq_f32(vacc0x0, vb0c2, va0, 2);
        vacc0x1 = vbfdotq_laneq_f32(vacc0x1, vb1c2, va0, 2);
        const bfloat16x8_t vb0c3 = vreinterpretq_bf16_u16(vld1q_u16(wk + 48));
        const bfloat16x8_t vb1c3 = vreinterpretq_bf16_u16(vld1q_u16(wk + 56));
        vacc0x0 = vbfdotq_laneq_f32(vacc0x0, vb0c3, va0, 3);
        vacc0x1 = vbfdotq_laneq_f32(vacc0x1, vb1c3, va0, 3);
        wk += 64;
      }
      // Remaining pairs of input channels. When the number of input channels
      // is odd, only the last one is loaded, so bytes past the end of the row
      // can't leak into the result through the zero-padded weights.
      while (k != 0) {
        uint32_t va0_pair;
        if XNN_LIKELY(k >= 2 * sizeof(xnn_bfloat16)) {
          va0_pair = unaligned_load_u32(a0);
          a0 += 2;
          k -= 2 * sizeof(xnn_bfloat16);
        } else {
          va0_pair = (uint32_t) *a0;
          k = 0;
        }
        const bfloat16x8_t va0 = vreinterpretq_bf16_u32(vdupq_n_u32(va0_pair));

        const bfloat16x8_t vb0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 0));
        const bfloat16x8_t vb1 = vreinterpretq_bf16_u16(vld1q_u16(wk + 8));
        wk += 16;

        vacc0x0 = vbfdotq_f32(vacc0x0, vb0, va0);
        vacc0x1 = vbfdotq_f32(vacc0x1, vb1, va0);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);
    w = (const void*) wk;

    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0 + 0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const xnn_bfloat16**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
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
