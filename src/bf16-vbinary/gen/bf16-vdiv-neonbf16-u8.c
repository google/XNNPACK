// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-vbinary/neon.c.in
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
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vbinary.h"

#if XNN_ARCH_ARM64

static inline void load_bf16x8_as_f32(
    const uint16_t* input, float32x4_t* lo, float32x4_t* hi) {
  const uint16x8_t vbf = vld1q_u16(input);
#if XNN_ARCH_ARM64
  const uint16x8_t vzero = vdupq_n_u16(0);
  *lo = vreinterpretq_f32_u16(vzip1q_u16(vzero, vbf));
  *hi = vreinterpretq_f32_u16(vzip2q_u16(vzero, vbf));
#else
  *lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vbf), 16));
  *hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vbf), 16));
#endif
}

static inline uint16x8_t convert_f32x8_to_bf16(float32x4_t lo, float32x4_t hi) {
  return vreinterpretq_u16_bf16(
      vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(lo), hi));
}

void xnn_bf16_vdiv_ukernel__neonbf16_u8(
    size_t batch,
    const xnn_bfloat16* input_a,
    const xnn_bfloat16* input_b,
    xnn_bfloat16* output,
    const struct xnn_bf16_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_bfloat16) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 8 * sizeof(xnn_bfloat16); batch -= 8 * sizeof(xnn_bfloat16)) {
    float32x4_t va0, va1;
    load_bf16x8_as_f32(a, &va0, &va1);
    a += 8;
    float32x4_t vb0, vb1;
    load_bf16x8_as_f32(b, &vb0, &vb1);
    b += 8;

    float32x4_t vy0 = vdivq_f32(va0, vb0);
    float32x4_t vy1 = vdivq_f32(va1, vb1);

    vst1q_u16(o, convert_f32x8_to_bf16(vy0, vy1));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    float32x4_t va0, va1;
    load_bf16x8_as_f32(a, &va0, &va1);
    float32x4_t vb0, vb1;
    load_bf16x8_as_f32(b, &vb0, &vb1);

    float32x4_t vy0 = vdivq_f32(va0, vb0);
    float32x4_t vy1 = vdivq_f32(va1, vb1);

    const uint16x8_t vy = convert_f32x8_to_bf16(vy0, vy1);
    uint16x4_t vy_lo = vget_low_u16(vy);
    if (batch & (4 * sizeof(xnn_bfloat16))) {
      vst1_u16(o, vy_lo); o += 4;
      vy_lo = vget_high_u16(vy);
    }
    if (batch & (2 * sizeof(xnn_bfloat16))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_u16(vy_lo), 0); o += 2;
      vy_lo = vext_u16(vy_lo, vy_lo, 2);
    }
    if (batch & (1 * sizeof(xnn_bfloat16))) {
      vst1_lane_u16(o, vy_lo, 0);
    }
  }
}
#endif  // XNN_ARCH_ARM64
