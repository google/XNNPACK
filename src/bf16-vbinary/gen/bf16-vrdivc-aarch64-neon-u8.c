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
  const uint32x4_t vbias = vdupq_n_u32(UINT32_C(0x00007FFF));
  const uint32x4_t vone = vdupq_n_u32(UINT32_C(1));

  const uint32x4_t vi_lo = vreinterpretq_u32_f32(lo);
  const uint32x4_t vi_hi = vreinterpretq_u32_f32(hi);
  const uint32x4_t vlsb_lo = vandq_u32(vshrq_n_u32(vi_lo, 16), vone);
  const uint32x4_t vlsb_hi = vandq_u32(vshrq_n_u32(vi_hi, 16), vone);
  const uint16x4_t vbf_lo = vaddhn_u32(vi_lo, vaddq_u32(vlsb_lo, vbias));
  #if XNN_ARCH_ARM64
    return vaddhn_high_u32(vbf_lo, vi_hi, vaddq_u32(vlsb_hi, vbias));
  #else
    return vcombine_u16(vbf_lo, vaddhn_u32(vi_hi, vaddq_u32(vlsb_hi, vbias)));
  #endif
}

void xnn_bf16_vrdivc_ukernel__aarch64_neon_u8(
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
  uint16_t* o = (uint16_t*) output;
  const float32x4_t vb = vreinterpretq_f32_u32(
      vdupq_n_u32((uint32_t) *((const uint16_t*) input_b) << 16));
  for (; batch >= 8 * sizeof(xnn_bfloat16); batch -= 8 * sizeof(xnn_bfloat16)) {
    float32x4_t va0, va1;
    load_bf16x8_as_f32(a, &va0, &va1);
    a += 8;

    float32x4_t vy0 = vdivq_f32(vb, va0);
    float32x4_t vy1 = vdivq_f32(vb, va1);

    vst1q_u16(o, convert_f32x8_to_bf16(vy0, vy1));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    float32x4_t va0, va1;
    load_bf16x8_as_f32(a, &va0, &va1);

    float32x4_t vy0 = vdivq_f32(vb, va0);
    float32x4_t vy1 = vdivq_f32(vb, va1);

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
