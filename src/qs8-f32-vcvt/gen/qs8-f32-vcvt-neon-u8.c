// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-f32-vcvt/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vcvt.h"


void xnn_qs8_f32_vcvt_ukernel__neon_u8(
    size_t batch,
    const int8_t* input,
    float* output,
    const struct xnn_qs8_f32_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float32x4_t vscale = vdupq_n_f32(params->scalar.scale);
#if XNN_ARCH_ARM64
  const uint8x16_t vsign_mask = vdupq_n_u8(UINT8_C(0x80));
  const uint8x16_t vmagic_bias = vreinterpretq_u8_u32(vdupq_n_u32(UINT32_C(0x4B000080) + (uint32_t) (int32_t) params->scalar.zero_point));
  static const uint8_t idx_table[64] = {
    0, 17, 18, 19,  1, 17, 18, 19,  2, 17, 18, 19,  3, 17, 18, 19,
    4, 17, 18, 19,  5, 17, 18, 19,  6, 17, 18, 19,  7, 17, 18, 19,
    8, 17, 18, 19,  9, 17, 18, 19, 10, 17, 18, 19, 11, 17, 18, 19,
   12, 17, 18, 19, 13, 17, 18, 19, 14, 17, 18, 19, 15, 17, 18, 19,
  };
  const uint8x16_t vidx0 = vld1q_u8(idx_table);
  const uint8x16_t vidx1 = vld1q_u8(idx_table + 16);
  const float32x4_t vbias = vreinterpretq_f32_u8(vmagic_bias);
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    const uint8x8_t vx_lo = vld1_u8((const uint8_t*) input); input += 8;
    const uint8x16_t vx = veorq_u8(vcombine_u8(vx_lo, vx_lo), vsign_mask);

    const uint8x16x2_t vtbl = {{ vx, vmagic_bias }};
    float32x4_t vy_lo = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl, vidx0)), vbias);
    float32x4_t vy_hi = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl, vidx1)), vbias);

    vy_lo = vmulq_f32(vy_lo, vscale);
    vy_hi = vmulq_f32(vy_hi, vscale);

    vst1q_f32(output, vy_lo); output += 4;
    vst1q_f32(output, vy_hi); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    const uint8x8_t vx_lo = vld1_u8((const uint8_t*) input);
    const uint8x16_t vx = veorq_u8(vcombine_u8(vx_lo, vx_lo), vsign_mask);

    const uint8x16x2_t vtbl = {{ vx, vmagic_bias }};
    float32x4_t vy = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl, vidx0)), vbias);
    vy = vmulq_f32(vy, vscale);

    if (batch & (4 * sizeof(int8_t))) {
      vst1q_f32(output, vy); output += 4;
      vy = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl, vidx1)), vbias);
      vy = vmulq_f32(vy, vscale);
    }
    float32x2_t vy_lo = vget_low_f32(vy);
    if (batch & (2 * sizeof(int8_t))) {
      vst1_f32(output, vy_lo); output += 2;
      vy_lo = vget_high_f32(vy);
    }
    if (batch & (1 * sizeof(int8_t))) {
      vst1_lane_f32(output, vy_lo, 0);
    }
  }
#else
  const int16x8_t vminus_zero_point = vdupq_n_s16(-params->scalar.zero_point);
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    const int8x8_t vx = vld1_s8(input); input += 8;

    const int16x8_t vhx = vaddw_s8(vminus_zero_point, vx);

    const int32x4_t vwx_lo = vmovl_s16(vget_low_s16(vhx));
    const int32x4_t vwx_hi = vmovl_s16(vget_high_s16(vhx));

    float32x4_t vy_lo = vcvtq_f32_s32(vwx_lo);
    float32x4_t vy_hi = vcvtq_f32_s32(vwx_hi);

    vy_lo = vmulq_f32(vy_lo, vscale);
    vy_hi = vmulq_f32(vy_hi, vscale);

    vst1q_f32(output, vy_lo); output += 4;
    vst1q_f32(output, vy_hi); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    const int8x8_t vx = vld1_s8(input);

    const int16x8_t vhx = vaddw_s8(vminus_zero_point, vx);

    const int32x4_t vwx_lo = vmovl_s16(vget_low_s16(vhx));
    const int32x4_t vwx_hi = vmovl_s16(vget_high_s16(vhx));

    float32x4_t vy = vcvtq_f32_s32(vwx_lo);
    vy = vmulq_f32(vy, vscale);

    if (batch & (4 * sizeof(int8_t))) {
      vst1q_f32(output, vy); output += 4;
      vy = vcvtq_f32_s32(vwx_hi);
      vy = vmulq_f32(vy, vscale);
    }
    float32x2_t vy_lo = vget_low_f32(vy);
    if (batch & (2 * sizeof(int8_t))) {
      vst1_f32(output, vy_lo); output += 2;
      vy_lo = vget_high_f32(vy);
    }
    if (batch & (1 * sizeof(int8_t))) {
      vst1_lane_f32(output, vy_lo, 0);
    }
  }
#endif
}
