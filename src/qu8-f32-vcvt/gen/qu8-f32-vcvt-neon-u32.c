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


void xnn_qu8_f32_vcvt_ukernel__neon_u32(
    size_t batch,
    const uint8_t* input,
    float* output,
    const struct xnn_qu8_f32_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float32x4_t vscale = vdupq_n_f32(params->scalar.scale);
#if XNN_ARCH_ARM64
  const uint8x16_t vmagic_bias = vreinterpretq_u8_u32(vdupq_n_u32(UINT32_C(0x4B000000) + (uint32_t) params->scalar.zero_point));
  static const uint8_t idx_table[64] = {
    0, 17, 18, 19,  1, 17, 18, 19,  2, 17, 18, 19,  3, 17, 18, 19,
    4, 17, 18, 19,  5, 17, 18, 19,  6, 17, 18, 19,  7, 17, 18, 19,
    8, 17, 18, 19,  9, 17, 18, 19, 10, 17, 18, 19, 11, 17, 18, 19,
   12, 17, 18, 19, 13, 17, 18, 19, 14, 17, 18, 19, 15, 17, 18, 19,
  };
  const uint8x16_t vidx0 = vld1q_u8(idx_table);
  const uint8x16_t vidx1 = vld1q_u8(idx_table + 16);
  const uint8x16_t vidx2 = vld1q_u8(idx_table + 32);
  const uint8x16_t vidx3 = vld1q_u8(idx_table + 48);
  const float32x4_t vbias = vreinterpretq_f32_u8(vmagic_bias);
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    uint8x16_t vx0123456789ABCDEF = vld1q_u8((const uint8_t*) input); input += 16;
    uint8x16_t vxGHIJKLMNOPQRSTUV = vld1q_u8((const uint8_t*) input); input += 16;


    const uint8x16x2_t vtbl0123456789ABCDEF = {{ vx0123456789ABCDEF, vmagic_bias }};
    float32x4_t vy0123 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEF, vidx0)), vbias);
    float32x4_t vy4567 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEF, vidx1)), vbias);
    float32x4_t vy89AB = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEF, vidx2)), vbias);
    float32x4_t vyCDEF = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEF, vidx3)), vbias);
    const uint8x16x2_t vtblGHIJKLMNOPQRSTUV = {{ vxGHIJKLMNOPQRSTUV, vmagic_bias }};
    float32x4_t vyGHIJ = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtblGHIJKLMNOPQRSTUV, vidx0)), vbias);
    float32x4_t vyKLMN = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtblGHIJKLMNOPQRSTUV, vidx1)), vbias);
    float32x4_t vyOPQR = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtblGHIJKLMNOPQRSTUV, vidx2)), vbias);
    float32x4_t vySTUV = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtblGHIJKLMNOPQRSTUV, vidx3)), vbias);

    vy0123 = vmulq_f32(vy0123, vscale);
    vy4567 = vmulq_f32(vy4567, vscale);
    vy89AB = vmulq_f32(vy89AB, vscale);
    vyCDEF = vmulq_f32(vyCDEF, vscale);
    vyGHIJ = vmulq_f32(vyGHIJ, vscale);
    vyKLMN = vmulq_f32(vyKLMN, vscale);
    vyOPQR = vmulq_f32(vyOPQR, vscale);
    vySTUV = vmulq_f32(vySTUV, vscale);

    vst1q_f32(output, vy0123); output += 4;
    vst1q_f32(output, vy4567); output += 4;
    vst1q_f32(output, vy89AB); output += 4;
    vst1q_f32(output, vyCDEF); output += 4;
    vst1q_f32(output, vyGHIJ); output += 4;
    vst1q_f32(output, vyKLMN); output += 4;
    vst1q_f32(output, vyOPQR); output += 4;
    vst1q_f32(output, vySTUV); output += 4;
  }
  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    const uint8x8_t vx_lo = vld1_u8((const uint8_t*) input); input += 8;
    const uint8x16_t vx = vcombine_u8(vx_lo, vx_lo);

    const uint8x16x2_t vtbl = {{ vx, vmagic_bias }};
    float32x4_t vy_lo = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl, vidx0)), vbias);
    float32x4_t vy_hi = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl, vidx1)), vbias);

    vy_lo = vmulq_f32(vy_lo, vscale);
    vy_hi = vmulq_f32(vy_hi, vscale);

    vst1q_f32(output, vy_lo); output += 4;
    vst1q_f32(output, vy_hi); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 7 * sizeof(uint8_t));

    const uint8x8_t vx_lo = vld1_u8((const uint8_t*) input);
    const uint8x16_t vx = vcombine_u8(vx_lo, vx_lo);

    const uint8x16x2_t vtbl = {{ vx, vmagic_bias }};
    float32x4_t vy = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl, vidx0)), vbias);
    vy = vmulq_f32(vy, vscale);

    if (batch & (4 * sizeof(uint8_t))) {
      vst1q_f32(output, vy); output += 4;
      vy = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl, vidx1)), vbias);
      vy = vmulq_f32(vy, vscale);
    }
    float32x2_t vy_lo = vget_low_f32(vy);
    if (batch & (2 * sizeof(uint8_t))) {
      vst1_f32(output, vy_lo); output += 2;
      vy_lo = vget_high_f32(vy);
    }
    if (batch & (1 * sizeof(uint8_t))) {
      vst1_lane_f32(output, vy_lo, 0);
    }
  }
#else
  const int16x8_t vminus_zero_point = vdupq_n_s16(-params->scalar.zero_point);
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    const uint8x8_t vx01234567 = vld1_u8(input); input += 8;
    const uint8x8_t vx89ABCDEF = vld1_u8(input); input += 8;
    const uint8x8_t vxGHIJKLMN = vld1_u8(input); input += 8;
    const uint8x8_t vxOPQRSTUV = vld1_u8(input); input += 8;

    const int16x8_t vhx01234567 = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(vminus_zero_point), vx01234567));
    const int16x8_t vhx89ABCDEF = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(vminus_zero_point), vx89ABCDEF));
    const int16x8_t vhxGHIJKLMN = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(vminus_zero_point), vxGHIJKLMN));
    const int16x8_t vhxOPQRSTUV = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(vminus_zero_point), vxOPQRSTUV));

    const int32x4_t vwx0123 = vmovl_s16(vget_low_s16(vhx01234567));
    const int32x4_t vwx4567 = vmovl_s16(vget_high_s16(vhx01234567));
    const int32x4_t vwx89AB = vmovl_s16(vget_low_s16(vhx89ABCDEF));
    const int32x4_t vwxCDEF = vmovl_s16(vget_high_s16(vhx89ABCDEF));
    const int32x4_t vwxGHIJ = vmovl_s16(vget_low_s16(vhxGHIJKLMN));
    const int32x4_t vwxKLMN = vmovl_s16(vget_high_s16(vhxGHIJKLMN));
    const int32x4_t vwxOPQR = vmovl_s16(vget_low_s16(vhxOPQRSTUV));
    const int32x4_t vwxSTUV = vmovl_s16(vget_high_s16(vhxOPQRSTUV));

    float32x4_t vy0123 = vcvtq_f32_s32(vwx0123);
    float32x4_t vy4567 = vcvtq_f32_s32(vwx4567);
    float32x4_t vy89AB = vcvtq_f32_s32(vwx89AB);
    float32x4_t vyCDEF = vcvtq_f32_s32(vwxCDEF);
    float32x4_t vyGHIJ = vcvtq_f32_s32(vwxGHIJ);
    float32x4_t vyKLMN = vcvtq_f32_s32(vwxKLMN);
    float32x4_t vyOPQR = vcvtq_f32_s32(vwxOPQR);
    float32x4_t vySTUV = vcvtq_f32_s32(vwxSTUV);

    vy0123 = vmulq_f32(vy0123, vscale);
    vy4567 = vmulq_f32(vy4567, vscale);
    vy89AB = vmulq_f32(vy89AB, vscale);
    vyCDEF = vmulq_f32(vyCDEF, vscale);
    vyGHIJ = vmulq_f32(vyGHIJ, vscale);
    vyKLMN = vmulq_f32(vyKLMN, vscale);
    vyOPQR = vmulq_f32(vyOPQR, vscale);
    vySTUV = vmulq_f32(vySTUV, vscale);

    vst1q_f32(output, vy0123); output += 4;
    vst1q_f32(output, vy4567); output += 4;
    vst1q_f32(output, vy89AB); output += 4;
    vst1q_f32(output, vyCDEF); output += 4;
    vst1q_f32(output, vyGHIJ); output += 4;
    vst1q_f32(output, vyKLMN); output += 4;
    vst1q_f32(output, vyOPQR); output += 4;
    vst1q_f32(output, vySTUV); output += 4;
  }
  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    const uint8x8_t vx = vld1_u8(input); input += 8;

    const int16x8_t vhx = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(vminus_zero_point), vx));

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
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 7 * sizeof(uint8_t));

    const uint8x8_t vx = vld1_u8(input);

    const int16x8_t vhx = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(vminus_zero_point), vx));

    const int32x4_t vwx_lo = vmovl_s16(vget_low_s16(vhx));
    const int32x4_t vwx_hi = vmovl_s16(vget_high_s16(vhx));

    float32x4_t vy = vcvtq_f32_s32(vwx_lo);
    vy = vmulq_f32(vy, vscale);

    if (batch & (4 * sizeof(uint8_t))) {
      vst1q_f32(output, vy); output += 4;
      vy = vcvtq_f32_s32(vwx_hi);
      vy = vmulq_f32(vy, vscale);
    }
    float32x2_t vy_lo = vget_low_f32(vy);
    if (batch & (2 * sizeof(uint8_t))) {
      vst1_f32(output, vy_lo); output += 2;
      vy_lo = vget_high_f32(vy);
    }
    if (batch & (1 * sizeof(uint8_t))) {
      vst1_lane_f32(output, vy_lo, 0);
    }
  }
#endif
}
