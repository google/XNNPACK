// Auto-generated file. Do not edit!
//   Template: src/qs8-f16-vcvt/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vcvt.h"


void xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u24(
    size_t batch,
    const int8_t* input,
    void* output,
    const union xnn_qs8_f16_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  uint16_t* o = (uint16_t*) output;
  const int16x8_t vminus_zero_point = vdupq_n_s16(-params->neon.zero_point);
#ifdef XNN_COMPILER_MSVC
  const float16x8_t vscale = vreinterpretq_f16_u16(vdupq_n_u16(params->neon.scale));
#else
  const float16x8_t vscale = vreinterpretq_f16_u16(vld1q_dup_u16(&params->neon.scale));
#endif
  for (; batch >= 24 * sizeof(int8_t); batch -= 24 * sizeof(int8_t)) {
    const int8x8_t vx01234567 = vld1_s8(input); input += 8;
    const int8x8_t vx89ABCDEF = vld1_s8(input); input += 8;
    const int8x8_t vxGHIJKLMN = vld1_s8(input); input += 8;

    const int16x8_t vhx01234567 = vaddw_s8(vminus_zero_point, vx01234567);
    const int16x8_t vhx89ABCDEF = vaddw_s8(vminus_zero_point, vx89ABCDEF);
    const int16x8_t vhxGHIJKLMN = vaddw_s8(vminus_zero_point, vxGHIJKLMN);

    float16x8_t vy01234567 = vcvtq_f16_s16(vhx01234567);
    float16x8_t vy89ABCDEF = vcvtq_f16_s16(vhx89ABCDEF);
    float16x8_t vyGHIJKLMN = vcvtq_f16_s16(vhxGHIJKLMN);

    vy01234567 = vmulq_f16(vy01234567, vscale);
    vy89ABCDEF = vmulq_f16(vy89ABCDEF, vscale);
    vyGHIJKLMN = vmulq_f16(vyGHIJKLMN, vscale);

    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy89ABCDEF)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vyGHIJKLMN)); o += 8;
  }
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    const int8x8_t vx = vld1_s8(input); input += 8;

    const int16x8_t vhx = vaddw_s8(vminus_zero_point, vx);

    float16x8_t vy = vcvtq_f16_s16(vhx);

    vy = vmulq_f16(vy, vscale);

    vst1q_u16(o, vreinterpretq_u16_f16(vy)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    const int8x8_t vx = vld1_s8(input);

    const int16x8_t vhx = vaddw_s8(vminus_zero_point, vx);

    float16x8_t vy = vcvtq_f16_s16(vhx);
    vy = vmulq_f16(vy, vscale);

    float16x4_t vy_lo = vget_low_f16(vy);
    if (batch & (4 * sizeof(int8_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vy_lo)); o += 4;
      vy_lo = vget_high_f16(vy);
    }
    if (batch & (2 * sizeof(int8_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy_lo), 0); o += 2;

      vy_lo = vext_f16(vy_lo, vy_lo, 2);
    }
    if (batch & (1 * sizeof(int8_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy_lo), 0);
    }
  }
}
