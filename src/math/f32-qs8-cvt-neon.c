// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_qs8_cvt__neon(
    size_t n,
    const float* input,
    int8_t* output,
    int8_t output_zero_point)
{
  assert(n % (8 * sizeof(int8_t)) == 0);

  const float32x4_t vfmagic = vdupq_n_f32(12582912.0f);
  const int32x4_t vimagic = vdupq_n_s32(INT32_C(0x4B400000) - (int32_t) output_zero_point);
  for (; n != 0; n -= 8 * sizeof(int8_t)) {
    float32x4_t vx_lo = vld1q_f32(input); input += 4;
    float32x4_t vx_hi = vld1q_f32(input); input += 4;

    vx_lo = vaddq_f32(vx_lo, vfmagic);
    vx_hi = vaddq_f32(vx_hi, vfmagic);

    int32x4_t vy_lo = vreinterpretq_s32_f32(vx_lo);
    int32x4_t vy_hi = vreinterpretq_s32_f32(vx_hi);

    vy_lo = vqsubq_s32(vy_lo, vimagic);
    vy_hi = vqsubq_s32(vy_hi, vimagic);

    const int16x8_t vy = vcombine_s16(vqmovn_s32(vy_lo), vqmovn_s32(vy_hi));

    const int8x8_t vout = vqmovn_s16(vy);
    vst1_s8(output, vout); output += 8;
  }
}
