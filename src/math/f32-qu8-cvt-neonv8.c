// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f32_qu8_cvt__neonv8(
    size_t n,
    const float* input,
    uint8_t* output,
    uint8_t output_zero_point)
{
  assert(n % (8 * sizeof(int8_t)) == 0);

  const int16x8_t voutput_zero_point = vdupq_n_s16((int16_t) (uint16_t) output_zero_point);
  for (; n != 0; n -= 8 * sizeof(int8_t)) {
    const float32x4_t vx_lo = vld1q_f32(input); input += 4;
    const float32x4_t vx_hi = vld1q_f32(input); input += 4;

    const int32x4_t vy_lo = vcvtnq_s32_f32(vx_lo);
    const int32x4_t vy_hi = vcvtnq_s32_f32(vx_hi);

    const int16x8_t vy = vqaddq_s16(vcombine_s16(vqmovn_s32(vy_lo), vqmovn_s32(vy_hi)), voutput_zero_point);

    const uint8x8_t vout = vqmovun_s16(vy);
    vst1_u8(output, vout); output += 8;
  }
}
