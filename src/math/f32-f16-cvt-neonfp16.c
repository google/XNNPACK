// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_f16_cvt__neonfp16(
    size_t n,
    const float* input,
    void* output)
{
  assert(n % (4 * sizeof(uint16_t)) == 0);

  uint16_t* o = (uint16_t*) output;
  for (; n != 0; n -= 4 * sizeof(uint16_t)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;
    const uint16x4_t vy = vreinterpret_u16_f16(vcvt_f16_f32(vx));
    vst1_u16(o, vy); o += 4;
  }
}
