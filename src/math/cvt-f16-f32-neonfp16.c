// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f16_f32_cvt__neonfp16(
    size_t n,
    const void* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  const uint16_t* i = (const uint16_t*) input;
  for (; n != 0; n -= 4 * sizeof(float)) {
    const float16x4_t vx = vreinterpret_f16_u16(vld1_u16(i)); i += 4;
    const float32x4_t vy = vcvt_f32_f16(vx);
    vst1q_f32(output, vy); output += 4;
  }
}
