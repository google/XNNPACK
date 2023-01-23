// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <arm_neon.h>

#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f16_sqrt__aarch64_neonfp16arith_sqrt(
    size_t n,
    const void* input,
    void* output)
{
  assert(n % (8 * sizeof(uint16_t)) == 0);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; n != 0; n -= 8 * sizeof(uint16_t)) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float16x8_t vy = vsqrtq_f16(vx);

    vst1q_u16(o, vreinterpretq_u16_f16(vy)); o += 8;
  }
}
