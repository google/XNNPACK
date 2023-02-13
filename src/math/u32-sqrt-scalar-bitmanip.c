// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_u32_sqrt__scalar_bitmanip(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n % sizeof(uint32_t) == 0);

  for (; n != 0; n -= sizeof(uint32_t)) {
    uint32_t vx = *input++;

    // Based on Hacker's Delight, Figure 11-4.
    uint32_t vm = UINT32_C(0x40000000);
    uint32_t vy = 0;
    for (uint32_t i = 0; i < 16; i++) {
      const uint32_t vb = vy | vm;
      vy >>= 1;
      if XNN_UNPREDICTABLE(vx >= vb) {
        vx -= vb;
        vy |= vm;
      }
      vm >>= 2;
    }

    // vy is sqrt(.) rounded down. Do the final rounding up if needed.
    if XNN_UNPREDICTABLE(vx > vy) {
      vy += 1;
    }

    *output++ = vy;
  }
}
