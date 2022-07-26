// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_u32_sqrt__scalar_clz_newton(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n % sizeof(uint32_t) == 0);

  for (; n != 0; n -= sizeof(uint32_t)) {
    const uint32_t vx = *input++;

    uint32_t vy = vx;

    // Based on Hacker's Delight, Figure 11-1.
    if (vx != 0) {
      const uint32_t vs = 16 - (math_clz_nonzero_u32(vx - 1) >> 1);

      uint32_t vg0 = UINT32_C(1) << vs;
      uint32_t vg1 = (vg0 + (vx >> vs)) >> 1;
      while XNN_LIKELY(vg1 < vg0) {
        vg0 = vg1;
        vg1 = (vg0 + vx / vg0) >> 1;
      }

      // vg0 is sqrt(vx) rounded down. Do the final rounding up if needed.
      if XNN_UNPREDICTABLE(vg0 * vg0 < vx - vg0) {
        vg0 += 1;
      }
      vy = vg0;
    }

    *output++ = vy;
  }
}
