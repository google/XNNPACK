// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_u32_sqrt__scalar_tflm(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n % sizeof(uint32_t) == 0);

  for (; n != 0; n -= sizeof(uint32_t)) {
    uint32_t vx = *input++;

    // Algorithm adapted from tensorflow/lite/experimental/microfrontend/lib/filterbank.c in TFLite-Micro
    uint32_t vy = 0;
    if (vx != 0) {
      const uint32_t vn = (math_clz_nonzero_u32(vx) | 1) ^ 31;
      uint32_t vb = UINT32_C(1) << vn;
      uint32_t iterations = (vn >> 1) + 1;
      while (iterations--) {
        const uint32_t vyb = vy + vb;
        if (vx >= vyb) {
          vx -= vyb;
          vy = (vy >> 1) + vb;
        } else {
          vy >>= 1;
        }
        vb >>= 2;
      }

      // vy is sqrt(.) rounded down. Do the final rounding up if needed.
      if (vx > vy) {
        // This condition prevents overflowing uint16_t, but produces incorrectly
        // rounded result for large inputs where square root should round to 0x10000.
        if (vy != UINT32_C(0xFFFF)) {
          vy += 1;
        }
      }
    }

    *output++ = vy;
  }
}
