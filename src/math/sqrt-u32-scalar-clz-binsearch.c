// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_u32_sqrt__scalar_clz_binsearch(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n % sizeof(uint32_t) == 0);

  for (; n != 0; n -= sizeof(uint32_t)) {
    const uint32_t vx = *input++;

    // Based on Hacker's Delight, Figure 11-3.
    uint32_t vb = (UINT32_C(1) << ((33 - math_clz_u32(vx)) / 2)) - 1;
    uint32_t va = (vb + 3) / 2;
    do {
      const uint32_t vm = (va + vb) >> 1;
      assert(vm <= UINT32_C(65535));
      if XNN_UNPREDICTABLE(vm * vm > vx) {
        vb = vm - 1;
      } else {
        va = vm + 1;
      }
    } while XNN_LIKELY(vb >= va);

    uint32_t vy = va - 1;
    // vy is sqrt(vx) rounded down. Do the final rounding up if needed.
    if XNN_UNPREDICTABLE(va * vy < vx) {
      vy += 1;
    }

    *output++ = vy;
  }
}
