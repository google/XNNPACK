// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_u32_sqrt__scalar_hashemian(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n % sizeof(uint32_t) == 0);

  for (; n != 0; n -= sizeof(uint32_t)) {
    const uint32_t vx = *input++;

    uint32_t vy = vx;
    if (vx != 0) {
      /*
       * Based on "Square Rooting Algorithms for Integer and Floating-Point Numbers" by Reza Hashemian
       * and StackOverflow answer https://stackoverflow.com/a/31149161
      */

      const uint32_t vn = math_clz_nonzero_u32(vx);
      const uint32_t vleft_shift = vn & 1;
      const uint32_t vm_minus_1 = 15 - (vn >> 1);
      const uint32_t vm_plus_1 = vm_minus_1 + 2;
      const uint32_t vexp2_m_minus_1 = UINT32_C(1) << vm_minus_1;
      const uint32_t vz = vexp2_m_minus_1 - (vx >> (vm_plus_1 - vleft_shift));

      vy = vz;
      // Iterate until y[i] == y[i-1]. Alternatively, we can do 7 iterations:
      //   for (uint32_t i = 0; i < 7; i++) {
      //     vy = vz + ((vy * vy) >> vm_plus_1);
      //   }
      uint32_t vy_prev;
      do {
        vy_prev = vy;
        vy = vz + ((vy * vy) >> vm_plus_1);
      } while (vy != vy_prev);

      // Reconstruct Y = 2**m - vy
      vy = (vexp2_m_minus_1 << 1) - vy;
      if XNN_UNPREDICTABLE(vleft_shift) {
        // Multiply by sqrt(0.5) by subtracting vy * (1 - sqrt(0.5)), 1 - sqrt(0.5) is represented
        // as a .16 fixed-point number to guarantee than the product doesn't overflow 32 bits.
        // Using 1 - sqrt(0.5) under these constraints is 1 bit more accurate than using sqrt(0.5) directly.
        vy -= (vy * UINT32_C(19195)) >> 16;
      }

      // When X has an even number of bits, Y can overestimate isqrt(X) by 1 due to truncations in fixed-point
      // arithmetics. When X has an odd number of bits, Y can overestimate isqrt(X) by an extra 1 (2 total) due to
      // truncation in the multiplication by sqrt(0.5).
      // We decrement Y once if X < Y * Y and decrement it once again if Y * Y - X > X - (Y - 1) * (Y - 1).
      uint32_t vsquared_y = vy * vy;
      if XNN_UNPREDICTABLE(vsquared_y > vx) {
        vsquared_y -= 2 * vy - 1;
        vy -= 1;
      }

      // Y is within a distance of 1 from properly rounded sqrt(X).
      // - Increment Y if (Y + 1) * (Y + 1) - X < X - Y * Y.
      // - Decrement Y if Y * Y - X > X - (Y - 1) * (Y - 1).
      // The increment + decrement are combined together to re-use the (Y * Y) value.
      if XNN_UNPREDICTABLE(vsquared_y < vx - vy) {
        vy += 1;
      } else if XNN_UNPREDICTABLE(vsquared_y - vy >= vx) {
        vy -= 1;
      }
    }

    *output++ = vy;
  }
}
