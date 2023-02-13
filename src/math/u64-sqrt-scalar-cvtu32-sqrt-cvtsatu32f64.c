// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <math.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_u64_sqrt__scalar_cvtu32_sqrt_cvtsatu32f64(
    size_t n,
    const uint64_t* input,
    uint64_t* output)
{
  assert(n % sizeof(uint32_t) == 0);

  for (; n != 0; n -= sizeof(uint64_t)) {
    const uint64_t vx = *input++;

    uint64_t vy = vx;
    if XNN_LIKELY(vx != 0) {
      const uint32_t vx_lo = (uint32_t) vx;
      const uint32_t vx_hi = (uint32_t) (vx >> 32);
      const double vf_hi = (double) vx_hi;
      const double vf_lo = (double) vx_lo;
      double vf = vf_hi * 0x1.0p+32 + vf_lo;
      vf = sqrt(vf);
      vy = math_cvt_sat_u32_f64(vf);
      #if XNN_ARCH_ARM || XNN_ARCH_X86
        const uint64_t vsquared_y_less_x = math_mulext_u32((uint32_t) vy, (uint32_t) vy) - vx;
      #else
        const uint64_t vsquared_y_less_x = vy * vy - vx;
      #endif
      if XNN_UNPREDICTABLE((int64_t) (vsquared_y_less_x + vy) < 0) {
        vy += 1;
      } else if XNN_UNPREDICTABLE((int64_t) (vsquared_y_less_x - vy) >= 0) {
        vy -= 1;
      }
    }

    *output++ = vy;
  }
}
