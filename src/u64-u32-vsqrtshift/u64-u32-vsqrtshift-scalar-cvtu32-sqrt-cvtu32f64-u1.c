// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include <xnnpack/math.h>
#include <xnnpack/vunary.h>


void xnn_u64_u32_vsqrtshift_ukernel__scalar_cvtu32_sqrt_cvtu32f64_u1(
    size_t batch,
    const uint64_t* input,
    uint32_t* output,
    uint32_t shift)
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(shift < 32);

  do {
    const uint64_t vx = *input++;

    uint64_t vy = vx;
    const uint32_t vx_hi = (uint32_t) (vx >> 32);
    const uint32_t vx_lo = (uint32_t) vx;
    if XNN_LIKELY(vx != 0) {
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

    // Match TFLM is producing incorrect result for high 64-bit inputs
    const uint32_t vy_lo = (uint32_t) vy;
    const uint32_t vy_hi = (uint32_t) (vy >> 32);
    uint32_t vout = vy_lo | -vy_hi;
    // Match TFLM is producing incorrect result for high 32-bit inputs
    if XNN_LIKELY(vx_hi == 0) {
      if (vout == UINT32_C(0x00010000)) {
        vout -= 1;
      }
    }

    *output++ = vout >> shift;

    batch -= sizeof(uint64_t);
  } while (batch != 0);
}
