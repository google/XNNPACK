// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <math.h>

#include <xnnpack/common.h>
#include <xnnpack/math-stubs.h>


void xnn_math_u32_sqrt__scalar_cvti64_sqrtf_lrintf(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n % sizeof(uint32_t) == 0);

  for (; n != 0; n -= sizeof(uint32_t)) {
    const uint32_t vx = *input++;

    uint32_t vy = vx;
    if XNN_LIKELY(vx != 0) {
      float vf = (float) (double) (int64_t) (uint64_t) vx;
      vf = sqrtf(vf);
      vy = (uint32_t) (int32_t) lrintf(vf);
      const uint32_t vsquared_y_less_x = vy * vy - vx;
      if XNN_UNPREDICTABLE((int32_t) (vsquared_y_less_x + vy) < 0) {
        vy += 1;
      } else if XNN_UNPREDICTABLE((int32_t) (vsquared_y_less_x - vy) >= 0) {
        vy -= 1;
      }
    }

    *output++ = vy;
  }
}
