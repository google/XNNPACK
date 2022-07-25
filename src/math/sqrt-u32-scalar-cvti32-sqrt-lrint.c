// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <math.h>

#include <xnnpack/math-stubs.h>


void xnn_math_u32_sqrt__scalar_cvti32_sqrt_lrint(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n % sizeof(uint32_t) == 0);

  for (; n != 0; n -= sizeof(uint32_t)) {
    const uint32_t vx = *input++;

    const int32_t vm = (int32_t) (vx ^ UINT32_C(0x80000000));
    double vf = (double) vm + 0x1.0p+31;
    vf = sqrt(vf);
    const uint32_t vy = (uint32_t) (int32_t) lrint(vf);

    *output++ = vy;
  }
}
