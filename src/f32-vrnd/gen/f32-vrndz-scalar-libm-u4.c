// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vrnd/scalar-libm.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f32_vrndz_ukernel__scalar_libm_u4(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    const float vx2 = input[2];
    const float vx3 = input[3];
    input += 4;

    const float vy0 = truncf(vx0);
    const float vy1 = truncf(vx1);
    const float vy2 = truncf(vx2);
    const float vy3 = truncf(vx3);

    output[0] = vy0;
    output[1] = vy1;
    output[2] = vy2;
    output[3] = vy3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vx = *input++;
      const float vy = truncf(vx);
      *output++ = vy;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}
