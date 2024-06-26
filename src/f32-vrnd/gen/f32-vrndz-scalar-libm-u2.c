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

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/vunary.h"


void xnn_f32_vrndz_ukernel__scalar_libm_u2(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    input += 2;

    const float vy0 = truncf(vx0);
    const float vy1 = truncf(vx1);

    output[0] = vy0;
    output[1] = vy1;
    output += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float vx = *input;
    const float vy = truncf(vx);
    *output = vy;
  }
}
