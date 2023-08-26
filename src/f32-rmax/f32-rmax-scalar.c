// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/math.h>
#include <xnnpack/rmax.h>


void xnn_f32_rmax_ukernel__scalar(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  float vmax0 = *input;
  float vmax1 = vmax0;
  float vmax2 = vmax0;
  float vmax3 = vmax0;
  for (; batch >= 16; batch -= 16) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    const float vx2 = input[2];
    const float vx3 = input[3];
    input += 4;

    vmax0 = math_max_f32(vx0, vmax0);
    vmax1 = math_max_f32(vx1, vmax1);
    vmax2 = math_max_f32(vx2, vmax2);
    vmax3 = math_max_f32(vx3, vmax3);
  }
  const float vmax01 = math_max_f32(vmax0, vmax1);
  const float vmax23 = math_max_f32(vmax2, vmax3);
  float vmax = math_max_f32(vmax01, vmax23);
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vx = *input++;
      vmax = math_max_f32(vx, vmax);
      batch -= 4;
    } while (batch != 0);
  }
  *output = vmax;
}
