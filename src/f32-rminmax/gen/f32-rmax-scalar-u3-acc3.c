// Auto-generated file. Do not edit!
//   Template: src/f32-rminmax/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/reduce.h"

void xnn_f32_rmax_ukernel__scalar_u3_acc3(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float* i = input;

  float vmax0 = *i;
  float vmax1 = vmax0;
  float vmax2 = vmax0;
  for (; batch >= 3 * sizeof(float); batch -= 3 * sizeof(float)) {
    const float vt0 = i[0];
    const float vt1 = i[1];
    const float vt2 = i[2];
    i += 3;

    vmax0 = math_max_f32(vmax0, vt0);
    vmax1 = math_max_f32(vmax1, vt1);
    vmax2 = math_max_f32(vmax2, vt2);
  }
  vmax0 = math_max_f32(vmax0, vmax1);
  vmax0 = math_max_f32(vmax0, vmax2);

  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vt = *i++;
      vmax0 = math_max_f32(vmax0, vt);
      batch -= sizeof(float);
    } while (batch != 0);
  }
  output[0] = vmax0;
}
