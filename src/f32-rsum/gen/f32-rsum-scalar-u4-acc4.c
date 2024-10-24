// Auto-generated file. Do not edit!
//   Template: src/f32-rsum/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"


void xnn_f32_rsum_ukernel__scalar_u4_acc4(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_scale_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  float vacc0 = 0.0f;
  float vacc1 = 0.0f;
  float vacc2 = 0.0f;
  float vacc3 = 0.0f;
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float vt0 = input[0];
    const float vt1 = input[1];
    const float vt2 = input[2];
    const float vt3 = input[3];
    input += 4;

    vacc0 += vt0;
    vacc1 += vt1;
    vacc2 += vt2;
    vacc3 += vt3;
  }
  vacc0 += vacc1;
  vacc2 += vacc3;
  vacc0 += vacc2;

  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vt = *input++;
      vacc0 += vt;
      batch -= sizeof(float);
    } while (batch != 0);
  }
  const float vscale = params->scalar.scale;
  vacc0 *= vscale;
  *output += vacc0;
}
