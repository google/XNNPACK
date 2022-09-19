// Auto-generated file. Do not edit!
//   Template: src/f32-vhswish/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/vunary.h>


void xnn_f32_vhswish_ukernel__scalar_x1(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_hswish_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vsixth = params->scalar.sixth;
  const float vthree = params->scalar.three;
  const float vsix = params->scalar.six;
  const float vzero = 0.0f;
  assert(vthree == 3.0f);
  assert(vsix == 6.0f);

  for (; batch >= sizeof(float); batch -= sizeof(float)) {
    float vx = *input++;
    float vacc = vx + vthree;
    vx *= vsixth;
    vacc = math_max_f32(vacc, vzero);
    vacc = math_min_f32(vacc, vsix);
    vacc *= vx;
    *output++ = vacc;
  }
}
