// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv2d-chw/3x3p1-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"


void xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_1x1_acc2(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top == 1);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

  const float vbias = weights[0];
  const float vk00 = weights[1];
  const float vk01 = weights[2];
  const float vk02 = weights[3];
  const float vk10 = weights[4];
  const float vk11 = weights[5];
  const float vk12 = weights[6];
  const float vk20 = weights[7];
  const float vk21 = weights[8];
  const float vk22 = weights[9];

  const float* i0 = zero;
  const float* i1 = input;
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);

  float* o0 = output;

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i2 = zero;
    }

    float vi0x0 = 0.0f;
    float vi1x0 = 0.0f;
    float vi2x0 = 0.0f;

    float vi0x1 = *i0++;
    float vi1x1 = *i1++;
    float vi2x1 = *i2++;

    size_t w = input_width;
    for (; w > 1 * sizeof(float); w -= 1 * sizeof(float)) {
      const float vi0x2 = *i0++;
      const float vi1x2 = *i1++;
      const float vi2x2 = *i2++;

      float vo0p0 = vbias + vi0x0 * vk00;
      float vo0p1 = vi1x0 * vk10;
      vo0p0 += vi2x0 * vk20;

      vi0x0 = vi0x1;
      vi1x0 = vi1x1;
      vi2x0 = vi2x1;

      vo0p1 += vi0x1 * vk01;
      vo0p0 += vi1x1 * vk11;
      vo0p1 += vi2x1 * vk21;

      vi0x1 = vi0x2;
      vi1x1 = vi1x2;
      vi2x1 = vi2x2;

      vo0p0 += vi0x2 * vk02;
      vo0p1 += vi1x2 * vk12;
      vo0p0 += vi2x2 * vk22;

      vo0p0 += vo0p1;

      float vo0 = math_max_f32(vo0p0, vmin);

      vo0 = math_min_f32(vo0, vmax);

      *o0++ = vo0;
    }
    // Always process the last pixel separately to account for right edge.
    assert(w == 1 * sizeof(float));
    {
      float vo0p0 = vbias + vi0x0 * vk00;
      float vo0p1 = vi1x0 * vk10;
      vo0p0 += vi2x0 * vk20;

      vo0p1 += vi0x1 * vk01;
      vo0p0 += vi1x1 * vk11;
      vo0p1 += vi2x1 * vk21;

      vo0p0 += vo0p1;

      float vo0 = math_max_f32(vo0p0, vmin);

      vo0 = math_min_f32(vo0, vmax);

      *o0++ = vo0;
    }

    i0 = (const float*) ((uintptr_t) i1 - input_width);


  } while (--output_height != 0);
}
