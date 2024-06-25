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


void xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_5x1(
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
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_width);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + input_width);
  float* o2 = (float*) ((uintptr_t) o1 + input_width);
  float* o3 = (float*) ((uintptr_t) o2 + input_width);
  float* o4 = (float*) ((uintptr_t) o3 + input_width);

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i2 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(output_height < 3) {
      i3 = zero;
      o2 = o1;
    }
    if XNN_UNPREDICTABLE(output_height < 4) {
      i4 = zero;
      o3 = o2;
    }
    if XNN_UNPREDICTABLE(output_height < 5) {
      i5 = zero;
      o4 = o3;
    }
    if XNN_UNPREDICTABLE(output_height < 6) {
      i6 = zero;
    }

    float vi0x0 = 0.0f;
    float vi1x0 = 0.0f;
    float vi2x0 = 0.0f;
    float vi3x0 = 0.0f;
    float vi4x0 = 0.0f;
    float vi5x0 = 0.0f;
    float vi6x0 = 0.0f;

    float vi0x1 = *i0++;
    float vi1x1 = *i1++;
    float vi2x1 = *i2++;
    float vi3x1 = *i3++;
    float vi4x1 = *i4++;
    float vi5x1 = *i5++;
    float vi6x1 = *i6++;

    size_t w = input_width;
    for (; w > 1 * sizeof(float); w -= 1 * sizeof(float)) {
      const float vi0x2 = *i0++;
      const float vi1x2 = *i1++;
      const float vi2x2 = *i2++;
      const float vi3x2 = *i3++;
      const float vi4x2 = *i4++;
      const float vi5x2 = *i5++;
      const float vi6x2 = *i6++;

      float vo0p0 = vbias + vi0x0 * vk00;
      float vo1p0 = vbias + vi1x0 * vk00;
      float vo2p0 = vbias + vi2x0 * vk00;
      float vo3p0 = vbias + vi3x0 * vk00;
      float vo4p0 = vbias + vi4x0 * vk00;
      vo0p0 += vi1x0 * vk10;
      vo1p0 += vi2x0 * vk10;
      vo2p0 += vi3x0 * vk10;
      vo3p0 += vi4x0 * vk10;
      vo4p0 += vi5x0 * vk10;
      vo0p0 += vi2x0 * vk20;
      vo1p0 += vi3x0 * vk20;
      vo2p0 += vi4x0 * vk20;
      vo3p0 += vi5x0 * vk20;
      vo4p0 += vi6x0 * vk20;

      vi0x0 = vi0x1;
      vi1x0 = vi1x1;
      vi2x0 = vi2x1;
      vi3x0 = vi3x1;
      vi4x0 = vi4x1;
      vi5x0 = vi5x1;
      vi6x0 = vi6x1;

      vo0p0 += vi0x1 * vk01;
      vo1p0 += vi1x1 * vk01;
      vo2p0 += vi2x1 * vk01;
      vo3p0 += vi3x1 * vk01;
      vo4p0 += vi4x1 * vk01;
      vo0p0 += vi1x1 * vk11;
      vo1p0 += vi2x1 * vk11;
      vo2p0 += vi3x1 * vk11;
      vo3p0 += vi4x1 * vk11;
      vo4p0 += vi5x1 * vk11;
      vo0p0 += vi2x1 * vk21;
      vo1p0 += vi3x1 * vk21;
      vo2p0 += vi4x1 * vk21;
      vo3p0 += vi5x1 * vk21;
      vo4p0 += vi6x1 * vk21;

      vi0x1 = vi0x2;
      vi1x1 = vi1x2;
      vi2x1 = vi2x2;
      vi3x1 = vi3x2;
      vi4x1 = vi4x2;
      vi5x1 = vi5x2;
      vi6x1 = vi6x2;

      vo0p0 += vi0x2 * vk02;
      vo1p0 += vi1x2 * vk02;
      vo2p0 += vi2x2 * vk02;
      vo3p0 += vi3x2 * vk02;
      vo4p0 += vi4x2 * vk02;
      vo0p0 += vi1x2 * vk12;
      vo1p0 += vi2x2 * vk12;
      vo2p0 += vi3x2 * vk12;
      vo3p0 += vi4x2 * vk12;
      vo4p0 += vi5x2 * vk12;
      vo0p0 += vi2x2 * vk22;
      vo1p0 += vi3x2 * vk22;
      vo2p0 += vi4x2 * vk22;
      vo3p0 += vi5x2 * vk22;
      vo4p0 += vi6x2 * vk22;


      float vo0 = math_max_f32(vo0p0, vmin);
      float vo1 = math_max_f32(vo1p0, vmin);
      float vo2 = math_max_f32(vo2p0, vmin);
      float vo3 = math_max_f32(vo3p0, vmin);
      float vo4 = math_max_f32(vo4p0, vmin);

      vo0 = math_min_f32(vo0, vmax);
      vo1 = math_min_f32(vo1, vmax);
      vo2 = math_min_f32(vo2, vmax);
      vo3 = math_min_f32(vo3, vmax);
      vo4 = math_min_f32(vo4, vmax);

      *o4++ = vo4;
      *o3++ = vo3;
      *o2++ = vo2;
      *o1++ = vo1;
      *o0++ = vo0;
    }
    // Always process the last pixel separately to account for right edge.
    assert(w == 1 * sizeof(float));
    {
      float vo0p0 = vbias + vi0x0 * vk00;
      float vo1p0 = vbias + vi1x0 * vk00;
      float vo2p0 = vbias + vi2x0 * vk00;
      float vo3p0 = vbias + vi3x0 * vk00;
      float vo4p0 = vbias + vi4x0 * vk00;
      vo0p0 += vi1x0 * vk10;
      vo1p0 += vi2x0 * vk10;
      vo2p0 += vi3x0 * vk10;
      vo3p0 += vi4x0 * vk10;
      vo4p0 += vi5x0 * vk10;
      vo0p0 += vi2x0 * vk20;
      vo1p0 += vi3x0 * vk20;
      vo2p0 += vi4x0 * vk20;
      vo3p0 += vi5x0 * vk20;
      vo4p0 += vi6x0 * vk20;

      vo0p0 += vi0x1 * vk01;
      vo1p0 += vi1x1 * vk01;
      vo2p0 += vi2x1 * vk01;
      vo3p0 += vi3x1 * vk01;
      vo4p0 += vi4x1 * vk01;
      vo0p0 += vi1x1 * vk11;
      vo1p0 += vi2x1 * vk11;
      vo2p0 += vi3x1 * vk11;
      vo3p0 += vi4x1 * vk11;
      vo4p0 += vi5x1 * vk11;
      vo0p0 += vi2x1 * vk21;
      vo1p0 += vi3x1 * vk21;
      vo2p0 += vi4x1 * vk21;
      vo3p0 += vi5x1 * vk21;
      vo4p0 += vi6x1 * vk21;


      float vo0 = math_max_f32(vo0p0, vmin);
      float vo1 = math_max_f32(vo1p0, vmin);
      float vo2 = math_max_f32(vo2p0, vmin);
      float vo3 = math_max_f32(vo3p0, vmin);
      float vo4 = math_max_f32(vo4p0, vmin);

      vo0 = math_min_f32(vo0, vmax);
      vo1 = math_min_f32(vo1, vmax);
      vo2 = math_min_f32(vo2, vmax);
      vo3 = math_min_f32(vo3, vmax);
      vo4 = math_min_f32(vo4, vmax);

      *o4++ = vo4;
      *o3++ = vo3;
      *o2++ = vo2;
      *o1++ = vo1;
      *o0++ = vo0;
    }

    i0 = (const float*) ((uintptr_t) i5 - input_width);
    i1 = (const float*) ((uintptr_t) i6 - input_width);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);
    i5 = (const float*) ((uintptr_t) i4 + input_width);
    i6 = (const float*) ((uintptr_t) i5 + input_width);

    o0 = o4;
    o1 = (float*) ((uintptr_t) o0 + input_width);
    o2 = (float*) ((uintptr_t) o1 + input_width);
    o3 = (float*) ((uintptr_t) o2 + input_width);
    o4 = (float*) ((uintptr_t) o3 + input_width);

    output_height = doz(output_height, 5);
  } while (output_height != 0);
}
