// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv2d-chw/3x3s2p1-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"


void xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_2x1_acc2(
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
  assert(padding_top >= 0);
  assert(padding_top <= 1);

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

  const size_t output_width = round_down_po2((input_width + (2 /* padding */ - 3 /* kernel size */ + 2 /* subsampling */) * sizeof(float)) / 2, sizeof(float));

  const float* i0 = (const float*) ((uintptr_t) input - ((-padding_top) & input_width));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_width);
  if XNN_UNPREDICTABLE(padding_top != 0) {
    i0 = zero;
  }
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + output_width);

  size_t padded_input_height = input_height + padding_top + 1 /* padding bottom */;
  size_t output_height = (padded_input_height - 3 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 4) {
      i2 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 5) {
      i3 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 6) {
      i4 = zero;
    }

    float vi0x0 = 0.0f;
    float vi1x0 = 0.0f;
    float vi2x0 = 0.0f;
    float vi3x0 = 0.0f;
    float vi4x0 = 0.0f;

    size_t w = input_width;
    for (; w >= 2 * sizeof(float); w -= 2 * sizeof(float)) {
      const float vi0x1 = i0[0];
      const float vi1x1 = i1[0];
      const float vi2x1 = i2[0];
      const float vi3x1 = i3[0];
      const float vi4x1 = i4[0];

      float vo0p0 = vbias + vi0x0 * vk00;
      float vo1p0 = vbias + vi2x0 * vk00;
      float vo0p1 = vi1x0 * vk10;
      float vo1p1 = vi3x0 * vk10;
      vo0p0 += vi2x0 * vk20;
      vo1p0 += vi4x0 * vk20;

      const float vi0x2 = i0[1];
      i0 += 2;
      const float vi1x2 = i1[1];
      i1 += 2;
      const float vi2x2 = i2[1];
      i2 += 2;
      const float vi3x2 = i3[1];
      i3 += 2;
      const float vi4x2 = i4[1];
      i4 += 2;

      vo0p1 += vi0x1 * vk01;
      vo1p1 += vi2x1 * vk01;
      vo0p0 += vi1x1 * vk11;
      vo1p0 += vi3x1 * vk11;
      vo0p1 += vi2x1 * vk21;
      vo1p1 += vi4x1 * vk21;

      vi0x0 = vi0x2;
      vi1x0 = vi1x2;
      vi2x0 = vi2x2;
      vi3x0 = vi3x2;
      vi4x0 = vi4x2;

      vo0p0 += vi0x2 * vk02;
      vo1p0 += vi2x2 * vk02;
      vo0p1 += vi1x2 * vk12;
      vo1p1 += vi3x2 * vk12;
      vo0p0 += vi2x2 * vk22;
      vo1p0 += vi4x2 * vk22;

      vo0p0 += vo0p1;
      vo1p0 += vo1p1;

      float vo0 = math_max_f32(vo0p0, vmin);
      float vo1 = math_max_f32(vo1p0, vmin);

      vo0 = math_min_f32(vo0, vmax);
      vo1 = math_min_f32(vo1, vmax);

      *o1++ = vo1;
      *o0++ = vo0;
    }
    // Potentially process the last pixel.
    assert(w <= 1 * sizeof(float));
    if (w != 0) {
      const float vi0x1 = *i0++;
      const float vi1x1 = *i1++;
      const float vi2x1 = *i2++;
      const float vi3x1 = *i3++;
      const float vi4x1 = *i4++;

      float vo0p0 = vbias + vi0x0 * vk00;
      float vo1p0 = vbias + vi2x0 * vk00;
      float vo0p1 = vi1x0 * vk10;
      float vo1p1 = vi3x0 * vk10;
      vo0p0 += vi2x0 * vk20;
      vo1p0 += vi4x0 * vk20;

      vo0p1 += vi0x1 * vk01;
      vo1p1 += vi2x1 * vk01;
      vo0p0 += vi1x1 * vk11;
      vo1p0 += vi3x1 * vk11;
      vo0p1 += vi2x1 * vk21;
      vo1p1 += vi4x1 * vk21;

      vo0p0 += vo0p1;
      vo1p0 += vo1p1;

      float vo0 = math_max_f32(vo0p0, vmin);
      float vo1 = math_max_f32(vo1p0, vmin);

      vo0 = math_min_f32(vo0, vmax);
      vo1 = math_min_f32(vo1, vmax);

      *o1++ = vo1;
      *o0++ = vo0;
    }

    i0 = (const float*) ((uintptr_t) i3);
    i1 = (const float*) ((uintptr_t) i4);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);

    o0 = o1;
    o1 = (float*) ((uintptr_t) o0 + output_width);

    output_height = doz(output_height, 2);
    padded_input_height = doz(padded_input_height, 4);
  } while (output_height != 0);
}
