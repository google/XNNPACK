// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv2d-chw/5x5s2p2-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"


void xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_3x1_acc2(
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
  assert(padding_top >= 1);
  assert(padding_top <= 2);

  const float vmax = params->scalar.max;
  const float vmin = params->scalar.min;

  const float vbias = weights[0];
  const float vk00 = weights[1];
  const float vk01 = weights[2];
  const float vk02 = weights[3];
  const float vk03 = weights[4];
  const float vk04 = weights[5];
  const float vk10 = weights[6];
  const float vk11 = weights[7];
  const float vk12 = weights[8];
  const float vk13 = weights[9];
  const float vk14 = weights[10];
  const float vk20 = weights[11];
  const float vk21 = weights[12];
  const float vk22 = weights[13];
  const float vk23 = weights[14];
  const float vk24 = weights[15];
  const float vk30 = weights[16];
  const float vk31 = weights[17];
  const float vk32 = weights[18];
  const float vk33 = weights[19];
  const float vk34 = weights[20];
  const float vk40 = weights[21];
  const float vk41 = weights[22];
  const float vk42 = weights[23];
  const float vk43 = weights[24];
  const float vk44 = weights[25];

  const uint32_t padding_top_less_1 = padding_top - 1;

  const float* i0 = zero;
  const float* i1 = (const float*) ((uintptr_t) input - ((-padding_top_less_1) & input_width));
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
  if XNN_UNPREDICTABLE(padding_top_less_1 != 0) {
    i1 = zero;
  }
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_width);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_width);
  const float* i7 = (const float*) ((uintptr_t) i6 + input_width);
  const float* i8 = (const float*) ((uintptr_t) i7 + input_width);

  const size_t output_width = round_down_po2((input_width + (2 /* padding */ - 3 /* kernel size */ + 2 /* subsampling */) * sizeof(float)) / 2, sizeof(float));

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + output_width);
  float* o2 = (float*) ((uintptr_t) o1 + output_width);

  size_t padded_input_height = input_height + (padding_top_less_1 + 1) + 2 /* padding bottom */;
  size_t output_height = (padded_input_height - 5 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 6) {
      i3 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 7) {
      i4 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 8) {
      i5 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 9) {
      i6 = zero;
      o2 = o1;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 10) {
      i7 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 11) {
      i8 = zero;
    }

    float vi0x0 = 0.0f;
    float vi1x0 = 0.0f;
    float vi2x0 = 0.0f;
    float vi3x0 = 0.0f;
    float vi4x0 = 0.0f;
    float vi5x0 = 0.0f;
    float vi6x0 = 0.0f;
    float vi7x0 = 0.0f;
    float vi8x0 = 0.0f;

    float vi0x1 = 0.0f;
    float vi1x1 = 0.0f;
    float vi2x1 = 0.0f;
    float vi3x1 = 0.0f;
    float vi4x1 = 0.0f;
    float vi5x1 = 0.0f;
    float vi6x1 = 0.0f;
    float vi7x1 = 0.0f;
    float vi8x1 = 0.0f;

    float vi0x2 = *i0++;
    float vi1x2 = *i1++;
    float vi2x2 = *i2++;
    float vi3x2 = *i3++;
    float vi4x2 = *i4++;
    float vi5x2 = *i5++;
    float vi6x2 = *i6++;
    float vi7x2 = *i7++;
    float vi8x2 = *i8++;

    size_t w = input_width;
    for (; w > 2 * sizeof(float); w -= 2 * sizeof(float)) {
      const float vi0x3 = i0[0];
      const float vi1x3 = i1[0];
      const float vi2x3 = i2[0];
      const float vi3x3 = i3[0];
      const float vi4x3 = i4[0];
      const float vi5x3 = i5[0];
      const float vi6x3 = i6[0];
      const float vi7x3 = i7[0];
      const float vi8x3 = i8[0];

      const float vi0x4 = i0[1];
      i0 += 2;
      const float vi1x4 = i1[1];
      i1 += 2;
      const float vi2x4 = i2[1];
      i2 += 2;
      const float vi3x4 = i3[1];
      i3 += 2;
      const float vi4x4 = i4[1];
      i4 += 2;
      const float vi5x4 = i5[1];
      i5 += 2;
      const float vi6x4 = i6[1];
      i6 += 2;
      const float vi7x4 = i7[1];
      i7 += 2;
      const float vi8x4 = i8[1];
      i8 += 2;

      float vo0p0 = vbias + vi0x0 * vk00;
      float vo1p0 = vbias + vi2x0 * vk00;
      float vo2p0 = vbias + vi4x0 * vk00;
      float vo0p1 = vi1x0 * vk10;
      float vo1p1 = vi3x0 * vk10;
      float vo2p1 = vi5x0 * vk10;
      vo0p0 += vi2x0 * vk20;
      vo1p0 += vi4x0 * vk20;
      vo2p0 += vi6x0 * vk20;
      vo0p1 += vi3x0 * vk30;
      vo1p1 += vi5x0 * vk30;
      vo2p1 += vi7x0 * vk30;
      vo0p0 += vi4x0 * vk40;
      vo1p0 += vi6x0 * vk40;
      vo2p0 += vi8x0 * vk40;

      vi0x0 = vi0x2;
      vi1x0 = vi1x2;
      vi2x0 = vi2x2;
      vi3x0 = vi3x2;
      vi4x0 = vi4x2;
      vi5x0 = vi5x2;
      vi6x0 = vi6x2;
      vi7x0 = vi7x2;
      vi8x0 = vi8x2;

      vo0p1 += vi0x1 * vk01;
      vo1p1 += vi2x1 * vk01;
      vo2p1 += vi4x1 * vk01;
      vo0p0 += vi1x1 * vk11;
      vo1p0 += vi3x1 * vk11;
      vo2p0 += vi5x1 * vk11;
      vo0p1 += vi2x1 * vk21;
      vo1p1 += vi4x1 * vk21;
      vo2p1 += vi6x1 * vk21;
      vo0p0 += vi3x1 * vk31;
      vo1p0 += vi5x1 * vk31;
      vo2p0 += vi7x1 * vk31;
      vo0p1 += vi4x1 * vk41;
      vo1p1 += vi6x1 * vk41;
      vo2p1 += vi8x1 * vk41;

      vi0x1 = vi0x3;
      vi1x1 = vi1x3;
      vi2x1 = vi2x3;
      vi3x1 = vi3x3;
      vi4x1 = vi4x3;
      vi5x1 = vi5x3;
      vi6x1 = vi6x3;
      vi7x1 = vi7x3;
      vi8x1 = vi8x3;

      vo0p0 += vi0x2 * vk02;
      vo1p0 += vi2x2 * vk02;
      vo2p0 += vi4x2 * vk02;
      vo0p1 += vi1x2 * vk12;
      vo1p1 += vi3x2 * vk12;
      vo2p1 += vi5x2 * vk12;
      vo0p0 += vi2x2 * vk22;
      vo1p0 += vi4x2 * vk22;
      vo2p0 += vi6x2 * vk22;
      vo0p1 += vi3x2 * vk32;
      vo1p1 += vi5x2 * vk32;
      vo2p1 += vi7x2 * vk32;
      vo0p0 += vi4x2 * vk42;
      vo1p0 += vi6x2 * vk42;
      vo2p0 += vi8x2 * vk42;

      vi0x2 = vi0x4;
      vi1x2 = vi1x4;
      vi2x2 = vi2x4;
      vi3x2 = vi3x4;
      vi4x2 = vi4x4;
      vi5x2 = vi5x4;
      vi6x2 = vi6x4;
      vi7x2 = vi7x4;
      vi8x2 = vi8x4;

      vo0p1 += vi0x3 * vk03;
      vo1p1 += vi2x3 * vk03;
      vo2p1 += vi4x3 * vk03;
      vo0p0 += vi1x3 * vk13;
      vo1p0 += vi3x3 * vk13;
      vo2p0 += vi5x3 * vk13;
      vo0p1 += vi2x3 * vk23;
      vo1p1 += vi4x3 * vk23;
      vo2p1 += vi6x3 * vk23;
      vo0p0 += vi3x3 * vk33;
      vo1p0 += vi5x3 * vk33;
      vo2p0 += vi7x3 * vk33;
      vo0p1 += vi4x3 * vk43;
      vo1p1 += vi6x3 * vk43;
      vo2p1 += vi8x3 * vk43;

      vo0p0 += vi0x4 * vk04;
      vo1p0 += vi2x4 * vk04;
      vo2p0 += vi4x4 * vk04;
      vo0p1 += vi1x4 * vk14;
      vo1p1 += vi3x4 * vk14;
      vo2p1 += vi5x4 * vk14;
      vo0p0 += vi2x4 * vk24;
      vo1p0 += vi4x4 * vk24;
      vo2p0 += vi6x4 * vk24;
      vo0p1 += vi3x4 * vk34;
      vo1p1 += vi5x4 * vk34;
      vo2p1 += vi7x4 * vk34;
      vo0p0 += vi4x4 * vk44;
      vo1p0 += vi6x4 * vk44;
      vo2p0 += vi8x4 * vk44;

      vo0p0 += vo0p1;
      vo1p0 += vo1p1;
      vo2p0 += vo2p1;

      float vo0 = math_max_f32(vo0p0, vmin);
      float vo1 = math_max_f32(vo1p0, vmin);
      float vo2 = math_max_f32(vo2p0, vmin);

      vo0 = math_min_f32(vo0, vmax);
      vo1 = math_min_f32(vo1, vmax);
      vo2 = math_min_f32(vo2, vmax);

      *o2++ = vo2;
      *o1++ = vo1;
      *o0++ = vo0;
    }
    if XNN_LIKELY(w == 2 * sizeof(float)) {
      const float vi0x3 = *i0++;
      const float vi1x3 = *i1++;
      const float vi2x3 = *i2++;
      const float vi3x3 = *i3++;
      const float vi4x3 = *i4++;
      const float vi5x3 = *i5++;
      const float vi6x3 = *i6++;
      const float vi7x3 = *i7++;
      const float vi8x3 = *i8++;

      float vo0p0 = vbias + vi0x0 * vk00;
      float vo1p0 = vbias + vi2x0 * vk00;
      float vo2p0 = vbias + vi4x0 * vk00;
      float vo0p1 = vi1x0 * vk10;
      float vo1p1 = vi3x0 * vk10;
      float vo2p1 = vi5x0 * vk10;
      vo0p0 += vi2x0 * vk20;
      vo1p0 += vi4x0 * vk20;
      vo2p0 += vi6x0 * vk20;
      vo0p1 += vi3x0 * vk30;
      vo1p1 += vi5x0 * vk30;
      vo2p1 += vi7x0 * vk30;
      vo0p0 += vi4x0 * vk40;
      vo1p0 += vi6x0 * vk40;
      vo2p0 += vi8x0 * vk40;

      vo0p1 += vi0x1 * vk01;
      vo1p1 += vi2x1 * vk01;
      vo2p1 += vi4x1 * vk01;
      vo0p0 += vi1x1 * vk11;
      vo1p0 += vi3x1 * vk11;
      vo2p0 += vi5x1 * vk11;
      vo0p1 += vi2x1 * vk21;
      vo1p1 += vi4x1 * vk21;
      vo2p1 += vi6x1 * vk21;
      vo0p0 += vi3x1 * vk31;
      vo1p0 += vi5x1 * vk31;
      vo2p0 += vi7x1 * vk31;
      vo0p1 += vi4x1 * vk41;
      vo1p1 += vi6x1 * vk41;
      vo2p1 += vi8x1 * vk41;

      vo0p0 += vi0x2 * vk02;
      vo1p0 += vi2x2 * vk02;
      vo2p0 += vi4x2 * vk02;
      vo0p1 += vi1x2 * vk12;
      vo1p1 += vi3x2 * vk12;
      vo2p1 += vi5x2 * vk12;
      vo0p0 += vi2x2 * vk22;
      vo1p0 += vi4x2 * vk22;
      vo2p0 += vi6x2 * vk22;
      vo0p1 += vi3x2 * vk32;
      vo1p1 += vi5x2 * vk32;
      vo2p1 += vi7x2 * vk32;
      vo0p0 += vi4x2 * vk42;
      vo1p0 += vi6x2 * vk42;
      vo2p0 += vi8x2 * vk42;

      vo0p1 += vi0x3 * vk03;
      vo1p1 += vi2x3 * vk03;
      vo2p1 += vi4x3 * vk03;
      vo0p0 += vi1x3 * vk13;
      vo1p0 += vi3x3 * vk13;
      vo2p0 += vi5x3 * vk13;
      vo0p1 += vi2x3 * vk23;
      vo1p1 += vi4x3 * vk23;
      vo2p1 += vi6x3 * vk23;
      vo0p0 += vi3x3 * vk33;
      vo1p0 += vi5x3 * vk33;
      vo2p0 += vi7x3 * vk33;
      vo0p1 += vi4x3 * vk43;
      vo1p1 += vi6x3 * vk43;
      vo2p1 += vi8x3 * vk43;

      vo0p0 += vo0p1;
      vo1p0 += vo1p1;
      vo2p0 += vo2p1;

      float vo0 = math_max_f32(vo0p0, vmin);
      float vo1 = math_max_f32(vo1p0, vmin);
      float vo2 = math_max_f32(vo2p0, vmin);

      vo0 = math_min_f32(vo0, vmax);
      vo1 = math_min_f32(vo1, vmax);
      vo2 = math_min_f32(vo2, vmax);

      *o2++ = vo2;
      *o1++ = vo1;
      *o0++ = vo0;
    } else {
      float vo0p0 = vbias + vi0x0 * vk00;
      float vo1p0 = vbias + vi2x0 * vk00;
      float vo2p0 = vbias + vi4x0 * vk00;
      float vo0p1 = vi1x0 * vk10;
      float vo1p1 = vi3x0 * vk10;
      float vo2p1 = vi5x0 * vk10;
      vo0p0 += vi2x0 * vk20;
      vo1p0 += vi4x0 * vk20;
      vo2p0 += vi6x0 * vk20;
      vo0p1 += vi3x0 * vk30;
      vo1p1 += vi5x0 * vk30;
      vo2p1 += vi7x0 * vk30;
      vo0p0 += vi4x0 * vk40;
      vo1p0 += vi6x0 * vk40;
      vo2p0 += vi8x0 * vk40;

      vo0p1 += vi0x1 * vk01;
      vo1p1 += vi2x1 * vk01;
      vo2p1 += vi4x1 * vk01;
      vo0p0 += vi1x1 * vk11;
      vo1p0 += vi3x1 * vk11;
      vo2p0 += vi5x1 * vk11;
      vo0p1 += vi2x1 * vk21;
      vo1p1 += vi4x1 * vk21;
      vo2p1 += vi6x1 * vk21;
      vo0p0 += vi3x1 * vk31;
      vo1p0 += vi5x1 * vk31;
      vo2p0 += vi7x1 * vk31;
      vo0p1 += vi4x1 * vk41;
      vo1p1 += vi6x1 * vk41;
      vo2p1 += vi8x1 * vk41;

      vo0p0 += vi0x2 * vk02;
      vo1p0 += vi2x2 * vk02;
      vo2p0 += vi4x2 * vk02;
      vo0p1 += vi1x2 * vk12;
      vo1p1 += vi3x2 * vk12;
      vo2p1 += vi5x2 * vk12;
      vo0p0 += vi2x2 * vk22;
      vo1p0 += vi4x2 * vk22;
      vo2p0 += vi6x2 * vk22;
      vo0p1 += vi3x2 * vk32;
      vo1p1 += vi5x2 * vk32;
      vo2p1 += vi7x2 * vk32;
      vo0p0 += vi4x2 * vk42;
      vo1p0 += vi6x2 * vk42;
      vo2p0 += vi8x2 * vk42;

      vo0p0 += vo0p1;
      vo1p0 += vo1p1;
      vo2p0 += vo2p1;

      float vo0 = math_max_f32(vo0p0, vmin);
      float vo1 = math_max_f32(vo1p0, vmin);
      float vo2 = math_max_f32(vo2p0, vmin);

      vo0 = math_min_f32(vo0, vmax);
      vo1 = math_min_f32(vo1, vmax);
      vo2 = math_min_f32(vo2, vmax);

      *o2++ = vo2;
      *o1++ = vo1;
      *o0++ = vo0;
    }

    i0 = (const float*) ((uintptr_t) i5);
    i1 = (const float*) ((uintptr_t) i6);
    i2 = (const float*) ((uintptr_t) i7);
    i3 = (const float*) ((uintptr_t) i8);
    i4 = (const float*) ((uintptr_t) i3 + input_width);
    i5 = (const float*) ((uintptr_t) i4 + input_width);
    i6 = (const float*) ((uintptr_t) i5 + input_width);
    i7 = (const float*) ((uintptr_t) i6 + input_width);
    i8 = (const float*) ((uintptr_t) i7 + input_width);

    o0 = o2;
    o1 = (float*) ((uintptr_t) o0 + output_width);
    o2 = (float*) ((uintptr_t) o1 + output_width);

    output_height = doz(output_height, 3);
    padded_input_height = doz(padded_input_height, 6);
  } while (output_height != 0);
}
