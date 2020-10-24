// Auto-generated file. Do not edit!
//   Template: src/f32-ibilinear-chw/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/ibilinear.h>


void xnn_f32_ibilinear_chw_ukernel__scalar_p2(
    size_t output_pixels,
    size_t channels,
    const float**restrict input,
    size_t input_offset,
    const float*restrict weights,
    float*restrict output,
    size_t input_increment)
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(input_increment % sizeof(float) == 0);

  size_t c = channels;
  do {
    const float** i = input;
    const float* w = weights;

    size_t p = output_pixels;
    for (; p >= 2; p -= 2) {
      const float* i0 = (const float*) ((uintptr_t) i[0] + input_offset);
      const float* i1 = i0 + 1;
      const float* i2 = (const float*) ((uintptr_t) i[1] + input_offset);
      const float* i3 = i2 + 1;
      const float* i4 = (const float*) ((uintptr_t) i[2] + input_offset);
      const float* i5 = i4 + 1;
      const float* i6 = (const float*) ((uintptr_t) i[3] + input_offset);
      const float* i7 = i6 + 1;
      i += 2 * 2;

      const float valphah0 = w[0];
      const float valphav0 = w[1];
      const float valphah1 = w[2];
      const float valphav1 = w[3];
      w += 2 * 2;

      const float vtl0 = *i0;
      const float vtr0 = *i1;
      const float vbl0 = *i2;
      const float vbr0 = *i3;
      const float vtl1 = *i4;
      const float vtr1 = *i5;
      const float vbl1 = *i6;
      const float vbr1 = *i7;

      const float vtd0 = vtr0 - vtl0;
      const float vbd0 = vbr0 - vbl0;
      const float vtd1 = vtr1 - vtl1;
      const float vbd1 = vbr1 - vbl1;

      const float vt0 = vtl0 + vtd0 * valphah0;
      const float vb0 = vbl0 + vbd0 * valphah0;
      const float vt1 = vtl1 + vtd1 * valphah1;
      const float vb1 = vbl1 + vbd1 * valphah1;

      const float vd0 = vb0 - vt0;
      const float vd1 = vb1 - vt1;

      const float vo0 = vt0 + vd0 * valphav0;
      const float vo1 = vt1 + vd1 * valphav1;

      output[0] = vo0;
      output[1] = vo1;
      output += 2;
    }

    for (; p >= 1; p -= 1) {
      const float* i0 = (const float*) ((uintptr_t) i[0] + input_offset);
      const float* i1 = i0 + 1;
      const float* i2 = (const float*) ((uintptr_t) i[1] + input_offset);
      const float* i3 = i2 + 1;
      i += 2;

      const float valphah = w[0];
      const float valphav = w[1];
      w += 2;

      const float vtl = *i0;
      const float vtr = *i1;
      const float vbl = *i2;
      const float vbr = *i3;

      const float vtd = vtr - vtl;
      const float vbd = vbr - vbl;

      const float vt = vtl + vtd * valphah;
      const float vb = vbl + vbd * valphah;

      const float vd = vb - vt;

      const float vo = vt + vd * valphav;

      *output++ = vo;
    }

    input_offset += input_increment;

    c--;
  } while (c != 0);
}
