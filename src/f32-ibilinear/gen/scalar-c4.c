// Auto-generated file. Do not edit!
//   Template: src/f32-ibilinear/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/ibilinear.h>


void xnn_f32_ibilinear_ukernel__scalar_c4(
    size_t output_pixels,
    size_t channels,
    const float**restrict input,
    size_t input_offset,
    const float*restrict weights,
    float*restrict output,
    size_t output_increment)
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  do {
    const float* i0 = (const float*) ((uintptr_t) input[0] + input_offset);
    const float* i1 = (const float*) ((uintptr_t) input[1] + input_offset);
    const float* i2 = (const float*) ((uintptr_t) input[2] + input_offset);
    const float* i3 = (const float*) ((uintptr_t) input[3] + input_offset);
    input += 4;

    const float valphah = weights[0];
    const float valphav = weights[1];
    weights += 2;

    size_t c = channels;
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const float vtl0 = i0[0];
      const float vtr0 = i1[0];
      const float vbl0 = i2[0];
      const float vbr0 = i3[0];
      const float vtl1 = i0[1];
      const float vtr1 = i1[1];
      const float vbl1 = i2[1];
      const float vbr1 = i3[1];
      const float vtl2 = i0[2];
      const float vtr2 = i1[2];
      const float vbl2 = i2[2];
      const float vbr2 = i3[2];
      const float vtl3 = i0[3];
      const float vtr3 = i1[3];
      const float vbl3 = i2[3];
      const float vbr3 = i3[3];
      i0 += 4;
      i1 += 4;
      i2 += 4;
      i3 += 4;

      const float vtd0 = vtr0 - vtl0;
      const float vbd0 = vbr0 - vbl0;
      const float vtd1 = vtr1 - vtl1;
      const float vbd1 = vbr1 - vbl1;
      const float vtd2 = vtr2 - vtl2;
      const float vbd2 = vbr2 - vbl2;
      const float vtd3 = vtr3 - vtl3;
      const float vbd3 = vbr3 - vbl3;

      const float vt0 = vtl0 + vtd0 * valphah;
      const float vb0 = vbl0 + vbd0 * valphah;
      const float vt1 = vtl1 + vtd1 * valphah;
      const float vb1 = vbl1 + vbd1 * valphah;
      const float vt2 = vtl2 + vtd2 * valphah;
      const float vb2 = vbl2 + vbd2 * valphah;
      const float vt3 = vtl3 + vtd3 * valphah;
      const float vb3 = vbl3 + vbd3 * valphah;

      const float vd0 = vb0 - vt0;
      const float vd1 = vb1 - vt1;
      const float vd2 = vb2 - vt2;
      const float vd3 = vb3 - vt3;

      const float vo0 = vt0 + vd0 * valphav;
      const float vo1 = vt1 + vd1 * valphav;
      const float vo2 = vt2 + vd2 * valphav;
      const float vo3 = vt3 + vd3 * valphav;

      output[0] = vo0;
      output[1] = vo1;
      output[2] = vo2;
      output[3] = vo3;
      output += 4;
    }
    for (; c >= sizeof(float); c -= sizeof(float)) {
      const float vtl = *i0++;
      const float vtr = *i1++;
      const float vbl = *i2++;
      const float vbr = *i3++;

      const float vtd = vtr - vtl;
      const float vbd = vbr - vbl;

      const float vt = vtl + vtd * valphah;
      const float vb = vbl + vbd * valphah;

      const float vd = vb - vt;

      const float vo = vt + vd * valphav;

      *output++ = vo;
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
