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


void xnn_f32_ibilinear_chw_ukernel__scalar_p4(
    size_t output_pixels,
    const float**restrict input,
    size_t input_offset,
    const float*restrict horizontal_weights,
    const float*restrict vertical_weights,
    float* output)
{
  assert(output_pixels != 0);

  const float** i = input;
  const float* wh = horizontal_weights;
  const float* wv = vertical_weights;

  size_t p = output_pixels;
  for (; p >= 4; p -= 4) {
    const float* i0 = (const float*) ((uintptr_t) i[0] + input_offset);
    const float* i1 = (const float*) ((uintptr_t) i[1] + input_offset);
    const float* i2 = (const float*) ((uintptr_t) i[2] + input_offset);
    const float* i3 = (const float*) ((uintptr_t) i[3] + input_offset);
    const float* i4 = (const float*) ((uintptr_t) i[4] + input_offset);
    const float* i5 = (const float*) ((uintptr_t) i[5] + input_offset);
    const float* i6 = (const float*) ((uintptr_t) i[6] + input_offset);
    const float* i7 = (const float*) ((uintptr_t) i[7] + input_offset);
    const float* i8 = (const float*) ((uintptr_t) i[8] + input_offset);
    const float* i9 = (const float*) ((uintptr_t) i[9] + input_offset);
    const float* i10 = (const float*) ((uintptr_t) i[10] + input_offset);
    const float* i11 = (const float*) ((uintptr_t) i[11] + input_offset);
    const float* i12 = (const float*) ((uintptr_t) i[12] + input_offset);
    const float* i13 = (const float*) ((uintptr_t) i[13] + input_offset);
    const float* i14 = (const float*) ((uintptr_t) i[14] + input_offset);
    const float* i15 = (const float*) ((uintptr_t) i[15] + input_offset);
    i += 4 * 4;

    const float valphah0 = wh[0];
    const float valphav0 = wv[0];
    const float valphah1 = wh[1];
    const float valphav1 = wv[1];
    const float valphah2 = wh[2];
    const float valphav2 = wv[2];
    const float valphah3 = wh[3];
    const float valphav3 = wv[3];
    wh += 4;
    wv += 4;

    const float vtl0 = *i0;
    const float vtr0 = *i1;
    const float vbl0 = *i2;
    const float vbr0 = *i3;
    const float vtl1 = *i4;
    const float vtr1 = *i5;
    const float vbl1 = *i6;
    const float vbr1 = *i7;
    const float vtl2 = *i8;
    const float vtr2 = *i9;
    const float vbl2 = *i10;
    const float vbr2 = *i11;
    const float vtl3 = *i12;
    const float vtr3 = *i13;
    const float vbl3 = *i14;
    const float vbr3 = *i15;

    const float vtd0 = vtr0 - vtl0;
    const float vbd0 = vbr0 - vbl0;
    const float vtd1 = vtr1 - vtl1;
    const float vbd1 = vbr1 - vbl1;
    const float vtd2 = vtr2 - vtl2;
    const float vbd2 = vbr2 - vbl2;
    const float vtd3 = vtr3 - vtl3;
    const float vbd3 = vbr3 - vbl3;

    const float vt0 = vtl0 + vtd0 * valphah0;
    const float vb0 = vbl0 + vbd0 * valphah0;
    const float vt1 = vtl1 + vtd1 * valphah1;
    const float vb1 = vbl1 + vbd1 * valphah1;
    const float vt2 = vtl2 + vtd2 * valphah2;
    const float vb2 = vbl2 + vbd2 * valphah2;
    const float vt3 = vtl3 + vtd3 * valphah3;
    const float vb3 = vbl3 + vbd3 * valphah3;

    const float vd0 = vb0 - vt0;
    const float vd1 = vb1 - vt1;
    const float vd2 = vb2 - vt2;
    const float vd3 = vb3 - vt3;

    const float vo0 = vt0 + vd0 * valphav0;
    const float vo1 = vt1 + vd1 * valphav1;
    const float vo2 = vt2 + vd2 * valphav2;
    const float vo3 = vt3 + vd3 * valphav3;

    output[0] = vo0;
    output[1] = vo1;
    output[2] = vo2;
    output[3] = vo3;
    output += 4;
  }

  for (; p >= 1; p -= 1) {
    const float* i0 = (const float*) ((uintptr_t) i[0] + input_offset);
    const float* i1 = (const float*) ((uintptr_t) i[1] + input_offset);
    const float* i2 = (const float*) ((uintptr_t) i[2] + input_offset);
    const float* i3 = (const float*) ((uintptr_t) i[3] + input_offset);
    i += 4;

    const float valphah = *wh++;
    const float valphav = *wv++;

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
}
