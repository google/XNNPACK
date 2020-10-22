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


void xnn_f32_ibilinear_chw_ukernel__scalar_p1(
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
  do {
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
  } while (--p != 0);
}
