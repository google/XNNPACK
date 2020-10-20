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
    size_t channels,
    const float**restrict input,
    size_t input_offset,
    const float*restrict weights,
    float*restrict output)
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const size_t input_offset_increment = output_pixels * 4 * sizeof(float);
  size_t c = channels;
  do {
    const float** i = input;
    const float* w = weights;

    size_t p = output_pixels;
    do {
      const float* i0 = (const float*) ((uintptr_t) i[0] + input_offset);
      const float* i1 = (const float*) ((uintptr_t) i[1] + input_offset);
      const float* i2 = (const float*) ((uintptr_t) i[2] + input_offset);
      const float* i3 = (const float*) ((uintptr_t) i[3] + input_offset);
      i += 4;

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

    } while (--p != 0);

    input_offset += input_offset_increment;

    c -= sizeof(float);
  } while (c != 0);
}
