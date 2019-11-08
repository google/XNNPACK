// Auto-generated file. Do not edit!
//   Template: src/f32-bilinear/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/bilinear.h>


void xnn_f32_bilinear_ukernel__scalar_c1(
    size_t output_pixels,
    size_t channels,
    const float**restrict input,
    const float*restrict weights,
    float*restrict output,
    size_t output_increment)
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  do {
    const float* i0 = input[0];
    const float* i1 = input[1];
    const float* i2 = input[2];
    const float* i3 = input[3];
    input += 4;

    const float valphah = weights[0];
    const float valphav = weights[1];
    weights += 2;

    size_t c = channels;
    do {
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

      c -= sizeof(float);
    } while (c != 0);

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
