// Auto-generated file. Do not edit!
//   Template: src/s8-ibilinear/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/ibilinear.h"
#include "xnnpack/math.h"


void xnn_u8_ibilinear_ukernel__scalar_c1(
    size_t output_pixels,
    size_t channels,
    const uint8_t** restrict input,
    size_t input_offset,
    const int16_t* restrict weights,
    uint8_t* restrict output,
    size_t output_increment)
{
  assert(output_pixels != 0);
  assert(channels != 0);

  do {
    const uint8_t* i0 = (const uint8_t*) ((uintptr_t) input[0] + input_offset);
    const uint8_t* i1 = (const uint8_t*) ((uintptr_t) input[1] + input_offset);
    const uint8_t* i2 = (const uint8_t*) ((uintptr_t) input[2] + input_offset);
    const uint8_t* i3 = (const uint8_t*) ((uintptr_t) input[3] + input_offset);
    input += 4;

    const int32_t valphah = (int32_t) (uint32_t) (uint16_t) weights[0];
    const int32_t valphav = (int32_t) (uint32_t) (uint16_t) weights[1];
    weights += 2;

    const int32_t vrounding = INT32_C(0x00200000);

    size_t c = channels;
    do {
      const int32_t vtl = (int32_t) *i0++;
      const int32_t vtr = (int32_t) *i1++;
      const int32_t vbl = (int32_t) *i2++;
      const int32_t vbr = (int32_t) *i3++;

      const int32_t vtd = vtr - vtl;
      const int32_t vbd = vbr - vbl;

      const int32_t vt = (int32_t) ((uint32_t) vtl << 11) + vtd * valphah;
      const int32_t vb = (int32_t) ((uint32_t) vbl << 11) + vbd * valphah;

      const int32_t vd = vb - vt;

      const int32_t vacc = (int32_t) ((uint32_t) vt << 11) + vd * valphav;

      const int32_t vo = math_asr_s32(vacc + vrounding, 22);

      *output++ = vo;

      c -= sizeof(uint8_t);
    } while (c != 0);

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
