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


void xnn_s8_ibilinear_ukernel__scalar_c2(
    size_t output_pixels,
    size_t channels,
    const int8_t** restrict input,
    size_t input_offset,
    const int16_t* restrict weights,
    int8_t* restrict output,
    size_t output_increment)
{
  assert(output_pixels != 0);
  assert(channels != 0);

  do {
    const int8_t* i0 = (const int8_t*) ((uintptr_t) input[0] + input_offset);
    const int8_t* i1 = (const int8_t*) ((uintptr_t) input[1] + input_offset);
    const int8_t* i2 = (const int8_t*) ((uintptr_t) input[2] + input_offset);
    const int8_t* i3 = (const int8_t*) ((uintptr_t) input[3] + input_offset);
    input += 4;

    const int32_t valphah = (int32_t) (uint32_t) (uint16_t) weights[0];
    const int32_t valphav = (int32_t) (uint32_t) (uint16_t) weights[1];
    weights += 2;

    const int32_t vrounding = INT32_C(0x00200000);

    size_t c = channels;
    for (; c >= 2 * sizeof(int8_t); c -= 2 * sizeof(int8_t)) {
      const int32_t vtl0 = (int32_t) i0[0];
      const int32_t vtr0 = (int32_t) i1[0];
      const int32_t vbl0 = (int32_t) i2[0];
      const int32_t vbr0 = (int32_t) i3[0];
      const int32_t vtl1 = (int32_t) i0[1];
      const int32_t vtr1 = (int32_t) i1[1];
      const int32_t vbl1 = (int32_t) i2[1];
      const int32_t vbr1 = (int32_t) i3[1];
      i0 += 2;
      i1 += 2;
      i2 += 2;
      i3 += 2;

      const int32_t vtd0 = vtr0 - vtl0;
      const int32_t vbd0 = vbr0 - vbl0;
      const int32_t vtd1 = vtr1 - vtl1;
      const int32_t vbd1 = vbr1 - vbl1;

      const int32_t vt0 = (int32_t) ((uint32_t) vtl0 << 11) + vtd0 * valphah;
      const int32_t vb0 = (int32_t) ((uint32_t) vbl0 << 11) + vbd0 * valphah;
      const int32_t vt1 = (int32_t) ((uint32_t) vtl1 << 11) + vtd1 * valphah;
      const int32_t vb1 = (int32_t) ((uint32_t) vbl1 << 11) + vbd1 * valphah;

      const int32_t vd0 = vb0 - vt0;
      const int32_t vd1 = vb1 - vt1;

      const int32_t vacc0 = (int32_t) ((uint32_t) vt0 << 11) + vd0 * valphav;
      const int32_t vacc1 = (int32_t) ((uint32_t) vt1 << 11) + vd1 * valphav;

      const int32_t vo0 = math_asr_s32(vacc0 + vrounding, 22);
      const int32_t vo1 = math_asr_s32(vacc1 + vrounding, 22);

      output[0] = (int8_t) vo0;
      output[1] = (int8_t) vo1;
      output += 2;
    }
    for (; c >= sizeof(int8_t); c -= sizeof(int8_t)) {
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
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
