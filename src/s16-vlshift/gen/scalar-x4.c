// Auto-generated file. Do not edit!
//   Template: src/s16-vlshift/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack/math.h>
#include <xnnpack/vlshift.h>


void xnn_s16_vlshift_ukernel__scalar_x4(
    size_t batch,
    const int16_t* input,
    int16_t* output,
    uint32_t shift)
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(shift < 16);

  for (; batch >= 4; batch -= 4) {
    const uint16_t vi0 = (uint16_t) input[0];
    const uint16_t vi1 = (uint16_t) input[1];
    const uint16_t vi2 = (uint16_t) input[2];
    const uint16_t vi3 = (uint16_t) input[3];
    input += 4;

    const uint16_t vout0 = vi0 << shift;
    const uint16_t vout1 = vi1 << shift;
    const uint16_t vout2 = vi2 << shift;
    const uint16_t vout3 = vi3 << shift;

    output[0] = (int16_t) vout0;
    output[1] = (int16_t) vout1;
    output[2] = (int16_t) vout2;
    output[3] = (int16_t) vout3;
    output += 4;
  }
 if XNN_UNLIKELY(batch != 0) {
   do {
     const uint16_t vi = (uint16_t) *input++;

     const uint16_t vout = vi << shift;

     *output++ = (int16_t) vout;
   } while (--batch != 0);
 }
}
