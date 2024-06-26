// Auto-generated file. Do not edit!
//   Template: src/x8-lut/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/lut.h"
#include "xnnpack/common.h"


void xnn_x8_lut_ukernel__scalar_u4(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const uint8_t table[restrict XNN_MIN_ELEMENTS(256)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 4 * sizeof(uint8_t); batch -= 4 * sizeof(uint8_t)) {
    const size_t vx0 = (size_t) input[0];
    const size_t vx1 = (size_t) input[1];
    const size_t vx2 = (size_t) input[2];
    const size_t vx3 = (size_t) input[3];
    input += 4;

    const uint32_t vt0 = (uint32_t) table[vx0];
    const uint32_t vt1 = (uint32_t) table[vx1];
    const uint32_t vt2 = (uint32_t) table[vx2];
    const uint32_t vt3 = (uint32_t) table[vx3];

    output[0] = (uint8_t) vt0;
    output[1] = (uint8_t) vt1;
    output[2] = (uint8_t) vt2;
    output[3] = (uint8_t) vt3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const size_t vx = (size_t) *input++;
      const uint32_t vt = (uint32_t) table[vx];
      *output++ = (uint8_t) vt;
      batch -= sizeof(uint8_t);
    } while (batch != 0);
  }
}
