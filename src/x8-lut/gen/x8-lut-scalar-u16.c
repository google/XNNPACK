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


void xnn_x8_lut_ukernel__scalar_u16(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const uint8_t table[restrict XNN_MIN_ELEMENTS(256)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    const size_t vx0 = (size_t) input[0];
    const size_t vx1 = (size_t) input[1];
    const size_t vx2 = (size_t) input[2];
    const size_t vx3 = (size_t) input[3];
    const size_t vx4 = (size_t) input[4];
    const size_t vx5 = (size_t) input[5];
    const size_t vx6 = (size_t) input[6];
    const size_t vx7 = (size_t) input[7];
    const size_t vx8 = (size_t) input[8];
    const size_t vx9 = (size_t) input[9];
    const size_t vx10 = (size_t) input[10];
    const size_t vx11 = (size_t) input[11];
    const size_t vx12 = (size_t) input[12];
    const size_t vx13 = (size_t) input[13];
    const size_t vx14 = (size_t) input[14];
    const size_t vx15 = (size_t) input[15];
    input += 16;

    const uint32_t vt0 = (uint32_t) table[vx0];
    const uint32_t vt1 = (uint32_t) table[vx1];
    const uint32_t vt2 = (uint32_t) table[vx2];
    const uint32_t vt3 = (uint32_t) table[vx3];
    const uint32_t vt4 = (uint32_t) table[vx4];
    const uint32_t vt5 = (uint32_t) table[vx5];
    const uint32_t vt6 = (uint32_t) table[vx6];
    const uint32_t vt7 = (uint32_t) table[vx7];
    const uint32_t vt8 = (uint32_t) table[vx8];
    const uint32_t vt9 = (uint32_t) table[vx9];
    const uint32_t vt10 = (uint32_t) table[vx10];
    const uint32_t vt11 = (uint32_t) table[vx11];
    const uint32_t vt12 = (uint32_t) table[vx12];
    const uint32_t vt13 = (uint32_t) table[vx13];
    const uint32_t vt14 = (uint32_t) table[vx14];
    const uint32_t vt15 = (uint32_t) table[vx15];

    output[0] = (uint8_t) vt0;
    output[1] = (uint8_t) vt1;
    output[2] = (uint8_t) vt2;
    output[3] = (uint8_t) vt3;
    output[4] = (uint8_t) vt4;
    output[5] = (uint8_t) vt5;
    output[6] = (uint8_t) vt6;
    output[7] = (uint8_t) vt7;
    output[8] = (uint8_t) vt8;
    output[9] = (uint8_t) vt9;
    output[10] = (uint8_t) vt10;
    output[11] = (uint8_t) vt11;
    output[12] = (uint8_t) vt12;
    output[13] = (uint8_t) vt13;
    output[14] = (uint8_t) vt14;
    output[15] = (uint8_t) vt15;
    output += 16;
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
