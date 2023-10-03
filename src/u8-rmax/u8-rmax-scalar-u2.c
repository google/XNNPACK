// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/reduce.h>


void xnn_u8_rmax_ukernel__scalar_u2(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  uint8_t vmax0 = 0;
  uint8_t vmax1 = 0;
  for (; batch >= 2 * sizeof(uint8_t); batch -= 2 * sizeof(uint8_t)) {
    const uint8_t vt0 = input[0];
    const uint8_t vt1 = input[1];
    input += 2;

    vmax0 = vt0 > vmax0 ? vt0 : vmax0;
    vmax1 = vt1 > vmax1 ? vt1 : vmax1;
  }
  uint8_t vmax = vmax0 > vmax1 ? vmax0 : vmax1;
  if (batch != 0) {
    const uint8_t vt = *input++;
    vmax = vt > vmax ? vt : vmax;
  }
  *output = vmax;
}
