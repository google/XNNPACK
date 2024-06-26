// Auto-generated file. Do not edit!
//   Template: src/s16-rmaxabs/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/math.h"
#include "xnnpack/rmaxabs.h"


void xnn_s16_rmaxabs_ukernel__scalar_x4(
    size_t batch,
    const int16_t* input,
    uint16_t* output)
{
  assert(batch != 0);
  assert(batch % sizeof(int16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  uint32_t vmax0 = 0;
  uint32_t vmax1 = 0;
  uint32_t vmax2 = 0;
  uint32_t vmax3 = 0;

  for (; batch >= 4 * sizeof(int16_t); batch -= 4 * sizeof(int16_t)) {
    const int32_t vi0 = (int32_t) input[0];
    const int32_t vi1 = (int32_t) input[1];
    const int32_t vi2 = (int32_t) input[2];
    const int32_t vi3 = (int32_t) input[3];
    input += 4;

    const uint32_t vabs0 = math_abs_s32(vi0);
    const uint32_t vabs1 = math_abs_s32(vi1);
    const uint32_t vabs2 = math_abs_s32(vi2);
    const uint32_t vabs3 = math_abs_s32(vi3);

    vmax0 = math_max_u32(vmax0, vabs0);
    vmax1 = math_max_u32(vmax1, vabs1);
    vmax2 = math_max_u32(vmax2, vabs2);
    vmax3 = math_max_u32(vmax3, vabs3);
  }

  vmax0 = math_max_u32(vmax0, vmax1);
  vmax2 = math_max_u32(vmax2, vmax3);
  vmax0 = math_max_u32(vmax0, vmax2);

  if (batch != 0) {
    do {
      const int32_t vi = (int32_t) *input++;
      const uint32_t vabs = math_abs_s32(vi);
      vmax0 = math_max_u32(vmax0, vabs);
      batch -= sizeof(int16_t);
    } while (batch != 0);
  }
  *output = (uint16_t) vmax0;
}
