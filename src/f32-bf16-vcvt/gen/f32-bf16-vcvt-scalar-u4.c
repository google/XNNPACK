// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-bf16-vcvt/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/vcvt.h"


void xnn_f32_bf16_vcvt_ukernel__scalar_u4(
    size_t batch,
    const float* input,
    xnn_bfloat16* output,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float* i = input;
  xnn_bfloat16* o = output;
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float vw0 = i[0];
    const float vw1 = i[1];
    const float vw2 = i[2];
    const float vw3 = i[3];
    i += 4;

    xnn_bfloat16 vbf0 = xnn_bfloat16_from_float(vw0);
    xnn_bfloat16 vbf1 = xnn_bfloat16_from_float(vw1);
    xnn_bfloat16 vbf2 = xnn_bfloat16_from_float(vw2);
    xnn_bfloat16 vbf3 = xnn_bfloat16_from_float(vw3);

    o[0] = vbf0;
    o[1] = vbf1;
    o[2] = vbf2;
    o[3] = vbf3;
    o += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vw = *i++;

      xnn_bfloat16 vbf = xnn_bfloat16_from_float(vw);

      *o++ = vbf;

      batch -= sizeof(float);
    } while (batch != 0);
  }
}
