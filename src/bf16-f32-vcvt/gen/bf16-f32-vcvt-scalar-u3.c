// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-f32-vcvt/scalar.c.in
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


void xnn_bf16_f32_vcvt_ukernel__scalar_u3(
    size_t batch,
    const xnn_bfloat16* input,
    float* output,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_bfloat16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const xnn_bfloat16* i = input;
  float* o = output;
  for (; batch >= 3 * sizeof(xnn_bfloat16); batch -= 3 * sizeof(xnn_bfloat16)) {
    const xnn_bfloat16 vh0 = i[0];
    const xnn_bfloat16 vh1 = i[1];
    const xnn_bfloat16 vh2 = i[2];
    i += 3;

    o[0] = xnn_bfloat16_to_float(vh0);
    o[1] = xnn_bfloat16_to_float(vh1);
    o[2] = xnn_bfloat16_to_float(vh2);
    o += 3;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const xnn_bfloat16 vh = *i++;

      *o++ = xnn_bfloat16_to_float(vh);

      batch -= sizeof(xnn_bfloat16);
    } while (batch != 0);
  }
}
