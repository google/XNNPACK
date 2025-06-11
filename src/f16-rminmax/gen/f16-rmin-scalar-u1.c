// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-rminmax/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/reduce.h"


void xnn_f16_rmin_ukernel__scalar_u1(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;

  int16_t vmin0 = math_signcomplement_f16(o[0]);
  do {
    int16_t vt = math_signcomplement_f16(*i++);
    vmin0 = math_min_s16(vmin0, vt);
    batch -= sizeof(uint16_t);
  } while (batch != 0);
  o[0] = (uint16_t) math_signcomplement_f16((uint16_t) vmin0);
}
