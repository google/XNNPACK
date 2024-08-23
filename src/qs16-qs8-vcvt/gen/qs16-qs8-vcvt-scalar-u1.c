// Auto-generated file. Do not edit!
//   Template: src/qs16-qs8-vcvt/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/math.h"
#include "xnnpack/vcvt.h"


void xnn_qs16_qs8_vcvt_ukernel__scalar_u1(
    size_t batch,
    const int16_t* input,
    int8_t* output,
    const union xnn_qs16_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int32_t vmultiplier = params->scalar.multiplier;
  const int64_t vbias = (int64_t) ((int32_t) (params->scalar.output_zero_point << 16) + 0x8000);
  do {
    const int32_t vx = (int32_t) *input++;

    int32_t vout = (int32_t) math_asr_s64(math_mulext_s32(vx, vmultiplier) + vbias, 16);

    vout = math_max_s32(vout, -128);
    vout = math_min_s32(vout, 127);
    *output++ = (int8_t) vout;

    batch -= sizeof(int16_t);
  } while (batch != 0);
}
