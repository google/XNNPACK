// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-vcvt/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vcvt.h"


void xnn_qu8_vcvt_ukernel__scalar_u1(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const struct xnn_qu8_cvt_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int32_t vmultiplier = (int32_t) params->scalar.multiplier << params->scalar.shift;
  const int32_t vbias =
      (int32_t) (((uint32_t) (int32_t) params->scalar.output_zero_point) << 15) -
      vmultiplier * (int32_t) params->scalar.input_zero_point +
      INT32_C(0x4000);
  do {
    int32_t vacc = *input++;
    vacc = vbias + vacc * vmultiplier;

    int32_t vout = math_asr_s32(vacc, 15);
    vout = math_max_s32(vout, 0);
    vout = math_min_s32(vout, 255);
    *output++ = (uint8_t) vout;

    batch -= sizeof(uint8_t);
  } while (batch != 0);
}
