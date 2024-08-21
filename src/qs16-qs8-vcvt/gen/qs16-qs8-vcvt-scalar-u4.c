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


void xnn_qs16_qs8_vcvt_ukernel__scalar_u4(
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
  for (; batch >= 4 * sizeof(int16_t); batch -= 4 * sizeof(int16_t)) {

    const int32_t vx0 = (int32_t) input[0];
    const int32_t vx1 = (int32_t) input[1];
    const int32_t vx2 = (int32_t) input[2];
    const int32_t vx3 = (int32_t) input[3];
    input += 4;

    int32_t vout0 = (int32_t) math_asr_s64(math_mulext_s32(vx0, vmultiplier) + vbias, 16);
    int32_t vout1 = (int32_t) math_asr_s64(math_mulext_s32(vx1, vmultiplier) + vbias, 16);
    int32_t vout2 = (int32_t) math_asr_s64(math_mulext_s32(vx2, vmultiplier) + vbias, 16);
    int32_t vout3 = (int32_t) math_asr_s64(math_mulext_s32(vx3, vmultiplier) + vbias, 16);

    vout0 = math_max_s32(vout0, -128);
    vout1 = math_max_s32(vout1, -128);
    vout2 = math_max_s32(vout2, -128);
    vout3 = math_max_s32(vout3, -128);

    vout0 = math_min_s32(vout0, 127);
    vout1 = math_min_s32(vout1, 127);
    vout2 = math_min_s32(vout2, 127);
    vout3 = math_min_s32(vout3, 127);

    output[0] = (int8_t) vout0;
    output[1] = (int8_t) vout1;
    output[2] = (int8_t) vout2;
    output[3] = (int8_t) vout3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const int32_t vx = (int32_t) *input++;

      int32_t vout = (int32_t) math_asr_s64(math_mulext_s32(vx, vmultiplier) + vbias, 16);

      vout = math_max_s32(vout, -128);
      vout = math_min_s32(vout, 127);
      *output++ = (int8_t) vout;

      batch -= sizeof(int16_t);
    } while (batch != 0);
  }
}
