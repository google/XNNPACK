// Auto-generated file. Do not edit!
//   Template: src/qs8-vcvt/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/math.h>
#include <xnnpack/vcvt.h>


void xnn_qu8_vcvt_ukernel__scalar_x4(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const union xnn_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  const int32_t vbias = params->scalar.bias;
  const int32_t vmultiplier = params->scalar.multiplier;
  for (; n >= 4 * sizeof(uint8_t); n -= 4 * sizeof(uint8_t)) {
    int32_t vacc0 = x[0];
    int32_t vacc1 = x[1];
    int32_t vacc2 = x[2];
    int32_t vacc3 = x[3];
    x += 4;

    vacc0 = vbias + vacc0 * vmultiplier;
    vacc1 = vbias + vacc1 * vmultiplier;
    vacc2 = vbias + vacc2 * vmultiplier;
    vacc3 = vbias + vacc3 * vmultiplier;

    int32_t vout0 = math_asr_s32(vacc0, 8);
    int32_t vout1 = math_asr_s32(vacc1, 8);
    int32_t vout2 = math_asr_s32(vacc2, 8);
    int32_t vout3 = math_asr_s32(vacc3, 8);

    vout0 = math_max_s32(vout0, 0);
    vout1 = math_max_s32(vout1, 0);
    vout2 = math_max_s32(vout2, 0);
    vout3 = math_max_s32(vout3, 0);

    vout0 = math_min_s32(vout0, 255);
    vout1 = math_min_s32(vout1, 255);
    vout2 = math_min_s32(vout2, 255);
    vout3 = math_min_s32(vout3, 255);

    y[0] = (uint8_t) vout0;
    y[1] = (uint8_t) vout1;
    y[2] = (uint8_t) vout2;
    y[3] = (uint8_t) vout3;
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      int32_t vacc = *x++;
      vacc = vbias + vacc * vmultiplier;

      int32_t vout = math_asr_s32(vacc, 8);
      vout = math_max_s32(vout, 0);
      vout = math_min_s32(vout, 255);
      *y++ = (uint8_t) vout;

      n -= sizeof(uint8_t);
    } while (n != 0);
  }
}
