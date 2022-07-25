// Auto-generated file. Do not edit!
//   Template: src/qs8-vlrelu/scalar-andxor.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/math.h>
#include <xnnpack/vlrelu.h>


void xnn_qs8_vlrelu_ukernel__scalar_andxor_x1(
    size_t n,
    const int8_t* x,
    int8_t* y,
    const union xnn_qs8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  const int32_t vinput_zero_point = params->scalar_andxor.input_zero_point;
  const int32_t vmultiplier_diff = params->scalar_andxor.multiplier_diff;
  const int32_t vmultiplier_base = params->scalar_andxor.multiplier_base;
  const int32_t vbias = params->scalar_andxor.bias;
  do {
    int32_t vacc = (int32_t) *x++ - vinput_zero_point;
    const int32_t vmultiplier = vmultiplier_base ^ (vmultiplier_diff & math_asr_s32(vacc, 31));
    vacc = vbias + vacc * vmultiplier;

    int32_t vout = math_asr_s32(vacc, 8);
    vout = math_max_s32(vout, -128);
    vout = math_min_s32(vout, 127);
    *y++ = (int8_t) vout;

    n -= sizeof(int8_t);
  } while (n != 0);
}
