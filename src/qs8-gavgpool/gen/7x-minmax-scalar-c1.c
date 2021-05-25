// Auto-generated file. Do not edit!
//   Template: src/qs8-gavgpool/unipass-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/gavgpool.h>
#include <xnnpack/math.h>


void xnn_qs8_gavgpool_minmax_ukernel_7x__scalar_c1(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int8_t* output,
    const union xnn_qs8_avgpool_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const int8_t* i0 = input;
  const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  const int8_t* i2 = (const int8_t*) ((uintptr_t) i1 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  const int8_t* i3 = (const int8_t*) ((uintptr_t) i2 + input_stride);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  const int8_t* i4 = (const int8_t*) ((uintptr_t) i3 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  const int8_t* i5 = (const int8_t*) ((uintptr_t) i4 + input_stride);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  const int8_t* i6 = (const int8_t*) ((uintptr_t) i5 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const int32_t vbias = params->scalar.bias;
  const int32_t vmultiplier = params->scalar.multiplier;
  const int64_t vrounding = params->scalar.rounding;
  const uint32_t vshift = params->scalar.shift;
  const int32_t vout_min = params->scalar.output_min_less_zero_point;
  const int32_t vout_max = params->scalar.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->scalar.output_zero_point;
  do {
    const int32_t vi0 = *i0++;
    const int32_t vi1 = *i1++;
    const int32_t vi2 = *i2++;
    const int32_t vi3 = *i3++;
    const int32_t vi4 = *i4++;
    const int32_t vi5 = *i5++;
    const int32_t vi6 = *i6++;

    int32_t vacc0 = vi0 + vi1;

    vacc0 += vi2;
    vacc0 += vi3;
    vacc0 += vi4;
    vacc0 += vi5;
    vacc0 += vi6;


    int32_t vacc = vbias + vacc0;
    const int64_t vprod = (int64_t) vacc * (int64_t) vmultiplier;
    const int64_t vadjprod = vprod - (int64_t) (vacc < 0);
    int32_t vout = (int32_t) asr_s64(vadjprod + vrounding, vshift);

    vout = math_max_s32(vout, vout_min);
    vout = math_min_s32(vout, vout_max);
    vout += voutput_zero_point;

    *output++ = (int8_t) vout;
  } while (--channels != 0);
}
