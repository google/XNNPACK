// Auto-generated file. Do not edit!
//   Template: src/qs8-gavgpool/unipass-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include "xnnpack/gavgpool.h"
#include "xnnpack/math.h"


void xnn_qu8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c1(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint8_t* output,
    const union xnn_qu8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const uint8_t* i0 = input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  const uint8_t* i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  const uint8_t* i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  const uint8_t* i4 = (const uint8_t*) ((uintptr_t) i3 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  const uint8_t* i5 = (const uint8_t*) ((uintptr_t) i4 + input_stride);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  const uint8_t* i6 = (const uint8_t*) ((uintptr_t) i5 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const int32_t vinit_bias = params->fp32_scalar_lrintf.init_bias;
  const float vscale = params->fp32_scalar_lrintf.scale;
  const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
  do {
    int32_t vacc = vinit_bias;
    const int32_t vi0 = (int32_t) *i0++;
    const int32_t vi1 = (int32_t) *i1++;

    vacc += vi0;
    const int32_t vi2 = (int32_t) *i2++;
    vacc += vi1;
    const int32_t vi3 = (int32_t) *i3++;
    vacc += vi2;
    const int32_t vi4 = (int32_t) *i4++;
    vacc += vi3;
    const int32_t vi5 = (int32_t) *i5++;
    vacc += vi4;
    const int32_t vi6 = (int32_t) *i6++;

    vacc += vi5;
    vacc += vi6;

    float vfpacc = (float) vacc * vscale;
    vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
    vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
    const int32_t vrndacc = (int32_t) lrintf(vfpacc);
    int32_t vout = vrndacc + voutput_zero_point;

    *output++ = (uint8_t) vout;
  } while (--channels != 0);
}
