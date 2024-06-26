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


void xnn_qu8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c2(
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
  for (; channels >= 2; channels -= 2) {
    const int32_t vi0x0 = (int32_t) i0[0];
    const int32_t vi0x1 = (int32_t) i0[1];
    i0 += 2;

    int32_t vacc0 = vi0x0 + vinit_bias;
    const int32_t vi1x0 = (int32_t) i1[0];
    int32_t vacc1 = vi0x1 + vinit_bias;
    const int32_t vi1x1 = (int32_t) i1[1];
    i1 += 2;

    vacc0 += vi1x0;
    const int32_t vi2x0 = (int32_t) i2[0];
    vacc1 += vi1x1;
    const int32_t vi2x1 = (int32_t) i2[1];
    i2 += 2;
    vacc0 += vi2x0;
    const int32_t vi3x0 = (int32_t) i3[0];
    vacc1 += vi2x1;
    const int32_t vi3x1 = (int32_t) i3[1];
    i3 += 2;
    vacc0 += vi3x0;
    const int32_t vi4x0 = (int32_t) i4[0];
    vacc1 += vi3x1;
    const int32_t vi4x1 = (int32_t) i4[1];
    i4 += 2;
    vacc0 += vi4x0;
    const int32_t vi5x0 = (int32_t) i5[0];
    vacc1 += vi4x1;
    const int32_t vi5x1 = (int32_t) i5[1];
    i5 += 2;
    vacc0 += vi5x0;
    const int32_t vi6x0 = (int32_t) i6[0];
    vacc1 += vi5x1;
    const int32_t vi6x1 = (int32_t) i6[1];
    i6 += 2;

    vacc0 += vi6x0;
    vacc1 += vi6x1;

    float vfpacc0 = (float) vacc0 * vscale;
    float vfpacc1 = (float) vacc1 * vscale;

    vfpacc0 = math_max_f32(vfpacc0, voutput_min_less_zero_point);
    vfpacc1 = math_max_f32(vfpacc1, voutput_min_less_zero_point);

    vfpacc0 = math_min_f32(vfpacc0, voutput_max_less_zero_point);
    vfpacc1 = math_min_f32(vfpacc1, voutput_max_less_zero_point);

    const int32_t vrndacc0 = (int32_t) lrintf(vfpacc0);
    const int32_t vrndacc1 = (int32_t) lrintf(vfpacc1);

    int32_t vout0 = vrndacc0 + voutput_zero_point;
    int32_t vout1 = vrndacc1 + voutput_zero_point;

    output[0] = (uint8_t) vout0;
    output[1] = (uint8_t) vout1;
    output += 2;
  }
  if XNN_UNLIKELY(channels != 0) {
    int32_t vacc = vinit_bias;
    const int32_t vi0 = (int32_t) *i0;
    const int32_t vi1 = (int32_t) *i1;

    vacc += vi0;
    const int32_t vi2 = (int32_t) *i2;
    vacc += vi1;
    const int32_t vi3 = (int32_t) *i3;
    vacc += vi2;
    const int32_t vi4 = (int32_t) *i4;
    vacc += vi3;
    const int32_t vi5 = (int32_t) *i5;
    vacc += vi4;
    const int32_t vi6 = (int32_t) *i6;

    vacc += vi5;
    vacc += vi6;

    float vfpacc = (float) vacc * vscale;
    vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
    vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
    const int32_t vrndacc = (int32_t) lrintf(vfpacc);
    int32_t vout = vrndacc + voutput_zero_point;

    *output = (uint8_t) vout;
  }
}
