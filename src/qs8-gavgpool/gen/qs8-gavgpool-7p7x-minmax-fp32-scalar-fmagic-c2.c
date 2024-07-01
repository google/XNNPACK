// Auto-generated file. Do not edit!
//   Template: src/qs8-gavgpool/multipass-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/gavgpool.h"
#include "xnnpack/math.h"


void xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int32_t* buffer,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows > 7);
  assert(channels != 0);

  const int8_t* i0 = input;
  const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
  const int8_t* i2 = (const int8_t*) ((uintptr_t) i1 + input_stride);
  const int8_t* i3 = (const int8_t*) ((uintptr_t) i2 + input_stride);
  const int8_t* i4 = (const int8_t*) ((uintptr_t) i3 + input_stride);
  const int8_t* i5 = (const int8_t*) ((uintptr_t) i4 + input_stride);
  const int8_t* i6 = (const int8_t*) ((uintptr_t) i5 + input_stride);
  const size_t input_increment = 7 * input_stride - round_up_po2(channels, 2) * sizeof(int8_t);

  const int32_t vinit_bias = params->fp32_scalar_fmagic.init_bias;
  int32_t* b = buffer;
  for (ptrdiff_t c = (ptrdiff_t) channels; c > 0; c -= 2) {
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

    b[0] = vacc0;
    b[1] = vacc1;
    b += 2;
  }

  for (rows -= 7; rows > 7; rows -= 7) {
    i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
    i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
    i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
    i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
    i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
    i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
    i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);

    int32_t* b = buffer;
    for (ptrdiff_t c = (ptrdiff_t) channels; c > 0; c -= 2) {
      int32_t vacc0 = b[0];
      const int32_t vi0x0 = (int32_t) i0[0];
      int32_t vacc1 = b[1];
      const int32_t vi0x1 = (int32_t) i0[1];
      i0 += 2;

      vacc0 += vi0x0;
      const int32_t vi1x0 = (int32_t) i1[0];
      vacc1 += vi0x1;
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

      b[0] = vacc0;
      b[1] = vacc1;
      b += 2;
    }
  }

  i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
  i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const float vscale = params->fp32_scalar_fmagic.scale;
  const float voutput_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
  const float vmagic_bias = params->fp32_scalar_fmagic.magic_bias;
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;
  for (; channels >= 2; channels -= 2) {
    int32_t vacc0 = buffer[0];
    const int32_t vi0x0 = (int32_t) i0[0];
    int32_t vacc1 = buffer[1];
    const int32_t vi0x1 = (int32_t) i0[1];
    buffer += 2;
    i0 += 2;

    vacc0 += vi0x0;
    const int32_t vi1x0 = (int32_t) i1[0];
    vacc1 += vi0x1;
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

    vfpacc0 += vmagic_bias;
    vfpacc1 += vmagic_bias;

    int32_t vout0 = (int32_t) float_as_uint32(vfpacc0) - vmagic_bias_less_output_zero_point;
    int32_t vout1 = (int32_t) float_as_uint32(vfpacc1) - vmagic_bias_less_output_zero_point;

    output[0] = (int8_t) vout0;
    output[1] = (int8_t) vout1;
    output += 2;
  }
  if XNN_UNLIKELY(channels != 0) {
    int32_t vacc = *buffer;
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
    vfpacc += vmagic_bias;
    int32_t vout = (int32_t) float_as_uint32(vfpacc) - vmagic_bias_less_output_zero_point;

    *output = (int8_t) vout;
  }
}
