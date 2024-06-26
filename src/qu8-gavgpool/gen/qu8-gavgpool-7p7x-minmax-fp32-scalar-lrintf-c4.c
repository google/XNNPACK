// Auto-generated file. Do not edit!
//   Template: src/qs8-gavgpool/multipass-scalar.c.in
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


void xnn_qu8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* output,
    const union xnn_qu8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows > 7);
  assert(channels != 0);

  const uint8_t* i0 = input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
  const uint8_t* i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
  const uint8_t* i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
  const uint8_t* i4 = (const uint8_t*) ((uintptr_t) i3 + input_stride);
  const uint8_t* i5 = (const uint8_t*) ((uintptr_t) i4 + input_stride);
  const uint8_t* i6 = (const uint8_t*) ((uintptr_t) i5 + input_stride);
  const size_t input_increment = 7 * input_stride - round_up_po2(channels, 4) * sizeof(uint8_t);

  const int32_t vinit_bias = params->fp32_scalar_lrintf.init_bias;
  int32_t* b = buffer;
  for (ptrdiff_t c = (ptrdiff_t) channels; c > 0; c -= 4) {
    const int32_t vi0x0 = (int32_t) i0[0];
    const int32_t vi0x1 = (int32_t) i0[1];
    const int32_t vi0x2 = (int32_t) i0[2];
    const int32_t vi0x3 = (int32_t) i0[3];
    i0 += 4;

    int32_t vacc0 = vi0x0 + vinit_bias;
    const int32_t vi1x0 = (int32_t) i1[0];
    int32_t vacc1 = vi0x1 + vinit_bias;
    const int32_t vi1x1 = (int32_t) i1[1];
    int32_t vacc2 = vi0x2 + vinit_bias;
    const int32_t vi1x2 = (int32_t) i1[2];
    int32_t vacc3 = vi0x3 + vinit_bias;
    const int32_t vi1x3 = (int32_t) i1[3];
    i1 += 4;

    vacc0 += vi1x0;
    const int32_t vi2x0 = (int32_t) i2[0];
    vacc1 += vi1x1;
    const int32_t vi2x1 = (int32_t) i2[1];
    vacc2 += vi1x2;
    const int32_t vi2x2 = (int32_t) i2[2];
    vacc3 += vi1x3;
    const int32_t vi2x3 = (int32_t) i2[3];
    i2 += 4;
    vacc0 += vi2x0;
    const int32_t vi3x0 = (int32_t) i3[0];
    vacc1 += vi2x1;
    const int32_t vi3x1 = (int32_t) i3[1];
    vacc2 += vi2x2;
    const int32_t vi3x2 = (int32_t) i3[2];
    vacc3 += vi2x3;
    const int32_t vi3x3 = (int32_t) i3[3];
    i3 += 4;
    vacc0 += vi3x0;
    const int32_t vi4x0 = (int32_t) i4[0];
    vacc1 += vi3x1;
    const int32_t vi4x1 = (int32_t) i4[1];
    vacc2 += vi3x2;
    const int32_t vi4x2 = (int32_t) i4[2];
    vacc3 += vi3x3;
    const int32_t vi4x3 = (int32_t) i4[3];
    i4 += 4;
    vacc0 += vi4x0;
    const int32_t vi5x0 = (int32_t) i5[0];
    vacc1 += vi4x1;
    const int32_t vi5x1 = (int32_t) i5[1];
    vacc2 += vi4x2;
    const int32_t vi5x2 = (int32_t) i5[2];
    vacc3 += vi4x3;
    const int32_t vi5x3 = (int32_t) i5[3];
    i5 += 4;
    vacc0 += vi5x0;
    const int32_t vi6x0 = (int32_t) i6[0];
    vacc1 += vi5x1;
    const int32_t vi6x1 = (int32_t) i6[1];
    vacc2 += vi5x2;
    const int32_t vi6x2 = (int32_t) i6[2];
    vacc3 += vi5x3;
    const int32_t vi6x3 = (int32_t) i6[3];
    i6 += 4;

    vacc0 += vi6x0;
    vacc1 += vi6x1;
    vacc2 += vi6x2;
    vacc3 += vi6x3;

    b[0] = vacc0;
    b[1] = vacc1;
    b[2] = vacc2;
    b[3] = vacc3;
    b += 4;
  }

  for (rows -= 7; rows > 7; rows -= 7) {
    i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
    i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
    i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
    i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
    i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
    i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
    i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);

    int32_t* b = buffer;
    for (ptrdiff_t c = (ptrdiff_t) channels; c > 0; c -= 4) {
      int32_t vacc0 = b[0];
      const int32_t vi0x0 = (int32_t) i0[0];
      int32_t vacc1 = b[1];
      const int32_t vi0x1 = (int32_t) i0[1];
      int32_t vacc2 = b[2];
      const int32_t vi0x2 = (int32_t) i0[2];
      int32_t vacc3 = b[3];
      const int32_t vi0x3 = (int32_t) i0[3];
      i0 += 4;

      vacc0 += vi0x0;
      const int32_t vi1x0 = (int32_t) i1[0];
      vacc1 += vi0x1;
      const int32_t vi1x1 = (int32_t) i1[1];
      vacc2 += vi0x2;
      const int32_t vi1x2 = (int32_t) i1[2];
      vacc3 += vi0x3;
      const int32_t vi1x3 = (int32_t) i1[3];
      i1 += 4;
      vacc0 += vi1x0;
      const int32_t vi2x0 = (int32_t) i2[0];
      vacc1 += vi1x1;
      const int32_t vi2x1 = (int32_t) i2[1];
      vacc2 += vi1x2;
      const int32_t vi2x2 = (int32_t) i2[2];
      vacc3 += vi1x3;
      const int32_t vi2x3 = (int32_t) i2[3];
      i2 += 4;
      vacc0 += vi2x0;
      const int32_t vi3x0 = (int32_t) i3[0];
      vacc1 += vi2x1;
      const int32_t vi3x1 = (int32_t) i3[1];
      vacc2 += vi2x2;
      const int32_t vi3x2 = (int32_t) i3[2];
      vacc3 += vi2x3;
      const int32_t vi3x3 = (int32_t) i3[3];
      i3 += 4;
      vacc0 += vi3x0;
      const int32_t vi4x0 = (int32_t) i4[0];
      vacc1 += vi3x1;
      const int32_t vi4x1 = (int32_t) i4[1];
      vacc2 += vi3x2;
      const int32_t vi4x2 = (int32_t) i4[2];
      vacc3 += vi3x3;
      const int32_t vi4x3 = (int32_t) i4[3];
      i4 += 4;
      vacc0 += vi4x0;
      const int32_t vi5x0 = (int32_t) i5[0];
      vacc1 += vi4x1;
      const int32_t vi5x1 = (int32_t) i5[1];
      vacc2 += vi4x2;
      const int32_t vi5x2 = (int32_t) i5[2];
      vacc3 += vi4x3;
      const int32_t vi5x3 = (int32_t) i5[3];
      i5 += 4;
      vacc0 += vi5x0;
      const int32_t vi6x0 = (int32_t) i6[0];
      vacc1 += vi5x1;
      const int32_t vi6x1 = (int32_t) i6[1];
      vacc2 += vi5x2;
      const int32_t vi6x2 = (int32_t) i6[2];
      vacc3 += vi5x3;
      const int32_t vi6x3 = (int32_t) i6[3];
      i6 += 4;

      vacc0 += vi6x0;
      vacc1 += vi6x1;
      vacc2 += vi6x2;
      vacc3 += vi6x3;

      b[0] = vacc0;
      b[1] = vacc1;
      b[2] = vacc2;
      b[3] = vacc3;
      b += 4;
    }
  }

  i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
  i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const float vscale = params->fp32_scalar_lrintf.scale;
  const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
  for (; channels >= 4; channels -= 4) {
    int32_t vacc0 = buffer[0];
    const int32_t vi0x0 = (int32_t) i0[0];
    int32_t vacc1 = buffer[1];
    const int32_t vi0x1 = (int32_t) i0[1];
    int32_t vacc2 = buffer[2];
    const int32_t vi0x2 = (int32_t) i0[2];
    int32_t vacc3 = buffer[3];
    const int32_t vi0x3 = (int32_t) i0[3];
    buffer += 4;
    i0 += 4;

    vacc0 += vi0x0;
    const int32_t vi1x0 = (int32_t) i1[0];
    vacc1 += vi0x1;
    const int32_t vi1x1 = (int32_t) i1[1];
    vacc2 += vi0x2;
    const int32_t vi1x2 = (int32_t) i1[2];
    vacc3 += vi0x3;
    const int32_t vi1x3 = (int32_t) i1[3];
    i1 += 4;
    vacc0 += vi1x0;
    const int32_t vi2x0 = (int32_t) i2[0];
    vacc1 += vi1x1;
    const int32_t vi2x1 = (int32_t) i2[1];
    vacc2 += vi1x2;
    const int32_t vi2x2 = (int32_t) i2[2];
    vacc3 += vi1x3;
    const int32_t vi2x3 = (int32_t) i2[3];
    i2 += 4;
    vacc0 += vi2x0;
    const int32_t vi3x0 = (int32_t) i3[0];
    vacc1 += vi2x1;
    const int32_t vi3x1 = (int32_t) i3[1];
    vacc2 += vi2x2;
    const int32_t vi3x2 = (int32_t) i3[2];
    vacc3 += vi2x3;
    const int32_t vi3x3 = (int32_t) i3[3];
    i3 += 4;
    vacc0 += vi3x0;
    const int32_t vi4x0 = (int32_t) i4[0];
    vacc1 += vi3x1;
    const int32_t vi4x1 = (int32_t) i4[1];
    vacc2 += vi3x2;
    const int32_t vi4x2 = (int32_t) i4[2];
    vacc3 += vi3x3;
    const int32_t vi4x3 = (int32_t) i4[3];
    i4 += 4;
    vacc0 += vi4x0;
    const int32_t vi5x0 = (int32_t) i5[0];
    vacc1 += vi4x1;
    const int32_t vi5x1 = (int32_t) i5[1];
    vacc2 += vi4x2;
    const int32_t vi5x2 = (int32_t) i5[2];
    vacc3 += vi4x3;
    const int32_t vi5x3 = (int32_t) i5[3];
    i5 += 4;
    vacc0 += vi5x0;
    const int32_t vi6x0 = (int32_t) i6[0];
    vacc1 += vi5x1;
    const int32_t vi6x1 = (int32_t) i6[1];
    vacc2 += vi5x2;
    const int32_t vi6x2 = (int32_t) i6[2];
    vacc3 += vi5x3;
    const int32_t vi6x3 = (int32_t) i6[3];
    i6 += 4;

    vacc0 += vi6x0;
    vacc1 += vi6x1;
    vacc2 += vi6x2;
    vacc3 += vi6x3;

    float vfpacc0 = (float) vacc0 * vscale;
    float vfpacc1 = (float) vacc1 * vscale;
    float vfpacc2 = (float) vacc2 * vscale;
    float vfpacc3 = (float) vacc3 * vscale;

    vfpacc0 = math_max_f32(vfpacc0, voutput_min_less_zero_point);
    vfpacc1 = math_max_f32(vfpacc1, voutput_min_less_zero_point);
    vfpacc2 = math_max_f32(vfpacc2, voutput_min_less_zero_point);
    vfpacc3 = math_max_f32(vfpacc3, voutput_min_less_zero_point);

    vfpacc0 = math_min_f32(vfpacc0, voutput_max_less_zero_point);
    vfpacc1 = math_min_f32(vfpacc1, voutput_max_less_zero_point);
    vfpacc2 = math_min_f32(vfpacc2, voutput_max_less_zero_point);
    vfpacc3 = math_min_f32(vfpacc3, voutput_max_less_zero_point);

    const int32_t vrndacc0 = (int32_t) lrintf(vfpacc0);
    const int32_t vrndacc1 = (int32_t) lrintf(vfpacc1);
    const int32_t vrndacc2 = (int32_t) lrintf(vfpacc2);
    const int32_t vrndacc3 = (int32_t) lrintf(vfpacc3);

    int32_t vout0 = vrndacc0 + voutput_zero_point;
    int32_t vout1 = vrndacc1 + voutput_zero_point;
    int32_t vout2 = vrndacc2 + voutput_zero_point;
    int32_t vout3 = vrndacc3 + voutput_zero_point;

    output[0] = (uint8_t) vout0;
    output[1] = (uint8_t) vout1;
    output[2] = (uint8_t) vout2;
    output[3] = (uint8_t) vout3;
    output += 4;
  }
  if XNN_UNLIKELY(channels != 0) {
    do {
      int32_t vacc = *buffer++;
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
}
