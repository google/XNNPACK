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

#include <xnnpack/gavgpool.h>
#include <xnnpack/math.h>


void xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c4(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
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

  const int32_t vinit_bias = params->fp32_scalar_lrintf.init_bias;
  const float vscale = params->fp32_scalar_lrintf.scale;
  const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
  while (channels >= 4) {
    const int32_t vi0x0 = i0[0];
    const int32_t vi0x1 = i0[1];
    const int32_t vi0x2 = i0[2];
    const int32_t vi0x3 = i0[3];
    i0 += 4;
    const int32_t vi1x0 = i1[0];
    const int32_t vi1x1 = i1[1];
    const int32_t vi1x2 = i1[2];
    const int32_t vi1x3 = i1[3];
    i1 += 4;
    const int32_t vi2x0 = i2[0];
    const int32_t vi2x1 = i2[1];
    const int32_t vi2x2 = i2[2];
    const int32_t vi2x3 = i2[3];
    i2 += 4;
    const int32_t vi3x0 = i3[0];
    const int32_t vi3x1 = i3[1];
    const int32_t vi3x2 = i3[2];
    const int32_t vi3x3 = i3[3];
    i3 += 4;
    const int32_t vi4x0 = i4[0];
    const int32_t vi4x1 = i4[1];
    const int32_t vi4x2 = i4[2];
    const int32_t vi4x3 = i4[3];
    i4 += 4;
    const int32_t vi5x0 = i5[0];
    const int32_t vi5x1 = i5[1];
    const int32_t vi5x2 = i5[2];
    const int32_t vi5x3 = i5[3];
    i5 += 4;
    const int32_t vi6x0 = i6[0];
    const int32_t vi6x1 = i6[1];
    const int32_t vi6x2 = i6[2];
    const int32_t vi6x3 = i6[3];
    i6 += 4;

    int32_t vacc0x0 = vi0x0 + vi1x0;
    int32_t vacc0x1 = vi0x1 + vi1x1;
    int32_t vacc0x2 = vi0x2 + vi1x2;
    int32_t vacc0x3 = vi0x3 + vi1x3;

    vacc0x0 += vi2x0;
    vacc0x1 += vi2x1;
    vacc0x2 += vi2x2;
    vacc0x3 += vi2x3;
    vacc0x0 += vi3x0;
    vacc0x1 += vi3x1;
    vacc0x2 += vi3x2;
    vacc0x3 += vi3x3;
    vacc0x0 += vi4x0;
    vacc0x1 += vi4x1;
    vacc0x2 += vi4x2;
    vacc0x3 += vi4x3;
    vacc0x0 += vi5x0;
    vacc0x1 += vi5x1;
    vacc0x2 += vi5x2;
    vacc0x3 += vi5x3;
    vacc0x0 += vi6x0;
    vacc0x1 += vi6x1;
    vacc0x2 += vi6x2;
    vacc0x3 += vi6x3;


    const int32_t vacc0 = vinit_bias + vacc0x0;
    const int32_t vacc1 = vinit_bias + vacc0x1;
    const int32_t vacc2 = vinit_bias + vacc0x2;
    const int32_t vacc3 = vinit_bias + vacc0x3;

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

    output[0] = vout0;
    output[1] = vout1;
    output[2] = vout2;
    output[3] = vout3;
    output += 4;

    channels -= 4;
  }
  if XNN_UNLIKELY(channels != 0) {
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


      const int32_t vacc = vinit_bias + vacc0;
      float vfpacc = (float) vacc * vscale;

      vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
      const int32_t vrndacc = (int32_t) lrintf(vfpacc);
      int32_t vout = vrndacc + voutput_zero_point;

      *output++ = (int8_t) vout;
    } while (--channels != 0);
  }
}
