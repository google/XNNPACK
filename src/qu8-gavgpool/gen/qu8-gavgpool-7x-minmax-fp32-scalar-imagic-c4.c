// Auto-generated file. Do not edit!
//   Template: src/qs8-gavgpool/unipass-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/gavgpool.h"
#include "xnnpack/math.h"


void xnn_qu8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c4(
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

  const int32_t vinit_bias = params->fp32_scalar_imagic.init_bias;
  const float vscale = params->fp32_scalar_imagic.scale;
  const float vmagic_bias = params->fp32_scalar_imagic.magic_bias;
  const int32_t vmagic_min = params->fp32_scalar_imagic.magic_min;
  const int32_t vmagic_max = params->fp32_scalar_imagic.magic_max;
  const int32_t vmagic_bias_less_zero_point = params->fp32_scalar_imagic.magic_bias_less_zero_point;
  for (; channels >= 4; channels -= 4) {
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

    float vfpacc0 = (float) vacc0 * vscale;
    float vfpacc1 = (float) vacc1 * vscale;
    float vfpacc2 = (float) vacc2 * vscale;
    float vfpacc3 = (float) vacc3 * vscale;

    vfpacc0 += vmagic_bias;
    vfpacc1 += vmagic_bias;
    vfpacc2 += vmagic_bias;
    vfpacc3 += vmagic_bias;

    int32_t vout0 = (int32_t) float_as_uint32(vfpacc0);
    int32_t vout1 = (int32_t) float_as_uint32(vfpacc1);
    int32_t vout2 = (int32_t) float_as_uint32(vfpacc2);
    int32_t vout3 = (int32_t) float_as_uint32(vfpacc3);

    vout0 = math_max_s32(vout0, vmagic_min);
    vout1 = math_max_s32(vout1, vmagic_min);
    vout2 = math_max_s32(vout2, vmagic_min);
    vout3 = math_max_s32(vout3, vmagic_min);

    vout0 = math_min_s32(vout0, vmagic_max);
    vout1 = math_min_s32(vout1, vmagic_max);
    vout2 = math_min_s32(vout2, vmagic_max);
    vout3 = math_min_s32(vout3, vmagic_max);

    vout0 -= vmagic_bias_less_zero_point;
    vout1 -= vmagic_bias_less_zero_point;
    vout2 -= vmagic_bias_less_zero_point;
    vout3 -= vmagic_bias_less_zero_point;

    output[0] = (uint8_t) vout0;
    output[1] = (uint8_t) vout1;
    output[2] = (uint8_t) vout2;
    output[3] = (uint8_t) vout3;
    output += 4;
  }
  if XNN_UNLIKELY(channels != 0) {
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
      vfpacc += vmagic_bias;
      int32_t vout = (int32_t) float_as_uint32(vfpacc);
      vout = math_max_s32(vout, vmagic_min);
      vout = math_min_s32(vout, vmagic_max);
      vout -= vmagic_bias_less_zero_point;

      *output++ = (uint8_t) vout;
    } while (--channels != 0);
  }
}
