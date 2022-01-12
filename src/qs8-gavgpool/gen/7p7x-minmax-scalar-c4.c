// Auto-generated file. Do not edit!
//   Template: src/qs8-gavgpool/multipass-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <fp16.h>

#include <xnnpack/gavgpool.h>
#include <xnnpack/math.h>


void xnn_qs8_gavgpool_minmax_ukernel_7p7x__scalar_c4(
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
  const size_t input_increment = 7 * input_stride - round_up_po2(channels, 4);

  const int32_t vinit_bias = params->fp32_scalar_imagic.init_bias;
  int32_t* b = buffer;
  for (ptrdiff_t c = (ptrdiff_t) channels; c > 0; c -= 4) {
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

    b[0] = vacc0;
    b[1] = vacc1;
    b[2] = vacc2;
    b[3] = vacc3;
    b += 4;
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
    for (ptrdiff_t c = (ptrdiff_t) channels; c > 0; c -= 4) {
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


      int32_t vacc0 = b[0] + vacc0x0;
      int32_t vacc1 = b[1] + vacc0x1;
      int32_t vacc2 = b[2] + vacc0x2;
      int32_t vacc3 = b[3] + vacc0x3;

      b[0] = vacc0;
      b[1] = vacc1;
      b[2] = vacc2;
      b[3] = vacc3;
      b += 4;
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

  const float vscale = params->fp32_scalar_imagic.scale;
  const float vmagic_bias = params->fp32_scalar_imagic.magic_bias;
  const int32_t vmagic_min = params->fp32_scalar_imagic.magic_min;
  const int32_t vmagic_max = params->fp32_scalar_imagic.magic_max;
  const int32_t vmagic_bias_less_zero_point = params->fp32_scalar_imagic.magic_bias_less_zero_point;
  for (; channels >= 4; channels -= 4) {
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


    const int32_t vacc0 = buffer[0] + vacc0x0;
    const int32_t vacc1 = buffer[1] + vacc0x1;
    const int32_t vacc2 = buffer[2] + vacc0x2;
    const int32_t vacc3 = buffer[3] + vacc0x3;
    buffer += 4;

    float vfpacc0 = (float) vacc0 * vscale;
    float vfpacc1 = (float) vacc1 * vscale;
    float vfpacc2 = (float) vacc2 * vscale;
    float vfpacc3 = (float) vacc3 * vscale;

    vfpacc0 += vmagic_bias;
    vfpacc1 += vmagic_bias;
    vfpacc2 += vmagic_bias;
    vfpacc3 += vmagic_bias;

    int32_t vout0 = (int32_t) fp32_to_bits(vfpacc0);
    int32_t vout1 = (int32_t) fp32_to_bits(vfpacc1);
    int32_t vout2 = (int32_t) fp32_to_bits(vfpacc2);
    int32_t vout3 = (int32_t) fp32_to_bits(vfpacc3);

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

    output[0] = (int8_t) vout0;
    output[1] = (int8_t) vout1;
    output[2] = (int8_t) vout2;
    output[3] = (int8_t) vout3;
    output += 4;
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


      const int32_t vacc = (*buffer++) + vacc0;
      float vfpacc = (float) vacc * vscale;

      vfpacc += vmagic_bias;
      int32_t vout = (int32_t) fp32_to_bits(vfpacc);
      vout = math_max_s32(vout, vmagic_min);
      vout = math_min_s32(vout, vmagic_max);
      vout -= vmagic_bias_less_zero_point;

      *output++ = (int8_t) vout;
    } while (--channels != 0);
  }
}
