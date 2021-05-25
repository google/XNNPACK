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


void xnn_qs8_gavgpool_minmax_ukernel_7x__scalar_c4(
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


    int32_t vacc0 = vbias + vacc0x0;
    int32_t vacc1 = vbias + vacc0x1;
    int32_t vacc2 = vbias + vacc0x2;
    int32_t vacc3 = vbias + vacc0x3;

    const int64_t vprod0 = (int64_t) vacc0 * (int64_t) vmultiplier;
    const int64_t vprod1 = (int64_t) vacc1 * (int64_t) vmultiplier;
    const int64_t vprod2 = (int64_t) vacc2 * (int64_t) vmultiplier;
    const int64_t vprod3 = (int64_t) vacc3 * (int64_t) vmultiplier;

    const int64_t vadjprod0 = vprod0 - (int64_t) (vacc0 < 0);
    const int64_t vadjprod1 = vprod1 - (int64_t) (vacc1 < 0);
    const int64_t vadjprod2 = vprod2 - (int64_t) (vacc2 < 0);
    const int64_t vadjprod3 = vprod3 - (int64_t) (vacc3 < 0);

    int32_t vout0 = (int32_t) asr_s64(vadjprod0 + vrounding, vshift);
    int32_t vout1 = (int32_t) asr_s64(vadjprod1 + vrounding, vshift);
    int32_t vout2 = (int32_t) asr_s64(vadjprod2 + vrounding, vshift);
    int32_t vout3 = (int32_t) asr_s64(vadjprod3 + vrounding, vshift);

    vout0 = XNN_UNPREDICTABLE(vout0 < vout_min) ? vout_min : vout0;
    vout1 = XNN_UNPREDICTABLE(vout1 < vout_min) ? vout_min : vout1;
    vout2 = XNN_UNPREDICTABLE(vout2 < vout_min) ? vout_min : vout2;
    vout3 = XNN_UNPREDICTABLE(vout3 < vout_min) ? vout_min : vout3;

    vout0 = XNN_UNPREDICTABLE(vout0 > vout_max) ? vout_max : vout0;
    vout1 = XNN_UNPREDICTABLE(vout1 > vout_max) ? vout_max : vout1;
    vout2 = XNN_UNPREDICTABLE(vout2 > vout_max) ? vout_max : vout2;
    vout3 = XNN_UNPREDICTABLE(vout3 > vout_max) ? vout_max : vout3;

    vout0 += voutput_zero_point;
    vout1 += voutput_zero_point;
    vout2 += voutput_zero_point;
    vout3 += voutput_zero_point;

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
}
