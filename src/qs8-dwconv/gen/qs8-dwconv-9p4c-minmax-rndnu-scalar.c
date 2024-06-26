// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/unipass-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"


void xnn_qs8_dwconv_minmax_rndnu_ukernel_9p4c__scalar(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const int32_t vmultiplier = params->rndnu_scalar.multiplier;
  const int64_t vrounding = params->rndnu_scalar.rounding;
  const uint32_t vshift = params->rndnu_scalar.shift;
  const int32_t voutput_min_less_zero_point = params->rndnu_scalar.output_min_less_zero_point;
  const int32_t voutput_max_less_zero_point = params->rndnu_scalar.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->rndnu_scalar.output_zero_point;
  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 4; c -= 4) {
      int32_t vacc0 = ((const int32_t*) w)[0];
      int32_t vacc1 = ((const int32_t*) w)[1];
      int32_t vacc2 = ((const int32_t*) w)[2];
      int32_t vacc3 = ((const int32_t*) w)[3];


      const int32_t vi0x0 = (int32_t) i0[0];
      const int32_t vi0x1 = (int32_t) i0[1];
      const int32_t vi0x2 = (int32_t) i0[2];
      const int32_t vi0x3 = (int32_t) i0[3];
      i0 += 4;

      const int32_t vk0x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[0];
      const int32_t vk0x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[1];
      const int32_t vk0x2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[2];
      const int32_t vk0x3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[3];

      vacc0 += vi0x0 * vk0x0;
      vacc1 += vi0x1 * vk0x1;
      vacc2 += vi0x2 * vk0x2;
      vacc3 += vi0x3 * vk0x3;

      const int32_t vi1x0 = (int32_t) i1[0];
      const int32_t vi1x1 = (int32_t) i1[1];
      const int32_t vi1x2 = (int32_t) i1[2];
      const int32_t vi1x3 = (int32_t) i1[3];
      i1 += 4;

      const int32_t vk1x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[4];
      const int32_t vk1x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[5];
      const int32_t vk1x2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[6];
      const int32_t vk1x3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[7];

      vacc0 += vi1x0 * vk1x0;
      vacc1 += vi1x1 * vk1x1;
      vacc2 += vi1x2 * vk1x2;
      vacc3 += vi1x3 * vk1x3;

      const int32_t vi2x0 = (int32_t) i2[0];
      const int32_t vi2x1 = (int32_t) i2[1];
      const int32_t vi2x2 = (int32_t) i2[2];
      const int32_t vi2x3 = (int32_t) i2[3];
      i2 += 4;

      const int32_t vk2x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[8];
      const int32_t vk2x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[9];
      const int32_t vk2x2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[10];
      const int32_t vk2x3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[11];

      vacc0 += vi2x0 * vk2x0;
      vacc1 += vi2x1 * vk2x1;
      vacc2 += vi2x2 * vk2x2;
      vacc3 += vi2x3 * vk2x3;

      const int32_t vi3x0 = (int32_t) i3[0];
      const int32_t vi3x1 = (int32_t) i3[1];
      const int32_t vi3x2 = (int32_t) i3[2];
      const int32_t vi3x3 = (int32_t) i3[3];
      i3 += 4;

      const int32_t vk3x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[12];
      const int32_t vk3x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[13];
      const int32_t vk3x2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[14];
      const int32_t vk3x3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[15];

      vacc0 += vi3x0 * vk3x0;
      vacc1 += vi3x1 * vk3x1;
      vacc2 += vi3x2 * vk3x2;
      vacc3 += vi3x3 * vk3x3;

      const int32_t vi4x0 = (int32_t) i4[0];
      const int32_t vi4x1 = (int32_t) i4[1];
      const int32_t vi4x2 = (int32_t) i4[2];
      const int32_t vi4x3 = (int32_t) i4[3];
      i4 += 4;

      const int32_t vk4x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[16];
      const int32_t vk4x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[17];
      const int32_t vk4x2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[18];
      const int32_t vk4x3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[19];

      vacc0 += vi4x0 * vk4x0;
      vacc1 += vi4x1 * vk4x1;
      vacc2 += vi4x2 * vk4x2;
      vacc3 += vi4x3 * vk4x3;

      const int32_t vi5x0 = (int32_t) i5[0];
      const int32_t vi5x1 = (int32_t) i5[1];
      const int32_t vi5x2 = (int32_t) i5[2];
      const int32_t vi5x3 = (int32_t) i5[3];
      i5 += 4;

      const int32_t vk5x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[20];
      const int32_t vk5x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[21];
      const int32_t vk5x2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[22];
      const int32_t vk5x3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[23];

      vacc0 += vi5x0 * vk5x0;
      vacc1 += vi5x1 * vk5x1;
      vacc2 += vi5x2 * vk5x2;
      vacc3 += vi5x3 * vk5x3;

      const int32_t vi6x0 = (int32_t) i6[0];
      const int32_t vi6x1 = (int32_t) i6[1];
      const int32_t vi6x2 = (int32_t) i6[2];
      const int32_t vi6x3 = (int32_t) i6[3];
      i6 += 4;

      const int32_t vk6x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[24];
      const int32_t vk6x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[25];
      const int32_t vk6x2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[26];
      const int32_t vk6x3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[27];

      vacc0 += vi6x0 * vk6x0;
      vacc1 += vi6x1 * vk6x1;
      vacc2 += vi6x2 * vk6x2;
      vacc3 += vi6x3 * vk6x3;

      const int32_t vi7x0 = (int32_t) i7[0];
      const int32_t vi7x1 = (int32_t) i7[1];
      const int32_t vi7x2 = (int32_t) i7[2];
      const int32_t vi7x3 = (int32_t) i7[3];
      i7 += 4;

      const int32_t vk7x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[28];
      const int32_t vk7x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[29];
      const int32_t vk7x2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[30];
      const int32_t vk7x3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[31];

      vacc0 += vi7x0 * vk7x0;
      vacc1 += vi7x1 * vk7x1;
      vacc2 += vi7x2 * vk7x2;
      vacc3 += vi7x3 * vk7x3;

      const int32_t vi8x0 = (int32_t) i8[0];
      const int32_t vi8x1 = (int32_t) i8[1];
      const int32_t vi8x2 = (int32_t) i8[2];
      const int32_t vi8x3 = (int32_t) i8[3];
      i8 += 4;

      const int32_t vk8x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[32];
      const int32_t vk8x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[33];
      const int32_t vk8x2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[34];
      const int32_t vk8x3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[35];

      vacc0 += vi8x0 * vk8x0;
      vacc1 += vi8x1 * vk8x1;
      vacc2 += vi8x2 * vk8x2;
      vacc3 += vi8x3 * vk8x3;

      w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t) + 36 * sizeof(int8_t));

      const int64_t vextacc0 = math_mulext_s32(vacc0, vmultiplier) + vrounding;
      const int64_t vextacc1 = math_mulext_s32(vacc1, vmultiplier) + vrounding;
      const int64_t vextacc2 = math_mulext_s32(vacc2, vmultiplier) + vrounding;
      const int64_t vextacc3 = math_mulext_s32(vacc3, vmultiplier) + vrounding;

      int32_t vout0 = (int32_t) math_asr_s64(vextacc0, vshift);
      int32_t vout1 = (int32_t) math_asr_s64(vextacc1, vshift);
      int32_t vout2 = (int32_t) math_asr_s64(vextacc2, vshift);
      int32_t vout3 = (int32_t) math_asr_s64(vextacc3, vshift);

      vout0 = math_max_s32(vout0, voutput_min_less_zero_point);
      vout1 = math_max_s32(vout1, voutput_min_less_zero_point);
      vout2 = math_max_s32(vout2, voutput_min_less_zero_point);
      vout3 = math_max_s32(vout3, voutput_min_less_zero_point);

      vout0 = math_min_s32(vout0, voutput_max_less_zero_point);
      vout1 = math_min_s32(vout1, voutput_max_less_zero_point);
      vout2 = math_min_s32(vout2, voutput_max_less_zero_point);
      vout3 = math_min_s32(vout3, voutput_max_less_zero_point);

      vout0 += voutput_zero_point;
      vout1 += voutput_zero_point;
      vout2 += voutput_zero_point;
      vout3 += voutput_zero_point;

      output[0] = (int8_t) vout0;
      output[1] = (int8_t) vout1;
      output[2] = (int8_t) vout2;
      output[3] = (int8_t) vout3;
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t));
      do {
        int32_t vacc = *((const int32_t*) w);
        w = (const int32_t*) w + 1;

        const int32_t vi0 = (int32_t) *i0++;
        const int32_t vk0 = (int32_t) k[0];
        vacc += vi0 * vk0;
        const int32_t vi1 = (int32_t) *i1++;
        const int32_t vk1 = (int32_t) k[4];
        vacc += vi1 * vk1;
        const int32_t vi2 = (int32_t) *i2++;
        const int32_t vk2 = (int32_t) k[8];
        vacc += vi2 * vk2;
        const int32_t vi3 = (int32_t) *i3++;
        const int32_t vk3 = (int32_t) k[12];
        vacc += vi3 * vk3;
        const int32_t vi4 = (int32_t) *i4++;
        const int32_t vk4 = (int32_t) k[16];
        vacc += vi4 * vk4;
        const int32_t vi5 = (int32_t) *i5++;
        const int32_t vk5 = (int32_t) k[20];
        vacc += vi5 * vk5;
        const int32_t vi6 = (int32_t) *i6++;
        const int32_t vk6 = (int32_t) k[24];
        vacc += vi6 * vk6;
        const int32_t vi7 = (int32_t) *i7++;
        const int32_t vk7 = (int32_t) k[28];
        vacc += vi7 * vk7;
        const int32_t vi8 = (int32_t) *i8++;
        const int32_t vk8 = (int32_t) k[32];
        vacc += vi8 * vk8;
        k += 1;

        const int64_t vextacc = math_mulext_s32(vacc, vmultiplier) + vrounding;
        int32_t vout = (int32_t) math_asr_s64(vextacc, vshift);
        vout = math_max_s32(vout, voutput_min_less_zero_point);
        vout = math_min_s32(vout, voutput_max_less_zero_point);
        vout += voutput_zero_point;

        *output++ = (int8_t) vout;
      } while (--c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
