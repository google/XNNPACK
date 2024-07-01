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
#include "xnnpack/unaligned.h"


void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p2c__scalar_fmagic(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float voutput_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
  const float vmagic_bias = params->fp32_scalar_fmagic.magic_bias;
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;
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
    const int8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const int8_t*) ((uintptr_t) i9 + input_offset);
    }
    const int8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const int8_t*) ((uintptr_t) i10 + input_offset);
    }
    const int8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const int8_t*) ((uintptr_t) i11 + input_offset);
    }
    const int8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const int8_t*) ((uintptr_t) i12 + input_offset);
    }
    const int8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const int8_t*) ((uintptr_t) i13 + input_offset);
    }
    const int8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const int8_t*) ((uintptr_t) i14 + input_offset);
    }
    const int8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const int8_t*) ((uintptr_t) i15 + input_offset);
    }
    const int8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const int8_t*) ((uintptr_t) i16 + input_offset);
    }
    const int8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const int8_t*) ((uintptr_t) i17 + input_offset);
    }
    const int8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const int8_t*) ((uintptr_t) i18 + input_offset);
    }
    const int8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const int8_t*) ((uintptr_t) i19 + input_offset);
    }
    const int8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const int8_t*) ((uintptr_t) i20 + input_offset);
    }
    const int8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const int8_t*) ((uintptr_t) i21 + input_offset);
    }
    const int8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const int8_t*) ((uintptr_t) i22 + input_offset);
    }
    const int8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const int8_t*) ((uintptr_t) i23 + input_offset);
    }
    const int8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const int8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 2; c -= 2) {
      int32_t vacc0 = unaligned_indexed_load_s32(w, 0);
      int32_t vacc1 = unaligned_indexed_load_s32(w, 1);


      const int32_t vi0x0 = (int32_t) i0[0];
      const int32_t vi0x1 = (int32_t) i0[1];
      i0 += 2;

      const int32_t vk0x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
      const int32_t vk0x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[1];

      vacc0 += vi0x0 * vk0x0;
      vacc1 += vi0x1 * vk0x1;

      const int32_t vi1x0 = (int32_t) i1[0];
      const int32_t vi1x1 = (int32_t) i1[1];
      i1 += 2;

      const int32_t vk1x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
      const int32_t vk1x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[3];

      vacc0 += vi1x0 * vk1x0;
      vacc1 += vi1x1 * vk1x1;

      const int32_t vi2x0 = (int32_t) i2[0];
      const int32_t vi2x1 = (int32_t) i2[1];
      i2 += 2;

      const int32_t vk2x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
      const int32_t vk2x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[5];

      vacc0 += vi2x0 * vk2x0;
      vacc1 += vi2x1 * vk2x1;

      const int32_t vi3x0 = (int32_t) i3[0];
      const int32_t vi3x1 = (int32_t) i3[1];
      i3 += 2;

      const int32_t vk3x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6];
      const int32_t vk3x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[7];

      vacc0 += vi3x0 * vk3x0;
      vacc1 += vi3x1 * vk3x1;

      const int32_t vi4x0 = (int32_t) i4[0];
      const int32_t vi4x1 = (int32_t) i4[1];
      i4 += 2;

      const int32_t vk4x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8];
      const int32_t vk4x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[9];

      vacc0 += vi4x0 * vk4x0;
      vacc1 += vi4x1 * vk4x1;

      const int32_t vi5x0 = (int32_t) i5[0];
      const int32_t vi5x1 = (int32_t) i5[1];
      i5 += 2;

      const int32_t vk5x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10];
      const int32_t vk5x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[11];

      vacc0 += vi5x0 * vk5x0;
      vacc1 += vi5x1 * vk5x1;

      const int32_t vi6x0 = (int32_t) i6[0];
      const int32_t vi6x1 = (int32_t) i6[1];
      i6 += 2;

      const int32_t vk6x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12];
      const int32_t vk6x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[13];

      vacc0 += vi6x0 * vk6x0;
      vacc1 += vi6x1 * vk6x1;

      const int32_t vi7x0 = (int32_t) i7[0];
      const int32_t vi7x1 = (int32_t) i7[1];
      i7 += 2;

      const int32_t vk7x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14];
      const int32_t vk7x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[15];

      vacc0 += vi7x0 * vk7x0;
      vacc1 += vi7x1 * vk7x1;

      const int32_t vi8x0 = (int32_t) i8[0];
      const int32_t vi8x1 = (int32_t) i8[1];
      i8 += 2;

      const int32_t vk8x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16];
      const int32_t vk8x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[17];

      vacc0 += vi8x0 * vk8x0;
      vacc1 += vi8x1 * vk8x1;

      const int32_t vi9x0 = (int32_t) i9[0];
      const int32_t vi9x1 = (int32_t) i9[1];
      i9 += 2;

      const int32_t vk9x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[18];
      const int32_t vk9x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[19];

      vacc0 += vi9x0 * vk9x0;
      vacc1 += vi9x1 * vk9x1;

      const int32_t vi10x0 = (int32_t) i10[0];
      const int32_t vi10x1 = (int32_t) i10[1];
      i10 += 2;

      const int32_t vk10x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[20];
      const int32_t vk10x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[21];

      vacc0 += vi10x0 * vk10x0;
      vacc1 += vi10x1 * vk10x1;

      const int32_t vi11x0 = (int32_t) i11[0];
      const int32_t vi11x1 = (int32_t) i11[1];
      i11 += 2;

      const int32_t vk11x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[22];
      const int32_t vk11x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[23];

      vacc0 += vi11x0 * vk11x0;
      vacc1 += vi11x1 * vk11x1;

      const int32_t vi12x0 = (int32_t) i12[0];
      const int32_t vi12x1 = (int32_t) i12[1];
      i12 += 2;

      const int32_t vk12x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[24];
      const int32_t vk12x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[25];

      vacc0 += vi12x0 * vk12x0;
      vacc1 += vi12x1 * vk12x1;

      const int32_t vi13x0 = (int32_t) i13[0];
      const int32_t vi13x1 = (int32_t) i13[1];
      i13 += 2;

      const int32_t vk13x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[26];
      const int32_t vk13x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[27];

      vacc0 += vi13x0 * vk13x0;
      vacc1 += vi13x1 * vk13x1;

      const int32_t vi14x0 = (int32_t) i14[0];
      const int32_t vi14x1 = (int32_t) i14[1];
      i14 += 2;

      const int32_t vk14x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[28];
      const int32_t vk14x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[29];

      vacc0 += vi14x0 * vk14x0;
      vacc1 += vi14x1 * vk14x1;

      const int32_t vi15x0 = (int32_t) i15[0];
      const int32_t vi15x1 = (int32_t) i15[1];
      i15 += 2;

      const int32_t vk15x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[30];
      const int32_t vk15x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[31];

      vacc0 += vi15x0 * vk15x0;
      vacc1 += vi15x1 * vk15x1;

      const int32_t vi16x0 = (int32_t) i16[0];
      const int32_t vi16x1 = (int32_t) i16[1];
      i16 += 2;

      const int32_t vk16x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[32];
      const int32_t vk16x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[33];

      vacc0 += vi16x0 * vk16x0;
      vacc1 += vi16x1 * vk16x1;

      const int32_t vi17x0 = (int32_t) i17[0];
      const int32_t vi17x1 = (int32_t) i17[1];
      i17 += 2;

      const int32_t vk17x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[34];
      const int32_t vk17x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[35];

      vacc0 += vi17x0 * vk17x0;
      vacc1 += vi17x1 * vk17x1;

      const int32_t vi18x0 = (int32_t) i18[0];
      const int32_t vi18x1 = (int32_t) i18[1];
      i18 += 2;

      const int32_t vk18x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[36];
      const int32_t vk18x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[37];

      vacc0 += vi18x0 * vk18x0;
      vacc1 += vi18x1 * vk18x1;

      const int32_t vi19x0 = (int32_t) i19[0];
      const int32_t vi19x1 = (int32_t) i19[1];
      i19 += 2;

      const int32_t vk19x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[38];
      const int32_t vk19x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[39];

      vacc0 += vi19x0 * vk19x0;
      vacc1 += vi19x1 * vk19x1;

      const int32_t vi20x0 = (int32_t) i20[0];
      const int32_t vi20x1 = (int32_t) i20[1];
      i20 += 2;

      const int32_t vk20x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[40];
      const int32_t vk20x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[41];

      vacc0 += vi20x0 * vk20x0;
      vacc1 += vi20x1 * vk20x1;

      const int32_t vi21x0 = (int32_t) i21[0];
      const int32_t vi21x1 = (int32_t) i21[1];
      i21 += 2;

      const int32_t vk21x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[42];
      const int32_t vk21x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[43];

      vacc0 += vi21x0 * vk21x0;
      vacc1 += vi21x1 * vk21x1;

      const int32_t vi22x0 = (int32_t) i22[0];
      const int32_t vi22x1 = (int32_t) i22[1];
      i22 += 2;

      const int32_t vk22x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[44];
      const int32_t vk22x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[45];

      vacc0 += vi22x0 * vk22x0;
      vacc1 += vi22x1 * vk22x1;

      const int32_t vi23x0 = (int32_t) i23[0];
      const int32_t vi23x1 = (int32_t) i23[1];
      i23 += 2;

      const int32_t vk23x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[46];
      const int32_t vk23x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[47];

      vacc0 += vi23x0 * vk23x0;
      vacc1 += vi23x1 * vk23x1;

      const int32_t vi24x0 = (int32_t) i24[0];
      const int32_t vi24x1 = (int32_t) i24[1];
      i24 += 2;

      const int32_t vk24x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[48];
      const int32_t vk24x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[49];

      vacc0 += vi24x0 * vk24x0;
      vacc1 += vi24x1 * vk24x1;

      w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 50 * sizeof(int8_t));

      float vfpacc0 = (float) vacc0;
      float vfpacc1 = (float) vacc1;

      const float vscale0 = unaligned_indexed_load_f32(w, 0);
      const float vscale1 = unaligned_indexed_load_f32(w, 1);
      w = (const void*) ((const float*) w + 2);

      vfpacc0 *= vscale0;
      vfpacc1 *= vscale1;

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
    if XNN_UNLIKELY(c != 0) {
      int32_t vacc = unaligned_load_s32(w);

      const int32_t vi0 = (int32_t) *i0;
      const int32_t vk0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
      vacc += vi0 * vk0;
      const int32_t vi1 = (int32_t) *i1;
      const int32_t vk1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
      vacc += vi1 * vk1;
      const int32_t vi2 = (int32_t) *i2;
      const int32_t vk2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
      vacc += vi2 * vk2;
      const int32_t vi3 = (int32_t) *i3;
      const int32_t vk3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6];
      vacc += vi3 * vk3;
      const int32_t vi4 = (int32_t) *i4;
      const int32_t vk4 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8];
      vacc += vi4 * vk4;
      const int32_t vi5 = (int32_t) *i5;
      const int32_t vk5 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10];
      vacc += vi5 * vk5;
      const int32_t vi6 = (int32_t) *i6;
      const int32_t vk6 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12];
      vacc += vi6 * vk6;
      const int32_t vi7 = (int32_t) *i7;
      const int32_t vk7 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14];
      vacc += vi7 * vk7;
      const int32_t vi8 = (int32_t) *i8;
      const int32_t vk8 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16];
      vacc += vi8 * vk8;
      const int32_t vi9 = (int32_t) *i9;
      const int32_t vk9 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[18];
      vacc += vi9 * vk9;
      const int32_t vi10 = (int32_t) *i10;
      const int32_t vk10 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[20];
      vacc += vi10 * vk10;
      const int32_t vi11 = (int32_t) *i11;
      const int32_t vk11 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[22];
      vacc += vi11 * vk11;
      const int32_t vi12 = (int32_t) *i12;
      const int32_t vk12 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[24];
      vacc += vi12 * vk12;
      const int32_t vi13 = (int32_t) *i13;
      const int32_t vk13 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[26];
      vacc += vi13 * vk13;
      const int32_t vi14 = (int32_t) *i14;
      const int32_t vk14 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[28];
      vacc += vi14 * vk14;
      const int32_t vi15 = (int32_t) *i15;
      const int32_t vk15 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[30];
      vacc += vi15 * vk15;
      const int32_t vi16 = (int32_t) *i16;
      const int32_t vk16 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[32];
      vacc += vi16 * vk16;
      const int32_t vi17 = (int32_t) *i17;
      const int32_t vk17 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[34];
      vacc += vi17 * vk17;
      const int32_t vi18 = (int32_t) *i18;
      const int32_t vk18 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[36];
      vacc += vi18 * vk18;
      const int32_t vi19 = (int32_t) *i19;
      const int32_t vk19 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[38];
      vacc += vi19 * vk19;
      const int32_t vi20 = (int32_t) *i20;
      const int32_t vk20 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[40];
      vacc += vi20 * vk20;
      const int32_t vi21 = (int32_t) *i21;
      const int32_t vk21 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[42];
      vacc += vi21 * vk21;
      const int32_t vi22 = (int32_t) *i22;
      const int32_t vk22 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[44];
      vacc += vi22 * vk22;
      const int32_t vi23 = (int32_t) *i23;
      const int32_t vk23 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[46];
      vacc += vi23 * vk23;
      const int32_t vi24 = (int32_t) *i24;
      const int32_t vk24 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[48];
      vacc += vi24 * vk24;

      const float vscale = unaligned_load_f32((const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 50 * sizeof(int8_t)));
      float vfpacc = (float) vacc * vscale;

      vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
      vfpacc += vmagic_bias;
      int32_t vout = (int32_t) float_as_uint32(vfpacc) - vmagic_bias_less_output_zero_point;

      *output++ = (int8_t) vout;
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
