// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/unipass-neon-mul8.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/intrinsics-polyfill.h"


void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__neonv8_mla8_ld128(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
  const int8x16_t voutput_min = vld1q_dup_s8(&params->fp32_neonv8.output_min);
  const int8x16_t voutput_max = vld1q_dup_s8(&params->fp32_neonv8.output_max);
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
    for (; c >= 16; c -= 16) {
      int32x4_t vacc0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
      int32x4_t vacc4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
      int32x4_t vacc89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
      int32x4_t vaccCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;

      const int8x16_t vi0x0123456789ABCDEF = vld1q_s8(i0); i0 += 16;
      const int8x16_t vk0x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      int16x8_t vprod01234567 = vmull_s8(vget_low_s8(vi0x0123456789ABCDEF), vget_low_s8(vk0x0123456789ABCDEF));
      int16x8_t vprod89ABCDEF = vmull_s8(vget_high_s8(vi0x0123456789ABCDEF), vget_high_s8(vk0x0123456789ABCDEF));

      const int8x16_t vi1x0123456789ABCDEF = vld1q_s8(i1); i1 += 16;
      const int8x16_t vk1x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmlal_s8(vprod01234567, vget_low_s8(vi1x0123456789ABCDEF), vget_low_s8(vk1x0123456789ABCDEF));
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vget_high_s8(vi1x0123456789ABCDEF), vget_high_s8(vk1x0123456789ABCDEF));

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x16_t vi2x0123456789ABCDEF = vld1q_s8(i2); i2 += 16;
      const int8x16_t vk2x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmull_s8(vget_low_s8(vi2x0123456789ABCDEF), vget_low_s8(vk2x0123456789ABCDEF));
      vprod89ABCDEF = vmull_s8(vget_high_s8(vi2x0123456789ABCDEF), vget_high_s8(vk2x0123456789ABCDEF));

      const int8x16_t vi3x0123456789ABCDEF = vld1q_s8(i3); i3 += 16;
      const int8x16_t vk3x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmlal_s8(vprod01234567, vget_low_s8(vi3x0123456789ABCDEF), vget_low_s8(vk3x0123456789ABCDEF));
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vget_high_s8(vi3x0123456789ABCDEF), vget_high_s8(vk3x0123456789ABCDEF));

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x16_t vi4x0123456789ABCDEF = vld1q_s8(i4); i4 += 16;
      const int8x16_t vk4x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmull_s8(vget_low_s8(vi4x0123456789ABCDEF), vget_low_s8(vk4x0123456789ABCDEF));
      vprod89ABCDEF = vmull_s8(vget_high_s8(vi4x0123456789ABCDEF), vget_high_s8(vk4x0123456789ABCDEF));

      const int8x16_t vi5x0123456789ABCDEF = vld1q_s8(i5); i5 += 16;
      const int8x16_t vk5x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmlal_s8(vprod01234567, vget_low_s8(vi5x0123456789ABCDEF), vget_low_s8(vk5x0123456789ABCDEF));
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vget_high_s8(vi5x0123456789ABCDEF), vget_high_s8(vk5x0123456789ABCDEF));

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x16_t vi6x0123456789ABCDEF = vld1q_s8(i6); i6 += 16;
      const int8x16_t vk6x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmull_s8(vget_low_s8(vi6x0123456789ABCDEF), vget_low_s8(vk6x0123456789ABCDEF));
      vprod89ABCDEF = vmull_s8(vget_high_s8(vi6x0123456789ABCDEF), vget_high_s8(vk6x0123456789ABCDEF));

      const int8x16_t vi7x0123456789ABCDEF = vld1q_s8(i7); i7 += 16;
      const int8x16_t vk7x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmlal_s8(vprod01234567, vget_low_s8(vi7x0123456789ABCDEF), vget_low_s8(vk7x0123456789ABCDEF));
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vget_high_s8(vi7x0123456789ABCDEF), vget_high_s8(vk7x0123456789ABCDEF));

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x16_t vi8x0123456789ABCDEF = vld1q_s8(i8); i8 += 16;
      const int8x16_t vk8x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmull_s8(vget_low_s8(vi8x0123456789ABCDEF), vget_low_s8(vk8x0123456789ABCDEF));
      vprod89ABCDEF = vmull_s8(vget_high_s8(vi8x0123456789ABCDEF), vget_high_s8(vk8x0123456789ABCDEF));

      const int8x16_t vi9x0123456789ABCDEF = vld1q_s8(i9); i9 += 16;
      const int8x16_t vk9x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmlal_s8(vprod01234567, vget_low_s8(vi9x0123456789ABCDEF), vget_low_s8(vk9x0123456789ABCDEF));
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vget_high_s8(vi9x0123456789ABCDEF), vget_high_s8(vk9x0123456789ABCDEF));

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x16_t vi10x0123456789ABCDEF = vld1q_s8(i10); i10 += 16;
      const int8x16_t vk10x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmull_s8(vget_low_s8(vi10x0123456789ABCDEF), vget_low_s8(vk10x0123456789ABCDEF));
      vprod89ABCDEF = vmull_s8(vget_high_s8(vi10x0123456789ABCDEF), vget_high_s8(vk10x0123456789ABCDEF));

      const int8x16_t vi11x0123456789ABCDEF = vld1q_s8(i11); i11 += 16;
      const int8x16_t vk11x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmlal_s8(vprod01234567, vget_low_s8(vi11x0123456789ABCDEF), vget_low_s8(vk11x0123456789ABCDEF));
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vget_high_s8(vi11x0123456789ABCDEF), vget_high_s8(vk11x0123456789ABCDEF));

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x16_t vi12x0123456789ABCDEF = vld1q_s8(i12); i12 += 16;
      const int8x16_t vk12x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmull_s8(vget_low_s8(vi12x0123456789ABCDEF), vget_low_s8(vk12x0123456789ABCDEF));
      vprod89ABCDEF = vmull_s8(vget_high_s8(vi12x0123456789ABCDEF), vget_high_s8(vk12x0123456789ABCDEF));

      const int8x16_t vi13x0123456789ABCDEF = vld1q_s8(i13); i13 += 16;
      const int8x16_t vk13x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmlal_s8(vprod01234567, vget_low_s8(vi13x0123456789ABCDEF), vget_low_s8(vk13x0123456789ABCDEF));
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vget_high_s8(vi13x0123456789ABCDEF), vget_high_s8(vk13x0123456789ABCDEF));

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x16_t vi14x0123456789ABCDEF = vld1q_s8(i14); i14 += 16;
      const int8x16_t vk14x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmull_s8(vget_low_s8(vi14x0123456789ABCDEF), vget_low_s8(vk14x0123456789ABCDEF));
      vprod89ABCDEF = vmull_s8(vget_high_s8(vi14x0123456789ABCDEF), vget_high_s8(vk14x0123456789ABCDEF));

      const int8x16_t vi15x0123456789ABCDEF = vld1q_s8(i15); i15 += 16;
      const int8x16_t vk15x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmlal_s8(vprod01234567, vget_low_s8(vi15x0123456789ABCDEF), vget_low_s8(vk15x0123456789ABCDEF));
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vget_high_s8(vi15x0123456789ABCDEF), vget_high_s8(vk15x0123456789ABCDEF));

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x16_t vi16x0123456789ABCDEF = vld1q_s8(i16); i16 += 16;
      const int8x16_t vk16x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmull_s8(vget_low_s8(vi16x0123456789ABCDEF), vget_low_s8(vk16x0123456789ABCDEF));
      vprod89ABCDEF = vmull_s8(vget_high_s8(vi16x0123456789ABCDEF), vget_high_s8(vk16x0123456789ABCDEF));

      const int8x16_t vi17x0123456789ABCDEF = vld1q_s8(i17); i17 += 16;
      const int8x16_t vk17x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmlal_s8(vprod01234567, vget_low_s8(vi17x0123456789ABCDEF), vget_low_s8(vk17x0123456789ABCDEF));
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vget_high_s8(vi17x0123456789ABCDEF), vget_high_s8(vk17x0123456789ABCDEF));

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x16_t vi18x0123456789ABCDEF = vld1q_s8(i18); i18 += 16;
      const int8x16_t vk18x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmull_s8(vget_low_s8(vi18x0123456789ABCDEF), vget_low_s8(vk18x0123456789ABCDEF));
      vprod89ABCDEF = vmull_s8(vget_high_s8(vi18x0123456789ABCDEF), vget_high_s8(vk18x0123456789ABCDEF));

      const int8x16_t vi19x0123456789ABCDEF = vld1q_s8(i19); i19 += 16;
      const int8x16_t vk19x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmlal_s8(vprod01234567, vget_low_s8(vi19x0123456789ABCDEF), vget_low_s8(vk19x0123456789ABCDEF));
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vget_high_s8(vi19x0123456789ABCDEF), vget_high_s8(vk19x0123456789ABCDEF));

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x16_t vi20x0123456789ABCDEF = vld1q_s8(i20); i20 += 16;
      const int8x16_t vk20x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmull_s8(vget_low_s8(vi20x0123456789ABCDEF), vget_low_s8(vk20x0123456789ABCDEF));
      vprod89ABCDEF = vmull_s8(vget_high_s8(vi20x0123456789ABCDEF), vget_high_s8(vk20x0123456789ABCDEF));

      const int8x16_t vi21x0123456789ABCDEF = vld1q_s8(i21); i21 += 16;
      const int8x16_t vk21x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmlal_s8(vprod01234567, vget_low_s8(vi21x0123456789ABCDEF), vget_low_s8(vk21x0123456789ABCDEF));
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vget_high_s8(vi21x0123456789ABCDEF), vget_high_s8(vk21x0123456789ABCDEF));

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x16_t vi22x0123456789ABCDEF = vld1q_s8(i22); i22 += 16;
      const int8x16_t vk22x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmull_s8(vget_low_s8(vi22x0123456789ABCDEF), vget_low_s8(vk22x0123456789ABCDEF));
      vprod89ABCDEF = vmull_s8(vget_high_s8(vi22x0123456789ABCDEF), vget_high_s8(vk22x0123456789ABCDEF));

      const int8x16_t vi23x0123456789ABCDEF = vld1q_s8(i23); i23 += 16;
      const int8x16_t vk23x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmlal_s8(vprod01234567, vget_low_s8(vi23x0123456789ABCDEF), vget_low_s8(vk23x0123456789ABCDEF));
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vget_high_s8(vi23x0123456789ABCDEF), vget_high_s8(vk23x0123456789ABCDEF));

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x16_t vi24x0123456789ABCDEF = vld1q_s8(i24); i24 += 16;
      const int8x16_t vk24x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmull_s8(vget_low_s8(vi24x0123456789ABCDEF), vget_low_s8(vk24x0123456789ABCDEF));
      vprod89ABCDEF = vmull_s8(vget_high_s8(vi24x0123456789ABCDEF), vget_high_s8(vk24x0123456789ABCDEF));

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

      float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
      float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);
      float32x4_t vfpacc89AB = vcvtq_f32_s32(vacc89AB);
      float32x4_t vfpaccCDEF = vcvtq_f32_s32(vaccCDEF);

      const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
      const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;
      const float32x4_t vscale89AB = vld1q_f32((const float*) w); w = (const float*) w + 4;
      const float32x4_t vscaleCDEF = vld1q_f32((const float*) w); w = (const float*) w + 4;

      vfpacc0123 = vmulq_f32(vfpacc0123, vscale0123);
      vfpacc4567 = vmulq_f32(vfpacc4567, vscale4567);
      vfpacc89AB = vmulq_f32(vfpacc89AB, vscale89AB);
      vfpaccCDEF = vmulq_f32(vfpaccCDEF, vscaleCDEF);

      vacc0123 = vcvtnq_s32_f32(vfpacc0123);
      vacc4567 = vcvtnq_s32_f32(vfpacc4567);
      vacc89AB = vcvtnq_s32_f32(vfpacc89AB);
      vaccCDEF = vcvtnq_s32_f32(vfpaccCDEF);

#if XNN_ARCH_ARM64
      int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
      int16x8_t vacc89ABCDEF = vqmovn_high_s32(vqmovn_s32(vacc89AB), vaccCDEF);

      vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);
      vacc89ABCDEF = vqaddq_s16(vacc89ABCDEF, voutput_zero_point);

      int8x16_t vout0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc01234567), vacc89ABCDEF);
#else  // !XNN_ARCH_ARM64
      int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
      int16x8_t vacc89ABCDEF = vcombine_s16(vqmovn_s32(vacc89AB), vqmovn_s32(vaccCDEF));

      vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);
      vacc89ABCDEF = vqaddq_s16(vacc89ABCDEF, voutput_zero_point);

      int8x16_t vout0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc01234567), vqmovn_s16(vacc89ABCDEF));
#endif  // !XNN_ARCH_ARM64

      vout0123456789ABCDEF = vmaxq_s8(vout0123456789ABCDEF, voutput_min);

      vout0123456789ABCDEF = vminq_s8(vout0123456789ABCDEF, voutput_max);

      vst1q_s8(output, vout0123456789ABCDEF); output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 16);
      do {
        int32x4_t vacc0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
        int32x4_t vacc4567 = vld1q_s32(w); w = (const int32_t*) w + 4;

        const int8x8_t vi0x01234567 = vld1_s8(i0); i0 += 8;
        const int8x8_t vk0x01234567 = vld1_s8(k); k += 8;

        int16x8_t vprod01234567 = vmull_s8(vi0x01234567, vk0x01234567);

        const int8x8_t vi1x01234567 = vld1_s8(i1); i1 += 8;
        const int8x8_t vk1x01234567 = vld1_s8((const void*) (k + 8));

        vprod01234567 = vmlal_s8(vprod01234567, vi1x01234567, vk1x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi2x01234567 = vld1_s8(i2); i2 += 8;
        const int8x8_t vk2x01234567 = vld1_s8((const void*) (k + 24));

        vprod01234567 = vmull_s8(vi2x01234567, vk2x01234567);

        const int8x8_t vi3x01234567 = vld1_s8(i3); i3 += 8;
        const int8x8_t vk3x01234567 = vld1_s8((const void*) (k + 40));

        vprod01234567 = vmlal_s8(vprod01234567, vi3x01234567, vk3x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi4x01234567 = vld1_s8(i4); i4 += 8;
        const int8x8_t vk4x01234567 = vld1_s8((const void*) (k + 56));

        vprod01234567 = vmull_s8(vi4x01234567, vk4x01234567);

        const int8x8_t vi5x01234567 = vld1_s8(i5); i5 += 8;
        const int8x8_t vk5x01234567 = vld1_s8((const void*) (k + 72));

        vprod01234567 = vmlal_s8(vprod01234567, vi5x01234567, vk5x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi6x01234567 = vld1_s8(i6); i6 += 8;
        const int8x8_t vk6x01234567 = vld1_s8((const void*) (k + 88));

        vprod01234567 = vmull_s8(vi6x01234567, vk6x01234567);

        const int8x8_t vi7x01234567 = vld1_s8(i7); i7 += 8;
        const int8x8_t vk7x01234567 = vld1_s8((const void*) (k + 104));

        vprod01234567 = vmlal_s8(vprod01234567, vi7x01234567, vk7x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi8x01234567 = vld1_s8(i8); i8 += 8;
        const int8x8_t vk8x01234567 = vld1_s8((const void*) (k + 120));

        vprod01234567 = vmull_s8(vi8x01234567, vk8x01234567);

        const int8x8_t vi9x01234567 = vld1_s8(i9); i9 += 8;
        const int8x8_t vk9x01234567 = vld1_s8((const void*) (k + 136));

        vprod01234567 = vmlal_s8(vprod01234567, vi9x01234567, vk9x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi10x01234567 = vld1_s8(i10); i10 += 8;
        const int8x8_t vk10x01234567 = vld1_s8((const void*) (k + 152));

        vprod01234567 = vmull_s8(vi10x01234567, vk10x01234567);

        const int8x8_t vi11x01234567 = vld1_s8(i11); i11 += 8;
        const int8x8_t vk11x01234567 = vld1_s8((const void*) (k + 168));

        vprod01234567 = vmlal_s8(vprod01234567, vi11x01234567, vk11x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi12x01234567 = vld1_s8(i12); i12 += 8;
        const int8x8_t vk12x01234567 = vld1_s8((const void*) (k + 184));

        vprod01234567 = vmull_s8(vi12x01234567, vk12x01234567);

        const int8x8_t vi13x01234567 = vld1_s8(i13); i13 += 8;
        const int8x8_t vk13x01234567 = vld1_s8((const void*) (k + 200));

        vprod01234567 = vmlal_s8(vprod01234567, vi13x01234567, vk13x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi14x01234567 = vld1_s8(i14); i14 += 8;
        const int8x8_t vk14x01234567 = vld1_s8((const void*) (k + 216));

        vprod01234567 = vmull_s8(vi14x01234567, vk14x01234567);

        const int8x8_t vi15x01234567 = vld1_s8(i15); i15 += 8;
        const int8x8_t vk15x01234567 = vld1_s8((const void*) (k + 232));

        vprod01234567 = vmlal_s8(vprod01234567, vi15x01234567, vk15x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi16x01234567 = vld1_s8(i16); i16 += 8;
        const int8x8_t vk16x01234567 = vld1_s8((const void*) (k + 248));

        vprod01234567 = vmull_s8(vi16x01234567, vk16x01234567);

        const int8x8_t vi17x01234567 = vld1_s8(i17); i17 += 8;
        const int8x8_t vk17x01234567 = vld1_s8((const void*) (k + 264));

        vprod01234567 = vmlal_s8(vprod01234567, vi17x01234567, vk17x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi18x01234567 = vld1_s8(i18); i18 += 8;
        const int8x8_t vk18x01234567 = vld1_s8((const void*) (k + 280));

        vprod01234567 = vmull_s8(vi18x01234567, vk18x01234567);

        const int8x8_t vi19x01234567 = vld1_s8(i19); i19 += 8;
        const int8x8_t vk19x01234567 = vld1_s8((const void*) (k + 296));

        vprod01234567 = vmlal_s8(vprod01234567, vi19x01234567, vk19x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi20x01234567 = vld1_s8(i20); i20 += 8;
        const int8x8_t vk20x01234567 = vld1_s8((const void*) (k + 312));

        vprod01234567 = vmull_s8(vi20x01234567, vk20x01234567);

        const int8x8_t vi21x01234567 = vld1_s8(i21); i21 += 8;
        const int8x8_t vk21x01234567 = vld1_s8((const void*) (k + 328));

        vprod01234567 = vmlal_s8(vprod01234567, vi21x01234567, vk21x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi22x01234567 = vld1_s8(i22); i22 += 8;
        const int8x8_t vk22x01234567 = vld1_s8((const void*) (k + 344));

        vprod01234567 = vmull_s8(vi22x01234567, vk22x01234567);

        const int8x8_t vi23x01234567 = vld1_s8(i23); i23 += 8;
        const int8x8_t vk23x01234567 = vld1_s8((const void*) (k + 360));

        vprod01234567 = vmlal_s8(vprod01234567, vi23x01234567, vk23x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi24x01234567 = vld1_s8(i24); i24 += 8;
        const int8x8_t vk24x01234567 = vld1_s8((const void*) (k + 376));

        vprod01234567 = vmull_s8(vi24x01234567, vk24x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));

        float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
        float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);

        const float32x4_t vscale0123 = vld1q_f32((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 400 * sizeof(int8_t)));
        const float32x4_t vscale4567 = vld1q_f32((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 400 * sizeof(int8_t) + 4 * sizeof(float)));
        vfpacc0123 = vmulq_f32(vfpacc0123, vscale0123);
        vfpacc4567 = vmulq_f32(vfpacc4567, vscale4567);

        vacc0123 = vcvtnq_s32_f32(vfpacc0123);
        vacc4567 = vcvtnq_s32_f32(vfpacc4567);

#if XNN_ARCH_ARM64
        int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
#else
        int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
#endif
        vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);

        int8x8_t vout01234567 = vqmovn_s16(vacc01234567);
        vout01234567 = vmax_s8(vout01234567, vget_low_s8(voutput_min));
        vout01234567 = vmin_s8(vout01234567, vget_low_s8(voutput_max));

        if XNN_LIKELY(c >= 8) {
          vst1_s8(output, vout01234567); output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            vst1_lane_u32((void*) output, vreinterpret_u32_s8(vout01234567), 0); output += 4;
            vout01234567 = vext_s8(vout01234567, vout01234567, 4);
          }
          if (c & 2) {
            vst1_lane_u16((void*) output, vreinterpret_u16_s8(vout01234567), 0); output += 2;
            vout01234567 = vext_s8(vout01234567, vout01234567, 2);
          }
          if (c & 1) {
            vst1_lane_s8(output, vout01234567, 0); output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
