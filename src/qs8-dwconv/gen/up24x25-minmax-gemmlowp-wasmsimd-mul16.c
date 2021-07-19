// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/unipass-wasmsimd-mul16.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/dwconv.h>


void xnn_qs8_dwconv_minmax_gemmlowp_ukernel_up24x25__wasmsimd_mul16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN XNN_DISABLE_MSAN
{
  assert(channels != 0);
  assert(output_width != 0);

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
    for (; c >= 24; c -= 24) {
      v128_t vacc0123 = wasm_v128_load(w);
      v128_t vacc4567 = wasm_v128_load((const void*) ((uintptr_t) w + 4 * sizeof(int32_t)));
      v128_t vacc89AB = wasm_v128_load((const void*) ((uintptr_t) w + 8 * sizeof(int32_t)));
      v128_t vaccCDEF = wasm_v128_load((const void*) ((uintptr_t) w + 12 * sizeof(int32_t)));
      v128_t vaccGHIJ = wasm_v128_load((const void*) ((uintptr_t) w + 16 * sizeof(int32_t)));
      v128_t vaccKLMN = wasm_v128_load((const void*) ((uintptr_t) w + 20 * sizeof(int32_t)));


      const v128_t vi0x01234567 = wasm_i16x8_load8x8(i0);
      const v128_t vk0x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      const v128_t vi0x89ABCDEF = wasm_i16x8_load8x8(i0 + 8);
      const v128_t vk0x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      const v128_t vi0xGHIJKLMN = wasm_i16x8_load8x8(i0 + 16);
      const v128_t vk0xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      i0 += 24;

      const v128_t vprod0x01234567 = wasm_i16x8_mul(vi0x01234567, vk0x01234567);
      const v128_t vprod0x89ABCDEF = wasm_i16x8_mul(vi0x89ABCDEF, vk0x89ABCDEF);
      const v128_t vprod0xGHIJKLMN = wasm_i16x8_mul(vi0xGHIJKLMN, vk0xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod0x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod0x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod0x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod0x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod0xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod0xGHIJKLMN));

      const v128_t vi1x01234567 = wasm_i16x8_load8x8(i1);
      const v128_t vk1x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      const v128_t vi1x89ABCDEF = wasm_i16x8_load8x8(i1 + 8);
      const v128_t vk1x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      const v128_t vi1xGHIJKLMN = wasm_i16x8_load8x8(i1 + 16);
      const v128_t vk1xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      i1 += 24;

      const v128_t vprod1x01234567 = wasm_i16x8_mul(vi1x01234567, vk1x01234567);
      const v128_t vprod1x89ABCDEF = wasm_i16x8_mul(vi1x89ABCDEF, vk1x89ABCDEF);
      const v128_t vprod1xGHIJKLMN = wasm_i16x8_mul(vi1xGHIJKLMN, vk1xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod1x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod1x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod1x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod1x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod1xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod1xGHIJKLMN));

      const v128_t vi2x01234567 = wasm_i16x8_load8x8(i2);
      const v128_t vk2x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      const v128_t vi2x89ABCDEF = wasm_i16x8_load8x8(i2 + 8);
      const v128_t vk2x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      const v128_t vi2xGHIJKLMN = wasm_i16x8_load8x8(i2 + 16);
      const v128_t vk2xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      i2 += 24;

      const v128_t vprod2x01234567 = wasm_i16x8_mul(vi2x01234567, vk2x01234567);
      const v128_t vprod2x89ABCDEF = wasm_i16x8_mul(vi2x89ABCDEF, vk2x89ABCDEF);
      const v128_t vprod2xGHIJKLMN = wasm_i16x8_mul(vi2xGHIJKLMN, vk2xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod2x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod2x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod2x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod2x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod2xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod2xGHIJKLMN));

      const v128_t vi3x01234567 = wasm_i16x8_load8x8(i3);
      const v128_t vk3x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 72 * sizeof(int8_t)));
      const v128_t vi3x89ABCDEF = wasm_i16x8_load8x8(i3 + 8);
      const v128_t vk3x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 80 * sizeof(int8_t)));
      const v128_t vi3xGHIJKLMN = wasm_i16x8_load8x8(i3 + 16);
      const v128_t vk3xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 88 * sizeof(int8_t)));
      i3 += 24;

      const v128_t vprod3x01234567 = wasm_i16x8_mul(vi3x01234567, vk3x01234567);
      const v128_t vprod3x89ABCDEF = wasm_i16x8_mul(vi3x89ABCDEF, vk3x89ABCDEF);
      const v128_t vprod3xGHIJKLMN = wasm_i16x8_mul(vi3xGHIJKLMN, vk3xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod3x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod3x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod3x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod3x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod3xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod3xGHIJKLMN));

      const v128_t vi4x01234567 = wasm_i16x8_load8x8(i4);
      const v128_t vk4x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 96 * sizeof(int8_t)));
      const v128_t vi4x89ABCDEF = wasm_i16x8_load8x8(i4 + 8);
      const v128_t vk4x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 104 * sizeof(int8_t)));
      const v128_t vi4xGHIJKLMN = wasm_i16x8_load8x8(i4 + 16);
      const v128_t vk4xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 112 * sizeof(int8_t)));
      i4 += 24;

      const v128_t vprod4x01234567 = wasm_i16x8_mul(vi4x01234567, vk4x01234567);
      const v128_t vprod4x89ABCDEF = wasm_i16x8_mul(vi4x89ABCDEF, vk4x89ABCDEF);
      const v128_t vprod4xGHIJKLMN = wasm_i16x8_mul(vi4xGHIJKLMN, vk4xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod4x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod4x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod4x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod4x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod4xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod4xGHIJKLMN));

      const v128_t vi5x01234567 = wasm_i16x8_load8x8(i5);
      const v128_t vk5x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 120 * sizeof(int8_t)));
      const v128_t vi5x89ABCDEF = wasm_i16x8_load8x8(i5 + 8);
      const v128_t vk5x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 128 * sizeof(int8_t)));
      const v128_t vi5xGHIJKLMN = wasm_i16x8_load8x8(i5 + 16);
      const v128_t vk5xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 136 * sizeof(int8_t)));
      i5 += 24;

      const v128_t vprod5x01234567 = wasm_i16x8_mul(vi5x01234567, vk5x01234567);
      const v128_t vprod5x89ABCDEF = wasm_i16x8_mul(vi5x89ABCDEF, vk5x89ABCDEF);
      const v128_t vprod5xGHIJKLMN = wasm_i16x8_mul(vi5xGHIJKLMN, vk5xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod5x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod5x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod5x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod5x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod5xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod5xGHIJKLMN));

      const v128_t vi6x01234567 = wasm_i16x8_load8x8(i6);
      const v128_t vk6x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 144 * sizeof(int8_t)));
      const v128_t vi6x89ABCDEF = wasm_i16x8_load8x8(i6 + 8);
      const v128_t vk6x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 152 * sizeof(int8_t)));
      const v128_t vi6xGHIJKLMN = wasm_i16x8_load8x8(i6 + 16);
      const v128_t vk6xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 160 * sizeof(int8_t)));
      i6 += 24;

      const v128_t vprod6x01234567 = wasm_i16x8_mul(vi6x01234567, vk6x01234567);
      const v128_t vprod6x89ABCDEF = wasm_i16x8_mul(vi6x89ABCDEF, vk6x89ABCDEF);
      const v128_t vprod6xGHIJKLMN = wasm_i16x8_mul(vi6xGHIJKLMN, vk6xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod6x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod6x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod6x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod6x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod6xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod6xGHIJKLMN));

      const v128_t vi7x01234567 = wasm_i16x8_load8x8(i7);
      const v128_t vk7x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 168 * sizeof(int8_t)));
      const v128_t vi7x89ABCDEF = wasm_i16x8_load8x8(i7 + 8);
      const v128_t vk7x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 176 * sizeof(int8_t)));
      const v128_t vi7xGHIJKLMN = wasm_i16x8_load8x8(i7 + 16);
      const v128_t vk7xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 184 * sizeof(int8_t)));
      i7 += 24;

      const v128_t vprod7x01234567 = wasm_i16x8_mul(vi7x01234567, vk7x01234567);
      const v128_t vprod7x89ABCDEF = wasm_i16x8_mul(vi7x89ABCDEF, vk7x89ABCDEF);
      const v128_t vprod7xGHIJKLMN = wasm_i16x8_mul(vi7xGHIJKLMN, vk7xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod7x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod7x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod7x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod7x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod7xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod7xGHIJKLMN));

      const v128_t vi8x01234567 = wasm_i16x8_load8x8(i8);
      const v128_t vk8x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 192 * sizeof(int8_t)));
      const v128_t vi8x89ABCDEF = wasm_i16x8_load8x8(i8 + 8);
      const v128_t vk8x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 200 * sizeof(int8_t)));
      const v128_t vi8xGHIJKLMN = wasm_i16x8_load8x8(i8 + 16);
      const v128_t vk8xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 208 * sizeof(int8_t)));
      i8 += 24;

      const v128_t vprod8x01234567 = wasm_i16x8_mul(vi8x01234567, vk8x01234567);
      const v128_t vprod8x89ABCDEF = wasm_i16x8_mul(vi8x89ABCDEF, vk8x89ABCDEF);
      const v128_t vprod8xGHIJKLMN = wasm_i16x8_mul(vi8xGHIJKLMN, vk8xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod8x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod8x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod8x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod8x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod8xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod8xGHIJKLMN));

      const v128_t vi9x01234567 = wasm_i16x8_load8x8(i9);
      const v128_t vk9x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 216 * sizeof(int8_t)));
      const v128_t vi9x89ABCDEF = wasm_i16x8_load8x8(i9 + 8);
      const v128_t vk9x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 224 * sizeof(int8_t)));
      const v128_t vi9xGHIJKLMN = wasm_i16x8_load8x8(i9 + 16);
      const v128_t vk9xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 232 * sizeof(int8_t)));
      i9 += 24;

      const v128_t vprod9x01234567 = wasm_i16x8_mul(vi9x01234567, vk9x01234567);
      const v128_t vprod9x89ABCDEF = wasm_i16x8_mul(vi9x89ABCDEF, vk9x89ABCDEF);
      const v128_t vprod9xGHIJKLMN = wasm_i16x8_mul(vi9xGHIJKLMN, vk9xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod9x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod9x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod9x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod9x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod9xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod9xGHIJKLMN));

      const v128_t vi10x01234567 = wasm_i16x8_load8x8(i10);
      const v128_t vk10x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 240 * sizeof(int8_t)));
      const v128_t vi10x89ABCDEF = wasm_i16x8_load8x8(i10 + 8);
      const v128_t vk10x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 248 * sizeof(int8_t)));
      const v128_t vi10xGHIJKLMN = wasm_i16x8_load8x8(i10 + 16);
      const v128_t vk10xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 256 * sizeof(int8_t)));
      i10 += 24;

      const v128_t vprod10x01234567 = wasm_i16x8_mul(vi10x01234567, vk10x01234567);
      const v128_t vprod10x89ABCDEF = wasm_i16x8_mul(vi10x89ABCDEF, vk10x89ABCDEF);
      const v128_t vprod10xGHIJKLMN = wasm_i16x8_mul(vi10xGHIJKLMN, vk10xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod10x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod10x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod10x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod10x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod10xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod10xGHIJKLMN));

      const v128_t vi11x01234567 = wasm_i16x8_load8x8(i11);
      const v128_t vk11x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 264 * sizeof(int8_t)));
      const v128_t vi11x89ABCDEF = wasm_i16x8_load8x8(i11 + 8);
      const v128_t vk11x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 272 * sizeof(int8_t)));
      const v128_t vi11xGHIJKLMN = wasm_i16x8_load8x8(i11 + 16);
      const v128_t vk11xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 280 * sizeof(int8_t)));
      i11 += 24;

      const v128_t vprod11x01234567 = wasm_i16x8_mul(vi11x01234567, vk11x01234567);
      const v128_t vprod11x89ABCDEF = wasm_i16x8_mul(vi11x89ABCDEF, vk11x89ABCDEF);
      const v128_t vprod11xGHIJKLMN = wasm_i16x8_mul(vi11xGHIJKLMN, vk11xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod11x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod11x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod11x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod11x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod11xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod11xGHIJKLMN));

      const v128_t vi12x01234567 = wasm_i16x8_load8x8(i12);
      const v128_t vk12x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 288 * sizeof(int8_t)));
      const v128_t vi12x89ABCDEF = wasm_i16x8_load8x8(i12 + 8);
      const v128_t vk12x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 296 * sizeof(int8_t)));
      const v128_t vi12xGHIJKLMN = wasm_i16x8_load8x8(i12 + 16);
      const v128_t vk12xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 304 * sizeof(int8_t)));
      i12 += 24;

      const v128_t vprod12x01234567 = wasm_i16x8_mul(vi12x01234567, vk12x01234567);
      const v128_t vprod12x89ABCDEF = wasm_i16x8_mul(vi12x89ABCDEF, vk12x89ABCDEF);
      const v128_t vprod12xGHIJKLMN = wasm_i16x8_mul(vi12xGHIJKLMN, vk12xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod12x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod12x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod12x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod12x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod12xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod12xGHIJKLMN));

      const v128_t vi13x01234567 = wasm_i16x8_load8x8(i13);
      const v128_t vk13x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 312 * sizeof(int8_t)));
      const v128_t vi13x89ABCDEF = wasm_i16x8_load8x8(i13 + 8);
      const v128_t vk13x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 320 * sizeof(int8_t)));
      const v128_t vi13xGHIJKLMN = wasm_i16x8_load8x8(i13 + 16);
      const v128_t vk13xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 328 * sizeof(int8_t)));
      i13 += 24;

      const v128_t vprod13x01234567 = wasm_i16x8_mul(vi13x01234567, vk13x01234567);
      const v128_t vprod13x89ABCDEF = wasm_i16x8_mul(vi13x89ABCDEF, vk13x89ABCDEF);
      const v128_t vprod13xGHIJKLMN = wasm_i16x8_mul(vi13xGHIJKLMN, vk13xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod13x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod13x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod13x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod13x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod13xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod13xGHIJKLMN));

      const v128_t vi14x01234567 = wasm_i16x8_load8x8(i14);
      const v128_t vk14x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 336 * sizeof(int8_t)));
      const v128_t vi14x89ABCDEF = wasm_i16x8_load8x8(i14 + 8);
      const v128_t vk14x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 344 * sizeof(int8_t)));
      const v128_t vi14xGHIJKLMN = wasm_i16x8_load8x8(i14 + 16);
      const v128_t vk14xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 352 * sizeof(int8_t)));
      i14 += 24;

      const v128_t vprod14x01234567 = wasm_i16x8_mul(vi14x01234567, vk14x01234567);
      const v128_t vprod14x89ABCDEF = wasm_i16x8_mul(vi14x89ABCDEF, vk14x89ABCDEF);
      const v128_t vprod14xGHIJKLMN = wasm_i16x8_mul(vi14xGHIJKLMN, vk14xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod14x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod14x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod14x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod14x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod14xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod14xGHIJKLMN));

      const v128_t vi15x01234567 = wasm_i16x8_load8x8(i15);
      const v128_t vk15x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 360 * sizeof(int8_t)));
      const v128_t vi15x89ABCDEF = wasm_i16x8_load8x8(i15 + 8);
      const v128_t vk15x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 368 * sizeof(int8_t)));
      const v128_t vi15xGHIJKLMN = wasm_i16x8_load8x8(i15 + 16);
      const v128_t vk15xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 376 * sizeof(int8_t)));
      i15 += 24;

      const v128_t vprod15x01234567 = wasm_i16x8_mul(vi15x01234567, vk15x01234567);
      const v128_t vprod15x89ABCDEF = wasm_i16x8_mul(vi15x89ABCDEF, vk15x89ABCDEF);
      const v128_t vprod15xGHIJKLMN = wasm_i16x8_mul(vi15xGHIJKLMN, vk15xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod15x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod15x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod15x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod15x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod15xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod15xGHIJKLMN));

      const v128_t vi16x01234567 = wasm_i16x8_load8x8(i16);
      const v128_t vk16x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 384 * sizeof(int8_t)));
      const v128_t vi16x89ABCDEF = wasm_i16x8_load8x8(i16 + 8);
      const v128_t vk16x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 392 * sizeof(int8_t)));
      const v128_t vi16xGHIJKLMN = wasm_i16x8_load8x8(i16 + 16);
      const v128_t vk16xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 400 * sizeof(int8_t)));
      i16 += 24;

      const v128_t vprod16x01234567 = wasm_i16x8_mul(vi16x01234567, vk16x01234567);
      const v128_t vprod16x89ABCDEF = wasm_i16x8_mul(vi16x89ABCDEF, vk16x89ABCDEF);
      const v128_t vprod16xGHIJKLMN = wasm_i16x8_mul(vi16xGHIJKLMN, vk16xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod16x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod16x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod16x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod16x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod16xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod16xGHIJKLMN));

      const v128_t vi17x01234567 = wasm_i16x8_load8x8(i17);
      const v128_t vk17x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 408 * sizeof(int8_t)));
      const v128_t vi17x89ABCDEF = wasm_i16x8_load8x8(i17 + 8);
      const v128_t vk17x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 416 * sizeof(int8_t)));
      const v128_t vi17xGHIJKLMN = wasm_i16x8_load8x8(i17 + 16);
      const v128_t vk17xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 424 * sizeof(int8_t)));
      i17 += 24;

      const v128_t vprod17x01234567 = wasm_i16x8_mul(vi17x01234567, vk17x01234567);
      const v128_t vprod17x89ABCDEF = wasm_i16x8_mul(vi17x89ABCDEF, vk17x89ABCDEF);
      const v128_t vprod17xGHIJKLMN = wasm_i16x8_mul(vi17xGHIJKLMN, vk17xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod17x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod17x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod17x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod17x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod17xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod17xGHIJKLMN));

      const v128_t vi18x01234567 = wasm_i16x8_load8x8(i18);
      const v128_t vk18x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 432 * sizeof(int8_t)));
      const v128_t vi18x89ABCDEF = wasm_i16x8_load8x8(i18 + 8);
      const v128_t vk18x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 440 * sizeof(int8_t)));
      const v128_t vi18xGHIJKLMN = wasm_i16x8_load8x8(i18 + 16);
      const v128_t vk18xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 448 * sizeof(int8_t)));
      i18 += 24;

      const v128_t vprod18x01234567 = wasm_i16x8_mul(vi18x01234567, vk18x01234567);
      const v128_t vprod18x89ABCDEF = wasm_i16x8_mul(vi18x89ABCDEF, vk18x89ABCDEF);
      const v128_t vprod18xGHIJKLMN = wasm_i16x8_mul(vi18xGHIJKLMN, vk18xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod18x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod18x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod18x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod18x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod18xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod18xGHIJKLMN));

      const v128_t vi19x01234567 = wasm_i16x8_load8x8(i19);
      const v128_t vk19x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 456 * sizeof(int8_t)));
      const v128_t vi19x89ABCDEF = wasm_i16x8_load8x8(i19 + 8);
      const v128_t vk19x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 464 * sizeof(int8_t)));
      const v128_t vi19xGHIJKLMN = wasm_i16x8_load8x8(i19 + 16);
      const v128_t vk19xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 472 * sizeof(int8_t)));
      i19 += 24;

      const v128_t vprod19x01234567 = wasm_i16x8_mul(vi19x01234567, vk19x01234567);
      const v128_t vprod19x89ABCDEF = wasm_i16x8_mul(vi19x89ABCDEF, vk19x89ABCDEF);
      const v128_t vprod19xGHIJKLMN = wasm_i16x8_mul(vi19xGHIJKLMN, vk19xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod19x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod19x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod19x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod19x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod19xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod19xGHIJKLMN));

      const v128_t vi20x01234567 = wasm_i16x8_load8x8(i20);
      const v128_t vk20x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 480 * sizeof(int8_t)));
      const v128_t vi20x89ABCDEF = wasm_i16x8_load8x8(i20 + 8);
      const v128_t vk20x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 488 * sizeof(int8_t)));
      const v128_t vi20xGHIJKLMN = wasm_i16x8_load8x8(i20 + 16);
      const v128_t vk20xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 496 * sizeof(int8_t)));
      i20 += 24;

      const v128_t vprod20x01234567 = wasm_i16x8_mul(vi20x01234567, vk20x01234567);
      const v128_t vprod20x89ABCDEF = wasm_i16x8_mul(vi20x89ABCDEF, vk20x89ABCDEF);
      const v128_t vprod20xGHIJKLMN = wasm_i16x8_mul(vi20xGHIJKLMN, vk20xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod20x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod20x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod20x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod20x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod20xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod20xGHIJKLMN));

      const v128_t vi21x01234567 = wasm_i16x8_load8x8(i21);
      const v128_t vk21x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 504 * sizeof(int8_t)));
      const v128_t vi21x89ABCDEF = wasm_i16x8_load8x8(i21 + 8);
      const v128_t vk21x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 512 * sizeof(int8_t)));
      const v128_t vi21xGHIJKLMN = wasm_i16x8_load8x8(i21 + 16);
      const v128_t vk21xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 520 * sizeof(int8_t)));
      i21 += 24;

      const v128_t vprod21x01234567 = wasm_i16x8_mul(vi21x01234567, vk21x01234567);
      const v128_t vprod21x89ABCDEF = wasm_i16x8_mul(vi21x89ABCDEF, vk21x89ABCDEF);
      const v128_t vprod21xGHIJKLMN = wasm_i16x8_mul(vi21xGHIJKLMN, vk21xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod21x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod21x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod21x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod21x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod21xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod21xGHIJKLMN));

      const v128_t vi22x01234567 = wasm_i16x8_load8x8(i22);
      const v128_t vk22x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 528 * sizeof(int8_t)));
      const v128_t vi22x89ABCDEF = wasm_i16x8_load8x8(i22 + 8);
      const v128_t vk22x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 536 * sizeof(int8_t)));
      const v128_t vi22xGHIJKLMN = wasm_i16x8_load8x8(i22 + 16);
      const v128_t vk22xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 544 * sizeof(int8_t)));
      i22 += 24;

      const v128_t vprod22x01234567 = wasm_i16x8_mul(vi22x01234567, vk22x01234567);
      const v128_t vprod22x89ABCDEF = wasm_i16x8_mul(vi22x89ABCDEF, vk22x89ABCDEF);
      const v128_t vprod22xGHIJKLMN = wasm_i16x8_mul(vi22xGHIJKLMN, vk22xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod22x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod22x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod22x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod22x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod22xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod22xGHIJKLMN));

      const v128_t vi23x01234567 = wasm_i16x8_load8x8(i23);
      const v128_t vk23x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 552 * sizeof(int8_t)));
      const v128_t vi23x89ABCDEF = wasm_i16x8_load8x8(i23 + 8);
      const v128_t vk23x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 560 * sizeof(int8_t)));
      const v128_t vi23xGHIJKLMN = wasm_i16x8_load8x8(i23 + 16);
      const v128_t vk23xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 568 * sizeof(int8_t)));
      i23 += 24;

      const v128_t vprod23x01234567 = wasm_i16x8_mul(vi23x01234567, vk23x01234567);
      const v128_t vprod23x89ABCDEF = wasm_i16x8_mul(vi23x89ABCDEF, vk23x89ABCDEF);
      const v128_t vprod23xGHIJKLMN = wasm_i16x8_mul(vi23xGHIJKLMN, vk23xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod23x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod23x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod23x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod23x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod23xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod23xGHIJKLMN));

      const v128_t vi24x01234567 = wasm_i16x8_load8x8(i24);
      const v128_t vk24x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 576 * sizeof(int8_t)));
      const v128_t vi24x89ABCDEF = wasm_i16x8_load8x8(i24 + 8);
      const v128_t vk24x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 584 * sizeof(int8_t)));
      const v128_t vi24xGHIJKLMN = wasm_i16x8_load8x8(i24 + 16);
      const v128_t vk24xGHIJKLMN = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 592 * sizeof(int8_t)));
      i24 += 24;

      const v128_t vprod24x01234567 = wasm_i16x8_mul(vi24x01234567, vk24x01234567);
      const v128_t vprod24x89ABCDEF = wasm_i16x8_mul(vi24x89ABCDEF, vk24x89ABCDEF);
      const v128_t vprod24xGHIJKLMN = wasm_i16x8_mul(vi24xGHIJKLMN, vk24xGHIJKLMN);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod24x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod24x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod24x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod24x89ABCDEF));
      vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_extend_low_i16x8(vprod24xGHIJKLMN));
      vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_extend_high_i16x8(vprod24xGHIJKLMN));


      w = (const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 600 * sizeof(int8_t));

      const v128_t vsign0123 = wasm_i32x4_shr(vacc0123, 31);
      const v128_t vsign4567 = wasm_i32x4_shr(vacc4567, 31);
      const v128_t vsign89AB = wasm_i32x4_shr(vacc89AB, 31);
      const v128_t vsignCDEF = wasm_i32x4_shr(vaccCDEF, 31);
      const v128_t vsignGHIJ = wasm_i32x4_shr(vaccGHIJ, 31);
      const v128_t vsignKLMN = wasm_i32x4_shr(vaccKLMN, 31);

      const v128_t vacc01 = wasm_v32x4_shuffle(vacc0123, vsign0123, 0, 4, 1, 5);
      const v128_t vacc23 = wasm_v32x4_shuffle(vacc0123, vsign0123, 2, 6, 3, 7);
      const v128_t vacc45 = wasm_v32x4_shuffle(vacc4567, vsign4567, 0, 4, 1, 5);
      const v128_t vacc67 = wasm_v32x4_shuffle(vacc4567, vsign4567, 2, 6, 3, 7);
      const v128_t vacc89 = wasm_v32x4_shuffle(vacc89AB, vsign89AB, 0, 4, 1, 5);
      const v128_t vaccAB = wasm_v32x4_shuffle(vacc89AB, vsign89AB, 2, 6, 3, 7);
      const v128_t vaccCD = wasm_v32x4_shuffle(vaccCDEF, vsignCDEF, 0, 4, 1, 5);
      const v128_t vaccEF = wasm_v32x4_shuffle(vaccCDEF, vsignCDEF, 2, 6, 3, 7);
      const v128_t vaccGH = wasm_v32x4_shuffle(vaccGHIJ, vsignGHIJ, 0, 4, 1, 5);
      const v128_t vaccIJ = wasm_v32x4_shuffle(vaccGHIJ, vsignGHIJ, 2, 6, 3, 7);
      const v128_t vaccKL = wasm_v32x4_shuffle(vaccKLMN, vsignKLMN, 0, 4, 1, 5);
      const v128_t vaccMN = wasm_v32x4_shuffle(vaccKLMN, vsignKLMN, 2, 6, 3, 7);

      const v128_t vmultiplier = wasm_v128_load(params->gemmlowp_wasmsimd.multiplier);
      const v128_t vrounding = wasm_v128_load(params->gemmlowp_wasmsimd.rounding);
      const v128_t vprod01 = wasm_i64x2_add(wasm_i64x2_mul(vacc01, vmultiplier), vrounding);
      const v128_t vprod23 = wasm_i64x2_add(wasm_i64x2_mul(vacc23, vmultiplier), vrounding);
      const v128_t vprod45 = wasm_i64x2_add(wasm_i64x2_mul(vacc45, vmultiplier), vrounding);
      const v128_t vprod67 = wasm_i64x2_add(wasm_i64x2_mul(vacc67, vmultiplier), vrounding);
      const v128_t vprod89 = wasm_i64x2_add(wasm_i64x2_mul(vacc89, vmultiplier), vrounding);
      const v128_t vprodAB = wasm_i64x2_add(wasm_i64x2_mul(vaccAB, vmultiplier), vrounding);
      const v128_t vprodCD = wasm_i64x2_add(wasm_i64x2_mul(vaccCD, vmultiplier), vrounding);
      const v128_t vprodEF = wasm_i64x2_add(wasm_i64x2_mul(vaccEF, vmultiplier), vrounding);
      const v128_t vprodGH = wasm_i64x2_add(wasm_i64x2_mul(vaccGH, vmultiplier), vrounding);
      const v128_t vprodIJ = wasm_i64x2_add(wasm_i64x2_mul(vaccIJ, vmultiplier), vrounding);
      const v128_t vprodKL = wasm_i64x2_add(wasm_i64x2_mul(vaccKL, vmultiplier), vrounding);
      const v128_t vprodMN = wasm_i64x2_add(wasm_i64x2_mul(vaccMN, vmultiplier), vrounding);

      const v128_t vq31prod0123 = wasm_v32x4_shuffle(vprod01, vprod23, 1, 3, 5, 7);
      const v128_t vq31prod4567 = wasm_v32x4_shuffle(vprod45, vprod67, 1, 3, 5, 7);
      const v128_t vq31prod89AB = wasm_v32x4_shuffle(vprod89, vprodAB, 1, 3, 5, 7);
      const v128_t vq31prodCDEF = wasm_v32x4_shuffle(vprodCD, vprodEF, 1, 3, 5, 7);
      const v128_t vq31prodGHIJ = wasm_v32x4_shuffle(vprodGH, vprodIJ, 1, 3, 5, 7);
      const v128_t vq31prodKLMN = wasm_v32x4_shuffle(vprodKL, vprodMN, 1, 3, 5, 7);

      const v128_t vremainder_mask = wasm_v128_load(params->gemmlowp_wasmsimd.remainder_mask);
      const v128_t vrem0123 = wasm_i32x4_add(wasm_v128_and(vq31prod0123, vremainder_mask), wasm_i32x4_shr(vq31prod0123, 31));
      const v128_t vrem4567 = wasm_i32x4_add(wasm_v128_and(vq31prod4567, vremainder_mask), wasm_i32x4_shr(vq31prod4567, 31));
      const v128_t vrem89AB = wasm_i32x4_add(wasm_v128_and(vq31prod89AB, vremainder_mask), wasm_i32x4_shr(vq31prod89AB, 31));
      const v128_t vremCDEF = wasm_i32x4_add(wasm_v128_and(vq31prodCDEF, vremainder_mask), wasm_i32x4_shr(vq31prodCDEF, 31));
      const v128_t vremGHIJ = wasm_i32x4_add(wasm_v128_and(vq31prodGHIJ, vremainder_mask), wasm_i32x4_shr(vq31prodGHIJ, 31));
      const v128_t vremKLMN = wasm_i32x4_add(wasm_v128_and(vq31prodKLMN, vremainder_mask), wasm_i32x4_shr(vq31prodKLMN, 31));

      const v128_t vthreshold = wasm_v128_load(params->gemmlowp_wasmsimd.remainder_threshold);
      const int32_t vshift = params->gemmlowp_wasmsimd.shift;
      vacc0123 = wasm_i32x4_sub(wasm_i32x4_shr(vq31prod0123, vshift), wasm_i32x4_gt(vrem0123, vthreshold));
      vacc4567 = wasm_i32x4_sub(wasm_i32x4_shr(vq31prod4567, vshift), wasm_i32x4_gt(vrem4567, vthreshold));
      vacc89AB = wasm_i32x4_sub(wasm_i32x4_shr(vq31prod89AB, vshift), wasm_i32x4_gt(vrem89AB, vthreshold));
      vaccCDEF = wasm_i32x4_sub(wasm_i32x4_shr(vq31prodCDEF, vshift), wasm_i32x4_gt(vremCDEF, vthreshold));
      vaccGHIJ = wasm_i32x4_sub(wasm_i32x4_shr(vq31prodGHIJ, vshift), wasm_i32x4_gt(vremGHIJ, vthreshold));
      vaccKLMN = wasm_i32x4_sub(wasm_i32x4_shr(vq31prodKLMN, vshift), wasm_i32x4_gt(vremKLMN, vthreshold));

      const v128_t voutput_zero_point = wasm_v128_load(params->gemmlowp_wasmsimd.output_zero_point);
      v128_t vout01234567 = wasm_i16x8_add_sat(wasm_i16x8_narrow_i32x4(vacc0123, vacc4567), voutput_zero_point);
      v128_t vout89ABCDEF = wasm_i16x8_add_sat(wasm_i16x8_narrow_i32x4(vacc89AB, vaccCDEF), voutput_zero_point);
      v128_t voutGHIJKLMN = wasm_i16x8_add_sat(wasm_i16x8_narrow_i32x4(vaccGHIJ, vaccKLMN), voutput_zero_point);

      const v128_t voutput_min = wasm_v128_load(params->gemmlowp_wasmsimd.output_min);
      const v128_t voutput_max = wasm_v128_load(params->gemmlowp_wasmsimd.output_max);
      v128_t vout0123456789ABCDEF = wasm_i8x16_min(wasm_i8x16_max(wasm_i8x16_narrow_i16x8(vout01234567, vout89ABCDEF), voutput_min), voutput_max);
      v128_t voutGHIJKLMNGHIJKLMN = wasm_i8x16_min(wasm_i8x16_max(wasm_i8x16_narrow_i16x8(voutGHIJKLMN, voutGHIJKLMN), voutput_min), voutput_max);

      wasm_v128_store(output, vout0123456789ABCDEF);
      *((double*) (output + 16)) = wasm_f64x2_extract_lane(voutGHIJKLMNGHIJKLMN, 0);
      output += 24;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((uintptr_t) w + 24 * sizeof(int32_t));
      do {
        v128_t vacc0123 = wasm_v128_load(w);
        v128_t vacc4567 = wasm_v128_load((const void*) ((uintptr_t) w + 4 * sizeof(int32_t)));


        const v128_t vi0x01234567 = wasm_i16x8_load8x8(i0);
        const v128_t vk0x01234567 = wasm_i16x8_load8x8(k);
        i0 += 8;

        const v128_t vprod0x01234567 = wasm_i16x8_mul(vi0x01234567, vk0x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod0x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod0x01234567));

        const v128_t vi1x01234567 = wasm_i16x8_load8x8(i1);
        const v128_t vk1x01234567 = wasm_i16x8_load8x8((const void*) (k + 24));
        i1 += 8;

        const v128_t vprod1x01234567 = wasm_i16x8_mul(vi1x01234567, vk1x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod1x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod1x01234567));

        const v128_t vi2x01234567 = wasm_i16x8_load8x8(i2);
        const v128_t vk2x01234567 = wasm_i16x8_load8x8((const void*) (k + 48));
        i2 += 8;

        const v128_t vprod2x01234567 = wasm_i16x8_mul(vi2x01234567, vk2x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod2x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod2x01234567));

        const v128_t vi3x01234567 = wasm_i16x8_load8x8(i3);
        const v128_t vk3x01234567 = wasm_i16x8_load8x8((const void*) (k + 72));
        i3 += 8;

        const v128_t vprod3x01234567 = wasm_i16x8_mul(vi3x01234567, vk3x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod3x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod3x01234567));

        const v128_t vi4x01234567 = wasm_i16x8_load8x8(i4);
        const v128_t vk4x01234567 = wasm_i16x8_load8x8((const void*) (k + 96));
        i4 += 8;

        const v128_t vprod4x01234567 = wasm_i16x8_mul(vi4x01234567, vk4x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod4x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod4x01234567));

        const v128_t vi5x01234567 = wasm_i16x8_load8x8(i5);
        const v128_t vk5x01234567 = wasm_i16x8_load8x8((const void*) (k + 120));
        i5 += 8;

        const v128_t vprod5x01234567 = wasm_i16x8_mul(vi5x01234567, vk5x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod5x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod5x01234567));

        const v128_t vi6x01234567 = wasm_i16x8_load8x8(i6);
        const v128_t vk6x01234567 = wasm_i16x8_load8x8((const void*) (k + 144));
        i6 += 8;

        const v128_t vprod6x01234567 = wasm_i16x8_mul(vi6x01234567, vk6x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod6x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod6x01234567));

        const v128_t vi7x01234567 = wasm_i16x8_load8x8(i7);
        const v128_t vk7x01234567 = wasm_i16x8_load8x8((const void*) (k + 168));
        i7 += 8;

        const v128_t vprod7x01234567 = wasm_i16x8_mul(vi7x01234567, vk7x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod7x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod7x01234567));

        const v128_t vi8x01234567 = wasm_i16x8_load8x8(i8);
        const v128_t vk8x01234567 = wasm_i16x8_load8x8((const void*) (k + 192));
        i8 += 8;

        const v128_t vprod8x01234567 = wasm_i16x8_mul(vi8x01234567, vk8x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod8x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod8x01234567));

        const v128_t vi9x01234567 = wasm_i16x8_load8x8(i9);
        const v128_t vk9x01234567 = wasm_i16x8_load8x8((const void*) (k + 216));
        i9 += 8;

        const v128_t vprod9x01234567 = wasm_i16x8_mul(vi9x01234567, vk9x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod9x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod9x01234567));

        const v128_t vi10x01234567 = wasm_i16x8_load8x8(i10);
        const v128_t vk10x01234567 = wasm_i16x8_load8x8((const void*) (k + 240));
        i10 += 8;

        const v128_t vprod10x01234567 = wasm_i16x8_mul(vi10x01234567, vk10x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod10x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod10x01234567));

        const v128_t vi11x01234567 = wasm_i16x8_load8x8(i11);
        const v128_t vk11x01234567 = wasm_i16x8_load8x8((const void*) (k + 264));
        i11 += 8;

        const v128_t vprod11x01234567 = wasm_i16x8_mul(vi11x01234567, vk11x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod11x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod11x01234567));

        const v128_t vi12x01234567 = wasm_i16x8_load8x8(i12);
        const v128_t vk12x01234567 = wasm_i16x8_load8x8((const void*) (k + 288));
        i12 += 8;

        const v128_t vprod12x01234567 = wasm_i16x8_mul(vi12x01234567, vk12x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod12x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod12x01234567));

        const v128_t vi13x01234567 = wasm_i16x8_load8x8(i13);
        const v128_t vk13x01234567 = wasm_i16x8_load8x8((const void*) (k + 312));
        i13 += 8;

        const v128_t vprod13x01234567 = wasm_i16x8_mul(vi13x01234567, vk13x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod13x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod13x01234567));

        const v128_t vi14x01234567 = wasm_i16x8_load8x8(i14);
        const v128_t vk14x01234567 = wasm_i16x8_load8x8((const void*) (k + 336));
        i14 += 8;

        const v128_t vprod14x01234567 = wasm_i16x8_mul(vi14x01234567, vk14x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod14x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod14x01234567));

        const v128_t vi15x01234567 = wasm_i16x8_load8x8(i15);
        const v128_t vk15x01234567 = wasm_i16x8_load8x8((const void*) (k + 360));
        i15 += 8;

        const v128_t vprod15x01234567 = wasm_i16x8_mul(vi15x01234567, vk15x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod15x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod15x01234567));

        const v128_t vi16x01234567 = wasm_i16x8_load8x8(i16);
        const v128_t vk16x01234567 = wasm_i16x8_load8x8((const void*) (k + 384));
        i16 += 8;

        const v128_t vprod16x01234567 = wasm_i16x8_mul(vi16x01234567, vk16x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod16x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod16x01234567));

        const v128_t vi17x01234567 = wasm_i16x8_load8x8(i17);
        const v128_t vk17x01234567 = wasm_i16x8_load8x8((const void*) (k + 408));
        i17 += 8;

        const v128_t vprod17x01234567 = wasm_i16x8_mul(vi17x01234567, vk17x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod17x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod17x01234567));

        const v128_t vi18x01234567 = wasm_i16x8_load8x8(i18);
        const v128_t vk18x01234567 = wasm_i16x8_load8x8((const void*) (k + 432));
        i18 += 8;

        const v128_t vprod18x01234567 = wasm_i16x8_mul(vi18x01234567, vk18x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod18x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod18x01234567));

        const v128_t vi19x01234567 = wasm_i16x8_load8x8(i19);
        const v128_t vk19x01234567 = wasm_i16x8_load8x8((const void*) (k + 456));
        i19 += 8;

        const v128_t vprod19x01234567 = wasm_i16x8_mul(vi19x01234567, vk19x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod19x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod19x01234567));

        const v128_t vi20x01234567 = wasm_i16x8_load8x8(i20);
        const v128_t vk20x01234567 = wasm_i16x8_load8x8((const void*) (k + 480));
        i20 += 8;

        const v128_t vprod20x01234567 = wasm_i16x8_mul(vi20x01234567, vk20x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod20x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod20x01234567));

        const v128_t vi21x01234567 = wasm_i16x8_load8x8(i21);
        const v128_t vk21x01234567 = wasm_i16x8_load8x8((const void*) (k + 504));
        i21 += 8;

        const v128_t vprod21x01234567 = wasm_i16x8_mul(vi21x01234567, vk21x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod21x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod21x01234567));

        const v128_t vi22x01234567 = wasm_i16x8_load8x8(i22);
        const v128_t vk22x01234567 = wasm_i16x8_load8x8((const void*) (k + 528));
        i22 += 8;

        const v128_t vprod22x01234567 = wasm_i16x8_mul(vi22x01234567, vk22x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod22x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod22x01234567));

        const v128_t vi23x01234567 = wasm_i16x8_load8x8(i23);
        const v128_t vk23x01234567 = wasm_i16x8_load8x8((const void*) (k + 552));
        i23 += 8;

        const v128_t vprod23x01234567 = wasm_i16x8_mul(vi23x01234567, vk23x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod23x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod23x01234567));

        const v128_t vi24x01234567 = wasm_i16x8_load8x8(i24);
        const v128_t vk24x01234567 = wasm_i16x8_load8x8((const void*) (k + 576));
        i24 += 8;

        const v128_t vprod24x01234567 = wasm_i16x8_mul(vi24x01234567, vk24x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod24x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod24x01234567));

        k += 8;


      const v128_t vsign0123 = wasm_i32x4_shr(vacc0123, 31);
      const v128_t vsign4567 = wasm_i32x4_shr(vacc4567, 31);

      const v128_t vacc01 = wasm_v32x4_shuffle(vacc0123, vsign0123, 0, 4, 1, 5);
      const v128_t vacc23 = wasm_v32x4_shuffle(vacc0123, vsign0123, 2, 6, 3, 7);
      const v128_t vacc45 = wasm_v32x4_shuffle(vacc4567, vsign4567, 0, 4, 1, 5);
      const v128_t vacc67 = wasm_v32x4_shuffle(vacc4567, vsign4567, 2, 6, 3, 7);

      const v128_t vmultiplier = wasm_v128_load(params->gemmlowp_wasmsimd.multiplier);
      const v128_t vrounding = wasm_v128_load(params->gemmlowp_wasmsimd.rounding);
      const v128_t vprod01 = wasm_i64x2_add(wasm_i64x2_mul(vacc01, vmultiplier), vrounding);
      const v128_t vprod23 = wasm_i64x2_add(wasm_i64x2_mul(vacc23, vmultiplier), vrounding);
      const v128_t vprod45 = wasm_i64x2_add(wasm_i64x2_mul(vacc45, vmultiplier), vrounding);
      const v128_t vprod67 = wasm_i64x2_add(wasm_i64x2_mul(vacc67, vmultiplier), vrounding);

      const v128_t vq31prod0123 = wasm_v32x4_shuffle(vprod01, vprod23, 1, 3, 5, 7);
      const v128_t vq31prod4567 = wasm_v32x4_shuffle(vprod45, vprod67, 1, 3, 5, 7);

      const v128_t vremainder_mask = wasm_v128_load(params->gemmlowp_wasmsimd.remainder_mask);
      const v128_t vrem0123 = wasm_i32x4_add(wasm_v128_and(vq31prod0123, vremainder_mask), wasm_i32x4_shr(vq31prod0123, 31));
      const v128_t vrem4567 = wasm_i32x4_add(wasm_v128_and(vq31prod4567, vremainder_mask), wasm_i32x4_shr(vq31prod4567, 31));

      const v128_t vthreshold = wasm_v128_load(params->gemmlowp_wasmsimd.remainder_threshold);
      const int32_t vshift = params->gemmlowp_wasmsimd.shift;
      vacc0123 = wasm_i32x4_sub(wasm_i32x4_shr(vq31prod0123, vshift), wasm_i32x4_gt(vrem0123, vthreshold));
      vacc4567 = wasm_i32x4_sub(wasm_i32x4_shr(vq31prod4567, vshift), wasm_i32x4_gt(vrem4567, vthreshold));

      const v128_t voutput_zero_point = wasm_v128_load(params->gemmlowp_wasmsimd.output_zero_point);
      v128_t vout01234567 = wasm_i16x8_add_sat(wasm_i16x8_narrow_i32x4(vacc0123, vacc4567), voutput_zero_point);

      const v128_t voutput_min = wasm_v128_load(params->gemmlowp_wasmsimd.output_min);
      const v128_t voutput_max = wasm_v128_load(params->gemmlowp_wasmsimd.output_max);
      v128_t vout0123456701234567 = wasm_i8x16_min(wasm_i8x16_max(wasm_i8x16_narrow_i16x8(vout01234567, vout01234567), voutput_min), voutput_max);

      w = (const void*) ((uintptr_t) w + 8 * sizeof(int32_t));

      if XNN_LIKELY(c >= 8) {
        *((double*) output) = wasm_f64x2_extract_lane(vout0123456701234567, 0);
        output += 8;
        c -= 8;
      } else {
        if (c & 4) {
          *((float*) output) = wasm_f32x4_extract_lane(vout0123456701234567, 0);
          vout0123456701234567 = wasm_u64x2_shr(vout0123456701234567, 32);
          output += 4;
        }
        if (c & 2) {
          *((uint16_t*) output) = (uint16_t) wasm_i16x8_extract_lane(vout0123456701234567, 0);
          vout0123456701234567 = wasm_u32x4_shr(vout0123456701234567, 16);
          output += 2;
        }
        if (c & 1) {
          *output = (int8_t) wasm_i8x16_extract_lane(vout0123456701234567, 0);
          output += 1;
        }
        c = 0;
      }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
