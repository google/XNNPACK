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


void xnn_qs8_dwconv_minmax_gemmlowp_ukernel_up24x9__wasmsimd_mul16(
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


      w = (const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 216 * sizeof(int8_t));

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
