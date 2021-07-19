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


void xnn_qs8_dwconv_minmax_fp32_ukernel_up24x9__wasmsimd_mul16(
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

      vacc0123 = wasm_f32x4_convert_i32x4(vacc0123);
      vacc4567 = wasm_f32x4_convert_i32x4(vacc4567);
      vacc89AB = wasm_f32x4_convert_i32x4(vacc89AB);
      vaccCDEF = wasm_f32x4_convert_i32x4(vaccCDEF);
      vaccGHIJ = wasm_f32x4_convert_i32x4(vaccGHIJ);
      vaccKLMN = wasm_f32x4_convert_i32x4(vaccKLMN);

      const v128_t vscale = wasm_v128_load(params->fp32_wasmsimd.scale);
      vacc0123 = wasm_f32x4_mul(vacc0123, vscale);
      vacc4567 = wasm_f32x4_mul(vacc4567, vscale);
      vacc89AB = wasm_f32x4_mul(vacc89AB, vscale);
      vaccCDEF = wasm_f32x4_mul(vaccCDEF, vscale);
      vaccGHIJ = wasm_f32x4_mul(vaccGHIJ, vscale);
      vaccKLMN = wasm_f32x4_mul(vaccKLMN, vscale);

      const v128_t voutput_min_less_zero_point = wasm_v128_load(params->fp32_wasmsimd.output_min_less_zero_point);
      vacc0123 = wasm_f32x4_max(vacc0123, voutput_min_less_zero_point);
      vacc4567 = wasm_f32x4_max(vacc4567, voutput_min_less_zero_point);
      vacc89AB = wasm_f32x4_max(vacc89AB, voutput_min_less_zero_point);
      vaccCDEF = wasm_f32x4_max(vaccCDEF, voutput_min_less_zero_point);
      vaccGHIJ = wasm_f32x4_max(vaccGHIJ, voutput_min_less_zero_point);
      vaccKLMN = wasm_f32x4_max(vaccKLMN, voutput_min_less_zero_point);

      const v128_t voutput_max_less_zero_point = wasm_v128_load(params->fp32_wasmsimd.output_max_less_zero_point);
      vacc0123 = wasm_f32x4_min(vacc0123, voutput_max_less_zero_point);
      vacc4567 = wasm_f32x4_min(vacc4567, voutput_max_less_zero_point);
      vacc89AB = wasm_f32x4_min(vacc89AB, voutput_max_less_zero_point);
      vaccCDEF = wasm_f32x4_min(vaccCDEF, voutput_max_less_zero_point);
      vaccGHIJ = wasm_f32x4_min(vaccGHIJ, voutput_max_less_zero_point);
      vaccKLMN = wasm_f32x4_min(vaccKLMN, voutput_max_less_zero_point);

      const v128_t vmagic_bias = wasm_v128_load(params->fp32_wasmsimd.magic_bias);
      vacc0123 = wasm_f32x4_add(vacc0123, vmagic_bias);
      vacc4567 = wasm_f32x4_add(vacc4567, vmagic_bias);
      vacc89AB = wasm_f32x4_add(vacc89AB, vmagic_bias);
      vaccCDEF = wasm_f32x4_add(vaccCDEF, vmagic_bias);
      vaccGHIJ = wasm_f32x4_add(vaccGHIJ, vmagic_bias);
      vaccKLMN = wasm_f32x4_add(vaccKLMN, vmagic_bias);

      const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load(params->fp32_wasmsimd.magic_bias_less_output_zero_point);
      vacc0123 = wasm_i32x4_sub(vacc0123, vmagic_bias_less_output_zero_point);
      vacc4567 = wasm_i32x4_sub(vacc4567, vmagic_bias_less_output_zero_point);
      vacc89AB = wasm_i32x4_sub(vacc89AB, vmagic_bias_less_output_zero_point);
      vaccCDEF = wasm_i32x4_sub(vaccCDEF, vmagic_bias_less_output_zero_point);
      vaccGHIJ = wasm_i32x4_sub(vaccGHIJ, vmagic_bias_less_output_zero_point);
      vaccKLMN = wasm_i32x4_sub(vaccKLMN, vmagic_bias_less_output_zero_point);

      v128_t vout01234567 = wasm_v16x8_shuffle(vacc0123, vacc4567, 0, 2, 4, 6, 8, 10, 12, 14);
      v128_t vout89ABCDEF = wasm_v16x8_shuffle(vacc89AB, vaccCDEF, 0, 2, 4, 6, 8, 10, 12, 14);
      v128_t voutGHIJKLMN = wasm_v16x8_shuffle(vaccGHIJ, vaccKLMN, 0, 2, 4, 6, 8, 10, 12, 14);

      v128_t vout0123456789ABCDEF = wasm_v8x16_shuffle(vout01234567, vout89ABCDEF, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
      v128_t voutGHIJKLMNGHIJKLMN = wasm_v8x16_shuffle(voutGHIJKLMN, voutGHIJKLMN, 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14);

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


      vacc0123 = wasm_f32x4_convert_i32x4(vacc0123);
      vacc4567 = wasm_f32x4_convert_i32x4(vacc4567);

      const v128_t vscale = wasm_v128_load(params->fp32_wasmsimd.scale);
      vacc0123 = wasm_f32x4_mul(vacc0123, vscale);
      vacc4567 = wasm_f32x4_mul(vacc4567, vscale);

      const v128_t voutput_min_less_zero_point = wasm_v128_load(params->fp32_wasmsimd.output_min_less_zero_point);
      vacc0123 = wasm_f32x4_max(vacc0123, voutput_min_less_zero_point);
      vacc4567 = wasm_f32x4_max(vacc4567, voutput_min_less_zero_point);

      const v128_t voutput_max_less_zero_point = wasm_v128_load(params->fp32_wasmsimd.output_max_less_zero_point);
      vacc0123 = wasm_f32x4_min(vacc0123, voutput_max_less_zero_point);
      vacc4567 = wasm_f32x4_min(vacc4567, voutput_max_less_zero_point);

      const v128_t vmagic_bias = wasm_v128_load(params->fp32_wasmsimd.magic_bias);
      vacc0123 = wasm_f32x4_add(vacc0123, vmagic_bias);
      vacc4567 = wasm_f32x4_add(vacc4567, vmagic_bias);

      const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load(params->fp32_wasmsimd.magic_bias_less_output_zero_point);
      vacc0123 = wasm_i32x4_sub(vacc0123, vmagic_bias_less_output_zero_point);
      vacc4567 = wasm_i32x4_sub(vacc4567, vmagic_bias_less_output_zero_point);

      v128_t vout01234567 = wasm_v16x8_shuffle(vacc0123, vacc4567, 0, 2, 4, 6, 8, 10, 12, 14);
      v128_t vout0123456701234567 = wasm_v8x16_shuffle(vout01234567, vout01234567, 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14);

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
