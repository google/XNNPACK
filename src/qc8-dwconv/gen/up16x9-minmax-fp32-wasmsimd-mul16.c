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


void xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__wasmsimd_mul16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN XNN_DISABLE_MSAN
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
    for (; c >= 16; c -= 16) {
      v128_t vacc0123 = wasm_v128_load(w);
      v128_t vacc4567 = wasm_v128_load((const void*) ((uintptr_t) w + 4 * sizeof(int32_t)));
      v128_t vacc89AB = wasm_v128_load((const void*) ((uintptr_t) w + 8 * sizeof(int32_t)));
      v128_t vaccCDEF = wasm_v128_load((const void*) ((uintptr_t) w + 12 * sizeof(int32_t)));


      const v128_t vi0x01234567 = wasm_i16x8_load8x8(i0);
      const v128_t vk0x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      const v128_t vi0x89ABCDEF = wasm_i16x8_load8x8(i0 + 8);
      const v128_t vk0x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      i0 += 16;

      const v128_t vprod0x01234567 = wasm_i16x8_mul(vi0x01234567, vk0x01234567);
      const v128_t vprod0x89ABCDEF = wasm_i16x8_mul(vi0x89ABCDEF, vk0x89ABCDEF);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod0x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod0x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod0x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod0x89ABCDEF));

      const v128_t vi1x01234567 = wasm_i16x8_load8x8(i1);
      const v128_t vk1x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      const v128_t vi1x89ABCDEF = wasm_i16x8_load8x8(i1 + 8);
      const v128_t vk1x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      i1 += 16;

      const v128_t vprod1x01234567 = wasm_i16x8_mul(vi1x01234567, vk1x01234567);
      const v128_t vprod1x89ABCDEF = wasm_i16x8_mul(vi1x89ABCDEF, vk1x89ABCDEF);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod1x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod1x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod1x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod1x89ABCDEF));

      const v128_t vi2x01234567 = wasm_i16x8_load8x8(i2);
      const v128_t vk2x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      const v128_t vi2x89ABCDEF = wasm_i16x8_load8x8(i2 + 8);
      const v128_t vk2x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      i2 += 16;

      const v128_t vprod2x01234567 = wasm_i16x8_mul(vi2x01234567, vk2x01234567);
      const v128_t vprod2x89ABCDEF = wasm_i16x8_mul(vi2x89ABCDEF, vk2x89ABCDEF);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod2x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod2x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod2x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod2x89ABCDEF));

      const v128_t vi3x01234567 = wasm_i16x8_load8x8(i3);
      const v128_t vk3x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      const v128_t vi3x89ABCDEF = wasm_i16x8_load8x8(i3 + 8);
      const v128_t vk3x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      i3 += 16;

      const v128_t vprod3x01234567 = wasm_i16x8_mul(vi3x01234567, vk3x01234567);
      const v128_t vprod3x89ABCDEF = wasm_i16x8_mul(vi3x89ABCDEF, vk3x89ABCDEF);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod3x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod3x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod3x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod3x89ABCDEF));

      const v128_t vi4x01234567 = wasm_i16x8_load8x8(i4);
      const v128_t vk4x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      const v128_t vi4x89ABCDEF = wasm_i16x8_load8x8(i4 + 8);
      const v128_t vk4x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(int8_t)));
      i4 += 16;

      const v128_t vprod4x01234567 = wasm_i16x8_mul(vi4x01234567, vk4x01234567);
      const v128_t vprod4x89ABCDEF = wasm_i16x8_mul(vi4x89ABCDEF, vk4x89ABCDEF);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod4x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod4x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod4x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod4x89ABCDEF));

      const v128_t vi5x01234567 = wasm_i16x8_load8x8(i5);
      const v128_t vk5x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(int8_t)));
      const v128_t vi5x89ABCDEF = wasm_i16x8_load8x8(i5 + 8);
      const v128_t vk5x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(int8_t)));
      i5 += 16;

      const v128_t vprod5x01234567 = wasm_i16x8_mul(vi5x01234567, vk5x01234567);
      const v128_t vprod5x89ABCDEF = wasm_i16x8_mul(vi5x89ABCDEF, vk5x89ABCDEF);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod5x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod5x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod5x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod5x89ABCDEF));

      const v128_t vi6x01234567 = wasm_i16x8_load8x8(i6);
      const v128_t vk6x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(int8_t)));
      const v128_t vi6x89ABCDEF = wasm_i16x8_load8x8(i6 + 8);
      const v128_t vk6x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 104 * sizeof(int8_t)));
      i6 += 16;

      const v128_t vprod6x01234567 = wasm_i16x8_mul(vi6x01234567, vk6x01234567);
      const v128_t vprod6x89ABCDEF = wasm_i16x8_mul(vi6x89ABCDEF, vk6x89ABCDEF);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod6x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod6x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod6x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod6x89ABCDEF));

      const v128_t vi7x01234567 = wasm_i16x8_load8x8(i7);
      const v128_t vk7x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(int8_t)));
      const v128_t vi7x89ABCDEF = wasm_i16x8_load8x8(i7 + 8);
      const v128_t vk7x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 120 * sizeof(int8_t)));
      i7 += 16;

      const v128_t vprod7x01234567 = wasm_i16x8_mul(vi7x01234567, vk7x01234567);
      const v128_t vprod7x89ABCDEF = wasm_i16x8_mul(vi7x89ABCDEF, vk7x89ABCDEF);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod7x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod7x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod7x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod7x89ABCDEF));

      const v128_t vi8x01234567 = wasm_i16x8_load8x8(i8);
      const v128_t vk8x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(int8_t)));
      const v128_t vi8x89ABCDEF = wasm_i16x8_load8x8(i8 + 8);
      const v128_t vk8x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 136 * sizeof(int8_t)));
      i8 += 16;

      const v128_t vprod8x01234567 = wasm_i16x8_mul(vi8x01234567, vk8x01234567);
      const v128_t vprod8x89ABCDEF = wasm_i16x8_mul(vi8x89ABCDEF, vk8x89ABCDEF);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod8x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod8x01234567));
      vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod8x89ABCDEF));
      vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod8x89ABCDEF));


      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t));

      vacc0123 = wasm_f32x4_convert_i32x4(vacc0123);
      vacc4567 = wasm_f32x4_convert_i32x4(vacc4567);
      vacc89AB = wasm_f32x4_convert_i32x4(vacc89AB);
      vaccCDEF = wasm_f32x4_convert_i32x4(vaccCDEF);

      const v128_t vscale0123 = wasm_v128_load(w);
      const v128_t vscale4567 = wasm_v128_load((const float*) w + 4);
      const v128_t vscale89AB = wasm_v128_load((const float*) w + 8);
      const v128_t vscaleCDEF = wasm_v128_load((const float*) w + 12);
      w = (const void*) ((const float*) w + 16);

      vacc0123 = wasm_f32x4_mul(vacc0123, vscale0123);
      vacc4567 = wasm_f32x4_mul(vacc4567, vscale4567);
      vacc89AB = wasm_f32x4_mul(vacc89AB, vscale89AB);
      vaccCDEF = wasm_f32x4_mul(vaccCDEF, vscaleCDEF);

      const v128_t voutput_min_less_zero_point = wasm_v128_load(params->wasmsimd.output_min_less_zero_point);
      vacc0123 = wasm_f32x4_max(vacc0123, voutput_min_less_zero_point);
      vacc4567 = wasm_f32x4_max(vacc4567, voutput_min_less_zero_point);
      vacc89AB = wasm_f32x4_max(vacc89AB, voutput_min_less_zero_point);
      vaccCDEF = wasm_f32x4_max(vaccCDEF, voutput_min_less_zero_point);

      const v128_t voutput_max_less_zero_point = wasm_v128_load(params->wasmsimd.output_max_less_zero_point);
      vacc0123 = wasm_f32x4_min(vacc0123, voutput_max_less_zero_point);
      vacc4567 = wasm_f32x4_min(vacc4567, voutput_max_less_zero_point);
      vacc89AB = wasm_f32x4_min(vacc89AB, voutput_max_less_zero_point);
      vaccCDEF = wasm_f32x4_min(vaccCDEF, voutput_max_less_zero_point);

      const v128_t vmagic_bias = wasm_v128_load(params->wasmsimd.magic_bias);
      vacc0123 = wasm_f32x4_add(vacc0123, vmagic_bias);
      vacc4567 = wasm_f32x4_add(vacc4567, vmagic_bias);
      vacc89AB = wasm_f32x4_add(vacc89AB, vmagic_bias);
      vaccCDEF = wasm_f32x4_add(vaccCDEF, vmagic_bias);

      const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load(params->wasmsimd.magic_bias_less_output_zero_point);
      vacc0123 = wasm_i32x4_sub(vacc0123, vmagic_bias_less_output_zero_point);
      vacc4567 = wasm_i32x4_sub(vacc4567, vmagic_bias_less_output_zero_point);
      vacc89AB = wasm_i32x4_sub(vacc89AB, vmagic_bias_less_output_zero_point);
      vaccCDEF = wasm_i32x4_sub(vaccCDEF, vmagic_bias_less_output_zero_point);

      v128_t vout01234567 = wasm_v16x8_shuffle(vacc0123, vacc4567, 0, 2, 4, 6, 8, 10, 12, 14);
      v128_t vout89ABCDEF = wasm_v16x8_shuffle(vacc89AB, vaccCDEF, 0, 2, 4, 6, 8, 10, 12, 14);

      v128_t vout0123456789ABCDEF = wasm_v8x16_shuffle(vout01234567, vout89ABCDEF, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);

      wasm_v128_store(output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((uintptr_t) w + 16 * sizeof(int32_t));
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
        const v128_t vk1x01234567 = wasm_i16x8_load8x8((const void*) (k + 16));
        i1 += 8;

        const v128_t vprod1x01234567 = wasm_i16x8_mul(vi1x01234567, vk1x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod1x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod1x01234567));

        const v128_t vi2x01234567 = wasm_i16x8_load8x8(i2);
        const v128_t vk2x01234567 = wasm_i16x8_load8x8((const void*) (k + 32));
        i2 += 8;

        const v128_t vprod2x01234567 = wasm_i16x8_mul(vi2x01234567, vk2x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod2x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod2x01234567));

        const v128_t vi3x01234567 = wasm_i16x8_load8x8(i3);
        const v128_t vk3x01234567 = wasm_i16x8_load8x8((const void*) (k + 48));
        i3 += 8;

        const v128_t vprod3x01234567 = wasm_i16x8_mul(vi3x01234567, vk3x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod3x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod3x01234567));

        const v128_t vi4x01234567 = wasm_i16x8_load8x8(i4);
        const v128_t vk4x01234567 = wasm_i16x8_load8x8((const void*) (k + 64));
        i4 += 8;

        const v128_t vprod4x01234567 = wasm_i16x8_mul(vi4x01234567, vk4x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod4x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod4x01234567));

        const v128_t vi5x01234567 = wasm_i16x8_load8x8(i5);
        const v128_t vk5x01234567 = wasm_i16x8_load8x8((const void*) (k + 80));
        i5 += 8;

        const v128_t vprod5x01234567 = wasm_i16x8_mul(vi5x01234567, vk5x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod5x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod5x01234567));

        const v128_t vi6x01234567 = wasm_i16x8_load8x8(i6);
        const v128_t vk6x01234567 = wasm_i16x8_load8x8((const void*) (k + 96));
        i6 += 8;

        const v128_t vprod6x01234567 = wasm_i16x8_mul(vi6x01234567, vk6x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod6x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod6x01234567));

        const v128_t vi7x01234567 = wasm_i16x8_load8x8(i7);
        const v128_t vk7x01234567 = wasm_i16x8_load8x8((const void*) (k + 112));
        i7 += 8;

        const v128_t vprod7x01234567 = wasm_i16x8_mul(vi7x01234567, vk7x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod7x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod7x01234567));

        const v128_t vi8x01234567 = wasm_i16x8_load8x8(i8);
        const v128_t vk8x01234567 = wasm_i16x8_load8x8((const void*) (k + 128));
        i8 += 8;

        const v128_t vprod8x01234567 = wasm_i16x8_mul(vi8x01234567, vk8x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod8x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod8x01234567));

        k += 8;


      vacc0123 = wasm_f32x4_convert_i32x4(vacc0123);
      vacc4567 = wasm_f32x4_convert_i32x4(vacc4567);

      const v128_t vscale0123 = wasm_v128_load((const float*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t)));
      const v128_t vscale4567 = wasm_v128_load((const float*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t) + 4 * sizeof(float)));

      vacc0123 = wasm_f32x4_mul(vacc0123, vscale0123);
      vacc4567 = wasm_f32x4_mul(vacc4567, vscale4567);

      const v128_t voutput_min_less_zero_point = wasm_v128_load(params->wasmsimd.output_min_less_zero_point);
      vacc0123 = wasm_f32x4_max(vacc0123, voutput_min_less_zero_point);
      vacc4567 = wasm_f32x4_max(vacc4567, voutput_min_less_zero_point);

      const v128_t voutput_max_less_zero_point = wasm_v128_load(params->wasmsimd.output_max_less_zero_point);
      vacc0123 = wasm_f32x4_min(vacc0123, voutput_max_less_zero_point);
      vacc4567 = wasm_f32x4_min(vacc4567, voutput_max_less_zero_point);

      const v128_t vmagic_bias = wasm_v128_load(params->wasmsimd.magic_bias);
      vacc0123 = wasm_f32x4_add(vacc0123, vmagic_bias);
      vacc4567 = wasm_f32x4_add(vacc4567, vmagic_bias);

      const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load(params->wasmsimd.magic_bias_less_output_zero_point);
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
