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


void xnn_qs8_dwconv_minmax_gemmlowp_ukernel_up8x9__wasmsimd_mul16(
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
    for (; c >= 8; c -= 8) {
      v128_t vacc0123 = wasm_v128_load(w);
      v128_t vacc4567 = wasm_v128_load((const void*) ((uintptr_t) w + 4 * sizeof(int32_t)));


      const v128_t vi0x01234567 = wasm_i16x8_load8x8(i0);
      const v128_t vk0x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      i0 += 8;

      const v128_t vprod0x01234567 = wasm_i16x8_mul(vi0x01234567, vk0x01234567);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod0x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod0x01234567));

      const v128_t vi1x01234567 = wasm_i16x8_load8x8(i1);
      const v128_t vk1x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      i1 += 8;

      const v128_t vprod1x01234567 = wasm_i16x8_mul(vi1x01234567, vk1x01234567);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod1x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod1x01234567));

      const v128_t vi2x01234567 = wasm_i16x8_load8x8(i2);
      const v128_t vk2x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      i2 += 8;

      const v128_t vprod2x01234567 = wasm_i16x8_mul(vi2x01234567, vk2x01234567);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod2x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod2x01234567));

      const v128_t vi3x01234567 = wasm_i16x8_load8x8(i3);
      const v128_t vk3x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      i3 += 8;

      const v128_t vprod3x01234567 = wasm_i16x8_mul(vi3x01234567, vk3x01234567);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod3x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod3x01234567));

      const v128_t vi4x01234567 = wasm_i16x8_load8x8(i4);
      const v128_t vk4x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      i4 += 8;

      const v128_t vprod4x01234567 = wasm_i16x8_mul(vi4x01234567, vk4x01234567);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod4x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod4x01234567));

      const v128_t vi5x01234567 = wasm_i16x8_load8x8(i5);
      const v128_t vk5x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      i5 += 8;

      const v128_t vprod5x01234567 = wasm_i16x8_mul(vi5x01234567, vk5x01234567);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod5x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod5x01234567));

      const v128_t vi6x01234567 = wasm_i16x8_load8x8(i6);
      const v128_t vk6x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      i6 += 8;

      const v128_t vprod6x01234567 = wasm_i16x8_mul(vi6x01234567, vk6x01234567);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod6x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod6x01234567));

      const v128_t vi7x01234567 = wasm_i16x8_load8x8(i7);
      const v128_t vk7x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      i7 += 8;

      const v128_t vprod7x01234567 = wasm_i16x8_mul(vi7x01234567, vk7x01234567);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod7x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod7x01234567));

      const v128_t vi8x01234567 = wasm_i16x8_load8x8(i8);
      const v128_t vk8x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      i8 += 8;

      const v128_t vprod8x01234567 = wasm_i16x8_mul(vi8x01234567, vk8x01234567);

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod8x01234567));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod8x01234567));


      w = (const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(int8_t));

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

      *((double*) output) = wasm_f64x2_extract_lane(vout0123456701234567, 0);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      {
        v128_t vacc0123 = wasm_v128_load(w);
        v128_t vacc4567 = wasm_v128_load((const void*) ((uintptr_t) w + 4 * sizeof(int32_t)));


        const v128_t vi0x01234567 = wasm_i16x8_load8x8(i0);
        const v128_t vk0x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t)));

        const v128_t vprod0x01234567 = wasm_i16x8_mul(vi0x01234567, vk0x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod0x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod0x01234567));

        const v128_t vi1x01234567 = wasm_i16x8_load8x8(i1);
        const v128_t vk1x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));

        const v128_t vprod1x01234567 = wasm_i16x8_mul(vi1x01234567, vk1x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod1x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod1x01234567));

        const v128_t vi2x01234567 = wasm_i16x8_load8x8(i2);
        const v128_t vk2x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));

        const v128_t vprod2x01234567 = wasm_i16x8_mul(vi2x01234567, vk2x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod2x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod2x01234567));

        const v128_t vi3x01234567 = wasm_i16x8_load8x8(i3);
        const v128_t vk3x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t)));

        const v128_t vprod3x01234567 = wasm_i16x8_mul(vi3x01234567, vk3x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod3x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod3x01234567));

        const v128_t vi4x01234567 = wasm_i16x8_load8x8(i4);
        const v128_t vk4x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(int8_t)));

        const v128_t vprod4x01234567 = wasm_i16x8_mul(vi4x01234567, vk4x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod4x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod4x01234567));

        const v128_t vi5x01234567 = wasm_i16x8_load8x8(i5);
        const v128_t vk5x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(int8_t)));

        const v128_t vprod5x01234567 = wasm_i16x8_mul(vi5x01234567, vk5x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod5x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod5x01234567));

        const v128_t vi6x01234567 = wasm_i16x8_load8x8(i6);
        const v128_t vk6x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t)));

        const v128_t vprod6x01234567 = wasm_i16x8_mul(vi6x01234567, vk6x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod6x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod6x01234567));

        const v128_t vi7x01234567 = wasm_i16x8_load8x8(i7);
        const v128_t vk7x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(int8_t)));

        const v128_t vprod7x01234567 = wasm_i16x8_mul(vi7x01234567, vk7x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod7x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod7x01234567));

        const v128_t vi8x01234567 = wasm_i16x8_load8x8(i8);
        const v128_t vk8x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(int8_t)));

        const v128_t vprod8x01234567 = wasm_i16x8_mul(vi8x01234567, vk8x01234567);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod8x01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod8x01234567));



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
      }
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
