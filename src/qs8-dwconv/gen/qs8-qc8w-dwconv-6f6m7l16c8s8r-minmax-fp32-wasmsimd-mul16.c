// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/multipass-wasmsimd-mul16.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>


void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__wasmsimd_mul16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    size_t kernel_size,
    int32_t* buffer,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 6);

  do {
    const void* w = weights;

    // First pass to process 6 inputs.
    {
      int32_t* b = buffer;
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
      input += 6;

      size_t c = round_up_po2(channels, 8);

      for (; c >= 16; c -= 16) {
        v128_t vacc0123 = wasm_v128_load(w);
        v128_t vacc4567 = wasm_v128_load((const void*) ((uintptr_t) w + 4 * sizeof(int32_t)));
        v128_t vacc89AB = wasm_v128_load((const void*) ((uintptr_t) w + 8 * sizeof(int32_t)));
        v128_t vaccCDEF = wasm_v128_load((const void*) ((uintptr_t) w + 12 * sizeof(int32_t)));

        const v128_t vi0x01234567 = wasm_i16x8_load8x8(i0);
        v128_t vk0x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(int8_t)));
        const v128_t vi0x89ABCDEF = wasm_i16x8_load8x8(i0 + 8);
        v128_t vk0x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(int8_t)));
        i0 += 16;

        v128_t vprod01234567 = wasm_i16x8_mul(vi0x01234567, vk0x01234567);
        v128_t vprod89ABCDEF = wasm_i16x8_mul(vi0x89ABCDEF, vk0x89ABCDEF);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));
        vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod89ABCDEF));
        vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod89ABCDEF));

        const v128_t vi1x01234567 = wasm_i16x8_load8x8(i1);
        v128_t vk1x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(int8_t)));
        const v128_t vi1x89ABCDEF = wasm_i16x8_load8x8(i1 + 8);
        v128_t vk1x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(int8_t)));
        i1 += 16;

        vprod01234567 = wasm_i16x8_mul(vi1x01234567, vk1x01234567);
        vprod89ABCDEF = wasm_i16x8_mul(vi1x89ABCDEF, vk1x89ABCDEF);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));
        vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod89ABCDEF));
        vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod89ABCDEF));

        const v128_t vi2x01234567 = wasm_i16x8_load8x8(i2);
        v128_t vk2x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(int8_t)));
        const v128_t vi2x89ABCDEF = wasm_i16x8_load8x8(i2 + 8);
        v128_t vk2x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(int8_t)));
        i2 += 16;

        vprod01234567 = wasm_i16x8_mul(vi2x01234567, vk2x01234567);
        vprod89ABCDEF = wasm_i16x8_mul(vi2x89ABCDEF, vk2x89ABCDEF);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));
        vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod89ABCDEF));
        vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod89ABCDEF));

        const v128_t vi3x01234567 = wasm_i16x8_load8x8(i3);
        v128_t vk3x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t)));
        const v128_t vi3x89ABCDEF = wasm_i16x8_load8x8(i3 + 8);
        v128_t vk3x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(int8_t)));
        i3 += 16;

        vprod01234567 = wasm_i16x8_mul(vi3x01234567, vk3x01234567);
        vprod89ABCDEF = wasm_i16x8_mul(vi3x89ABCDEF, vk3x89ABCDEF);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));
        vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod89ABCDEF));
        vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod89ABCDEF));

        const v128_t vi4x01234567 = wasm_i16x8_load8x8(i4);
        v128_t vk4x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(int8_t)));
        const v128_t vi4x89ABCDEF = wasm_i16x8_load8x8(i4 + 8);
        v128_t vk4x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(int8_t)));
        i4 += 16;

        vprod01234567 = wasm_i16x8_mul(vi4x01234567, vk4x01234567);
        vprod89ABCDEF = wasm_i16x8_mul(vi4x89ABCDEF, vk4x89ABCDEF);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));
        vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod89ABCDEF));
        vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod89ABCDEF));

        const v128_t vi5x01234567 = wasm_i16x8_load8x8(i5);
        v128_t vk5x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(int8_t)));
        const v128_t vi5x89ABCDEF = wasm_i16x8_load8x8(i5 + 8);
        v128_t vk5x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(int8_t)));
        i5 += 16;

        vprod01234567 = wasm_i16x8_mul(vi5x01234567, vk5x01234567);
        vprod89ABCDEF = wasm_i16x8_mul(vi5x89ABCDEF, vk5x89ABCDEF);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));
        vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod89ABCDEF));
        vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod89ABCDEF));

        w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(int8_t));

        wasm_v128_store(b, vacc0123);
        wasm_v128_store(b + 4, vacc4567);
        wasm_v128_store(b + 8, vacc89AB);
        wasm_v128_store(b + 12, vaccCDEF);
        b += 16;
      }

      if XNN_UNLIKELY(c != 0) {
        do {
          v128_t vacc0123 = wasm_v128_load(w);
          v128_t vacc4567 = wasm_v128_load((const int32_t*) w + 4);

          const v128_t vi0x01234567 = wasm_i16x8_load8x8(i0);
          v128_t vk0x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t)));
          i0 += 8;

          v128_t vprod01234567 = wasm_i16x8_mul(vi0x01234567, vk0x01234567);

          vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
          vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));

          const v128_t vi1x01234567 = wasm_i16x8_load8x8(i1);
          v128_t vk1x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));
          i1 += 8;

          vprod01234567 = wasm_i16x8_mul(vi1x01234567, vk1x01234567);

          vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
          vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));

          const v128_t vi2x01234567 = wasm_i16x8_load8x8(i2);
          v128_t vk2x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));
          i2 += 8;

          vprod01234567 = wasm_i16x8_mul(vi2x01234567, vk2x01234567);

          vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
          vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));

          const v128_t vi3x01234567 = wasm_i16x8_load8x8(i3);
          v128_t vk3x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t)));
          i3 += 8;

          vprod01234567 = wasm_i16x8_mul(vi3x01234567, vk3x01234567);

          vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
          vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));

          const v128_t vi4x01234567 = wasm_i16x8_load8x8(i4);
          v128_t vk4x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(int8_t)));
          i4 += 8;

          vprod01234567 = wasm_i16x8_mul(vi4x01234567, vk4x01234567);

          vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
          vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));

          const v128_t vi5x01234567 = wasm_i16x8_load8x8(i5);
          v128_t vk5x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(int8_t)));
          i5 += 8;

          vprod01234567 = wasm_i16x8_mul(vi5x01234567, vk5x01234567);

          vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
          vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));


          w = (const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t));

          wasm_v128_store(b, vacc0123);
          wasm_v128_store(b + 4, vacc4567);
          b += 8;
          c -= 8;
        } while (c != 0);
      }
    }


    // Middle pass to process 6 inputs in each iteration.
    for (size_t ks = kernel_size - 6; ks > 7; ks -= 6) {
      int32_t* b = buffer;
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
      input += 6;

      size_t c = round_up_po2(channels, 8);

      for (; c >= 16; c -= 16) {
        v128_t vacc0123 = wasm_v128_load(b);
        v128_t vacc4567 = wasm_v128_load(b + 4);
        v128_t vacc89AB = wasm_v128_load(b + 8);
        v128_t vaccCDEF = wasm_v128_load(b + 12);

        const v128_t vi0x01234567 = wasm_i16x8_load8x8(i0);
        v128_t vk0x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 0 * sizeof(int8_t)));
        const v128_t vi0x89ABCDEF = wasm_i16x8_load8x8(i0 + 8);
        v128_t vk0x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int8_t)));
        i0 += 16;

        v128_t vprod01234567 = wasm_i16x8_mul(vi0x01234567, vk0x01234567);
        v128_t vprod89ABCDEF = wasm_i16x8_mul(vi0x89ABCDEF, vk0x89ABCDEF);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));
        vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod89ABCDEF));
        vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod89ABCDEF));

        const v128_t vi1x01234567 = wasm_i16x8_load8x8(i1);
        v128_t vk1x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int8_t)));
        const v128_t vi1x89ABCDEF = wasm_i16x8_load8x8(i1 + 8);
        v128_t vk1x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int8_t)));
        i1 += 16;

        vprod01234567 = wasm_i16x8_mul(vi1x01234567, vk1x01234567);
        vprod89ABCDEF = wasm_i16x8_mul(vi1x89ABCDEF, vk1x89ABCDEF);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));
        vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod89ABCDEF));
        vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod89ABCDEF));

        const v128_t vi2x01234567 = wasm_i16x8_load8x8(i2);
        v128_t vk2x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 32 * sizeof(int8_t)));
        const v128_t vi2x89ABCDEF = wasm_i16x8_load8x8(i2 + 8);
        v128_t vk2x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 40 * sizeof(int8_t)));
        i2 += 16;

        vprod01234567 = wasm_i16x8_mul(vi2x01234567, vk2x01234567);
        vprod89ABCDEF = wasm_i16x8_mul(vi2x89ABCDEF, vk2x89ABCDEF);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));
        vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod89ABCDEF));
        vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod89ABCDEF));

        const v128_t vi3x01234567 = wasm_i16x8_load8x8(i3);
        v128_t vk3x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 48 * sizeof(int8_t)));
        const v128_t vi3x89ABCDEF = wasm_i16x8_load8x8(i3 + 8);
        v128_t vk3x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 56 * sizeof(int8_t)));
        i3 += 16;

        vprod01234567 = wasm_i16x8_mul(vi3x01234567, vk3x01234567);
        vprod89ABCDEF = wasm_i16x8_mul(vi3x89ABCDEF, vk3x89ABCDEF);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));
        vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod89ABCDEF));
        vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod89ABCDEF));

        const v128_t vi4x01234567 = wasm_i16x8_load8x8(i4);
        v128_t vk4x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 64 * sizeof(int8_t)));
        const v128_t vi4x89ABCDEF = wasm_i16x8_load8x8(i4 + 8);
        v128_t vk4x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 72 * sizeof(int8_t)));
        i4 += 16;

        vprod01234567 = wasm_i16x8_mul(vi4x01234567, vk4x01234567);
        vprod89ABCDEF = wasm_i16x8_mul(vi4x89ABCDEF, vk4x89ABCDEF);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));
        vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod89ABCDEF));
        vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod89ABCDEF));

        const v128_t vi5x01234567 = wasm_i16x8_load8x8(i5);
        v128_t vk5x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 80 * sizeof(int8_t)));
        const v128_t vi5x89ABCDEF = wasm_i16x8_load8x8(i5 + 8);
        v128_t vk5x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 88 * sizeof(int8_t)));
        i5 += 16;

        vprod01234567 = wasm_i16x8_mul(vi5x01234567, vk5x01234567);
        vprod89ABCDEF = wasm_i16x8_mul(vi5x89ABCDEF, vk5x89ABCDEF);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));
        vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod89ABCDEF));
        vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod89ABCDEF));

        w = (const void*) ((uintptr_t) w + 96 * sizeof(int8_t));

        wasm_v128_store(b, vacc0123);
        wasm_v128_store(b + 4, vacc4567);
        wasm_v128_store(b + 8, vacc89AB);
        wasm_v128_store(b + 12, vaccCDEF);
        b += 16;
      }

      if XNN_UNLIKELY(c != 0) {
        do {
          v128_t vacc0123 = wasm_v128_load(b);
          v128_t vacc4567 = wasm_v128_load(b + 4);

          const v128_t vi0x01234567 = wasm_i16x8_load8x8(i0);
          v128_t vk0x01234567 = wasm_i16x8_load8x8(w);
          i0 += 8;

          v128_t vprod01234567 = wasm_i16x8_mul(vi0x01234567, vk0x01234567);

          vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
          vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));

          const v128_t vi1x01234567 = wasm_i16x8_load8x8(i1);
          v128_t vk1x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int8_t)));
          i1 += 8;

          vprod01234567 = wasm_i16x8_mul(vi1x01234567, vk1x01234567);

          vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
          vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));

          const v128_t vi2x01234567 = wasm_i16x8_load8x8(i2);
          v128_t vk2x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int8_t)));
          i2 += 8;

          vprod01234567 = wasm_i16x8_mul(vi2x01234567, vk2x01234567);

          vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
          vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));

          const v128_t vi3x01234567 = wasm_i16x8_load8x8(i3);
          v128_t vk3x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int8_t)));
          i3 += 8;

          vprod01234567 = wasm_i16x8_mul(vi3x01234567, vk3x01234567);

          vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
          vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));

          const v128_t vi4x01234567 = wasm_i16x8_load8x8(i4);
          v128_t vk4x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 32 * sizeof(int8_t)));
          i4 += 8;

          vprod01234567 = wasm_i16x8_mul(vi4x01234567, vk4x01234567);

          vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
          vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));

          const v128_t vi5x01234567 = wasm_i16x8_load8x8(i5);
          v128_t vk5x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 40 * sizeof(int8_t)));
          i5 += 8;

          vprod01234567 = wasm_i16x8_mul(vi5x01234567, vk5x01234567);

          vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
          vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));


          w = (const void*) ((uintptr_t) w + 48 * sizeof(int8_t));

          wasm_v128_store(b, vacc0123);
          wasm_v128_store(b + 4, vacc4567);
          b += 8;
          c -= 8;
        } while (c != 0);
      }
    }

    // Last pass to process up to 7 inputs.
    {
      const int32_t* b = buffer;
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

      size_t c = channels;

      for (; c >= 16; c -= 16) {
        v128_t vacc0123 = wasm_v128_load(b);
        v128_t vacc4567 = wasm_v128_load(b + 4);
        v128_t vacc89AB = wasm_v128_load(b + 8);
        v128_t vaccCDEF = wasm_v128_load(b + 12);
        b += 16;

        const v128_t vi0x01234567 = wasm_i16x8_load8x8(i0);
        v128_t vk0x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 0 * sizeof(int8_t)));
        const v128_t vi0x89ABCDEF = wasm_i16x8_load8x8(i0 + 8);
        v128_t vk0x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int8_t)));
        i0 += 16;

        v128_t vprod01234567 = wasm_i16x8_mul(vi0x01234567, vk0x01234567);
        v128_t vprod89ABCDEF = wasm_i16x8_mul(vi0x89ABCDEF, vk0x89ABCDEF);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));
        vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod89ABCDEF));
        vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod89ABCDEF));

        const v128_t vi1x01234567 = wasm_i16x8_load8x8(i1);
        v128_t vk1x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int8_t)));
        const v128_t vi1x89ABCDEF = wasm_i16x8_load8x8(i1 + 8);
        v128_t vk1x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int8_t)));
        i1 += 16;

        vprod01234567 = wasm_i16x8_mul(vi1x01234567, vk1x01234567);
        vprod89ABCDEF = wasm_i16x8_mul(vi1x89ABCDEF, vk1x89ABCDEF);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));
        vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod89ABCDEF));
        vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod89ABCDEF));

        const v128_t vi2x01234567 = wasm_i16x8_load8x8(i2);
        v128_t vk2x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 32 * sizeof(int8_t)));
        const v128_t vi2x89ABCDEF = wasm_i16x8_load8x8(i2 + 8);
        v128_t vk2x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 40 * sizeof(int8_t)));
        i2 += 16;

        vprod01234567 = wasm_i16x8_mul(vi2x01234567, vk2x01234567);
        vprod89ABCDEF = wasm_i16x8_mul(vi2x89ABCDEF, vk2x89ABCDEF);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));
        vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod89ABCDEF));
        vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod89ABCDEF));

        const v128_t vi3x01234567 = wasm_i16x8_load8x8(i3);
        v128_t vk3x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 48 * sizeof(int8_t)));
        const v128_t vi3x89ABCDEF = wasm_i16x8_load8x8(i3 + 8);
        v128_t vk3x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 56 * sizeof(int8_t)));
        i3 += 16;

        vprod01234567 = wasm_i16x8_mul(vi3x01234567, vk3x01234567);
        vprod89ABCDEF = wasm_i16x8_mul(vi3x89ABCDEF, vk3x89ABCDEF);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));
        vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod89ABCDEF));
        vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod89ABCDEF));

        const v128_t vi4x01234567 = wasm_i16x8_load8x8(i4);
        v128_t vk4x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 64 * sizeof(int8_t)));
        const v128_t vi4x89ABCDEF = wasm_i16x8_load8x8(i4 + 8);
        v128_t vk4x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 72 * sizeof(int8_t)));
        i4 += 16;

        vprod01234567 = wasm_i16x8_mul(vi4x01234567, vk4x01234567);
        vprod89ABCDEF = wasm_i16x8_mul(vi4x89ABCDEF, vk4x89ABCDEF);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));
        vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod89ABCDEF));
        vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod89ABCDEF));

        const v128_t vi5x01234567 = wasm_i16x8_load8x8(i5);
        v128_t vk5x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 80 * sizeof(int8_t)));
        const v128_t vi5x89ABCDEF = wasm_i16x8_load8x8(i5 + 8);
        v128_t vk5x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 88 * sizeof(int8_t)));
        i5 += 16;

        vprod01234567 = wasm_i16x8_mul(vi5x01234567, vk5x01234567);
        vprod89ABCDEF = wasm_i16x8_mul(vi5x89ABCDEF, vk5x89ABCDEF);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));
        vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod89ABCDEF));
        vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod89ABCDEF));

        const v128_t vi6x01234567 = wasm_i16x8_load8x8(i6);
        v128_t vk6x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 96 * sizeof(int8_t)));
        const v128_t vi6x89ABCDEF = wasm_i16x8_load8x8(i6 + 8);
        v128_t vk6x89ABCDEF = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 104 * sizeof(int8_t)));
        i6 += 16;

        vprod01234567 = wasm_i16x8_mul(vi6x01234567, vk6x01234567);
        vprod89ABCDEF = wasm_i16x8_mul(vi6x89ABCDEF, vk6x89ABCDEF);

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));
        vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vprod89ABCDEF));
        vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vprod89ABCDEF));


        w = (const void*) ((uintptr_t) w + 112 * sizeof(int8_t));

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

        const v128_t vmagic_bias = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias);
        vacc0123 = wasm_f32x4_add(vacc0123, vmagic_bias);
        vacc4567 = wasm_f32x4_add(vacc4567, vmagic_bias);
        vacc89AB = wasm_f32x4_add(vacc89AB, vmagic_bias);
        vaccCDEF = wasm_f32x4_add(vaccCDEF, vmagic_bias);

        const v128_t vmagic_min = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_min);
        vacc0123 = wasm_i32x4_max(vacc0123, vmagic_min);
        vacc4567 = wasm_i32x4_max(vacc4567, vmagic_min);
        vacc89AB = wasm_i32x4_max(vacc89AB, vmagic_min);
        vaccCDEF = wasm_i32x4_max(vaccCDEF, vmagic_min);

        const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias_less_output_zero_point);
        vacc0123 = wasm_i32x4_sub(vacc0123, vmagic_bias_less_output_zero_point);
        vacc4567 = wasm_i32x4_sub(vacc4567, vmagic_bias_less_output_zero_point);
        vacc89AB = wasm_i32x4_sub(vacc89AB, vmagic_bias_less_output_zero_point);
        vaccCDEF = wasm_i32x4_sub(vaccCDEF, vmagic_bias_less_output_zero_point);

        v128_t vout01234567 = wasm_i16x8_narrow_i32x4(vacc0123, vacc4567);
        v128_t vout89ABCDEF = wasm_i16x8_narrow_i32x4(vacc89AB, vaccCDEF);

        v128_t vout0123456789ABCDEF = wasm_i8x16_narrow_i16x8(vout01234567, vout89ABCDEF);

        const v128_t voutput_max = wasm_v128_load64_splat(params->fp32_wasmsimd.output_max);
        vout0123456789ABCDEF = wasm_i8x16_min(vout0123456789ABCDEF, voutput_max);

        wasm_v128_store(output, vout0123456789ABCDEF);
        output += 16;
      }

      if XNN_UNLIKELY(c != 0) {
        do {
          v128_t vacc0123 = wasm_v128_load(b);
          v128_t vacc4567 = wasm_v128_load(b + 4);
          b += 8;

          const v128_t vi0x01234567 = wasm_i16x8_load8x8(i0);
          v128_t vk0x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w));
          i0 += 8;

          v128_t vprod01234567 = wasm_i16x8_mul(vi0x01234567, vk0x01234567);

          vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
          vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));

          const v128_t vi1x01234567 = wasm_i16x8_load8x8(i1);
          v128_t vk1x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 8 * sizeof(int8_t)));
          i1 += 8;

          vprod01234567 = wasm_i16x8_mul(vi1x01234567, vk1x01234567);

          vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
          vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));

          const v128_t vi2x01234567 = wasm_i16x8_load8x8(i2);
          v128_t vk2x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 16 * sizeof(int8_t)));
          i2 += 8;

          vprod01234567 = wasm_i16x8_mul(vi2x01234567, vk2x01234567);

          vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
          vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));

          const v128_t vi3x01234567 = wasm_i16x8_load8x8(i3);
          v128_t vk3x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 24 * sizeof(int8_t)));
          i3 += 8;

          vprod01234567 = wasm_i16x8_mul(vi3x01234567, vk3x01234567);

          vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
          vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));

          const v128_t vi4x01234567 = wasm_i16x8_load8x8(i4);
          v128_t vk4x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 32 * sizeof(int8_t)));
          i4 += 8;

          vprod01234567 = wasm_i16x8_mul(vi4x01234567, vk4x01234567);

          vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
          vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));

          const v128_t vi5x01234567 = wasm_i16x8_load8x8(i5);
          v128_t vk5x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 40 * sizeof(int8_t)));
          i5 += 8;

          vprod01234567 = wasm_i16x8_mul(vi5x01234567, vk5x01234567);

          vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
          vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));

          const v128_t vi6x01234567 = wasm_i16x8_load8x8(i6);
          v128_t vk6x01234567 = wasm_i16x8_load8x8((const void*) ((uintptr_t) w + 48 * sizeof(int8_t)));
          i6 += 8;

          vprod01234567 = wasm_i16x8_mul(vi6x01234567, vk6x01234567);

          vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vprod01234567));
          vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vprod01234567));


          vacc0123 = wasm_f32x4_convert_i32x4(vacc0123);
          vacc4567 = wasm_f32x4_convert_i32x4(vacc4567);

          const v128_t vscale0123 = wasm_v128_load((const float*) ((uintptr_t) w + 56 * sizeof(int8_t)));
          const v128_t vscale4567 = wasm_v128_load((const float*) ((uintptr_t) w + 56 * sizeof(int8_t) + 4 * sizeof(float)));

          vacc0123 = wasm_f32x4_mul(vacc0123, vscale0123);
          vacc4567 = wasm_f32x4_mul(vacc4567, vscale4567);

          w = (void*) ((uintptr_t) w + 56 + 8 * sizeof(float));

          const v128_t vmagic_bias = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias);
          vacc0123 = wasm_f32x4_add(vacc0123, vmagic_bias);
          vacc4567 = wasm_f32x4_add(vacc4567, vmagic_bias);

          const v128_t vmagic_min = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_min);
          vacc0123 = wasm_i32x4_max(vacc0123, vmagic_min);
          vacc4567 = wasm_i32x4_max(vacc4567, vmagic_min);

          const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias_less_output_zero_point);
          vacc0123 = wasm_i32x4_sub(vacc0123, vmagic_bias_less_output_zero_point);
          vacc4567 = wasm_i32x4_sub(vacc4567, vmagic_bias_less_output_zero_point);

          v128_t vout01234567 = wasm_i16x8_narrow_i32x4(vacc0123, vacc4567);
          v128_t vout0123456701234567 = wasm_i8x16_narrow_i16x8(vout01234567, vout01234567);

          const v128_t voutput_max = wasm_v128_load64_splat(params->fp32_wasmsimd.output_max);
          vout0123456701234567 = wasm_i8x16_min(vout0123456701234567, voutput_max);

          if XNN_LIKELY(c >= 8) {
            wasm_v128_store64_lane(output, vout0123456701234567, 0);
            output += 8;
            c -= 8;
          } else {
            if (c & 4) {
              wasm_v128_store32_lane(output, vout0123456701234567, 0);
              vout0123456701234567 = wasm_u64x2_shr(vout0123456701234567, 32);
              output += 4;
            }
            if (c & 2) {
              wasm_v128_store16_lane(output, vout0123456701234567, 0);
              vout0123456701234567 = wasm_u32x4_shr(vout0123456701234567, 16);
              output += 2;
            }
            if (c & 1) {
              wasm_v128_store8_lane(output, vout0123456701234567, 0);
              output += 1;
            }
            c = 0;
          }
        } while (c != 0);
      }
    }

    input = (const int8_t**) ((uintptr_t) input + input_stride);
    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
