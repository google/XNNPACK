// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/unipass-wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/dwconv.h>


void xnn_f32_dwconv_minmax_ukernel_9p8c__wasmsimd_x86(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const v128_t vmin = wasm_v128_load64_splat(params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load64_splat(params->wasmsimd.max);
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      v128_t vacc0123p0 = wasm_v128_load(w);
      v128_t vacc4567p0 = wasm_v128_load(w + 4);


      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vi0x4567 = wasm_v128_load(i0 + 4);
      i0 += 8;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      const v128_t vk0x4567 = wasm_v128_load(w + 12);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi0x0123, vk0x0123), vacc0123p0);
      vacc4567p0 = wasm_f32x4_add(wasm_f32x4_mul(vi0x4567, vk0x4567), vacc4567p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vi1x4567 = wasm_v128_load(i1 + 4);
      i1 += 8;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      const v128_t vk1x4567 = wasm_v128_load(w + 20);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi1x0123, vk1x0123), vacc0123p0);
      vacc4567p0 = wasm_f32x4_add(wasm_f32x4_mul(vi1x4567, vk1x4567), vacc4567p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vi2x4567 = wasm_v128_load(i2 + 4);
      i2 += 8;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      const v128_t vk2x4567 = wasm_v128_load(w + 28);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi2x0123, vk2x0123), vacc0123p0);
      vacc4567p0 = wasm_f32x4_add(wasm_f32x4_mul(vi2x4567, vk2x4567), vacc4567p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vi3x4567 = wasm_v128_load(i3 + 4);
      i3 += 8;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      const v128_t vk3x4567 = wasm_v128_load(w + 36);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi3x0123, vk3x0123), vacc0123p0);
      vacc4567p0 = wasm_f32x4_add(wasm_f32x4_mul(vi3x4567, vk3x4567), vacc4567p0);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      const v128_t vi4x4567 = wasm_v128_load(i4 + 4);
      i4 += 8;

      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      const v128_t vk4x4567 = wasm_v128_load(w + 44);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi4x0123, vk4x0123), vacc0123p0);
      vacc4567p0 = wasm_f32x4_add(wasm_f32x4_mul(vi4x4567, vk4x4567), vacc4567p0);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      const v128_t vi5x4567 = wasm_v128_load(i5 + 4);
      i5 += 8;

      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      const v128_t vk5x4567 = wasm_v128_load(w + 52);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi5x0123, vk5x0123), vacc0123p0);
      vacc4567p0 = wasm_f32x4_add(wasm_f32x4_mul(vi5x4567, vk5x4567), vacc4567p0);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      const v128_t vi6x4567 = wasm_v128_load(i6 + 4);
      i6 += 8;

      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      const v128_t vk6x4567 = wasm_v128_load(w + 60);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi6x0123, vk6x0123), vacc0123p0);
      vacc4567p0 = wasm_f32x4_add(wasm_f32x4_mul(vi6x4567, vk6x4567), vacc4567p0);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      const v128_t vi7x4567 = wasm_v128_load(i7 + 4);
      i7 += 8;

      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      const v128_t vk7x4567 = wasm_v128_load(w + 68);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi7x0123, vk7x0123), vacc0123p0);
      vacc4567p0 = wasm_f32x4_add(wasm_f32x4_mul(vi7x4567, vk7x4567), vacc4567p0);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      const v128_t vi8x4567 = wasm_v128_load(i8 + 4);
      i8 += 8;

      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      const v128_t vk8x4567 = wasm_v128_load(w + 76);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi8x0123, vk8x0123), vacc0123p0);
      vacc4567p0 = wasm_f32x4_add(wasm_f32x4_mul(vi8x4567, vk8x4567), vacc4567p0);

      w += 80;


      v128_t vacc0123 = wasm_f32x4_pmax(vmin, vacc0123p0);
      v128_t vacc4567 = wasm_f32x4_pmax(vmin, vacc4567p0);

      vacc0123 = wasm_f32x4_pmin(vmax, vacc0123);
      vacc4567 = wasm_f32x4_pmin(vmax, vacc4567);

      wasm_v128_store(output, vacc0123);
      wasm_v128_store(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi0x0123, vk0x0123), vacc0123p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      i1 += 4;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi1x0123, vk1x0123), vacc0123p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      i2 += 4;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi2x0123, vk2x0123), vacc0123p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      i3 += 4;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi3x0123, vk3x0123), vacc0123p0);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      i4 += 4;

      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi4x0123, vk4x0123), vacc0123p0);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      i5 += 4;

      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi5x0123, vk5x0123), vacc0123p0);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      i6 += 4;

      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi6x0123, vk6x0123), vacc0123p0);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      i7 += 4;

      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi7x0123, vk7x0123), vacc0123p0);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      i8 += 4;

      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi8x0123, vk8x0123), vacc0123p0);

      w += 4;


      v128_t vacc0123 = wasm_f32x4_pmax(vmin, vacc0123p0);
      vacc0123 = wasm_f32x4_pmin(vmax, vacc0123);

      wasm_v128_store(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi0x0123, vk0x0123), vacc0123p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi1x0123, vk1x0123), vacc0123p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi2x0123, vk2x0123), vacc0123p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi3x0123, vk3x0123), vacc0123p0);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi4x0123, vk4x0123), vacc0123p0);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi5x0123, vk5x0123), vacc0123p0);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi6x0123, vk6x0123), vacc0123p0);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi7x0123, vk7x0123), vacc0123p0);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi8x0123, vk8x0123), vacc0123p0);


      v128_t vacc0123 = wasm_f32x4_pmax(vmin, vacc0123p0);
      vacc0123 = wasm_f32x4_pmin(vmax, vacc0123);

      if (c & 2) {
        wasm_v128_store64_lane(output, vacc0123, 0);
        vacc0123 = wasm_v64x2_shuffle(vacc0123, vacc0123, 1, 1);
        output += 2;
      }
      if (c & 1) {
        wasm_v128_store32_lane(output, vacc0123, 0);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
