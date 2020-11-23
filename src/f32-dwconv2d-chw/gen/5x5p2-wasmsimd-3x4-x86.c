// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv2d-chw/5x5p2-wasmsimd-3x4.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>


#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>



void xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_3x4(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float *zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top == 2);

  const v128_t vmask = wasm_v128_load(params->scalar.mask);
  const v128_t vmax = wasm_v32x4_load_splat(&params->scalar.max);
  const v128_t vmin = wasm_v32x4_load_splat(&params->scalar.min);

  const v128_t vbias = wasm_v32x4_load_splat(weights);
  const v128_t vk00 = wasm_v32x4_load_splat(weights + 1);
  const v128_t vk01 = wasm_v32x4_load_splat(weights + 2);
  const v128_t vk02 = wasm_v32x4_load_splat(weights + 3);
  const v128_t vk03 = wasm_v32x4_load_splat(weights + 4);
  const v128_t vk04 = wasm_v32x4_load_splat(weights + 5);
  const v128_t vk10 = wasm_v32x4_load_splat(weights + 6);
  const v128_t vk11 = wasm_v32x4_load_splat(weights + 7);
  const v128_t vk12 = wasm_v32x4_load_splat(weights + 8);
  const v128_t vk13 = wasm_v32x4_load_splat(weights + 9);
  const v128_t vk14 = wasm_v32x4_load_splat(weights + 10);
  const v128_t vk20 = wasm_v32x4_load_splat(weights + 11);
  const v128_t vk21 = wasm_v32x4_load_splat(weights + 12);
  const v128_t vk22 = wasm_v32x4_load_splat(weights + 13);
  const v128_t vk23 = wasm_v32x4_load_splat(weights + 14);
  const v128_t vk24 = wasm_v32x4_load_splat(weights + 15);
  const v128_t vk30 = wasm_v32x4_load_splat(weights + 16);
  const v128_t vk31 = wasm_v32x4_load_splat(weights + 17);
  const v128_t vk32 = wasm_v32x4_load_splat(weights + 18);
  const v128_t vk33 = wasm_v32x4_load_splat(weights + 19);
  const v128_t vk34 = wasm_v32x4_load_splat(weights + 20);
  const v128_t vk40 = wasm_v32x4_load_splat(weights + 21);
  const v128_t vk41 = wasm_v32x4_load_splat(weights + 22);
  const v128_t vk42 = wasm_v32x4_load_splat(weights + 23);
  const v128_t vk43 = wasm_v32x4_load_splat(weights + 24);
  const v128_t vk44 = wasm_v32x4_load_splat(weights + 25);

  const v128_t vzero = wasm_f32x4_splat(0.0f);

  const size_t input_decrement = round_up_po2(input_width, 4 * sizeof(float));

  const float* i0 = zero;
  const float* i1 = zero;
  const float* i2 = input;
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_width);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + input_width);
  float* o2 = (float*) ((uintptr_t) o1 + input_width);

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i3 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(output_height <= 2) {
      i4 = zero;
      o2 = o1;
    }
    if XNN_UNPREDICTABLE(output_height < 4) {
      i5 = zero;
    }
    if XNN_UNPREDICTABLE(output_height <= 4) {
      i6 = zero;
    }

    v128_t vi0x0123 = vzero;
    v128_t vi1x0123 = vzero;
    v128_t vi2x0123 = vzero;
    v128_t vi3x0123 = vzero;
    v128_t vi4x0123 = vzero;
    v128_t vi5x0123 = vzero;
    v128_t vi6x0123 = vzero;
    v128_t vi0x4567 = wasm_v128_load(i0); i0 += 4;
    v128_t vi1x4567 = wasm_v128_load(i1); i1 += 4;
    v128_t vi2x4567 = wasm_v128_load(i2); i2 += 4;
    v128_t vi3x4567 = wasm_v128_load(i3); i3 += 4;
    v128_t vi4x4567 = wasm_v128_load(i4); i4 += 4;
    v128_t vi5x4567 = wasm_v128_load(i5); i5 += 4;
    v128_t vi6x4567 = wasm_v128_load(i6); i6 += 4;

    size_t w = input_width;
    for (; w > 8 * sizeof(float); w -= 4 * sizeof(float)) {
      v128_t vo4567p0 = vbias;
      v128_t vo4567p1 = vbias;
      v128_t vo4567p2 = vbias;

      const v128_t vi0x89AB = wasm_v128_load(i0); i0 += 4;
      const v128_t vi1x89AB = wasm_v128_load(i1); i1 += 4;
      const v128_t vi2x89AB = wasm_v128_load(i2); i2 += 4;
      const v128_t vi3x89AB = wasm_v128_load(i3); i3 += 4;
      const v128_t vi4x89AB = wasm_v128_load(i4); i4 += 4;
      const v128_t vi5x89AB = wasm_v128_load(i5); i5 += 4;
      const v128_t vi6x89AB = wasm_v128_load(i6); i6 += 4;

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi0x4567, vk02));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi1x4567, vk02));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi2x4567, vk02));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi1x4567, vk12));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi2x4567, vk12));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi3x4567, vk12));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi2x4567, vk22));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi3x4567, vk22));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi4x4567, vk22));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi3x4567, vk32));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi4x4567, vk32));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi5x4567, vk32));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi4x4567, vk42));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi5x4567, vk42));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi6x4567, vk42));

      const v128_t vi0x3456 = wasm_v32x4_shuffle(vi0x0123, vi0x4567, 3, 4, 5, 6);
      const v128_t vi1x3456 = wasm_v32x4_shuffle(vi1x0123, vi1x4567, 3, 4, 5, 6);
      const v128_t vi2x3456 = wasm_v32x4_shuffle(vi2x0123, vi2x4567, 3, 4, 5, 6);
      const v128_t vi3x3456 = wasm_v32x4_shuffle(vi3x0123, vi3x4567, 3, 4, 5, 6);
      const v128_t vi4x3456 = wasm_v32x4_shuffle(vi4x0123, vi4x4567, 3, 4, 5, 6);
      const v128_t vi5x3456 = wasm_v32x4_shuffle(vi5x0123, vi5x4567, 3, 4, 5, 6);
      const v128_t vi6x3456 = wasm_v32x4_shuffle(vi6x0123, vi6x4567, 3, 4, 5, 6);

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi0x3456, vk01));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi1x3456, vk01));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi2x3456, vk01));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi1x3456, vk11));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi2x3456, vk11));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi3x3456, vk11));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi2x3456, vk21));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi3x3456, vk21));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi4x3456, vk21));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi3x3456, vk31));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi4x3456, vk31));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi5x3456, vk31));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi4x3456, vk41));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi5x3456, vk41));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi6x3456, vk41));

      const v128_t vi0x2345 = wasm_v32x4_shuffle(vi0x0123, vi0x4567, 2, 3, 4, 5);
      const v128_t vi1x2345 = wasm_v32x4_shuffle(vi1x0123, vi1x4567, 2, 3, 4, 5);
      const v128_t vi2x2345 = wasm_v32x4_shuffle(vi2x0123, vi2x4567, 2, 3, 4, 5);
      const v128_t vi3x2345 = wasm_v32x4_shuffle(vi3x0123, vi3x4567, 2, 3, 4, 5);
      const v128_t vi4x2345 = wasm_v32x4_shuffle(vi4x0123, vi4x4567, 2, 3, 4, 5);
      const v128_t vi5x2345 = wasm_v32x4_shuffle(vi5x0123, vi5x4567, 2, 3, 4, 5);
      const v128_t vi6x2345 = wasm_v32x4_shuffle(vi6x0123, vi6x4567, 2, 3, 4, 5);

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi0x2345, vk00));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi1x2345, vk00));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi2x2345, vk00));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi1x2345, vk10));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi2x2345, vk10));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi3x2345, vk10));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi2x2345, vk20));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi3x2345, vk20));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi4x2345, vk20));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi3x2345, vk30));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi4x2345, vk30));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi5x2345, vk30));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi4x2345, vk40));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi5x2345, vk40));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi6x2345, vk40));

      vi0x0123 = vi0x4567;
      vi1x0123 = vi1x4567;
      vi2x0123 = vi2x4567;
      vi3x0123 = vi3x4567;
      vi4x0123 = vi4x4567;
      vi5x0123 = vi5x4567;
      vi6x0123 = vi6x4567;

      const v128_t vi0x5678 = wasm_v32x4_shuffle(vi0x4567, vi0x89AB, 1, 2, 3, 4);
      const v128_t vi1x5678 = wasm_v32x4_shuffle(vi1x4567, vi1x89AB, 1, 2, 3, 4);
      const v128_t vi2x5678 = wasm_v32x4_shuffle(vi2x4567, vi2x89AB, 1, 2, 3, 4);
      const v128_t vi3x5678 = wasm_v32x4_shuffle(vi3x4567, vi3x89AB, 1, 2, 3, 4);
      const v128_t vi4x5678 = wasm_v32x4_shuffle(vi4x4567, vi4x89AB, 1, 2, 3, 4);
      const v128_t vi5x5678 = wasm_v32x4_shuffle(vi5x4567, vi5x89AB, 1, 2, 3, 4);
      const v128_t vi6x5678 = wasm_v32x4_shuffle(vi6x4567, vi6x89AB, 1, 2, 3, 4);

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi0x5678, vk03));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi1x5678, vk03));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi2x5678, vk03));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi1x5678, vk13));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi2x5678, vk13));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi3x5678, vk13));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi2x5678, vk23));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi3x5678, vk23));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi4x5678, vk23));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi3x5678, vk33));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi4x5678, vk33));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi5x5678, vk33));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi4x5678, vk43));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi5x5678, vk43));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi6x5678, vk43));

      const v128_t vi0x6789 = wasm_v32x4_shuffle(vi0x4567, vi0x89AB, 2, 3, 4, 5);
      const v128_t vi1x6789 = wasm_v32x4_shuffle(vi1x4567, vi1x89AB, 2, 3, 4, 5);
      const v128_t vi2x6789 = wasm_v32x4_shuffle(vi2x4567, vi2x89AB, 2, 3, 4, 5);
      const v128_t vi3x6789 = wasm_v32x4_shuffle(vi3x4567, vi3x89AB, 2, 3, 4, 5);
      const v128_t vi4x6789 = wasm_v32x4_shuffle(vi4x4567, vi4x89AB, 2, 3, 4, 5);
      const v128_t vi5x6789 = wasm_v32x4_shuffle(vi5x4567, vi5x89AB, 2, 3, 4, 5);
      const v128_t vi6x6789 = wasm_v32x4_shuffle(vi6x4567, vi6x89AB, 2, 3, 4, 5);

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi0x6789, vk04));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi1x6789, vk04));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi2x6789, vk04));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi1x6789, vk14));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi2x6789, vk14));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi3x6789, vk14));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi2x6789, vk24));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi3x6789, vk24));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi4x6789, vk24));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi3x6789, vk34));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi4x6789, vk34));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi5x6789, vk34));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi4x6789, vk44));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi5x6789, vk44));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi6x6789, vk44));

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;
      vi4x4567 = vi4x89AB;
      vi5x4567 = vi5x89AB;
      vi6x4567 = vi6x89AB;

      v128_t vo0 = vo4567p0;
      v128_t vo1 = vo4567p1;
      v128_t vo2 = vo4567p2;

      vo0 = wasm_v128_bitselect(vmin, vo0, wasm_f32x4_lt(vo0, vmin));
      vo1 = wasm_v128_bitselect(vmin, vo1, wasm_f32x4_lt(vo1, vmin));
      vo2 = wasm_v128_bitselect(vmin, vo2, wasm_f32x4_lt(vo2, vmin));
      vo0 = wasm_v128_bitselect(vo0, vmax, wasm_f32x4_le(vo0, vmax));
      vo1 = wasm_v128_bitselect(vo1, vmax, wasm_f32x4_le(vo1, vmax));
      vo2 = wasm_v128_bitselect(vo2, vmax, wasm_f32x4_le(vo2, vmax));

      wasm_v128_store(o2, vo2); o2 += 4;
      wasm_v128_store(o1, vo1); o1 += 4;
      wasm_v128_store(o0, vo0); o0 += 4;
    }
    // Always process the last block of 5..8 pixels.
    if XNN_LIKELY(w > 4 * sizeof(float)) {
      v128_t vo4567p0 = vbias;
      v128_t vo4567p1 = vbias;
      v128_t vo4567p2 = vbias;

      v128_t vi0x89AB = wasm_v128_load(i0); i0 += 4;
      v128_t vi1x89AB = wasm_v128_load(i1); i1 += 4;
      v128_t vi2x89AB = wasm_v128_load(i2); i2 += 4;
      v128_t vi3x89AB = wasm_v128_load(i3); i3 += 4;
      v128_t vi4x89AB = wasm_v128_load(i4); i4 += 4;
      v128_t vi5x89AB = wasm_v128_load(i5); i5 += 4;
      v128_t vi6x89AB = wasm_v128_load(i6); i6 += 4;

      vi0x89AB = wasm_v128_and(vmask, vi0x89AB);
      vi1x89AB = wasm_v128_and(vmask, vi1x89AB);
      vi2x89AB = wasm_v128_and(vmask, vi2x89AB);
      vi3x89AB = wasm_v128_and(vmask, vi3x89AB);
      vi4x89AB = wasm_v128_and(vmask, vi4x89AB);
      vi5x89AB = wasm_v128_and(vmask, vi5x89AB);
      vi6x89AB = wasm_v128_and(vmask, vi6x89AB);

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi0x4567, vk02));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi1x4567, vk02));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi2x4567, vk02));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi1x4567, vk12));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi2x4567, vk12));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi3x4567, vk12));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi2x4567, vk22));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi3x4567, vk22));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi4x4567, vk22));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi3x4567, vk32));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi4x4567, vk32));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi5x4567, vk32));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi4x4567, vk42));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi5x4567, vk42));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi6x4567, vk42));

      const v128_t vi0x3456 = wasm_v32x4_shuffle(vi0x0123, vi0x4567, 3, 4, 5, 6);
      const v128_t vi1x3456 = wasm_v32x4_shuffle(vi1x0123, vi1x4567, 3, 4, 5, 6);
      const v128_t vi2x3456 = wasm_v32x4_shuffle(vi2x0123, vi2x4567, 3, 4, 5, 6);
      const v128_t vi3x3456 = wasm_v32x4_shuffle(vi3x0123, vi3x4567, 3, 4, 5, 6);
      const v128_t vi4x3456 = wasm_v32x4_shuffle(vi4x0123, vi4x4567, 3, 4, 5, 6);
      const v128_t vi5x3456 = wasm_v32x4_shuffle(vi5x0123, vi5x4567, 3, 4, 5, 6);
      const v128_t vi6x3456 = wasm_v32x4_shuffle(vi6x0123, vi6x4567, 3, 4, 5, 6);

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi0x3456, vk01));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi1x3456, vk01));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi2x3456, vk01));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi1x3456, vk11));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi2x3456, vk11));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi3x3456, vk11));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi2x3456, vk21));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi3x3456, vk21));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi4x3456, vk21));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi3x3456, vk31));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi4x3456, vk31));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi5x3456, vk31));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi4x3456, vk41));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi5x3456, vk41));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi6x3456, vk41));

      const v128_t vi0x2345 = wasm_v32x4_shuffle(vi0x0123, vi0x4567, 2, 3, 4, 5);
      const v128_t vi1x2345 = wasm_v32x4_shuffle(vi1x0123, vi1x4567, 2, 3, 4, 5);
      const v128_t vi2x2345 = wasm_v32x4_shuffle(vi2x0123, vi2x4567, 2, 3, 4, 5);
      const v128_t vi3x2345 = wasm_v32x4_shuffle(vi3x0123, vi3x4567, 2, 3, 4, 5);
      const v128_t vi4x2345 = wasm_v32x4_shuffle(vi4x0123, vi4x4567, 2, 3, 4, 5);
      const v128_t vi5x2345 = wasm_v32x4_shuffle(vi5x0123, vi5x4567, 2, 3, 4, 5);
      const v128_t vi6x2345 = wasm_v32x4_shuffle(vi6x0123, vi6x4567, 2, 3, 4, 5);

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi0x2345, vk00));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi1x2345, vk00));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi2x2345, vk00));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi1x2345, vk10));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi2x2345, vk10));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi3x2345, vk10));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi2x2345, vk20));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi3x2345, vk20));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi4x2345, vk20));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi3x2345, vk30));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi4x2345, vk30));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi5x2345, vk30));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi4x2345, vk40));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi5x2345, vk40));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi6x2345, vk40));

      vi0x0123 = vi0x4567;
      vi1x0123 = vi1x4567;
      vi2x0123 = vi2x4567;
      vi3x0123 = vi3x4567;
      vi4x0123 = vi4x4567;
      vi5x0123 = vi5x4567;
      vi6x0123 = vi6x4567;

      const v128_t vi0x5678 = wasm_v32x4_shuffle(vi0x4567, vi0x89AB, 1, 2, 3, 4);
      const v128_t vi1x5678 = wasm_v32x4_shuffle(vi1x4567, vi1x89AB, 1, 2, 3, 4);
      const v128_t vi2x5678 = wasm_v32x4_shuffle(vi2x4567, vi2x89AB, 1, 2, 3, 4);
      const v128_t vi3x5678 = wasm_v32x4_shuffle(vi3x4567, vi3x89AB, 1, 2, 3, 4);
      const v128_t vi4x5678 = wasm_v32x4_shuffle(vi4x4567, vi4x89AB, 1, 2, 3, 4);
      const v128_t vi5x5678 = wasm_v32x4_shuffle(vi5x4567, vi5x89AB, 1, 2, 3, 4);
      const v128_t vi6x5678 = wasm_v32x4_shuffle(vi6x4567, vi6x89AB, 1, 2, 3, 4);

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi0x5678, vk03));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi1x5678, vk03));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi2x5678, vk03));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi1x5678, vk13));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi2x5678, vk13));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi3x5678, vk13));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi2x5678, vk23));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi3x5678, vk23));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi4x5678, vk23));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi3x5678, vk33));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi4x5678, vk33));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi5x5678, vk33));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi4x5678, vk43));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi5x5678, vk43));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi6x5678, vk43));

      const v128_t vi0x6789 = wasm_v32x4_shuffle(vi0x4567, vi0x89AB, 2, 3, 4, 5);
      const v128_t vi1x6789 = wasm_v32x4_shuffle(vi1x4567, vi1x89AB, 2, 3, 4, 5);
      const v128_t vi2x6789 = wasm_v32x4_shuffle(vi2x4567, vi2x89AB, 2, 3, 4, 5);
      const v128_t vi3x6789 = wasm_v32x4_shuffle(vi3x4567, vi3x89AB, 2, 3, 4, 5);
      const v128_t vi4x6789 = wasm_v32x4_shuffle(vi4x4567, vi4x89AB, 2, 3, 4, 5);
      const v128_t vi5x6789 = wasm_v32x4_shuffle(vi5x4567, vi5x89AB, 2, 3, 4, 5);
      const v128_t vi6x6789 = wasm_v32x4_shuffle(vi6x4567, vi6x89AB, 2, 3, 4, 5);

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi0x6789, vk04));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi1x6789, vk04));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi2x6789, vk04));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi1x6789, vk14));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi2x6789, vk14));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi3x6789, vk14));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi2x6789, vk24));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi3x6789, vk24));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi4x6789, vk24));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi3x6789, vk34));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi4x6789, vk34));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi5x6789, vk34));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi4x6789, vk44));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi5x6789, vk44));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi6x6789, vk44));

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;
      vi4x4567 = vi4x89AB;
      vi5x4567 = vi5x89AB;
      vi6x4567 = vi6x89AB;

      v128_t vo0 = vo4567p0;
      v128_t vo1 = vo4567p1;
      v128_t vo2 = vo4567p2;

      vo0 = wasm_v128_bitselect(vmin, vo0, wasm_f32x4_lt(vo0, vmin));
      vo1 = wasm_v128_bitselect(vmin, vo1, wasm_f32x4_lt(vo1, vmin));
      vo2 = wasm_v128_bitselect(vmin, vo2, wasm_f32x4_lt(vo2, vmin));
      vo0 = wasm_v128_bitselect(vo0, vmax, wasm_f32x4_le(vo0, vmax));
      vo1 = wasm_v128_bitselect(vo1, vmax, wasm_f32x4_le(vo1, vmax));
      vo2 = wasm_v128_bitselect(vo2, vmax, wasm_f32x4_le(vo2, vmax));

      wasm_v128_store(o2, vo2); o2 += 4;
      wasm_v128_store(o1, vo1); o1 += 4;
      wasm_v128_store(o0, vo0); o0 += 4;
      w -= 4 * sizeof(float);
    }
    assert(w >= 1 * sizeof(float));
    assert(w <= 4 * sizeof(float));
    {
      v128_t vo4567p0 = vbias;
      v128_t vo4567p1 = vbias;
      v128_t vo4567p2 = vbias;

      // This might have already happened if there are more than 4 pixels, but we can't count on it.
      vi0x4567 = wasm_v128_and(vmask, vi0x4567);
      vi1x4567 = wasm_v128_and(vmask, vi1x4567);
      vi2x4567 = wasm_v128_and(vmask, vi2x4567);
      vi3x4567 = wasm_v128_and(vmask, vi3x4567);
      vi4x4567 = wasm_v128_and(vmask, vi4x4567);
      vi5x4567 = wasm_v128_and(vmask, vi5x4567);
      vi6x4567 = wasm_v128_and(vmask, vi6x4567);

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi0x4567, vk02));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi1x4567, vk02));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi2x4567, vk02));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi1x4567, vk12));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi2x4567, vk12));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi3x4567, vk12));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi2x4567, vk22));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi3x4567, vk22));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi4x4567, vk22));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi3x4567, vk32));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi4x4567, vk32));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi5x4567, vk32));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi4x4567, vk42));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi5x4567, vk42));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi6x4567, vk42));

      const v128_t vi0x3456 = wasm_v32x4_shuffle(vi0x0123, vi0x4567, 3, 4, 5, 6);
      const v128_t vi1x3456 = wasm_v32x4_shuffle(vi1x0123, vi1x4567, 3, 4, 5, 6);
      const v128_t vi2x3456 = wasm_v32x4_shuffle(vi2x0123, vi2x4567, 3, 4, 5, 6);
      const v128_t vi3x3456 = wasm_v32x4_shuffle(vi3x0123, vi3x4567, 3, 4, 5, 6);
      const v128_t vi4x3456 = wasm_v32x4_shuffle(vi4x0123, vi4x4567, 3, 4, 5, 6);
      const v128_t vi5x3456 = wasm_v32x4_shuffle(vi5x0123, vi5x4567, 3, 4, 5, 6);
      const v128_t vi6x3456 = wasm_v32x4_shuffle(vi6x0123, vi6x4567, 3, 4, 5, 6);

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi0x3456, vk01));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi1x3456, vk01));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi2x3456, vk01));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi1x3456, vk11));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi2x3456, vk11));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi3x3456, vk11));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi2x3456, vk21));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi3x3456, vk21));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi4x3456, vk21));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi3x3456, vk31));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi4x3456, vk31));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi5x3456, vk31));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi4x3456, vk41));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi5x3456, vk41));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi6x3456, vk41));

      const v128_t vi0x2345 = wasm_v32x4_shuffle(vi0x0123, vi0x4567, 2, 3, 4, 5);
      const v128_t vi1x2345 = wasm_v32x4_shuffle(vi1x0123, vi1x4567, 2, 3, 4, 5);
      const v128_t vi2x2345 = wasm_v32x4_shuffle(vi2x0123, vi2x4567, 2, 3, 4, 5);
      const v128_t vi3x2345 = wasm_v32x4_shuffle(vi3x0123, vi3x4567, 2, 3, 4, 5);
      const v128_t vi4x2345 = wasm_v32x4_shuffle(vi4x0123, vi4x4567, 2, 3, 4, 5);
      const v128_t vi5x2345 = wasm_v32x4_shuffle(vi5x0123, vi5x4567, 2, 3, 4, 5);
      const v128_t vi6x2345 = wasm_v32x4_shuffle(vi6x0123, vi6x4567, 2, 3, 4, 5);

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi0x2345, vk00));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi1x2345, vk00));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi2x2345, vk00));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi1x2345, vk10));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi2x2345, vk10));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi3x2345, vk10));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi2x2345, vk20));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi3x2345, vk20));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi4x2345, vk20));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi3x2345, vk30));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi4x2345, vk30));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi5x2345, vk30));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi4x2345, vk40));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi5x2345, vk40));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi6x2345, vk40));

      const v128_t vi0x5678 = wasm_v32x4_shuffle(vi0x4567, vzero, 1, 2, 3, 4);
      const v128_t vi1x5678 = wasm_v32x4_shuffle(vi1x4567, vzero, 1, 2, 3, 4);
      const v128_t vi2x5678 = wasm_v32x4_shuffle(vi2x4567, vzero, 1, 2, 3, 4);
      const v128_t vi3x5678 = wasm_v32x4_shuffle(vi3x4567, vzero, 1, 2, 3, 4);
      const v128_t vi4x5678 = wasm_v32x4_shuffle(vi4x4567, vzero, 1, 2, 3, 4);
      const v128_t vi5x5678 = wasm_v32x4_shuffle(vi5x4567, vzero, 1, 2, 3, 4);
      const v128_t vi6x5678 = wasm_v32x4_shuffle(vi6x4567, vzero, 1, 2, 3, 4);

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi0x5678, vk03));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi1x5678, vk03));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi2x5678, vk03));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi1x5678, vk13));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi2x5678, vk13));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi3x5678, vk13));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi2x5678, vk23));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi3x5678, vk23));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi4x5678, vk23));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi3x5678, vk33));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi4x5678, vk33));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi5x5678, vk33));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi4x5678, vk43));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi5x5678, vk43));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi6x5678, vk43));

      const v128_t vi0x6789 = wasm_v32x4_shuffle(vi0x4567, vzero, 2, 3, 4, 5);
      const v128_t vi1x6789 = wasm_v32x4_shuffle(vi1x4567, vzero, 2, 3, 4, 5);
      const v128_t vi2x6789 = wasm_v32x4_shuffle(vi2x4567, vzero, 2, 3, 4, 5);
      const v128_t vi3x6789 = wasm_v32x4_shuffle(vi3x4567, vzero, 2, 3, 4, 5);
      const v128_t vi4x6789 = wasm_v32x4_shuffle(vi4x4567, vzero, 2, 3, 4, 5);
      const v128_t vi5x6789 = wasm_v32x4_shuffle(vi5x4567, vzero, 2, 3, 4, 5);
      const v128_t vi6x6789 = wasm_v32x4_shuffle(vi6x4567, vzero, 2, 3, 4, 5);

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi0x6789, vk04));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi1x6789, vk04));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi2x6789, vk04));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi1x6789, vk14));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi2x6789, vk14));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi3x6789, vk14));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi2x6789, vk24));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi3x6789, vk24));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi4x6789, vk24));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi3x6789, vk34));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi4x6789, vk34));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi5x6789, vk34));

      vo4567p0 = wasm_f32x4_add(vo4567p0, wasm_f32x4_mul(vi4x6789, vk44));
      vo4567p1 = wasm_f32x4_add(vo4567p1, wasm_f32x4_mul(vi5x6789, vk44));
      vo4567p2 = wasm_f32x4_add(vo4567p2, wasm_f32x4_mul(vi6x6789, vk44));

      v128_t vo0 = vo4567p0;
      v128_t vo1 = vo4567p1;
      v128_t vo2 = vo4567p2;

      vo0 = wasm_f32x4_max(vo0, vmin);
      vo1 = wasm_f32x4_max(vo1, vmin);
      vo2 = wasm_f32x4_max(vo2, vmin);

      vo0 = wasm_f32x4_min(vo0, vmax);
      vo1 = wasm_f32x4_min(vo1, vmax);
      vo2 = wasm_f32x4_min(vo2, vmax);

      if XNN_LIKELY(w & (4 * sizeof(float))) {
        wasm_v128_store(o2, vo2);
        o2 += 4;
        wasm_v128_store(o1, vo1);
        o1 += 4;
        wasm_v128_store(o0, vo0);
        o0 += 4;
      } else {
        if (w & (2 * sizeof(float))) {
          *((double*) o2) = wasm_f64x2_extract_lane(vo2, 0);
          o2 += 2;
          *((double*) o1) = wasm_f64x2_extract_lane(vo1, 0);
          o1 += 2;
          *((double*) o0) = wasm_f64x2_extract_lane(vo0, 0);
          o0 += 2;

          vo0 = wasm_v32x4_shuffle(vo0, vo0, 2, 3, 0, 1);
          vo1 = wasm_v32x4_shuffle(vo1, vo1, 2, 3, 0, 1);
          vo2 = wasm_v32x4_shuffle(vo2, vo2, 2, 3, 0, 1);
        }
        if (w & (1 * sizeof(float))) {
          *o2 = wasm_f32x4_extract_lane(vo2, 0);
          o2 += 1;
          *o1 = wasm_f32x4_extract_lane(vo1, 0);
          o1 += 1;
          *o0 = wasm_f32x4_extract_lane(vo0, 0);
          o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i3 - input_decrement);
    i1 = (const float*) ((uintptr_t) i4 - input_decrement);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);
    i5 = (const float*) ((uintptr_t) i4 + input_width);
    i6 = (const float*) ((uintptr_t) i5 + input_width);

    o0 = o2;
    o1 = (float*) ((uintptr_t) o0 + input_width);
    o2 = (float*) ((uintptr_t) o1 + input_width);

    output_height = doz(output_height, 3);
  } while (output_height != 0);
}
