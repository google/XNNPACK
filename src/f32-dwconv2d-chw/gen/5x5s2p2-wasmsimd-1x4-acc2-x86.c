// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-1x4-acc2.c.in
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



void xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_1x4_acc2(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top >= 1);
  assert(padding_top <= 2);

  const v128_t vmask_even = wasm_v128_load(params->scalar.mask_even);
  const v128_t vmask_odd = wasm_v128_load(params->scalar.mask_odd);
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

  const uint32_t padding_top_less_1 = padding_top - 1;
  const size_t input_decrement = round_down_po2(input_width - 1 * sizeof(float), 4 * sizeof(float)) + 4 * sizeof(float);

  const float* i0 = zero;
  const float* i1 = (const float*) ((uintptr_t) input - ((-padding_top_less_1) & input_width));
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
  if XNN_UNPREDICTABLE(padding_top_less_1 != 0) {
    i1 = zero;
  }
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);

  size_t padded_input_height = input_height + (padding_top_less_1 + 1) + 2 /* padding bottom */;
  size_t output_height = (padded_input_height - 5 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height <= 6) {
      i4 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 6) {
      i3 = zero;
    }

    v128_t vi0x0123 = vzero;
    v128_t vi1x0123 = vzero;
    v128_t vi2x0123 = vzero;
    v128_t vi3x0123 = vzero;
    v128_t vi4x0123 = vzero;
    v128_t vi0x4567 = wasm_v128_load(i0);
    i0 += 4;
    v128_t vi1x4567 = wasm_v128_load(i1);
    i1 += 4;
    v128_t vi2x4567 = wasm_v128_load(i2);
    i2 += 4;
    v128_t vi3x4567 = wasm_v128_load(i3);
    i3 += 4;
    v128_t vi4x4567 = wasm_v128_load(i4);
    i4 += 4;

    size_t w = input_width;
    for (; w > 8 * sizeof(float); w -= 8 * sizeof(float)) {
      v128_t vo468Ap0 = vbias;

      const v128_t vi0x89AB = wasm_v128_load(i0);
      const v128_t vi1x89AB = wasm_v128_load(i1);
      const v128_t vi2x89AB = wasm_v128_load(i2);
      const v128_t vi3x89AB = wasm_v128_load(i3);
      const v128_t vi4x89AB = wasm_v128_load(i4);

      const v128_t vi0xCDEF = wasm_v128_load(i0 + 4);
      i0 += 8;
      const v128_t vi1xCDEF = wasm_v128_load(i1 + 4);
      i1 += 8;
      const v128_t vi2xCDEF = wasm_v128_load(i2 + 4);
      i2 += 8;
      const v128_t vi3xCDEF = wasm_v128_load(i3 + 4);
      i3 += 8;
      const v128_t vi4xCDEF = wasm_v128_load(i4 + 4);
      i4 += 8;

      const v128_t vi0x468A = wasm_v32x4_shuffle(vi0x4567, vi0x89AB, 0, 2, 4 + 0, 4 + 2);
      const v128_t vi0x579B = wasm_v32x4_shuffle(vi0x4567, vi0x89AB, 1, 3, 4 + 1, 4 + 3);
      const v128_t vi1x468A = wasm_v32x4_shuffle(vi1x4567, vi1x89AB, 0, 2, 4 + 0, 4 + 2);
      const v128_t vi1x579B = wasm_v32x4_shuffle(vi1x4567, vi1x89AB, 1, 3, 4 + 1, 4 + 3);
      const v128_t vi2x468A = wasm_v32x4_shuffle(vi2x4567, vi2x89AB, 0, 2, 4 + 0, 4 + 2);
      const v128_t vi2x579B = wasm_v32x4_shuffle(vi2x4567, vi2x89AB, 1, 3, 4 + 1, 4 + 3);
      const v128_t vi3x468A = wasm_v32x4_shuffle(vi3x4567, vi3x89AB, 0, 2, 4 + 0, 4 + 2);
      const v128_t vi3x579B = wasm_v32x4_shuffle(vi3x4567, vi3x89AB, 1, 3, 4 + 1, 4 + 3);
      const v128_t vi4x468A = wasm_v32x4_shuffle(vi4x4567, vi4x89AB, 0, 2, 4 + 0, 4 + 2);
      const v128_t vi4x579B = wasm_v32x4_shuffle(vi4x4567, vi4x89AB, 1, 3, 4 + 1, 4 + 3);

      // middle tap
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi0x468A, vk02));
      v128_t vo468Ap1 = wasm_f32x4_mul(vi1x468A, vk12);
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi2x468A, vk22));
      vo468Ap1 = wasm_f32x4_add(vo468Ap1, wasm_f32x4_mul(vi3x468A, vk32));
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi4x468A, vk42));

      // one left
      const v128_t vi0x3579 = wasm_v32x4_shuffle(vi0x0123, vi0x579B, 3, 4, 5, 6);
      const v128_t vi1x3579 = wasm_v32x4_shuffle(vi1x0123, vi1x579B, 3, 4, 5, 6);
      const v128_t vi2x3579 = wasm_v32x4_shuffle(vi2x0123, vi2x579B, 3, 4, 5, 6);
      const v128_t vi3x3579 = wasm_v32x4_shuffle(vi3x0123, vi3x579B, 3, 4, 5, 6);
      const v128_t vi4x3579 = wasm_v32x4_shuffle(vi4x0123, vi4x579B, 3, 4, 5, 6);

      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi0x3579, vk01));
      vo468Ap1 = wasm_f32x4_add(vo468Ap1, wasm_f32x4_mul(vi1x3579, vk11));
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi2x3579, vk21));
      vo468Ap1 = wasm_f32x4_add(vo468Ap1, wasm_f32x4_mul(vi3x3579, vk31));
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi4x3579, vk41));

      // two left
      const v128_t vi0x2468 = wasm_v32x4_shuffle(vi0x0123, vi0x468A, 2, 4, 5, 6);
      const v128_t vi1x2468 = wasm_v32x4_shuffle(vi1x0123, vi1x468A, 2, 4, 5, 6);
      const v128_t vi2x2468 = wasm_v32x4_shuffle(vi2x0123, vi2x468A, 2, 4, 5, 6);
      const v128_t vi3x2468 = wasm_v32x4_shuffle(vi3x0123, vi3x468A, 2, 4, 5, 6);
      const v128_t vi4x2468 = wasm_v32x4_shuffle(vi4x0123, vi4x468A, 2, 4, 5, 6);

      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi0x2468, vk00));
      vo468Ap1 = wasm_f32x4_add(vo468Ap1, wasm_f32x4_mul(vi1x2468, vk10));
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi2x2468, vk20));
      vo468Ap1 = wasm_f32x4_add(vo468Ap1, wasm_f32x4_mul(vi3x2468, vk30));
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi4x2468, vk40));

      vi0x0123 = vi0x89AB;
      vi1x0123 = vi1x89AB;
      vi2x0123 = vi2x89AB;
      vi3x0123 = vi3x89AB;
      vi4x0123 = vi4x89AB;

      // one right
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi0x579B, vk03));
      vo468Ap1 = wasm_f32x4_add(vo468Ap1, wasm_f32x4_mul(vi1x579B, vk13));
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi2x579B, vk23));
      vo468Ap1 = wasm_f32x4_add(vo468Ap1, wasm_f32x4_mul(vi3x579B, vk33));
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi4x579B, vk43));

      // two right
      const v128_t vi0x68AC = wasm_v32x4_shuffle(vi0x468A, vi0xCDEF, 1, 2, 3, 4);
      const v128_t vi1x68AC = wasm_v32x4_shuffle(vi1x468A, vi1xCDEF, 1, 2, 3, 4);
      const v128_t vi2x68AC = wasm_v32x4_shuffle(vi2x468A, vi2xCDEF, 1, 2, 3, 4);
      const v128_t vi3x68AC = wasm_v32x4_shuffle(vi3x468A, vi3xCDEF, 1, 2, 3, 4);
      const v128_t vi4x68AC = wasm_v32x4_shuffle(vi4x468A, vi4xCDEF, 1, 2, 3, 4);

      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi0x68AC, vk04));
      vo468Ap1 = wasm_f32x4_add(vo468Ap1, wasm_f32x4_mul(vi1x68AC, vk14));
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi2x68AC, vk24));
      vo468Ap1 = wasm_f32x4_add(vo468Ap1, wasm_f32x4_mul(vi3x68AC, vk34));
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi4x68AC, vk44));

      vi0x4567 = vi0xCDEF;
      vi1x4567 = vi1xCDEF;
      vi2x4567 = vi2xCDEF;
      vi3x4567 = vi3xCDEF;
      vi4x4567 = vi4xCDEF;

      v128_t vo0 = wasm_f32x4_add(vo468Ap0, vo468Ap1);

      vo0 = wasm_v128_bitselect(vmin, vo0, wasm_f32x4_lt(vo0, vmin));
      vo0 = wasm_v128_bitselect(vo0, vmax, wasm_f32x4_le(vo0, vmax));

      size_t w_tmp = (w + 1 * sizeof(float)) / (2 * sizeof(float));
      if XNN_LIKELY(w_tmp >= 4) {
        wasm_v128_store(output, vo0);
        output += 4;
      } else {
        if (w_tmp & 2) {
          *((double*) output) = wasm_f64x2_extract_lane(vo0, 0);
          output += 2;
          vo0 = wasm_v32x4_shuffle(vo0, vo0, 2, 3, 0, 1);
        }
        if (w_tmp & 1) {
          *output = wasm_f32x4_extract_lane(vo0, 0);
          output += 1;
        }
      }
    }

    {
      v128_t vo468Ap0 = vbias;

      v128_t vi0x89AB = vzero;
      v128_t vi1x89AB = vzero;
      v128_t vi2x89AB = vzero;
      v128_t vi3x89AB = vzero;
      v128_t vi4x89AB = vzero;
      if XNN_LIKELY(w > 4 * sizeof(float)) {
        vi0x89AB = wasm_v128_load(i0);
        i0 += 4;
        vi1x89AB = wasm_v128_load(i1);
        i1 += 4;
        vi2x89AB = wasm_v128_load(i2);
        i2 += 4;
        vi3x89AB = wasm_v128_load(i3);
        i3 += 4;
        vi4x89AB = wasm_v128_load(i4);
        i4 += 4;
      }

      v128_t vi0xCDEF = vzero;
      v128_t vi1xCDEF = vzero;
      v128_t vi2xCDEF = vzero;
      v128_t vi3xCDEF = vzero;
      v128_t vi4xCDEF = vzero;
      if XNN_LIKELY(w > 8 * sizeof(float)) {
        vi0xCDEF = wasm_v128_load(i0);
        i0 += 4;
        vi1xCDEF = wasm_v128_load(i1);
        i1 += 4;
        vi2xCDEF = wasm_v128_load(i2);
        i2 += 4;
        vi3xCDEF = wasm_v128_load(i3);
        i3 += 4;
        vi4xCDEF = wasm_v128_load(i4);
        i4 += 4;
      }

      v128_t vi0x468A = wasm_v32x4_shuffle(vi0x4567, vi0x89AB, 0, 2, 4 + 0, 4 + 2);
      v128_t vi0x579B = wasm_v32x4_shuffle(vi0x4567, vi0x89AB, 1, 3, 4 + 1, 4 + 3);
      v128_t vi1x468A = wasm_v32x4_shuffle(vi1x4567, vi1x89AB, 0, 2, 4 + 0, 4 + 2);
      v128_t vi1x579B = wasm_v32x4_shuffle(vi1x4567, vi1x89AB, 1, 3, 4 + 1, 4 + 3);
      v128_t vi2x468A = wasm_v32x4_shuffle(vi2x4567, vi2x89AB, 0, 2, 4 + 0, 4 + 2);
      v128_t vi2x579B = wasm_v32x4_shuffle(vi2x4567, vi2x89AB, 1, 3, 4 + 1, 4 + 3);
      v128_t vi3x468A = wasm_v32x4_shuffle(vi3x4567, vi3x89AB, 0, 2, 4 + 0, 4 + 2);
      v128_t vi3x579B = wasm_v32x4_shuffle(vi3x4567, vi3x89AB, 1, 3, 4 + 1, 4 + 3);
      v128_t vi4x468A = wasm_v32x4_shuffle(vi4x4567, vi4x89AB, 0, 2, 4 + 0, 4 + 2);
      v128_t vi4x579B = wasm_v32x4_shuffle(vi4x4567, vi4x89AB, 1, 3, 4 + 1, 4 + 3);

      vi0x468A = wasm_v128_and(vmask_even, vi0x468A);
      vi1x468A = wasm_v128_and(vmask_even, vi1x468A);
      vi2x468A = wasm_v128_and(vmask_even, vi2x468A);
      vi3x468A = wasm_v128_and(vmask_even, vi3x468A);
      vi4x468A = wasm_v128_and(vmask_even, vi4x468A);

      vi0x579B = wasm_v128_and(vmask_odd, vi0x579B);
      vi1x579B = wasm_v128_and(vmask_odd, vi1x579B);
      vi2x579B = wasm_v128_and(vmask_odd, vi2x579B);
      vi3x579B = wasm_v128_and(vmask_odd, vi3x579B);
      vi4x579B = wasm_v128_and(vmask_odd, vi4x579B);

      // middle tap
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi0x468A, vk02));
      v128_t vo468Ap1 = wasm_f32x4_mul(vi1x468A, vk12);
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi2x468A, vk22));
      vo468Ap1 = wasm_f32x4_add(vo468Ap1, wasm_f32x4_mul(vi3x468A, vk32));
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi4x468A, vk42));

      // one left
      const v128_t vi0x3579 = wasm_v32x4_shuffle(vi0x0123, vi0x579B, 3, 4, 5, 6);
      const v128_t vi1x3579 = wasm_v32x4_shuffle(vi1x0123, vi1x579B, 3, 4, 5, 6);
      const v128_t vi2x3579 = wasm_v32x4_shuffle(vi2x0123, vi2x579B, 3, 4, 5, 6);
      const v128_t vi3x3579 = wasm_v32x4_shuffle(vi3x0123, vi3x579B, 3, 4, 5, 6);
      const v128_t vi4x3579 = wasm_v32x4_shuffle(vi4x0123, vi4x579B, 3, 4, 5, 6);

      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi0x3579, vk01));
      vo468Ap1 = wasm_f32x4_add(vo468Ap1, wasm_f32x4_mul(vi1x3579, vk11));
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi2x3579, vk21));
      vo468Ap1 = wasm_f32x4_add(vo468Ap1, wasm_f32x4_mul(vi3x3579, vk31));
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi4x3579, vk41));

      // two left
      const v128_t vi0x2468 = wasm_v32x4_shuffle(vi0x0123, vi0x468A, 2, 4, 5, 6);
      const v128_t vi1x2468 = wasm_v32x4_shuffle(vi1x0123, vi1x468A, 2, 4, 5, 6);
      const v128_t vi2x2468 = wasm_v32x4_shuffle(vi2x0123, vi2x468A, 2, 4, 5, 6);
      const v128_t vi3x2468 = wasm_v32x4_shuffle(vi3x0123, vi3x468A, 2, 4, 5, 6);
      const v128_t vi4x2468 = wasm_v32x4_shuffle(vi4x0123, vi4x468A, 2, 4, 5, 6);

      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi0x2468, vk00));
      vo468Ap1 = wasm_f32x4_add(vo468Ap1, wasm_f32x4_mul(vi1x2468, vk10));
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi2x2468, vk20));
      vo468Ap1 = wasm_f32x4_add(vo468Ap1, wasm_f32x4_mul(vi3x2468, vk30));
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi4x2468, vk40));

      vi0x0123 = vi0x89AB;
      vi1x0123 = vi1x89AB;
      vi2x0123 = vi2x89AB;
      vi3x0123 = vi3x89AB;
      vi4x0123 = vi4x89AB;

      // one right
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi0x579B, vk03));
      vo468Ap1 = wasm_f32x4_add(vo468Ap1, wasm_f32x4_mul(vi1x579B, vk13));
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi2x579B, vk23));
      vo468Ap1 = wasm_f32x4_add(vo468Ap1, wasm_f32x4_mul(vi3x579B, vk33));
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi4x579B, vk43));

      // two right
      const v128_t vi0x68AC = wasm_v32x4_shuffle(vi0x468A, vi0xCDEF, 1, 2, 3, 4);
      const v128_t vi1x68AC = wasm_v32x4_shuffle(vi1x468A, vi1xCDEF, 1, 2, 3, 4);
      const v128_t vi2x68AC = wasm_v32x4_shuffle(vi2x468A, vi2xCDEF, 1, 2, 3, 4);
      const v128_t vi3x68AC = wasm_v32x4_shuffle(vi3x468A, vi3xCDEF, 1, 2, 3, 4);
      const v128_t vi4x68AC = wasm_v32x4_shuffle(vi4x468A, vi4xCDEF, 1, 2, 3, 4);

      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi0x68AC, vk04));
      vo468Ap1 = wasm_f32x4_add(vo468Ap1, wasm_f32x4_mul(vi1x68AC, vk14));
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi2x68AC, vk24));
      vo468Ap1 = wasm_f32x4_add(vo468Ap1, wasm_f32x4_mul(vi3x68AC, vk34));
      vo468Ap0 = wasm_f32x4_add(vo468Ap0, wasm_f32x4_mul(vi4x68AC, vk44));

      vi0x4567 = vi0xCDEF;
      vi1x4567 = vi1xCDEF;
      vi2x4567 = vi2xCDEF;
      vi3x4567 = vi3xCDEF;
      vi4x4567 = vi4xCDEF;

      v128_t vo0 = wasm_f32x4_add(vo468Ap0, vo468Ap1);

      vo0 = wasm_v128_bitselect(vmin, vo0, wasm_f32x4_lt(vo0, vmin));
      vo0 = wasm_v128_bitselect(vo0, vmax, wasm_f32x4_le(vo0, vmax));

      size_t w_tmp = (w + 1 * sizeof(float)) / (2 * sizeof(float));
      if XNN_LIKELY(w_tmp >= 4) {
        wasm_v128_store(output, vo0);
        output += 4;
      } else {
        if (w_tmp & 2) {
          *((double*) output) = wasm_f64x2_extract_lane(vo0, 0);
          output += 2;
          vo0 = wasm_v32x4_shuffle(vo0, vo0, 2, 3, 0, 1);
        }
        if (w_tmp & 1) {
          *output = wasm_f32x4_extract_lane(vo0, 0);
          output += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i2 - input_decrement);
    i1 = (const float*) ((uintptr_t) i3 - input_decrement);
    i2 = (const float*) ((uintptr_t) i4 - input_decrement);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);

    output_height -= 1;
    padded_input_height -= 2;
  } while (output_height != 0);
}
