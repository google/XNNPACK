// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"



void xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_1x4_acc4(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top == 2);

  const v128_t vmax = wasm_v128_load32_splat(&params->scalar.max);
  const v128_t vmin = wasm_v128_load32_splat(&params->scalar.min);
  XNN_FORCE_REALIZATION(vmax);
  XNN_FORCE_REALIZATION(vmin);

  static const int32_t mask_table[7] = {-1, -1, -1, -1, 0, 0, 0};
  const v128_t vmask = wasm_v128_load(&mask_table[3 - (((input_width >> 2) - 1) & 3)]);

  const v128_t vw0123 = wasm_v128_load(weights);
  const v128_t vw4567 = wasm_v128_load(weights + 4);
  const v128_t vw89AB = wasm_v128_load(weights + 8);
  const v128_t vwCDEF = wasm_v128_load(weights + 12);
  const v128_t vwGHIJ = wasm_v128_load(weights + 16);
  const v128_t vwKLMN = wasm_v128_load(weights + 20);
  const v128_t vwOP = wasm_v128_load64_splat(weights + 24);
  const v128_t vbias = wasm_v32x4_shuffle(vw0123, vw0123, 0, 0, 0, 0);
  const v128_t vk00 =  wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1);
  const v128_t vk01 =  wasm_v32x4_shuffle(vw0123, vw0123, 2, 2, 2, 2);
  const v128_t vk02 =  wasm_v32x4_shuffle(vw0123, vw0123, 3, 3, 3, 3);
  const v128_t vk03 =  wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0);
  const v128_t vk04 =  wasm_v32x4_shuffle(vw4567, vw4567, 1, 1, 1, 1);
  const v128_t vk10 =  wasm_v32x4_shuffle(vw4567, vw4567, 2, 2, 2, 2);
  const v128_t vk11 =  wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3);
  const v128_t vk12 =  wasm_v32x4_shuffle(vw89AB, vw89AB, 0, 0, 0, 0);
  const v128_t vk13 =  wasm_v32x4_shuffle(vw89AB, vw89AB, 1, 1, 1, 1);
  const v128_t vk14 =  wasm_v32x4_shuffle(vw89AB, vw89AB, 2, 2, 2, 2);
  const v128_t vk20 =  wasm_v32x4_shuffle(vw89AB, vw89AB, 3, 3, 3, 3);
  const v128_t vk21 =  wasm_v32x4_shuffle(vwCDEF, vwCDEF, 0, 0, 0, 0);
  const v128_t vk22 =  wasm_v32x4_shuffle(vwCDEF, vwCDEF, 1, 1, 1, 1);
  const v128_t vk23 =  wasm_v32x4_shuffle(vwCDEF, vwCDEF, 2, 2, 2, 2);
  const v128_t vk24 =  wasm_v32x4_shuffle(vwCDEF, vwCDEF, 3, 3, 3, 3);
  const v128_t vk30 =  wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 0, 0, 0, 0);
  const v128_t vk31 =  wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 1, 1, 1, 1);
  const v128_t vk32 =  wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 2, 2, 2, 2);
  const v128_t vk33 =  wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 3, 3, 3, 3);
  const v128_t vk34 =  wasm_v32x4_shuffle(vwKLMN, vwKLMN, 0, 0, 0, 0);
  const v128_t vk40 =  wasm_v32x4_shuffle(vwKLMN, vwKLMN, 1, 1, 1, 1);
  const v128_t vk41 =  wasm_v32x4_shuffle(vwKLMN, vwKLMN, 2, 2, 2, 2);
  const v128_t vk42 =  wasm_v32x4_shuffle(vwKLMN, vwKLMN, 3, 3, 3, 3);
  const v128_t vk43 =  wasm_v32x4_shuffle(vwOP,   vwOP,   0, 0, 0, 0);
  const v128_t vk44 =  wasm_v32x4_shuffle(vwOP,   vwOP,   1, 1, 1, 1);

  const size_t input_decrement = round_up_po2(input_width, 4 * sizeof(float));

  const float* i0 = zero;
  const float* i1 = zero;
  const float* i2 = input;
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);

  float* o0 = output;

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i3 = zero;
    }
    if XNN_UNPREDICTABLE(output_height < 3) {
      i4 = zero;
    }

    v128_t vi0x0123 = wasm_f32x4_const_splat(0.0f);
    v128_t vi1x0123 = wasm_f32x4_const_splat(0.0f);
    v128_t vi2x0123 = wasm_f32x4_const_splat(0.0f);
    v128_t vi3x0123 = wasm_f32x4_const_splat(0.0f);
    v128_t vi4x0123 = wasm_f32x4_const_splat(0.0f);

    v128_t vi0x4567 = wasm_v128_load(i0); i0 += 4;
    v128_t vi1x4567 = wasm_v128_load(i1); i1 += 4;
    v128_t vi2x4567 = wasm_v128_load(i2); i2 += 4;
    v128_t vi3x4567 = wasm_v128_load(i3); i3 += 4;
    v128_t vi4x4567 = wasm_v128_load(i4); i4 += 4;

    size_t w = input_width;
    for (; w > 8 * sizeof(float); w -= 4 * sizeof(float)) {
      v128_t vo0p0 = vbias;

      const v128_t vi0x89AB = wasm_v128_load(i0); i0 += 4;
      const v128_t vi1x89AB = wasm_v128_load(i1); i1 += 4;
      const v128_t vi2x89AB = wasm_v128_load(i2); i2 += 4;
      const v128_t vi3x89AB = wasm_v128_load(i3); i3 += 4;
      const v128_t vi4x89AB = wasm_v128_load(i4); i4 += 4;

      v128_t vo0p1 = wasm_f32x4_mul(vi0x4567, vk02);

      v128_t vo0p2 = wasm_f32x4_mul(vi1x4567, vk12);

      v128_t vo0p3 = wasm_f32x4_mul(vi2x4567, vk22);

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi3x4567, vk32));

      vo0p2 = wasm_f32x4_add(vo0p2, wasm_f32x4_mul(vi4x4567, vk42));

      const v128_t vi0x3456 = wasm_v32x4_shuffle(vi0x0123, vi0x4567, 3, 4, 5, 6);
      const v128_t vi1x3456 = wasm_v32x4_shuffle(vi1x0123, vi1x4567, 3, 4, 5, 6);
      const v128_t vi2x3456 = wasm_v32x4_shuffle(vi2x0123, vi2x4567, 3, 4, 5, 6);
      const v128_t vi3x3456 = wasm_v32x4_shuffle(vi3x0123, vi3x4567, 3, 4, 5, 6);
      const v128_t vi4x3456 = wasm_v32x4_shuffle(vi4x0123, vi4x4567, 3, 4, 5, 6);

      vo0p3 = wasm_f32x4_add(vo0p3, wasm_f32x4_mul(vi0x3456, vk01));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi1x3456, vk11));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi2x3456, vk21));

      vo0p2 = wasm_f32x4_add(vo0p2, wasm_f32x4_mul(vi3x3456, vk31));

      vo0p3 = wasm_f32x4_add(vo0p3, wasm_f32x4_mul(vi4x3456, vk41));

      const v128_t vi0x2345 = wasm_v32x4_shuffle(vi0x0123, vi0x4567, 2, 3, 4, 5);
      vi0x0123 = vi0x4567;
      const v128_t vi1x2345 = wasm_v32x4_shuffle(vi1x0123, vi1x4567, 2, 3, 4, 5);
      vi1x0123 = vi1x4567;
      const v128_t vi2x2345 = wasm_v32x4_shuffle(vi2x0123, vi2x4567, 2, 3, 4, 5);
      vi2x0123 = vi2x4567;
      const v128_t vi3x2345 = wasm_v32x4_shuffle(vi3x0123, vi3x4567, 2, 3, 4, 5);
      vi3x0123 = vi3x4567;
      const v128_t vi4x2345 = wasm_v32x4_shuffle(vi4x0123, vi4x4567, 2, 3, 4, 5);
      vi4x0123 = vi4x4567;

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi0x2345, vk00));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi1x2345, vk10));

      vo0p2 = wasm_f32x4_add(vo0p2, wasm_f32x4_mul(vi2x2345, vk20));

      vo0p3 = wasm_f32x4_add(vo0p3, wasm_f32x4_mul(vi3x2345, vk30));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi4x2345, vk40));

      const v128_t vi0x5678 = wasm_v32x4_shuffle(vi0x4567, vi0x89AB, 1, 2, 3, 4);
      const v128_t vi1x5678 = wasm_v32x4_shuffle(vi1x4567, vi1x89AB, 1, 2, 3, 4);
      const v128_t vi2x5678 = wasm_v32x4_shuffle(vi2x4567, vi2x89AB, 1, 2, 3, 4);
      const v128_t vi3x5678 = wasm_v32x4_shuffle(vi3x4567, vi3x89AB, 1, 2, 3, 4);
      const v128_t vi4x5678 = wasm_v32x4_shuffle(vi4x4567, vi4x89AB, 1, 2, 3, 4);

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi0x5678, vk03));

      vo0p2 = wasm_f32x4_add(vo0p2, wasm_f32x4_mul(vi1x5678, vk13));

      vo0p3 = wasm_f32x4_add(vo0p3, wasm_f32x4_mul(vi2x5678, vk23));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi3x5678, vk33));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi4x5678, vk43));

      const v128_t vi0x6789 = wasm_v32x4_shuffle(vi0x4567, vi0x89AB, 2, 3, 4, 5);
      vi0x4567 = vi0x89AB;
      const v128_t vi1x6789 = wasm_v32x4_shuffle(vi1x4567, vi1x89AB, 2, 3, 4, 5);
      vi1x4567 = vi1x89AB;
      const v128_t vi2x6789 = wasm_v32x4_shuffle(vi2x4567, vi2x89AB, 2, 3, 4, 5);
      vi2x4567 = vi2x89AB;
      const v128_t vi3x6789 = wasm_v32x4_shuffle(vi3x4567, vi3x89AB, 2, 3, 4, 5);
      vi3x4567 = vi3x89AB;
      const v128_t vi4x6789 = wasm_v32x4_shuffle(vi4x4567, vi4x89AB, 2, 3, 4, 5);
      vi4x4567 = vi4x89AB;

      vo0p2 = wasm_f32x4_add(vo0p2, wasm_f32x4_mul(vi0x6789, vk04));

      vo0p3 = wasm_f32x4_add(vo0p3, wasm_f32x4_mul(vi1x6789, vk14));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi2x6789, vk24));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi3x6789, vk34));

      vo0p2 = wasm_f32x4_add(vo0p2, wasm_f32x4_mul(vi4x6789, vk44));

      vo0p0 = wasm_f32x4_add(vo0p0, vo0p1);
      vo0p2 = wasm_f32x4_add(vo0p2, vo0p3);
      vo0p0 = wasm_f32x4_add(vo0p0, vo0p2);

      v128_t vo0 = wasm_f32x4_pmax(vmin, vo0p0);
      vo0 = wasm_f32x4_pmin(vmax, vo0);

      wasm_v128_store(o0, vo0); o0 += 4;
    }
    // Always process the last block of 5..8 pixels.
    if XNN_LIKELY(w > 4 * sizeof(float)) {
      v128_t vo0p0 = vbias;

      v128_t vi0x89AB = wasm_v128_load(i0); i0 += 4;
      v128_t vi1x89AB = wasm_v128_load(i1); i1 += 4;
      v128_t vi2x89AB = wasm_v128_load(i2); i2 += 4;
      v128_t vi3x89AB = wasm_v128_load(i3); i3 += 4;
      v128_t vi4x89AB = wasm_v128_load(i4); i4 += 4;

      vi0x89AB = wasm_v128_and(vmask, vi0x89AB);
      vi1x89AB = wasm_v128_and(vmask, vi1x89AB);
      vi2x89AB = wasm_v128_and(vmask, vi2x89AB);
      vi3x89AB = wasm_v128_and(vmask, vi3x89AB);
      vi4x89AB = wasm_v128_and(vmask, vi4x89AB);

      v128_t vo0p1 = wasm_f32x4_mul(vi0x4567, vk02);

      v128_t vo0p2 = wasm_f32x4_mul(vi1x4567, vk12);

      v128_t vo0p3 = wasm_f32x4_mul(vi2x4567, vk22);

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi3x4567, vk32));

      vo0p2 = wasm_f32x4_add(vo0p2, wasm_f32x4_mul(vi4x4567, vk42));

      const v128_t vi0x3456 = wasm_v32x4_shuffle(vi0x0123, vi0x4567, 3, 4, 5, 6);
      const v128_t vi1x3456 = wasm_v32x4_shuffle(vi1x0123, vi1x4567, 3, 4, 5, 6);
      const v128_t vi2x3456 = wasm_v32x4_shuffle(vi2x0123, vi2x4567, 3, 4, 5, 6);
      const v128_t vi3x3456 = wasm_v32x4_shuffle(vi3x0123, vi3x4567, 3, 4, 5, 6);
      const v128_t vi4x3456 = wasm_v32x4_shuffle(vi4x0123, vi4x4567, 3, 4, 5, 6);

      vo0p3 = wasm_f32x4_add(vo0p3, wasm_f32x4_mul(vi0x3456, vk01));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi1x3456, vk11));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi2x3456, vk21));

      vo0p2 = wasm_f32x4_add(vo0p2, wasm_f32x4_mul(vi3x3456, vk31));

      vo0p3 = wasm_f32x4_add(vo0p3, wasm_f32x4_mul(vi4x3456, vk41));

      const v128_t vi0x2345 = wasm_v32x4_shuffle(vi0x0123, vi0x4567, 2, 3, 4, 5);
      vi0x0123 = vi0x4567;
      const v128_t vi1x2345 = wasm_v32x4_shuffle(vi1x0123, vi1x4567, 2, 3, 4, 5);
      vi1x0123 = vi1x4567;
      const v128_t vi2x2345 = wasm_v32x4_shuffle(vi2x0123, vi2x4567, 2, 3, 4, 5);
      vi2x0123 = vi2x4567;
      const v128_t vi3x2345 = wasm_v32x4_shuffle(vi3x0123, vi3x4567, 2, 3, 4, 5);
      vi3x0123 = vi3x4567;
      const v128_t vi4x2345 = wasm_v32x4_shuffle(vi4x0123, vi4x4567, 2, 3, 4, 5);
      vi4x0123 = vi4x4567;

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi0x2345, vk00));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi1x2345, vk10));

      vo0p2 = wasm_f32x4_add(vo0p2, wasm_f32x4_mul(vi2x2345, vk20));

      vo0p3 = wasm_f32x4_add(vo0p3, wasm_f32x4_mul(vi3x2345, vk30));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi4x2345, vk40));

      const v128_t vi0x5678 = wasm_v32x4_shuffle(vi0x4567, vi0x89AB, 1, 2, 3, 4);
      const v128_t vi1x5678 = wasm_v32x4_shuffle(vi1x4567, vi1x89AB, 1, 2, 3, 4);
      const v128_t vi2x5678 = wasm_v32x4_shuffle(vi2x4567, vi2x89AB, 1, 2, 3, 4);
      const v128_t vi3x5678 = wasm_v32x4_shuffle(vi3x4567, vi3x89AB, 1, 2, 3, 4);
      const v128_t vi4x5678 = wasm_v32x4_shuffle(vi4x4567, vi4x89AB, 1, 2, 3, 4);

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi0x5678, vk03));

      vo0p2 = wasm_f32x4_add(vo0p2, wasm_f32x4_mul(vi1x5678, vk13));

      vo0p3 = wasm_f32x4_add(vo0p3, wasm_f32x4_mul(vi2x5678, vk23));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi3x5678, vk33));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi4x5678, vk43));

      const v128_t vi0x6789 = wasm_v32x4_shuffle(vi0x4567, vi0x89AB, 2, 3, 4, 5);
      vi0x4567 = vi0x89AB;
      const v128_t vi1x6789 = wasm_v32x4_shuffle(vi1x4567, vi1x89AB, 2, 3, 4, 5);
      vi1x4567 = vi1x89AB;
      const v128_t vi2x6789 = wasm_v32x4_shuffle(vi2x4567, vi2x89AB, 2, 3, 4, 5);
      vi2x4567 = vi2x89AB;
      const v128_t vi3x6789 = wasm_v32x4_shuffle(vi3x4567, vi3x89AB, 2, 3, 4, 5);
      vi3x4567 = vi3x89AB;
      const v128_t vi4x6789 = wasm_v32x4_shuffle(vi4x4567, vi4x89AB, 2, 3, 4, 5);
      vi4x4567 = vi4x89AB;

      vo0p2 = wasm_f32x4_add(vo0p2, wasm_f32x4_mul(vi0x6789, vk04));

      vo0p3 = wasm_f32x4_add(vo0p3, wasm_f32x4_mul(vi1x6789, vk14));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi2x6789, vk24));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi3x6789, vk34));

      vo0p2 = wasm_f32x4_add(vo0p2, wasm_f32x4_mul(vi4x6789, vk44));

      vo0p0 = wasm_f32x4_add(vo0p0, vo0p1);
      vo0p2 = wasm_f32x4_add(vo0p2, vo0p3);
      vo0p0 = wasm_f32x4_add(vo0p0, vo0p2);

      v128_t vo0 = wasm_f32x4_pmax(vmin, vo0p0);
      vo0 = wasm_f32x4_pmin(vmax, vo0);

      wasm_v128_store(o0, vo0); o0 += 4;

      w -= 4 * sizeof(float);
    }
    assert(w >= 1 * sizeof(float));
    assert(w <= 4 * sizeof(float));
    {
      v128_t vo0p0 = vbias;

      vi0x4567 = wasm_v128_and(vmask, vi0x4567);
      vi1x4567 = wasm_v128_and(vmask, vi1x4567);
      vi2x4567 = wasm_v128_and(vmask, vi2x4567);
      vi3x4567 = wasm_v128_and(vmask, vi3x4567);
      vi4x4567 = wasm_v128_and(vmask, vi4x4567);

      v128_t vo0p1 = wasm_f32x4_mul(vi0x4567, vk02);

      v128_t vo0p2 = wasm_f32x4_mul(vi1x4567, vk12);

      v128_t vo0p3 = wasm_f32x4_mul(vi2x4567, vk22);

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi3x4567, vk32));

      vo0p2 = wasm_f32x4_add(vo0p2, wasm_f32x4_mul(vi4x4567, vk42));

      const v128_t vi0x3456 = wasm_v32x4_shuffle(vi0x0123, vi0x4567, 3, 4, 5, 6);
      const v128_t vi1x3456 = wasm_v32x4_shuffle(vi1x0123, vi1x4567, 3, 4, 5, 6);
      const v128_t vi2x3456 = wasm_v32x4_shuffle(vi2x0123, vi2x4567, 3, 4, 5, 6);
      const v128_t vi3x3456 = wasm_v32x4_shuffle(vi3x0123, vi3x4567, 3, 4, 5, 6);
      const v128_t vi4x3456 = wasm_v32x4_shuffle(vi4x0123, vi4x4567, 3, 4, 5, 6);

      vo0p3 = wasm_f32x4_add(vo0p3, wasm_f32x4_mul(vi0x3456, vk01));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi1x3456, vk11));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi2x3456, vk21));

      vo0p2 = wasm_f32x4_add(vo0p2, wasm_f32x4_mul(vi3x3456, vk31));

      vo0p3 = wasm_f32x4_add(vo0p3, wasm_f32x4_mul(vi4x3456, vk41));

      const v128_t vi0x2345 = wasm_v32x4_shuffle(vi0x0123, vi0x4567, 2, 3, 4, 5);
      const v128_t vi1x2345 = wasm_v32x4_shuffle(vi1x0123, vi1x4567, 2, 3, 4, 5);
      const v128_t vi2x2345 = wasm_v32x4_shuffle(vi2x0123, vi2x4567, 2, 3, 4, 5);
      const v128_t vi3x2345 = wasm_v32x4_shuffle(vi3x0123, vi3x4567, 2, 3, 4, 5);
      const v128_t vi4x2345 = wasm_v32x4_shuffle(vi4x0123, vi4x4567, 2, 3, 4, 5);

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi0x2345, vk00));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi1x2345, vk10));

      vo0p2 = wasm_f32x4_add(vo0p2, wasm_f32x4_mul(vi2x2345, vk20));

      vo0p3 = wasm_f32x4_add(vo0p3, wasm_f32x4_mul(vi3x2345, vk30));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi4x2345, vk40));

      const v128_t vzero = wasm_f32x4_const_splat(0.0f);
      const v128_t vi0x5678 = wasm_v32x4_shuffle(vi0x4567, vzero, 1, 2, 3, 4);
      const v128_t vi1x5678 = wasm_v32x4_shuffle(vi1x4567, vzero, 1, 2, 3, 4);
      const v128_t vi2x5678 = wasm_v32x4_shuffle(vi2x4567, vzero, 1, 2, 3, 4);
      const v128_t vi3x5678 = wasm_v32x4_shuffle(vi3x4567, vzero, 1, 2, 3, 4);
      const v128_t vi4x5678 = wasm_v32x4_shuffle(vi4x4567, vzero, 1, 2, 3, 4);

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi0x5678, vk03));

      vo0p2 = wasm_f32x4_add(vo0p2, wasm_f32x4_mul(vi1x5678, vk13));

      vo0p3 = wasm_f32x4_add(vo0p3, wasm_f32x4_mul(vi2x5678, vk23));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi3x5678, vk33));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi4x5678, vk43));

      const v128_t vi0x6789 = wasm_v32x4_shuffle(vi0x5678, vzero, 1, 2, 3, 4);
      const v128_t vi1x6789 = wasm_v32x4_shuffle(vi1x5678, vzero, 1, 2, 3, 4);
      const v128_t vi2x6789 = wasm_v32x4_shuffle(vi2x5678, vzero, 1, 2, 3, 4);
      const v128_t vi3x6789 = wasm_v32x4_shuffle(vi3x5678, vzero, 1, 2, 3, 4);
      const v128_t vi4x6789 = wasm_v32x4_shuffle(vi4x5678, vzero, 1, 2, 3, 4);

      vo0p2 = wasm_f32x4_add(vo0p2, wasm_f32x4_mul(vi0x6789, vk04));

      vo0p3 = wasm_f32x4_add(vo0p3, wasm_f32x4_mul(vi1x6789, vk14));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi2x6789, vk24));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi3x6789, vk34));

      vo0p2 = wasm_f32x4_add(vo0p2, wasm_f32x4_mul(vi4x6789, vk44));

      vo0p0 = wasm_f32x4_add(vo0p0, vo0p1);
      vo0p2 = wasm_f32x4_add(vo0p2, vo0p3);
      vo0p0 = wasm_f32x4_add(vo0p0, vo0p2);

      v128_t vo0 = wasm_f32x4_pmax(vmin, vo0p0);
      vo0 = wasm_f32x4_pmin(vmax, vo0);

      if XNN_LIKELY(w & (4 * sizeof(float))) {
        wasm_v128_store(o0, vo0); o0 += 4;
      } else {
        if (w & (2 * sizeof(float))) {
          wasm_v128_store64_lane(o0, vo0, 0);
          o0 += 2;

          vo0 = wasm_v64x2_shuffle(vo0, vo0, 1, 1);
        }
        if (w & (1 * sizeof(float))) {
          wasm_v128_store32_lane(o0, vo0, 0);
          o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i1 - input_decrement);
    i1 = (const float*) ((uintptr_t) i2 - input_decrement);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);


  } while (--output_height != 0);
}
