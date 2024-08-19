// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-splat.c.in
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



void xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_3x4_acc2(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top >= 1);
  assert(padding_top <= 2);

  const v128_t vmask_even = wasm_v128_load(params->wasmsimd_stride2.mask_even);
  const v128_t vmask_odd  = wasm_v128_load(params->wasmsimd_stride2.mask_odd);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd_stride2.max);
  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd_stride2.min);
  XNN_FORCE_REALIZATION(vmax);
  XNN_FORCE_REALIZATION(vmin);

  const v128_t vw0123 = wasm_v128_load(weights);
  const v128_t vw4567 = wasm_v128_load(weights + 4);
  const v128_t vw89AB = wasm_v128_load(weights + 8);
  const v128_t vwCDEF = wasm_v128_load(weights + 12);
  const v128_t vwGHIJ = wasm_v128_load(weights + 16);
  const v128_t vwKLMN = wasm_v128_load(weights + 20);
  const v128_t vwOP = wasm_v128_load64_splat(weights + 24);

  const uint32_t padding_top_less_1 = padding_top - 1;
  const size_t input_decrement = round_up_po2(input_width, 8 * sizeof(float));

  const float* i0 = zero;
  const float* i1 = (const float*) ((uintptr_t) input - ((-padding_top_less_1) & input_width));
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
  if XNN_UNPREDICTABLE(padding_top_less_1 != 0) {
    i1 = zero;
  }
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_width);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_width);
  const float* i7 = (const float*) ((uintptr_t) i6 + input_width);
  const float* i8 = (const float*) ((uintptr_t) i7 + input_width);

  const size_t output_width = round_down_po2((input_width + (2 /* padding */ - 3 /* kernel size */ + 2 /* subsampling */) * sizeof(float)) / 2, sizeof(float));

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + output_width);
  float* o2 = (float*) ((uintptr_t) o1 + output_width);

  size_t padded_input_height = input_height + (padding_top_less_1 + 1) + 2 /* padding bottom */;
  size_t output_height = (padded_input_height - 5 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 6) {
      i3 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 7) {
      i4 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 8) {
      i5 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 9) {
      i6 = zero;
      o2 = o1;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 10) {
      i7 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 11) {
      i8 = zero;
    }

    v128_t vi0x0246 = wasm_f32x4_const_splat(0.0f);
    v128_t vi1x0246 = wasm_f32x4_const_splat(0.0f);
    v128_t vi2x0246 = wasm_f32x4_const_splat(0.0f);
    v128_t vi3x0246 = wasm_f32x4_const_splat(0.0f);
    v128_t vi4x0246 = wasm_f32x4_const_splat(0.0f);
    v128_t vi5x0246 = wasm_f32x4_const_splat(0.0f);
    v128_t vi6x0246 = wasm_f32x4_const_splat(0.0f);
    v128_t vi7x0246 = wasm_f32x4_const_splat(0.0f);
    v128_t vi8x0246 = wasm_f32x4_const_splat(0.0f);

    v128_t vi0x1357 = wasm_f32x4_const_splat(0.0f);
    v128_t vi1x1357 = wasm_f32x4_const_splat(0.0f);
    v128_t vi2x1357 = wasm_f32x4_const_splat(0.0f);
    v128_t vi3x1357 = wasm_f32x4_const_splat(0.0f);
    v128_t vi4x1357 = wasm_f32x4_const_splat(0.0f);
    v128_t vi5x1357 = wasm_f32x4_const_splat(0.0f);
    v128_t vi6x1357 = wasm_f32x4_const_splat(0.0f);
    v128_t vi7x1357 = wasm_f32x4_const_splat(0.0f);
    v128_t vi8x1357 = wasm_f32x4_const_splat(0.0f);

    const v128_t vi0x89AB = wasm_v128_load(i0);
    const v128_t vi0xCDEF = wasm_v128_load(i0 + 4);
    i0 += 8;
    const v128_t vi1x89AB = wasm_v128_load(i1);
    const v128_t vi1xCDEF = wasm_v128_load(i1 + 4);
    i1 += 8;
    const v128_t vi2x89AB = wasm_v128_load(i2);
    const v128_t vi2xCDEF = wasm_v128_load(i2 + 4);
    i2 += 8;
    const v128_t vi3x89AB = wasm_v128_load(i3);
    const v128_t vi3xCDEF = wasm_v128_load(i3 + 4);
    i3 += 8;
    const v128_t vi4x89AB = wasm_v128_load(i4);
    const v128_t vi4xCDEF = wasm_v128_load(i4 + 4);
    i4 += 8;
    const v128_t vi5x89AB = wasm_v128_load(i5);
    const v128_t vi5xCDEF = wasm_v128_load(i5 + 4);
    i5 += 8;
    const v128_t vi6x89AB = wasm_v128_load(i6);
    const v128_t vi6xCDEF = wasm_v128_load(i6 + 4);
    i6 += 8;
    const v128_t vi7x89AB = wasm_v128_load(i7);
    const v128_t vi7xCDEF = wasm_v128_load(i7 + 4);
    i7 += 8;
    const v128_t vi8x89AB = wasm_v128_load(i8);
    const v128_t vi8xCDEF = wasm_v128_load(i8 + 4);
    i8 += 8;

    v128_t vi0x8ACE = wasm_v32x4_shuffle(vi0x89AB, vi0xCDEF, 0, 2, 4, 6);
    v128_t vi0x9BDF = wasm_v32x4_shuffle(vi0x89AB, vi0xCDEF, 1, 3, 5, 7);
    v128_t vi1x8ACE = wasm_v32x4_shuffle(vi1x89AB, vi1xCDEF, 0, 2, 4, 6);
    v128_t vi1x9BDF = wasm_v32x4_shuffle(vi1x89AB, vi1xCDEF, 1, 3, 5, 7);
    v128_t vi2x8ACE = wasm_v32x4_shuffle(vi2x89AB, vi2xCDEF, 0, 2, 4, 6);
    v128_t vi2x9BDF = wasm_v32x4_shuffle(vi2x89AB, vi2xCDEF, 1, 3, 5, 7);
    v128_t vi3x8ACE = wasm_v32x4_shuffle(vi3x89AB, vi3xCDEF, 0, 2, 4, 6);
    v128_t vi3x9BDF = wasm_v32x4_shuffle(vi3x89AB, vi3xCDEF, 1, 3, 5, 7);
    v128_t vi4x8ACE = wasm_v32x4_shuffle(vi4x89AB, vi4xCDEF, 0, 2, 4, 6);
    v128_t vi4x9BDF = wasm_v32x4_shuffle(vi4x89AB, vi4xCDEF, 1, 3, 5, 7);
    v128_t vi5x8ACE = wasm_v32x4_shuffle(vi5x89AB, vi5xCDEF, 0, 2, 4, 6);
    v128_t vi5x9BDF = wasm_v32x4_shuffle(vi5x89AB, vi5xCDEF, 1, 3, 5, 7);
    v128_t vi6x8ACE = wasm_v32x4_shuffle(vi6x89AB, vi6xCDEF, 0, 2, 4, 6);
    v128_t vi6x9BDF = wasm_v32x4_shuffle(vi6x89AB, vi6xCDEF, 1, 3, 5, 7);
    v128_t vi7x8ACE = wasm_v32x4_shuffle(vi7x89AB, vi7xCDEF, 0, 2, 4, 6);
    v128_t vi7x9BDF = wasm_v32x4_shuffle(vi7x89AB, vi7xCDEF, 1, 3, 5, 7);
    v128_t vi8x8ACE = wasm_v32x4_shuffle(vi8x89AB, vi8xCDEF, 0, 2, 4, 6);
    v128_t vi8x9BDF = wasm_v32x4_shuffle(vi8x89AB, vi8xCDEF, 1, 3, 5, 7);

    size_t w = input_width;
    for (; w > 8 * sizeof(float); w -= 8 * sizeof(float)) {
      v128_t vo0p0 = wasm_v32x4_shuffle(vw0123, vw0123, 0, 0, 0, 0);
      v128_t vo1p0 = wasm_v32x4_shuffle(vw0123, vw0123, 0, 0, 0, 0);
      v128_t vo2p0 = wasm_v32x4_shuffle(vw0123, vw0123, 0, 0, 0, 0);

      v128_t vo0p1 = wasm_f32x4_mul(vi0x8ACE, wasm_v32x4_shuffle(vw0123, vw0123, 3, 3, 3, 3));
      v128_t vo1p1 = wasm_f32x4_mul(vi2x8ACE, wasm_v32x4_shuffle(vw0123, vw0123, 3, 3, 3, 3));
      v128_t vo2p1 = wasm_f32x4_mul(vi4x8ACE, wasm_v32x4_shuffle(vw0123, vw0123, 3, 3, 3, 3));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi1x8ACE, wasm_v32x4_shuffle(vw89AB, vw89AB, 0, 0, 0, 0)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi3x8ACE, wasm_v32x4_shuffle(vw89AB, vw89AB, 0, 0, 0, 0)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi5x8ACE, wasm_v32x4_shuffle(vw89AB, vw89AB, 0, 0, 0, 0)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi2x8ACE, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 1, 1, 1, 1)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi4x8ACE, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 1, 1, 1, 1)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi6x8ACE, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 1, 1, 1, 1)));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi3x8ACE, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 2, 2, 2, 2)));
      vo1p1 = wasm_f32x4_add(vo1p1, wasm_f32x4_mul(vi5x8ACE, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 2, 2, 2, 2)));
      vo2p1 = wasm_f32x4_add(vo2p1, wasm_f32x4_mul(vi7x8ACE, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 2, 2, 2, 2)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi4x8ACE, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 3, 3, 3, 3)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi6x8ACE, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 3, 3, 3, 3)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi8x8ACE, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 3, 3, 3, 3)));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi0x9BDF, wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0)));
      vo1p1 = wasm_f32x4_add(vo1p1, wasm_f32x4_mul(vi2x9BDF, wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0)));
      vo2p1 = wasm_f32x4_add(vo2p1, wasm_f32x4_mul(vi4x9BDF, wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi1x9BDF, wasm_v32x4_shuffle(vw89AB, vw89AB, 1, 1, 1, 1)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi3x9BDF, wasm_v32x4_shuffle(vw89AB, vw89AB, 1, 1, 1, 1)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi5x9BDF, wasm_v32x4_shuffle(vw89AB, vw89AB, 1, 1, 1, 1)));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi2x9BDF, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 2, 2, 2, 2)));
      vo1p1 = wasm_f32x4_add(vo1p1, wasm_f32x4_mul(vi4x9BDF, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 2, 2, 2, 2)));
      vo2p1 = wasm_f32x4_add(vo2p1, wasm_f32x4_mul(vi6x9BDF, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 2, 2, 2, 2)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi3x9BDF, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 3, 3, 3, 3)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi5x9BDF, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 3, 3, 3, 3)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi7x9BDF, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 3, 3, 3, 3)));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi4x9BDF, wasm_v32x4_shuffle(vwOP, vwOP, 0, 0, 0, 0)));
      vo1p1 = wasm_f32x4_add(vo1p1, wasm_f32x4_mul(vi6x9BDF, wasm_v32x4_shuffle(vwOP, vwOP, 0, 0, 0, 0)));
      vo2p1 = wasm_f32x4_add(vo2p1, wasm_f32x4_mul(vi8x9BDF, wasm_v32x4_shuffle(vwOP, vwOP, 0, 0, 0, 0)));

      const v128_t vi0x68AC = wasm_v32x4_shuffle(vi0x0246, vi0x8ACE, 3, 4, 5, 6);
      vi0x0246 = vi0x8ACE;
      const v128_t vi1x68AC = wasm_v32x4_shuffle(vi1x0246, vi1x8ACE, 3, 4, 5, 6);
      vi1x0246 = vi1x8ACE;
      const v128_t vi2x68AC = wasm_v32x4_shuffle(vi2x0246, vi2x8ACE, 3, 4, 5, 6);
      vi2x0246 = vi2x8ACE;
      const v128_t vi3x68AC = wasm_v32x4_shuffle(vi3x0246, vi3x8ACE, 3, 4, 5, 6);
      vi3x0246 = vi3x8ACE;
      const v128_t vi4x68AC = wasm_v32x4_shuffle(vi4x0246, vi4x8ACE, 3, 4, 5, 6);
      vi4x0246 = vi4x8ACE;
      const v128_t vi5x68AC = wasm_v32x4_shuffle(vi5x0246, vi5x8ACE, 3, 4, 5, 6);
      vi5x0246 = vi5x8ACE;
      const v128_t vi6x68AC = wasm_v32x4_shuffle(vi6x0246, vi6x8ACE, 3, 4, 5, 6);
      vi6x0246 = vi6x8ACE;
      const v128_t vi7x68AC = wasm_v32x4_shuffle(vi7x0246, vi7x8ACE, 3, 4, 5, 6);
      vi7x0246 = vi7x8ACE;
      const v128_t vi8x68AC = wasm_v32x4_shuffle(vi8x0246, vi8x8ACE, 3, 4, 5, 6);
      vi8x0246 = vi8x8ACE;

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi0x68AC, wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi2x68AC, wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi4x68AC, wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1)));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi1x68AC, wasm_v32x4_shuffle(vw4567, vw4567, 2, 2, 2, 2)));
      vo1p1 = wasm_f32x4_add(vo1p1, wasm_f32x4_mul(vi3x68AC, wasm_v32x4_shuffle(vw4567, vw4567, 2, 2, 2, 2)));
      vo2p1 = wasm_f32x4_add(vo2p1, wasm_f32x4_mul(vi5x68AC, wasm_v32x4_shuffle(vw4567, vw4567, 2, 2, 2, 2)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi2x68AC, wasm_v32x4_shuffle(vw89AB, vw89AB, 3, 3, 3, 3)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi4x68AC, wasm_v32x4_shuffle(vw89AB, vw89AB, 3, 3, 3, 3)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi6x68AC, wasm_v32x4_shuffle(vw89AB, vw89AB, 3, 3, 3, 3)));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi3x68AC, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 0, 0, 0, 0)));
      vo1p1 = wasm_f32x4_add(vo1p1, wasm_f32x4_mul(vi5x68AC, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 0, 0, 0, 0)));
      vo2p1 = wasm_f32x4_add(vo2p1, wasm_f32x4_mul(vi7x68AC, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 0, 0, 0, 0)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi4x68AC, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 1, 1, 1, 1)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi6x68AC, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 1, 1, 1, 1)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi8x68AC, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 1, 1, 1, 1)));

      const v128_t vi0x79BD = wasm_v32x4_shuffle(vi0x1357, vi0x9BDF, 3, 4, 5, 6);
      vi0x1357 = vi0x9BDF;
      const v128_t vi1x79BD = wasm_v32x4_shuffle(vi1x1357, vi1x9BDF, 3, 4, 5, 6);
      vi1x1357 = vi1x9BDF;
      const v128_t vi2x79BD = wasm_v32x4_shuffle(vi2x1357, vi2x9BDF, 3, 4, 5, 6);
      vi2x1357 = vi2x9BDF;
      const v128_t vi3x79BD = wasm_v32x4_shuffle(vi3x1357, vi3x9BDF, 3, 4, 5, 6);
      vi3x1357 = vi3x9BDF;
      const v128_t vi4x79BD = wasm_v32x4_shuffle(vi4x1357, vi4x9BDF, 3, 4, 5, 6);
      vi4x1357 = vi4x9BDF;
      const v128_t vi5x79BD = wasm_v32x4_shuffle(vi5x1357, vi5x9BDF, 3, 4, 5, 6);
      vi5x1357 = vi5x9BDF;
      const v128_t vi6x79BD = wasm_v32x4_shuffle(vi6x1357, vi6x9BDF, 3, 4, 5, 6);
      vi6x1357 = vi6x9BDF;
      const v128_t vi7x79BD = wasm_v32x4_shuffle(vi7x1357, vi7x9BDF, 3, 4, 5, 6);
      vi7x1357 = vi7x9BDF;
      const v128_t vi8x79BD = wasm_v32x4_shuffle(vi8x1357, vi8x9BDF, 3, 4, 5, 6);
      vi8x1357 = vi8x9BDF;

      const v128_t vi0xGHIJ = wasm_v128_load(i0);
      const v128_t vi0xKLMN = wasm_v128_load(i0 + 4);
      i0 += 8;
      const v128_t vi1xGHIJ = wasm_v128_load(i1);
      const v128_t vi1xKLMN = wasm_v128_load(i1 + 4);
      i1 += 8;
      const v128_t vi2xGHIJ = wasm_v128_load(i2);
      const v128_t vi2xKLMN = wasm_v128_load(i2 + 4);
      i2 += 8;
      const v128_t vi3xGHIJ = wasm_v128_load(i3);
      const v128_t vi3xKLMN = wasm_v128_load(i3 + 4);
      i3 += 8;
      const v128_t vi4xGHIJ = wasm_v128_load(i4);
      const v128_t vi4xKLMN = wasm_v128_load(i4 + 4);
      i4 += 8;
      const v128_t vi5xGHIJ = wasm_v128_load(i5);
      const v128_t vi5xKLMN = wasm_v128_load(i5 + 4);
      i5 += 8;
      const v128_t vi6xGHIJ = wasm_v128_load(i6);
      const v128_t vi6xKLMN = wasm_v128_load(i6 + 4);
      i6 += 8;
      const v128_t vi7xGHIJ = wasm_v128_load(i7);
      const v128_t vi7xKLMN = wasm_v128_load(i7 + 4);
      i7 += 8;
      const v128_t vi8xGHIJ = wasm_v128_load(i8);
      const v128_t vi8xKLMN = wasm_v128_load(i8 + 4);
      i8 += 8;

      const v128_t vi0xGIKM = wasm_v32x4_shuffle(vi0xGHIJ, vi0xKLMN, 0, 2, 4, 6);
      const v128_t vi0xHJLN = wasm_v32x4_shuffle(vi0xGHIJ, vi0xKLMN, 1, 3, 5, 7);
      const v128_t vi1xGIKM = wasm_v32x4_shuffle(vi1xGHIJ, vi1xKLMN, 0, 2, 4, 6);
      const v128_t vi1xHJLN = wasm_v32x4_shuffle(vi1xGHIJ, vi1xKLMN, 1, 3, 5, 7);
      const v128_t vi2xGIKM = wasm_v32x4_shuffle(vi2xGHIJ, vi2xKLMN, 0, 2, 4, 6);
      const v128_t vi2xHJLN = wasm_v32x4_shuffle(vi2xGHIJ, vi2xKLMN, 1, 3, 5, 7);
      const v128_t vi3xGIKM = wasm_v32x4_shuffle(vi3xGHIJ, vi3xKLMN, 0, 2, 4, 6);
      const v128_t vi3xHJLN = wasm_v32x4_shuffle(vi3xGHIJ, vi3xKLMN, 1, 3, 5, 7);
      const v128_t vi4xGIKM = wasm_v32x4_shuffle(vi4xGHIJ, vi4xKLMN, 0, 2, 4, 6);
      const v128_t vi4xHJLN = wasm_v32x4_shuffle(vi4xGHIJ, vi4xKLMN, 1, 3, 5, 7);
      const v128_t vi5xGIKM = wasm_v32x4_shuffle(vi5xGHIJ, vi5xKLMN, 0, 2, 4, 6);
      const v128_t vi5xHJLN = wasm_v32x4_shuffle(vi5xGHIJ, vi5xKLMN, 1, 3, 5, 7);
      const v128_t vi6xGIKM = wasm_v32x4_shuffle(vi6xGHIJ, vi6xKLMN, 0, 2, 4, 6);
      const v128_t vi6xHJLN = wasm_v32x4_shuffle(vi6xGHIJ, vi6xKLMN, 1, 3, 5, 7);
      const v128_t vi7xGIKM = wasm_v32x4_shuffle(vi7xGHIJ, vi7xKLMN, 0, 2, 4, 6);
      const v128_t vi7xHJLN = wasm_v32x4_shuffle(vi7xGHIJ, vi7xKLMN, 1, 3, 5, 7);
      const v128_t vi8xGIKM = wasm_v32x4_shuffle(vi8xGHIJ, vi8xKLMN, 0, 2, 4, 6);
      const v128_t vi8xHJLN = wasm_v32x4_shuffle(vi8xGHIJ, vi8xKLMN, 1, 3, 5, 7);

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi0x79BD, wasm_v32x4_shuffle(vw0123, vw0123, 2, 2, 2, 2)));
      vo1p1 = wasm_f32x4_add(vo1p1, wasm_f32x4_mul(vi2x79BD, wasm_v32x4_shuffle(vw0123, vw0123, 2, 2, 2, 2)));
      vo2p1 = wasm_f32x4_add(vo2p1, wasm_f32x4_mul(vi4x79BD, wasm_v32x4_shuffle(vw0123, vw0123, 2, 2, 2, 2)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi1x79BD, wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi3x79BD, wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi5x79BD, wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3)));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi2x79BD, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 0, 0, 0, 0)));
      vo1p1 = wasm_f32x4_add(vo1p1, wasm_f32x4_mul(vi4x79BD, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 0, 0, 0, 0)));
      vo2p1 = wasm_f32x4_add(vo2p1, wasm_f32x4_mul(vi6x79BD, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 0, 0, 0, 0)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi3x79BD, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 1, 1, 1, 1)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi5x79BD, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 1, 1, 1, 1)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi7x79BD, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 1, 1, 1, 1)));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi4x79BD, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 2, 2, 2, 2)));
      vo1p1 = wasm_f32x4_add(vo1p1, wasm_f32x4_mul(vi6x79BD, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 2, 2, 2, 2)));
      vo2p1 = wasm_f32x4_add(vo2p1, wasm_f32x4_mul(vi8x79BD, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 2, 2, 2, 2)));

      const v128_t vi0xACEG = wasm_v32x4_shuffle(vi0x8ACE, vi0xGIKM, 1, 2, 3, 4);
      vi0x8ACE = vi0xGIKM;
      vi0x9BDF = vi0xHJLN;
      const v128_t vi1xACEG = wasm_v32x4_shuffle(vi1x8ACE, vi1xGIKM, 1, 2, 3, 4);
      vi1x8ACE = vi1xGIKM;
      vi1x9BDF = vi1xHJLN;
      const v128_t vi2xACEG = wasm_v32x4_shuffle(vi2x8ACE, vi2xGIKM, 1, 2, 3, 4);
      vi2x8ACE = vi2xGIKM;
      vi2x9BDF = vi2xHJLN;
      const v128_t vi3xACEG = wasm_v32x4_shuffle(vi3x8ACE, vi3xGIKM, 1, 2, 3, 4);
      vi3x8ACE = vi3xGIKM;
      vi3x9BDF = vi3xHJLN;
      const v128_t vi4xACEG = wasm_v32x4_shuffle(vi4x8ACE, vi4xGIKM, 1, 2, 3, 4);
      vi4x8ACE = vi4xGIKM;
      vi4x9BDF = vi4xHJLN;
      const v128_t vi5xACEG = wasm_v32x4_shuffle(vi5x8ACE, vi5xGIKM, 1, 2, 3, 4);
      vi5x8ACE = vi5xGIKM;
      vi5x9BDF = vi5xHJLN;
      const v128_t vi6xACEG = wasm_v32x4_shuffle(vi6x8ACE, vi6xGIKM, 1, 2, 3, 4);
      vi6x8ACE = vi6xGIKM;
      vi6x9BDF = vi6xHJLN;
      const v128_t vi7xACEG = wasm_v32x4_shuffle(vi7x8ACE, vi7xGIKM, 1, 2, 3, 4);
      vi7x8ACE = vi7xGIKM;
      vi7x9BDF = vi7xHJLN;
      const v128_t vi8xACEG = wasm_v32x4_shuffle(vi8x8ACE, vi8xGIKM, 1, 2, 3, 4);
      vi8x8ACE = vi8xGIKM;
      vi8x9BDF = vi8xHJLN;

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi0xACEG, wasm_v32x4_shuffle(vw4567, vw4567, 1, 1, 1, 1)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi2xACEG, wasm_v32x4_shuffle(vw4567, vw4567, 1, 1, 1, 1)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi4xACEG, wasm_v32x4_shuffle(vw4567, vw4567, 1, 1, 1, 1)));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi1xACEG, wasm_v32x4_shuffle(vw89AB, vw89AB, 2, 2, 2, 2)));
      vo1p1 = wasm_f32x4_add(vo1p1, wasm_f32x4_mul(vi3xACEG, wasm_v32x4_shuffle(vw89AB, vw89AB, 2, 2, 2, 2)));
      vo2p1 = wasm_f32x4_add(vo2p1, wasm_f32x4_mul(vi5xACEG, wasm_v32x4_shuffle(vw89AB, vw89AB, 2, 2, 2, 2)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi2xACEG, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 3, 3, 3, 3)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi4xACEG, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 3, 3, 3, 3)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi6xACEG, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 3, 3, 3, 3)));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi3xACEG, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 0, 0, 0, 0)));
      vo1p1 = wasm_f32x4_add(vo1p1, wasm_f32x4_mul(vi5xACEG, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 0, 0, 0, 0)));
      vo2p1 = wasm_f32x4_add(vo2p1, wasm_f32x4_mul(vi7xACEG, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 0, 0, 0, 0)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi4xACEG, wasm_v32x4_shuffle(vwOP, vwOP, 1, 1, 1, 1)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi6xACEG, wasm_v32x4_shuffle(vwOP, vwOP, 1, 1, 1, 1)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi8xACEG, wasm_v32x4_shuffle(vwOP, vwOP, 1, 1, 1, 1)));

      vo0p0 = wasm_f32x4_add(vo0p0, vo0p1);
      vo1p0 = wasm_f32x4_add(vo1p0, vo1p1);
      vo2p0 = wasm_f32x4_add(vo2p0, vo2p1);

      v128_t vo0 = wasm_f32x4_pmax(vmin, vo0p0);
      v128_t vo1 = wasm_f32x4_pmax(vmin, vo1p0);
      v128_t vo2 = wasm_f32x4_pmax(vmin, vo2p0);
      vo0 = wasm_f32x4_pmin(vmax, vo0);
      vo1 = wasm_f32x4_pmin(vmax, vo1);
      vo2 = wasm_f32x4_pmin(vmax, vo2);

      wasm_v128_store(o2, vo2); o2 += 4;
      wasm_v128_store(o1, vo1); o1 += 4;
      wasm_v128_store(o0, vo0); o0 += 4;
    }
    // Last block has 1-8 pixels to process.
    assert(w <= 8 * sizeof(float));
    assert(w >= 1 * sizeof(float));
    {
      v128_t vo0p0 = wasm_v32x4_shuffle(vw0123, vw0123, 0, 0, 0, 0);
      v128_t vo1p0 = wasm_v32x4_shuffle(vw0123, vw0123, 0, 0, 0, 0);
      v128_t vo2p0 = wasm_v32x4_shuffle(vw0123, vw0123, 0, 0, 0, 0);

      vi0x8ACE = wasm_v128_and(vmask_even, vi0x8ACE);
      vi1x8ACE = wasm_v128_and(vmask_even, vi1x8ACE);
      vi2x8ACE = wasm_v128_and(vmask_even, vi2x8ACE);
      vi3x8ACE = wasm_v128_and(vmask_even, vi3x8ACE);
      vi4x8ACE = wasm_v128_and(vmask_even, vi4x8ACE);
      vi5x8ACE = wasm_v128_and(vmask_even, vi5x8ACE);
      vi6x8ACE = wasm_v128_and(vmask_even, vi6x8ACE);
      vi7x8ACE = wasm_v128_and(vmask_even, vi7x8ACE);
      vi8x8ACE = wasm_v128_and(vmask_even, vi8x8ACE);

      vi0x9BDF = wasm_v128_and(vmask_odd, vi0x9BDF);
      vi1x9BDF = wasm_v128_and(vmask_odd, vi1x9BDF);
      vi2x9BDF = wasm_v128_and(vmask_odd, vi2x9BDF);
      vi3x9BDF = wasm_v128_and(vmask_odd, vi3x9BDF);
      vi4x9BDF = wasm_v128_and(vmask_odd, vi4x9BDF);
      vi5x9BDF = wasm_v128_and(vmask_odd, vi5x9BDF);
      vi6x9BDF = wasm_v128_and(vmask_odd, vi6x9BDF);
      vi7x9BDF = wasm_v128_and(vmask_odd, vi7x9BDF);
      vi8x9BDF = wasm_v128_and(vmask_odd, vi8x9BDF);

      v128_t vo0p1 = wasm_f32x4_mul(vi0x8ACE, wasm_v32x4_shuffle(vw0123, vw0123, 3, 3, 3, 3));
      v128_t vo1p1 = wasm_f32x4_mul(vi2x8ACE, wasm_v32x4_shuffle(vw0123, vw0123, 3, 3, 3, 3));
      v128_t vo2p1 = wasm_f32x4_mul(vi4x8ACE, wasm_v32x4_shuffle(vw0123, vw0123, 3, 3, 3, 3));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi1x8ACE, wasm_v32x4_shuffle(vw89AB, vw89AB, 0, 0, 0, 0)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi3x8ACE, wasm_v32x4_shuffle(vw89AB, vw89AB, 0, 0, 0, 0)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi5x8ACE, wasm_v32x4_shuffle(vw89AB, vw89AB, 0, 0, 0, 0)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi2x8ACE, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 1, 1, 1, 1)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi4x8ACE, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 1, 1, 1, 1)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi6x8ACE, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 1, 1, 1, 1)));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi3x8ACE, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 2, 2, 2, 2)));
      vo1p1 = wasm_f32x4_add(vo1p1, wasm_f32x4_mul(vi5x8ACE, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 2, 2, 2, 2)));
      vo2p1 = wasm_f32x4_add(vo2p1, wasm_f32x4_mul(vi7x8ACE, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 2, 2, 2, 2)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi4x8ACE, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 3, 3, 3, 3)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi6x8ACE, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 3, 3, 3, 3)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi8x8ACE, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 3, 3, 3, 3)));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi0x9BDF, wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0)));
      vo1p1 = wasm_f32x4_add(vo1p1, wasm_f32x4_mul(vi2x9BDF, wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0)));
      vo2p1 = wasm_f32x4_add(vo2p1, wasm_f32x4_mul(vi4x9BDF, wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi1x9BDF, wasm_v32x4_shuffle(vw89AB, vw89AB, 1, 1, 1, 1)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi3x9BDF, wasm_v32x4_shuffle(vw89AB, vw89AB, 1, 1, 1, 1)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi5x9BDF, wasm_v32x4_shuffle(vw89AB, vw89AB, 1, 1, 1, 1)));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi2x9BDF, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 2, 2, 2, 2)));
      vo1p1 = wasm_f32x4_add(vo1p1, wasm_f32x4_mul(vi4x9BDF, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 2, 2, 2, 2)));
      vo2p1 = wasm_f32x4_add(vo2p1, wasm_f32x4_mul(vi6x9BDF, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 2, 2, 2, 2)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi3x9BDF, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 3, 3, 3, 3)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi5x9BDF, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 3, 3, 3, 3)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi7x9BDF, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 3, 3, 3, 3)));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi4x9BDF, wasm_v32x4_shuffle(vwOP, vwOP, 0, 0, 0, 0)));
      vo1p1 = wasm_f32x4_add(vo1p1, wasm_f32x4_mul(vi6x9BDF, wasm_v32x4_shuffle(vwOP, vwOP, 0, 0, 0, 0)));
      vo2p1 = wasm_f32x4_add(vo2p1, wasm_f32x4_mul(vi8x9BDF, wasm_v32x4_shuffle(vwOP, vwOP, 0, 0, 0, 0)));

      const v128_t vi0x68AC = wasm_v32x4_shuffle(vi0x0246, vi0x8ACE, 3, 4, 5, 6);
      const v128_t vi1x68AC = wasm_v32x4_shuffle(vi1x0246, vi1x8ACE, 3, 4, 5, 6);
      const v128_t vi2x68AC = wasm_v32x4_shuffle(vi2x0246, vi2x8ACE, 3, 4, 5, 6);
      const v128_t vi3x68AC = wasm_v32x4_shuffle(vi3x0246, vi3x8ACE, 3, 4, 5, 6);
      const v128_t vi4x68AC = wasm_v32x4_shuffle(vi4x0246, vi4x8ACE, 3, 4, 5, 6);
      const v128_t vi5x68AC = wasm_v32x4_shuffle(vi5x0246, vi5x8ACE, 3, 4, 5, 6);
      const v128_t vi6x68AC = wasm_v32x4_shuffle(vi6x0246, vi6x8ACE, 3, 4, 5, 6);
      const v128_t vi7x68AC = wasm_v32x4_shuffle(vi7x0246, vi7x8ACE, 3, 4, 5, 6);
      const v128_t vi8x68AC = wasm_v32x4_shuffle(vi8x0246, vi8x8ACE, 3, 4, 5, 6);

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi0x68AC, wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi2x68AC, wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi4x68AC, wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1)));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi1x68AC, wasm_v32x4_shuffle(vw4567, vw4567, 2, 2, 2, 2)));
      vo1p1 = wasm_f32x4_add(vo1p1, wasm_f32x4_mul(vi3x68AC, wasm_v32x4_shuffle(vw4567, vw4567, 2, 2, 2, 2)));
      vo2p1 = wasm_f32x4_add(vo2p1, wasm_f32x4_mul(vi5x68AC, wasm_v32x4_shuffle(vw4567, vw4567, 2, 2, 2, 2)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi2x68AC, wasm_v32x4_shuffle(vw89AB, vw89AB, 3, 3, 3, 3)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi4x68AC, wasm_v32x4_shuffle(vw89AB, vw89AB, 3, 3, 3, 3)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi6x68AC, wasm_v32x4_shuffle(vw89AB, vw89AB, 3, 3, 3, 3)));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi3x68AC, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 0, 0, 0, 0)));
      vo1p1 = wasm_f32x4_add(vo1p1, wasm_f32x4_mul(vi5x68AC, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 0, 0, 0, 0)));
      vo2p1 = wasm_f32x4_add(vo2p1, wasm_f32x4_mul(vi7x68AC, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 0, 0, 0, 0)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi4x68AC, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 1, 1, 1, 1)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi6x68AC, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 1, 1, 1, 1)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi8x68AC, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 1, 1, 1, 1)));

      const v128_t vi0x79BD = wasm_v32x4_shuffle(vi0x1357, vi0x9BDF, 3, 4, 5, 6);
      const v128_t vi1x79BD = wasm_v32x4_shuffle(vi1x1357, vi1x9BDF, 3, 4, 5, 6);
      const v128_t vi2x79BD = wasm_v32x4_shuffle(vi2x1357, vi2x9BDF, 3, 4, 5, 6);
      const v128_t vi3x79BD = wasm_v32x4_shuffle(vi3x1357, vi3x9BDF, 3, 4, 5, 6);
      const v128_t vi4x79BD = wasm_v32x4_shuffle(vi4x1357, vi4x9BDF, 3, 4, 5, 6);
      const v128_t vi5x79BD = wasm_v32x4_shuffle(vi5x1357, vi5x9BDF, 3, 4, 5, 6);
      const v128_t vi6x79BD = wasm_v32x4_shuffle(vi6x1357, vi6x9BDF, 3, 4, 5, 6);
      const v128_t vi7x79BD = wasm_v32x4_shuffle(vi7x1357, vi7x9BDF, 3, 4, 5, 6);
      const v128_t vi8x79BD = wasm_v32x4_shuffle(vi8x1357, vi8x9BDF, 3, 4, 5, 6);

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi0x79BD, wasm_v32x4_shuffle(vw0123, vw0123, 2, 2, 2, 2)));
      vo1p1 = wasm_f32x4_add(vo1p1, wasm_f32x4_mul(vi2x79BD, wasm_v32x4_shuffle(vw0123, vw0123, 2, 2, 2, 2)));
      vo2p1 = wasm_f32x4_add(vo2p1, wasm_f32x4_mul(vi4x79BD, wasm_v32x4_shuffle(vw0123, vw0123, 2, 2, 2, 2)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi1x79BD, wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi3x79BD, wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi5x79BD, wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3)));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi2x79BD, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 0, 0, 0, 0)));
      vo1p1 = wasm_f32x4_add(vo1p1, wasm_f32x4_mul(vi4x79BD, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 0, 0, 0, 0)));
      vo2p1 = wasm_f32x4_add(vo2p1, wasm_f32x4_mul(vi6x79BD, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 0, 0, 0, 0)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi3x79BD, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 1, 1, 1, 1)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi5x79BD, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 1, 1, 1, 1)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi7x79BD, wasm_v32x4_shuffle(vwGHIJ, vwGHIJ, 1, 1, 1, 1)));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi4x79BD, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 2, 2, 2, 2)));
      vo1p1 = wasm_f32x4_add(vo1p1, wasm_f32x4_mul(vi6x79BD, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 2, 2, 2, 2)));
      vo2p1 = wasm_f32x4_add(vo2p1, wasm_f32x4_mul(vi8x79BD, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 2, 2, 2, 2)));

      const v128_t vzero = wasm_f32x4_const_splat(0.0f);
      const v128_t vi0xACEG = wasm_v32x4_shuffle(vi0x8ACE, vzero, 1, 2, 3, 4);
      const v128_t vi1xACEG = wasm_v32x4_shuffle(vi1x8ACE, vzero, 1, 2, 3, 4);
      const v128_t vi2xACEG = wasm_v32x4_shuffle(vi2x8ACE, vzero, 1, 2, 3, 4);
      const v128_t vi3xACEG = wasm_v32x4_shuffle(vi3x8ACE, vzero, 1, 2, 3, 4);
      const v128_t vi4xACEG = wasm_v32x4_shuffle(vi4x8ACE, vzero, 1, 2, 3, 4);
      const v128_t vi5xACEG = wasm_v32x4_shuffle(vi5x8ACE, vzero, 1, 2, 3, 4);
      const v128_t vi6xACEG = wasm_v32x4_shuffle(vi6x8ACE, vzero, 1, 2, 3, 4);
      const v128_t vi7xACEG = wasm_v32x4_shuffle(vi7x8ACE, vzero, 1, 2, 3, 4);
      const v128_t vi8xACEG = wasm_v32x4_shuffle(vi8x8ACE, vzero, 1, 2, 3, 4);

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi0xACEG, wasm_v32x4_shuffle(vw4567, vw4567, 1, 1, 1, 1)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi2xACEG, wasm_v32x4_shuffle(vw4567, vw4567, 1, 1, 1, 1)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi4xACEG, wasm_v32x4_shuffle(vw4567, vw4567, 1, 1, 1, 1)));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi1xACEG, wasm_v32x4_shuffle(vw89AB, vw89AB, 2, 2, 2, 2)));
      vo1p1 = wasm_f32x4_add(vo1p1, wasm_f32x4_mul(vi3xACEG, wasm_v32x4_shuffle(vw89AB, vw89AB, 2, 2, 2, 2)));
      vo2p1 = wasm_f32x4_add(vo2p1, wasm_f32x4_mul(vi5xACEG, wasm_v32x4_shuffle(vw89AB, vw89AB, 2, 2, 2, 2)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi2xACEG, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 3, 3, 3, 3)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi4xACEG, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 3, 3, 3, 3)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi6xACEG, wasm_v32x4_shuffle(vwCDEF, vwCDEF, 3, 3, 3, 3)));

      vo0p1 = wasm_f32x4_add(vo0p1, wasm_f32x4_mul(vi3xACEG, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 0, 0, 0, 0)));
      vo1p1 = wasm_f32x4_add(vo1p1, wasm_f32x4_mul(vi5xACEG, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 0, 0, 0, 0)));
      vo2p1 = wasm_f32x4_add(vo2p1, wasm_f32x4_mul(vi7xACEG, wasm_v32x4_shuffle(vwKLMN, vwKLMN, 0, 0, 0, 0)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi4xACEG, wasm_v32x4_shuffle(vwOP, vwOP, 1, 1, 1, 1)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi6xACEG, wasm_v32x4_shuffle(vwOP, vwOP, 1, 1, 1, 1)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi8xACEG, wasm_v32x4_shuffle(vwOP, vwOP, 1, 1, 1, 1)));

      vo0p0 = wasm_f32x4_add(vo0p0, vo0p1);
      vo1p0 = wasm_f32x4_add(vo1p0, vo1p1);
      vo2p0 = wasm_f32x4_add(vo2p0, vo2p1);

      v128_t vo0 = wasm_f32x4_pmax(vmin, vo0p0);
      v128_t vo1 = wasm_f32x4_pmax(vmin, vo1p0);
      v128_t vo2 = wasm_f32x4_pmax(vmin, vo2p0);
      vo0 = wasm_f32x4_pmin(vmax, vo0);
      vo1 = wasm_f32x4_pmin(vmax, vo1);
      vo2 = wasm_f32x4_pmin(vmax, vo2);

      size_t w_tmp = (w + 1 * sizeof(float)) / (2 * sizeof(float));
      if XNN_LIKELY(w_tmp >= 4) {
        wasm_v128_store(o2, vo2);
        o2 += 4;
        wasm_v128_store(o1, vo1);
        o1 += 4;
        wasm_v128_store(o0, vo0);
        o0 += 4;
      } else {
        if (w_tmp & 2) {
          wasm_v128_store64_lane(o2, vo2, 0);
          o2 += 2;
          wasm_v128_store64_lane(o1, vo1, 0);
          o1 += 2;
          wasm_v128_store64_lane(o0, vo0, 0);
          o0 += 2;

          vo0 = wasm_v64x2_shuffle(vo0, vo0, 1, 1);
          vo1 = wasm_v64x2_shuffle(vo1, vo1, 1, 1);
          vo2 = wasm_v64x2_shuffle(vo2, vo2, 1, 1);
        }
        if (w_tmp & 1) {
          wasm_v128_store32_lane(o2, vo2, 0);
          o2 += 1;
          wasm_v128_store32_lane(o1, vo1, 0);
          o1 += 1;
          wasm_v128_store32_lane(o0, vo0, 0);
          o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i6 - input_decrement);
    i1 = (const float*) ((uintptr_t) i7 - input_decrement);
    i2 = (const float*) ((uintptr_t) i8 - input_decrement);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);
    i5 = (const float*) ((uintptr_t) i4 + input_width);
    i6 = (const float*) ((uintptr_t) i5 + input_width);
    i7 = (const float*) ((uintptr_t) i6 + input_width);
    i8 = (const float*) ((uintptr_t) i7 + input_width);

    o0 = o2;
    o1 = (float*) ((uintptr_t) o0 + output_width);
    o2 = (float*) ((uintptr_t) o1 + output_width);

    output_height = doz(output_height, 3);
    padded_input_height = doz(padded_input_height, 6);
  } while (output_height != 0);
}
