// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-splat.c.in
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



void xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_splat_4x4(
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
  assert(padding_top >= 0);
  assert(padding_top <= 1);

  const v128_t vmax = wasm_v128_load32_splat(&params->scalar.max);
  const v128_t vmin = wasm_v128_load32_splat(&params->scalar.min);
  XNN_FORCE_REALIZATION(vmax);
  XNN_FORCE_REALIZATION(vmin);

  static const int32_t mask_table[8] = {-1, -1, -1, -1, 0, 0, 0, 0};
  const v128_t vmask_even = wasm_v128_load(&mask_table[4 - (((input_width & 31) + 4) >> 3)]);
  const v128_t vmask_odd = wasm_v128_load(&mask_table[4 - ((input_width & 31) >> 3)]);

  const v128_t vw0123 = wasm_v128_load(weights);
  const v128_t vw4567 = wasm_v128_load(weights + 4);
  const v128_t vw89 = wasm_v128_load64_splat(weights + 8);

  const size_t input_decrement = round_down_po2(input_width, 4 /* SIMD output width */ * 2 /* subsampling */ * sizeof(float));
  const size_t output_width = round_down_po2((input_width + (2 /* padding */ - 3 /* kernel size */ + 2 /* subsampling */) * sizeof(float)) / 2, sizeof(float));

  const float* i0 = (const float*) ((uintptr_t) input - ((-padding_top) & input_width));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_width);
  if XNN_UNPREDICTABLE(padding_top != 0) {
    i0 = zero;
  }
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_width);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_width);
  const float* i7 = (const float*) ((uintptr_t) i6 + input_width);
  const float* i8 = (const float*) ((uintptr_t) i7 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + output_width);
  float* o2 = (float*) ((uintptr_t) o1 + output_width);
  float* o3 = (float*) ((uintptr_t) o2 + output_width);

  size_t padded_input_height = input_height + padding_top + 1 /* padding bottom */;
  size_t output_height = (padded_input_height - 3 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 4) {
      i2 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 5) {
      i3 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 6) {
      i4 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 7) {
      i5 = zero;
      o2 = o1;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 8) {
      i6 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 9) {
      i7 = zero;
      o3 = o2;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 10) {
      i8 = zero;
    }

    v128_t vi0x1357 = wasm_f32x4_const_splat(0.0f);
    v128_t vi1x1357 = wasm_f32x4_const_splat(0.0f);
    v128_t vi2x1357 = wasm_f32x4_const_splat(0.0f);
    v128_t vi3x1357 = wasm_f32x4_const_splat(0.0f);
    v128_t vi4x1357 = wasm_f32x4_const_splat(0.0f);
    v128_t vi5x1357 = wasm_f32x4_const_splat(0.0f);
    v128_t vi6x1357 = wasm_f32x4_const_splat(0.0f);
    v128_t vi7x1357 = wasm_f32x4_const_splat(0.0f);
    v128_t vi8x1357 = wasm_f32x4_const_splat(0.0f);

    size_t w = input_width;
    for (; w >= 8 * sizeof(float); w -= 8 * sizeof(float)) {
      v128_t vo0p0 = wasm_v32x4_shuffle(vw0123, vw0123, 0, 0, 0, 0);
      v128_t vo1p0 = wasm_v32x4_shuffle(vw0123, vw0123, 0, 0, 0, 0);
      v128_t vo2p0 = wasm_v32x4_shuffle(vw0123, vw0123, 0, 0, 0, 0);
      v128_t vo3p0 = wasm_v32x4_shuffle(vw0123, vw0123, 0, 0, 0, 0);

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

      const v128_t vi0x8ACE = wasm_v32x4_shuffle(vi0x89AB, vi0xCDEF, 0, 2, 4, 6);
      const v128_t vi0x9BDF = wasm_v32x4_shuffle(vi0x89AB, vi0xCDEF, 1, 3, 5, 7);
      const v128_t vi1x8ACE = wasm_v32x4_shuffle(vi1x89AB, vi1xCDEF, 0, 2, 4, 6);
      const v128_t vi1x9BDF = wasm_v32x4_shuffle(vi1x89AB, vi1xCDEF, 1, 3, 5, 7);
      const v128_t vi2x8ACE = wasm_v32x4_shuffle(vi2x89AB, vi2xCDEF, 0, 2, 4, 6);
      const v128_t vi2x9BDF = wasm_v32x4_shuffle(vi2x89AB, vi2xCDEF, 1, 3, 5, 7);
      const v128_t vi3x8ACE = wasm_v32x4_shuffle(vi3x89AB, vi3xCDEF, 0, 2, 4, 6);
      const v128_t vi3x9BDF = wasm_v32x4_shuffle(vi3x89AB, vi3xCDEF, 1, 3, 5, 7);
      const v128_t vi4x8ACE = wasm_v32x4_shuffle(vi4x89AB, vi4xCDEF, 0, 2, 4, 6);
      const v128_t vi4x9BDF = wasm_v32x4_shuffle(vi4x89AB, vi4xCDEF, 1, 3, 5, 7);
      const v128_t vi5x8ACE = wasm_v32x4_shuffle(vi5x89AB, vi5xCDEF, 0, 2, 4, 6);
      const v128_t vi5x9BDF = wasm_v32x4_shuffle(vi5x89AB, vi5xCDEF, 1, 3, 5, 7);
      const v128_t vi6x8ACE = wasm_v32x4_shuffle(vi6x89AB, vi6xCDEF, 0, 2, 4, 6);
      const v128_t vi6x9BDF = wasm_v32x4_shuffle(vi6x89AB, vi6xCDEF, 1, 3, 5, 7);
      const v128_t vi7x8ACE = wasm_v32x4_shuffle(vi7x89AB, vi7xCDEF, 0, 2, 4, 6);
      const v128_t vi7x9BDF = wasm_v32x4_shuffle(vi7x89AB, vi7xCDEF, 1, 3, 5, 7);
      const v128_t vi8x8ACE = wasm_v32x4_shuffle(vi8x89AB, vi8xCDEF, 0, 2, 4, 6);
      const v128_t vi8x9BDF = wasm_v32x4_shuffle(vi8x89AB, vi8xCDEF, 1, 3, 5, 7);

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi0x8ACE, wasm_v32x4_shuffle(vw0123, vw0123, 2, 2, 2, 2)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi2x8ACE, wasm_v32x4_shuffle(vw0123, vw0123, 2, 2, 2, 2)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi4x8ACE, wasm_v32x4_shuffle(vw0123, vw0123, 2, 2, 2, 2)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi6x8ACE, wasm_v32x4_shuffle(vw0123, vw0123, 2, 2, 2, 2)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi1x8ACE, wasm_v32x4_shuffle(vw4567, vw4567, 1, 1, 1, 1)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi3x8ACE, wasm_v32x4_shuffle(vw4567, vw4567, 1, 1, 1, 1)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi5x8ACE, wasm_v32x4_shuffle(vw4567, vw4567, 1, 1, 1, 1)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi7x8ACE, wasm_v32x4_shuffle(vw4567, vw4567, 1, 1, 1, 1)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi2x8ACE, wasm_v32x4_shuffle(vw89, vw89, 0, 0, 0, 0)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi4x8ACE, wasm_v32x4_shuffle(vw89, vw89, 0, 0, 0, 0)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi6x8ACE, wasm_v32x4_shuffle(vw89, vw89, 0, 0, 0, 0)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi8x8ACE, wasm_v32x4_shuffle(vw89, vw89, 0, 0, 0, 0)));

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

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi0x79BD, wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi2x79BD, wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi4x79BD, wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi6x79BD, wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi1x79BD, wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi3x79BD, wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi5x79BD, wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi7x79BD, wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi2x79BD, wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi4x79BD, wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi6x79BD, wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi8x79BD, wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi0x9BDF, wasm_v32x4_shuffle(vw0123, vw0123, 3, 3, 3, 3)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi2x9BDF, wasm_v32x4_shuffle(vw0123, vw0123, 3, 3, 3, 3)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi4x9BDF, wasm_v32x4_shuffle(vw0123, vw0123, 3, 3, 3, 3)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi6x9BDF, wasm_v32x4_shuffle(vw0123, vw0123, 3, 3, 3, 3)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi1x9BDF, wasm_v32x4_shuffle(vw4567, vw4567, 2, 2, 2, 2)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi3x9BDF, wasm_v32x4_shuffle(vw4567, vw4567, 2, 2, 2, 2)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi5x9BDF, wasm_v32x4_shuffle(vw4567, vw4567, 2, 2, 2, 2)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi7x9BDF, wasm_v32x4_shuffle(vw4567, vw4567, 2, 2, 2, 2)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi2x9BDF, wasm_v32x4_shuffle(vw89, vw89, 1, 1, 1, 1)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi4x9BDF, wasm_v32x4_shuffle(vw89, vw89, 1, 1, 1, 1)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi6x9BDF, wasm_v32x4_shuffle(vw89, vw89, 1, 1, 1, 1)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi8x9BDF, wasm_v32x4_shuffle(vw89, vw89, 1, 1, 1, 1)));


      v128_t vo0 = wasm_f32x4_pmax(vmin, vo0p0);
      v128_t vo1 = wasm_f32x4_pmax(vmin, vo1p0);
      v128_t vo2 = wasm_f32x4_pmax(vmin, vo2p0);
      v128_t vo3 = wasm_f32x4_pmax(vmin, vo3p0);
      vo0 = wasm_f32x4_pmin(vmax, vo0);
      vo1 = wasm_f32x4_pmin(vmax, vo1);
      vo2 = wasm_f32x4_pmin(vmax, vo2);
      vo3 = wasm_f32x4_pmin(vmax, vo3);

      wasm_v128_store(o3, vo3); o3 += 4;
      wasm_v128_store(o2, vo2); o2 += 4;
      wasm_v128_store(o1, vo1); o1 += 4;
      wasm_v128_store(o0, vo0); o0 += 4;
    }
    // Last block has 0-7 pixels to process.
    assert(w < 8 * sizeof(float));
    if XNN_LIKELY(w != 0) {
      v128_t vo0p0 = wasm_v32x4_shuffle(vw0123, vw0123, 0, 0, 0, 0);
      v128_t vo1p0 = wasm_v32x4_shuffle(vw0123, vw0123, 0, 0, 0, 0);
      v128_t vo2p0 = wasm_v32x4_shuffle(vw0123, vw0123, 0, 0, 0, 0);
      v128_t vo3p0 = wasm_v32x4_shuffle(vw0123, vw0123, 0, 0, 0, 0);

      const v128_t vi0x89AB = wasm_v128_load(i0);
      const v128_t vi0xCDEF = wasm_v128_load(i0 + 4);
      const v128_t vi1x89AB = wasm_v128_load(i1);
      const v128_t vi1xCDEF = wasm_v128_load(i1 + 4);
      const v128_t vi2x89AB = wasm_v128_load(i2);
      const v128_t vi2xCDEF = wasm_v128_load(i2 + 4);
      const v128_t vi3x89AB = wasm_v128_load(i3);
      const v128_t vi3xCDEF = wasm_v128_load(i3 + 4);
      const v128_t vi4x89AB = wasm_v128_load(i4);
      const v128_t vi4xCDEF = wasm_v128_load(i4 + 4);
      const v128_t vi5x89AB = wasm_v128_load(i5);
      const v128_t vi5xCDEF = wasm_v128_load(i5 + 4);
      const v128_t vi6x89AB = wasm_v128_load(i6);
      const v128_t vi6xCDEF = wasm_v128_load(i6 + 4);
      const v128_t vi7x89AB = wasm_v128_load(i7);
      const v128_t vi7xCDEF = wasm_v128_load(i7 + 4);
      const v128_t vi8x89AB = wasm_v128_load(i8);
      const v128_t vi8xCDEF = wasm_v128_load(i8 + 4);

      const v128_t vi0x8ACE = wasm_v128_and(vmask_even, wasm_v32x4_shuffle(vi0x89AB, vi0xCDEF, 0, 2, 4, 6));
      const v128_t vi0x9BDF = wasm_v128_and(vmask_odd,  wasm_v32x4_shuffle(vi0x89AB, vi0xCDEF, 1, 3, 5, 7));
      const v128_t vi1x8ACE = wasm_v128_and(vmask_even, wasm_v32x4_shuffle(vi1x89AB, vi1xCDEF, 0, 2, 4, 6));
      const v128_t vi1x9BDF = wasm_v128_and(vmask_odd,  wasm_v32x4_shuffle(vi1x89AB, vi1xCDEF, 1, 3, 5, 7));
      const v128_t vi2x8ACE = wasm_v128_and(vmask_even, wasm_v32x4_shuffle(vi2x89AB, vi2xCDEF, 0, 2, 4, 6));
      const v128_t vi2x9BDF = wasm_v128_and(vmask_odd,  wasm_v32x4_shuffle(vi2x89AB, vi2xCDEF, 1, 3, 5, 7));
      const v128_t vi3x8ACE = wasm_v128_and(vmask_even, wasm_v32x4_shuffle(vi3x89AB, vi3xCDEF, 0, 2, 4, 6));
      const v128_t vi3x9BDF = wasm_v128_and(vmask_odd,  wasm_v32x4_shuffle(vi3x89AB, vi3xCDEF, 1, 3, 5, 7));
      const v128_t vi4x8ACE = wasm_v128_and(vmask_even, wasm_v32x4_shuffle(vi4x89AB, vi4xCDEF, 0, 2, 4, 6));
      const v128_t vi4x9BDF = wasm_v128_and(vmask_odd,  wasm_v32x4_shuffle(vi4x89AB, vi4xCDEF, 1, 3, 5, 7));
      const v128_t vi5x8ACE = wasm_v128_and(vmask_even, wasm_v32x4_shuffle(vi5x89AB, vi5xCDEF, 0, 2, 4, 6));
      const v128_t vi5x9BDF = wasm_v128_and(vmask_odd,  wasm_v32x4_shuffle(vi5x89AB, vi5xCDEF, 1, 3, 5, 7));
      const v128_t vi6x8ACE = wasm_v128_and(vmask_even, wasm_v32x4_shuffle(vi6x89AB, vi6xCDEF, 0, 2, 4, 6));
      const v128_t vi6x9BDF = wasm_v128_and(vmask_odd,  wasm_v32x4_shuffle(vi6x89AB, vi6xCDEF, 1, 3, 5, 7));
      const v128_t vi7x8ACE = wasm_v128_and(vmask_even, wasm_v32x4_shuffle(vi7x89AB, vi7xCDEF, 0, 2, 4, 6));
      const v128_t vi7x9BDF = wasm_v128_and(vmask_odd,  wasm_v32x4_shuffle(vi7x89AB, vi7xCDEF, 1, 3, 5, 7));
      const v128_t vi8x8ACE = wasm_v128_and(vmask_even, wasm_v32x4_shuffle(vi8x89AB, vi8xCDEF, 0, 2, 4, 6));
      const v128_t vi8x9BDF = wasm_v128_and(vmask_odd,  wasm_v32x4_shuffle(vi8x89AB, vi8xCDEF, 1, 3, 5, 7));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi0x8ACE, wasm_v32x4_shuffle(vw0123, vw0123, 2, 2, 2, 2)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi2x8ACE, wasm_v32x4_shuffle(vw0123, vw0123, 2, 2, 2, 2)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi4x8ACE, wasm_v32x4_shuffle(vw0123, vw0123, 2, 2, 2, 2)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi6x8ACE, wasm_v32x4_shuffle(vw0123, vw0123, 2, 2, 2, 2)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi1x8ACE, wasm_v32x4_shuffle(vw4567, vw4567, 1, 1, 1, 1)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi3x8ACE, wasm_v32x4_shuffle(vw4567, vw4567, 1, 1, 1, 1)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi5x8ACE, wasm_v32x4_shuffle(vw4567, vw4567, 1, 1, 1, 1)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi7x8ACE, wasm_v32x4_shuffle(vw4567, vw4567, 1, 1, 1, 1)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi2x8ACE, wasm_v32x4_shuffle(vw89, vw89, 0, 0, 0, 0)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi4x8ACE, wasm_v32x4_shuffle(vw89, vw89, 0, 0, 0, 0)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi6x8ACE, wasm_v32x4_shuffle(vw89, vw89, 0, 0, 0, 0)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi8x8ACE, wasm_v32x4_shuffle(vw89, vw89, 0, 0, 0, 0)));

      const v128_t vi0x79BD = wasm_v32x4_shuffle(vi0x1357, vi0x9BDF, 3, 4, 5, 6);
      const v128_t vi1x79BD = wasm_v32x4_shuffle(vi1x1357, vi1x9BDF, 3, 4, 5, 6);
      const v128_t vi2x79BD = wasm_v32x4_shuffle(vi2x1357, vi2x9BDF, 3, 4, 5, 6);
      const v128_t vi3x79BD = wasm_v32x4_shuffle(vi3x1357, vi3x9BDF, 3, 4, 5, 6);
      const v128_t vi4x79BD = wasm_v32x4_shuffle(vi4x1357, vi4x9BDF, 3, 4, 5, 6);
      const v128_t vi5x79BD = wasm_v32x4_shuffle(vi5x1357, vi5x9BDF, 3, 4, 5, 6);
      const v128_t vi6x79BD = wasm_v32x4_shuffle(vi6x1357, vi6x9BDF, 3, 4, 5, 6);
      const v128_t vi7x79BD = wasm_v32x4_shuffle(vi7x1357, vi7x9BDF, 3, 4, 5, 6);
      const v128_t vi8x79BD = wasm_v32x4_shuffle(vi8x1357, vi8x9BDF, 3, 4, 5, 6);

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi0x79BD, wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi2x79BD, wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi4x79BD, wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi6x79BD, wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi1x79BD, wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi3x79BD, wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi5x79BD, wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi7x79BD, wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi2x79BD, wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi4x79BD, wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi6x79BD, wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi8x79BD, wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi0x9BDF, wasm_v32x4_shuffle(vw0123, vw0123, 3, 3, 3, 3)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi2x9BDF, wasm_v32x4_shuffle(vw0123, vw0123, 3, 3, 3, 3)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi4x9BDF, wasm_v32x4_shuffle(vw0123, vw0123, 3, 3, 3, 3)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi6x9BDF, wasm_v32x4_shuffle(vw0123, vw0123, 3, 3, 3, 3)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi1x9BDF, wasm_v32x4_shuffle(vw4567, vw4567, 2, 2, 2, 2)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi3x9BDF, wasm_v32x4_shuffle(vw4567, vw4567, 2, 2, 2, 2)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi5x9BDF, wasm_v32x4_shuffle(vw4567, vw4567, 2, 2, 2, 2)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi7x9BDF, wasm_v32x4_shuffle(vw4567, vw4567, 2, 2, 2, 2)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi2x9BDF, wasm_v32x4_shuffle(vw89, vw89, 1, 1, 1, 1)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi4x9BDF, wasm_v32x4_shuffle(vw89, vw89, 1, 1, 1, 1)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi6x9BDF, wasm_v32x4_shuffle(vw89, vw89, 1, 1, 1, 1)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi8x9BDF, wasm_v32x4_shuffle(vw89, vw89, 1, 1, 1, 1)));


      v128_t vo0 = wasm_f32x4_pmax(vmin, vo0p0);
      v128_t vo1 = wasm_f32x4_pmax(vmin, vo1p0);
      v128_t vo2 = wasm_f32x4_pmax(vmin, vo2p0);
      v128_t vo3 = wasm_f32x4_pmax(vmin, vo3p0);
      vo0 = wasm_f32x4_pmin(vmax, vo0);
      vo1 = wasm_f32x4_pmin(vmax, vo1);
      vo2 = wasm_f32x4_pmin(vmax, vo2);
      vo3 = wasm_f32x4_pmin(vmax, vo3);

      w += 1 * sizeof(float);
      if (w & (8 * sizeof(float))) {
        wasm_v128_store(o3, vo3); o3 += 4;
        wasm_v128_store(o2, vo2); o2 += 4;
        wasm_v128_store(o1, vo1); o1 += 4;
        wasm_v128_store(o0, vo0); o0 += 4;
      } else {
        if (w & (4 * sizeof(float))) {
          wasm_v128_store64_lane(o3, vo3, 0);
          o3 += 2;
          wasm_v128_store64_lane(o2, vo2, 0);
          o2 += 2;
          wasm_v128_store64_lane(o1, vo1, 0);
          o1 += 2;
          wasm_v128_store64_lane(o0, vo0, 0);
          o0 += 2;

          vo0 = wasm_v64x2_shuffle(vo0, vo0, 1, 1);
          vo1 = wasm_v64x2_shuffle(vo1, vo1, 1, 1);
          vo2 = wasm_v64x2_shuffle(vo2, vo2, 1, 1);
          vo3 = wasm_v64x2_shuffle(vo3, vo3, 1, 1);
        }
        if (w & (2 * sizeof(float))) {
          wasm_v128_store32_lane(o3, vo3, 0);
          o3 += 1;
          wasm_v128_store32_lane(o2, vo2, 0);
          o2 += 1;
          wasm_v128_store32_lane(o1, vo1, 0);
          o1 += 1;
          wasm_v128_store32_lane(o0, vo0, 0);
          o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i8 - input_decrement);
    i1 = (const float*) ((uintptr_t) i0 + input_width);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);
    i5 = (const float*) ((uintptr_t) i4 + input_width);
    i6 = (const float*) ((uintptr_t) i5 + input_width);
    i7 = (const float*) ((uintptr_t) i6 + input_width);
    i8 = (const float*) ((uintptr_t) i7 + input_width);

    o0 = o3;
    o1 = (float*) ((uintptr_t) o0 + output_width);
    o2 = (float*) ((uintptr_t) o1 + output_width);
    o3 = (float*) ((uintptr_t) o2 + output_width);

    output_height = doz(output_height, 4);
    padded_input_height = doz(padded_input_height, 8);
  } while (output_height != 0);
}
