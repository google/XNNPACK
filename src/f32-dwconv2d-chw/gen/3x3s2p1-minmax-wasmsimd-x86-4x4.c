// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv2d-chw/3x3s2p1-wasmsimd.c.in
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


void xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_4x4(
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
  assert(padding_top >= 0);
  assert(padding_top <= 1);

  const v128_t vmask_even = wasm_v128_load(params->scalar.mask_even);
  const v128_t vmask_odd  = wasm_v128_load(params->scalar.mask_odd);
  const v128_t vmax = wasm_v32x4_load_splat(&params->scalar.max);
  const v128_t vmin = wasm_v32x4_load_splat(&params->scalar.min);

  const v128_t vw0123 = wasm_v128_load(weights);
  const v128_t vw4567 = wasm_v128_load(weights + 4);
  const v128_t vw89 = wasm_v64x2_load_splat(weights + 8);

  const v128_t vzero = wasm_f32x4_splat(0.0f);

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

    v128_t vi0x1357 = vzero;
    v128_t vi1x1357 = vzero;
    v128_t vi2x1357 = vzero;
    v128_t vi3x1357 = vzero;
    v128_t vi4x1357 = vzero;
    v128_t vi5x1357 = vzero;
    v128_t vi6x1357 = vzero;
    v128_t vi7x1357 = vzero;
    v128_t vi8x1357 = vzero;

    size_t w = input_width;
    for (; w >= 8 * sizeof(float); w -= 8 * sizeof(float)) {
      v128_t vo0p0 = wasm_v32x4_shuffle(vw0123, vw0123, 0, 0, 0, 0);
      v128_t vo1p0 = wasm_v32x4_shuffle(vw0123, vw0123, 0, 0, 0, 0);
      v128_t vo2p0 = wasm_v32x4_shuffle(vw0123, vw0123, 0, 0, 0, 0);
      v128_t vo3p0 = wasm_v32x4_shuffle(vw0123, vw0123, 0, 0, 0, 0);

      const v128_t vi0x8ACE = wasm_v128_load(i0);
      const v128_t vi0x9BDF = wasm_v128_load(i0 + 4);
      i0 += 8;
      const v128_t vi1x8ACE = wasm_v128_load(i1);
      const v128_t vi1x9BDF = wasm_v128_load(i1 + 4);
      i1 += 8;
      const v128_t vi2x8ACE = wasm_v128_load(i2);
      const v128_t vi2x9BDF = wasm_v128_load(i2 + 4);
      i2 += 8;
      const v128_t vi3x8ACE = wasm_v128_load(i3);
      const v128_t vi3x9BDF = wasm_v128_load(i3 + 4);
      i3 += 8;
      const v128_t vi4x8ACE = wasm_v128_load(i4);
      const v128_t vi4x9BDF = wasm_v128_load(i4 + 4);
      i4 += 8;
      const v128_t vi5x8ACE = wasm_v128_load(i5);
      const v128_t vi5x9BDF = wasm_v128_load(i5 + 4);
      i5 += 8;
      const v128_t vi6x8ACE = wasm_v128_load(i6);
      const v128_t vi6x9BDF = wasm_v128_load(i6 + 4);
      i6 += 8;
      const v128_t vi7x8ACE = wasm_v128_load(i7);
      const v128_t vi7x9BDF = wasm_v128_load(i7 + 4);
      i7 += 8;
      const v128_t vi8x8ACE = wasm_v128_load(i8);
      const v128_t vi8x9BDF = wasm_v128_load(i8 + 4);
      i8 += 8;

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

      const v128_t vi0x7BDF = wasm_v32x4_shuffle(vi0x1357, vi0x9BDF, 3, 4, 5, 6);
      vi0x1357 = vi0x9BDF;
      const v128_t vi1x7BDF = wasm_v32x4_shuffle(vi1x1357, vi1x9BDF, 3, 4, 5, 6);
      vi1x1357 = vi1x9BDF;
      const v128_t vi2x7BDF = wasm_v32x4_shuffle(vi2x1357, vi2x9BDF, 3, 4, 5, 6);
      vi2x1357 = vi2x9BDF;
      const v128_t vi3x7BDF = wasm_v32x4_shuffle(vi3x1357, vi3x9BDF, 3, 4, 5, 6);
      vi3x1357 = vi3x9BDF;
      const v128_t vi4x7BDF = wasm_v32x4_shuffle(vi4x1357, vi4x9BDF, 3, 4, 5, 6);
      vi4x1357 = vi4x9BDF;
      const v128_t vi5x7BDF = wasm_v32x4_shuffle(vi5x1357, vi5x9BDF, 3, 4, 5, 6);
      vi5x1357 = vi5x9BDF;
      const v128_t vi6x7BDF = wasm_v32x4_shuffle(vi6x1357, vi6x9BDF, 3, 4, 5, 6);
      vi6x1357 = vi6x9BDF;
      const v128_t vi7x7BDF = wasm_v32x4_shuffle(vi7x1357, vi7x9BDF, 3, 4, 5, 6);
      vi7x1357 = vi7x9BDF;
      const v128_t vi8x7BDF = wasm_v32x4_shuffle(vi8x1357, vi8x9BDF, 3, 4, 5, 6);
      vi8x1357 = vi8x9BDF;

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi0x7BDF, wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi2x7BDF, wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi4x7BDF, wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi6x7BDF, wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi1x7BDF, wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi3x7BDF, wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi5x7BDF, wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi7x7BDF, wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi2x7BDF, wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi4x7BDF, wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi6x7BDF, wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi8x7BDF, wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3)));

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


      v128_t vo0 = wasm_v128_bitselect(vmin, vo0p0, wasm_f32x4_lt(vo0p0, vmin));
      v128_t vo1 = wasm_v128_bitselect(vmin, vo1p0, wasm_f32x4_lt(vo1p0, vmin));
      v128_t vo2 = wasm_v128_bitselect(vmin, vo2p0, wasm_f32x4_lt(vo2p0, vmin));
      v128_t vo3 = wasm_v128_bitselect(vmin, vo3p0, wasm_f32x4_lt(vo3p0, vmin));
      vo0 = wasm_v128_bitselect(vo0, vmax, wasm_f32x4_le(vo0, vmax));
      vo1 = wasm_v128_bitselect(vo1, vmax, wasm_f32x4_le(vo1, vmax));
      vo2 = wasm_v128_bitselect(vo2, vmax, wasm_f32x4_le(vo2, vmax));
      vo3 = wasm_v128_bitselect(vo3, vmax, wasm_f32x4_le(vo3, vmax));

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

      v128_t vi0x8ACE = wasm_v128_load(i0);
      v128_t vi0x9BDF = wasm_v128_load(i0 + 4);
      v128_t vi1x8ACE = wasm_v128_load(i1);
      v128_t vi1x9BDF = wasm_v128_load(i1 + 4);
      v128_t vi2x8ACE = wasm_v128_load(i2);
      v128_t vi2x9BDF = wasm_v128_load(i2 + 4);
      v128_t vi3x8ACE = wasm_v128_load(i3);
      v128_t vi3x9BDF = wasm_v128_load(i3 + 4);
      v128_t vi4x8ACE = wasm_v128_load(i4);
      v128_t vi4x9BDF = wasm_v128_load(i4 + 4);
      v128_t vi5x8ACE = wasm_v128_load(i5);
      v128_t vi5x9BDF = wasm_v128_load(i5 + 4);
      v128_t vi6x8ACE = wasm_v128_load(i6);
      v128_t vi6x9BDF = wasm_v128_load(i6 + 4);
      v128_t vi7x8ACE = wasm_v128_load(i7);
      v128_t vi7x9BDF = wasm_v128_load(i7 + 4);
      v128_t vi8x8ACE = wasm_v128_load(i8);
      v128_t vi8x9BDF = wasm_v128_load(i8 + 4);

      vi0x8ACE = wasm_v128_and(vmask_even, vi0x8ACE);
      vi0x9BDF = wasm_v128_and(vmask_odd,  vi0x9BDF);
      vi1x8ACE = wasm_v128_and(vmask_even, vi1x8ACE);
      vi1x9BDF = wasm_v128_and(vmask_odd,  vi1x9BDF);
      vi2x8ACE = wasm_v128_and(vmask_even, vi2x8ACE);
      vi2x9BDF = wasm_v128_and(vmask_odd,  vi2x9BDF);
      vi3x8ACE = wasm_v128_and(vmask_even, vi3x8ACE);
      vi3x9BDF = wasm_v128_and(vmask_odd,  vi3x9BDF);
      vi4x8ACE = wasm_v128_and(vmask_even, vi4x8ACE);
      vi4x9BDF = wasm_v128_and(vmask_odd,  vi4x9BDF);
      vi5x8ACE = wasm_v128_and(vmask_even, vi5x8ACE);
      vi5x9BDF = wasm_v128_and(vmask_odd,  vi5x9BDF);
      vi6x8ACE = wasm_v128_and(vmask_even, vi6x8ACE);
      vi6x9BDF = wasm_v128_and(vmask_odd,  vi6x9BDF);
      vi7x8ACE = wasm_v128_and(vmask_even, vi7x8ACE);
      vi7x9BDF = wasm_v128_and(vmask_odd,  vi7x9BDF);
      vi8x8ACE = wasm_v128_and(vmask_even, vi8x8ACE);
      vi8x9BDF = wasm_v128_and(vmask_odd,  vi8x9BDF);

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

      const v128_t vi0x7BDF = wasm_v32x4_shuffle(vi0x1357, vi0x9BDF, 3, 4, 5, 6);
      const v128_t vi1x7BDF = wasm_v32x4_shuffle(vi1x1357, vi1x9BDF, 3, 4, 5, 6);
      const v128_t vi2x7BDF = wasm_v32x4_shuffle(vi2x1357, vi2x9BDF, 3, 4, 5, 6);
      const v128_t vi3x7BDF = wasm_v32x4_shuffle(vi3x1357, vi3x9BDF, 3, 4, 5, 6);
      const v128_t vi4x7BDF = wasm_v32x4_shuffle(vi4x1357, vi4x9BDF, 3, 4, 5, 6);
      const v128_t vi5x7BDF = wasm_v32x4_shuffle(vi5x1357, vi5x9BDF, 3, 4, 5, 6);
      const v128_t vi6x7BDF = wasm_v32x4_shuffle(vi6x1357, vi6x9BDF, 3, 4, 5, 6);
      const v128_t vi7x7BDF = wasm_v32x4_shuffle(vi7x1357, vi7x9BDF, 3, 4, 5, 6);
      const v128_t vi8x7BDF = wasm_v32x4_shuffle(vi8x1357, vi8x9BDF, 3, 4, 5, 6);

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi0x7BDF, wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi2x7BDF, wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi4x7BDF, wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi6x7BDF, wasm_v32x4_shuffle(vw0123, vw0123, 1, 1, 1, 1)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi1x7BDF, wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi3x7BDF, wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi5x7BDF, wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi7x7BDF, wasm_v32x4_shuffle(vw4567, vw4567, 0, 0, 0, 0)));

      vo0p0 = wasm_f32x4_add(vo0p0, wasm_f32x4_mul(vi2x7BDF, wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3)));
      vo1p0 = wasm_f32x4_add(vo1p0, wasm_f32x4_mul(vi4x7BDF, wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3)));
      vo2p0 = wasm_f32x4_add(vo2p0, wasm_f32x4_mul(vi6x7BDF, wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3)));
      vo3p0 = wasm_f32x4_add(vo3p0, wasm_f32x4_mul(vi8x7BDF, wasm_v32x4_shuffle(vw4567, vw4567, 3, 3, 3, 3)));

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


      v128_t vo0 = wasm_v128_bitselect(vmin, vo0p0, wasm_f32x4_lt(vo0p0, vmin));
      v128_t vo1 = wasm_v128_bitselect(vmin, vo1p0, wasm_f32x4_lt(vo1p0, vmin));
      v128_t vo2 = wasm_v128_bitselect(vmin, vo2p0, wasm_f32x4_lt(vo2p0, vmin));
      v128_t vo3 = wasm_v128_bitselect(vmin, vo3p0, wasm_f32x4_lt(vo3p0, vmin));
      vo0 = wasm_v128_bitselect(vo0, vmax, wasm_f32x4_le(vo0, vmax));
      vo1 = wasm_v128_bitselect(vo1, vmax, wasm_f32x4_le(vo1, vmax));
      vo2 = wasm_v128_bitselect(vo2, vmax, wasm_f32x4_le(vo2, vmax));
      vo3 = wasm_v128_bitselect(vo3, vmax, wasm_f32x4_le(vo3, vmax));

      w += 1 * sizeof(float);
      if (w & (8 * sizeof(float))) {
        wasm_v128_store(o3, vo3); o3 += 4;
        wasm_v128_store(o2, vo2); o2 += 4;
        wasm_v128_store(o1, vo1); o1 += 4;
        wasm_v128_store(o0, vo0); o0 += 4;
      } else {
        if (w & (4 * sizeof(float))) {
          *((double*) o3) = wasm_f64x2_extract_lane(vo3, 0); o3 += 2;
          *((double*) o2) = wasm_f64x2_extract_lane(vo2, 0); o2 += 2;
          *((double*) o1) = wasm_f64x2_extract_lane(vo1, 0); o1 += 2;
          *((double*) o0) = wasm_f64x2_extract_lane(vo0, 0); o0 += 2;

          vo0 = wasm_v32x4_shuffle(vo0, vo0, 2, 3, 0, 1);
          vo1 = wasm_v32x4_shuffle(vo1, vo1, 2, 3, 0, 1);
          vo2 = wasm_v32x4_shuffle(vo2, vo2, 2, 3, 0, 1);
          vo3 = wasm_v32x4_shuffle(vo3, vo3, 2, 3, 0, 1);
        }
        if (w & (2 * sizeof(float))) {
          *o3 = wasm_f32x4_extract_lane(vo3, 0); o3 += 1;
          *o2 = wasm_f32x4_extract_lane(vo2, 0); o2 += 1;
          *o1 = wasm_f32x4_extract_lane(vo1, 0); o1 += 1;
          *o0 = wasm_f32x4_extract_lane(vo0, 0); o0 += 1;
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
