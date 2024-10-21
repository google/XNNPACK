// Auto-generated file. Do not edit!
//   Template: src/f32-prelu/wasmsimd-iminmax.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/math.h"
#include "xnnpack/prelu.h"


void xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x16(
    size_t rows,
    size_t channels,
    const float* restrict input,
    size_t input_stride,
    const float* restrict weights,
    float* restrict output,
    size_t output_stride) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const float* i0 = input;
  float* o0 = output;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_stride);
  float* o2 = (float*) ((uintptr_t) o1 + output_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_stride);
  float* o3 = (float*) ((uintptr_t) o2 + output_stride);

  const size_t input_increment = input_stride * 4 - channels;
  const size_t output_increment = output_stride * 4 - channels;

  const v128_t vzero = wasm_i32x4_const_splat(0);
  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(rows <= 2) {
      i2 = i1;
      o2 = o1;
    }
    if XNN_UNPREDICTABLE(rows < 4) {
      i3 = i2;
      o3 = o2;
    }

    const float* w = weights;
    size_t c = channels;
    for (; c >= 16 * sizeof(float); c -= 16 * sizeof(float)) {
      const v128_t vw0123 = wasm_v128_load(w);
      const v128_t vw4567 = wasm_v128_load(w + 4);
      const v128_t vw89AB = wasm_v128_load(w + 8);
      const v128_t vwCDEF = wasm_v128_load(w + 12);
      w += 16;

      v128_t vi0x0123 = wasm_v128_load(i0);
      v128_t vi0x4567 = wasm_v128_load(i0 + 4);
      v128_t vi0x89AB = wasm_v128_load(i0 + 8);
      v128_t vi0xCDEF = wasm_v128_load(i0 + 12);
      i0 += 16;
      v128_t vi1x0123 = wasm_v128_load(i1);
      v128_t vi1x4567 = wasm_v128_load(i1 + 4);
      v128_t vi1x89AB = wasm_v128_load(i1 + 8);
      v128_t vi1xCDEF = wasm_v128_load(i1 + 12);
      i1 += 16;
      v128_t vi2x0123 = wasm_v128_load(i2);
      v128_t vi2x4567 = wasm_v128_load(i2 + 4);
      v128_t vi2x89AB = wasm_v128_load(i2 + 8);
      v128_t vi2xCDEF = wasm_v128_load(i2 + 12);
      i2 += 16;
      v128_t vi3x0123 = wasm_v128_load(i3);
      v128_t vi3x4567 = wasm_v128_load(i3 + 4);
      v128_t vi3x89AB = wasm_v128_load(i3 + 8);
      v128_t vi3xCDEF = wasm_v128_load(i3 + 12);
      i3 += 16;

      v128_t vacc0x0123 = wasm_i32x4_max(vi0x0123, vzero);
      vi0x0123 = wasm_i32x4_min(vi0x0123, vzero);
      v128_t vacc0x4567 = wasm_i32x4_max(vi0x4567, vzero);
      vi0x4567 = wasm_i32x4_min(vi0x4567, vzero);
      v128_t vacc0x89AB = wasm_i32x4_max(vi0x89AB, vzero);
      vi0x89AB = wasm_i32x4_min(vi0x89AB, vzero);
      v128_t vacc0xCDEF = wasm_i32x4_max(vi0xCDEF, vzero);
      vi0xCDEF = wasm_i32x4_min(vi0xCDEF, vzero);
      v128_t vacc1x0123 = wasm_i32x4_max(vi1x0123, vzero);
      vi1x0123 = wasm_i32x4_min(vi1x0123, vzero);
      v128_t vacc1x4567 = wasm_i32x4_max(vi1x4567, vzero);
      vi1x4567 = wasm_i32x4_min(vi1x4567, vzero);
      v128_t vacc1x89AB = wasm_i32x4_max(vi1x89AB, vzero);
      vi1x89AB = wasm_i32x4_min(vi1x89AB, vzero);
      v128_t vacc1xCDEF = wasm_i32x4_max(vi1xCDEF, vzero);
      vi1xCDEF = wasm_i32x4_min(vi1xCDEF, vzero);
      v128_t vacc2x0123 = wasm_i32x4_max(vi2x0123, vzero);
      vi2x0123 = wasm_i32x4_min(vi2x0123, vzero);
      v128_t vacc2x4567 = wasm_i32x4_max(vi2x4567, vzero);
      vi2x4567 = wasm_i32x4_min(vi2x4567, vzero);
      v128_t vacc2x89AB = wasm_i32x4_max(vi2x89AB, vzero);
      vi2x89AB = wasm_i32x4_min(vi2x89AB, vzero);
      v128_t vacc2xCDEF = wasm_i32x4_max(vi2xCDEF, vzero);
      vi2xCDEF = wasm_i32x4_min(vi2xCDEF, vzero);
      v128_t vacc3x0123 = wasm_i32x4_max(vi3x0123, vzero);
      vi3x0123 = wasm_i32x4_min(vi3x0123, vzero);
      v128_t vacc3x4567 = wasm_i32x4_max(vi3x4567, vzero);
      vi3x4567 = wasm_i32x4_min(vi3x4567, vzero);
      v128_t vacc3x89AB = wasm_i32x4_max(vi3x89AB, vzero);
      vi3x89AB = wasm_i32x4_min(vi3x89AB, vzero);
      v128_t vacc3xCDEF = wasm_i32x4_max(vi3xCDEF, vzero);
      vi3xCDEF = wasm_i32x4_min(vi3xCDEF, vzero);

      vacc0x0123 = wasm_f32x4_relaxed_madd(vi0x0123, vw0123, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(vi0x4567, vw4567, vacc0x4567);
      vacc0x89AB = wasm_f32x4_relaxed_madd(vi0x89AB, vw89AB, vacc0x89AB);
      vacc0xCDEF = wasm_f32x4_relaxed_madd(vi0xCDEF, vwCDEF, vacc0xCDEF);
      vacc1x0123 = wasm_f32x4_relaxed_madd(vi1x0123, vw0123, vacc1x0123);
      vacc1x4567 = wasm_f32x4_relaxed_madd(vi1x4567, vw4567, vacc1x4567);
      vacc1x89AB = wasm_f32x4_relaxed_madd(vi1x89AB, vw89AB, vacc1x89AB);
      vacc1xCDEF = wasm_f32x4_relaxed_madd(vi1xCDEF, vwCDEF, vacc1xCDEF);
      vacc2x0123 = wasm_f32x4_relaxed_madd(vi2x0123, vw0123, vacc2x0123);
      vacc2x4567 = wasm_f32x4_relaxed_madd(vi2x4567, vw4567, vacc2x4567);
      vacc2x89AB = wasm_f32x4_relaxed_madd(vi2x89AB, vw89AB, vacc2x89AB);
      vacc2xCDEF = wasm_f32x4_relaxed_madd(vi2xCDEF, vwCDEF, vacc2xCDEF);
      vacc3x0123 = wasm_f32x4_relaxed_madd(vi3x0123, vw0123, vacc3x0123);
      vacc3x4567 = wasm_f32x4_relaxed_madd(vi3x4567, vw4567, vacc3x4567);
      vacc3x89AB = wasm_f32x4_relaxed_madd(vi3x89AB, vw89AB, vacc3x89AB);
      vacc3xCDEF = wasm_f32x4_relaxed_madd(vi3xCDEF, vwCDEF, vacc3xCDEF);

      wasm_v128_store(o0, vacc0x0123);
      wasm_v128_store(o0 + 4, vacc0x4567);
      wasm_v128_store(o0 + 8, vacc0x89AB);
      wasm_v128_store(o0 + 12, vacc0xCDEF);
      o0 += 16;
      wasm_v128_store(o1, vacc1x0123);
      wasm_v128_store(o1 + 4, vacc1x4567);
      wasm_v128_store(o1 + 8, vacc1x89AB);
      wasm_v128_store(o1 + 12, vacc1xCDEF);
      o1 += 16;
      wasm_v128_store(o2, vacc2x0123);
      wasm_v128_store(o2 + 4, vacc2x4567);
      wasm_v128_store(o2 + 8, vacc2x89AB);
      wasm_v128_store(o2 + 12, vacc2xCDEF);
      o2 += 16;
      wasm_v128_store(o3, vacc3x0123);
      wasm_v128_store(o3 + 4, vacc3x4567);
      wasm_v128_store(o3 + 8, vacc3x89AB);
      wasm_v128_store(o3 + 12, vacc3xCDEF);
      o3 += 16;
    }
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const v128_t vw0123 = wasm_v128_load(w);
      w += 4;

      v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;
      v128_t vi1x0123 = wasm_v128_load(i1);
      i1 += 4;
      v128_t vi2x0123 = wasm_v128_load(i2);
      i2 += 4;
      v128_t vi3x0123 = wasm_v128_load(i3);
      i3 += 4;

      v128_t vacc0x0123 = wasm_i32x4_max(vi0x0123, vzero);
      vi0x0123 = wasm_i32x4_min(vi0x0123, vzero);
      v128_t vacc1x0123 = wasm_i32x4_max(vi1x0123, vzero);
      vi1x0123 = wasm_i32x4_min(vi1x0123, vzero);
      v128_t vacc2x0123 = wasm_i32x4_max(vi2x0123, vzero);
      vi2x0123 = wasm_i32x4_min(vi2x0123, vzero);
      v128_t vacc3x0123 = wasm_i32x4_max(vi3x0123, vzero);
      vi3x0123 = wasm_i32x4_min(vi3x0123, vzero);

      vacc0x0123 = wasm_f32x4_relaxed_madd(vi0x0123, vw0123, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(vi1x0123, vw0123, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(vi2x0123, vw0123, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(vi3x0123, vw0123, vacc3x0123);

      wasm_v128_store(o0, vacc0x0123);
      o0 += 4;
      wasm_v128_store(o1, vacc1x0123);
      o1 += 4;
      wasm_v128_store(o2, vacc2x0123);
      o2 += 4;
      wasm_v128_store(o3, vacc3x0123);
      o3 += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      const v128_t vw0123 = wasm_v128_load(w);
      w = (const float*) ((uintptr_t) w + c);

      v128_t vi0x0123 = wasm_v128_load(i0);
      i0 = (const float*) ((uintptr_t) i0 + c);
      v128_t vi1x0123 = wasm_v128_load(i1);
      i1 = (const float*) ((uintptr_t) i1 + c);
      v128_t vi2x0123 = wasm_v128_load(i2);
      i2 = (const float*) ((uintptr_t) i2 + c);
      v128_t vi3x0123 = wasm_v128_load(i3);
      i3 = (const float*) ((uintptr_t) i3 + c);

      v128_t vacc0x0123 = wasm_i32x4_max(vi0x0123, vzero);
      vi0x0123 = wasm_i32x4_min(vi0x0123, vzero);
      v128_t vacc1x0123 = wasm_i32x4_max(vi1x0123, vzero);
      vi1x0123 = wasm_i32x4_min(vi1x0123, vzero);
      v128_t vacc2x0123 = wasm_i32x4_max(vi2x0123, vzero);
      vi2x0123 = wasm_i32x4_min(vi2x0123, vzero);
      v128_t vacc3x0123 = wasm_i32x4_max(vi3x0123, vzero);
      vi3x0123 = wasm_i32x4_min(vi3x0123, vzero);

      vacc0x0123 = wasm_f32x4_relaxed_madd(vi0x0123, vw0123, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(vi1x0123, vw0123, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(vi2x0123, vw0123, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(vi3x0123, vw0123, vacc3x0123);

      if (c & (2 * sizeof(float))) {
        wasm_v128_store64_lane(o0, vacc0x0123, 0);
        wasm_v128_store64_lane(o1, vacc1x0123, 0);
        wasm_v128_store64_lane(o2, vacc2x0123, 0);
        wasm_v128_store64_lane(o3, vacc3x0123, 0);

        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);

        o0 += 2;
        o1 += 2;
        o2 += 2;
        o3 += 2;
      }
      if (c & (1 * sizeof(float))) {
        wasm_v128_store32_lane(o0, vacc0x0123, 0);
        wasm_v128_store32_lane(o1, vacc1x0123, 0);
        wasm_v128_store32_lane(o2, vacc2x0123, 0);
        wasm_v128_store32_lane(o3, vacc3x0123, 0);

        o0 += 1;
        o1 += 1;
        o2 += 1;
        o3 += 1;
      }
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    i2 = (const float*) ((uintptr_t) i2 + input_increment);
    o2 = (float*) ((uintptr_t) o2 + output_increment);
    i3 = (const float*) ((uintptr_t) i3 + input_increment);
    o3 = (float*) ((uintptr_t) o3 + output_increment);
    rows = doz(rows, 4);
  } while (rows != 0);
}
