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

#include <xnnpack/math.h>
#include <xnnpack/prelu.h>


void xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x16(
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

  const size_t input_increment = input_stride * 1 - channels;
  const size_t output_increment = output_stride * 1 - channels;

  const v128_t vzero = wasm_i32x4_const_splat(0);
  do {

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

      v128_t vacc0x0123 = wasm_i32x4_max(vi0x0123, vzero);
      vi0x0123 = wasm_i32x4_min(vi0x0123, vzero);
      v128_t vacc0x4567 = wasm_i32x4_max(vi0x4567, vzero);
      vi0x4567 = wasm_i32x4_min(vi0x4567, vzero);
      v128_t vacc0x89AB = wasm_i32x4_max(vi0x89AB, vzero);
      vi0x89AB = wasm_i32x4_min(vi0x89AB, vzero);
      v128_t vacc0xCDEF = wasm_i32x4_max(vi0xCDEF, vzero);
      vi0xCDEF = wasm_i32x4_min(vi0xCDEF, vzero);

      vacc0x0123 = wasm_f32x4_add(wasm_f32x4_mul(vi0x0123, vw0123), vacc0x0123);
      vacc0x4567 = wasm_f32x4_add(wasm_f32x4_mul(vi0x4567, vw4567), vacc0x4567);
      vacc0x89AB = wasm_f32x4_add(wasm_f32x4_mul(vi0x89AB, vw89AB), vacc0x89AB);
      vacc0xCDEF = wasm_f32x4_add(wasm_f32x4_mul(vi0xCDEF, vwCDEF), vacc0xCDEF);

      wasm_v128_store(o0, vacc0x0123);
      wasm_v128_store(o0 + 4, vacc0x4567);
      wasm_v128_store(o0 + 8, vacc0x89AB);
      wasm_v128_store(o0 + 12, vacc0xCDEF);
      o0 += 16;
    }
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const v128_t vw0123 = wasm_v128_load(w);
      w += 4;

      v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;

      v128_t vacc0x0123 = wasm_i32x4_max(vi0x0123, vzero);
      vi0x0123 = wasm_i32x4_min(vi0x0123, vzero);

      vacc0x0123 = wasm_f32x4_add(wasm_f32x4_mul(vi0x0123, vw0123), vacc0x0123);

      wasm_v128_store(o0, vacc0x0123);
      o0 += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      const v128_t vw0123 = wasm_v128_load(w);
      w = (const float*) ((uintptr_t) w + c);

      v128_t vi0x0123 = wasm_v128_load(i0);
      i0 = (const float*) ((uintptr_t) i0 + c);

      v128_t vacc0x0123 = wasm_i32x4_max(vi0x0123, vzero);
      vi0x0123 = wasm_i32x4_min(vi0x0123, vzero);

      vacc0x0123 = wasm_f32x4_add(wasm_f32x4_mul(vi0x0123, vw0123), vacc0x0123);

      if (c & (2 * sizeof(float))) {
        wasm_v128_store64_lane(o0, vacc0x0123, 0);

        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        o0 += 2;
      }
      if (c & (1 * sizeof(float))) {
        wasm_v128_store32_lane(o0, vacc0x0123, 0);

        o0 += 1;
      }
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    rows = doz(rows, 1);
  } while (rows != 0);
}
