// Auto-generated file. Do not edit!
//   Template: src/f32-prelu/wasmsimd-laneselect.c.in
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


void xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x4(
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
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const v128_t vw0123 = wasm_v128_load(w);
      w += 4;

      const v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;
      const v128_t vi1x0123 = wasm_v128_load(i1);
      i1 += 4;
      const v128_t vi2x0123 = wasm_v128_load(i2);
      i2 += 4;
      const v128_t vi3x0123 = wasm_v128_load(i3);
      i3 += 4;

      v128_t vacc0x0123 = wasm_f32x4_mul(vi0x0123, vw0123);
      const v128_t vmask0x0123 = wasm_i32x4_shr(vi0x0123, 31);
      v128_t vacc1x0123 = wasm_f32x4_mul(vi1x0123, vw0123);
      const v128_t vmask1x0123 = wasm_i32x4_shr(vi1x0123, 31);
      v128_t vacc2x0123 = wasm_f32x4_mul(vi2x0123, vw0123);
      const v128_t vmask2x0123 = wasm_i32x4_shr(vi2x0123, 31);
      v128_t vacc3x0123 = wasm_f32x4_mul(vi3x0123, vw0123);
      const v128_t vmask3x0123 = wasm_i32x4_shr(vi3x0123, 31);

      vacc0x0123 = wasm_v128_bitselect(vacc0x0123, vi0x0123, vmask0x0123);
      vacc1x0123 = wasm_v128_bitselect(vacc1x0123, vi1x0123, vmask1x0123);
      vacc2x0123 = wasm_v128_bitselect(vacc2x0123, vi2x0123, vmask2x0123);
      vacc3x0123 = wasm_v128_bitselect(vacc3x0123, vi3x0123, vmask3x0123);

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

      const v128_t vi0x0123 = wasm_v128_load(i0);
      i0 = (const float*) ((uintptr_t) i0 + c);
      const v128_t vi1x0123 = wasm_v128_load(i1);
      i1 = (const float*) ((uintptr_t) i1 + c);
      const v128_t vi2x0123 = wasm_v128_load(i2);
      i2 = (const float*) ((uintptr_t) i2 + c);
      const v128_t vi3x0123 = wasm_v128_load(i3);
      i3 = (const float*) ((uintptr_t) i3 + c);

      v128_t vacc0x0123 = wasm_f32x4_mul(vi0x0123, vw0123);
      const v128_t vmask0x0123 = wasm_i32x4_shr(vi0x0123, 31);
      v128_t vacc1x0123 = wasm_f32x4_mul(vi1x0123, vw0123);
      const v128_t vmask1x0123 = wasm_i32x4_shr(vi1x0123, 31);
      v128_t vacc2x0123 = wasm_f32x4_mul(vi2x0123, vw0123);
      const v128_t vmask2x0123 = wasm_i32x4_shr(vi2x0123, 31);
      v128_t vacc3x0123 = wasm_f32x4_mul(vi3x0123, vw0123);
      const v128_t vmask3x0123 = wasm_i32x4_shr(vi3x0123, 31);

      vacc0x0123 = wasm_v128_bitselect(vacc0x0123, vi0x0123, vmask0x0123);
      vacc1x0123 = wasm_v128_bitselect(vacc1x0123, vi1x0123, vmask1x0123);
      vacc2x0123 = wasm_v128_bitselect(vacc2x0123, vi2x0123, vmask2x0123);
      vacc3x0123 = wasm_v128_bitselect(vacc3x0123, vi3x0123, vmask3x0123);

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
