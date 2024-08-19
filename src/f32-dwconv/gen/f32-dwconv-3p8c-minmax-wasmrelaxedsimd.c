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

#include "xnnpack/dwconv.h"


void xnn_f32_dwconv_minmax_ukernel_3p8c__wasmrelaxedsimd(
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

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
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

      w += 32;


      v128_t vacc0123 = wasm_f32x4_relaxed_max(vmin, vacc0123p0);
      v128_t vacc4567 = wasm_f32x4_relaxed_max(vmin, vacc4567p0);

      vacc0123 = wasm_f32x4_relaxed_min(vmax, vacc0123);
      vacc4567 = wasm_f32x4_relaxed_min(vmax, vacc4567);

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

      w += 4;


      v128_t vacc0123 = wasm_f32x4_relaxed_max(vmin, vacc0123p0);
      vacc0123 = wasm_f32x4_relaxed_min(vmax, vacc0123);

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


      v128_t vacc0123 = wasm_f32x4_relaxed_max(vmin, vacc0123p0);
      vacc0123 = wasm_f32x4_relaxed_min(vmax, vacc0123);

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
