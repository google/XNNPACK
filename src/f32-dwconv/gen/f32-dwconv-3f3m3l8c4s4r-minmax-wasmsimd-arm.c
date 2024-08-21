// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/multipass-wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <wasm_simd128.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"


void xnn_f32_dwconv_minmax_ukernel_3f3m3l8c4s4r__wasmsimd_arm(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    size_t kernel_size,
    float* buffer,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 3);

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    const float* w = weights;

    // First pass to process 3 inputs.
    {
      float* b = buffer;
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
      input += 3;

      // Process c channels and write to buffer.
      size_t c = round_up_po2(channels, 4);
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


        wasm_v128_store(b, vacc0123p0);
        wasm_v128_store(b + 4, vacc4567p0);
        b += 8;
      }

      if (c != 0) {
        v128_t vacc0p0 = wasm_v128_load(w);


        const v128_t vi0x0123 = wasm_v128_load(i0);
        i0 += 4;

        const v128_t vk0x0123 = wasm_v128_load(w + 4);
        vacc0p0 = wasm_f32x4_add(wasm_f32x4_mul(vi0x0123, vk0x0123), vacc0p0);

        const v128_t vi1x0123 = wasm_v128_load(i1);
        i1 += 4;

        const v128_t vk1x0123 = wasm_v128_load(w + 8);
        vacc0p0 = wasm_f32x4_add(wasm_f32x4_mul(vi1x0123, vk1x0123), vacc0p0);

        const v128_t vi2x0123 = wasm_v128_load(i2);
        i2 += 4;

        const v128_t vk2x0123 = wasm_v128_load(w + 12);
        vacc0p0 = wasm_f32x4_add(wasm_f32x4_mul(vi2x0123, vk2x0123), vacc0p0);

        w += 16;


        wasm_v128_store(b, vacc0p0);
        b += 4;
      }
    }

    // Middle pass to process 3 inputs in each iteration.
    for (size_t ks = kernel_size - 3; ks > 3; ks -= 3) {
      float* b = buffer;
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
      input += 3;

      size_t c = round_up_po2(channels, 4);
      for (; c >= 8; c -= 8) {
        v128_t vacc0123p0 = wasm_v128_load(b);
        v128_t vacc4567p0 = wasm_v128_load(b + 4);


        const v128_t vi0x0123 = wasm_v128_load(i0);
        const v128_t vi0x4567 = wasm_v128_load(i0 + 4);
        i0 += 8;

        const v128_t vk0x0123 = wasm_v128_load(w);
        const v128_t vk0x4567 = wasm_v128_load(w + 4);
        vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi0x0123, vk0x0123), vacc0123p0);
        vacc4567p0 = wasm_f32x4_add(wasm_f32x4_mul(vi0x4567, vk0x4567), vacc4567p0);

        const v128_t vi1x0123 = wasm_v128_load(i1);
        const v128_t vi1x4567 = wasm_v128_load(i1 + 4);
        i1 += 8;

        const v128_t vk1x0123 = wasm_v128_load(w + 8);
        const v128_t vk1x4567 = wasm_v128_load(w + 12);
        vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi1x0123, vk1x0123), vacc0123p0);
        vacc4567p0 = wasm_f32x4_add(wasm_f32x4_mul(vi1x4567, vk1x4567), vacc4567p0);

        const v128_t vi2x0123 = wasm_v128_load(i2);
        const v128_t vi2x4567 = wasm_v128_load(i2 + 4);
        i2 += 8;

        const v128_t vk2x0123 = wasm_v128_load(w + 16);
        const v128_t vk2x4567 = wasm_v128_load(w + 20);
        vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi2x0123, vk2x0123), vacc0123p0);
        vacc4567p0 = wasm_f32x4_add(wasm_f32x4_mul(vi2x4567, vk2x4567), vacc4567p0);

        w += 24;


        wasm_v128_store(b, vacc0123p0);
        wasm_v128_store(b + 4, vacc4567p0);
        b += 8;
      }

      if (c != 0) {
        v128_t vacc0p0 = wasm_v128_load(b);


        const v128_t vi0x0123 = wasm_v128_load(i0);
        i0 += 4;

        const v128_t vk0x0123 = wasm_v128_load(w);
        vacc0p0 = wasm_f32x4_add(wasm_f32x4_mul(vi0x0123, vk0x0123), vacc0p0);

        const v128_t vi1x0123 = wasm_v128_load(i1);
        i1 += 4;

        const v128_t vk1x0123 = wasm_v128_load(w + 4);
        vacc0p0 = wasm_f32x4_add(wasm_f32x4_mul(vi1x0123, vk1x0123), vacc0p0);

        const v128_t vi2x0123 = wasm_v128_load(i2);
        i2 += 4;

        const v128_t vk2x0123 = wasm_v128_load(w + 8);
        vacc0p0 = wasm_f32x4_add(wasm_f32x4_mul(vi2x0123, vk2x0123), vacc0p0);

        w += 12;


        wasm_v128_store(b, vacc0p0);
        b += 4;
      }
    }

    // Last pass to process up to 3 inputs.
    {
      float* b = buffer;
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

      size_t c = channels;
      for (; c >= 8; c -= 8) {
        v128_t vacc0123p0 = wasm_v128_load(b);
        v128_t vacc4567p0 = wasm_v128_load(b + 4);
        b += 8;


        const v128_t vi0x0123 = wasm_v128_load(i0);
        const v128_t vi0x4567 = wasm_v128_load(i0 + 4);
        i0 += 8;

        v128_t vk0x0123 = wasm_v128_load(w);
        v128_t vk0x4567 = wasm_v128_load(w + 4);

        vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi0x0123, vk0x0123), vacc0123p0);
        vacc4567p0 = wasm_f32x4_add(wasm_f32x4_mul(vi0x4567, vk0x4567), vacc4567p0);

        const v128_t vi1x0123 = wasm_v128_load(i1);
        const v128_t vi1x4567 = wasm_v128_load(i1 + 4);
        i1 += 8;

        v128_t vk1x0123 = wasm_v128_load(w + 8);
        v128_t vk1x4567 = wasm_v128_load(w + 12);

        vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi1x0123, vk1x0123), vacc0123p0);
        vacc4567p0 = wasm_f32x4_add(wasm_f32x4_mul(vi1x4567, vk1x4567), vacc4567p0);

        const v128_t vi2x0123 = wasm_v128_load(i2);
        const v128_t vi2x4567 = wasm_v128_load(i2 + 4);
        i2 += 8;

        v128_t vk2x0123 = wasm_v128_load(w + 16);
        v128_t vk2x4567 = wasm_v128_load(w + 20);

        vacc0123p0 = wasm_f32x4_add(wasm_f32x4_mul(vi2x0123, vk2x0123), vacc0123p0);
        vacc4567p0 = wasm_f32x4_add(wasm_f32x4_mul(vi2x4567, vk2x4567), vacc4567p0);

        w += 24;


        v128_t vacc0123 = wasm_f32x4_max(vacc0123p0, vmin);
        v128_t vacc4567 = wasm_f32x4_max(vacc4567p0, vmin);

        vacc0123 = wasm_f32x4_min(vacc0123, vmax);
        vacc4567 = wasm_f32x4_min(vacc4567, vmax);

        wasm_v128_store(output, vacc0123);
        wasm_v128_store(output + 4, vacc4567);
        output += 8;
      }


      for (; c >= 4; c -= 4) {
        v128_t vacc0p0 = wasm_v128_load(b);
        b += 4;


        const v128_t vi0x0123 = wasm_v128_load(i0);
        i0 += 4;

        v128_t vk0x0123 = wasm_v128_load(w);

        vacc0p0 = wasm_f32x4_add(wasm_f32x4_mul(vi0x0123, vk0x0123), vacc0p0);

        const v128_t vi1x0123 = wasm_v128_load(i1);
        i1 += 4;

        v128_t vk1x0123 = wasm_v128_load(w + 4);

        vacc0p0 = wasm_f32x4_add(wasm_f32x4_mul(vi1x0123, vk1x0123), vacc0p0);

        const v128_t vi2x0123 = wasm_v128_load(i2);
        i2 += 4;

        v128_t vk2x0123 = wasm_v128_load(w + 8);

        vacc0p0 = wasm_f32x4_add(wasm_f32x4_mul(vi2x0123, vk2x0123), vacc0p0);

        w += 12;



        v128_t vacc0 = wasm_f32x4_max(vacc0p0, vmin);

        vacc0 = wasm_f32x4_min(vacc0, vmax);

        wasm_v128_store(output, vacc0);
        output += 4;
      }

      if XNN_UNLIKELY(c != 0) {
        v128_t vacc0p0 = wasm_v128_load(b);

        const v128_t vi0x0123 = wasm_v128_load(i0);
        v128_t vk0x0123 = wasm_v128_load(w);
        vacc0p0 = wasm_f32x4_add(wasm_f32x4_mul(vi0x0123, vk0x0123), vacc0p0);

        const v128_t vi1x0123 = wasm_v128_load(i1);
        v128_t vk1x0123 = wasm_v128_load(w + 4);
        vacc0p0 = wasm_f32x4_add(wasm_f32x4_mul(vi1x0123, vk1x0123), vacc0p0);

        const v128_t vi2x0123 = wasm_v128_load(i2);
        v128_t vk2x0123 = wasm_v128_load(w + 8);
        vacc0p0 = wasm_f32x4_add(wasm_f32x4_mul(vi2x0123, vk2x0123), vacc0p0);


        v128_t vacc0 = wasm_f32x4_max(vacc0p0, vmin);

        vacc0 = wasm_f32x4_min(vacc0, vmax);

        if (c & 2) {
          wasm_v128_store64_lane(output, vacc0, 0);
          vacc0 = wasm_v64x2_shuffle(vacc0, vacc0, 1, 1);
          output += 2;
        }
        if (c & 1) {
          wasm_v128_store32_lane(output, vacc0, 0);
          output += 1;
        }
      }

    }
    input = (const float**) ((uintptr_t) input + input_stride);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
