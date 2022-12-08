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

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>


void xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__wasmsimd_x86(
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
  assert(kernel_size > 2);

  const v128_t vmin = wasm_v128_load64_splat(params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load64_splat(params->wasmsimd.max);
  do {
    const float* w = weights;

    // First pass to process 2 inputs.
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
      input += 2;

      // Process c channels and write to buffer.
      size_t c = round_up_po2(channels, 4);
      for (; c >= 16; c -= 16) {
        v128_t vacc0123p0 = wasm_v128_load(w);
        v128_t vacc4567p0 = wasm_v128_load(w + 4);
        v128_t vacc89ABp0 = wasm_v128_load(w + 8);
        v128_t vaccCDEFp0 = wasm_v128_load(w + 12);


        const v128_t vi0x0123 = wasm_v128_load(i0);
        const v128_t vi0x4567 = wasm_v128_load(i0 + 4);
        const v128_t vi0x89AB = wasm_v128_load(i0 + 8);
        const v128_t vi0xCDEF = wasm_v128_load(i0 + 12);
        i0 += 16;

        const v128_t vk0x0123 = wasm_v128_load(w + 16);
        const v128_t vk0x4567 = wasm_v128_load(w + 20);
        const v128_t vk0x89AB = wasm_v128_load(w + 24);
        const v128_t vk0xCDEF = wasm_v128_load(w + 28);
        vacc0123p0 = wasm_f32x4_add(vacc0123p0, wasm_f32x4_mul(vi0x0123, vk0x0123));
        vacc4567p0 = wasm_f32x4_add(vacc4567p0, wasm_f32x4_mul(vi0x4567, vk0x4567));
        vacc89ABp0 = wasm_f32x4_add(vacc89ABp0, wasm_f32x4_mul(vi0x89AB, vk0x89AB));
        vaccCDEFp0 = wasm_f32x4_add(vaccCDEFp0, wasm_f32x4_mul(vi0xCDEF, vk0xCDEF));

        const v128_t vi1x0123 = wasm_v128_load(i1);
        const v128_t vi1x4567 = wasm_v128_load(i1 + 4);
        const v128_t vi1x89AB = wasm_v128_load(i1 + 8);
        const v128_t vi1xCDEF = wasm_v128_load(i1 + 12);
        i1 += 16;

        const v128_t vk1x0123 = wasm_v128_load(w + 32);
        const v128_t vk1x4567 = wasm_v128_load(w + 36);
        const v128_t vk1x89AB = wasm_v128_load(w + 40);
        const v128_t vk1xCDEF = wasm_v128_load(w + 44);
        vacc0123p0 = wasm_f32x4_add(vacc0123p0, wasm_f32x4_mul(vi1x0123, vk1x0123));
        vacc4567p0 = wasm_f32x4_add(vacc4567p0, wasm_f32x4_mul(vi1x4567, vk1x4567));
        vacc89ABp0 = wasm_f32x4_add(vacc89ABp0, wasm_f32x4_mul(vi1x89AB, vk1x89AB));
        vaccCDEFp0 = wasm_f32x4_add(vaccCDEFp0, wasm_f32x4_mul(vi1xCDEF, vk1xCDEF));

        w += 48;


        wasm_v128_store(b, vacc0123p0);
        wasm_v128_store(b + 4, vacc4567p0);
        wasm_v128_store(b + 8, vacc89ABp0);
        wasm_v128_store(b + 12, vaccCDEFp0);
        b += 16;
      }

      for (; c != 0; c -= 4) {
        v128_t vacc0p0 = wasm_v128_load(w);


        const v128_t vi0x0123 = wasm_v128_load(i0);
        i0 += 4;

        const v128_t vk0x0123 = wasm_v128_load(w + 4);
        vacc0p0 = wasm_f32x4_add(vacc0p0, wasm_f32x4_mul(vi0x0123, vk0x0123));

        const v128_t vi1x0123 = wasm_v128_load(i1);
        i1 += 4;

        const v128_t vk1x0123 = wasm_v128_load(w + 8);
        vacc0p0 = wasm_f32x4_add(vacc0p0, wasm_f32x4_mul(vi1x0123, vk1x0123));

        w += 12;


        wasm_v128_store(b, vacc0p0);
        b += 4;
      }
    }

    // Middle pass to process 2 inputs in each iteration.
    for (size_t ks = kernel_size - 2; ks > 2; ks -= 2) {
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
      input += 2;

      size_t c = round_up_po2(channels, 4);
      for (; c >= 16; c -= 16) {
        v128_t vacc0123p0 = wasm_v128_load(b);
        v128_t vacc4567p0 = wasm_v128_load(b + 4);
        v128_t vacc89ABp0 = wasm_v128_load(b + 8);
        v128_t vaccCDEFp0 = wasm_v128_load(b + 12);


        const v128_t vi0x0123 = wasm_v128_load(i0);
        const v128_t vi0x4567 = wasm_v128_load(i0 + 4);
        const v128_t vi0x89AB = wasm_v128_load(i0 + 8);
        const v128_t vi0xCDEF = wasm_v128_load(i0 + 12);
        i0 += 16;

        const v128_t vk0x0123 = wasm_v128_load(w);
        const v128_t vk0x4567 = wasm_v128_load(w + 4);
        const v128_t vk0x89AB = wasm_v128_load(w + 8);
        const v128_t vk0xCDEF = wasm_v128_load(w + 12);
        vacc0123p0 = wasm_f32x4_add(vacc0123p0, wasm_f32x4_mul(vi0x0123, vk0x0123));
        vacc4567p0 = wasm_f32x4_add(vacc4567p0, wasm_f32x4_mul(vi0x4567, vk0x4567));
        vacc89ABp0 = wasm_f32x4_add(vacc89ABp0, wasm_f32x4_mul(vi0x89AB, vk0x89AB));
        vaccCDEFp0 = wasm_f32x4_add(vaccCDEFp0, wasm_f32x4_mul(vi0xCDEF, vk0xCDEF));

        const v128_t vi1x0123 = wasm_v128_load(i1);
        const v128_t vi1x4567 = wasm_v128_load(i1 + 4);
        const v128_t vi1x89AB = wasm_v128_load(i1 + 8);
        const v128_t vi1xCDEF = wasm_v128_load(i1 + 12);
        i1 += 16;

        const v128_t vk1x0123 = wasm_v128_load(w + 16);
        const v128_t vk1x4567 = wasm_v128_load(w + 20);
        const v128_t vk1x89AB = wasm_v128_load(w + 24);
        const v128_t vk1xCDEF = wasm_v128_load(w + 28);
        vacc0123p0 = wasm_f32x4_add(vacc0123p0, wasm_f32x4_mul(vi1x0123, vk1x0123));
        vacc4567p0 = wasm_f32x4_add(vacc4567p0, wasm_f32x4_mul(vi1x4567, vk1x4567));
        vacc89ABp0 = wasm_f32x4_add(vacc89ABp0, wasm_f32x4_mul(vi1x89AB, vk1x89AB));
        vaccCDEFp0 = wasm_f32x4_add(vaccCDEFp0, wasm_f32x4_mul(vi1xCDEF, vk1xCDEF));

        w += 32;


        wasm_v128_store(b, vacc0123p0);
        wasm_v128_store(b + 4, vacc4567p0);
        wasm_v128_store(b + 8, vacc89ABp0);
        wasm_v128_store(b + 12, vaccCDEFp0);
        b += 16;
      }

      for (; c != 0; c -= 4) {
        v128_t vacc0p0 = wasm_v128_load(b);


        const v128_t vi0x0123 = wasm_v128_load(i0);
        i0 += 4;

        const v128_t vk0x0123 = wasm_v128_load(w);
        vacc0p0 = wasm_f32x4_add(vacc0p0, wasm_f32x4_mul(vi0x0123, vk0x0123));

        const v128_t vi1x0123 = wasm_v128_load(i1);
        i1 += 4;

        const v128_t vk1x0123 = wasm_v128_load(w + 4);
        vacc0p0 = wasm_f32x4_add(vacc0p0, wasm_f32x4_mul(vi1x0123, vk1x0123));

        w += 8;


        wasm_v128_store(b, vacc0p0);
        b += 4;
      }
    }

    // Last pass to process up to 2 inputs.
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

      size_t c = channels;
      for (; c >= 16; c -= 16) {
        v128_t vacc0123p0 = wasm_v128_load(b);
        v128_t vacc4567p0 = wasm_v128_load(b + 4);
        v128_t vacc89ABp0 = wasm_v128_load(b + 8);
        v128_t vaccCDEFp0 = wasm_v128_load(b + 12);
        b += 16;


        const v128_t vi0x0123 = wasm_v128_load(i0);
        const v128_t vi0x4567 = wasm_v128_load(i0 + 4);
        const v128_t vi0x89AB = wasm_v128_load(i0 + 8);
        const v128_t vi0xCDEF = wasm_v128_load(i0 + 12);
        i0 += 16;

        v128_t vk0x0123 = wasm_v128_load(w);
        v128_t vk0x4567 = wasm_v128_load(w + 4);
        v128_t vk0x89AB = wasm_v128_load(w + 8);
        v128_t vk0xCDEF = wasm_v128_load(w + 12);

        vacc0123p0 = wasm_f32x4_add(vacc0123p0, wasm_f32x4_mul(vi0x0123, vk0x0123));
        vacc4567p0 = wasm_f32x4_add(vacc4567p0, wasm_f32x4_mul(vi0x4567, vk0x4567));
        vacc89ABp0 = wasm_f32x4_add(vacc89ABp0, wasm_f32x4_mul(vi0x89AB, vk0x89AB));
        vaccCDEFp0 = wasm_f32x4_add(vaccCDEFp0, wasm_f32x4_mul(vi0xCDEF, vk0xCDEF));

        const v128_t vi1x0123 = wasm_v128_load(i1);
        const v128_t vi1x4567 = wasm_v128_load(i1 + 4);
        const v128_t vi1x89AB = wasm_v128_load(i1 + 8);
        const v128_t vi1xCDEF = wasm_v128_load(i1 + 12);
        i1 += 16;

        v128_t vk1x0123 = wasm_v128_load(w + 16);
        v128_t vk1x4567 = wasm_v128_load(w + 20);
        v128_t vk1x89AB = wasm_v128_load(w + 24);
        v128_t vk1xCDEF = wasm_v128_load(w + 28);

        vacc0123p0 = wasm_f32x4_add(vacc0123p0, wasm_f32x4_mul(vi1x0123, vk1x0123));
        vacc4567p0 = wasm_f32x4_add(vacc4567p0, wasm_f32x4_mul(vi1x4567, vk1x4567));
        vacc89ABp0 = wasm_f32x4_add(vacc89ABp0, wasm_f32x4_mul(vi1x89AB, vk1x89AB));
        vaccCDEFp0 = wasm_f32x4_add(vaccCDEFp0, wasm_f32x4_mul(vi1xCDEF, vk1xCDEF));

        w += 32;


        v128_t vacc0123 = wasm_f32x4_pmax(vacc0123p0, vmin);
        v128_t vacc4567 = wasm_f32x4_pmax(vacc4567p0, vmin);
        v128_t vacc89AB = wasm_f32x4_pmax(vacc89ABp0, vmin);
        v128_t vaccCDEF = wasm_f32x4_pmax(vaccCDEFp0, vmin);

        vacc0123 = wasm_f32x4_pmin(vacc0123, vmax);
        vacc4567 = wasm_f32x4_pmin(vacc4567, vmax);
        vacc89AB = wasm_f32x4_pmin(vacc89AB, vmax);
        vaccCDEF = wasm_f32x4_pmin(vaccCDEF, vmax);

        wasm_v128_store(output, vacc0123);
        wasm_v128_store(output + 4, vacc4567);
        wasm_v128_store(output + 8, vacc89AB);
        wasm_v128_store(output + 12, vaccCDEF);
        output += 16;
      }


      for (; c >= 4; c -= 4) {
        v128_t vacc0p0 = wasm_v128_load(b);
        b += 4;


        const v128_t vi0x0123 = wasm_v128_load(i0);
        i0 += 4;

        v128_t vk0x0123 = wasm_v128_load(w);

        vacc0p0 = wasm_f32x4_add(vacc0p0, wasm_f32x4_mul(vi0x0123, vk0x0123));

        const v128_t vi1x0123 = wasm_v128_load(i1);
        i1 += 4;

        v128_t vk1x0123 = wasm_v128_load(w + 4);

        vacc0p0 = wasm_f32x4_add(vacc0p0, wasm_f32x4_mul(vi1x0123, vk1x0123));

        w += 8;



        v128_t vacc0 = wasm_f32x4_pmax(vacc0p0, vmin);

        vacc0 = wasm_f32x4_pmin(vacc0, vmax);

        wasm_v128_store(output, vacc0);
        output += 4;
      }

      if XNN_UNLIKELY(c != 0) {
        v128_t vacc0p0 = wasm_v128_load(b);

        const v128_t vi0x0123 = wasm_v128_load(i0);
        v128_t vk0x0123 = wasm_v128_load(w);
        vacc0p0 = wasm_f32x4_add(vacc0p0, wasm_f32x4_mul(vi0x0123, vk0x0123));

        const v128_t vi1x0123 = wasm_v128_load(i1);
        v128_t vk1x0123 = wasm_v128_load(w + 4);
        vacc0p0 = wasm_f32x4_add(vacc0p0, wasm_f32x4_mul(vi1x0123, vk1x0123));


        v128_t vacc0 = wasm_f32x4_pmax(vacc0p0, vmin);

        vacc0 = wasm_f32x4_pmin(vacc0, vmax);

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
