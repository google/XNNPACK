// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/avgpool.h>


void xnn_f32_avgpool_minmax_ukernel_9p8x__wasmsimd_arm_c4(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* zero,
    float* buffer,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements > 9);
  assert(channels != 0);

  const v128_t vscale = wasm_v128_load32_splat(&params->scalar.scale);
  const v128_t vmin = wasm_v128_load32_splat(&params->scalar.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->scalar.max);

  do {
    {
      const float* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      const float* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      const float* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      const float* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }
      const float* i8 = *input++;
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 = (const float*) ((uintptr_t) i8 + input_offset);
      }

      float* b = buffer;
      for (size_t c = 0; c < channels; c += 4) {
        const v128_t vi0 = wasm_v128_load(i0);
        i0 += 4;
        const v128_t vi1 = wasm_v128_load(i1);
        i1 += 4;
        const v128_t vi2 = wasm_v128_load(i2);
        i2 += 4;
        const v128_t vi3 = wasm_v128_load(i3);
        i3 += 4;
        const v128_t vi4 = wasm_v128_load(i4);
        i4 += 4;
        const v128_t vi5 = wasm_v128_load(i5);
        i5 += 4;
        const v128_t vi6 = wasm_v128_load(i6);
        i6 += 4;
        const v128_t vi7 = wasm_v128_load(i7);
        i7 += 4;
        const v128_t vi8 = wasm_v128_load(i8);
        i8 += 4;

        const v128_t vsum01 = wasm_f32x4_add(vi0, vi1);
        const v128_t vsum23 = wasm_f32x4_add(vi2, vi3);
        const v128_t vsum45 = wasm_f32x4_add(vi4, vi5);
        const v128_t vsum67 = wasm_f32x4_add(vi6, vi7);
        const v128_t vsum018 = wasm_f32x4_add(vsum01, vi8);
        const v128_t vsum2345 = wasm_f32x4_add(vsum23, vsum45);
        const v128_t vsum01678 = wasm_f32x4_add(vsum018, vsum67);
        const v128_t vsum = wasm_f32x4_add(vsum2345, vsum01678);

        wasm_v128_store(b, vsum);
        b += 4;
      }
    }

    size_t k = kernel_elements;
    for (k -= 9; k > 8; k -= 8) {
      const float* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      const float* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      const float* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      const float* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }

      float* b = buffer;
      for (size_t c = 0; c < channels; c += 4) {
        const v128_t vi0 = wasm_v128_load(i0);
        i0 += 4;
        const v128_t vi1 = wasm_v128_load(i1);
        i1 += 4;
        const v128_t vi2 = wasm_v128_load(i2);
        i2 += 4;
        const v128_t vi3 = wasm_v128_load(i3);
        i3 += 4;
        const v128_t vi4 = wasm_v128_load(i4);
        i4 += 4;
        const v128_t vi5 = wasm_v128_load(i5);
        i5 += 4;
        const v128_t vi6 = wasm_v128_load(i6);
        i6 += 4;
        const v128_t vi7 = wasm_v128_load(i7);
        i7 += 4;
        const v128_t vacc = wasm_v128_load(b);

        const v128_t vsum01 = wasm_f32x4_add(vi0, vi1);
        const v128_t vsum23 = wasm_f32x4_add(vi2, vi3);
        const v128_t vsum45 = wasm_f32x4_add(vi4, vi5);
        const v128_t vsum67 = wasm_f32x4_add(vi6, vi7);
        const v128_t vsum01a = wasm_f32x4_add(vsum01, vacc);
        const v128_t vsum2345 = wasm_f32x4_add(vsum23, vsum45);
        const v128_t vsum0167a = wasm_f32x4_add(vsum01a, vsum67);
        const v128_t vsum = wasm_f32x4_add(vsum2345, vsum0167a);

        wasm_v128_store(b, vsum);
        b += 4;
      }
    }

    {
      const float* i0 = input[0];
      assert(i0 != NULL);
      const float* i1 = input[1];
      const float* i2 = input[2];
      const float* i3 = input[3];
      const float* i4 = input[4];
      const float* i5 = input[5];
      const float* i6 = input[6];
      const float* i7 = input[7];
      input = (const float**) ((uintptr_t) input + input_increment);
      if (k < 2) {
        i1 = zero;
      }
      assert(i1 != NULL);
      if (k <= 2) {
        i2 = zero;
      }
      assert(i2 != NULL);
      if (k < 4) {
        i3 = zero;
      }
      assert(i3 != NULL);
      if (k <= 4) {
        i4 = zero;
      }
      assert(i4 != NULL);
      if (k < 6) {
        i5 = zero;
      }
      assert(i5 != NULL);
      if (k <= 6) {
        i6 = zero;
      }
      assert(i6 != NULL);
      if (k < 8) {
        i7 = zero;
      }
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }

      size_t c = channels;
      float* b = buffer;
      while (c >= 4) {
        const v128_t vi0 = wasm_v128_load(i0);
        i0 += 4;
        const v128_t vi1 = wasm_v128_load(i1);
        i1 += 4;
        const v128_t vi2 = wasm_v128_load(i2);
        i2 += 4;
        const v128_t vi3 = wasm_v128_load(i3);
        i3 += 4;
        const v128_t vi4 = wasm_v128_load(i4);
        i4 += 4;
        const v128_t vi5 = wasm_v128_load(i5);
        i5 += 4;
        const v128_t vi6 = wasm_v128_load(i6);
        i6 += 4;
        const v128_t vi7 = wasm_v128_load(i7);
        i7 += 4;
        const v128_t vacc = wasm_v128_load(b);
        b += 4;

        const v128_t vsum01 = wasm_f32x4_add(vi0, vi1);
        const v128_t vsum23 = wasm_f32x4_add(vi2, vi3);
        const v128_t vsum45 = wasm_f32x4_add(vi4, vi5);
        const v128_t vsum67 = wasm_f32x4_add(vi6, vi7);
        const v128_t vsum01a = wasm_f32x4_add(vsum01, vacc);
        const v128_t vsum2345 = wasm_f32x4_add(vsum23, vsum45);
        const v128_t vsum0167a = wasm_f32x4_add(vsum01a, vsum67);
        const v128_t vsum = wasm_f32x4_add(vsum2345, vsum0167a);

        v128_t vout = wasm_f32x4_mul(vsum, vscale);
        vout = wasm_f32x4_max(vout, vmin);
        vout = wasm_f32x4_min(vout, vmax);

        wasm_v128_store(output, vout);
        output += 4;

        c -= 4;
      }
      if (c != 0) {
        const v128_t vi0 = wasm_v128_load(i0);
        const v128_t vi1 = wasm_v128_load(i1);
        const v128_t vi2 = wasm_v128_load(i2);
        const v128_t vi3 = wasm_v128_load(i3);
        const v128_t vi4 = wasm_v128_load(i4);
        const v128_t vi5 = wasm_v128_load(i5);
        const v128_t vi6 = wasm_v128_load(i6);
        const v128_t vi7 = wasm_v128_load(i7);
        const v128_t vacc = wasm_v128_load(b);

        const v128_t vsum01 = wasm_f32x4_add(vi0, vi1);
        const v128_t vsum23 = wasm_f32x4_add(vi2, vi3);
        const v128_t vsum45 = wasm_f32x4_add(vi4, vi5);
        const v128_t vsum67 = wasm_f32x4_add(vi6, vi7);
        const v128_t vsum01a = wasm_f32x4_add(vsum01, vacc);
        const v128_t vsum2345 = wasm_f32x4_add(vsum23, vsum45);
        const v128_t vsum0167a = wasm_f32x4_add(vsum01a, vsum67);
        const v128_t vsum = wasm_f32x4_add(vsum2345, vsum0167a);

        v128_t vout = wasm_f32x4_mul(vsum, vscale);
        vout = wasm_f32x4_max(vout, vmin);
        vout = wasm_f32x4_min(vout, vmax);

        if (c & 2) {
          wasm_v128_store64_lane(output, vout, 0);
          vout = wasm_v64x2_shuffle(vout, vout, 1, 1);
          output += 2;
        }
        if (c & 1) {
          wasm_v128_store32_lane(output, vout, 0);
          output += 1;
        }
      }
    }
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
