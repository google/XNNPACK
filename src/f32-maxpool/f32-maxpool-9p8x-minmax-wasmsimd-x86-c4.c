// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/maxpool.h"


void xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const v128_t voutput_max = wasm_v128_load32_splat(&params->scalar.max);
  const v128_t voutput_min = wasm_v128_load32_splat(&params->scalar.min);
  do {
    float* o = output;
    {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      const float* i8 = *input++;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
      if (kernel_elements < 2) {
        i1 = i0;
      }
      if (kernel_elements <= 2) {
        i2 = i0;
      }
      if (kernel_elements < 4) {
        i3 = i0;
      }
      if (kernel_elements <= 4) {
        i4 = i0;
      }
      if (kernel_elements < 6) {
        i5 = i0;
      }
      if (kernel_elements <= 6) {
        i6 = i0;
      }
      if (kernel_elements < 8) {
        i7 = i0;
      }
      if (kernel_elements <= 8) {
        i8 = i0;
      }

      size_t c = channels;
      for (; c >= 4; c -= 4) {
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

        const v128_t vmax01 = wasm_f32x4_pmax(vi1, vi0);
        const v128_t vmax23 = wasm_f32x4_pmax(vi3, vi2);
        const v128_t vmax45 = wasm_f32x4_pmax(vi5, vi4);
        const v128_t vmax018 = wasm_f32x4_pmax(vi8, vmax01);
        const v128_t vmax67 = wasm_f32x4_pmax(vi7, vi6);

        const v128_t vmax2345 = wasm_f32x4_pmax(vmax45, vmax23);
        const v128_t vmax01678 = wasm_f32x4_pmax(vmax67, vmax018);
        const v128_t vmax = wasm_f32x4_pmax(vmax2345, vmax01678);

        v128_t vout = wasm_f32x4_pmax(voutput_min, vmax);
        vout = wasm_f32x4_pmin(voutput_max, vout);

        wasm_v128_store(o, vout);
        o += 4;
      }
      if (c != 0) {
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

        const v128_t vmax01 = wasm_f32x4_pmax(vi1, vi0);
        const v128_t vmax23 = wasm_f32x4_pmax(vi3, vi2);
        const v128_t vmax45 = wasm_f32x4_pmax(vi5, vi4);
        const v128_t vmax018 = wasm_f32x4_pmax(vi8, vmax01);
        const v128_t vmax67 = wasm_f32x4_pmax(vi7, vi6);

        const v128_t vmax2345 = wasm_f32x4_pmax(vmax45, vmax23);
        const v128_t vmax01678 = wasm_f32x4_pmax(vmax67, vmax018);
        const v128_t vmax = wasm_f32x4_pmax(vmax2345, vmax01678);

        v128_t vout = wasm_f32x4_pmax(voutput_min, vmax);
        vout = wasm_f32x4_pmin(voutput_max, vout);

        if (c & 2) {
          wasm_v128_store64_lane(o, vout, 0);
          vout = wasm_v64x2_shuffle(vout, vout, 1, 1);
          o += 2;
        }
        if (c & 1) {
          wasm_v128_store32_lane(o, vout, 0);
          o += 1;
        }
      }
    }

    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      if (k < 2) {
        i1 = i0;
      }
      if (k <= 2) {
        i2 = i0;
      }
      if (k < 4) {
        i3 = i0;
      }
      if (k <= 4) {
        i4 = i0;
      }
      if (k < 6) {
        i5 = i0;
      }
      if (k <= 6) {
        i6 = i0;
      }
      if (k < 8) {
        i7 = i0;
      }

      o = output;
      size_t c = channels;
      for (; c >= 4; c -= 4) {
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
        const v128_t vo = wasm_v128_load(o);

        const v128_t vmax01 = wasm_f32x4_pmax(vi1, vi0);
        const v128_t vmax23 = wasm_f32x4_pmax(vi3, vi2);
        const v128_t vmax45 = wasm_f32x4_pmax(vi5, vi4);
        const v128_t vmax01o = wasm_f32x4_pmax(vo, vmax01);
        const v128_t vmax67 = wasm_f32x4_pmax(vi7, vi6);

        const v128_t vmax2345 = wasm_f32x4_pmax(vmax45, vmax23);
        const v128_t vmax0167 = wasm_f32x4_pmax(vmax67, vmax01o);
        const v128_t vmax = wasm_f32x4_pmax(vmax2345, vmax0167);

        v128_t vout = wasm_f32x4_pmax(voutput_min, vmax);
        vout = wasm_f32x4_pmin(voutput_max, vout);

        wasm_v128_store(o, vout);
        o += 4;
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
        const v128_t vo = wasm_v128_load(o);

        const v128_t vmax01 = wasm_f32x4_pmax(vi1, vi0);
        const v128_t vmax23 = wasm_f32x4_pmax(vi3, vi2);
        const v128_t vmax45 = wasm_f32x4_pmax(vi5, vi4);
        const v128_t vmax01o = wasm_f32x4_pmax(vo, vmax01);
        const v128_t vmax67 = wasm_f32x4_pmax(vi7, vi6);

        const v128_t vmax2345 = wasm_f32x4_pmax(vmax45, vmax23);
        const v128_t vmax0167 = wasm_f32x4_pmax(vmax67, vmax01o);
        const v128_t vmax = wasm_f32x4_pmax(vmax2345, vmax0167);

        v128_t vout = wasm_f32x4_pmax(voutput_min, vmax);
        vout = wasm_f32x4_pmin(voutput_max, vout);

        if (c & 2) {
          wasm_v128_store64_lane(o, vout, 0);
          vout = wasm_v64x2_shuffle(vout, vout, 1, 1);
          o += 2;
        }
        if (c & 1) {
          wasm_v128_store32_lane(o, vout, 0);
          o += 1;
        }
      }
    }
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}
