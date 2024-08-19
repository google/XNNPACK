// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/maxpool.h"


void xnn_u8_maxpool_minmax_ukernel_9p8x__wasmsimd_c16(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const uint8_t** input,
    size_t input_offset,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_u8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const v128_t voutput_max = wasm_v128_load8_splat(&params->scalar.max);
  const v128_t voutput_min = wasm_v128_load8_splat(&params->scalar.min);
  XNN_FORCE_REALIZATION(voutput_max);
  XNN_FORCE_REALIZATION(voutput_min);

  do {
    uint8_t* o = output;
    {
      const uint8_t* i0 = *input++;
      const uint8_t* i1 = *input++;
      const uint8_t* i2 = *input++;
      const uint8_t* i3 = *input++;
      const uint8_t* i4 = *input++;
      const uint8_t* i5 = *input++;
      const uint8_t* i6 = *input++;
      const uint8_t* i7 = *input++;
      const uint8_t* i8 = *input++;
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
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
      for (; c >= 16; c -= 16) {
        const v128_t vi0 = wasm_v128_load(i0);
        i0 += 16;
        const v128_t vi1 = wasm_v128_load(i1);
        i1 += 16;
        const v128_t vi2 = wasm_v128_load(i2);
        i2 += 16;
        const v128_t vi3 = wasm_v128_load(i3);
        i3 += 16;
        const v128_t vi4 = wasm_v128_load(i4);
        i4 += 16;
        const v128_t vi5 = wasm_v128_load(i5);
        i5 += 16;
        const v128_t vi6 = wasm_v128_load(i6);
        i6 += 16;
        const v128_t vi7 = wasm_v128_load(i7);
        i7 += 16;
        const v128_t vi8 = wasm_v128_load(i8);
        i8 += 16;

        const v128_t vmax018 = wasm_u8x16_max(wasm_u8x16_max(vi0, vi1), vi8);
        const v128_t vmax23 = wasm_u8x16_max(vi2, vi3);
        const v128_t vmax45 = wasm_u8x16_max(vi4, vi5);
        const v128_t vmax67 = wasm_u8x16_max(vi6, vi7);

        const v128_t vmax2345 = wasm_u8x16_max(vmax23, vmax45);
        const v128_t vmax01678 = wasm_u8x16_max(vmax018, vmax67);
        v128_t vout = wasm_u8x16_max(vmax2345, vmax01678);
        vout = wasm_u8x16_min(vout, voutput_max);
        vout = wasm_u8x16_max(vout, voutput_min);

        wasm_v128_store(o, vout); o += 16;
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
        const v128_t vi8 = wasm_v128_load(i8);

        const v128_t vmax018 = wasm_u8x16_max(wasm_u8x16_max(vi0, vi1), vi8);
        const v128_t vmax23 = wasm_u8x16_max(vi2, vi3);
        const v128_t vmax45 = wasm_u8x16_max(vi4, vi5);
        const v128_t vmax67 = wasm_u8x16_max(vi6, vi7);

        const v128_t vmax2345 = wasm_u8x16_max(vmax23, vmax45);
        const v128_t vmax01678 = wasm_u8x16_max(vmax018, vmax67);
        v128_t vout = wasm_u8x16_max(vmax2345, vmax01678);
        vout = wasm_u8x16_min(vout, voutput_max);
        vout = wasm_u8x16_max(vout, voutput_min);

        if (c & 8) {
          wasm_v128_store64_lane(o, vout, 0);
          vout = wasm_v64x2_shuffle(vout, vout, 1, 1);
          o += 8;
        }
        if (c & 4) {
          wasm_v128_store32_lane(o, vout, 0);
          vout = wasm_u64x2_shr(vout, 32);
          o += 4;
        }
        if (c & 2) {
          wasm_v128_store16_lane(o, vout, 0);
          vout = wasm_u32x4_shr(vout, 16);
          o += 2;
        }
        if (c & 1) {
          wasm_v128_store8_lane(o, vout, 0);
          o += 1;
        }
      }
    }

    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
      const uint8_t* i0 = *input++;
      const uint8_t* i1 = *input++;
      const uint8_t* i2 = *input++;
      const uint8_t* i3 = *input++;
      const uint8_t* i4 = *input++;
      const uint8_t* i5 = *input++;
      const uint8_t* i6 = *input++;
      const uint8_t* i7 = *input++;
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
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
      for (; c >= 16; c -= 16) {
        const v128_t vi0 = wasm_v128_load(i0);
        i0 += 16;
        const v128_t vi1 = wasm_v128_load(i1);
        i1 += 16;
        const v128_t vi2 = wasm_v128_load(i2);
        i2 += 16;
        const v128_t vi3 = wasm_v128_load(i3);
        i3 += 16;
        const v128_t vi4 = wasm_v128_load(i4);
        i4 += 16;
        const v128_t vi5 = wasm_v128_load(i5);
        i5 += 16;
        const v128_t vi6 = wasm_v128_load(i6);
        i6 += 16;
        const v128_t vi7 = wasm_v128_load(i7);
        i7 += 16;
        const v128_t vo = wasm_v128_load(o);

        const v128_t vmax01 = wasm_u8x16_max(wasm_u8x16_max(vi0, vi1), vo);
        const v128_t vmax23 = wasm_u8x16_max(vi2, vi3);
        const v128_t vmax45 = wasm_u8x16_max(vi4, vi5);
        const v128_t vmax67 = wasm_u8x16_max(vi6, vi7);

        const v128_t vmax2345 = wasm_u8x16_max(vmax23, vmax45);
        const v128_t vmax0167 = wasm_u8x16_max(vmax01, vmax67);
        v128_t vout = wasm_u8x16_max(vmax2345, vmax0167);
        vout = wasm_u8x16_min(vout, voutput_max);
        vout = wasm_u8x16_max(vout, voutput_min);

        wasm_v128_store(o, vout);
        o += 16;
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

        const v128_t vmax01 = wasm_u8x16_max(wasm_u8x16_max(vi0, vi1), vo);
        const v128_t vmax23 = wasm_u8x16_max(vi2, vi3);
        const v128_t vmax45 = wasm_u8x16_max(vi4, vi5);
        const v128_t vmax67 = wasm_u8x16_max(vi6, vi7);

        const v128_t vmax2345 = wasm_u8x16_max(vmax23, vmax45);
        const v128_t vmax0167 = wasm_u8x16_max(vmax01, vmax67);
        v128_t vout = wasm_u8x16_max(vmax2345, vmax0167);
        vout = wasm_u8x16_min(vout, voutput_max);
        vout = wasm_u8x16_max(vout, voutput_min);

        if (c & 8) {
          wasm_v128_store64_lane(o, vout, 0);
          vout = wasm_v64x2_shuffle(vout, vout, 1, 1);
          o += 8;
        }
        if (c & 4) {
          wasm_v128_store32_lane(o, vout, 0);
          vout = wasm_u64x2_shr(vout, 32);
          o += 4;
        }
        if (c & 2) {
          wasm_v128_store16_lane(o, vout, 0);
          vout = wasm_u32x4_shr(vout, 16);
          o += 2;
        }
        if (c & 1) {
          wasm_v128_store8_lane(o, vout, 0);
          o += 1;
        }
      }
    }
    input = (const uint8_t**) ((uintptr_t) input + input_increment);
    output = (uint8_t*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}
