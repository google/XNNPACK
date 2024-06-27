// Auto-generated file. Do not edit!
//   Template: src/f32-ibilinear-chw/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/ibilinear.h"


void xnn_f32_ibilinear_chw_ukernel__wasmsimd_p4(
    size_t output_pixels,
    size_t channels,
    const float** restrict input,
    size_t input_offset,
    const float* restrict weights,
    float* restrict output,
    size_t input_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(input_increment % sizeof(float) == 0);

  do {
    const float** i = input;
    const float* w = weights;
    size_t p = output_pixels;

    for (; p >= 4; p -= 4) {
      const float* itl0 = (const float*) ((uintptr_t) i[0] + input_offset);
      const float* ibl0 = (const float*) ((uintptr_t) i[1] + input_offset);
      const float* itl1 = (const float*) ((uintptr_t) i[2] + input_offset);
      const float* ibl1 = (const float*) ((uintptr_t) i[3] + input_offset);
      const float* itl2 = (const float*) ((uintptr_t) i[4] + input_offset);
      const float* ibl2 = (const float*) ((uintptr_t) i[5] + input_offset);
      const float* itl3 = (const float*) ((uintptr_t) i[6] + input_offset);
      const float* ibl3 = (const float*) ((uintptr_t) i[7] + input_offset);
      i += 8;

      const v128_t vw0 = wasm_v128_load(w);
      const v128_t vw1 = wasm_v128_load(w + 4);
      w += 8;

      const v128_t vtltr0 = wasm_v128_load64_splat(itl0);
      const v128_t vblbr0 = wasm_v128_load64_splat(ibl0);
      const v128_t vtltr2 = wasm_v128_load64_splat(itl2);
      const v128_t vblbr2 = wasm_v128_load64_splat(ibl2);

      const v128_t vtltr01 = wasm_v128_load64_lane(itl1, vtltr0, 1);
      const v128_t vblbr01 = wasm_v128_load64_lane(ibl1, vblbr0, 1);
      const v128_t vtltr23 = wasm_v128_load64_lane(itl3, vtltr2, 1);
      const v128_t vblbr23 = wasm_v128_load64_lane(ibl3, vblbr2, 1);

      const v128_t valphah = wasm_v32x4_shuffle(vw0, vw1, 0, 2, 4, 6);
      const v128_t valphav = wasm_v32x4_shuffle(vw0, vw1, 1, 3, 5, 7);

      const v128_t vldrd01 = wasm_f32x4_sub(vblbr01, vtltr01);
      const v128_t vldrd23 = wasm_f32x4_sub(vblbr23, vtltr23);

      const v128_t vld = wasm_v32x4_shuffle(vldrd01, vldrd23, 0, 2, 4, 6);
      const v128_t vrd = wasm_v32x4_shuffle(vldrd01, vldrd23, 1, 3, 5, 7);

      const v128_t vtl = wasm_v32x4_shuffle(vtltr01, vtltr23, 0, 2, 4, 6);
      const v128_t vtr = wasm_v32x4_shuffle(vtltr01, vtltr23, 1, 3, 5, 7);

      const v128_t vl = wasm_f32x4_add(vtl, wasm_f32x4_mul(vld, valphav));
      const v128_t vr = wasm_f32x4_add(vtr, wasm_f32x4_mul(vrd, valphav));

      const v128_t vd = wasm_f32x4_sub(vr, vl);
      const v128_t vo = wasm_f32x4_add(vl, wasm_f32x4_mul(vd, valphah));

      wasm_v128_store(output, vo);
      output += 4;
    }

    if XNN_UNLIKELY(p != 0) {
      if (p & 2) {
        const v128_t vw = wasm_v128_load(w);
        w += 4;

        const v128_t valphah = wasm_v32x4_shuffle(vw, vw, 0, 2, 0, 2);
        const v128_t valphav = wasm_v32x4_shuffle(vw, vw, 1, 3, 1, 3);

        const float* itl0 = (const float*) ((uintptr_t) i[0] + input_offset);
        const float* ibl0 = (const float*) ((uintptr_t) i[1] + input_offset);
        const float* itl1 = (const float*) ((uintptr_t) i[2] + input_offset);
        const float* ibl1 = (const float*) ((uintptr_t) i[3] + input_offset);
        i += 4;

        const v128_t vtltr = wasm_v128_load64_lane(itl1, wasm_v128_load64_zero(itl0), 1);
        const v128_t vblbr = wasm_v128_load64_lane(ibl1, wasm_v128_load64_zero(ibl0), 1);

        const v128_t vldrd = wasm_f32x4_sub(vblbr, vtltr);
        const v128_t vld = wasm_v32x4_shuffle(vldrd, vldrd, 0, 2, 0, 2);
        const v128_t vrd = wasm_v32x4_shuffle(vldrd, vldrd, 1, 3, 1, 3);

        const v128_t vtl = wasm_v32x4_shuffle(vtltr, vtltr, 0, 2, 0, 2);
        const v128_t vtr = wasm_v32x4_shuffle(vtltr, vtltr, 1, 3, 1, 3);

        const v128_t vl = wasm_f32x4_add(vtl, wasm_f32x4_mul(vld, valphav));
        const v128_t vr = wasm_f32x4_add(vtr, wasm_f32x4_mul(vrd, valphav));

        const v128_t vd = wasm_f32x4_sub(vr, vl);
        const v128_t vo = wasm_f32x4_add(vl, wasm_f32x4_mul(vd, valphah));

        wasm_v128_store64_lane(output, vo, 0);
        output += 2;
      }

      if (p & 1) {
        // We are computing the following formula:
        //   result = (1 - alpha_h) * (1 - alpha_v) * top_left +
        //                 alpha_h  * (1 - alpha_v) * top_right +
        //            (1 - alpha_h) *      alpha_v  * bottom_left +
        //                 alpha_h  *      alpha_v  * bottom_right.
        //
        // Rearranging gives
        //   result =    left + alpha_h * (right        - left),
        // where
        //   left =  top_left + alpha_v * (bottom_left  - top_left),
        //  right = top_right + alpha_v * (bottom_right - top_right).

        const float alphah = *w;
        const v128_t valphav = wasm_v128_load32_splat(w + 1);
        w += 2;

        const float* itl = (const float*) ((uintptr_t) i[0] + input_offset);
        const float* ibl = (const float*) ((uintptr_t) i[1] + input_offset);
        i += 2;

        const v128_t vtltr = wasm_v128_load64_zero(itl);
        const v128_t vblbr = wasm_v128_load64_zero(ibl);

        // Compute at once
        //    left_diff = bottom_left  - top_left
        //   right_diff = bottom_right - top_right
        const v128_t vldrd = wasm_f32x4_sub(vblbr, vtltr);
        const v128_t vlr = wasm_f32x4_add(vtltr, wasm_f32x4_mul(vldrd, valphav));

        // Extract them and compute the result.
        const float l = wasm_f32x4_extract_lane(vlr, 0);
        const float r = wasm_f32x4_extract_lane(vlr, 1);

        *output++ = l + alphah * (r - l);
      }
    }

    input_offset += input_increment;
  } while (--channels != 0);
}
