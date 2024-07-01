// Auto-generated file. Do not edit!
//   Template: src/f32-ibilinear/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/ibilinear.h"


void xnn_f32_ibilinear_ukernel__wasmsimd_c8(
    size_t output_pixels,
    size_t channels,
    const float** restrict input,
    size_t input_offset,
    const float* restrict weights,
    float* restrict output,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  do {
    const float* i0 = (const float*) ((uintptr_t) input[0] + input_offset);
    const float* i1 = (const float*) ((uintptr_t) input[1] + input_offset);
    const float* i2 = (const float*) ((uintptr_t) input[2] + input_offset);
    const float* i3 = (const float*) ((uintptr_t) input[3] + input_offset);
    input += 4;

    const v128_t valphah = wasm_v128_load32_splat(weights);
    const v128_t valphav = wasm_v128_load32_splat(weights + 1);
    weights += 2;

    size_t c = channels;
    for (; c >= 8 * sizeof(float); c -= 8 * sizeof(float)) {
      const v128_t vtl0123 = wasm_v128_load(i0);
      const v128_t vtr0123 = wasm_v128_load(i1);
      const v128_t vbl0123 = wasm_v128_load(i2);
      const v128_t vbr0123 = wasm_v128_load(i3);
      const v128_t vtl4567 = wasm_v128_load(i0 + 4);
      const v128_t vtr4567 = wasm_v128_load(i1 + 4);
      const v128_t vbl4567 = wasm_v128_load(i2 + 4);
      const v128_t vbr4567 = wasm_v128_load(i3 + 4);
      i0 += 8;
      i1 += 8;
      i2 += 8;
      i3 += 8;

      const v128_t vtd0123 = wasm_f32x4_sub(vtr0123, vtl0123);
      const v128_t vbd0123 = wasm_f32x4_sub(vbr0123, vbl0123);
      const v128_t vtd4567 = wasm_f32x4_sub(vtr4567, vtl4567);
      const v128_t vbd4567 = wasm_f32x4_sub(vbr4567, vbl4567);

      const v128_t vt0123 = wasm_f32x4_add(wasm_f32x4_mul(vtd0123, valphah), vtl0123);
      const v128_t vb0123 = wasm_f32x4_add(wasm_f32x4_mul(vbd0123, valphah), vbl0123);
      const v128_t vt4567 = wasm_f32x4_add(wasm_f32x4_mul(vtd4567, valphah), vtl4567);
      const v128_t vb4567 = wasm_f32x4_add(wasm_f32x4_mul(vbd4567, valphah), vbl4567);

      const v128_t vd0123 = wasm_f32x4_sub(vb0123, vt0123);
      const v128_t vd4567 = wasm_f32x4_sub(vb4567, vt4567);

      const v128_t vo0123 = wasm_f32x4_add(wasm_f32x4_mul(vd0123, valphav), vt0123);
      const v128_t vo4567 = wasm_f32x4_add(wasm_f32x4_mul(vd4567, valphav), vt4567);

      wasm_v128_store(output, vo0123);
      wasm_v128_store(output + 4, vo4567);
      output += 8;
    }
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const v128_t vtl = wasm_v128_load(i0);
      const v128_t vtr = wasm_v128_load(i1);
      const v128_t vbl = wasm_v128_load(i2);
      const v128_t vbr = wasm_v128_load(i3);
      i0 += 4;
      i1 += 4;
      i2 += 4;
      i3 += 4;

      const v128_t vtd = wasm_f32x4_sub(vtr, vtl);
      const v128_t vbd = wasm_f32x4_sub(vbr, vbl);
      const v128_t vt = wasm_f32x4_add(wasm_f32x4_mul(vtd, valphah), vtl);
      const v128_t vb = wasm_f32x4_add(wasm_f32x4_mul(vbd, valphah), vbl);
      const v128_t vd = wasm_f32x4_sub(vb, vt);
      const v128_t vo = wasm_f32x4_add(wasm_f32x4_mul(vd, valphav), vt);

      wasm_v128_store(output, vo);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      const v128_t vtl = wasm_v128_load(i0);
      const v128_t vtr = wasm_v128_load(i1);
      const v128_t vbl = wasm_v128_load(i2);
      const v128_t vbr = wasm_v128_load(i3);

      const v128_t vtd = wasm_f32x4_sub(vtr, vtl);
      const v128_t vbd = wasm_f32x4_sub(vbr, vbl);
      const v128_t vt = wasm_f32x4_add(wasm_f32x4_mul(vtd, valphah), vtl);
      const v128_t vb = wasm_f32x4_add(wasm_f32x4_mul(vbd, valphah), vbl);
      const v128_t vd = wasm_f32x4_sub(vb, vt);
      v128_t vo = wasm_f32x4_add(wasm_f32x4_mul(vd, valphav), vt);

      if (c & (2 * sizeof(float))) {
        wasm_v128_store64_lane(output, vo, 0);
        vo = wasm_v64x2_shuffle(vo, vo, 1, 1);
        output += 2;
      }
      if (c & (1 * sizeof(float))) {
        wasm_v128_store32_lane(output, vo, 0);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
