// Auto-generated file. Do not edit!
//   Template: src/x8-lut/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/lut.h"
#include "xnnpack/common.h"


void xnn_x8_lut_ukernel__wasmsimd_u32(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const uint8_t table[restrict XNN_MIN_ELEMENTS(256)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vtable0 = wasm_v128_load(table);
  const v128_t vtable1 = wasm_v128_load(table + 16);
  const v128_t vtable2 = wasm_v128_load(table + 32);
  const v128_t vtable3 = wasm_v128_load(table + 48);
  const v128_t vtable4 = wasm_v128_load(table + 64);
  const v128_t vtable5 = wasm_v128_load(table + 80);
  const v128_t vtable6 = wasm_v128_load(table + 96);
  const v128_t vtable7 = wasm_v128_load(table + 112);
  const v128_t vtable8 = wasm_v128_load(table + 128);
  const v128_t vtable9 = wasm_v128_load(table + 144);
  const v128_t vtable10 = wasm_v128_load(table + 160);
  const v128_t vtable11 = wasm_v128_load(table + 176);
  const v128_t vtable12 = wasm_v128_load(table + 192);
  const v128_t vtable13 = wasm_v128_load(table + 208);
  const v128_t vtable14 = wasm_v128_load(table + 224);
  const v128_t vtable15 = wasm_v128_load(table + 240);
  const v128_t voffset = wasm_i8x16_const_splat(16);
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    v128_t vx0 = wasm_v128_load(input);
    v128_t vx1 = wasm_v128_load(input + 16);
    input += 32;

    v128_t vy0 = wasm_i8x16_swizzle(vtable0, vx0);
    v128_t vy1 = wasm_i8x16_swizzle(vtable0, vx1);

    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable1, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable1, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable2, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable2, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable3, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable3, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable4, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable4, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable5, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable5, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable6, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable6, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable7, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable7, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable8, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable8, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable9, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable9, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable10, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable10, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable11, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable11, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable12, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable12, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable13, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable13, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable14, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable14, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable15, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable15, vx1));

    wasm_v128_store(output, vy0);
    wasm_v128_store(output + 16, vy1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    v128_t vx = wasm_v128_load(input);
    input += 16;

    v128_t vy = wasm_i8x16_swizzle(vtable0, vx);

    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable1, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable2, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable3, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable4, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable5, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable6, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable7, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable8, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable9, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable10, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable11, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable12, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable13, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable14, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable15, vx));

    wasm_v128_store(output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    v128_t vx = wasm_v128_load(input);

    v128_t vy = wasm_i8x16_swizzle(vtable0, vx);

    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable1, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable2, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable3, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable4, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable5, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable6, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable7, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable8, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable9, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable10, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable11, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable12, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable13, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable14, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_or(vy, wasm_i8x16_swizzle(vtable15, vx));

    if (batch & (8 * sizeof(uint8_t))) {
      wasm_v128_store64_lane(output, vy, 0);
      vy = wasm_v64x2_shuffle(vy, vy, 1, 1);
      output += 8;
    }
    if (batch & (4 * sizeof(uint8_t))) {
      wasm_v128_store32_lane(output, vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(uint8_t))) {
      wasm_v128_store16_lane(output, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      wasm_v128_store8_lane(output, vy, 0);
    }
  }
}
