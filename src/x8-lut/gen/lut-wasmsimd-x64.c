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

#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/lut.h>
#include <xnnpack/common.h>


void xnn_x8_lut_ukernel__wasmsimd_x64(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const uint8_t t[restrict XNN_MIN_ELEMENTS(256)])
{
  assert(n != 0);
  assert(x != NULL);
  assert(y != NULL);

  const v128_t vtable0 = wasm_v128_load(t);
  const v128_t vtable1 = wasm_v128_load(t + 16);
  const v128_t vtable2 = wasm_v128_load(t + 32);
  const v128_t vtable3 = wasm_v128_load(t + 48);
  const v128_t vtable4 = wasm_v128_load(t + 64);
  const v128_t vtable5 = wasm_v128_load(t + 80);
  const v128_t vtable6 = wasm_v128_load(t + 96);
  const v128_t vtable7 = wasm_v128_load(t + 112);
  const v128_t vtable8 = wasm_v128_load(t + 128);
  const v128_t vtable9 = wasm_v128_load(t + 144);
  const v128_t vtable10 = wasm_v128_load(t + 160);
  const v128_t vtable11 = wasm_v128_load(t + 176);
  const v128_t vtable12 = wasm_v128_load(t + 192);
  const v128_t vtable13 = wasm_v128_load(t + 208);
  const v128_t vtable14 = wasm_v128_load(t + 224);
  const v128_t vtable15 = wasm_v128_load(t + 240);
  const v128_t voffset = wasm_i8x16_const_splat(16);
  for (; n >= 64 * sizeof(uint8_t); n -= 64 * sizeof(uint8_t)) {
    v128_t vx0 = wasm_v128_load(x);
    v128_t vx1 = wasm_v128_load(x + 16);
    v128_t vx2 = wasm_v128_load(x + 32);
    v128_t vx3 = wasm_v128_load(x + 48);
    x += 64;

    v128_t vy0 = wasm_i8x16_swizzle(vtable0, vx0);
    v128_t vy1 = wasm_i8x16_swizzle(vtable0, vx1);
    v128_t vy2 = wasm_i8x16_swizzle(vtable0, vx2);
    v128_t vy3 = wasm_i8x16_swizzle(vtable0, vx3);

    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable1, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable1, vx1));
    vx2 = wasm_i8x16_sub(vx2, voffset);
    vy2 = wasm_v128_or(vy2, wasm_i8x16_swizzle(vtable1, vx2));
    vx3 = wasm_i8x16_sub(vx3, voffset);
    vy3 = wasm_v128_or(vy3, wasm_i8x16_swizzle(vtable1, vx3));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable2, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable2, vx1));
    vx2 = wasm_i8x16_sub(vx2, voffset);
    vy2 = wasm_v128_or(vy2, wasm_i8x16_swizzle(vtable2, vx2));
    vx3 = wasm_i8x16_sub(vx3, voffset);
    vy3 = wasm_v128_or(vy3, wasm_i8x16_swizzle(vtable2, vx3));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable3, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable3, vx1));
    vx2 = wasm_i8x16_sub(vx2, voffset);
    vy2 = wasm_v128_or(vy2, wasm_i8x16_swizzle(vtable3, vx2));
    vx3 = wasm_i8x16_sub(vx3, voffset);
    vy3 = wasm_v128_or(vy3, wasm_i8x16_swizzle(vtable3, vx3));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable4, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable4, vx1));
    vx2 = wasm_i8x16_sub(vx2, voffset);
    vy2 = wasm_v128_or(vy2, wasm_i8x16_swizzle(vtable4, vx2));
    vx3 = wasm_i8x16_sub(vx3, voffset);
    vy3 = wasm_v128_or(vy3, wasm_i8x16_swizzle(vtable4, vx3));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable5, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable5, vx1));
    vx2 = wasm_i8x16_sub(vx2, voffset);
    vy2 = wasm_v128_or(vy2, wasm_i8x16_swizzle(vtable5, vx2));
    vx3 = wasm_i8x16_sub(vx3, voffset);
    vy3 = wasm_v128_or(vy3, wasm_i8x16_swizzle(vtable5, vx3));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable6, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable6, vx1));
    vx2 = wasm_i8x16_sub(vx2, voffset);
    vy2 = wasm_v128_or(vy2, wasm_i8x16_swizzle(vtable6, vx2));
    vx3 = wasm_i8x16_sub(vx3, voffset);
    vy3 = wasm_v128_or(vy3, wasm_i8x16_swizzle(vtable6, vx3));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable7, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable7, vx1));
    vx2 = wasm_i8x16_sub(vx2, voffset);
    vy2 = wasm_v128_or(vy2, wasm_i8x16_swizzle(vtable7, vx2));
    vx3 = wasm_i8x16_sub(vx3, voffset);
    vy3 = wasm_v128_or(vy3, wasm_i8x16_swizzle(vtable7, vx3));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable8, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable8, vx1));
    vx2 = wasm_i8x16_sub(vx2, voffset);
    vy2 = wasm_v128_or(vy2, wasm_i8x16_swizzle(vtable8, vx2));
    vx3 = wasm_i8x16_sub(vx3, voffset);
    vy3 = wasm_v128_or(vy3, wasm_i8x16_swizzle(vtable8, vx3));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable9, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable9, vx1));
    vx2 = wasm_i8x16_sub(vx2, voffset);
    vy2 = wasm_v128_or(vy2, wasm_i8x16_swizzle(vtable9, vx2));
    vx3 = wasm_i8x16_sub(vx3, voffset);
    vy3 = wasm_v128_or(vy3, wasm_i8x16_swizzle(vtable9, vx3));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable10, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable10, vx1));
    vx2 = wasm_i8x16_sub(vx2, voffset);
    vy2 = wasm_v128_or(vy2, wasm_i8x16_swizzle(vtable10, vx2));
    vx3 = wasm_i8x16_sub(vx3, voffset);
    vy3 = wasm_v128_or(vy3, wasm_i8x16_swizzle(vtable10, vx3));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable11, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable11, vx1));
    vx2 = wasm_i8x16_sub(vx2, voffset);
    vy2 = wasm_v128_or(vy2, wasm_i8x16_swizzle(vtable11, vx2));
    vx3 = wasm_i8x16_sub(vx3, voffset);
    vy3 = wasm_v128_or(vy3, wasm_i8x16_swizzle(vtable11, vx3));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable12, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable12, vx1));
    vx2 = wasm_i8x16_sub(vx2, voffset);
    vy2 = wasm_v128_or(vy2, wasm_i8x16_swizzle(vtable12, vx2));
    vx3 = wasm_i8x16_sub(vx3, voffset);
    vy3 = wasm_v128_or(vy3, wasm_i8x16_swizzle(vtable12, vx3));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable13, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable13, vx1));
    vx2 = wasm_i8x16_sub(vx2, voffset);
    vy2 = wasm_v128_or(vy2, wasm_i8x16_swizzle(vtable13, vx2));
    vx3 = wasm_i8x16_sub(vx3, voffset);
    vy3 = wasm_v128_or(vy3, wasm_i8x16_swizzle(vtable13, vx3));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable14, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable14, vx1));
    vx2 = wasm_i8x16_sub(vx2, voffset);
    vy2 = wasm_v128_or(vy2, wasm_i8x16_swizzle(vtable14, vx2));
    vx3 = wasm_i8x16_sub(vx3, voffset);
    vy3 = wasm_v128_or(vy3, wasm_i8x16_swizzle(vtable14, vx3));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vy0 = wasm_v128_or(vy0, wasm_i8x16_swizzle(vtable15, vx0));
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy1 = wasm_v128_or(vy1, wasm_i8x16_swizzle(vtable15, vx1));
    vx2 = wasm_i8x16_sub(vx2, voffset);
    vy2 = wasm_v128_or(vy2, wasm_i8x16_swizzle(vtable15, vx2));
    vx3 = wasm_i8x16_sub(vx3, voffset);
    vy3 = wasm_v128_or(vy3, wasm_i8x16_swizzle(vtable15, vx3));

    wasm_v128_store(y, vy0);
    wasm_v128_store(y + 16, vy1);
    wasm_v128_store(y + 32, vy2);
    wasm_v128_store(y + 48, vy3);
    y += 64;
  }
  for (; n >= 16 * sizeof(uint8_t); n -= 16 * sizeof(uint8_t)) {
    v128_t vx = wasm_v128_load(x);
    x += 16;

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

    wasm_v128_store(y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    v128_t vx = wasm_v128_load(x);

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

    if (n & (8 * sizeof(uint8_t))) {
      *((double*) y) = wasm_f64x2_extract_lane(vy, 0);
      vy = wasm_v64x2_shuffle(vy, vy, 1, 1);
      y += 8;
    }
    if (n & (4 * sizeof(uint8_t))) {
      *((float*) y) = wasm_f32x4_extract_lane(vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      y += 4;
    }
    uint32_t vy_lo = wasm_i32x4_extract_lane(vy, 0);
    if (n & (2 * sizeof(uint8_t))) {
      *((uint16_t*) y) = (uint16_t) vy_lo;
      vy_lo >>= 16;
      y += 2;
    }
    if (n & (1 * sizeof(uint8_t))) {
      *y = (uint8_t) vy_lo;
    }
  }
}
