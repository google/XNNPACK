// Auto-generated file. Do not edit!
//   Template: src/f32-vrnd/vrndz-wasmsimd-addsub.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/vunary.h>


void xnn_f32_vrndz_ukernel__wasmsimd_addsub_x8(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const v128_t vsign_mask = wasm_f32x4_const_splat(-0.0f);
  const v128_t vmagic_number = wasm_f32x4_const_splat(0x1.000000p+23f);
  const v128_t vone = wasm_f32x4_const_splat(1.0f);
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const v128_t vx0123 = wasm_v128_load(x);
    const v128_t vx4567 = wasm_v128_load(x + 4);
    x += 8;

    const v128_t vabsx0123 = wasm_v128_andnot(vx0123, vsign_mask);
    const v128_t vabsx4567 = wasm_v128_andnot(vx4567, vsign_mask);

    const v128_t vrndmask0123 = wasm_v128_or(vsign_mask, wasm_f32x4_le(vmagic_number, vabsx0123));
    const v128_t vrndmask4567 = wasm_v128_or(vsign_mask, wasm_f32x4_le(vmagic_number, vabsx4567));

    const v128_t vrndabsx0123 = wasm_f32x4_sub(wasm_f32x4_add(vabsx0123, vmagic_number), vmagic_number);
    const v128_t vrndabsx4567 = wasm_f32x4_sub(wasm_f32x4_add(vabsx4567, vmagic_number), vmagic_number);

    const v128_t vadjustment0123 = wasm_v128_and(wasm_f32x4_lt(vabsx0123, vrndabsx0123), vone);
    const v128_t vadjustment4567 = wasm_v128_and(wasm_f32x4_lt(vabsx4567, vrndabsx4567), vone);

    const v128_t vflrabsx0123 = wasm_f32x4_sub(vrndabsx0123, vadjustment0123);
    const v128_t vflrabsx4567 = wasm_f32x4_sub(vrndabsx4567, vadjustment4567);

    const v128_t vy0123 = wasm_v128_bitselect(vx0123, vflrabsx0123, vrndmask0123);
    const v128_t vy4567 = wasm_v128_bitselect(vx4567, vflrabsx4567, vrndmask4567);

    wasm_v128_store(y, vy0123);
    wasm_v128_store(y + 4, vy4567);
    y += 8;
  }
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(x);
    x += 4;

    const v128_t vabsx = wasm_v128_andnot(vx, vsign_mask);
    const v128_t vrndmask = wasm_v128_or(vsign_mask, wasm_f32x4_le(vmagic_number, vabsx));
    const v128_t vrndabsx = wasm_f32x4_sub(wasm_f32x4_add(vabsx, vmagic_number), vmagic_number);
    const v128_t vadjustment = wasm_v128_and(wasm_f32x4_lt(vabsx, vrndabsx), vone);
    const v128_t vflrabsx = wasm_f32x4_sub(vrndabsx, vadjustment);
    const v128_t vy = wasm_v128_bitselect(vx, vflrabsx, vrndmask);

    wasm_v128_store(y, vy);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const v128_t vx = wasm_v128_load(x);

    const v128_t vabsx = wasm_v128_andnot(vx, vsign_mask);
    const v128_t vrndmask = wasm_v128_or(vsign_mask, wasm_f32x4_le(vmagic_number, vabsx));
    const v128_t vrndabsx = wasm_f32x4_sub(wasm_f32x4_add(vabsx, vmagic_number), vmagic_number);
    const v128_t vadjustment = wasm_v128_and(wasm_f32x4_lt(vabsx, vrndabsx), vone);
    const v128_t vflrabsx = wasm_f32x4_sub(vrndabsx, vadjustment);
    v128_t vy = wasm_v128_bitselect(vx, vflrabsx, vrndmask);

    if (n & (2 * sizeof(float))) {
      *((double*) y) = wasm_f64x2_extract_lane(vy, 0);
      vy = wasm_v32x4_shuffle(vy, vy, 2, 3, 2, 3);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      *y = wasm_f32x4_extract_lane(vy, 0);
    }
  }
}
