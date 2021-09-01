// Auto-generated file. Do not edit!
//   Template: src/f32-vrnd/vrndu-wasmsimd-cvt.c.in
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


void xnn_f32_vrndu_ukernel__wasmsimd_cvt_x8(
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

    const v128_t vintx0123 = wasm_i32x4_trunc_sat_f32x4(vx0123);
    const v128_t vabsx0123 = wasm_f32x4_abs(vx0123);
    const v128_t vintx4567 = wasm_i32x4_trunc_sat_f32x4(vx4567);
    const v128_t vabsx4567 = wasm_f32x4_abs(vx4567);

    const v128_t vprerndx0123 = wasm_f32x4_convert_i32x4(vintx0123);
    const v128_t vrndmask0123 = wasm_v128_andnot(wasm_f32x4_lt(vabsx0123, vmagic_number), vsign_mask);
    const v128_t vprerndx4567 = wasm_f32x4_convert_i32x4(vintx4567);
    const v128_t vrndmask4567 = wasm_v128_andnot(wasm_f32x4_lt(vabsx4567, vmagic_number), vsign_mask);

    const v128_t vrndx0123 = wasm_v128_bitselect(vprerndx0123, vx0123, vrndmask0123);
    const v128_t vrndx4567 = wasm_v128_bitselect(vprerndx4567, vx4567, vrndmask4567);

    const v128_t vadjmask0123 = wasm_v128_or(vsign_mask, wasm_f32x4_le(vx0123, vrndx0123));
    const v128_t vadjmask4567 = wasm_v128_or(vsign_mask, wasm_f32x4_le(vx4567, vrndx4567));

    const v128_t vadjrndx0123 = wasm_f32x4_add(vrndx0123, vone);
    const v128_t vadjrndx4567 = wasm_f32x4_add(vrndx4567, vone);

    const v128_t vy0123 = wasm_v128_bitselect(vrndx0123, vadjrndx0123, vadjmask0123);
    const v128_t vy4567 = wasm_v128_bitselect(vrndx4567, vadjrndx4567, vadjmask4567);

    wasm_v128_store(y, vy0123);
    wasm_v128_store(y + 4, vy4567);
    y += 8;
  }
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(x);
    x += 4;

    const v128_t vintx = wasm_i32x4_trunc_sat_f32x4(vx);
    const v128_t vabsx = wasm_f32x4_abs(vx);
    const v128_t vprerndx = wasm_f32x4_convert_i32x4(vintx);
    const v128_t vrndmask = wasm_v128_andnot(wasm_f32x4_lt(vabsx, vmagic_number), vsign_mask);
    const v128_t vrndx = wasm_v128_bitselect(vprerndx, vx, vrndmask);
    const v128_t vadjmask = wasm_v128_or(vsign_mask, wasm_f32x4_le(vx, vrndx));
    const v128_t vadjrndx = wasm_f32x4_add(vrndx, vone);
    const v128_t vy = wasm_v128_bitselect(vrndx, vadjrndx, vadjmask);

    wasm_v128_store(y, vy);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const v128_t vx = wasm_v128_load(x);

    const v128_t vintx = wasm_i32x4_trunc_sat_f32x4(vx);
    const v128_t vabsx = wasm_f32x4_abs(vx);
    const v128_t vprerndx = wasm_f32x4_convert_i32x4(vintx);
    const v128_t vrndmask = wasm_v128_andnot(wasm_f32x4_lt(vabsx, vmagic_number), vsign_mask);
    const v128_t vrndx = wasm_v128_bitselect(vprerndx, vx, vrndmask);
    const v128_t vadjmask = wasm_v128_or(vsign_mask, wasm_f32x4_le(vx, vrndx));
    const v128_t vadjrndx = wasm_f32x4_add(vrndx, vone);
    v128_t vy = wasm_v128_bitselect(vrndx, vadjrndx, vadjmask);

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
