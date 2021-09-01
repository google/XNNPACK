// Auto-generated file. Do not edit!
//   Template: src/f32-vrnd/vrndd-wasmsimd-cvt.c.in
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


void xnn_f32_vrndd_ukernel__wasmsimd_cvt_x4(
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
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(x);
    x += 4;

    const v128_t vintx = wasm_i32x4_trunc_sat_f32x4(vx);
    const v128_t vabsx = wasm_f32x4_abs(vx);
    const v128_t vprerndx = wasm_f32x4_convert_i32x4(vintx);
    const v128_t vrndmask = wasm_v128_andnot(wasm_f32x4_lt(vabsx, vmagic_number), vsign_mask);
    const v128_t vrndx = wasm_v128_bitselect(vprerndx, vx, vrndmask);
    const v128_t vadj = wasm_v128_and(wasm_f32x4_lt(vx, vrndx), vone);
    const v128_t vy = wasm_f32x4_sub(vrndx, vadj);

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
    const v128_t vadj = wasm_v128_and(wasm_f32x4_lt(vx, vrndx), vone);
    v128_t vy = wasm_f32x4_sub(vrndx, vadj);

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
