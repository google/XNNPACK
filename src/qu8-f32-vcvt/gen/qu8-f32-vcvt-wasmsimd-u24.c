// Auto-generated file. Do not edit!
//   Template: src/qs8-f32-vcvt/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vcvt.h"


void xnn_qu8_f32_vcvt_ukernel__wasmsimd_u24(
    size_t batch,
    const uint8_t* input,
    float* output,
    const union xnn_qu8_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vminus_zero_point = wasm_v128_load64_splat(params->wasmsimd.minus_zero_point);
  const v128_t vscale = wasm_v128_load64_splat(params->wasmsimd.scale);
  for (; batch >= 24 * sizeof(uint8_t); batch -= 24 * sizeof(uint8_t)) {
    v128_t vx01234567 = wasm_u16x8_load8x8(input);
    v128_t vx89ABCDEF = wasm_u16x8_load8x8(input + 8);
    v128_t vxGHIJKLMN = wasm_u16x8_load8x8(input + 16);
    input += 24;

    vx01234567 = wasm_i16x8_add(vx01234567, vminus_zero_point);
    vx89ABCDEF = wasm_i16x8_add(vx89ABCDEF, vminus_zero_point);
    vxGHIJKLMN = wasm_i16x8_add(vxGHIJKLMN, vminus_zero_point);

    v128_t vy0123 = wasm_i32x4_extend_low_i16x8(vx01234567);
    v128_t vy4567 = wasm_i32x4_extend_high_i16x8(vx01234567);
    v128_t vy89AB = wasm_i32x4_extend_low_i16x8(vx89ABCDEF);
    v128_t vyCDEF = wasm_i32x4_extend_high_i16x8(vx89ABCDEF);
    v128_t vyGHIJ = wasm_i32x4_extend_low_i16x8(vxGHIJKLMN);
    v128_t vyKLMN = wasm_i32x4_extend_high_i16x8(vxGHIJKLMN);

    vy0123 = wasm_f32x4_convert_i32x4(vy0123);
    vy4567 = wasm_f32x4_convert_i32x4(vy4567);
    vy89AB = wasm_f32x4_convert_i32x4(vy89AB);
    vyCDEF = wasm_f32x4_convert_i32x4(vyCDEF);
    vyGHIJ = wasm_f32x4_convert_i32x4(vyGHIJ);
    vyKLMN = wasm_f32x4_convert_i32x4(vyKLMN);

    vy0123 = wasm_f32x4_mul(vy0123, vscale);
    vy4567 = wasm_f32x4_mul(vy4567, vscale);
    vy89AB = wasm_f32x4_mul(vy89AB, vscale);
    vyCDEF = wasm_f32x4_mul(vyCDEF, vscale);
    vyGHIJ = wasm_f32x4_mul(vyGHIJ, vscale);
    vyKLMN = wasm_f32x4_mul(vyKLMN, vscale);

    wasm_v128_store(output, vy0123);
    wasm_v128_store(output + 4, vy4567);
    wasm_v128_store(output + 8, vy89AB);
    wasm_v128_store(output + 12, vyCDEF);
    wasm_v128_store(output + 16, vyGHIJ);
    wasm_v128_store(output + 20, vyKLMN);
    output += 24;
  }
  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    v128_t vx = wasm_u16x8_load8x8(input);
    vx = wasm_i16x8_add(vx, vminus_zero_point);
    input += 8;

    v128_t vy_lo = wasm_i32x4_extend_low_i16x8(vx);
    v128_t vy_hi = wasm_i32x4_extend_high_i16x8(vx);

    vy_lo = wasm_f32x4_convert_i32x4(vy_lo);
    vy_hi = wasm_f32x4_convert_i32x4(vy_hi);

    vy_lo = wasm_f32x4_mul(vy_lo, vscale);
    vy_hi = wasm_f32x4_mul(vy_hi, vscale);

    wasm_v128_store(output, vy_lo);
    wasm_v128_store(output + 4, vy_hi);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 7 * sizeof(uint8_t));

    v128_t vx = wasm_u16x8_load8x8(input);
    vx = wasm_i16x8_add(vx, vminus_zero_point);
    input += 8;

    v128_t vy = wasm_i32x4_extend_low_i16x8(vx);
    vy = wasm_f32x4_convert_i32x4(vy);
    vy = wasm_f32x4_mul(vy, vscale);

    if (batch & (4 * sizeof(uint8_t))) {
      wasm_v128_store(output, vy); output += 4;
      vy = wasm_i32x4_extend_high_i16x8(vx);
      vy = wasm_f32x4_convert_i32x4(vy);
      vy = wasm_f32x4_mul(vy, vscale);
    }
    if (batch & (2 * sizeof(uint8_t))) {
      wasm_v128_store64_lane(output, vy, 0);
      vy = wasm_v64x2_shuffle(vy, vy, 1, 1);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      wasm_v128_store32_lane(output, vy, 0);
    }
  }
}
