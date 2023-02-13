// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <wasm_simd128.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_qu8_cvt__wasmsimd(
    size_t n,
    const float* input,
    uint8_t* output,
    uint8_t output_zero_point)
{
  assert(n % (16 * sizeof(uint8_t)) == 0);

  const v128_t vmin = wasm_f32x4_splat(12582912.0f - (float) (int32_t) output_zero_point);
  const v128_t vfmagic = wasm_f32x4_const_splat(12582912.0f);
  const v128_t vimagic = wasm_i32x4_splat(INT32_C(0x4B400000) - (int32_t) output_zero_point);
  for (; n != 0; n -= 16 * sizeof(uint8_t)) {
    const v128_t vx_ll = wasm_v128_load(input);
    const v128_t vx_lh = wasm_v128_load(input + 4);
    const v128_t vx_hl = wasm_v128_load(input + 8);
    const v128_t vx_hh = wasm_v128_load(input + 12);
    input += 16;

    v128_t vy_ll = wasm_f32x4_add(vx_ll, vfmagic);
    v128_t vy_lh = wasm_f32x4_add(vx_lh, vfmagic);
    v128_t vy_hl = wasm_f32x4_add(vx_hl, vfmagic);
    v128_t vy_hh = wasm_f32x4_add(vx_hh, vfmagic);

    vy_ll = wasm_i32x4_max(vy_ll, vmin);
    vy_lh = wasm_i32x4_max(vy_lh, vmin);
    vy_hl = wasm_i32x4_max(vy_hl, vmin);
    vy_hh = wasm_i32x4_max(vy_hh, vmin);

    vy_ll = wasm_i32x4_sub(vy_ll, vimagic);
    vy_lh = wasm_i32x4_sub(vy_lh, vimagic);
    vy_hl = wasm_i32x4_sub(vy_hl, vimagic);
    vy_hh = wasm_i32x4_sub(vy_hh, vimagic);

    const v128_t vy_lo = wasm_i16x8_narrow_i32x4(vy_ll, vy_lh);
    const v128_t vy_hi = wasm_i16x8_narrow_i32x4(vy_hl, vy_hh);

    const v128_t vout = wasm_u8x16_narrow_i16x8(vy_lo, vy_hi);
    wasm_v128_store(output, vout);
    output += 16;
  }
}
