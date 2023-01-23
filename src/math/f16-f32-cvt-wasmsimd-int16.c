// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <wasm_simd128.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f16_f32_cvt__wasmsimd_int16(
    size_t n,
    const void* input,
    float* output)
{
  assert(n % (8 * sizeof(float)) == 0);

  const v128_t vsign_mask = wasm_i16x8_const_splat(0x8000);
  const v128_t vexp_offset = wasm_i16x8_const_splat(0x7000);
  const v128_t vexp_scale = wasm_f32x4_const_splat(0x1.0p-112f);
  const v128_t vmagic_mask = wasm_i16x8_const_splat(0x3F00);
  const v128_t vmagic_bias = wasm_f32x4_const_splat(0.5f);
  const v128_t vdenorm_cutoff = wasm_i16x8_const_splat(0x0400);

  const uint16_t* i = (const uint16_t*) input;
  for (; n != 0; n -= 8 * sizeof(float)) {
    const v128_t vh = wasm_v128_load(i);
    i += 8;

    const v128_t vsign = wasm_v128_and(vh, vsign_mask);

    const v128_t vnonsign = wasm_v128_xor(vh, vsign);

    const v128_t vprenorm_lo = wasm_i16x8_shl(vnonsign, 13);
    const v128_t vprenorm_hi = wasm_i16x8_add(wasm_u16x8_shr(vnonsign, 3), vexp_offset);

    const v128_t vnorm_lo = wasm_f32x4_mul(wasm_v16x8_shuffle(vprenorm_lo, vprenorm_hi, 0,  8, 1,  9, 2, 10, 3, 11), vexp_scale);
    const v128_t vnorm_hi = wasm_f32x4_mul(wasm_v16x8_shuffle(vprenorm_lo, vprenorm_hi, 4, 12, 5, 13, 6, 14, 7, 15), vexp_scale);

    const v128_t vdenorm_lo = wasm_f32x4_sub(wasm_v16x8_shuffle(vnonsign, vmagic_mask, 0,  8, 1,  9, 2, 10, 3, 11), vmagic_bias);
    const v128_t vdenorm_hi = wasm_f32x4_sub(wasm_v16x8_shuffle(vnonsign, vmagic_mask, 4, 12, 5, 13, 6, 14, 7, 15), vmagic_bias);

    const v128_t vmask = wasm_i16x8_gt(vnonsign, vdenorm_cutoff);
    const v128_t vmask_lo = wasm_i32x4_extend_low_i16x8(vmask);
    const v128_t vmask_hi = wasm_i32x4_extend_high_i16x8(vmask);

    const v128_t vzero = wasm_i16x8_const_splat(0);
    const v128_t vf_lo = wasm_v128_or(wasm_v16x8_shuffle(vzero, vsign, 0,  8, 1,  9, 2, 10, 3, 11),
      wasm_v128_bitselect(vnorm_lo, vdenorm_lo, vmask_lo));
    const v128_t vf_hi = wasm_v128_or(wasm_v16x8_shuffle(vzero, vsign, 4, 12, 5, 13, 6, 14, 7, 15),
      wasm_v128_bitselect(vnorm_hi, vdenorm_hi, vmask_hi));

    wasm_v128_store(output, vf_lo);
    wasm_v128_store(output + 4, vf_hi);
    output += 8;
  }
}
