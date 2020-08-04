// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdint.h>
#include <stddef.h>

#include <wasm_simd128.h>

#include <fp16/bitcasts.h>

#include <xnnpack/requantization-stubs.h>


void xnn_qs8_requantize_q31__wasmsimd(
    size_t n,
    const int32_t* input,
    float scale,
    int8_t zero_point,
    int8_t qmin,
    int8_t qmax,
    int8_t* output)
{
  assert(n % 16 == 0);
  assert(scale < 1.0f);
  assert(scale >= 0x1.0p-32f);

  // Compute requantization parameters.
  const uint32_t scale_bits = fp32_to_bits(scale);

  // Multiplier is in [0x40000000, 0x7FFFFF80] range.
  const int32_t multiplier = (int32_t) (((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  // Shift is in [0, 31] range.
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
  assert(shift >= 0);
  assert(shift < 32);

  const v128_t vzero = wasm_f64x2_splat(0.0);
  const v128_t vmultiplier = wasm_i64x2_make((int64_t) multiplier, (int64_t) multiplier);
  const v128_t vzero_point = wasm_i16x8_splat((int16_t) zero_point);

  const v128_t vqmin = wasm_i8x16_splat(qmin);
  const v128_t vqmax = wasm_i8x16_splat(qmax);
  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
  const v128_t vremainder_mask = wasm_i32x4_splat((int32_t) remainder_mask);
  const v128_t vthreshold = wasm_i32x4_splat((int32_t) (remainder_mask >> 1));
  const v128_t vq31rounding = wasm_i64x2_splat(UINT64_C(0x40000000));
  for (; n != 0; n -= 16) {
    const v128_t x = wasm_v128_load(input);
    const v128_t y = wasm_v128_load(input + 4);
    const v128_t z = wasm_v128_load(input + 8);
    const v128_t w = wasm_v128_load(input + 12);
    input += 16;

    const v128_t x_sign = wasm_i32x4_lt(x, vzero);
    const v128_t y_sign = wasm_i32x4_lt(y, vzero);
    const v128_t z_sign = wasm_i32x4_lt(z, vzero);
    const v128_t w_sign = wasm_i32x4_lt(w, vzero);

    const v128_t x_even = wasm_v32x4_shuffle(x, x_sign, 0, 4, 2, 6);
    const v128_t y_even = wasm_v32x4_shuffle(y, y_sign, 0, 4, 2, 6);
    const v128_t z_even = wasm_v32x4_shuffle(z, z_sign, 0, 4, 2, 6);
    const v128_t w_even = wasm_v32x4_shuffle(w, w_sign, 0, 4, 2, 6);

    const v128_t x_odd = wasm_v32x4_shuffle(x, x_sign, 1, 5, 3, 7);
    const v128_t y_odd = wasm_v32x4_shuffle(y, y_sign, 1, 5, 3, 7);
    const v128_t z_odd = wasm_v32x4_shuffle(z, z_sign, 1, 5, 3, 7);
    const v128_t w_odd = wasm_v32x4_shuffle(w, w_sign, 1, 5, 3, 7);

    const v128_t x_product_even = wasm_i64x2_add(wasm_i64x2_mul(x_even, vmultiplier), vq31rounding);
    const v128_t y_product_even = wasm_i64x2_add(wasm_i64x2_mul(y_even, vmultiplier), vq31rounding);
    const v128_t z_product_even = wasm_i64x2_add(wasm_i64x2_mul(z_even, vmultiplier), vq31rounding);
    const v128_t w_product_even = wasm_i64x2_add(wasm_i64x2_mul(w_even, vmultiplier), vq31rounding);

    const v128_t x_product_odd = wasm_i64x2_add(wasm_i64x2_mul(x_odd, vmultiplier), vq31rounding);
    const v128_t y_product_odd = wasm_i64x2_add(wasm_i64x2_mul(y_odd, vmultiplier), vq31rounding);
    const v128_t z_product_odd = wasm_i64x2_add(wasm_i64x2_mul(z_odd, vmultiplier), vq31rounding);
    const v128_t w_product_odd = wasm_i64x2_add(wasm_i64x2_mul(w_odd, vmultiplier), vq31rounding);

    const v128_t x_q31product_even = wasm_u64x2_shr(x_product_even, 31);
    const v128_t x_q31product_odd = wasm_i64x2_add(x_product_odd, x_product_odd);
    const v128_t y_q31product_even = wasm_u64x2_shr(y_product_even, 31);
    const v128_t y_q31product_odd = wasm_i64x2_add(y_product_odd, y_product_odd);
    const v128_t z_q31product_even = wasm_u64x2_shr(z_product_even, 31);
    const v128_t z_q31product_odd = wasm_i64x2_add(z_product_odd, z_product_odd);
    const v128_t w_q31product_even = wasm_u64x2_shr(w_product_even, 31);
    const v128_t w_q31product_odd = wasm_i64x2_add(w_product_odd, w_product_odd);

    const v128_t x_q31product = wasm_v32x4_shuffle(x_q31product_even, x_q31product_odd, 0, 5, 2, 7);
    const v128_t y_q31product = wasm_v32x4_shuffle(y_q31product_even, y_q31product_odd, 0, 5, 2, 7);
    const v128_t z_q31product = wasm_v32x4_shuffle(z_q31product_even, z_q31product_odd, 0, 5, 2, 7);
    const v128_t w_q31product = wasm_v32x4_shuffle(w_q31product_even, w_q31product_odd, 0, 5, 2, 7);

    const v128_t x_remainder =
        wasm_i32x4_add(wasm_v128_and(x_q31product, vremainder_mask), wasm_i32x4_lt(x_q31product, vzero));
    const v128_t y_remainder =
        wasm_i32x4_add(wasm_v128_and(y_q31product, vremainder_mask), wasm_i32x4_lt(y_q31product, vzero));
    const v128_t z_remainder =
        wasm_i32x4_add(wasm_v128_and(z_q31product, vremainder_mask), wasm_i32x4_lt(z_q31product, vzero));
    const v128_t w_remainder =
        wasm_i32x4_add(wasm_v128_and(w_q31product, vremainder_mask), wasm_i32x4_lt(w_q31product, vzero));

    const v128_t x_scaled =
        wasm_i32x4_sub(wasm_i32x4_shr(x_q31product, shift), wasm_i32x4_gt(x_remainder, vthreshold));
    const v128_t y_scaled =
        wasm_i32x4_sub(wasm_i32x4_shr(y_q31product, shift), wasm_i32x4_gt(y_remainder, vthreshold));
    const v128_t z_scaled =
        wasm_i32x4_sub(wasm_i32x4_shr(z_q31product, shift), wasm_i32x4_gt(z_remainder, vthreshold));
    const v128_t w_scaled =
        wasm_i32x4_sub(wasm_i32x4_shr(w_q31product, shift), wasm_i32x4_gt(w_remainder, vthreshold));

    const v128_t xy_packed = wasm_i16x8_add_saturate(wasm_i16x8_narrow_i32x4(x_scaled, y_scaled), vzero_point);
    const v128_t zw_packed = wasm_i16x8_add_saturate(wasm_i16x8_narrow_i32x4(z_scaled, w_scaled), vzero_point);
    const v128_t xyzw_packed = wasm_i8x16_narrow_i16x8(xy_packed, zw_packed);
    const v128_t xyzw_clamped = wasm_i8x16_min(wasm_i8x16_max(xyzw_packed, vqmin), vqmax);

    // 12x v128.shuffle
    // 8x i32x4.lt
    // 12x i64x2.add
    // 8x i64x2.mul
    // 4x i64x2.shr_u
    // 4x v128.and
    // 4x i32x4.add
    // 4x i32x4.sub
    // 4x i32x4.gt
    // 4x i32x4.shr_s
    // 2x i16x8.narrow_i32x4_s
    // 2x i16x8.add_saturate_s
    // 1x i8x16.narrow_i16x8_s
    // 1x i8x16.max_s
    // 1x i8x16.min_s
    // ---------------------
    // 71 instructions total

    wasm_v128_store(output, xyzw_clamped);
    output += 16;
  }
}
