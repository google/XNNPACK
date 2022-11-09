// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdint.h>
#include <stddef.h>

#include <wasm_simd128.h>

#include <xnnpack/requantization-stubs.h>


void xnn_qs8_requantize_fp32__wasmsimd(
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

  const v128_t vscale = wasm_f32x4_splat(scale);
  const v128_t vfmin = wasm_f32x4_splat((float) ((int32_t) qmin - (int32_t) zero_point));
  const v128_t vfmax = wasm_f32x4_splat((float) ((int32_t) qmax - (int32_t) zero_point));
  const v128_t vfmagic = wasm_f32x4_const_splat(12582912.0f);
  const v128_t vimagic = wasm_i32x4_splat(INT32_C(0x4B400000) - (int32_t) zero_point);
  for (; n != 0; n -= 16) {
    const v128_t x = wasm_v128_load(input);
    const v128_t y = wasm_v128_load(input + 4);
    const v128_t z = wasm_v128_load(input + 8);
    const v128_t w = wasm_v128_load(input + 12);
    input += 16;

    // Convert int32_t input to FP32 and multiply by FP32 scale.
    // Both operations involve statistically unbiased roundings:
    // - Large int32_t values can't be exactly represented as FP32. The conversion instruction in WAsm SIMD would
    //   round it to nearest FP32 value with ties to even.
    // - Product of two FP32 values is generally not exactly representation as an FP32 value, and will be rounded
    //   to nearest FP32 value with ties to even.
    const v128_t x_scaled = wasm_f32x4_mul(wasm_f32x4_convert_i32x4(x), vscale);
    const v128_t y_scaled = wasm_f32x4_mul(wasm_f32x4_convert_i32x4(y), vscale);
    const v128_t z_scaled = wasm_f32x4_mul(wasm_f32x4_convert_i32x4(z), vscale);
    const v128_t w_scaled = wasm_f32x4_mul(wasm_f32x4_convert_i32x4(w), vscale);

    // WAsm SIMD offers only a floating-point to integer conversion instruction with rounding towards zero.
    // In lieu of conversion instruction with rounding-to-nearest-even, we use a magic trick of adding a large
    // number (1.5 * 2**23) to scaled value to cause rounding to integer, and then substracing this magic number as
    // integer. This trick works only in a limited range (absolute value of input must be less than 2**22), so
    // generally we have to clamp input to this range before using the magic. However, clamping to any smaller range
    // works just as well, and thus we clamp to [qmin - zero point, qmax - zero point] range so that after we add
    // zero point to the result, it gets into target [qmin, qmax] range.
    const v128_t x_clamped = wasm_f32x4_min(wasm_f32x4_max(x_scaled, vfmin), vfmax);
    const v128_t y_clamped = wasm_f32x4_min(wasm_f32x4_max(y_scaled, vfmin), vfmax);
    const v128_t z_clamped = wasm_f32x4_min(wasm_f32x4_max(z_scaled, vfmin), vfmax);
    const v128_t w_clamped = wasm_f32x4_min(wasm_f32x4_max(w_scaled, vfmin), vfmax);

    // Conversion to integer using the "magic trick". Rounding is performed in the output of addition operation,
    // and result is rounded to nearest even integer with ties to even.
    const v128_t x_biased = wasm_i32x4_sub(wasm_f32x4_add(x_clamped, vfmagic), vimagic);
    const v128_t y_biased = wasm_i32x4_sub(wasm_f32x4_add(y_clamped, vfmagic), vimagic);
    const v128_t z_biased = wasm_i32x4_sub(wasm_f32x4_add(z_clamped, vfmagic), vimagic);
    const v128_t w_biased = wasm_i32x4_sub(wasm_f32x4_add(w_clamped, vfmagic), vimagic);

    // Select low 8 bits of each 32-bit integer in the vectors for the output.
    // Since result is already clamped to [qmin, qmax] subrange of [0, 255], saturation is not needed.
    const v128_t xy_packed = wasm_v16x8_shuffle(x_biased, y_biased, 0, 2, 4, 6, 8, 10, 12, 14);
    const v128_t zw_packed = wasm_v16x8_shuffle(z_biased, w_biased, 0, 2, 4, 6, 8, 10, 12, 14);
    const v128_t xyzw_packed = wasm_v8x16_shuffle(xy_packed, zw_packed, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);

    // 4x f32x4.convert_i32x4_s
    // 4x f32x4.mul
    // 4x f32x4.max
    // 4x f32x4.min
    // 4x f32x4.add
    // 4x i32x4.sub
    // 3x v8x16.shuffle
    // ---------------------
    // 29 instructions total

    wasm_v128_store(output, xyzw_packed);
    output += 16;
  }
}
