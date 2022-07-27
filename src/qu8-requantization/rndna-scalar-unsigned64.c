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

#include <xnnpack/math.h>
#include <xnnpack/requantization-stubs.h>


void xnn_qu8_requantize_rndna__scalar_unsigned64(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output)
{
  assert(n % 4 == 0);
  assert(scale < 1.0f);
  assert(scale >= 0x1.0p-32f);

  const uint32_t scale_bits = float_as_uint32(scale);
  const uint32_t multiplier = (scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000);
  const uint32_t shift = 127 + 23 - (scale_bits >> 23);
  assert(shift >= 24);
  assert(shift < 56);

  const uint64_t rounding = UINT64_C(1) << (shift - 1);
  const int32_t smin = (int32_t) (uint32_t) qmin - (int32_t) (uint32_t) zero_point;
  const int32_t smax = (int32_t) (uint32_t) qmax - (int32_t) (uint32_t) zero_point;
  for (; n != 0; n -= 4) {
    const int32_t x = input[0];
    const int32_t y = input[1];
    const int32_t z = input[2];
    const int32_t w = input[3];
    input += 4;

    // Compute absolute value of input as unsigned 32-bit int.
    // All further computations will work with unsigned values to avoid undefined behaviour on signed operations.
    const uint32_t x_abs = (x >= 0) ? (uint32_t) x : -(uint32_t) x;
    const uint32_t y_abs = (y >= 0) ? (uint32_t) y : -(uint32_t) y;
    const uint32_t z_abs = (z >= 0) ? (uint32_t) z : -(uint32_t) z;
    const uint32_t w_abs = (w >= 0) ? (uint32_t) w : -(uint32_t) w;

    // Compute full 64-bit product of 32-bit factors.
    const uint64_t x_product = (uint64_t) x_abs * (uint64_t) multiplier;
    const uint64_t y_product = (uint64_t) y_abs * (uint64_t) multiplier;
    const uint64_t z_product = (uint64_t) z_abs * (uint64_t) multiplier;
    const uint64_t w_product = (uint64_t) w_abs * (uint64_t) multiplier;

    // Shift the full 64-bit product right with rounding.
    // Rounding is performed towards closest integer, with midpoints rounded up (same as away from zero).
    //
    // Note that although rounding is precomputed, it is dependent on shift value, and on processors with 64-bit
    // "right shift with rounding" instruction each line below can be represented by just one such instruction
    // (e.g. VRSHL.U64 on ARM NEON, URSHL in ARM64 Advanced SIMD).
    const uint32_t x_abs_scaled = (uint32_t) ((x_product + rounding) >> shift);
    const uint32_t y_abs_scaled = (uint32_t) ((y_product + rounding) >> shift);
    const uint32_t z_abs_scaled = (uint32_t) ((z_product + rounding) >> shift);
    const uint32_t w_abs_scaled = (uint32_t) ((w_product + rounding) >> shift);

    // Copy the sign of input to scaled absolute input value.
    //
    // On x86 processors with SSSE3 instruction set, this operation nicely maps to PSIGND instruction.
    const int32_t x_scaled = (int32_t) (x >= 0 ? x_abs_scaled : -x_abs_scaled);
    const int32_t y_scaled = (int32_t) (y >= 0 ? y_abs_scaled : -y_abs_scaled);
    const int32_t z_scaled = (int32_t) (z >= 0 ? z_abs_scaled : -z_abs_scaled);
    const int32_t w_scaled = (int32_t) (w >= 0 ? w_abs_scaled : -w_abs_scaled);

    // Clamp scaled value with zero point between (qmin - zero point) and (qmax - zero point).
    const int32_t x_clamped = math_min_s32(math_max_s32(x_scaled, smin), smax);
    const int32_t y_clamped = math_min_s32(math_max_s32(y_scaled, smin), smax);
    const int32_t z_clamped = math_min_s32(math_max_s32(z_scaled, smin), smax);
    const int32_t w_clamped = math_min_s32(math_max_s32(w_scaled, smin), smax);

    // Add zero point to clamped value.
    // The result is guaranteed to be in [qmin, qmax] range.
    //
    // This addition can not be safely done before clamping, because scaled values are in [-2147483520, 2147483519]
    // range, so addition of zero point (which can be up to 255) can overflow signed 32-bit integer.
    const int32_t x_biased = x_clamped + zero_point;
    const int32_t y_biased = y_clamped + zero_point;
    const int32_t z_biased = z_clamped + zero_point;
    const int32_t w_biased = w_clamped + zero_point;

    output[0] = (uint8_t) x_biased;
    output[1] = (uint8_t) y_biased;
    output[2] = (uint8_t) z_biased;
    output[3] = (uint8_t) w_biased;
    output += 4;
  }
}
