// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/wasm_simd128.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum.h"

namespace ynn {

using simd::f32x4;
using simd::f32x8;
using simd::s16x8;
using simd::s8x16;
using simd::u8x16;

MIN_MAX_K1_KERNEL(min_max_fp32_k1_simd128, f32x4, f32x4, float, 4);
MIN_MAX_KN_KERNEL(min_max_fp32_kn_simd128, f32x4, f32x4, float, 4);
MIN_MAX_K1_KERNEL(min_max_uint8_k1_simd128, u8x16, u8x16, uint8_t, 16);
MIN_MAX_KN_KERNEL(min_max_uint8_kn_simd128, u8x16, u8x16, uint8_t, 16);
MIN_MAX_K1_KERNEL(min_max_int8_k1_simd128, s8x16, s8x16, int8_t, 16);
MIN_MAX_KN_KERNEL(min_max_int8_kn_simd128, s8x16, s8x16, int8_t, 16);

MIN_MAX_K1_KERNEL(min_fp32_k1_simd128, f32x4, dummy_t, float, 4);
MIN_MAX_KN_KERNEL(min_fp32_kn_simd128, f32x4, dummy_t, float, 4);
MIN_MAX_K1_KERNEL(min_uint8_k1_simd128, u8x16, dummy_t, uint8_t, 16);
MIN_MAX_KN_KERNEL(min_uint8_kn_simd128, u8x16, dummy_t, uint8_t, 16);
MIN_MAX_K1_KERNEL(min_int8_k1_simd128, s8x16, dummy_t, int8_t, 16);
MIN_MAX_KN_KERNEL(min_int8_kn_simd128, s8x16, dummy_t, int8_t, 16);

MIN_MAX_K1_KERNEL(max_fp32_k1_simd128, dummy_t, f32x4, float, 4);
MIN_MAX_KN_KERNEL(max_fp32_kn_simd128, dummy_t, f32x4, float, 4);
MIN_MAX_K1_KERNEL(max_uint8_k1_simd128, dummy_t, u8x16, uint8_t, 16);
MIN_MAX_KN_KERNEL(max_uint8_kn_simd128, dummy_t, u8x16, uint8_t, 16);
MIN_MAX_K1_KERNEL(max_int8_k1_simd128, dummy_t, s8x16, int8_t, 16);
MIN_MAX_KN_KERNEL(max_int8_kn_simd128, dummy_t, s8x16, int8_t, 16);

SUM_K1_KERNEL(sum_fp32_k1_simd128, float, float, 4, 1, identity);
SUM_KN_KERNEL(sum_fp32_kn_simd128, float, float, 4, identity);

SUM_K1_KERNEL(sum_squared_fp32_k1_simd128, float, float, 4, 1, square);
SUM_KN_KERNEL(sum_squared_fp32_kn_simd128, float, float, 4, square);

}  // namespace ynn
