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
#include "ynnpack/kernels/reduce/min_max_accumulator.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

using simd::f32x4;
using simd::f32x8;
using simd::s16x8;
using simd::s8x16;
using simd::u8x16;

MIN_MAX_KERNEL(min_max_fp32_4x4_wasm_simd128, f32x4, f32x4, float, 4);
MIN_MAX_KERNEL(min_max_uint8_4x16_wasm_simd128, u8x16, u8x16, uint8_t, 16);
MIN_MAX_KERNEL(min_max_int8_4x16_wasm_simd128, s8x16, s8x16, int8_t, 16);

MIN_MAX_KERNEL(min_fp32_4x4_wasm_simd128, f32x4, dummy_t, float, 4);
MIN_MAX_KERNEL(min_uint8_4x16_wasm_simd128, u8x16, dummy_t, uint8_t, 16);
MIN_MAX_KERNEL(min_int8_4x16_wasm_simd128, s8x16, dummy_t, int8_t, 16);

MIN_MAX_KERNEL(max_fp32_4x4_wasm_simd128, dummy_t, f32x4, float, 4);
MIN_MAX_KERNEL(max_uint8_4x16_wasm_simd128, dummy_t, u8x16, uint8_t, 16);
MIN_MAX_KERNEL(max_int8_4x16_wasm_simd128, dummy_t, s8x16, int8_t, 16);

void sum_fp32_wasm_simd128(size_t n, size_t k3, size_t k2, size_t k1,
                           size_t a_stride_n, size_t a_stride_k3,
                           size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(float)) {
    stream_reduce<sum_accumulator_k1_1<f32x4>, float, float>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const float*>(a),
        /*C_stride_m=*/0, reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_fp32<1, Identity>, float, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const float*>(a),
        /*C_stride_m=*/0, reinterpret_cast<float*>(c));
  }
}

void sum_squared_fp32_wasm_simd128(size_t n, size_t k3, size_t k2, size_t k1,
                                   size_t a_stride_n, size_t a_stride_k3,
                                   size_t a_stride_k2, const void* a, size_t,
                                   void* c) {
  if (k1 == 1 && a_stride_n == sizeof(float)) {
    stream_reduce<sum_accumulator_k1_1<f32x4, Square>, float, float>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const float*>(a),
        /*C_stride_m=*/0, reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_fp32<1, Square>, float, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const float*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

}  // namespace ynn
