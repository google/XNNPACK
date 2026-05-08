// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_avx.h"

#include <immintrin.h>

#include <cstddef>

#include "ynnpack/kernels/reduce/min_max.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum.h"

namespace ynn {

using simd::f32x16;
using simd::f32x8;
using simd::f64x4;

MIN_MAX_K1_KERNEL(min_max_fp32_k1_avx, f32x8, f32x8, float, 8);
MIN_MAX_KN_KERNEL(min_max_fp32_kn_avx, f32x8, f32x8, float, 8);
MIN_MAX_K1_KERNEL(min_fp32_k1_avx, f32x8, dummy_t, float, 8);
MIN_MAX_KN_KERNEL(min_fp32_kn_avx, f32x8, dummy_t, float, 8);
MIN_MAX_K1_KERNEL(max_fp32_k1_avx, dummy_t, f32x8, float, 8);
MIN_MAX_KN_KERNEL(max_fp32_kn_avx, dummy_t, f32x8, float, 8);

MIN_MAX_K1_KERNEL(min_max_fp64_k1_avx, f64x4, f64x4, double, 4);
MIN_MAX_KN_KERNEL(min_max_fp64_kn_avx, f64x4, f64x4, double, 4);
MIN_MAX_K1_KERNEL(min_fp64_k1_avx, f64x4, dummy_t, double, 4);
MIN_MAX_KN_KERNEL(min_fp64_kn_avx, f64x4, dummy_t, double, 4);
MIN_MAX_K1_KERNEL(max_fp64_k1_avx, dummy_t, f64x4, double, 4);
MIN_MAX_KN_KERNEL(max_fp64_kn_avx, dummy_t, f64x4, double, 4);

SUM_K1_KERNEL(sum_fp64_k1_avx, double, double, consistent_tile_k_fp64, 1,
              identity);
SUM_KN_KERNEL(sum_fp64_kn_avx, double, double, 4, identity);
SUM_K1_KERNEL(sum_fp32_k1_avx, float, float, consistent_tile_k_fp32, 1,
              identity);
SUM_KN_KERNEL(sum_fp32_kn_avx, float, float, 8, identity);

SUM_K1_KERNEL(sum_squared_fp64_k1_avx, double, double, consistent_tile_k_fp64,
              1, square);
SUM_KN_KERNEL(sum_squared_fp64_kn_avx, double, double, 4, square);
SUM_K1_KERNEL(sum_squared_fp32_k1_avx, float, float, consistent_tile_k_fp32, 1,
              square);
SUM_KN_KERNEL(sum_squared_fp32_kn_avx, float, float, 8, square);

}  // namespace ynn
