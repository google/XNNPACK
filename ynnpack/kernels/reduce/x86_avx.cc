// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <cstddef>

#include "ynnpack/base/simd/x86_vec256.h"
#include "ynnpack/kernels/reduce/min_max.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum.h"

namespace ynn {

using simd::f32x8;
using simd::f64x4;

MIN_MAX_K1_KERNEL(min_max_k1_fp32_avx, f32x8, f32x8, float, 8);
MIN_MAX_KN_KERNEL(min_max_kn_fp32_avx, f32x8, f32x8, float, 8);
MIN_MAX_K1_KERNEL(min_k1_fp32_avx, f32x8, dummy_t, float, 8);
MIN_MAX_KN_KERNEL(min_kn_fp32_avx, f32x8, dummy_t, float, 8);
MIN_MAX_K1_KERNEL(max_k1_fp32_avx, dummy_t, f32x8, float, 8);
MIN_MAX_KN_KERNEL(max_kn_fp32_avx, dummy_t, f32x8, float, 8);

MIN_MAX_K1_KERNEL(min_max_k1_fp64_avx, f64x4, f64x4, double, 4);
MIN_MAX_KN_KERNEL(min_max_kn_fp64_avx, f64x4, f64x4, double, 4);
MIN_MAX_K1_KERNEL(min_k1_fp64_avx, f64x4, dummy_t, double, 4);
MIN_MAX_KN_KERNEL(min_kn_fp64_avx, f64x4, dummy_t, double, 4);
MIN_MAX_K1_KERNEL(max_k1_fp64_avx, dummy_t, f64x4, double, 4);
MIN_MAX_KN_KERNEL(max_kn_fp64_avx, dummy_t, f64x4, double, 4);

SUM_FLOAT_K1_KERNEL(sum_k1_fp64_avx, double, double, 0, 1, identity);
SUM_FLOAT_KN_KERNEL(sum_kn_fp64_avx, double, double, 4, identity);
SUM_FLOAT_K1_KERNEL(sum_k1_fp32_avx, float, float, 0, 1, identity);
SUM_FLOAT_KN_KERNEL(sum_kn_fp32_avx, float, float, 8, identity);

SUM_FLOAT_K1_KERNEL(sum_squared_k1_fp64_avx, double, double, 0, 1, square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_fp64_avx, double, double, 4, square);
SUM_FLOAT_K1_KERNEL(sum_squared_k1_fp32_avx, float, float, 0, 1, square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_fp32_avx, float, float, 8, square);

}  // namespace ynn
