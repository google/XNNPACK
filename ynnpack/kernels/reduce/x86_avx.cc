// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <cstddef>

#include "ynnpack/base/simd/x86_avx.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max_accumulator.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

using simd::f32x16;
using simd::f32x8;
using simd::f64x4;
using f64x8 = simd::vec<double, 8>;

MIN_MAX_KERNEL(min_max_fp32_4x8_avx, f32x8, f32x8, float, 8);
MIN_MAX_KERNEL(min_max_fp64_4x4_avx, f64x4, f64x4, double, 4);

MIN_MAX_KERNEL(min_fp32_4x8_avx, f32x8, dummy_t, float, 8);
MIN_MAX_KERNEL(min_fp64_4x4_avx, f64x4, dummy_t, double, 4);

MIN_MAX_KERNEL(max_fp32_4x8_avx, dummy_t, f32x8, float, 8);
MIN_MAX_KERNEL(max_fp64_4x4_avx, dummy_t, f64x4, double, 4);

SUM_KERNEL(sum_fp32_avx, f32x16, float, float, 16);
SUM_KERNEL(sum_fp64_avx, f64x8, double, double, 8);

SUM_SQUARED_KERNEL(sum_squared_fp32_avx, f32x16, float, float, 16);
SUM_SQUARED_KERNEL(sum_squared_fp64_avx, f64x8, double, double, 8);

}  // namespace ynn
