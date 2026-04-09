// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/simd/arm_neon.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max_accumulator.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

using simd::f64x2;

MIN_MAX_KERNEL(min_max_fp64_4x2_neon, f64x2, f64x2, double, 2);
MIN_MAX_KERNEL(min_fp64_4x2_neon, f64x2, dummy_t, double, 2);
MIN_MAX_KERNEL(max_fp64_4x2_neon, dummy_t, f64x2, double, 2);

void sum_fp64_neon(size_t n, size_t k3, size_t k2, size_t k1, size_t a_stride_n,
                   size_t a_stride_k3, size_t a_stride_k2, const void* a,
                   size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(double)) {
    stream_reduce<sum_accumulator_k1_1<f64x2>, double, double>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const double*>(a),
        /*C_stride_m=*/0, reinterpret_cast<double*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<f64x2, 2>, double, double>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const double*>(a), /*C_stride_m=*/0,
        reinterpret_cast<double*>(c));
  }
}

void sum_squared_fp64_neon(size_t n, size_t k3, size_t k2, size_t k1,
                           size_t a_stride_n, size_t a_stride_k3,
                           size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(double)) {
    stream_reduce<sum_accumulator_k1_1<f64x2, Square>, double, double>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const double*>(a),
        /*C_stride_m=*/0, reinterpret_cast<double*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<f64x2, 2, Square>, double, double>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const double*>(a), /*C_stride_m=*/0,
        reinterpret_cast<double*>(c));
  }
}

}  // namespace ynn
