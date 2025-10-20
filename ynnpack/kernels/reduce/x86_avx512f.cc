// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <cstring>

#include "ynnpack/base/simd/multi_vec.h"
#include "ynnpack/base/simd/x86_avx512.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max_accumulator.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

using simd::f32x16;

MIN_MAX_KERNEL(min_max_fp32_4x16_avx512f, f32x16, f32x16, float, 16);
MIN_MAX_KERNEL(min_fp32_4x16_avx512f, f32x16, dummy_t, float, 16);
MIN_MAX_KERNEL(max_fp32_4x16_avx512f, dummy_t, f32x16, float, 16);

using f32x16x16 = simd::multi_vec<f32x16, 16>;

void sum_fp32_avx512f(size_t n, size_t k3, size_t k2, size_t k1,
                      size_t a_stride_n, size_t a_stride_k3,
                      size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(float)) {
    tiled_reduce<sum_accumulator_k1_1<f32x16x16, f32x16x16>, float, float>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const float*>(a),
        /*C_stride_m=*/0, reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<f32x16, 16>, float, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const float*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

}  // namespace ynn
