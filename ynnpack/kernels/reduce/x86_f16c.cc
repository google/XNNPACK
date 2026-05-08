// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_f16c.h"

#include <immintrin.h>

#include <cstddef>

#include "ynnpack/base/half.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/sum.h"

namespace ynn {

using simd::f16x8;
using simd::f16x16;
using simd::f32x8;
using simd::f32x16;

SUM_K1_KERNEL(sum_fp16_fp32_k1_f16c, half, float, consistent_tile_k_fp32, 1,
              identity);
SUM_KN_KERNEL(sum_fp16_fp32_kn_f16c, half, float, 16, identity);

SUM_K1_KERNEL(sum_squared_fp16_fp32_k1_f16c, half, float,
              consistent_tile_k_fp32, 1, square);
SUM_KN_KERNEL(sum_squared_fp16_fp32_kn_f16c, half, float, 16, square);

}  // namespace ynn
