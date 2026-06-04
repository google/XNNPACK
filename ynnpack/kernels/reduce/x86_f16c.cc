// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <cstddef>

#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/x86_vec256.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/sum.h"

namespace ynn {

using simd::f16x8;
using simd::f16x16;
using simd::f32x8;

SUM_FLOAT_K1_KERNEL(sum_k1_fp16_fp32_f16c, half, float, 0, 1, identity);
SUM_FLOAT_KN_KERNEL(sum_kn_fp16_fp32_f16c, half, float, 16, identity);

SUM_FLOAT_K1_KERNEL(sum_squared_k1_fp16_fp32_f16c, half, float, 0, 1, square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_fp16_fp32_f16c, half, float, 16, square);

}  // namespace ynn
