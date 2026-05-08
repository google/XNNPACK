// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <cstddef>

#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/x86_f16c.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

using simd::f16x8;
using simd::f16x16;
using simd::f32x8;
using simd::f32x16;

SUM_KERNEL(sum_fp16_fp32_f16c, f32x16, half, float, 16);

SUM_SQUARED_KERNEL(sum_squared_fp16_fp32_f16c, f32x16, half, float, 16);

}  // namespace ynn
