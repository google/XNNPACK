// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/arm_neonfp16.h"

#include <arm_neon.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum.h"

namespace ynn {

using simd::f16x8;
using simd::f32x8;

SUM_K1_KERNEL(sum_fp16_fp32_k1_neonfp16, half, float, 8, 1, identity);
SUM_KN_KERNEL(sum_fp16_fp32_kn_neonfp16, half, float, 8, identity);

SUM_K1_KERNEL(sum_squared_fp16_fp32_k1_neonfp16, half, float, 8, 1, square);
SUM_KN_KERNEL(sum_squared_fp16_fp32_kn_neonfp16, half, float, 8, square);

}  // namespace ynn
