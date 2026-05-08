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

SUM_KERNEL(sum_fp64_neon, f64x2, double, double, 2);

SUM_SQUARED_KERNEL(sum_squared_fp64_neon, f64x2, double, double, 2);

}  // namespace ynn
