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
#include "ynnpack/kernels/reduce/min_max.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum.h"

namespace ynn {

using simd::f64x2;

MIN_MAX_K1_KERNEL(min_max_k1_fp64_neon, f64x2, f64x2, double, 2);
MIN_MAX_KN_KERNEL(min_max_kn_fp64_neon, f64x2, f64x2, double, 2);
MIN_MAX_K1_KERNEL(min_k1_fp64_neon, f64x2, dummy_t, double, 2);
MIN_MAX_KN_KERNEL(min_kn_fp64_neon, f64x2, dummy_t, double, 2);
MIN_MAX_K1_KERNEL(max_k1_fp64_neon, dummy_t, f64x2, double, 2);
MIN_MAX_KN_KERNEL(max_kn_fp64_neon, dummy_t, f64x2, double, 2);

SUM_FLOAT_K1_KERNEL(sum_k1_fp64_neon, double, double, 2, 1, identity);
SUM_FLOAT_KN_KERNEL(sum_kn_fp64_neon, double, double, 4, identity);

SUM_FLOAT_K1_KERNEL(sum_squared_k1_fp64_neon, double, double, 2, 1, square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_fp64_neon, double, double, 4, square);

}  // namespace ynn
