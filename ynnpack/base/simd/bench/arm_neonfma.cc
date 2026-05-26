// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/arm_vec128.h"

// This must be included last
#include "ynnpack/base/simd/bench/generic.h"

namespace ynn {
namespace simd {

BENCH_UNARY(neonfma, exp, f32, 4);
BENCH_UNARY(neonfma, expm1, f32, 4);
BENCH_UNARY(neonfma, log, f32, 4);
BENCH_UNARY(neonfma, log1p, f32, 4);
BENCH_UNARY(neonfma, erf, f32, 4);
BENCH_UNARY(neonfma, tanh, f32, 4);

BENCH_UNARY(neonfma, fast_erf, f32, 4);

}  // namespace simd
}  // namespace ynn
