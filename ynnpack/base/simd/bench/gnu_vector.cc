// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/gnu_vector.h"

// This must be included last
#include "ynnpack/base/simd/bench/generic.h"

namespace ynn {
namespace simd {

BENCH_PARTIAL_LOAD_STORE(gnu_vector, s8, 16);
BENCH_PARTIAL_LOAD_STORE(gnu_vector, s16, 8);
BENCH_PARTIAL_LOAD_STORE(gnu_vector, s32, 4);

BENCH_FMA(gnu_vector, f32, 4);

BENCH_UNARY(gnu_vector, floor_log2, f32, 4);
BENCH_UNARY(gnu_vector, floor_log2, f64, 2);
BENCH_UNARY(gnu_vector, exp2_round, f32, 4);
BENCH_UNARY(gnu_vector, exp2_round, f64, 2);

BENCH_UNARY(gnu_vector, exp, f32, 4);
BENCH_UNARY(gnu_vector, exp, f64, 2);
BENCH_UNARY(gnu_vector, expm1, f32, 4);
BENCH_UNARY(gnu_vector, expm1, f64, 2);
BENCH_UNARY(gnu_vector, log, f32, 4);
BENCH_UNARY(gnu_vector, log, f64, 2);
BENCH_UNARY(gnu_vector, log1p, f32, 4);
BENCH_UNARY(gnu_vector, log1p, f64, 2);
BENCH_UNARY(gnu_vector, erf, f32, 4);
BENCH_UNARY(gnu_vector, erf, f64, 2);
BENCH_UNARY(gnu_vector, tanh, f32, 4);
BENCH_UNARY(gnu_vector, tanh, f64, 2);

BENCH_UNARY(gnu_vector, approx_erf, f32, 4);
BENCH_UNARY(gnu_vector, approx_tanh, f32, 4);

}  // namespace simd
}  // namespace ynn
