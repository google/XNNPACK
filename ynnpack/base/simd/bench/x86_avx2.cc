// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_vec256.h"

// This must be included last
#include "ynnpack/base/simd/bench/generic.h"

namespace ynn {
namespace simd {

BENCH_UNARY(avx2, floor_log2, f32, 8);
BENCH_UNARY(avx2, floor_log2, f64, 4);
BENCH_UNARY(avx2, exp2_round, f32, 8);
BENCH_UNARY(avx2, exp2_round, f64, 4);

BENCH_CAST(avx2, f32, bf16, 16);

BENCH_UNARY(avx2, exp, f32, 8);
BENCH_UNARY(avx2, exp, f64, 4);
BENCH_UNARY(avx2, expm1, f32, 8);
BENCH_UNARY(avx2, expm1, f64, 4);
BENCH_UNARY(avx2, log, f32, 8);
BENCH_UNARY(avx2, log, f64, 4);
BENCH_UNARY(avx2, log1p, f32, 8);
BENCH_UNARY(avx2, log1p, f64, 4);
BENCH_UNARY(avx2, erf, f32, 8);
BENCH_UNARY(avx2, erf, f64, 4);
BENCH_UNARY(avx2, tanh, f32, 8);
BENCH_UNARY(avx2, tanh, f64, 4);

BENCH_UNARY(avx2, approx_erf, f32, 8);
BENCH_UNARY(avx2, approx_tanh, f32, 8);

}  // namespace simd
}  // namespace ynn
