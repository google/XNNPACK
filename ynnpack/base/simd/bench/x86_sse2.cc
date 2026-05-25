// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/emulate_fma.h"
#include "ynnpack/base/simd/x86_vec128.h"

// This must be included last
#include "ynnpack/base/simd/bench/generic.h"

namespace ynn {
namespace simd {

BENCH_PARTIAL_LOAD_STORE(sse2, s8, 16);
BENCH_PARTIAL_LOAD_STORE(sse2, s16, 8);
BENCH_PARTIAL_LOAD_STORE(sse2, s32, 4);

BENCH_UNARY(sse2, floor_log2, f32, 4);
BENCH_UNARY(sse2, floor_log2, f64, 2);
BENCH_UNARY(sse2, exp2_round, f32, 4);
BENCH_UNARY(sse2, exp2_round, f64, 2);

BENCH_UNARY(sse2, exp, f32, 4);
BENCH_UNARY(sse2, exp, f64, 2);
BENCH_UNARY(sse2, expm1, f32, 4);
BENCH_UNARY(sse2, expm1, f64, 2);
BENCH_UNARY(sse2, log, f32, 4);
BENCH_UNARY(sse2, log, f64, 2);
BENCH_UNARY(sse2, log1p, f32, 4);
BENCH_UNARY(sse2, log1p, f64, 2);
BENCH_UNARY(sse2, erf, f32, 4);
BENCH_UNARY(sse2, erf, f64, 2);

BENCH_UNARY(sse2, fast_erf, f32, 4);

}  // namespace simd
}  // namespace ynn
