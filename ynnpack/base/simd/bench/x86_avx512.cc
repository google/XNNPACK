// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/x86_vec512.h"

// This must be included last
#include "ynnpack/base/simd/bench/generic.h"

namespace ynn {
namespace simd {

BENCH_PARTIAL_LOAD_STORE(avx512, s8, 16);
BENCH_PARTIAL_LOAD_STORE(avx512, s16, 8);
BENCH_PARTIAL_LOAD_STORE(avx512, s32, 4);

BENCH_PARTIAL_LOAD_STORE(avx512, s8, 32);
BENCH_PARTIAL_LOAD_STORE(avx512, s16, 16);
BENCH_PARTIAL_LOAD_STORE(avx512, s32, 8);

BENCH_PARTIAL_LOAD_STORE(avx512, s8, 64);
BENCH_PARTIAL_LOAD_STORE(avx512, s16, 32);
BENCH_PARTIAL_LOAD_STORE(avx512, s32, 16);

BENCH_FMA(avx512, f32, 16);

BENCH_CAST(avx512, f32, bf16, 32);

BENCH_UNARY(avx512, floor_log2, f32, 16);
BENCH_UNARY(avx512, floor_log2, f32, 8);
BENCH_UNARY(avx512, floor_log2, f32, 4);
BENCH_UNARY(avx512, floor_log2, f64, 8);
BENCH_UNARY(avx512, floor_log2, f64, 4);
BENCH_UNARY(avx512, floor_log2, f64, 2);
BENCH_UNARY(avx512, exp2_round, f32, 16);
BENCH_UNARY(avx512, exp2_round, f64, 8);

BENCH_UNARY(avx512, exp, f32, 16);
BENCH_UNARY(avx512, exp, f64, 8);
BENCH_UNARY(avx512, expm1, f32, 16);
BENCH_UNARY(avx512, expm1, f64, 8);
BENCH_UNARY(avx512, log, f32, 16);
BENCH_UNARY(avx512, log, f64, 8);
BENCH_UNARY(avx512, log1p, f32, 16);
BENCH_UNARY(avx512, log1p, f64, 8);
BENCH_UNARY(avx512, erf, f32, 16);
BENCH_UNARY(avx512, erf, f64, 8);
BENCH_UNARY(avx512, tanh, f32, 16);
BENCH_UNARY(avx512, tanh, f64, 8);

BENCH_UNARY(avx512, fast_erf, f32, 16);

}  // namespace simd
}  // namespace ynn
