// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/x86_vec256.h"

// This must be included last
#include "ynnpack/base/simd/bench/generic.h"

namespace ynn {
namespace simd {

BENCH_PARTIAL_LOAD_STORE(avx, s8, 32);
BENCH_PARTIAL_LOAD_STORE(avx, s16, 16);
BENCH_PARTIAL_LOAD_STORE(avx, s32, 8);

BENCH_UNARY(avx, exp, f32, 8);
BENCH_UNARY(avx, exp, f64, 4);
BENCH_UNARY(avx, expm1, f32, 8);
BENCH_UNARY(avx, expm1, f64, 4);
BENCH_UNARY(avx, log, f32, 8);
BENCH_UNARY(avx, log, f64, 4);
BENCH_UNARY(avx, log1p, f32, 8);
BENCH_UNARY(avx, log1p, f64, 4);
BENCH_UNARY(avx, erf, f32, 8);
BENCH_UNARY(avx, erf, f64, 4);

BENCH_UNARY(avx, fast_erf, f32, 8);

}  // namespace simd
}  // namespace ynn
