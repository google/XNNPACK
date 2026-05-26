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

BENCH_PARTIAL_LOAD_STORE(neon, s8, 16);
BENCH_PARTIAL_LOAD_STORE(neon, s16, 8);
BENCH_PARTIAL_LOAD_STORE(neon, s32, 4);

BENCH_UNARY(neon, floor_log2, f32, 4);
BENCH_UNARY(neon, exp2_round, f32, 4);

BENCH_CAST(neon, f32, bf16, 4);

BENCH_UNARY(neon, exp, f32, 4);
BENCH_UNARY(neon, expm1, f32, 4);
BENCH_UNARY(neon, log, f32, 4);
BENCH_UNARY(neon, log1p, f32, 4);
BENCH_UNARY(neon, erf, f32, 4);
BENCH_UNARY(neon, tanh, f32, 4);

BENCH_UNARY(neon, fast_erf, f32, 4);

}  // namespace simd
}  // namespace ynn
