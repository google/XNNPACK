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

BENCH_FMA(fma3, f32, 8);

BENCH_UNARY(fma3, exp, f32, 8);
BENCH_UNARY(fma3, exp, f64, 4);
BENCH_UNARY(fma3, expm1, f32, 8);
BENCH_UNARY(fma3, expm1, f64, 4);
BENCH_UNARY(fma3, log, f32, 8);
BENCH_UNARY(fma3, log, f64, 4);
BENCH_UNARY(fma3, log1p, f32, 8);
BENCH_UNARY(fma3, log1p, f64, 4);

}  // namespace simd
}  // namespace ynn
