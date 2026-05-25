// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/x86_vec128.h"

// This must be included last
#include "ynnpack/base/simd/bench/generic.h"

namespace ynn {
namespace simd {

BENCH_FMA(sse2_fma, f32, 4);

BENCH_UNARY(sse2_fma, exp, f32, 4);
BENCH_UNARY(sse2_fma, expm1, f32, 4);
BENCH_UNARY(sse2_fma, log, f32, 4);
BENCH_UNARY(sse2_fma, log1p, f32, 4);
BENCH_UNARY(sse2_fma, erf, f32, 4);

BENCH_UNARY(sse2_fma, fast_erf, f32, 4);

}  // namespace simd
}  // namespace ynn
