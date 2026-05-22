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

BENCH_FMA(sse2fma, f32, 4);

BENCH_UNARY(sse2fma, exp, f32, 4);
BENCH_UNARY(sse2fma, expm1, f32, 4);
BENCH_UNARY(sse2fma, log, f32, 4);
BENCH_UNARY(sse2fma, log1p, f32, 4);

}  // namespace simd
}  // namespace ynn
