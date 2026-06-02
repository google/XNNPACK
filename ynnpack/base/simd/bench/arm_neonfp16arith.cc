// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/arm_vec128.h"

// This must be included last
#include "ynnpack/base/simd/bench/generic.h"

namespace ynn {
namespace simd {

BENCH_UNARY(arm_neonfp16arith, floor_log2, f16, 32);
BENCH_UNARY(arm_neonfp16arith, exp2_round, f16, 32);

BENCH_UNARY(arm_neonfp16arith, approx_log, f16, 32);
BENCH_UNARY(arm_neonfp16arith, approx_log1p, f16, 32);
#ifdef YNN_ARCH_ARM64
BENCH_UNARY(arm_neonfp16arith, approx_erf, f16, 32);
BENCH_UNARY(arm_neonfp16arith, approx_tanh, f16, 32);
#endif

}  // namespace simd
}  // namespace ynn
