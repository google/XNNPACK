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

BENCH_UNARY(x86_avx512fp16, floor_log2, f16, 32);
BENCH_UNARY(x86_avx512fp16, exp2_round, f16, 32);

}  // namespace simd
}  // namespace ynn
