// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_avx2.h"

#include "ynnpack/base/simd/bench/generic.h"

namespace ynn {
namespace simd {

BENCH_UNARY(avx2, floor_log2, f32, 8);
BENCH_UNARY(avx2, floor_log2, f64, 4);

}  // namespace simd
}  // namespace ynn
