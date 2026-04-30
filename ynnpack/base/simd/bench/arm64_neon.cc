// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/arm_neon.h"
#include "ynnpack/base/simd/bench/generic.h"

namespace ynn {
namespace simd {

BENCH_UNARY(neon, floor_log2, f64, 2);

}  // namespace simd
}  // namespace ynn
