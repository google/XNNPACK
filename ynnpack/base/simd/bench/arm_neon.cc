// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/arm_neon.h"

#include <cstdint>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/bench/generic.h"

namespace ynn {
namespace simd {

BENCH_PARTIAL_LOAD_STORE(neon, s8, 16);
BENCH_PARTIAL_LOAD_STORE(neon, s16, 8);
BENCH_PARTIAL_LOAD_STORE(neon, s32, 4);

}  // namespace simd
}  // namespace ynn
