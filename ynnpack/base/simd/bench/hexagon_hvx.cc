// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/hexagon_hvx.h"

#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/bench/generic.h"

namespace ynn {
namespace simd {

BENCH_PARTIAL_LOAD_STORE(hvx, s8, 128);
BENCH_PARTIAL_LOAD_STORE(hvx, s16, 64);
BENCH_PARTIAL_LOAD_STORE(hvx, s32, 32);

}  // namespace simd
}  // namespace ynn
