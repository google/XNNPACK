// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_avx.h"

#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/bench/generic.h"

namespace ynn {
namespace simd {

BENCH_PARTIAL_LOAD_STORE(avx, s8, 32);
BENCH_PARTIAL_LOAD_STORE(avx, s16, 16);
BENCH_PARTIAL_LOAD_STORE(avx, s32, 8);

}  // namespace simd
}  // namespace ynn
