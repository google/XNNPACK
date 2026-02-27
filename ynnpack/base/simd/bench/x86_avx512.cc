// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_avx512.h"

#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/bench/generic.h"

namespace ynn {
namespace simd {

BENCH_PARTIAL_LOAD_STORE(avx512, s8, 16);
BENCH_PARTIAL_LOAD_STORE(avx512, s16, 8);
BENCH_PARTIAL_LOAD_STORE(avx512, s32, 4);

BENCH_PARTIAL_LOAD_STORE(avx512, s8, 32);
BENCH_PARTIAL_LOAD_STORE(avx512, s16, 16);
BENCH_PARTIAL_LOAD_STORE(avx512, s32, 8);

BENCH_PARTIAL_LOAD_STORE(avx512, s8, 64);
BENCH_PARTIAL_LOAD_STORE(avx512, s16, 32);
BENCH_PARTIAL_LOAD_STORE(avx512, s32, 16);

BENCH_FMA(avx512, f32, 16);

}  // namespace simd
}  // namespace ynn
