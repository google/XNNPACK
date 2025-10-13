// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstdint>

#include "ynnpack/base/simd/x86_sse.h"
#include "ynnpack/kernels/reduce/min_max_accumulator.h"

namespace ynn {

using simd::s32x4;
using simd::s8x16;

MIN_MAX_KERNEL(min_max_int8_4x16_sse41, s8x16, s8x16, int8_t, 16);
MIN_MAX_KERNEL(min_int8_4x16_sse41, s8x16, dummy_t, int8_t, 16);
MIN_MAX_KERNEL(max_int8_4x16_sse41, dummy_t, s8x16, int8_t, 16);

}  // namespace ynn
