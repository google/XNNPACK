// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_sse41.h"

#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

TEST_MULTIPLY(x86_sse41, int32_t, 4);

TEST_MIN(x86_sse41, int8_t, 16);
TEST_MIN(x86_sse41, int32_t, 4);

TEST_MAX(x86_sse41, int8_t, 16);
TEST_MAX(x86_sse41, int32_t, 4);

TEST_HORIZONTAL_MIN(x86_sse41, int32_t, 4);

TEST_HORIZONTAL_MAX(x86_sse41, int32_t, 4);

TEST_CONVERT(x86_sse41, int32_t, u8x16);
TEST_CONVERT(x86_sse41, int32_t, s8x16);

}  // namespace simd
}  // namespace ynn
