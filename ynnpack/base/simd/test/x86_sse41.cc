// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_sse41.h"

#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

TEST_MULTIPLY(x86_sse41, s32x4, arch_flag::sse41);

TEST_MIN(x86_sse41, s8x16, arch_flag::sse41);
TEST_MIN(x86_sse41, s32x4, arch_flag::sse41);

TEST_MAX(x86_sse41, s8x16, arch_flag::sse41);
TEST_MAX(x86_sse41, s32x4, arch_flag::sse41);

TEST_HORIZONTAL_MIN(x86_sse41, s32x4, arch_flag::sse41);

TEST_HORIZONTAL_MAX(x86_sse41, s32x4, arch_flag::sse41);

}  // namespace simd
}  // namespace ynn
