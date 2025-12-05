// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_avx2.h"

#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

TEST_ADD(x86_avx2, u8x32, arch_flag::avx2);
TEST_ADD(x86_avx2, s8x32, arch_flag::avx2);
TEST_ADD(x86_avx2, s32x8, arch_flag::avx2);

TEST_SUBTRACT(x86_avx2, u8x32, arch_flag::avx2);
TEST_SUBTRACT(x86_avx2, s8x32, arch_flag::avx2);
TEST_SUBTRACT(x86_avx2, s32x8, arch_flag::avx2);

TEST_MULTIPLY(x86_avx2, s32x8, arch_flag::avx2);

TEST_MIN(x86_avx2, u8x32, arch_flag::avx2);
TEST_MIN(x86_avx2, s8x32, arch_flag::avx2);
TEST_MIN(x86_avx2, s16x16, arch_flag::avx2);
TEST_MIN(x86_avx2, s32x8, arch_flag::avx2);

TEST_MAX(x86_avx2, u8x32, arch_flag::avx2);
TEST_MAX(x86_avx2, s8x32, arch_flag::avx2);
TEST_MAX(x86_avx2, s16x16, arch_flag::avx2);
TEST_MAX(x86_avx2, s32x8, arch_flag::avx2);

TEST_HORIZONTAL_MIN(x86_avx2, u8x32, arch_flag::avx2);
TEST_HORIZONTAL_MIN(x86_avx2, s8x32, arch_flag::avx2);
TEST_HORIZONTAL_MIN(x86_avx2, s16x16, arch_flag::avx2);
TEST_HORIZONTAL_MIN(x86_avx2, s32x8, arch_flag::avx2);

TEST_HORIZONTAL_MAX(x86_avx2, u8x32, arch_flag::avx2);
TEST_HORIZONTAL_MAX(x86_avx2, s8x32, arch_flag::avx2);
TEST_HORIZONTAL_MAX(x86_avx2, s16x16, arch_flag::avx2);
TEST_HORIZONTAL_MAX(x86_avx2, s32x8, arch_flag::avx2);

}  // namespace simd
}  // namespace ynn
