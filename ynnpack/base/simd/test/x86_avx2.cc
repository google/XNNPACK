// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_avx2.h"

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

class x86_avx2 : public ::testing::Test {
  void SetUp() override {
    if (!is_arch_supported(arch_flag::avx2)) {
      GTEST_SKIP() << "avx2 not supported on this hardware";
    }
  }
};

TEST_ADD(x86_avx2, u8, 32);
TEST_ADD(x86_avx2, s8, 32);
TEST_ADD(x86_avx2, s32, 8);

TEST_SUBTRACT(x86_avx2, u8, 32);
TEST_SUBTRACT(x86_avx2, s8, 32);
TEST_SUBTRACT(x86_avx2, s32, 8);

TEST_MULTIPLY(x86_avx2, s32, 8);

TEST_MIN(x86_avx2, u8, 32);
TEST_MIN(x86_avx2, s8, 32);
TEST_MIN(x86_avx2, s16, 16);
TEST_MIN(x86_avx2, s32, 8);

TEST_MAX(x86_avx2, u8, 32);
TEST_MAX(x86_avx2, s8, 32);
TEST_MAX(x86_avx2, s16, 16);
TEST_MAX(x86_avx2, s32, 8);

TEST_HORIZONTAL_MIN(x86_avx2, u8, 32);
TEST_HORIZONTAL_MIN(x86_avx2, s8, 32);
TEST_HORIZONTAL_MIN(x86_avx2, s16, 16);
TEST_HORIZONTAL_MIN(x86_avx2, s32, 8);

TEST_HORIZONTAL_MAX(x86_avx2, u8, 32);
TEST_HORIZONTAL_MAX(x86_avx2, s8, 32);
TEST_HORIZONTAL_MAX(x86_avx2, s16, 16);
TEST_HORIZONTAL_MAX(x86_avx2, s32, 8);

TEST_CONVERT(x86_avx2, f32, bf16x8);
TEST_CONVERT(x86_avx2, s32, u8x16);
TEST_CONVERT(x86_avx2, s32, s8x16);

}  // namespace simd
}  // namespace ynn
