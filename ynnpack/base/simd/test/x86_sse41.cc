// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_sse41.h"

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

class x86_sse41 : public ::testing::Test {
  void SetUp() override {
    if (!is_arch_supported(arch_flag::sse41)) {
      GTEST_SKIP() << "sse41 not supported on this hardware";
    }
  }
};

TEST_MULTIPLY(x86_sse41, s32, 4);

TEST_MIN(x86_sse41, s8, 16);
TEST_MIN(x86_sse41, s32, 4);

TEST_MAX(x86_sse41, s8, 16);
TEST_MAX(x86_sse41, s32, 4);

TEST_HORIZONTAL_MIN(x86_sse41, s32, 4);

TEST_HORIZONTAL_MAX(x86_sse41, s32, 4);

TEST_CONVERT(x86_sse41, s32, u8x16);
TEST_CONVERT(x86_sse41, s32, s8x16);

}  // namespace simd
}  // namespace ynn
