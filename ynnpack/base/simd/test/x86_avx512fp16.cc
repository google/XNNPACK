// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/x86_vec512.h"

// This must be included last
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

class x86_avx512fp16 : public ::testing::Test {
  void SetUp() override {
    if (!is_arch_supported(arch_flag::avx512fp16)) {
      GTEST_SKIP() << "avx512fp16 not supported on this hardware";
    }
  }
};

TEST_LOAD_STORE(x86_avx512fp16, f16, 8);
TEST_LOAD_STORE(x86_avx512fp16, f16, 16);
TEST_LOAD_STORE(x86_avx512fp16, f16, 32);

TEST_ADD(x86_avx512fp16, f16, 8);
TEST_ADD(x86_avx512fp16, f16, 16);
TEST_ADD(x86_avx512fp16, f16, 32);

TEST_SUBTRACT(x86_avx512fp16, f16, 8);
TEST_SUBTRACT(x86_avx512fp16, f16, 16);
TEST_SUBTRACT(x86_avx512fp16, f16, 32);

TEST_MULTIPLY(x86_avx512fp16, f16, 8);
TEST_MULTIPLY(x86_avx512fp16, f16, 16);
TEST_MULTIPLY(x86_avx512fp16, f16, 32);

TEST_DIVIDE(x86_avx512fp16, f16, 8);
TEST_DIVIDE(x86_avx512fp16, f16, 16);
TEST_DIVIDE(x86_avx512fp16, f16, 32);

TEST_MIN(x86_avx512fp16, f16, 8);
TEST_MIN(x86_avx512fp16, f16, 16);
TEST_MIN(x86_avx512fp16, f16, 32);

TEST_MAX(x86_avx512fp16, f16, 8);
TEST_MAX(x86_avx512fp16, f16, 16);
TEST_MAX(x86_avx512fp16, f16, 32);

TEST_FLOOR_LOG2(x86_avx512fp16, f16, 8);
TEST_FLOOR_LOG2(x86_avx512fp16, f16, 16);
TEST_FLOOR_LOG2(x86_avx512fp16, f16, 32);

TEST_EXP2_ROUND(x86_avx512fp16, f16, 8);
TEST_EXP2_ROUND(x86_avx512fp16, f16, 16);
TEST_EXP2_ROUND(x86_avx512fp16, f16, 32);

}  // namespace simd
}  // namespace ynn
