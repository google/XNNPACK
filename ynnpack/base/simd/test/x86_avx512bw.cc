// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_avx512bw.h"

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

class x86_avx512bw : public ::testing::Test {
  void SetUp() override {
    if (!is_arch_supported(arch_flag::avx512bw)) {
      GTEST_SKIP() << "avx512bw not supported on this hardware";
    }
  }
};

TEST_PARTIAL_LOAD_STORE(x86_avx512bw, u8, 64);
TEST_PARTIAL_LOAD_STORE(x86_avx512bw, s8, 64);
TEST_PARTIAL_LOAD_STORE(x86_avx512bw, s16, 32);
TEST_PARTIAL_LOAD_STORE(x86_avx512bw, f16, 32);
TEST_PARTIAL_LOAD_STORE(x86_avx512bw, bf16, 32);

TEST_ADD(x86_avx512bw, u8, 64);
TEST_ADD(x86_avx512bw, s8, 64);

TEST_SUBTRACT(x86_avx512bw, u8, 64);
TEST_SUBTRACT(x86_avx512bw, s8, 64);

TEST_MULTIPLY(x86_avx512bw, f32, 16);
TEST_MULTIPLY(x86_avx512bw, s32, 16);

TEST_MIN(x86_avx512bw, u8, 64);
TEST_MIN(x86_avx512bw, s8, 64);
TEST_MIN(x86_avx512bw, s16, 32);

TEST_MAX(x86_avx512bw, u8, 64);
TEST_MAX(x86_avx512bw, s8, 64);
TEST_MAX(x86_avx512bw, s16, 32);

TEST_CONVERT(x86_avx512bw, s32, s8x16);
TEST_CONVERT(x86_avx512bw, s32, u8x16);
TEST_CONVERT(x86_avx512bw, s32, s8x32);
TEST_CONVERT(x86_avx512bw, s32, u8x32);

}  // namespace simd
}  // namespace ynn
