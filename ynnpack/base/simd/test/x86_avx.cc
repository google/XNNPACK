// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_avx.h"

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

class x86_avx : public ::testing::Test {
  void SetUp() override {
    if (!is_arch_supported(arch_flag::avx)) {
      GTEST_SKIP() << "avx not supported on this hardware";
    }
  }
};

TEST_BROADCAST(x86_avx, u8, 32);
TEST_BROADCAST(x86_avx, s8, 32);
TEST_BROADCAST(x86_avx, s16, 16);
TEST_BROADCAST(x86_avx, f16, 16);
TEST_BROADCAST(x86_avx, bf16, 16);
TEST_BROADCAST(x86_avx, f32, 8);
TEST_BROADCAST(x86_avx, s32, 8);

TEST_LOAD_STORE(x86_avx, u8, 32);
TEST_LOAD_STORE(x86_avx, s8, 32);
TEST_LOAD_STORE(x86_avx, s16, 16);
TEST_LOAD_STORE(x86_avx, f16, 16);
TEST_LOAD_STORE(x86_avx, bf16, 16);
TEST_LOAD_STORE(x86_avx, f32, 8);
TEST_LOAD_STORE(x86_avx, s32, 8);

TEST_ALIGNED_LOAD_STORE(x86_avx, u8, 32);
TEST_ALIGNED_LOAD_STORE(x86_avx, s8, 32);
TEST_ALIGNED_LOAD_STORE(x86_avx, s16, 16);
TEST_ALIGNED_LOAD_STORE(x86_avx, f16, 16);
TEST_ALIGNED_LOAD_STORE(x86_avx, bf16, 16);
TEST_ALIGNED_LOAD_STORE(x86_avx, f32, 8);
TEST_ALIGNED_LOAD_STORE(x86_avx, s32, 8);

TEST_PARTIAL_LOAD_STORE(x86_avx, u8, 32);
TEST_PARTIAL_LOAD_STORE(x86_avx, s8, 32);
TEST_PARTIAL_LOAD_STORE(x86_avx, s16, 16);
TEST_PARTIAL_LOAD_STORE(x86_avx, f16, 16);
TEST_PARTIAL_LOAD_STORE(x86_avx, bf16, 16);
TEST_PARTIAL_LOAD_STORE(x86_avx, f32, 8);
TEST_PARTIAL_LOAD_STORE(x86_avx, s32, 8);

TEST_ADD(x86_avx, f32, 8);
TEST_SUBTRACT(x86_avx, f32, 8);
TEST_MULTIPLY(x86_avx, f32, 8);
TEST_MIN(x86_avx, f32, 8);
TEST_MAX(x86_avx, f32, 8);

TEST_EXTRACT(x86_avx, s32x8, 4);
TEST_EXTRACT(x86_avx, f32x8, 4);
TEST_EXTRACT(x86_avx, bf16x16, 8);
TEST_EXTRACT(x86_avx, f16x16, 8);
TEST_EXTRACT(x86_avx, s8x32, 16);
TEST_EXTRACT(x86_avx, u8x32, 16);

TEST_CONCAT(x86_avx, s32x4);
TEST_CONCAT(x86_avx, f32x4);
TEST_CONCAT(x86_avx, bf16x8);
TEST_CONCAT(x86_avx, f16x8);
TEST_CONCAT(x86_avx, s8x16);
TEST_CONCAT(x86_avx, u8x16);

TEST_HORIZONTAL_MIN(x86_avx, f32, 8);
TEST_HORIZONTAL_MAX(x86_avx, f32, 8);

}  // namespace simd
}  // namespace ynn
