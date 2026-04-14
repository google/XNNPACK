// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/test/generic.h"
#include "ynnpack/base/simd/x86_avx512.h"

namespace ynn {
namespace simd {

class x86_avx512bf16 : public ::testing::Test {
  void SetUp() override {
    if (!is_arch_supported(arch_flag::avx512bf16)) {
      GTEST_SKIP() << "avx512bf16 not supported on this hardware";
    }
  }
};

TEST_CAST(x86_avx512bf16, bf16, f32x32);
TEST_CAST(x86_avx512bf16, bf16, f32x16);

}  // namespace simd
}  // namespace ynn
