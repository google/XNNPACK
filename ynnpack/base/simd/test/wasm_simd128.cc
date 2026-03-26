// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/wasm_simd128.h"

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

class wasm : public ::testing::Test {
  void SetUp() override {
    if (!is_arch_supported(arch_flag::wasm_simd128)) {
      GTEST_SKIP() << "wasm simd128 not supported on this hardware";
    }
  }
};

TEST_BROADCAST(wasm, f32, 4);

}  // namespace simd
}  // namespace ynn
