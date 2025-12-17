// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"

int main(int argc, char* argv[]) {
  #ifdef YNN_UNSUPPORTED_ARCH
  return 0;
  #else
  if (!ynn::is_arch_supported(ynn::arch_flag::YNN_ARCH)) {
    return 0;
  }
  #endif

  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
