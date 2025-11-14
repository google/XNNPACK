// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>
#include <gmock/gmock.h>

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  // This calls InitGoogleTest as well. It consume any arguments it understands
  // from argv.
  testing::InitGoogleMock(&argc, argv);

  return RUN_ALL_TESTS();
}
