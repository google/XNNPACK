// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"

TEST(BuildIdentifierTest, SizeIsCorrect) {
  // The current implmentation uses a SHA256 sum, so we expect the size of the
  // identifier to be 32 bytes long.
  EXPECT_EQ(xnn_experimental_get_build_identifier_size(), 32);
}

TEST(BuildIdentifierTest, ReadingIdentifierDoesNotTriggerAsan) {
  size_t value = 0;
  const void* id = xnn_experimental_get_build_identifier_data();
  for(size_t i = 0; i < xnn_experimental_get_build_identifier_size(); ++i) {
    value += reinterpret_cast<const uint8_t*>(id)[i];
  }
  EXPECT_GT(value, 0);
}

TEST(BuildIdentifierTest, CheckSucceedsForIdentity) {
  EXPECT_TRUE(xnn_experimental_check_build_identifier(xnn_experimental_get_build_identifier_data(),
                                                      xnn_experimental_get_build_identifier_size()));
}

TEST(BuildIdentifierTest, CheckFailsWithDifferentDataSize) {
  EXPECT_FALSE(xnn_experimental_check_build_identifier(nullptr, xnn_experimental_get_build_identifier_size() - 1));
  EXPECT_FALSE(xnn_experimental_check_build_identifier(nullptr, xnn_experimental_get_build_identifier_size() + 1));
}

TEST(BuildIdentifierTest, CheckFailsWithDifferentData) {
  std::vector<uint8_t> wrong_data(xnn_experimental_get_build_identifier_size());
  for(size_t i = 0; i < xnn_experimental_get_build_identifier_size(); ++i) {
    std::copy_n(reinterpret_cast<const uint8_t*>(xnn_experimental_get_build_identifier_data()),
                xnn_experimental_get_build_identifier_size(), wrong_data.begin());
    ++wrong_data[i];
    EXPECT_FALSE(xnn_experimental_check_build_identifier(wrong_data.data(),
                                                         xnn_experimental_get_build_identifier_size())) << "byte " << i;
  }
}
