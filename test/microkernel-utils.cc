// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/microkernel-utils.h>
#include <gtest/gtest.h>

TEST(MULTIPASS_DWCONV_WEIGHTS_COUNT, channels_le_channel_tile) {
  ASSERT_EQ((1 + 1) * 4, xnn_dwconv_multipass_weights_count(1, 1, 8, 4, 4));
  ASSERT_EQ((1 + 1) * 4, xnn_dwconv_multipass_weights_count(1, 2, 8, 4, 4));
  ASSERT_EQ((1 + 1) * 4, xnn_dwconv_multipass_weights_count(1, 3, 8, 4, 4));
  ASSERT_EQ((1 + 1) * 4, xnn_dwconv_multipass_weights_count(1, 4, 8, 4, 4));
  ASSERT_EQ((1 + 1) * 4 * 2, xnn_dwconv_multipass_weights_count(1, 5, 8, 4, 4));
  ASSERT_EQ((1 + 1) * 4 * 2, xnn_dwconv_multipass_weights_count(1, 6, 8, 4, 4));
  ASSERT_EQ((1 + 1) * 4 * 2, xnn_dwconv_multipass_weights_count(1, 7, 8, 4, 4));
  ASSERT_EQ((1 + 1) * 8, xnn_dwconv_multipass_weights_count(1, 8, 8, 4, 4));
}

TEST(MULTIPASS_DWCONV_WEIGHTS_COUNT, channels_gt_channel_tile) {
  ASSERT_EQ((1 + 1) * (8 + 4), xnn_dwconv_multipass_weights_count(1, 9, 8, 4, 4));
  ASSERT_EQ((1 + 1) * (8 + 4), xnn_dwconv_multipass_weights_count(1, 10, 8, 4, 4));
  ASSERT_EQ((1 + 1) * (8 + 4), xnn_dwconv_multipass_weights_count(1, 11, 8, 4, 4));
  ASSERT_EQ((1 + 1) * (8 + 4), xnn_dwconv_multipass_weights_count(1, 12, 8, 4, 4));
  ASSERT_EQ((1 + 1) * (8 + 4 * 2), xnn_dwconv_multipass_weights_count(1, 13, 8, 4, 4));
  ASSERT_EQ((1 + 1) * (8 + 4 * 2), xnn_dwconv_multipass_weights_count(1, 14, 8, 4, 4));
  ASSERT_EQ((1 + 1) * (8 + 4 * 2), xnn_dwconv_multipass_weights_count(1, 15, 8, 4, 4));
  ASSERT_EQ((1 + 1) * (8 + 8), xnn_dwconv_multipass_weights_count(1, 16, 8, 4, 4));

  ASSERT_EQ((1 + 1) * 16, xnn_dwconv_multipass_weights_count(1, 16, 16, 4, 4));
  ASSERT_EQ((1 + 1) * (16 + 4), xnn_dwconv_multipass_weights_count(1, 17, 16, 4, 4));
  ASSERT_EQ((1 + 1) * (16 + 4), xnn_dwconv_multipass_weights_count(1, 20, 16, 4, 4));
  ASSERT_EQ((1 + 1) * (16 + 4 * 2), xnn_dwconv_multipass_weights_count(1, 21, 16, 4, 4));
  ASSERT_EQ((1 + 1) * (16 + 4 * 2), xnn_dwconv_multipass_weights_count(1, 24, 16, 4, 4));
  ASSERT_EQ((1 + 1) * (16 + 4 * 3), xnn_dwconv_multipass_weights_count(1, 28, 16, 4, 4));
  ASSERT_EQ((1 + 1) * (16 + 16), xnn_dwconv_multipass_weights_count(1, 29, 16, 4, 4));
}

TEST(MULTIPASS_DWCONV_WEIGHTS_COUNT, channels_gt_channel_tile_channel_round) {
  // Simulate AVX channel tile.
  ASSERT_EQ((1 + 1) * (16 + 8), xnn_dwconv_multipass_weights_count(1, 17, 16, 8, 4));
  ASSERT_EQ((1 + 1) * (16 + 8 + 8), xnn_dwconv_multipass_weights_count(1, 25, 16, 8, 4));
}
