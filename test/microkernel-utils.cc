// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/microkernel-utils.h>
#include <gtest/gtest.h>

TEST(MULTIPASS_DWCONV_WEIGHTS_COUNT, channels_le_channel_tile) {
  ASSERT_EQ((1 + 1) * 4, xnn_dwconv_multipass_weights_size(1, 1, 8, 4, 4, 1, 0, 0));
  ASSERT_EQ((1 + 1) * 4, xnn_dwconv_multipass_weights_size(1, 2, 8, 4, 4, 1, 0, 0));
  ASSERT_EQ((1 + 1) * 4, xnn_dwconv_multipass_weights_size(1, 3, 8, 4, 4, 1, 0, 0));
  ASSERT_EQ((1 + 1) * 4, xnn_dwconv_multipass_weights_size(1, 4, 8, 4, 4, 1, 0, 0));
  ASSERT_EQ((1 + 1) * 4 * 2, xnn_dwconv_multipass_weights_size(1, 5, 8, 4, 4, 1, 0, 0));
  ASSERT_EQ((1 + 1) * 4 * 2, xnn_dwconv_multipass_weights_size(1, 6, 8, 4, 4, 1, 0, 0));
  ASSERT_EQ((1 + 1) * 4 * 2, xnn_dwconv_multipass_weights_size(1, 7, 8, 4, 4, 1, 0, 0));
  ASSERT_EQ((1 + 1) * 8, xnn_dwconv_multipass_weights_size(1, 8, 8, 4, 4, 1, 0, 0));
}

TEST(MULTIPASS_DWCONV_WEIGHTS_size, channels_gt_channel_tile) {
  ASSERT_EQ((1 + 1) * (8 + 4), xnn_dwconv_multipass_weights_size(1, 9, 8, 4, 4, 1, 0, 0));
  ASSERT_EQ((1 + 1) * (8 + 4), xnn_dwconv_multipass_weights_size(1, 10, 8, 4, 4, 1, 0, 0));
  ASSERT_EQ((1 + 1) * (8 + 4), xnn_dwconv_multipass_weights_size(1, 11, 8, 4, 4, 1, 0, 0));
  ASSERT_EQ((1 + 1) * (8 + 4), xnn_dwconv_multipass_weights_size(1, 12, 8, 4, 4, 1, 0, 0));
  ASSERT_EQ((1 + 1) * (8 + 4 * 2), xnn_dwconv_multipass_weights_size(1, 13, 8, 4, 4, 1, 0, 0));
  ASSERT_EQ((1 + 1) * (8 + 4 * 2), xnn_dwconv_multipass_weights_size(1, 14, 8, 4, 4, 1, 0, 0));
  ASSERT_EQ((1 + 1) * (8 + 4 * 2), xnn_dwconv_multipass_weights_size(1, 15, 8, 4, 4, 1, 0, 0));
  ASSERT_EQ((1 + 1) * (8 + 8), xnn_dwconv_multipass_weights_size(1, 16, 8, 4, 4, 1, 0, 0));

  ASSERT_EQ((1 + 1) * 16, xnn_dwconv_multipass_weights_size(1, 16, 16, 4, 4, 1, 0, 0));
  ASSERT_EQ((1 + 1) * (16 + 4), xnn_dwconv_multipass_weights_size(1, 17, 16, 4, 4, 1, 0, 0));
  ASSERT_EQ((1 + 1) * (16 + 4), xnn_dwconv_multipass_weights_size(1, 20, 16, 4, 4, 1, 0, 0));
  ASSERT_EQ((1 + 1) * (16 + 4 * 2), xnn_dwconv_multipass_weights_size(1, 21, 16, 4, 4, 1, 0, 0));
  ASSERT_EQ((1 + 1) * (16 + 4 * 2), xnn_dwconv_multipass_weights_size(1, 24, 16, 4, 4, 1, 0, 0));
  ASSERT_EQ((1 + 1) * (16 + 4 * 3), xnn_dwconv_multipass_weights_size(1, 28, 16, 4, 4, 1, 0, 0));
  ASSERT_EQ((1 + 1) * (16 + 16), xnn_dwconv_multipass_weights_size(1, 29, 16, 4, 4, 1, 0, 0));
}

TEST(MULTIPASS_DWCONV_WEIGHTS_size, channels_gt_channel_tile_channel_round) {
  // Simulate AVX channel tile.
  ASSERT_EQ((1 + 1) * (16 + 8), xnn_dwconv_multipass_weights_size(1, 17, 16, 8, 4, 1, 0, 0));
  ASSERT_EQ((1 + 1) * (16 + 8 + 8), xnn_dwconv_multipass_weights_size(1, 25, 16, 8, 4, 1, 0, 0));
}

TEST(MULTIPASS_DWCONV_ELEMENTS_READ, elements_read) {
  // | First pass | middle pass | last pass | total            |
  // |------------|-------------|-----------|------------------|
  // | 3 * 16     | 3 * 16      | 3 * 16    | 144 inputs read  |
  // | 1 * 16     | 0           | 0         | 16 bias read     |
  // | 3 * 16     | 3 * 16      | 3 * 16    | 144 weights read |
  // | 0          | 16          | 16        | 32 buffers read  |
  ASSERT_EQ(144 + 16 + 144 + 32,
            xnn_dwconv_multipass_bytes_read(
              /*kernel_size=*/3 * 3,
              /*first_pass_tile=*/3, /*middle_pass_tile=*/3, /*last_pass_tile=*/3,
              /*channels=*/16, /*channel_tile=*/4, /*channel_subtile=*/4, /*channel_round=*/4,
              /*log2_input_size=*/0, /*log2_filter_size=*/0, /*bias_element_size=*/1, /*log2_accumulator_size=*/0));
}

TEST(MULTIPASS_DWCONV_ELEMENTS_READ, different_sizes_for_weights) {
  // | First pass | middle pass | last pass | size | total            |
  // |------------|-------------|-----------|------|------------------|
  // | 3 * 16     | 3 * 16      | 3 * 16    |   1  | 144 inputs read  |
  // | 1 * 16     | 0           | 0         |   1  | 16 bias read     |
  // | 3 * 16     | 3 * 16      | 3 * 16    |   4  | 144 weights read |
  // | 0          | 16          | 16        |   4  | 32 buffers read  |
  ASSERT_EQ(144 * 1 + 16 * 1 + 144 * 4 + 32 * 4,
            xnn_dwconv_multipass_bytes_read(
              /*kernel_size=*/3 * 3,
              /*first_pass_tile=*/3, /*middle_pass_tile=*/3, /*last_pass_tile=*/3,
              /*channels=*/16, /*channel_tile=*/4, /*channel_subtile=*/4, /*channel_round=*/4,
              /*log2_input_size=*/0, /*log2_filter_size=*/2, /*bias_element_size=*/1, /*log2_accumulator_size=*/2));
}

TEST(MULTIPASS_DWCONV_ELEMENTS_WRITTEN, elements_written) {
  // 16 + 16 = 32 buffers written.
  // 16 outputs written.
  // | First pass | middle pass | last pass | total              |
  // |------------|-------------|-----------|--------------------|
  // | 0          | 16          | 16        | 32 buffers written |
  // | 0          | 0           | 16        | 16 outputs written |
  ASSERT_EQ(32 + 16,
            xnn_dwconv_multipass_bytes_written(
              /*kernel_size=*/3 * 3,
              /*first_pass_tile=*/3, /*middle_pass_tile=*/3, /*last_pass_tile=*/3,
              /*channels=*/16, /*channel_round=*/4,
              /*log2_accumulator_size=*/0, /*log2_output_size=*/0));
}

TEST(MULTIPASS_DWCONV_ELEMENTS_WRITTEN, different_accumulator_size) {
  // 16 + 16 = 32 buffers written.
  // 16 outputs written.
  // | First pass | middle pass | last pass | size | total              |
  // |------------|-------------|-----------|------|--------------------|
  // | 0          | 16          | 16        |  4   | 32 buffers written |
  // | 0          | 0           | 16        |  1   | 16 outputs written |
  ASSERT_EQ(32 * 4 + 16 * 1,
            xnn_dwconv_multipass_bytes_written(
              /*kernel_size=*/3 * 3,
              /*first_pass_tile=*/3, /*middle_pass_tile=*/3, /*last_pass_tile=*/3,
              /*channels=*/16, /*channel_round=*/4,
              /*log2_accumulator_size=*/2, /*log2_output_size=*/0));
}
