// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/microkernel-utils.h"

#include <cstddef>
#include <random>

#include <gtest/gtest.h>
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "replicable_random_device.h"

TEST(GEMM_BEST_NC, min_tiles_per_thread) {
  xnnpack::ReplicableRandomDevice rnd;
  std::uniform_int_distribution<size_t> rnd_kernel_dim(1, XNN_MAX_MR);
  std::uniform_int_distribution<size_t> rnd_tensor_dim(1, 100);
  std::uniform_int_distribution<size_t> rnd_thread_dim(2, 16);
  const size_t kNumTrials = 1000;

  for (size_t trial = 0; trial < kNumTrials; trial++) {
    const size_t mr = rnd_kernel_dim(rnd);
    const size_t nr = 8 * rnd_kernel_dim(rnd);
    const size_t m = rnd_tensor_dim(rnd);
    const size_t n = nr + rnd_tensor_dim(rnd);
    const size_t num_threads = rnd_thread_dim(rnd);

    const size_t num_tiles_m = divide_round_up(m, mr);
    const size_t min_num_tiles = XNN_GEMM_TILES_PER_THREAD * num_threads;

    for (size_t num_groups :
         {(size_t)1, num_threads, 5 * num_threads, 10 * num_threads}) {
      const size_t nc = xnn_gemm_best_nc(num_groups, m, n, mr, nr, num_threads);

      // Check that `nc` is a multiple of `nr` if it is less than `n`.
      if (nc < nr) {
        EXPECT_EQ(nc % nr, 0) << "Not a multiple of `nr`";
      }

      // If an `nc` larger than `nr` was chosen, make sure we still have enough
      // tiles.
      if (nr < nc) {
        const size_t num_tiles_n = divide_round_up(n, nc);
        const size_t num_tiles = num_groups * num_tiles_m * num_tiles_n;
        EXPECT_LE(min_num_tiles, num_tiles)
            << "Didn't generate enough tiles, num_groups=" << num_groups
            << ", m=" << m << ", n=" << n << ", " << "mr=" << mr << " , "
            << "nr=" << nr << " , " << "nc=" << nc
            << ", num_threads=" << num_threads;
      }

      // Verify that the next-smallest `nc` would increase the number of tiles.
      if (nr < nc && nc < n) {
        EXPECT_NE(divide_round_up(n, nc), divide_round_up(n, nc - nr))
            << "Failed to get minimal `nc` for num_groups=" << num_groups
            << ", m=" << m << ", n=" << n << ", " << "mr=" << mr << " , "
            << "nr=" << nr << " , " << "nc=" << nc
            << ", num_threads=" << num_threads;
      }
    }
  }
}

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
