// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/microkernel-utils.h"

#include <cstddef>
#include <random>

#include <gtest/gtest.h>
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "test/replicable_random_device.h"

TEST(GEMM_BEST_TILE_SIZE, min_tiles_per_thread) {
  xnnpack::ReplicableRandomDevice rnd;
  std::uniform_int_distribution<size_t> rnd_kernel_dim(1, XNN_MAX_MR);
  std::uniform_int_distribution<size_t> rnd_tensor_dim(1, 100);
  std::uniform_int_distribution<size_t> rnd_thread_dim(2, 16);
  const size_t kNumTrials = 1000;

  for (size_t trial = 0; trial < kNumTrials; trial++) {
    const size_t mr = rnd_kernel_dim(rnd);
    const size_t nr = 8 * rnd_kernel_dim(rnd);
    const size_t m = rnd_tensor_dim(rnd);
    const size_t k = rnd_tensor_dim(rnd);
    const size_t n = nr + rnd_tensor_dim(rnd);
    const size_t num_threads = rnd_thread_dim(rnd);

    const size_t min_num_tiles = XNN_GEMM_MIN_TILES_PER_THREAD * num_threads;

    for (size_t num_groups :
         {(size_t)1, num_threads, 5 * num_threads, 10 * num_threads}) {
      const size_t nc = xnn_gemm_best_tile_size(
          num_groups, m, n, /*m_stride=*/k * sizeof(float),
          /*n_stride=*/k * sizeof(float),
          /*cn_stride=*/sizeof(float), mr, nr, num_threads);

      // Check that `nc` is an integer multiple of `nr` if it is less than `n`.
      if (nc < n) {
        EXPECT_EQ(nc % nr, 0)
            << "mc=" << nc << " is not a multiple of nr=" << nr;
      }

      // If an `nc` larger than `nr`, or `mc` larger than `mr`, was chosen, make
      // sure we still have enough tiles.
      if (nr < nc) {
        const size_t num_tiles_m = divide_round_up(m, mr);
        const size_t num_tiles_n = divide_round_up(n, nc);
        const size_t num_tiles = num_groups * num_tiles_m * num_tiles_n;
        EXPECT_LE(min_num_tiles, num_tiles)
            << "Didn't generate enough tiles, num_groups=" << num_groups
            << ", m=" << m << ", n=" << n << ", " << "mr=" << mr
            << ", nr=" << nr << ", nc=" << nc
            << ", num_threads=" << num_threads;
      }

      // Verify that the next-smallest `nc` or `mc` would increase the number of
      // tiles.
      if (nr < nc && nc < n) {
        EXPECT_NE(divide_round_up(n, nc), divide_round_up(n, nc - nr))
            << "Failed to get minimal `nc` for num_groups=" << num_groups
            << ", m=" << m << ", n=" << n << ", mr=" << mr << ", nr=" << nr
            << ", nc=" << nc << ", num_threads=" << num_threads;
      }
    }
  }
}

TEST(GEMM_BEST_TILE_SIZE, zero_inputs) {
  EXPECT_EQ(0, xnn_gemm_best_tile_size(1, 0, 10, 4, 4, 4, 2, 4, 1));
  EXPECT_EQ(0, xnn_gemm_best_tile_size(1, 10, 0, 4, 4, 4, 2, 4, 1));
  EXPECT_EQ(0, xnn_gemm_best_tile_size(1, 10, 10, 4, 4, 4, 0, 4, 1));
  EXPECT_EQ(0, xnn_gemm_best_tile_size(1, 10, 10, 4, 4, 4, 2, 0, 1));
}

TEST(GEMM_BEST_TILE_SIZE, overflow_inputs) {
  const size_t nc = xnn_gemm_best_tile_size(
      SIZE_MAX, SIZE_MAX, 100, SIZE_MAX, SIZE_MAX, SIZE_MAX, 4, 8, 4);
  EXPECT_GT(nc, 0);
  EXPECT_LE(nc, 100);
}

TEST(SHOULD_INLINE_LHS_PACKING, null_config) {
  EXPECT_FALSE(xnn_should_inline_lhs_packing(nullptr, 16, 16, 16, 4, 8));
}

TEST(USE_NR2, zero_nr) {
  EXPECT_FALSE(xnn_use_nr2(0, 4, 16));
  EXPECT_FALSE(xnn_use_nr2(4, 0, 16));
  EXPECT_FALSE(xnn_use_nr2(0, 0, 16));
}

TEST(USE_NR2, normal_cases) {
  EXPECT_TRUE(xnn_use_nr2(8, 4, 3));
  EXPECT_FALSE(xnn_use_nr2(8, 4, 8));
}
