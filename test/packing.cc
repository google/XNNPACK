// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <numeric>

#include <xnnpack/aligned-allocator.h>
#include <xnnpack/microkernel-utils.h>
#include <xnnpack/operator-utils.h>
#include <xnnpack/pack.h>

#include <gtest/gtest.h>
#include <fp16.h>

TEST(PACK_QU8_DWCONV_GHW_W, primary_tile_eq_kernel_size) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<uint8_t> k(c * h * w);  // k = [2, 3, 4, 5, 6, 7]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));

  std::vector<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 48387);
  const std::vector<uint8_t> expected = {
    // bias first
    // 48387 + 0 - (2 + 3 + 4) * 127 = 47,244 = 0xB88C
    0x8C, 0xB8, 0, 0,
    // 48387 + 1 - (5 + 6 + 7) * 127 = 46,102 = 0xB416
    0x16, 0xB4, 0, 0,
    // then weights, channels first
    2, 5,
    3, 6,
    4, 7,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_DWCONV_GHW_W, primary_tile_eq_kernel_size_channels_gt_cr) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint8_t> k(c * h * w);  // k = [
                                      //   5, 6, 7,
                                      //   8, 9, 10,
                                      //   11, 12, 13,
                                      //   14, 15, 16,
                                      //   17, 18, 19]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 48387);
  const std::vector<uint8_t> expected = {
    // cr blocks
    // bias first (cr == 2 of them)
    // 48387 + 0 - (5 + 6 + 7) * 127 = 46,101 = 0xB415
    0x15, 0xB4, 0, 0,
    // 48387 + 1 - (8 + 9 + 10) * 127 = 44,959 = 0xAF9F
    0x9F, 0xAF, 0, 0,
    // then weights, channels first
    5, 8, 6, 9, 7, 10,
    // bias again
    // 48387 + 2 - (11 + 12 + 13) * 127 = 43,817 = 0xAB29
    0x29, 0xAB, 0, 0,
    // 48387 + 3 - (14 + 15 + 16) * 127 = 42,675 = 0xA6B3
    0xB3, 0xA6, 0, 0,
    // then weights, channels first
    11, 14, 12, 15, 13, 16,
    // bias again
    // 48387 + 4 - (17 + 18 + 19) * 127 = 41,533 = 0xA23D
    0x3D, 0xA2, 0, 0,
    0, 0, 0, 0,
    // then weights, channels first
    17, 0, 18, 0, 19, 0,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_DWCONV_GHW_W, primary_tile_gt_kernel_size) {
  const size_t primary_tile = 9;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<uint8_t> k(c * h * w);  // k = [
                                      //   2, 3,
                                      //   4, 5,
                                      //   6, 7,
                                      //   8, 9]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<uint8_t> expected = {
    // bias first (cr == 2 of them)
    // 64516 + 0 - (2 + 3 + 4 + 5) * 127 = 62,738 = 0xF512
    0x12, 0xF5, 0, 0,
    // 64516 + 1 - (6 + 7 + 8 + 9) * 127 = 60,707 = 0xED23
    0x23, 0xED, 0, 0,
    // then weights, channels first
    2, 6,
    // go down the columns first
    4, 8, 3, 7, 5, 9,
    // followed by 10 zeros to make up the difference with primary_tile
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_DWCONV_GHW_W, primary_tile_gt_kernel_size_channels_gt_cr) {
  const size_t primary_tile = 9;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint8_t> k(c * h * w);  // k = [
                                      //   5, 6,
                                      //   7, 8,
                                      //   9, 10,
                                      //   11, 12,
                                      //   13, 14,
                                      //   15, 16,
                                      //   17, 18,
                                      //   19, 20,
                                      //   21, 22,
                                      //   23, 24]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<uint8_t> expected = {
    // bias first (cr == 2 of them)
    // 64516 + 0 - (5 + 6 + 7 + 8) * 127 = 61,214 = 0xEF1E
    0x1E, 0xEF, 0, 0,
    // 64516 + 1 - (9 + 10 + 11 + 12) * 127 = 59,183 = 0xE72F
    0x2F, 0xE7, 0, 0,
    // then weights, channels first
    5, 9,
    // go down the columns first
    7, 11,
    6, 10,
    8, 12,
    // followed by 10 zeros to make up the difference with primary_tile
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // bias first (cr == 2 of them)
    // 64516 + 2 - (13 + 14 + 15 + 16) * 127 = 57,152 = 0xDF40
    0x40, 0xDF, 0, 0,
    // 64516 + 3 - (17 + 18 + 19 + 20) * 127 = 55,121 = 0xD751
    0x51, 0xD7, 0, 0,
    // then weights, channels first
    13, 17, 15, 19, 14, 18, 16, 20,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // bias
    // 64516 + 4 - (21 + 22 + 23 + 24) * 127 = 53,090 = 0xCF62
    0x62, 0xCF, 0, 0,
    0, 0, 0, 0,
    // weights
    21, 0, 23, 0, 22, 0, 24, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_DWCONV_HWG_W, primary_tile_eq_kernel_size) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<uint8_t> k(c * h * w);  // k = [2, 3, 4, 5, 6, 7]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));

  std::vector<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 48387);
  const std::vector<uint8_t> expected = {
    // bias first
    // 48387 + 0 - (2 + 4 + 6) * 127 = 46,863 = 0xB70F
    0x0F, 0xB7, 0, 0,
    // 48387 + 1 - (3 + 5 + 7) * 127 = 46,483 = 0xB593
    0x93, 0xB5, 0, 0,
    // then weights, channels first
    2, 3,
    4, 5,
    6, 7,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_DWCONV_HWG_W, primary_tile_eq_kernel_size_channels_gt_cr) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint8_t> k(c * h * w);  // k = [
                                      //   5, 6, 7, 8, 9,
                                      //   10, 11, 12, 13, 14,
                                      //   15, 16, 17, 18, 19]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 48387);
  const std::vector<uint8_t> expected = {
    // cr blocks
    // bias first (cr == 2 of them)
    // 48387 + 0 - (5 + 10 + 15) * 127 = 44577 = 0xAE21
    0x21, 0xAE, 0, 0,
    // 48387 + 1 - (6 + 11 + 16) * 127 = 44197 = 0xACA5
    0xA5, 0xAC, 0, 0,
    // then weights, channels first
    5, 6, 10, 11, 15, 16,
    // bias again
    // 48387 + 2 - (7, 12, 17) * 127 = 43817 = 0xAB29
    0x29, 0xAB, 0, 0,
    // 48387 + 3 - (8, 13, 18) * 127 = 43434 = 0xA9AD
    0xAD, 0xA9, 0, 0,
    // then weights, channels first
    7, 8, 12, 13, 17, 18,
    // bias again
    // 48387 + 4 - (9, 14, 19) * 127 = 43053 = 0xA831
    0x31, 0xA8, 0, 0,
    0, 0, 0, 0,
    // then weights, channels first
    9, 0, 14, 0, 19, 0,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_DWCONV_HWG_W, primary_tile_gt_kernel_size) {
  const size_t primary_tile = 9;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<uint8_t> k(c * h * w);  // k = [
                                      //   2, 3,
                                      //   4, 5,
                                      //   6, 7,
                                      //   8, 9]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<uint8_t> expected = {
    // bias first (cr == 2 of them)
    // 64516 + 0 - (2 + 4 + 6 + 8) * 127 = 61976 = 0xF218
    0x18, 0xF2, 0, 0,
    // 64516 + 1 - (3 + 5 + 7 + 9) * 127 = 61469 = 0xF01D
    0x1D, 0xF0, 0, 0,
    // then weights, channels first
    2, 3,
    // go down the columns first
    6, 7, 4, 5, 8, 9,
    // followed by 10 zeros to make up the difference with primary_tile
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_DWCONV_HWG_W, primary_tile_gt_kernel_size_channels_gt_cr) {
  const size_t primary_tile = 9;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint8_t> k(c * h * w);  // k = [
                                      //   5, 6, 7, 8, 9,
                                      //   10, 11, 12, 13, 14,
                                      //   15, 16, 17, 18, 19,
                                      //   20, 21, 22, 23, 24]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<uint8_t> expected = {
    // bias first (cr == 2 of them)
    // 64516 + 0 - (5 + 10 + 15 + 20) * 127 = 58166 = 0xE336
    0x36, 0xE3, 0, 0,
    // 64516 + 1 - (6 + 11 + 16 + 21) * 127 = 57659 = 0xE13B
    0x3B, 0xE1, 0, 0,
    // then weights, channels first
    5, 6,
    // go down the columns first
    15, 16,
    10, 11,
    20, 21,
    // followed by 10 zeros to make up the difference with primary_tile
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // bias first (cr == 2 of them)
    // 64516 + 2 - (7 + 12 + 17 + 22) * 127 = 57152 = 0xDF40
    0x40, 0xDF, 0, 0,
    // 64516 + 3 - (8 + 13 + 18 + 23) * 127 = 56645 = 0xDD45
    0x45, 0xDD, 0, 0,
    // then weights, channels first
    7, 8, 17, 18, 12, 13, 22, 23,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // bias
    // 64516 + 4 - (9 + 14 + 19 + 24) * 127 = 56138 = 0xDB4A
    0x4A, 0xDB, 0, 0,
    0, 0, 0, 0,
    // weights
    9, 0, 19, 0, 14, 0, 24, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_DWCONV_GHW_W, primary_tile_eq_kernel_size) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<int8_t> k(c * h * w);  // k = [2, 3, 4, 5, 6, 7]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));

  std::vector<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // bias first
    // (2 + 3 + 4) * 127 = -1143 = 0xFFFFFB89
    0x89, 0xFB, 0xFF, 0xFF,
    // (5 + 6 + 7) * 127 = -2285 = 0xFFFFF713
    0x13, 0xF7, 0xFF, 0xFF,
    // then weights, channels first
    2, 5,
    3, 6,
    4, 7,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_DWCONV_GHW_W, primary_tile_eq_kernel_size_channels_gt_cr) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<int8_t> k(c * h * w);  // k = [
                                      //   5, 6, 7,
                                      //   8, 9, 10,
                                      //   11, 12, 13,
                                      //   14, 15, 16,
                                      //   17, 18, 19]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // cr blocks
    // bias first (cr == 2 of them)
    // 0 - (5 + 6 + 7) * 127 = -2286 = 0xFFFFF712
    0x12, 0xF7, 0xFF, 0xFF,
    // 1 - (8 + 9 + 10) * 127 = -3428 = 0xFFFFF29C
    0x9C, 0xF2, 0xFF, 0xFF,
    // then weights, channels first
    5, 8, 6, 9, 7, 10,
    // bias again
    // 2 - (11 + 12 + 13) * 127 = -4570 = 0xFFFFEE26
    0x26, 0xEE, 0xFF, 0xFF,
    // 3 - (14 + 15 + 16) * 127 = -5712 = 0xFFFFE9B0
    0xB0, 0xE9, 0xFF, 0xFF,
    // then weights, channels first
    11, 14, 12, 15, 13, 16,
    // bias again
    // 4 - (17 + 18 + 19) * 127 = -6854 = 0xFFFFE53A
    0x3A, 0xE5, 0xFF, 0xFF,
    0, 0, 0, 0,
    // then weights, channels first
    17, 0, 18, 0, 19, 0,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_DWCONV_GHW_W, primary_tile_gt_kernel_size) {
  const size_t primary_tile = 9;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<int8_t> k(c * h * w);  // k = [
                                      //   2, 3,
                                      //   4, 5,
                                      //   6, 7,
                                      //   8, 9]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // bias first (cr == 2 of them)
    // 0 - (2 + 3 + 4 + 5) * 127 = -1778 = 0xFFFFF90E
    0x0E, 0xF9, 0xFF, 0xFF,
    // 1 - (6 + 7 + 8 + 9) * 127 = -3809 = 0xFFFFF11F
    0x1F, 0xF1, 0xFF, 0xFF,
    // then weights, channels first
    2, 6,
    // go down the columns first
    4, 8, 3, 7, 5, 9,
    // followed by 10 zeros to make up the difference with primary_tile
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_DWCONV_GHW_W, primary_tile_gt_kernel_size_channels_gt_cr) {
  const size_t primary_tile = 9;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<int8_t> k(c * h * w);  // k = [
                                      //   5, 6,
                                      //   7, 8,
                                      //   9, 10,
                                      //   11, 12,
                                      //   13, 14,
                                      //   15, 16,
                                      //   17, 18,
                                      //   19, 20,
                                      //   21, 22,
                                      //   23, 24]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // bias first (cr == 2 of them)
    // 0 - (5 + 6 + 7 + 8) * 127 = -3302 = 0xFFFFF31A
    0x1A, 0xF3, 0xFF, 0xFF,
    // 1 - (9 + 10 + 11 + 12) * 127 = -5333 = 0xFFFFEB2B
    0x2B, 0xEB, 0xFF, 0xFF,
    // then weights, channels first
    5, 9,
    // go down the columns first
    7, 11,
    6, 10,
    8, 12,
    // followed by 10 zeros to make up the difference with primary_tile
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // bias first (cr == 2 of them)
    // 2 - (13 + 14 + 15 + 16) * 127 = -7364 = 0xFFFFE33C
    0x3C, 0xE3, 0xFF, 0xFF,
    // 3 - (17 + 18 + 19 + 20) * 127 = -9395 = 0xFFFFDB4D
    0x4D, 0xDB, 0xFF, 0xFF,
    // then weights, channels first
    13, 17, 15, 19, 14, 18, 16, 20,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // bias
    // 4 - (21 + 22 + 23 + 24) * 127 = -11426 = 0xFFFFD35E
    0x5E, 0xD3, 0xFF, 0xFF,
    0, 0, 0, 0,
    // weights
    21, 0, 23, 0, 22, 0, 24, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_DWCONV_HWG_W, primary_tile_eq_kernel_size) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<int8_t> k(c * h * w);  // k = [2, 3, 4, 5, 6, 7]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));

  std::vector<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // bias first
    // 0 - (2 + 4 + 6) * 127 = -1524 = 0xFFFFFA0C
    0x0C, 0xFA, 0xFF, 0xFF,
    // 1 - (3 + 5 + 7) * 127 = -1904 = 0xFFFFF890
    0x90, 0xF8, 0xFF, 0xFF,
    // then weights, channels first
    2, 3,
    4, 5,
    6, 7,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_DWCONV_HWG_W, primary_tile_eq_kernel_size_channels_gt_cr) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<int8_t> k(c * h * w);  // k = [
                                      //   5, 6, 7, 8, 9,
                                      //   10, 11, 12, 13, 14,
                                      //   15, 16, 17, 18, 19]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // cr blocks
    // bias first (cr == 2 of them)
    // 0 - (5 + 10 + 15) * 127 = -3810 = 0xFFFFF11E
    0x1E, 0xF1, 0xFF, 0xFF,
    // 1 - (6 + 11 + 16) * 127 = -4190 = 0xFFFFEFA2
    0xA2, 0xEF, 0xFF, 0xFF,
    // then weights, channels first
    5, 6, 10, 11, 15, 16,
    // bias again
    // 2 - (7, 12, 17) * 127 = -45709 = 0xFFFFEE26
    0x26, 0xEE, 0xFF, 0xFF,
    // 3 - (8, 13, 18) * 127 = -4950 = 0xFFFFECAA
    0xAA, 0xEC, 0xFF, 0xFF,
    // then weights, channels first
    7, 8, 12, 13, 17, 18,
    // bias again
    // 4 - (9, 14, 19) * 127 = -5330 = 0xFFFFEB2E
    0x2E, 0xEB, 0xFF, 0xFF,
    0, 0, 0, 0,
    // then weights, channels first
    9, 0, 14, 0, 19, 0,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_DWCONV_HWG_W, primary_tile_gt_kernel_size) {
  const size_t primary_tile = 9;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<int8_t> k(c * h * w);  // k = [
                                      //   2, 3,
                                      //   4, 5,
                                      //   6, 7,
                                      //   8, 9]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // bias first (cr == 2 of them)
    // 0 - (2 + 4 + 6 + 8) * 127 = -2540 = 0xFFFFF614
    0x14, 0xF6, 0xFF, 0xFF,
    // 1 - (3 + 5 + 7 + 9) * 127 = -3047 = 0xFFFFF419
    0x19, 0xF4, 0xFF, 0xFF,
    // then weights, channels first
    2, 3,
    // go down the columns first
    6, 7, 4, 5, 8, 9,
    // followed by 10 zeros to make up the difference with primary_tile
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_DWCONV_HWG_W, primary_tile_gt_kernel_size_channels_gt_cr) {
  const size_t primary_tile = 9;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<int8_t> k(c * h * w);  // k = [
                                      //   5, 6, 7, 8, 9,
                                      //   10, 11, 12, 13, 14,
                                      //   15, 16, 17, 18, 19,
                                      //   20, 21, 22, 23, 24]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // bias first (cr == 2 of them)
    // 0 - (5 + 10 + 15 + 20) * 127 = -6350 = 0xFFFFE732
    0x32, 0xE7, 0xFF, 0xFF,
    // 1 - (6 + 11 + 16 + 21) * 127 = -6857 = 0xFFFFE537
    0x37, 0xE5, 0xFF, 0xFF,
    // then weights, channels first
    5, 6,
    // go down the columns first
    15, 16,
    10, 11,
    20, 21,
    // followed by 10 zeros to make up the difference with primary_tile
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // bias first (cr == 2 of them)
    // 2 - (7 + 12 + 17 + 22) * 127 = -7364 = 0xFFFFE33C
    0x3C, 0xE3, 0xFF, 0xFF,
    // 3 - (8 + 13 + 18 + 23) * 127 = -7871 = 0xFFFFE141
    0x41, 0xE1, 0xFF, 0xFF,
    // then weights, channels first
    7, 8, 17, 18, 12, 13, 22, 23,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // bias
    // 4 - (9 + 14 + 19 + 24) * 127 = -8378 = 0xFFFFDF46
    0x46, 0xDF, 0xFF, 0xFF,
    0, 0, 0, 0,
    // weights
    9, 0, 19, 0, 14, 0, 24, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_DWCONV_GHW_W, primary_tile_eq_kernel_size) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<uint16_t> k(c * h * w);  // k = [2, 3, 4, 5, 6, 7]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f16_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // bias first
    0, 1,
    // then weights, channels first
    2, 5,
    3, 6,
    4, 7,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_DWCONV_GHW_W, primary_tile_eq_kernel_size_channels_gt_cr) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint16_t> k(c * h * w);  // k = [
                                      //   5, 6, 7,
                                      //   8, 9, 10,
                                      //   11, 12, 13,
                                      //   14, 15, 16,
                                      //   17, 18, 19]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f16_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // cr blocks
    // bias first (cr == 2 of them)
    0, 1,
    // then weights, channels first
    5, 8, 6, 9, 7, 10,
    // bias again
    2, 3,
    // then weights, channels first
    11, 14, 12, 15, 13, 16,
    // bias again
    4, 0,
    // then weights, channels first
    17, 0, 18, 0, 19, 0,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_DWCONV_GHW_W, primary_tile_gt_kernel_size) {
  const size_t primary_tile = 9;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<uint16_t> k(c * h * w);  // k = [
                                      //   2, 3,
                                      //   4, 5,
                                      //   6, 7,
                                      //   8, 9]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f16_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // bias first (cr == 2 of them)
    0, 1,
    // then weights, channels first
    2, 6,
    // go down the columns first
    4, 8, 3, 7, 5, 9,
    // followed by 10 zeros to make up the difference with primary_tile
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_DWCONV_GHW_W, primary_tile_gt_kernel_size_channels_gt_cr) {
  const size_t primary_tile = 9;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint16_t> k(c * h * w);  // k = [
                                      //   5, 6,
                                      //   7, 8,
                                      //   9, 10,
                                      //   11, 12,
                                      //   13, 14,
                                      //   15, 16,
                                      //   17, 18,
                                      //   19, 20,
                                      //   21, 22,
                                      //   23, 24]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f16_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // bias first (cr == 2 of them)
    0, 1,
    // then weights, channels first
    5, 9,
    // go down the columns first
    7, 11,
    6, 10,
    8, 12,
    // followed by 10 zeros to make up the difference with primary_tile
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // bias first (cr == 2 of them)
    2, 3,
    // then weights, channels first
    13, 17, 15, 19, 14, 18, 16, 20,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // bias
    4, 0,
    // weights
    21, 0, 23, 0, 22, 0, 24, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_DWCONV_HWG_W, primary_tile_eq_kernel_size) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<uint16_t> k(c * h * w);  // k = [2, 3, 4, 5, 6, 7]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f16_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // bias first
    0, 1,
    // then weights, channels first
    2, 3,
    4, 5,
    6, 7,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_DWCONV_HWG_W, primary_tile_eq_kernel_size_channels_gt_cr) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint16_t> k(c * h * w);  // k = [
                                      //   5, 6, 7, 8, 9,
                                      //   10, 11, 12, 13, 14,
                                      //   15, 16, 17, 18, 19]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f16_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // cr blocks
    // bias first (cr == 2 of them)
    0, 1,
    // then weights, channels first
    5, 6, 10, 11, 15, 16,
    // bias again
    2, 3,
    // then weights, channels first
    7, 8, 12, 13, 17, 18,
    // bias again
    4, 0,
    // then weights, channels first
    9, 0, 14, 0, 19, 0,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_DWCONV_HWG_W, primary_tile_gt_kernel_size) {
  const size_t primary_tile = 9;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<uint16_t> k(c * h * w);  // k = [
                                      //   2, 3,
                                      //   4, 5,
                                      //   6, 7,
                                      //   8, 9]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f16_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // bias first (cr == 2 of them)
    0, 1,
    // then weights, channels first
    2, 3,
    // go down the columns first
    6, 7, 4, 5, 8, 9,
    // followed by 10 zeros to make up the difference with primary_tile
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_DWCONV_HWG_W, primary_tile_gt_kernel_size_channels_gt_cr) {
  const size_t primary_tile = 9;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint16_t> k(c * h * w);  // k = [
                                      //   5, 6, 7, 8, 9,
                                      //   10, 11, 12, 13, 14,
                                      //   15, 16, 17, 18, 19,
                                      //   20, 21, 22, 23, 24]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f16_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // bias first (cr == 2 of them)
    0, 1,
    // then weights, channels first
    5, 6,
    // go down the columns first
    15, 16,
    10, 11,
    20, 21,
    // followed by 10 zeros to make up the difference with primary_tile
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // bias first (cr == 2 of them)
    2, 3,
    // then weights, channels first
    7, 8, 17, 18, 12, 13, 22, 23,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // bias
    4, 0,
    // weights
    9, 0, 19, 0, 14, 0, 24, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_DWCONV_GHW_W, primary_tile_eq_kernel_size) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1]
  std::vector<float> k(c * h * w);  // k = [2, 3, 4, 5, 6, 7]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // bias first
    0.0f, 1.0f,
    // then weights, channels first
    2.0f, 5.0f,
    3.0f, 6.0f,
    4.0f, 7.0f,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_DWCONV_GHW_W, primary_tile_eq_kernel_size_channels_gt_cr) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4]
  std::vector<float> k(c * h * w);  // k = [
                                      //   5, 6, 7,
                                      //   8, 9, 10,
                                      //   11, 12, 13,
                                      //   14, 15, 16,
                                      //   17, 18, 19]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // cr blocks
    // bias first (cr == 2 of them)
    0.0f, 1.0f,
    // then weights, channels first
    5.0f, 8.0f, 6.0f, 9.0f, 7.0f, 10.0f,
    // bias again
    2.0f, 3.0f,
    // then weights, channels first
    11.0f, 14.0f, 12.0f, 15.0f, 13.0f, 16.0f,
    // bias again
    4.0f, 0.0f,
    // then weights, channels first
    17.0f, 0.0f, 18.0f, 0.0f, 19.0f, 0.0f,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_DWCONV_GHW_W, primary_tile_gt_kernel_size) {
  const size_t primary_tile = 9;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1]
  std::vector<float> k(c * h * w);  // k = [
                                      //   2, 3,
                                      //   4, 5,
                                      //   6, 7,
                                      //   8, 9]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // bias first (cr == 2 of them)
    0.0f, 1.0f,
    // then weights, channels first
    2.0f, 6.0f,
    // go down the columns first
    4.0f, 8.0f, 3.0f, 7.0f, 5.0f, 9.0f,
    // followed by 10 zeros to make up the difference with primary_tile
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_DWCONV_GHW_W, primary_tile_gt_kernel_size_channels_gt_cr) {
  const size_t primary_tile = 9;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4]
  std::vector<float> k(c * h * w);  // k = [
                                      //   5, 6,
                                      //   7, 8,
                                      //   9, 10,
                                      //   11, 12,
                                      //   13, 14,
                                      //   15, 16,
                                      //   17, 18,
                                      //   19, 20,
                                      //   21, 22,
                                      //   23, 24]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // bias first (cr == 2 of them)
    0.0f, 1.0f,
    // then weights, channels first
    5.0f, 9.0f,
    // go down the columns first
    7.0f, 11.0f,
    6.0f, 10.0f,
    8.0f, 12.0f,
    // followed by 10 zeros to make up the difference with primary_tile
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    // bias first (cr == 2 of them)
    2.0f, 3.0f,
    // then weights, channels first
    13.0f, 17.0f, 15.0f, 19.0f, 14.0f, 18.0f, 16.0f, 20.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    // bias
    4.0f, 0.0f,
    // weights
    21.0f, 0.0f, 23.0f, 0.0f, 22.0f, 0.0f, 24.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_DWCONV_HWG_W, primary_tile_eq_kernel_size) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1]
  std::vector<float> k(c * h * w);  // k = [2, 3, 4, 5, 6, 7]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // bias first
    0.0f, 1.0f,
    // then weights, channels first
    2.0f, 3.0f,
    4.0f, 5.0f,
    6.0f, 7.0f,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_DWCONV_HWG_W, primary_tile_eq_kernel_size_channels_gt_cr) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4]
  std::vector<float> k(c * h * w);  // k = [
                                      //   5, 6, 7, 8, 9,
                                      //   10, 11, 12, 13, 14,
                                      //   15, 16, 17, 18, 19]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // cr blocks
    // bias first (cr == 2 of them)
    0.0f, 1.0f,
    // then weights, channels first
    5.0f, 6.0f, 10.0f, 11.0f, 15.0f, 16.0f,
    // bias again
    2.0f, 3.0f,
    // then weights, channels first
    7.0f, 8.0f, 12.0f, 13.0f, 17.0f, 18.0f,
    // bias again
    4.0f, 0.0f,
    // then weights, channels first
    9.0f, 0.0f, 14.0f, 0.0f, 19.0f, 0.0f,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_DWCONV_HWG_W, primary_tile_gt_kernel_size) {
  const size_t primary_tile = 9;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1]
  std::vector<float> k(c * h * w);  // k = [
                                      //   2, 3,
                                      //   4, 5,
                                      //   6, 7,
                                      //   8, 9]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // bias first (cr == 2 of them)
    0.0f, 1.0f,
    // then weights, channels first
    2.0f, 3.0f,
    // go down the columns first
    6.0f, 7.0f, 4.0f, 5.0f, 8.0f, 9.0f,
    // followed by 10 zeros to make up the difference with primary_tile
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_DWCONV_HWG_W, primary_tile_gt_kernel_size_channels_gt_cr) {
  const size_t primary_tile = 9;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4]
  std::vector<float> k(c * h * w);  // k = [
                                      //   5, 6, 7, 8, 9,
                                      //   10, 11, 12, 13, 14,
                                      //   15, 16, 17, 18, 19,
                                      //   20, 21, 22, 23, 24]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // bias first (cr == 2 of them)
    0.0f, 1.0f,
    // then weights, channels first
    5.0f, 6.0f,
    // go down the columns first
    15.0f, 16.0f,
    10.0f, 11.0f,
    20.0f, 21.0f,
    // followed by 10 zeros to make up the difference with primary_tile
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    // bias first (cr == 2 of them)
    2.0f, 3.0f,
    // then weights, channels first
    7.0f, 8.0f, 17.0f, 18.0f, 12.0f, 13.0f, 22.0f, 23.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    // bias
    4.0f, 0.0f,
    // weights
    9.0f, 0.0f, 19.0f, 0.0f, 14.0f, 0.0f, 24.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  };
  EXPECT_EQ(expected, packed_weights);
}

// Calculates the size (number of elements) of packed weights required for a multi-pass dwconv.
static size_t multipass_weights_count(
  size_t kernel_size,
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t channels,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round)
{
  return xnn_dwconv_multipass_weights_count(
    xnn_dwconv_multipass_tile_size(kernel_size, first_pass_tile, middle_pass_tile, last_pass_tile), channels,
    channel_tile, channel_subtile, channel_round);
}

// Convenient overload when channel_tile == channel_subtile.
static size_t multipass_weights_count(
  size_t kernel_size,
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t channels,
  size_t channel_tile)
{
  return multipass_weights_count(
    kernel_size, first_pass_tile, middle_pass_tile, last_pass_tile, channels, channel_tile, channel_tile, channel_tile);
}

// Convenient overload when channel_subtile == channel_round.
static size_t multipass_weights_count(
  size_t kernel_size,
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t channels,
  size_t channel_tile,
  size_t channel_subtile)
{
  return multipass_weights_count(
    kernel_size, first_pass_tile, middle_pass_tile, last_pass_tile, channels, channel_tile, channel_subtile, channel_subtile);
}

TEST(PACK_F32_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1]
  std::vector<float> k(c * h * w);  // k = [2, 3, // first 2x2 kernel
                                    //      4, 5,
                                    //      6, 7, // second 2x2 kernel
                                    //      8, 9]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_dwconv_multipass_ghw_w(
      first_pass_tile,
      middle_pass_tile,
      last_pass_tile,
      h,
      w,
      c,
      cr,
      cr,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass.
    0.0f, 1.0f,  // bias
    2.0f, 6.0f,  // 2 weights, channels first, then columns
    4.0f, 8.0f,
    // No middle pass.
    3.0f, 7.0f,  // last pass, 2 weights
    5.0f, 9.0f,
    0.0f, 0.0f,  // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once_channels_gt_cr) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4]
  std::vector<float> k(c * h * w);  // k = [5, 6, 7, 8, // first 2x2 kernel
                                    //      9, 10, 11, 12, // second 2x2 kernel
                                    //      13, 14, 15, 16, // third 2x2 kernel
                                    //      17, 18, 19, 20, // fourth 2x2 kernel
                                    //      21, 22, 23, 24, // fifth 2x2 kernel
                                    //      ]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_dwconv_multipass_ghw_w(
      first_pass_tile,
      middle_pass_tile,
      last_pass_tile,
      h,
      w,
      c,
      cr,
      cr,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass.
    0.0f, 1.0f, // bias
    5.0f, 9.0f, // 2 weights, 2 channels first, then columns
    7.0f, 11.0f,
    2.0f, 3.0f, // bias
    13.0f, 17.0f, // 2 weights, 2 channels first, then columns
    15.0f, 19.0f,
    4.0f, 0.0f, // bias
    21.0f, 0.0f, // 2 weights, 1 remainder channels first, then columns
    23.0f, 0.0f,
    // No middle pass.
    6.0f, 10.0f, // last pass, 2 weights, 2 channels first, then columns
    8.0f, 12.0f,
    0.0f, 0.0f,  // padding
    14.0f, 18.0f,
    16.0f, 20.0f,
    0.0f, 0.0f,  // padding
    22.0f, 0.0f,
    24.0f, 0.0f,
    0.0f, 0.0f,  // padding

  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_MULTIPASS_DWCONV_GHW_W, one_middle_pass_tile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 1;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1]
  std::vector<float> k(c * h * w);  // k = [2, 3, // first 2x2 kernel
                                    //      4, 5,
                                    //      6, 7, // second 2x2 kernel
                                    //      8, 9]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_dwconv_multipass_ghw_w(
      first_pass_tile,
      middle_pass_tile,
      last_pass_tile,
      h,
      w,
      c,
      cr,
      cr,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass only has 1 element.
    0.0f, 1.0f, // bias
    2.0f, 6.0f, // weights, 2 channels, 1 element.
    // Middle pass has 2 elements, columns first.
    4.0f, 8.0f,
    3.0f, 7.0f,
    // Last pass.
    5.0f, 9.0f,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_MULTIPASS_DWCONV_GHW_W, one_middle_pass_tile_channels_gt_cr) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4]
  std::vector<float> k(c * h * w);  // k = [5, 6, // first 2x2 kernel
                                    //      7, 8,
                                    //      9, 10, // second 2x2 kernel
                                    //      11, 12,
                                    //      13, 14, // third 2x2 kernel
                                    //      15, 16,
                                    //      17, 18, // fourth 2x2 kernel
                                    //      19, 20,
                                    //      21, 22, // fifth 2x2 kernel
                                    //      23, 24,
                                    //      ]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_dwconv_multipass_ghw_w(
      first_pass_tile,
      middle_pass_tile,
      last_pass_tile,
      h,
      w,
      c,
      cr,
      cr,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass only has 1 element.
    0.0f, 1.0f, // bias
    5.0f, 9.0f, // weights, 2 channels, 1 element.
    2.0f, 3.0f, // bias
    13.0f, 17.0f, // weights, 2 channels, 1 element.
    4.0f, 0.0f, // bias
    21.0f, 0.0f, // weights, 1 remainder channel, 1 element.
    // Middle pass has 2 elements, channels first, then columns.
    7.0f, 11.0f,
    6.0f, 10.0f,
    15.0f, 19.0f,
    14.0f, 18.0f,
    // Middle pass, 1 remainder channel.
    23.0f, 0.0f,
    22.0f, 0.0f,
    // Last pass,
    8.0f, 12.0f,
    0.0f, 0.0f,  // padding
    16.0f, 20.0f,
    0.0f, 0.0f,  // padding
    24.0f, 0.0f,
    0.0f, 0.0f,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_MULTIPASS_DWCONV_GHW_W, multiple_middle_pass_tile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 3;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1]
  std::vector<float> k(c * h * w);  // k = [2, 3, 4, // first 2x3 kernel
                                    //      5, 6, 7,
                                    //      8, 9, 10, // second 2x3 kernel
                                    //      11, 12, 13]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_dwconv_multipass_ghw_w(
      first_pass_tile,
      middle_pass_tile,
      last_pass_tile,
      h,
      w,
      c,
      cr,
      cr,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass has 2 elements.
    0.0f, 1.0f, // bias
    2.0f, 8.0f, // 1 weight, 2 channels first, then columns
    // Middle pass 1 (2 elements per pass).
    5.0f, 11.0f,
    3.0f, 9.0f,
    // Middle pass 2 (2 elements per pass).
    6.0f, 12.0f,
    4.0f, 10.0f,
    // Last pass.
    7.0f, 13.0f,
    0.0f, 0.0f,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_MULTIPASS_DWCONV_GHW_W, multiple_middle_pass_tile_channels_gt_cr) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 3;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4]
  std::vector<float> k(c * h * w);  // k = [5, 6, 7, // first 2x3 kernel
                                    //      8, 9, 10,
                                    //      11, 12, 13, // second 2x3 kernel
                                    //      14, 15, 16,
                                    //      17, 18, 19, // third 2x3 kernel
                                    //      20, 21, 22,
                                    //      23, 24, 25, // fourth 2x3 kernel
                                    //      26, 27, 28,
                                    //      29, 30, 31, // fifth 2x3 kernel
                                    //      32, 33, 34,
                                    //      ]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_dwconv_multipass_ghw_w(
      first_pass_tile,
      middle_pass_tile,
      last_pass_tile,
      h,
      w,
      c,
      cr,
      cr,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);


  const std::vector<float> expected = {
    // First pass has 1 element.
    0.0f, 1.0f, // bias
    5.0f, 11.0f, // 1 weight, 2 channels, 2 elements.
    2.0f, 3.0f, // bias
    17.0f, 23.0f, // 1 weight, 2 channels, 2 elements.
    4.0f, 0.0f, // bias
    29.0f, 0.0f, // 1 weight, 1 remainder channel, 2 elements.
    // Middle pass has 2 elements, channels first, then columns.
    8.0f, 14.0f,
    6.0f, 12.0f,
    20.0f, 26.0f,
    18.0f, 24.0f,
    // 1 remainder channel.
    32.0f, 0.0f,
    30.0f, 0.0f,
    // Second middle pass, 2 elements.
    9.0f, 15.0f,
    7.0f, 13.0f,
    21.0f, 27.0f,
    19.0f, 25.0f,
    // 1 remainder channel.
    33.0f, 0.0f,
    31.0f, 0.0f,
    // Last pass
    10.0f, 16.0f,
    0.0f, 0.0f,  // padding
    22.0f, 28.0f,
    0.0f, 0.0f,  // padding
    // 1 remainder channel.
    34.0f, 0.0f,
    0.0f, 0.0f,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once_channel_subtile) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 4;
  const size_t channel_subtile = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4]
  std::vector<float> k(c * h * w);  // k = [5, 6, // first 2x2 kernel
                                    //      7, 8,
                                    //      9, 10, // second 2x2 kernel
                                    //      11, 12,
                                    //      13, 14, // third 2x2 kernel
                                    //      15, 16,
                                    //      17, 18, // fourth 2x2 kernel
                                    //      19, 20,
                                    //      21, 22, // fifth 2x2 kernel
                                    //      23, 24 ]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f32_dwconv_multipass_ghw_w(
      first_pass_tile,
      middle_pass_tile,
      last_pass_tile,
      h,
      w,
      c,
      cr,
      channel_subtile,
      channel_subtile,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass.
    0.0f, 1.0f, 2.0f, 3.0f,  // bias, 4 channels
    5.0f, 9.0f, 13.0, 17.0f,  // 2 weights, 4 channels first, then columns
    7.0f, 11.0f, 15.0f, 19.0f,
    4.0f, 0.0f, // bias, 1 last channel, 1 padding up to channel_subtile
    21.0f, 0.0f, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    23.0f, 0.0f,
    // No middle pass.
    6.0f, 10.0f, 14.0f, 18.0f,  // last pass, 2 weights, 4 channels first
    8.0f, 12.0f, 16.0f, 20.0f,
    0.0f, 0.0f, 0.0f, 0.0f, // padding to last_pass_tile
    22.0f, 0.0f, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    24.0f, 0.0f,
    0.0f, 0.0f, // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_MULTIPASS_DWCONV_GHW_W, one_middle_pass_channel_subtile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 4;
  const size_t channel_subtile = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4]
  std::vector<float> k(c * h * w);  // k = [5, 6, // first 2x2 kernel
                                    //      7, 8,
                                    //      9, 10, // second 2x2 kernel
                                    //      11, 12,
                                    //      13, 14, // third 2x2 kernel
                                    //      15, 16,
                                    //      17, 18, // fourth 2x2 kernel
                                    //      19, 20,
                                    //      21, 22, // fifth 2x2 kernel
                                    //      23, 24 ]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f32_dwconv_multipass_ghw_w(
      first_pass_tile,
      middle_pass_tile,
      last_pass_tile,
      h,
      w,
      c,
      cr,
      channel_subtile,
      channel_subtile,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass.
    0.0f, 1.0f, 2.0f, 3.0f,  // bias, 4 channels
    5.0f, 9.0f, 13.0, 17.0f,  // 1 weight, 4 channels first, then columns
    4.0f, 0.0f, // bias, 1 last channel, 1 padding up to channel_subtile
    21.0f, 0.0f, // 1 weight, 1 last channel, 1 padding up to channel_subtile
    // 1 middle pass
    7.0f, 11.0f, 15.0f, 19.0f, // 2 weights, 4 channels first, then columns
    6.0f, 10.0f, 14.0f, 18.0f,
    23.0f, 0.0f, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    22.0f, 0.0f,
    // Last pass.
    8.0f, 12.0f, 16.0f, 20.0f,  // 1 weight, 4 channels first
    0.0f, 0.0f, 0.0f, 0.0f, // padding to last_pass_tile
    24.0f, 0.0f, // 1 weight, 1 last channel, 1 padding up to channel_subtile
    0.0f, 0.0f, // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once_channel_subtile_rounded) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 7;
  const size_t cr = 4;
  const size_t channel_subtile = 2;
  // c rounded to channel_subtile is 8, so we will have 2 channel_tile loops in the first and middle pass.

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4, 5, 6]
  std::vector<float> k(c * h * w);  // k = [7, 8, // first 2x2 kernel
                                    //      9, 10,
                                    //      11, 12, // second 2x2 kernel
                                    //      13, 14,
                                    //      15, 16, // third 2x2 kernel
                                    //      17, 18,
                                    //      19, 20, // fourth 2x2 kernel
                                    //      21, 22,
                                    //      23, 24, // fifth 2x2 kernel
                                    //      25, 26,
                                    //      27, 28, // sixth 2x2 kernel
                                    //      29, 30,
                                    //      31, 32, // seventh 2x2 kernel
                                    //      33, 34]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f32_dwconv_multipass_ghw_w(
      first_pass_tile,
      middle_pass_tile,
      last_pass_tile,
      h,
      w,
      c,
      cr,
      channel_subtile,
      channel_subtile,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass.
    0.0f, 1.0f, 2.0f, 3.0f,  // bias, 4 channels
    7.0f, 11.0f, 15.0, 19.0f,  // 2 weights, 4 channels first, then columns
    9.0f, 13.0f, 17.0f, 21.0f,
    4.0f, 5.0f, 6.0f, 0.0f, // bias, 3 remainder channels, 1 padding up to channel_tile
    23.0f, 27.0f, 31.0f, 0.0f,  // 2 weights, 3 remainder channels, 1 padding up to channel_tile
    25.0f, 29.0f, 33.0f, 0.0f,
    // No middle pass.
    // Last pass.
    8.0f, 12.0f, 16.0f, 20.0f,  // last pass, 2 weights, 4 channels first
    10.0f, 14.0f, 18.0f, 22.0f,
    0.0f, 0.0f, 0.0f, 0.0f, // padding to last_pass_tile
    24.0f, 28.0f, // last pass, 2 weights, channel_subtile (2)
    26.0f, 30.0f,
    0.0f, 0.0f, // padding to last_pass_tile
    32.0f, 0.0f, // 1 remainder channel, 1 padding up to channel_subtile
    34.0f, 0.0f, // 1 remainder channel, 1 padding up to channel_subtile
    0.0f, 0.0f, // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once_channel_subtile_rounded_to_channel_round) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 6;
  const size_t cr = 8;
  const size_t channel_subtile = 4;
  const size_t channel_round = 2;
  // c rounded to channel_round is 6, so we will have 0 channel_tile and 2 channel_subtile loops
  // for first and middle pass.

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4, 5]
  std::vector<float> k(c * h * w);  // k = [6, 7, 8, 9,
                                    //      10, 11, 12, 13
                                    //      14, 15, 16, 17,
                                    //      18, 19, 20, 21,
                                    //      22, 23, 24, 25,
                                    //      26, 27, 28, 29,]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(
    multipass_weights_count(
      h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile, channel_round));

  xnn_pack_f32_dwconv_multipass_ghw_w(
      first_pass_tile,
      middle_pass_tile,
      last_pass_tile,
      h,
      w,
      c,
      cr,
      channel_subtile,
      channel_round,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass.
    0.0f, 1.0f, 2.0f, 3.0f, // bias, 4 channels
    6.0f, 10.0f, 14.0f, 18.0f, // 2 weights, 4 channels first, then columns
    8.0f, 12.0f, 16.0f, 20.0f,
    4.0f, 5.0f, 0.0f, 0.0f, // bias, 2 remainder channels, 1 padding up to channel_subtile
    22.0f, 26.0f, 0.0f, 0.0f, // 2 weights, 2 remainder channel, 1 padding up to channel_subtile
    24.0f, 28.0f, 0.0f, 0.0f,
    // No middle pass.
    7.0f, 11.0f, 15.0f, 19.0f, // last pass, 2 weights, 4 channels first.
    9.0f, 13.0f, 17.0f, 21.0f,
    0.0f, 0.0f, 0.0f, 0.0f, // padding to last_pass_tile
    23.0f, 27.0f, 0.0f, 0.0f, // 2 weights, channel_subtile (4)
    25.0f, 29.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_MULTIPASS_DWCONV_GHW_W, one_middle_pass_channel_subtile_rounded) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 7;
  const size_t cr = 4;
  const size_t channel_subtile = 2;
  // c rounded to channel_subtile is 8, so we will have 2 channel_tile loops in first and middle pass.

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4, 5, 6]
  std::vector<float> k(c * h * w);  // k = [7, 8, // first 2x2 kernel
                                    //      9, 10,
                                    //      11, 12, // second 2x2 kernel
                                    //      13, 14,
                                    //      15, 16, // third 2x2 kernel
                                    //      17, 18,
                                    //      19, 20, // fourth 2x2 kernel
                                    //      21, 22,
                                    //      23, 24, // fifth 2x2 kernel
                                    //      25, 26,
                                    //      27, 28, // sixth 2x2 kernel
                                    //      29, 30,
                                    //      31, 32, // seventh 2x2 kernel
                                    //      33, 34]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f32_dwconv_multipass_ghw_w(
      first_pass_tile,
      middle_pass_tile,
      last_pass_tile,
      h,
      w,
      c,
      cr,
      channel_subtile,
      channel_subtile,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass.
    0.0f, 1.0f, 2.0f, 3.0f,  // bias, 4 channels
    7.0f, 11.0f, 15.0, 19.0f,  // 1 weight, 4 channels first, then columns
    4.0f, 5.0f, 6.0f, 0.0f, // bias, 4 channels, 1 padding up to channel_tile
    23.0f, 27.0f, 31.0f, 0.0f,  // 1 weight, 3 remainder channels, 1 padding up to channel_tile
    // 1 middle pass.
    9.0f, 13.0f, 17.0f, 21.0f, // 2 weights, 4 channels first
    8.0f, 12.0f, 16.0f, 20.0f,
    25.0f, 29.0f, 33.0f, 0.0f, // 3 remainder channels, 1 padding up to channel_tile
    24.0f, 28.0f, 32.0f, 0.0f,
    // Last pass.
    10.0f, 14.0f, 18.0f, 22.0f,  // last pass, 1 weight, 4 channels first
    0.0f, 0.0f, 0.0f, 0.0f, // padding to last_pass_tile
    26.0f, 30.0f, // 1 weight, channel_subtile,
    0.0f, 0.0f, // padding to last_pass_tile
    34.0f, 0.0f, // 1 weight, 1 remainder channel, 1 padding up to channel_subtile
    0.0f, 0.0f, // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1]
  std::vector<float> k(c * h * w);  // k = [2, 3,
                                    //      4, 5,
                                    //      6, 7,
                                    //      8, 9]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_dwconv_multipass_hwg_w(
      first_pass_tile,
      middle_pass_tile,
      last_pass_tile,
      h,
      w,
      c,
      cr,
      cr,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass.
    0.0f, 1.0f, // bias
    2.0f, 3.0f, // First pass, 2 weights, channels first, then columns
    6.0f, 7.0f,
    // No middle pass.
    4.0f, 5.0f, // Last pass, 2 weights
    8.0f, 9.0f,
    0.0f, 0.0f,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once_channels_gt_cr) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4]
  std::vector<float> k(c * h * w);  // k = [5, 6, 7, 8, 9,
                                    //      10, 11, 12, 13, 14,
                                    //      15, 16, 17, 18, 19,
                                    //      20, 21, 22, 23, 24,
                                    //      ]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_dwconv_multipass_hwg_w(
      first_pass_tile,
      middle_pass_tile,
      last_pass_tile,
      h,
      w,
      c,
      cr,
      cr,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass.
    0.0f, 1.0f, // bias
    5.0f, 6.0f, // 2 weights, 2 channels first, then columns
    15.0f, 16.0f,
    2.0f, 3.0f, // bias
    7.0f, 8.0f, // weights, 2 channels first, then columns
    17.0f, 18.0f,
    4.0f, 0.0f, // bias
    9.0f, 0.0f, // weights, 1 remainder channels first, then columns
    19.0f, 0.0f,
    // No middle pass.
    // Last pass, 2 weights
    10.0f, 11.0f,
    20.0f, 21.0f,
    0.0f, 0.0f,  // padding
    12.0f, 13.0f,
    22.0f, 23.0f,
    0.0f, 0.0f,  // padding
    14.0f, 0.0f,
    24.0f, 0.0f,
    0.0f, 0.0f,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_MULTIPASS_DWCONV_HWG_W, one_middle_pass_tile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 1;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1]
  std::vector<float> k(c * h * w);  // k = [2, 3,
                                    //      4, 5,
                                    //      6, 7,
                                    //      8, 9]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_dwconv_multipass_hwg_w(
      first_pass_tile,
      middle_pass_tile,
      last_pass_tile,
      h,
      w,
      c,
      cr,
      cr,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass only has 1 element.
    0.0f, 1.0f, // bias
    2.0f, 3.0f, // weights, 2 channels, 1 element.
    // Middle pass has 2 elements, columns first.
    6.0f, 7.0f,
    4.0f, 5.0f,
    // Last pass.
    8.0f, 9.0f,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_MULTIPASS_DWCONV_HWG_W, one_middle_pass_tile_channels_gt_cr) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4]
  std::vector<float> k(c * h * w);  // k = [5, 6, 7, 8, 9,
                                    //      10, 11, 12, 13, 14,
                                    //      15, 16, 17, 18, 19,
                                    //      20, 21, 22, 23, 24,
                                    //      ]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_dwconv_multipass_hwg_w(
      first_pass_tile,
      middle_pass_tile,
      last_pass_tile,
      h,
      w,
      c,
      cr,
      cr,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass only has 1 element.
    0.0f, 1.0f, // bias
    5.0f, 6.0f, // weights, 2 channels, 1 element.
    2.0f, 3.0f, // bias
    7.0f, 8.0f, // weights, 2 channels, 1 element.
    4.0f, 0.0f, // bias
    9.0f, 0.0f, // weights, 1 remainder channel, 1 element.
    // Middle pass has 2 elements, channels first, then columns.
    15.0f, 16.0f,
    10.0f, 11.0f,
    17.0f, 18.0f,
    12.0f, 13.0f,
    // Middle pass, 1 remainder channel.
    19.0f, 0.0f,
    14.0f, 0.0f,
    // Last pass.
    20.0f, 21.0f,
    0.0f, 0.0f,  // padding
    22.0f, 23.0f,
    0.0f, 0.0f,  // padding
    24.0f, 0.0f,
    0.0f, 0.0f,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_MULTIPASS_DWCONV_HWG_W, multiple_middle_pass_tile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 3;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1]
  std::vector<float> k(c * h * w);  // k = [2, 3,
                                    //      4, 5,
                                    //      6, 7,
                                    //      8, 9,
                                    //      10, 11,
                                    //      12, 13]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_dwconv_multipass_hwg_w(
      first_pass_tile,
      middle_pass_tile,
      last_pass_tile,
      h,
      w,
      c,
      cr,
      cr,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass has 2 elements.
    0.0f, 1.0f, // bias
    2.0f, 3.0f, // 1 weight, 2 channels first, then columns
    // 2 passes of middle pass (2 elements per pass).
    8.0f, 9.0f,
    4.0f, 5.0f,
    10.0f, 11.0f,
    6.0f, 7.0f,
    // Last pass.
    12.0f, 13.0f,
    0.0f, 0.0f,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_MULTIPASS_DWCONV_HWG_W, multiple_middle_pass_tile_channels_gt_cr) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 3;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4]
  std::vector<float> k(c * h * w);  // k = [5, 6, 7, 8, 9,
                                    //      10, 11, 12, 13, 14,
                                    //      15, 16, 17, 18, 19,
                                    //      20, 21, 22, 23, 24,
                                    //      25, 26, 27, 28, 29,
                                    //      30, 31, 32, 33, 34,
                                    //      ]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_dwconv_multipass_hwg_w(
      first_pass_tile,
      middle_pass_tile,
      last_pass_tile,
      h,
      w,
      c,
      cr,
      cr,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);


  const std::vector<float> expected = {
    // First pass has 1 element.
    0.0f, 1.0f, // bias
    5.0f, 6.0f, // weights, 2 channels, 2 elements.
    2.0f, 3.0f, // bias
    7.0f, 8.0f, // weights, 2 channels, 2 elements.
    4.0f, 0.0f, // bias
    9.0f, 0.0f, // weights, 1 remainder channel, 2 elements.
    // Middle pass has 2 elements, channels first, then columns.
    20.0f, 21.0f,
    10.0f, 11.0f,
    22.0f, 23.0f,
    12.0f, 13.0f,
    // 1 remainder channel.
    24.0f, 0.0f,
    14.0f, 0.0f,
    // Second middle pass, 2 elements.
    25.0f, 26.0f,
    15.0f, 16.0f,
    27.0f, 28.0f,
    17.0f, 18.0f,
    29.0f, 0.0f,
    // 1 remainder channel.
    19.0f, 0.0f,
    // Last pass.
    30.0f, 31.0f,
    0.0f, 0.0f,  // padding
    32.0f, 33.0f,
    0.0f, 0.0f,  // padding
    // 1 remainder channel.
    34.0f, 0.0f,
    0.0f, 0.0f,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once_channel_subtile) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 4;
  const size_t channel_subtile = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4]
  std::vector<float> k(c * h * w);  // k = [5, 6, 7, 8, 9, // first channel
                                    //      10, 11, 12, 13, 14, // second channel
                                    //      15, 16, 17, 18, 19, // third channel
                                    //      20, 21, 22, 23, 24] // fourth channel
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f32_dwconv_multipass_hwg_w(
      first_pass_tile,
      middle_pass_tile,
      last_pass_tile,
      h,
      w,
      c,
      cr,
      channel_subtile,
      channel_subtile,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass.
    0.0f, 1.0f, 2.0f, 3.0f, // bias, 4 channels
    5.0f, 6.0f, 7.0f, 8.0f, // first pass, 2 weights, 4 channels first, then columns
    15.0f, 16.0f, 17.0f, 18.0f,
    4.0f, 0.0f, // bias, 1 last channel, 1 padding up to channel_subtile
    9.0f, 0.0f, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    19.0f, 0.0f,
    // No middle pass.
    10.0f, 11.0f, 12.0f, 13.0f, // last pass, 2 weights, 4 channels first.
    20.0f, 21.0f, 22.0f, 23.0f,
    0.0f, 0.0f, 0.0f, 0.0f, // padding to last_pass_tile
    14.0f, 0.0f, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    24.0f, 0.0f, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    0.0f, 0.0f,  // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_MULTIPASS_DWCONV_HWG_W, one_middle_pass_channel_subtile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 4;
  const size_t channel_subtile = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4]
  std::vector<float> k(c * h * w);  // k = [5, 6, 7, 8, 9,
                                    //      10, 11, 12, 13, 14,
                                    //      15, 16, 17, 18, 19,
                                    //      20, 21, 22, 23, 24]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f32_dwconv_multipass_hwg_w(
      first_pass_tile,
      middle_pass_tile,
      last_pass_tile,
      h,
      w,
      c,
      cr,
      channel_subtile,
      channel_subtile,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass.
    0.0f, 1.0f, 2.0f, 3.0f, // bias, 4 channels
    5.0f, 6.0f, 7.0f, 8.0f, // first pass, 1 weight, 4 channels first, then columns
    4.0f, 0.0f, // bias, 1 last channel, 1 padding up to channel_subtile
    9.0f, 0.0f, // 1 weight, 1 last channel, 1 padding up to channel_subtile
    // 1 middle pass.
    15.0f, 16.0f, 17.0f, 18.0f, // 2 weights, 4 channels first.
    10.0f, 11.0f, 12.0f, 13.0f,
    19.0f, 0.0f, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    14.0f, 0.0f,
    // Last pass.
    20.0f, 21.0f, 22.0f, 23.0f, // 1 weight, 4 channels first.
    0.0f, 0.0f, 0.0f, 0.0f, // padding to last_pass_tile
    24.0f, 0.0f, // 1 weight, 1 last channel, 1 padding up to channel_subtile
    0.0f, 0.0f,  // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once_channel_subtile_rounded) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 7;
  const size_t cr = 4;
  const size_t channel_subtile = 2;
  // c rounded to channel_subtile is 8, so we will have 2 channel_tile loops for first and middle pass.

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4, 5, 6]
  std::vector<float> k(c * h * w);  // k = [7, 8, 9, 10, 11, 12, 13,
                                    //      14, 15, 16, 17, 18, 19, 20,
                                    //      21, 22, 23, 24, 25, 26, 27,
                                    //      28, 29, 30, 31, 32, 33, 34]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f32_dwconv_multipass_hwg_w(
      first_pass_tile,
      middle_pass_tile,
      last_pass_tile,
      h,
      w,
      c,
      cr,
      channel_subtile,
      channel_subtile,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass.
    0.0f, 1.0f, 2.0f, 3.0f, // bias, 4 channels
    7.0f, 8.0f, 9.0f, 10.0f, // 2 weights, 4 channels first, then columns
    21.0f, 22.0f, 23.0f, 24.0f,
    4.0f, 5.0f, 6.0f, 0.0f, // bias, 3 remainder channels, 1 padding up to channel_subtile
    11.0f, 12.0f, 13.0f, 0.0f, // 2 weights, 3 remainder channel, 1 padding up to channel_subtile
    25.0f, 26.0f, 27.0f, 0.0f,
    // No middle pass.
    14.0f, 15.0f, 16.0f, 17.0f, // last pass, 2 weights, 4 channels first.
    28.0f, 29.0f, 30.0f, 31.0f,
    0.0f, 0.0f, 0.0f, 0.0f, // padding to last_pass_tile
    18.0f, 19.0f, // 2 weights, channel_subtile (2)
    32.0f, 33.0f,
    0.0f, 0.0f,  // padding to last_pass_tile
    20.0f, 0.0f, // 2 weights, 1 remainder channel, 1 padding up to channel_subtile
    34.0f, 0.0f,
    0.0f, 0.0f,  // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_MULTIPASS_DWCONV_HWG_W, one_middle_pass_channel_subtile_rounded) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 7;
  const size_t cr = 4;
  const size_t channel_subtile = 2;
  // c rounded to channel_subtile is 8, so we will have 2 channel_tile loops for first and middle pass.

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4, 5, 6]
  std::vector<float> k(c * h * w);  // k = [7, 8, 9, 10, 11, 12, 13,
                                    //      14, 15, 16, 17, 18, 19, 20,
                                    //      21, 22, 23, 24, 25, 26, 27,
                                    //      28, 29, 30, 31, 32, 33, 34]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f32_dwconv_multipass_hwg_w(
      first_pass_tile,
      middle_pass_tile,
      last_pass_tile,
      h,
      w,
      c,
      cr,
      channel_subtile,
      channel_subtile,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass.
    0.0f, 1.0f, 2.0f, 3.0f, // bias, 4 channels
    7.0f, 8.0f, 9.0f, 10.0f, // 1 weight, 4 channels first, then columns
    4.0f, 5.0f, 6.0f, 0.0f, // bias, 3 remainder channels, 1 padding up to channel_subtile
    11.0f, 12.0f, 13.0f, 0.0f, // 1 weight, 3 remainder channel, 1 padding up to channel_subtile
    // 1 middle pass.
    21.0f, 22.0f, 23.0f, 24.0f, // 2 weights, 4 channels first
    14.0f, 15.0f, 16.0f, 17.0f,
    25.0f, 26.0f, 27.0f, 0.0f, // 3 remainder channels first, 1 padding up to channel_tile
    18.0f, 19.0f, 20.0f, 0.0f,
    // Last pass.
    28.0f, 29.0f, 30.0f, 31.0f, // last pass, 1 weight, 4 channels first.
    0.0f, 0.0f, 0.0f, 0.0f, // padding to last_pass_tile
    32.0f, 33.0f, // last pass, 1 weight, channel_subtile (2)
    0.0f, 0.0f,  // padding to last_pass_tile
    34.0f, 0.0f, // last pass, 1 remainder channel, 1 padding up to channel_subtile
    0.0f, 0.0f,  // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once_channel_subtile_rounded_to_channel_round) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 6;
  const size_t cr = 8;
  const size_t channel_subtile = 4;
  const size_t channel_round = 2;
  // c rounded to channel_round is 6, so we will have 0 channel_tile and 2 channel_subtile loops
  // for first and middle pass.

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4, 5]
  std::vector<float> k(c * h * w);  // k = [6, 7, 8, 9, 10, 11,
                                    //      12, 13, 14, 15, 16, 17,
                                    //      18, 19, 20, 21, 22, 23,
                                    //      24, 25, 26, 27, 28, 29,]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(
    multipass_weights_count(
      h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile, channel_round));

  xnn_pack_f32_dwconv_multipass_hwg_w(
      first_pass_tile,
      middle_pass_tile,
      last_pass_tile,
      h,
      w,
      c,
      cr,
      channel_subtile,
      channel_round,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass.
    0.0f, 1.0f, 2.0f, 3.0f, // bias, 4 channels
    6.0f, 7.0f, 8.0f, 9.0f, // 2 weights, 4 channels first, then columns
    18.0f, 19.0f, 20.0f, 21.0f,
    4.0f, 5.0f, 0.0f, 0.0f, // bias, 2 remainder channels, 1 padding up to channel_subtile
    10.0f, 11.0f, 0.0f, 0.0f, // 2 weights, 2 remainder channel, 1 padding up to channel_subtile
    22.0f, 23.0f, 0.0f, 0.0f,
    // No middle pass.
    12.0f, 13.0f, 14.0f, 15.0f, // last pass, 2 weights, 4 channels first.
    24.0f, 25.0f, 26.0f, 27.0f,
    0.0f, 0.0f, 0.0f, 0.0f, // padding to last_pass_tile
    16.0f, 17.0f, 0.0f, 0.0f, // 2 weights, channel_subtile (4)
    28.0f, 29.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_TO_F16_DWCONV_GHW_W, primary_tile_eq_kernel_size) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1]
  std::vector<float> k(c * h * w);  // k = [2, 3, 4, 5, 6, 7]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_to_f16_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected_float = {
    // bias first
    0.0f, 1.0f,
    // then weights, channels first
    2.0f, 5.0f,
    3.0f, 6.0f,
    4.0f, 7.0f,
  };
  std::vector<uint16_t> expected(expected_float.size());
  std::transform(expected_float.begin(), expected_float.end(), expected.begin(),
                 [](float f) { return fp16_ieee_from_fp32_value(f); });
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_TO_F16_DWCONV_GHW_W, primary_tile_eq_kernel_size_channels_gt_cr) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4]
  std::vector<float> k(c * h * w);  // k = [
                                      //   5, 6, 7,
                                      //   8, 9, 10,
                                      //   11, 12, 13,
                                      //   14, 15, 16,
                                      //   17, 18, 19]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_to_f16_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected_float = {
    // cr blocks
    // bias first (cr == 2 of them)
    0.0f, 1.0f,
    // then weights, channels first
    5.0f, 8.0f, 6.0f, 9.0f, 7.0f, 10.0f,
    // bias again
    2.0f, 3.0f,
    // then weights, channels first
    11.0f, 14.0f, 12.0f, 15.0f, 13.0f, 16.0f,
    // bias again
    4.0f, 0.0f,
    // then weights, channels first
    17.0f, 0.0f, 18.0f, 0.0f, 19.0f, 0.0f,
  };
  std::vector<uint16_t> expected(expected_float.size());
  std::transform(expected_float.begin(), expected_float.end(), expected.begin(),
                 [](float f) { return fp16_ieee_from_fp32_value(f); });
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_TO_F16_DWCONV_GHW_W, primary_tile_gt_kernel_size) {
  const size_t primary_tile = 9;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1]
  std::vector<float> k(c * h * w);  // k = [
                                      //   2, 3,
                                      //   4, 5,
                                      //   6, 7,
                                      //   8, 9]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_to_f16_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected_float = {
    // bias first (cr == 2 of them)
    0.0f, 1.0f,
    // then weights, channels first
    2.0f, 6.0f,
    // go down the columns first
    4.0f, 8.0f, 3.0f, 7.0f, 5.0f, 9.0f,
    // followed by 10 zeros to make up the difference with primary_tile
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  };
  std::vector<uint16_t> expected(expected_float.size());
  std::transform(expected_float.begin(), expected_float.end(), expected.begin(),
                 [](float f) { return fp16_ieee_from_fp32_value(f); });
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_TO_F16_DWCONV_GHW_W, primary_tile_gt_kernel_size_channels_gt_cr) {
  const size_t primary_tile = 9;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4]
  std::vector<float> k(c * h * w);  // k = [
                                      //   5, 6,
                                      //   7, 8,
                                      //   9, 10,
                                      //   11, 12,
                                      //   13, 14,
                                      //   15, 16,
                                      //   17, 18,
                                      //   19, 20,
                                      //   21, 22,
                                      //   23, 24]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_to_f16_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected_float = {
    // bias first (cr == 2 of them)
    0.0f, 1.0f,
    // then weights, channels first
    5.0f, 9.0f,
    // go down the columns first
    7.0f, 11.0f,
    6.0f, 10.0f,
    8.0f, 12.0f,
    // followed by 10 zeros to make up the difference with primary_tile
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    // bias first (cr == 2 of them)
    2.0f, 3.0f,
    // then weights, channels first
    13.0f, 17.0f, 15.0f, 19.0f, 14.0f, 18.0f, 16.0f, 20.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    // bias
    4.0f, 0.0f,
    // weights
    21.0f, 0.0f, 23.0f, 0.0f, 22.0f, 0.0f, 24.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  };
  std::vector<uint16_t> expected(expected_float.size());
  std::transform(expected_float.begin(), expected_float.end(), expected.begin(),
                 [](float f) { return fp16_ieee_from_fp32_value(f); });
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_TO_F16_DWCONV_HWG_W, primary_tile_eq_kernel_size) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1]
  std::vector<float> k(c * h * w);  // k = [2, 3, 4, 5, 6, 7]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_to_f16_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected_float = {
    // bias first
    0.0f, 1.0f,
    // then weights, channels first
    2.0f, 3.0f,
    4.0f, 5.0f,
    6.0f, 7.0f,
  };
  std::vector<uint16_t> expected(expected_float.size());
  std::transform(expected_float.begin(), expected_float.end(), expected.begin(),
                 [](float f) { return fp16_ieee_from_fp32_value(f); });
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_TO_F16_DWCONV_HWG_W, primary_tile_eq_kernel_size_channels_gt_cr) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4]
  std::vector<float> k(c * h * w);  // k = [
                                      //   5, 6, 7, 8, 9,
                                      //   10, 11, 12, 13, 14,
                                      //   15, 16, 17, 18, 19]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_to_f16_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected_float = {
    // cr blocks
    // bias first (cr == 2 of them)
    0.0f, 1.0f,
    // then weights, channels first
    5.0f, 6.0f, 10.0f, 11.0f, 15.0f, 16.0f,
    // bias again
    2.0f, 3.0f,
    // then weights, channels first
    7.0f, 8.0f, 12.0f, 13.0f, 17.0f, 18.0f,
    // bias again
    4.0f, 0.0f,
    // then weights, channels first
    9.0f, 0.0f, 14.0f, 0.0f, 19.0f, 0.0f,
  };
  std::vector<uint16_t> expected(expected_float.size());
  std::transform(expected_float.begin(), expected_float.end(), expected.begin(),
                 [](float f) { return fp16_ieee_from_fp32_value(f); });
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_TO_F16_DWCONV_HWG_W, primary_tile_gt_kernel_size) {
  const size_t primary_tile = 9;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1]
  std::vector<float> k(c * h * w);  // k = [
                                      //   2, 3,
                                      //   4, 5,
                                      //   6, 7,
                                      //   8, 9]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_to_f16_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected_float = {
    // bias first (cr == 2 of them)
    0.0f, 1.0f,
    // then weights, channels first
    2.0f, 3.0f,
    // go down the columns first
    6.0f, 7.0f, 4.0f, 5.0f, 8.0f, 9.0f,
    // followed by 10 zeros to make up the difference with primary_tile
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  };
  std::vector<uint16_t> expected(expected_float.size());
  std::transform(expected_float.begin(), expected_float.end(), expected.begin(),
                 [](float f) { return fp16_ieee_from_fp32_value(f); });
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_TO_F16_DWCONV_HWG_W, primary_tile_gt_kernel_size_channels_gt_cr) {
  const size_t primary_tile = 9;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<float> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4]
  std::vector<float> k(c * h * w);  // k = [
                                      //   5, 6, 7, 8, 9,
                                      //   10, 11, 12, 13, 14,
                                      //   15, 16, 17, 18, 19,
                                      //   20, 21, 22, 23, 24]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_to_f16_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<float> expected_float = {
    // bias first (cr == 2 of them)
    0.0f, 1.0f,
    // then weights, channels first
    5.0f, 6.0f,
    // go down the columns first
    15.0f, 16.0f,
    10.0f, 11.0f,
    20.0f, 21.0f,
    // followed by 10 zeros to make up the difference with primary_tile
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    // bias first (cr == 2 of them)
    2.0f, 3.0f,
    // then weights, channels first
    7.0f, 8.0f, 17.0f, 18.0f, 12.0f, 13.0f, 22.0f, 23.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    // bias
    4.0f, 0.0f,
    // weights
    9.0f, 0.0f, 19.0f, 0.0f, 14.0f, 0.0f, 24.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  };
  std::vector<uint16_t> expected(expected_float.size());
  std::transform(expected_float.begin(), expected_float.end(), expected.begin(),
                 [](float f) { return fp16_ieee_from_fp32_value(f); });
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_TO_F16_CHW_DWCONV_HWG_W, primary_tile_eq_kernel_size) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;

  std::vector<float> b(1);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0]
  std::vector<float> k(h * w);  // k = [1, 2, 3]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<uint16_t> packed_weights(primary_tile + 1);

  xnn_pack_f32_to_f16_chw_dwconv_hwg_w(
      primary_tile,  // kernel size
      1, // groups
      k.data(),
      b.data(),
      packed_weights.data(),
      nullptr);

  const std::vector<float> expected_float = {
    // bias first
    0.0f,
    // then weights
    1.0f,
    2.0f,
    3.0f,
  };
  std::vector<uint16_t> expected(expected_float.size());
  std::transform(expected_float.begin(), expected_float.end(), expected.begin(),
                 [](float f) { return fp16_ieee_from_fp32_value(f); });
  EXPECT_EQ(expected, packed_weights);
}


TEST(PACK_F32_TO_F16_CHW_DWCONV_HWG_W, groups_gt_1) {
  const size_t primary_tile = 3;
  const size_t g = 3;
  const size_t h = 3;
  const size_t w = 1;

  std::vector<float> b(g);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2]
  std::vector<float> k(g * h * w);  // k = [3, 4, 5, 6, 7, 8, 9, 10, 11 ]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<uint16_t> packed_weights(g + g * h * w);

  xnn_pack_f32_to_f16_chw_dwconv_hwg_w(
      primary_tile,  // kernel size
      g, // groups
      k.data(),
      b.data(),
      packed_weights.data(),
      nullptr);

  const std::vector<float> expected_float = {
    // bias first
    0.0f,
    // then weights
    3.0f,
    6.0f,
    9.0f,

    // 2nd group, bias first
    1.0f,
    // then weights
    4.0f,
    7.0f,
    10.0f,

    // 3rd group, bias first
    2.0f,
    // then weights
    5.0f,
    8.0f,
    11.0f,
  };
  std::vector<uint16_t> expected(expected_float.size());
  std::transform(expected_float.begin(), expected_float.end(), expected.begin(),
                 [](float f) { return fp16_ieee_from_fp32_value(f); });
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_CHW_DWCONV_HWG_W, primary_tile_eq_kernel_size) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;

  std::vector<uint16_t> b(1);
  std::iota(b.begin(), b.end(), 0);  // b = [0]
  std::vector<uint16_t> k(h * w);  // k = [1, 2, 3]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(primary_tile + 1);

  xnn_pack_f16_chw_dwconv_hwg_w(
      primary_tile,  // kernel size
      1, // groups
      k.data(),
      b.data(),
      packed_weights.data(),
      nullptr);

  const std::vector<uint16_t> expected = {
    // bias first
    0,
    // then weights
    1,
    2,
    3,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_CHW_DWCONV_HWG_W, groups_gt_1) {
  const size_t primary_tile = 3;
  const size_t g = 3;
  const size_t h = 3;
  const size_t w = 1;

  std::vector<uint16_t> b(g);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2]
  std::vector<uint16_t> k(g * h * w);  // k = [3, 4, 5, 6, 7, 8, 9, 10, 11 ]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(g + g * h * w);

  xnn_pack_f16_chw_dwconv_hwg_w(
      primary_tile,  // kernel size
      g, // groups
      k.data(),
      b.data(),
      packed_weights.data(),
      nullptr);

  const std::vector<uint16_t> expected = {
    // bias first
    0,
    // then weights
    3,
    6,
    9,

    // 2nd group, bias first
    1,
    // then weights
    4,
    7,
    10,

    // 3rd group, bias first
    2,
    // then weights
    5,
    8,
    11,
  };
  EXPECT_EQ(expected, packed_weights);
}


TEST(PACK_F16_CHW_DWCONV_GHW_W, primary_tile_eq_kernel_size) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;

  std::vector<uint16_t> b(1);
  std::iota(b.begin(), b.end(), 0);  // b = [0]
  std::vector<uint16_t> k(h * w);  // k = [1, 2, 3]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(primary_tile + 1);

  xnn_pack_f16_chw_dwconv_ghw_w(
      primary_tile,  // kernel size
      1, // groups
      k.data(),
      b.data(),
      packed_weights.data(),
      nullptr);

  const std::vector<uint16_t> expected = {
    // bias first
    0,
    // then weights
    1,
    2,
    3,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_CHW_DWCONV_GHW_W, groups_gt_1) {
  const size_t primary_tile = 3;
  const size_t g = 3;
  const size_t h = 3;
  const size_t w = 1;

  std::vector<uint16_t> b(g);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2]
  std::vector<uint16_t> k(g * h * w);  // k = [3, 4, 5, 6, 7, 8, 9, 10, 11 ]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(g + g * h * w);

  xnn_pack_f16_chw_dwconv_ghw_w(
      primary_tile,  // kernel size
      g, // groups
      k.data(),
      b.data(),
      packed_weights.data(),
      nullptr);

  const std::vector<uint16_t> expected = {
    // bias first
    0,
    // then weights
    3,
    4,
    5,

    // 2nd group, bias first
    1,
    // then weights
    6,
    7,
    8,

    // 3rd group, bias first
    2,
    // then weights
    9,
    10,
    11,
  };
  EXPECT_EQ(expected, packed_weights);
}


TEST(PACK_F32_DWCONV_OKI_W, primary_tile_eq_kernel_size) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;

  std::vector<float> b(3);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0]
  std::vector<float> k(h * w);  // k = [3, 4, 5]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<float> packed_weights(primary_tile * 2);

  xnn_pack_f32_dconv_oki_w(
      h, // nc
      1, // kc
      w, // nr
      1, // kh
      1, // kw
      k.data(),
      b.data(),
      packed_weights.data(),
      nullptr);

  const std::vector<float> expected = {
    // bias first
    0.0f,
    // then weight
    3.0f,
    // bias first
    1.0f,
    // then weight
    4.0f,
    // bias first
    2.0f,
    // then weight
    5.0f,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_TO_F16_DWCONV_OKI_W, primary_tile_eq_kernel_size) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;

  std::vector<float> b(3);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0]
  std::vector<float> k(h * w);  // k = [3, 4, 5]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  std::vector<uint16_t> packed_weights(primary_tile * 2);

  xnn_pack_f32_to_f16_dconv_oki_w(
      h, // nc
      1, // kc
      w, // nr
      1, // kh
      1, // kw
      k.data(),
      b.data(),
      packed_weights.data(),
      nullptr);

  const std::vector<float> expected_float = {
    // bias first
    0.0f,
    // then weight
    3.0f,
    // bias first
    1.0f,
    // then weight
    4.0f,
    // bias first
    2.0f,
    // then weight
    5.0f,
  };
  std::vector<uint16_t> expected(expected_float.size());
  std::transform(expected_float.begin(), expected_float.end(), expected.begin(),
                 [](float f) { return fp16_ieee_from_fp32_value(f); });
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F32_TO_F16_DWCONV_OKI_W, null_bias) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;

  std::vector<float> k(h * w);  // k = [3, 4, 5]
  std::iota(k.begin(), k.end(), 3);
  std::vector<uint16_t> packed_weights(primary_tile * 2);

  xnn_pack_f32_to_f16_dconv_oki_w(
      h, // nc
      1, // kc
      w, // nr
      1, // kh
      1, // kw
      k.data(),
      nullptr, // bias,
      packed_weights.data(),
      nullptr);

  const std::vector<float> expected_float = {
    // bias first
    0.0f,
    // then weight
    3.0f,
    // bias first
    0.0f,
    // then weight
    4.0f,
    // bias first
    0.0f,
    // then weight
    5.0f,
  };
  std::vector<uint16_t> expected(expected_float.size());
  std::transform(expected_float.begin(), expected_float.end(), expected.begin(),
                 [](float f) { return fp16_ieee_from_fp32_value(f); });
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_DWCONV_OKI_W, primary_tile_eq_kernel_size) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;

  std::vector<uint16_t> b(3);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2]
  std::vector<uint16_t> k(h * w);  // k = [3, 4, 5]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(primary_tile * 2);

  xnn_pack_f16_dconv_oki_w(
      h, // nc
      1, // kc
      w, // nr
      1, // kh
      1, // kw
      k.data(),
      b.data(),
      packed_weights.data(),
      nullptr);

  const std::vector<uint16_t> expected = {
    // bias first
    0,
    // then weight
    3,
    // bias first
    1,
    // then weight
    4,
    // bias first
    2,
    // then weight
    5,
  };
  EXPECT_EQ(expected, packed_weights);
}


