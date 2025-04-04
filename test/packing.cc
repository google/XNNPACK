// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// clang-format off

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microkernel-utils.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/pack.h"

// QD8-F32-QC4W GEMM packing tests.

namespace {

using testing::ElementsAreArray;
using testing::Matcher;
using testing::_;

TEST(PACK_QD8_F32_QC4W_GEMM_GOI_W, kr_eq_4) {
  size_t g = 1;
  size_t nc = 1;
  size_t kc = 16;
  size_t nr = 1;
  size_t kr = 4;
  size_t sr = 1;

  std::vector<int32_t> b(g * nc);
  std::iota(b.begin(), b.end(), 0);
  std::vector<uint8_t> k(g * nc * kc / 2);
  k[0] = 0x98; k[1] = 0xBA; k[2] = 0xDC; k[3] = 0xFE; k[4] = 0x10; k[5] = 0x32; k[6] = 0x54; k[7] = 0x76;
  xnnpack::Buffer<uint8_t> packed_weights(g * round_up(nc, nr) * (sizeof(float) + round_up_po2(kc, kr * sr) / 2));
  auto a = xnn_qs8_qc4w_packing_params{ 0, 0x8 };
  xnn_pack_qs8_qc4w_gemm_goi_w(g, nc, kc, nr, kr, sr,
    k.data(), b.data(), /*scale=*/nullptr, packed_weights.data(), /*extra_bytes=*/0, /*params=*/&a);

  const std::vector<Matcher<uint8_t>> expected = {
    // 1 bias.
    0x00, 0x00, 0x00, 0x00,
    0x40, 0x51, 0x62, 0x73, 0xC8, 0xD9, 0xEA, 0xFB,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
}

TEST(PACK_QD8_F32_QC4W_GEMM_GIO_W, kr_eq_4) {
  size_t g = 1;
  size_t nc = 1;
  size_t kc = 16;
  size_t nr = 1;
  size_t kr = 4;
  size_t sr = 1;

  std::vector<int32_t> b(g * nc);
  std::iota(b.begin(), b.end(), 0);
  std::vector<uint8_t> k(g * nc * kc);  // round up rows of nc to bytes

  k[0] = 0x88;
  k[1] = 0x89;
  k[2] = 0x8A;
  k[3] = 0x8B;
  k[4] = 0x8C;
  k[5] = 0x8D;
  k[6] = 0x8E;
  k[7] = 0x8F;
  k[8] = 0x80;
  k[9] = 0x81;
  k[10] = 0x82;
  k[11] = 0x83;
  k[12] = 0x84;
  k[13] = 0x85;
  k[14] = 0x86;
  k[15] = 0x87;
  xnnpack::Buffer<uint8_t> packed_weights(g * round_up(nc, nr) * (sizeof(float) + round_up_po2(kc, kr * sr) / 2));
  auto a = xnn_qs8_qc4w_packing_params{ 0, 0x8 };
  xnn_pack_qs8_qc4w_gemm_gio_w(g, nc, kc, nr, kr, sr,/*k_stride=*/round_up_po2(nc, 2),
    k.data(), b.data(), /*scale=*/nullptr, packed_weights.data(), /*extra_bytes=*/0, /*params=*/&a);

  const std::vector<Matcher<uint8_t>> expected = {
    // 1 bias.
    0x00, 0x00, 0x00, 0x00,
    0x40, 0x51, 0x62, 0x73, 0xC8, 0xD9, 0xEA, 0xFB,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
}

TEST(PACK_QD8_F32_QC4W_GEMM_GOI_W, kr_eq_4_nr_eq_2) {
  size_t g = 1;
  size_t nc = 2;
  size_t kc = 8;
  size_t nr = 2;
  size_t kr = 4;
  size_t sr = 1;

  std::vector<int32_t> b(g * nc);
  std::iota(b.begin(), b.end(), 0);
  std::vector<uint8_t> k(g * nc * kc / 2);
  k[0] = 0x98; k[1] = 0xBA; k[2] = 0xDC; k[3] = 0xFE;
  k[4] = 0x10; k[5] = 0x32; k[6] = 0x54; k[7] = 0x76;
  xnnpack::Buffer<uint8_t> packed_weights(g * round_up(nc, nr) * (sizeof(float) + round_up_po2(kc, kr * sr) / 2));
  auto a = xnn_qs8_qc4w_packing_params{ 0, 0x8 };
  xnn_pack_qs8_qc4w_gemm_goi_w(g, nc, kc, nr, kr, sr,
    k.data(), b.data(), /*scale=*/nullptr, packed_weights.data(), /*extra_bytes=*/0, /*params=*/&a);

  const std::vector<Matcher<uint8_t>> expected = {
    // 2 bias.
    0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00,
    0x40, 0x51, 0x62, 0x73,
    0xC8, 0xD9, 0xEA, 0xFB,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
}

TEST(PACK_QD8_F32_QC4UW_GEMM_GOI_W, kr_eq_4_nr_eq_2) {
  size_t g = 1;
  size_t nc = 2;
  size_t kc = 8;
  size_t nr = 2;
  size_t kr = 4;
  size_t sr = 1;

  std::vector<int32_t> b(g * nc);
  std::iota(b.begin(), b.end(), 0);
  std::vector<uint8_t> k(g * nc * kc / 2);
  k[0] = 0x98; k[1] = 0xBA; k[2] = 0xDC; k[3] = 0xFE;
  k[4] = 0x10; k[5] = 0x32; k[6] = 0x54; k[7] = 0x76;
  xnnpack::Buffer<uint8_t> packed_weights(g * round_up(nc, nr) * (sizeof(float) + round_up_po2(kc, kr * sr) / 2));
  auto a = xnn_qs8_qc4w_packing_params{ 0, 0x8 };
  xnn_pack_qs8_qc4uw_gemm_goi_w(g, nc, kc, nr, kr, sr,
    k.data(), b.data(), /*scale=*/nullptr, packed_weights.data(), /*extra_bytes=*/0, /*params=*/&a);

  const std::vector<Matcher<uint8_t>> expected = {
    // 2 bias.
    0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00,
    0xC8, 0xD9, 0xEA, 0xFB,
    0x40, 0x51, 0x62, 0x73,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
}

TEST(PACK_QD8_F32_QC4W_GEMM_GIO_W, kr_eq_4_nr_eq_2) {
  size_t g = 1;
  size_t nc = 2;
  size_t kc = 8;
  size_t nr = 2;
  size_t kr = 4;
  size_t sr = 1;

  std::vector<int32_t> b(g * nc);
  std::iota(b.begin(), b.end(), 0);
  std::vector<uint8_t> k(g * nc * kc / 2);
  k[0] = 0x08;
  k[1] = 0x19;
  k[2] = 0x2A;
  k[3] = 0x3B;
  k[4] = 0x4C;
  k[5] = 0x5D;
  k[6] = 0x6E;
  k[7] = 0x7F;
  xnnpack::Buffer<uint8_t> packed_weights(g * round_up(nc, nr) * (sizeof(float) + round_up_po2(kc, kr * sr) / 2));
  auto a = xnn_qs8_qc4w_packing_params{ 0, 0x8 };
  xnn_pack_qs8_qc4w_gemm_gio_w(g, nc, kc, nr, kr, sr, /*k_stride=*/nc,
    k.data(), b.data(), /*scale=*/nullptr, packed_weights.data(), /*extra_bytes=*/0, /*params=*/&a);

  const std::vector<Matcher<uint8_t>> expected = {
    // 2 bias.
    0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00,
    0x40, 0x51, 0x62, 0x73,
    0xC8, 0xD9, 0xEA, 0xFB,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
}

TEST(PACK_QD8_F32_QB4W_GEMM_GOI_W, bl_eq_kc) {
  size_t g = 1;
  size_t nc = 1;
  size_t kc = 16;
  size_t nr = 1;
  size_t kr = 4;
  size_t sr = 1;
  size_t bl = kc;
  size_t k_num_blocks = round_up_po2(kc, kr) / bl;

  std::vector<uint8_t> k(g * nc * kc / 2);
  k[0] = 0x98; k[1] = 0xBA; k[2] = 0xDC; k[3] = 0xFE; k[4] = 0x10; k[5] = 0x32; k[6] = 0x54; k[7] = 0x76;
  size_t extra_bytes_bl = sizeof(uint16_t);
  size_t extra_bytes_n = sizeof(float);
  std::vector<uint8_t> packed_weights(g * round_up(nc, nr) * (sizeof(float) + round_up_po2(kc, kr * sr) / 2)
    + k_num_blocks * round_up(nc, nr) * extra_bytes_bl + round_up(nc, nr) * extra_bytes_n);
  std::vector<xnn_bfloat16> scale(nc * k_num_blocks, 853.6010);
  auto a = xnn_qs8_qc4w_packing_params{ -1, 0x8 };

  xnn_pack_qs8_qb4w_gemm_goi_w(g, nc, kc, nr, kr, sr, bl,
    k.data(), /*bias=*/nullptr, /*scale=*/scale.data(), packed_weights.data(), extra_bytes_bl, extra_bytes_n, /*params=*/&a);

  size_t k_stride = round_up_po2(kc, kr * sr * 2 /* planes */);

  // If filter is 4-bit, half k_stride (since we will scale k_stride by log2_filter_element_size, and we pass 0 for qc4).
  k_stride = round_up_po2(k_stride, 2) >> 1;

  size_t k_bytes = sizeof(int8_t) * k_stride * nr;
  size_t bias_bytes = sizeof(float) * nr;
  size_t ksum_bytes = sizeof(float) * nr;
  size_t block_bytes = sizeof(uint16_t) * k_num_blocks * nr;

  size_t start_offset = ksum_bytes + k_bytes / k_num_blocks;
  size_t stride = ksum_bytes + k_bytes + block_bytes + bias_bytes;
  size_t block_stride = (bl * nr) / 2 + (sizeof(uint16_t) * nr);

  // Fill in scales.
  xnn_init_blockwise_scale_bf16_params(
    /*channels=*/ nc,
    /*channels_tile=*/ nr,
    /*stride=*/ stride,
    /*num_blocks=*/k_num_blocks,
    /*block_stride=*/block_stride,
    /*scale=*/ scale.data(),
    /*packed_weight=*/ packed_weights.data() + start_offset);

  const std::vector<Matcher<uint8_t>> expected = {
    // kscaledsum
    // scaled row sum converted to bf16.
    0x00, 0x00, 0xd5, 0xc5, // -1 * 853.6010 (bf16) * sum(-8..+7) = 0xc5d50000

    // weights.
    0x40, 0x51, 0x62, 0x73, // kr0 | kr1
    0xC8, 0xD9, 0xEA, 0xFB, // kr2 | kr3
    // extra bytes bl
    0x55, 0x42,

    // extra bytes n - no bias for this test
    0, 0, 0, 0
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
}

TEST(PACK_QD8_F32_QB4W_GEMM_GOI_W, nc_gt_1) {
  size_t g = 1;
  size_t nc = 2;
  size_t kc = 8;
  size_t nr = 1;
  size_t kr = 4;
  size_t sr = 1;
  size_t bl = 8;
  size_t k_num_blocks = kc / bl;

  std::vector<int32_t> b(g * nc);
  std::vector<uint8_t> k(g * nc * kc / 2);
  k[0] = 0xA8; k[1] = 0xEC; k[2] = 0x20; k[3] = 0x64; k[4] = 0xB9; k[5] = 0xFD; k[6] = 0x31; k[7] = 0x75;

  // Example:
  // ------------------
  // 8 A C E | 0 2 4 6 |
  // ------------------
  // 9 B D F | 1 3 5 7 |
  // ------------------

  size_t extra_bytes_n = sizeof(float);
  size_t extra_bytes_bl = sizeof(uint16_t);
  std::vector<uint8_t> packed_weights(g * round_up(nc, nr) * (sizeof(float) + round_up_po2(kc, kr * sr) / 2)
    + k_num_blocks * round_up(nc, nr) * extra_bytes_bl + round_up(nc, nr) * extra_bytes_n);
  std::vector<xnn_bfloat16> scale(nc * k_num_blocks, 853.6010);

  auto a = xnn_qs8_qc4w_packing_params{ -1, 0x8 };
  xnn_pack_qs8_qb4w_gemm_goi_w(g, nc, kc, nr, kr, sr, bl,
    k.data(), nullptr, /*scale=*/scale.data(), packed_weights.data(), extra_bytes_bl, extra_bytes_n, /*params=*/&a);

    size_t k_stride = round_up_po2(kc, kr * sr * 2 /* planes */);

  k_stride = round_up_po2(k_stride, 2) >> 1;

  size_t k_bytes = sizeof(int8_t) * k_stride * nr;
  size_t bias_bytes = sizeof(float) * nr;
  size_t ksum_bytes = sizeof(float) * nr;
  size_t block_bytes = sizeof(uint16_t) * k_num_blocks * nr;

  size_t start_offset = ksum_bytes + k_bytes / k_num_blocks;
  size_t stride = ksum_bytes + k_bytes + block_bytes + bias_bytes;
  size_t block_stride = (bl * nr) / 2 + (sizeof(uint16_t) * nr);

  xnn_init_blockwise_scale_bf16_params(
    /*channels=*/nc,
    /*channels_tile=*/nr,
    /*stride=*/stride,
    /*num_blocks=*/k_num_blocks,
    /*block_stride=*/block_stride,
    /*scale=*/scale.data(),
    /*packed_w=*/packed_weights.data() + start_offset);

  const std::vector<Matcher<uint8_t>> expected = {
    // kscaledsum
    // scaled row sum converted to bf16.
    0x00, 0x00, 0xd5, 0xc5, // -1 * 853.6010 (bf16) * (sum(-8..+7) = 0xc5d50000

    // weights
    0x80, 0xA2, 0xC4, 0xE6, // kr0 | kr1
    // extra bytes bl
    0x55, 0x42,

    // extra bytes n
    0, 0, 0, 0,

    // kscaledsum
    0x00, 0x00, 0x00, 0x00, // -1 * 853.6010 * (sum(-7, -5, -3, ..., 7)) = 0

    // weights
    0x91, 0xB3, 0xD5, 0xF7, // kr2 | kr3
    // extra bytes bl
    0x55, 0x42,

    // extra bytes n
    0, 0, 0, 0
  };

  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
}

TEST(PACK_QD8_F32_QB4W_GEMM_GOI_W, bl_lt_kc) {
  size_t g = 1;
  size_t nc = 1;
  size_t kc = 16;
  size_t nr = 1;
  size_t kr = 4;
  size_t sr = 1;
  size_t bl = 8;
  size_t k_num_blocks = kc / bl;

  std::vector<uint8_t> k(g * nc * kc / 2);
  k[0] = 0x98; k[1] = 0xBA; k[2] = 0xDC; k[3] = 0xFE; k[4] = 0x10; k[5] = 0x32; k[6] = 0x54; k[7] = 0x76;
  size_t extra_bytes_n = sizeof(float);
  size_t extra_bytes_bl = sizeof(uint16_t);
  std::vector<uint8_t> packed_weights(g * round_up(nc, nr) * (sizeof(float) + round_up_po2(kc, kr * sr) / 2)
    + k_num_blocks * round_up(nc, nr) * extra_bytes_bl + round_up(nc, nr) * extra_bytes_n);
  std::vector<xnn_bfloat16> scale(nc * k_num_blocks, 853.6010);


  auto a = xnn_qs8_qc4w_packing_params{ -1, 0x8 };
  xnn_pack_qs8_qb4w_gemm_goi_w(g, nc, kc, nr, kr, sr, bl,
    k.data(), /*bias=*/nullptr, /*scale=*/scale.data(), packed_weights.data(), extra_bytes_bl, extra_bytes_n, /*params=*/&a);

    size_t k_stride = round_up_po2(kc, kr * sr * 2 /* planes */);

  k_stride = round_up_po2(k_stride, 2) >> 1;

  size_t k_bytes = sizeof(int8_t) * k_stride * nr;
  size_t bias_bytes = sizeof(float) * nr;
  size_t ksum_bytes = sizeof(float) * nr;
  size_t block_bytes = sizeof(uint16_t) * k_num_blocks * nr;

  size_t start_offset = ksum_bytes + k_bytes / k_num_blocks;
  size_t stride = ksum_bytes + k_bytes + block_bytes + bias_bytes;
  size_t block_stride = (bl * nr) / 2 + (sizeof(uint16_t) * nr);

  // Fill in scales.
  xnn_init_blockwise_scale_bf16_params(
    /*channels=*/ nc,
    /*channels_tile=*/ nr,
    /*stride=*/ stride,
    /*num_blocks=*/k_num_blocks,
    /*block_stride=*/block_stride,
    /*scale=*/ scale.data(),
    /*packed_weight=*/ packed_weights.data() + start_offset);


  const std::vector<Matcher<uint8_t>> expected = {
    // kscaledsum
    // scaled row sum converted to bf16.
    0x00, 0x00, 0xd5, 0xc5, // -1 * 853.6010 (bf16) * (sum(-8..+7) = 0xc5d50000

    // weights
    0x40, 0x51, 0x62, 0x73, // kr0 | kr1
    // extra bytes bl
    0x55, 0x42,

    // weights
    0xC8, 0xD9, 0xEA, 0xFB, // kr2 | kr3
    // extra bytes bl
    0x55, 0x42,

    // extra bytes n - no bias for this test
    0, 0, 0, 0
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
}

TEST(PACK_QD8_F32_QB4W_GEMM_GIO_W, bl_eq_kc) {
  size_t g = 1;
  size_t nc = 2;
  size_t kc = 8;
  size_t nr = 1;
  size_t kr = 4;
  size_t sr = 1;
  size_t bl = 8;
  size_t k_num_blocks = kc / bl;

  std::vector<int32_t> b(g * nc);
  std::iota(b.begin(), b.end(), 0);
  std::vector<uint8_t> k(g * nc * kc / 2);
  k[0] = 0x98; k[1] = 0xBA; k[2] = 0xDC; k[3] = 0xFE; k[4] = 0x10; k[5] = 0x32; k[6] = 0x54; k[7] = 0x76;
  // Example:
  // |-----------|
  // |  8  |  9  |
  // |  A  |  B  |
  // |  C  |  D  |
  // |  E  |  F  |
  // |-----------|
  // |  0  |  1  |
  // |  2  |  3  |
  // |  4  |  5  |
  // |  6  |  7  |
  // |-----------|

  size_t extra_bytes_n = sizeof(float);
  size_t extra_bytes_bl = sizeof(uint16_t);
  std::vector<uint8_t> packed_weights(g * round_up(nc, nr) * (sizeof(float) + round_up_po2(kc, kr * sr) / 2)
    + k_num_blocks * round_up(nc, nr) * extra_bytes_bl + round_up(nc, nr) * extra_bytes_n);
  std::vector<xnn_bfloat16> scale(nc * k_num_blocks, 853.6010);


  auto a = xnn_qs8_qc4w_packing_params{ -1, 0x8 };

  size_t k_stride = round_up_po2(kc, kr * sr * 2 /* planes */);

  k_stride = round_up_po2(k_stride, 2) >> 1;

  xnn_pack_qs8_qb4w_gemm_gio_w(g, nc, kc, nr, kr, sr, /*k_stride =*/nc, bl,
    k.data(), nullptr, /*scale=*/scale.data(), packed_weights.data(), extra_bytes_bl, extra_bytes_n, /*params=*/&a);

  size_t k_bytes = sizeof(int8_t) * k_stride * nr;
  size_t bias_bytes = sizeof(float) * nr;
  size_t ksum_bytes = sizeof(float) * nr;
  size_t block_bytes = sizeof(uint16_t) * k_num_blocks * nr;

  size_t start_offset = ksum_bytes + k_bytes / k_num_blocks;
  size_t stride = ksum_bytes + k_bytes + block_bytes + bias_bytes;
  size_t block_stride = (bl * nr) / 2 + (sizeof(uint16_t) * nr);

  xnn_init_blockwise_scale_bf16_params(
    /*channels=*/nc,
    /*channels_tile=*/nr,
    /*stride=*/stride,
    /*num_blocks=*/k_num_blocks,
    /*block_stride=*/block_stride,
    /*scale=*/scale.data(),
    /*packed_w=*/packed_weights.data() + start_offset);

  const std::vector<Matcher<uint8_t>> expected = {
    // kscaledsum
    0x00, 0x00, 0xd5, 0xc5, // -1 * 853.6010 (bf16) * (sum(-8..+7) = 0xc5d50000

    // weights
    0x80, 0xA2, 0xC4, 0xE6, // kr0 | kr1
    // extra bytes bl
    0x55, 0x42,

    // extra bytes n
    0, 0, 0, 0,

    // kscaledsum
    // scaled row sum converted to bf16.
    0x00, 0x00, 0x00, 0x00, // -1 * 853.6010 * (sum(-7, -5, -3, ..., 7)) = 0

    // weights
    0x91, 0xB3, 0xD5, 0xF7, // kr2 | kr3
    // extra bytes bl
    0x55, 0x42,

    // extra bytes n
    0, 0, 0, 0
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
}

TEST(PACK_F32_GEMM_GIO_W, g_eq_1) {
  size_t g = 1;
  size_t nc = 2;
  size_t kc = 2;
  size_t nr = 1;
  size_t kr = 1;
  size_t sr = 1;

  std::vector<float> b(g * nc);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1]
  std::vector<float> k(g * nc * kc);
  std::iota(k.begin(), k.end(), static_cast<float>(b.size())); // k = [2, 3, 4, 5]
  xnnpack::Buffer<float> packed_weights(g * round_up(nc, nr) * (1 + round_up_po2(kc, kr * sr)));
  xnn_pack_f32_gemm_gio_w(g, nc, kc, nr, kr, sr, nc,
    k.data(), b.data(), /*scale=*/nullptr, packed_weights.data(), /*extra_bytes=*/0, /*params=*/nullptr);

  const std::vector<Matcher<float>> expected = {
    0.0f,
    2.0f, 4.f,
    1.0f,
    3.0f, 5.0f,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
}

TEST(PACK_F32_GEMM_GIO_W, g_eq_1_nr_gt_1_kr_gt_1) {
  size_t g = 1;
  size_t nc = 3;
  size_t kc = 3;
  size_t nr = 2;
  size_t kr = 2;
  size_t sr = 1;

  std::vector<float> b(g * nc);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2]
  std::vector<float> k(g * nc * kc);
  std::iota(k.begin(), k.end(), static_cast<float>(b.size())); // k = [3, 4, 5, 6, 7, 8, 9, 10, 11]
  xnnpack::Buffer<float> packed_weights(g * round_up(nc, nr) * (1 + round_up_po2(kc, kr * sr)));
  xnn_pack_f32_gemm_gio_w(g, nc, kc, nr, kr, sr, nc,
    k.data(), b.data(), /*scale=*/nullptr, packed_weights.data(), /*extra_bytes=*/0, /*params=*/nullptr);

  const std::vector<Matcher<float>> expected = {
    0.0f, 1.0f,
    3.0f, 6.0f, 4.0f, 7.0f,
    9.0f, 0.0f, 10.0f, 0.0f,

    2.0f, _,
    5.0f, 8.0f, _, _,
    11.0f, 0.0f, _, _,
  };
  EXPECT_THAT(packed_weights, testing::ElementsAreArray(expected));
}

TEST(PACK_F32_GEMM_GIO_W, g_gt_1) {
  size_t g = 3;
  size_t nc = 2;
  size_t kc = 2;
  size_t nr = 1;
  size_t kr = 1;
  size_t sr = 1;

  std::vector<float> b(g * nc);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4, 5]
  std::vector<float> k(g * nc * kc);
  std::iota(k.begin(), k.end(), static_cast<float>(b.size())); // k = [6,7,8,9,10,11,12,13,14,15,16,17]
  xnnpack::Buffer<float> packed_weights(g * round_up(nc, nr) * (1 + round_up_po2(kc, kr * sr)));
  xnn_pack_f32_gemm_gio_w(g, nc, kc, nr, kr, sr, nc,
    k.data(), b.data(), /*scale=*/nullptr, packed_weights.data(), /*extra_bytes=*/0, /*params=*/nullptr);

  const std::vector<Matcher<float>> expected = {
    0.0f,
    6.0f, 8.f,
    1.0f,
    7.0f, 9.0f,
    2.0f,
    10.0f, 12.0f,
    3.0f,
    11.0f, 13.0f,
    4.0f,
    14.0f, 16.0f,
    5.0f,
    15.0f, 17.0f,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
}

TEST(PACK_F32_GEMM_GIO_W, g_gt_1_nr_gt_1_kr_gt_1) {
  size_t g = 3;
  size_t nc = 3;
  size_t kc = 3;
  size_t nr = 2;
  size_t kr = 2;
  size_t sr = 1;

  std::vector<float> b(g * nc);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4, 5, 6, 7, 8]
  std::vector<float> k(g * nc * kc);
  std::iota(k.begin(), k.end(), static_cast<float>(b.size())); // k = [
                                                               //   9,10,11,12,13,14,15,16,17,
                                                               //   18,19,20,21,22,23,24,25,26,
                                                               //   27,28,29,30,31,32,33,34,35
                                                               // ]
  xnnpack::Buffer<float> packed_weights(g * round_up(nc, nr) * (1 + round_up_po2(kc, kr * sr)));
  xnn_pack_f32_gemm_gio_w(g, nc, kc, nr, kr, sr, nc,
    k.data(), b.data(), /*scale=*/nullptr, packed_weights.data(), /*extra_bytes=*/0, /*params=*/nullptr);

  const std::vector<Matcher<float>> expected = {
    // Group 1.
    0.0f, 1.0f,
    9.0f, 12.0f, 10.0f, 13.0f,
    15.0f, 0.0f, 16.0f, 0.0f,
    2.0f, _,
    11.0f, 14.0f, _, _,
    17.0f, 0.0f, _, _,
    // Group 2.
    3.0f, 4.0f,
    18.0f, 21.0f, 19.0f, 22.0f,
    24.0f, 0.0f, 25.0f, 0.0f,
    5.0f, _,
    20.0f, 23.0f, _, _,
    26.0f, 0.0f, _, _,
    // Group 3.
    6.0f, 7.0f,
    27.0f, 30.0f, 28.0f, 31.0f,
    33.0f, 0.0f, 34.0f, 0.0f,
    8.0f, _,
    29.0f, 32.0f, _, _,
    35.0f, 0.0f, _, _,
  };
  EXPECT_THAT(packed_weights, testing::ElementsAreArray(expected));
}

float packed_bf16(xnn_bfloat16 a, xnn_bfloat16 b) {
  union {
    float f;
    xnn_bfloat16 bf[2];
  } result;
  result.bf[0] = a;
  result.bf[1] = b;
  return result.f;
}

TEST(PACK_BF16_F32_GEMM_GIO_W, g_eq_1) {
  size_t g = 1;
  size_t nc = 2;
  size_t kc = 2;
  size_t nr = 1;
  size_t kr = 1;
  size_t sr = 1;

  std::vector<float> b(g * nc);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1]
  std::vector<xnn_bfloat16> k(g * nc * kc);
  std::iota(k.begin(), k.end(), static_cast<float>(b.size())); // k = [2, 3, 4, 5]
  xnnpack::Buffer<float> packed_weights(g * round_up(nc, nr) + g * round_up(nc, nr) * round_up_po2(kc, kr * sr) / 2);
  xnn_pack_bf16_f32_gemm_gio_w(g, nc, kc, nr, kr, sr, nc,
    k.data(), b.data(), /*scale=*/nullptr, packed_weights.data(), /*extra_bytes=*/0, /*params=*/nullptr);

  const std::vector<Matcher<float>> expected = {
    0.0f,
    packed_bf16(2.0f, 4.f),
    1.0f,
    packed_bf16(3.0f, 5.0f),
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
}

TEST(PACK_BF16_F32_GEMM_GIO_W, g_eq_1_nr_gt_1_kr_gt_1) {
  size_t g = 1;
  size_t nc = 3;
  size_t kc = 3;
  size_t nr = 2;
  size_t kr = 2;
  size_t sr = 1;

  std::vector<float> b(g * nc);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2]
  std::vector<xnn_bfloat16> k(g * nc * kc);
  std::iota(k.begin(), k.end(), static_cast<float>(b.size())); // k = [3, 4, 5, 6, 7, 8, 9, 10, 11]
  xnnpack::Buffer<float> packed_weights(g * round_up(nc, nr) + g * round_up(nc, nr) * round_up_po2(kc, kr * sr) / 2);
  xnn_pack_bf16_f32_gemm_gio_w(g, nc, kc, nr, kr, sr, nc,
    k.data(), b.data(), /*scale=*/nullptr, packed_weights.data(), /*extra_bytes=*/0, /*params=*/nullptr);

  const std::vector<Matcher<float>> expected = {
    0.0f, 1.0f,
    packed_bf16(3.0f, 6.0f), packed_bf16(4.0f, 7.0f),
    packed_bf16(9.0f, 0.0f), packed_bf16(10.0f, 0.0f),
    2.0f, _,
    packed_bf16(5.0f, 8.0f), _,
    packed_bf16(11.0f, 0.0f), _,
  };
  EXPECT_THAT(packed_weights, testing::ElementsAreArray(expected));
}

TEST(PACK_BF16_F32_GEMM_GIO_W, g_gt_1) {
  size_t g = 3;
  size_t nc = 2;
  size_t kc = 2;
  size_t nr = 1;
  size_t kr = 1;
  size_t sr = 1;

  std::vector<float> b(g * nc);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4, 5]
  std::vector<xnn_bfloat16> k(g * nc * kc);
  std::iota(k.begin(), k.end(), static_cast<float>(b.size())); // k = [6,7,8,9,10,11,12,13,14,15,16,17]
  xnnpack::Buffer<float> packed_weights(g * round_up(nc, nr) + g * round_up(nc, nr) * round_up_po2(kc, kr * sr) / 2);
  xnn_pack_bf16_f32_gemm_gio_w(g, nc, kc, nr, kr, sr, nc,
    k.data(), b.data(), /*scale=*/nullptr, packed_weights.data(), /*extra_bytes=*/0, /*params=*/nullptr);

  const std::vector<Matcher<float>> expected = {
    0.0f,
    packed_bf16(6.0f, 8.f),
    1.0f,
    packed_bf16(7.0f, 9.0f),
    2.0f,
    packed_bf16(10.0f, 12.0f),
    3.0f,
    packed_bf16(11.0f, 13.0f),
    4.0f,
    packed_bf16(14.0f, 16.0f),
    5.0f,
    packed_bf16(15.0f, 17.0f),
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
}

TEST(PACK_BF16_F32_GEMM_GIO_W, g_gt_1_nr_gt_1_kr_gt_1) {
  size_t g = 3;
  size_t nc = 3;
  size_t kc = 3;
  size_t nr = 2;
  size_t kr = 2;
  size_t sr = 1;

  std::vector<float> b(g * nc);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4, 5, 6, 7, 8]
  std::vector<xnn_bfloat16> k(g * nc * kc);
  std::iota(k.begin(), k.end(), static_cast<float>(b.size())); // k = [
                                                               //   9,10,11,12,13,14,15,16,17,
                                                               //   18,19,20,21,22,23,24,25,26,
                                                               //   27,28,29,30,31,32,33,34,35
                                                               // ]
  xnnpack::Buffer<float> packed_weights(g * round_up(nc, nr) + g * round_up(nc, nr) * round_up_po2(kc, kr * sr) / 2);
  xnn_pack_bf16_f32_gemm_gio_w(g, nc, kc, nr, kr, sr, nc,
    k.data(), b.data(), /*scale=*/nullptr, packed_weights.data(), /*extra_bytes=*/0, /*params=*/nullptr);

  const std::vector<Matcher<float>> expected = {
    // Group 1.
    0.0f, 1.0f,
    packed_bf16(9.0f, 12.0f), packed_bf16(10.0f, 13.0f),
    packed_bf16(15.0f, 0.0f), packed_bf16(16.0f, 0.0f),
    2.0f, _,
    packed_bf16(11.0f, 14.0f), _,
    packed_bf16(17.0f, 0.0f), _,
    // Group 2.
    3.0f, 4.0f,
    packed_bf16(18.0f, 21.0f), packed_bf16(19.0f, 22.0f),
    packed_bf16(24.0f, 0.0f), packed_bf16(25.0f, 0.0f),
    5.0f, _,
    packed_bf16(20.0f, 23.0f), _,
    packed_bf16(26.0f, 0.0f), _,
    // Group 3.
    6.0f, 7.0f,
    packed_bf16(27.0f, 30.0f), packed_bf16(28.0f, 31.0f),
    packed_bf16(33.0f, 0.0f), packed_bf16(34.0f, 0.0f),
    8.0f, _,
    packed_bf16(29.0f, 32.0f), _,
    packed_bf16(35.0f, 0.0f), _,
  };
  EXPECT_THAT(packed_weights, testing::ElementsAreArray(expected));
}

// DWCONV packing tests.

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

  xnnpack::Buffer<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 48387);
  const std::vector<Matcher<uint8_t>> expected = {
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
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 48387);
  const std::vector<Matcher<uint8_t>> expected = {
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
    _, _, _, _,
    // then weights, channels first
    17, _, 18, _, 19, _,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<Matcher<uint8_t>> expected = {
    // bias first (cr == 2 of them)
    // 64516 + 0 - (2 + 3 + 4 + 5) * 127 = 62,738 = 0xF512
    0x12, 0xF5, 0, 0,
    // 64516 + 1 - (6 + 7 + 8 + 9) * 127 = 60,707 = 0xED23
    0x23, 0xED, 0, 0,
    // then weights, channels first
    2, 6,
    // go down the columns first
    4, 8, 3, 7, 5, 9,
    // followed by 10 kernel zero values to make up the difference with
    // primary_tile
    127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<Matcher<uint8_t>> expected = {
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
    // followed by 10 kernel zero values to make up the difference with
    // primary_tile
    127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    // bias first (cr == 2 of them)
    // 64516 + 2 - (13 + 14 + 15 + 16) * 127 = 57,152 = 0xDF40
    0x40, 0xDF, 0, 0,
    // 64516 + 3 - (17 + 18 + 19 + 20) * 127 = 55,121 = 0xD751
    0x51, 0xD7, 0, 0,
    // then weights, channels first
    13, 17, 15, 19, 14, 18, 16, 20,
    127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    // bias
    // 64516 + 4 - (21 + 22 + 23 + 24) * 127 = 53,090 = 0xCF62
    0x62, 0xCF, 0, 0,
    _, _, _, _,
    // weights
    21, _, 23, _, 22, _, 24, _,
    127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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

  xnnpack::Buffer<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 48387);
  const std::vector<Matcher<uint8_t>> expected = {
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
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 48387);
  const std::vector<Matcher<uint8_t>> expected = {
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
    _, _, _, _,
    // then weights, channels first
    9, _, 14, _, 19, _,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<Matcher<uint8_t>> expected = {
    // bias first (cr == 2 of them)
    // 64516 + 0 - (2 + 4 + 6 + 8) * 127 = 61976 = 0xF218
    0x18, 0xF2, 0, 0,
    // 64516 + 1 - (3 + 5 + 7 + 9) * 127 = 61469 = 0xF01D
    0x1D, 0xF0, 0, 0,
    // then weights, channels first
    2, 3,
    // go down the columns first
    6, 7, 4, 5, 8, 9,
    // followed by 10 kernel zero values to make up the difference with
    // primary_tile
    127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<Matcher<uint8_t>> expected = {
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
    // followed by 10 kernel zero values to make up the difference with
    // primary_tile
    127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    // bias first (cr == 2 of them)
    // 64516 + 2 - (7 + 12 + 17 + 22) * 127 = 57152 = 0xDF40
    0x40, 0xDF, 0, 0,
    // 64516 + 3 - (8 + 13 + 18 + 23) * 127 = 56645 = 0xDD45
    0x45, 0xDD, 0, 0,
    // then weights, channels first
    7, 8, 17, 18, 12, 13, 22, 23,
    127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    // bias
    // 64516 + 4 - (9 + 14 + 19 + 24) * 127 = 56138 = 0xDB4A
    0x4A, 0xDB, 0, 0,
    _, _, _, _,
    // weights
    9, _, 19, _, 14, _, 24, _,
    127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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

  xnnpack::Buffer<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      &params);

  const std::vector<Matcher<uint8_t>> expected = {
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
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      &params);

  const std::vector<Matcher<uint8_t>> expected = {
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
    _, _, _, _,
    // then weights, channels first
    17, _, 18, _, 19, _,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      &params);

  const std::vector<Matcher<uint8_t>> expected = {
    // bias first (cr == 2 of them)
    // 0 - (2 + 3 + 4 + 5) * 127 = -1778 = 0xFFFFF90E
    0x0E, 0xF9, 0xFF, 0xFF,
    // 1 - (6 + 7 + 8 + 9) * 127 = -3809 = 0xFFFFF11F
    0x1F, 0xF1, 0xFF, 0xFF,
    // then weights, channels first
    2, 6,
    // go down the columns first
    4, 8, 3, 7, 5, 9,
    // followed by 10 zero values to make up the difference with primary_tile
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      &params);

  const std::vector<Matcher<uint8_t>> expected = {
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
    // followed by 10 zero values to make up the difference with primary_tile
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
    _, _, _, _,
    // weights
    21, _, 23, _, 22, _, 24, _,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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

  xnnpack::Buffer<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      &params);

  const std::vector<Matcher<uint8_t>> expected = {
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
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      &params);

  const std::vector<Matcher<uint8_t>> expected = {
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
    _, _, _, _,
    // then weights, channels first
    9, _, 14, _, 19, _,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      &params);

  const std::vector<Matcher<uint8_t>> expected = {
    // bias first (cr == 2 of them)
    // 0 - (2 + 4 + 6 + 8) * 127 = -2540 = 0xFFFFF614
    0x14, 0xF6, 0xFF, 0xFF,
    // 1 - (3 + 5 + 7 + 9) * 127 = -3047 = 0xFFFFF419
    0x19, 0xF4, 0xFF, 0xFF,
    // then weights, channels first
    2, 3,
    // go down the columns first
    6, 7, 4, 5, 8, 9,
    // followed by 10 zero values to make up the difference with primary_tile
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<uint8_t> packed_weights(((primary_tile + sizeof(int32_t)/sizeof(uint8_t)) * round_up_po2(c, cr)));

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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      &params);

  const std::vector<Matcher<uint8_t>> expected = {
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
    // followed by 10 zero values to make up the difference with primary_tile
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
    _, _, _, _,
    // weights
    9, _, 19, _, 14, _, 24, _,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f16_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<uint16_t>> expected = {
    // bias first
    0, 1,
    // then weights, channels first
    2, 5,
    3, 6,
    4, 7,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f16_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<uint16_t>> expected = {
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
    4, _,
    // then weights, channels first
    17, _, 18, _, 19, _,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f16_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<uint16_t>> expected = {
    // bias first (cr == 2 of them)
    0, 1,
    // then weights, channels first
    2, 6,
    // go down the columns first
    4, 8, 3, 7, 5, 9,
    // followed by 10 zero values to make up the difference with primary_tile
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f16_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<uint16_t>> expected = {
    // bias first (cr == 2 of them)
    0, 1,
    // then weights, channels first
    5, 9,
    // go down the columns first
    7, 11,
    6, 10,
    8, 12,
    // followed by 10 zero values to make up the difference with primary_tile
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // bias first (cr == 2 of them)
    2, 3,
    // then weights, channels first
    13, 17, 15, 19, 14, 18, 16, 20,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // bias
    4, _,
    // weights
    21, _, 23, _, 22, _, 24, _,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f16_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<uint16_t>> expected = {
    // bias first
    0, 1,
    // then weights, channels first
    2, 3,
    4, 5,
    6, 7,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f16_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<uint16_t>> expected = {
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
    4, _,
    // then weights, channels first
    9, _, 14, _, 19, _,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f16_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<uint16_t>> expected = {
    // bias first (cr == 2 of them)
    0, 1,
    // then weights, channels first
    2, 3,
    // go down the columns first
    6, 7, 4, 5, 8, 9,
    // followed by 10 zero values to make up the difference with primary_tile
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<uint16_t> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f16_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<uint16_t>> expected = {
    // bias first (cr == 2 of them)
    0, 1,
    // then weights, channels first
    5, 6,
    // go down the columns first
    15, 16,
    10, 11,
    20, 21,
    // followed by 10 zero values to make up the difference with primary_tile
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // bias first (cr == 2 of them)
    2, 3,
    // then weights, channels first
    7, 8, 17, 18, 12, 13, 22, 23,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // bias
    4, _,
    // weights
    9, _, 19, _, 14, _, 24, _,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<float> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<float>> expected = {
    // bias first
    0.0f, 1.0f,
    // then weights, channels first
    2.0f, 5.0f,
    3.0f, 6.0f,
    4.0f, 7.0f,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<float> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<float>> expected = {
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
    4.0f, _,
    // then weights, channels first
    17.0f, _, 18.0f, _, 19.0f, _,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<float> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<float>> expected = {
    // bias first (cr == 2 of them)
    0.0f, 1.0f,
    // then weights, channels first
    2.0f, 6.0f,
    // go down the columns first
    4.0f, 8.0f, 3.0f, 7.0f, 5.0f, 9.0f,
    // followed by 10 zero values to make up the difference with primary_tile
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<float> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<float>> expected = {
    // bias first (cr == 2 of them)
    0.0f, 1.0f,
    // then weights, channels first
    5.0f, 9.0f,
    // go down the columns first
    7.0f, 11.0f,
    6.0f, 10.0f,
    8.0f, 12.0f,
    // followed by 10 zero values to make up the difference with primary_tile
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    // bias first (cr == 2 of them)
    2.0f, 3.0f,
    // then weights, channels first
    13.0f, 17.0f, 15.0f, 19.0f, 14.0f, 18.0f, 16.0f, 20.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    // bias
    4.0f, _,
    // weights
    21.0f, _, 23.0f, _, 22.0f, _, 24.0f, _,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<float> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<float>> expected = {
    // bias first
    0.0f, 1.0f,
    // then weights, channels first
    2.0f, 3.0f,
    4.0f, 5.0f,
    6.0f, 7.0f,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<float> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<float>> expected = {
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
    4.0f, _,
    // then weights, channels first
    9.0f, _, 14.0f, _, 19.0f, _,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<float> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<float>> expected = {
    // bias first (cr == 2 of them)
    0.0f, 1.0f,
    // then weights, channels first
    2.0f, 3.0f,
    // go down the columns first
    6.0f, 7.0f, 4.0f, 5.0f, 8.0f, 9.0f,
    // followed by 10 zero values to make up the difference with primary_tile
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<float> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<float>> expected = {
    // bias first (cr == 2 of them)
    0.0f, 1.0f,
    // then weights, channels first
    5.0f, 6.0f,
    // go down the columns first
    15.0f, 16.0f,
    10.0f, 11.0f,
    20.0f, 21.0f,
    // followed by 10 zero values to make up the difference with primary_tile
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    // bias first (cr == 2 of them)
    2.0f, 3.0f,
    // then weights, channels first
    7.0f, 8.0f, 17.0f, 18.0f, 12.0f, 13.0f, 22.0f, 23.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    // bias
    4.0f, _,
    // weights
    9.0f, _, 19.0f, _, 14.0f, _, 24.0f, _,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<xnn_float16> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_to_f16_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<float>> expected = {
    // bias first
    0.0f, 1.0f,
    // then weights, channels first
    2.0f, 5.0f,
    3.0f, 6.0f,
    4.0f, 7.0f,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<xnn_float16> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_to_f16_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<float>> expected = {
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
    4.0f, _,
    // then weights, channels first
    17.0f, _, 18.0f, _, 19.0f, _,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<xnn_float16> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_to_f16_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<float>> expected = {
    // bias first (cr == 2 of them)
    0.0f, 1.0f,
    // then weights, channels first
    2.0f, 6.0f,
    // go down the columns first
    4.0f, 8.0f, 3.0f, 7.0f, 5.0f, 9.0f,
    // followed by 10 zero values to make up the difference with primary_tile
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<xnn_float16> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_to_f16_dwconv_ghw_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<float>> expected = {
    // bias first (cr == 2 of them)
    0.0f, 1.0f,
    // then weights, channels first
    5.0f, 9.0f,
    // go down the columns first
    7.0f, 11.0f,
    6.0f, 10.0f,
    8.0f, 12.0f,
    // followed by 10 zero values to make up the difference with primary_tile
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    // bias first (cr == 2 of them)
    2.0f, 3.0f,
    // then weights, channels first
    13.0f, 17.0f, 15.0f, 19.0f, 14.0f, 18.0f, 16.0f, 20.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    // bias
    4.0f, _,
    // weights
    21.0f, _, 23.0f, _, 22.0f, _, 24.0f, _,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<xnn_float16> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_to_f16_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<float>> expected = {
    // bias first
    0.0f, 1.0f,
    // then weights, channels first
    2.0f, 3.0f,
    4.0f, 5.0f,
    6.0f, 7.0f,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<xnn_float16> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_to_f16_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<float>> expected = {
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
    4.0f, _,
    // then weights, channels first
    9.0f, _, 14.0f, _, 19.0f, _,
  };
  EXPECT_THAT(packed_weights, testing::ElementsAreArray(expected));
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
  xnnpack::Buffer<xnn_float16> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_to_f16_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<float>> expected = {
    // bias first (cr == 2 of them)
    0.0f, 1.0f,
    // then weights, channels first
    2.0f, 3.0f,
    // go down the columns first
    6.0f, 7.0f, 4.0f, 5.0f, 8.0f, 9.0f,
    // followed by 10 zero values to make up the difference with primary_tile
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  };
  EXPECT_THAT(packed_weights, testing::ElementsAreArray(expected));

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
  xnnpack::Buffer<xnn_float16> packed_weights(((primary_tile + 1) * round_up_po2(c, cr)));

  xnn_pack_f32_to_f16_dwconv_hwg_w(
      primary_tile,
      h,
      w,
      c,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      nullptr);

  const std::vector<Matcher<float>> expected = {
    // bias first (cr == 2 of them)
    0.0f, 1.0f,
    // then weights, channels first
    5.0f, 6.0f,
    // go down the columns first
    15.0f, 16.0f,
    10.0f, 11.0f,
    20.0f, 21.0f,
    // followed by 10 zero values to make up the difference with primary_tile
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    // bias first (cr == 2 of them)
    2.0f, 3.0f,
    // then weights, channels first
    7.0f, 8.0f, 17.0f, 18.0f, 12.0f, 13.0f, 22.0f, 23.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    // bias
    4.0f, _,
    // weights
    9.0f, _, 19.0f, _, 14.0f, _, 24.0f, _,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  };
  EXPECT_THAT(packed_weights, testing::ElementsAreArray(expected));

}

TEST(PACK_F32_TO_F16_CHW_DWCONV_HWG_W, primary_tile_eq_kernel_size) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;

  std::vector<float> b(1);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0]
  std::vector<float> k(h * w);  // k = [1, 2, 3]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  xnnpack::Buffer<xnn_float16> packed_weights(primary_tile + 1);

  xnn_pack_f32_to_f16_chw_dwconv_hwg_w(
      primary_tile,  // kernel size
      1, // groups
      k.data(),
      b.data(),
      packed_weights.data(),
      nullptr);

  const std::vector<Matcher<float>> expected = {
    // bias first
    0.0f,
    // then weights
    1.0f,
    2.0f,
    3.0f,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));

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
  xnnpack::Buffer<xnn_float16> packed_weights(g + g * h * w);

  xnn_pack_f32_to_f16_chw_dwconv_hwg_w(
      primary_tile,  // kernel size
      g, // groups
      k.data(),
      b.data(),
      packed_weights.data(),
      nullptr);

  const std::vector<Matcher<float>> expected = {
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
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));

}

TEST(PACK_F16_CHW_DWCONV_HWG_W, primary_tile_eq_kernel_size) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;

  std::vector<uint16_t> b(1);
  std::iota(b.begin(), b.end(), 0);  // b = [0]
  std::vector<uint16_t> k(h * w);  // k = [1, 2, 3]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  xnnpack::Buffer<uint16_t> packed_weights(primary_tile + 1);

  xnn_pack_f16_chw_dwconv_hwg_w(
      primary_tile,  // kernel size
      1, // groups
      k.data(),
      b.data(),
      packed_weights.data(),
      nullptr);

  const std::vector<Matcher<uint16_t>> expected = {
    // bias first
    0,
    // then weights
    1,
    2,
    3,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<uint16_t> packed_weights(g + g * h * w);

  xnn_pack_f16_chw_dwconv_hwg_w(
      primary_tile,  // kernel size
      g, // groups
      k.data(),
      b.data(),
      packed_weights.data(),
      nullptr);

  const std::vector<Matcher<uint16_t>> expected = {
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
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
}


TEST(PACK_F16_CHW_DWCONV_GHW_W, primary_tile_eq_kernel_size) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;

  std::vector<uint16_t> b(1);
  std::iota(b.begin(), b.end(), 0);  // b = [0]
  std::vector<uint16_t> k(h * w);  // k = [1, 2, 3]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  xnnpack::Buffer<uint16_t> packed_weights(primary_tile + 1);

  xnn_pack_f16_chw_dwconv_ghw_w(
      primary_tile,  // kernel size
      1, // groups
      k.data(),
      b.data(),
      packed_weights.data(),
      nullptr);

  const std::vector<Matcher<uint16_t>> expected = {
    // bias first
    0,
    // then weights
    1,
    2,
    3,
  };
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
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
  xnnpack::Buffer<uint16_t> packed_weights(g + g * h * w);

  xnn_pack_f16_chw_dwconv_ghw_w(
      primary_tile,  // kernel size
      g, // groups
      k.data(),
      b.data(),
      packed_weights.data(),
      nullptr);

  const std::vector<Matcher<uint16_t>> expected = {
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
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
}


TEST(PACK_F32_DWCONV_OKI_W, primary_tile_eq_kernel_size) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;

  std::vector<float> b(3);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0]
  std::vector<float> k(h * w);  // k = [3, 4, 5]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  xnnpack::Buffer<float> packed_weights(primary_tile * 2);

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

  const std::vector<Matcher<float>> expected = {
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
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
}

TEST(PACK_F32_TO_F16_DWCONV_OKI_W, primary_tile_eq_kernel_size) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;

  std::vector<float> b(3);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0]
  std::vector<float> k(h * w);  // k = [3, 4, 5]
  std::iota(k.begin(), k.end(), static_cast<float>(b.size()));
  xnnpack::Buffer<xnn_float16> packed_weights(primary_tile * 2);

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

  const std::vector<Matcher<float>> expected = {
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
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
}

TEST(PACK_F32_TO_F16_DWCONV_OKI_W, null_bias) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;

  std::vector<float> k(h * w);  // k = [3, 4, 5]
  std::iota(k.begin(), k.end(), 3.0f);
  xnnpack::Buffer<xnn_float16> packed_weights(primary_tile * 2);

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

  const std::vector<Matcher<float>> expected = {
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
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
}

TEST(PACK_F16_DWCONV_OKI_W, primary_tile_eq_kernel_size) {
  const size_t primary_tile = 3;
  const size_t h = 3;
  const size_t w = 1;

  std::vector<uint16_t> b(3);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2]
  std::vector<uint16_t> k(h * w);  // k = [3, 4, 5]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  xnnpack::Buffer<uint16_t> packed_weights(primary_tile * 2);

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

  const std::vector<Matcher<uint16_t>> expected = {
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
  EXPECT_THAT(packed_weights, ElementsAreArray(expected));
}

}  // namespace
