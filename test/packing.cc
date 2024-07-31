// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack/math.h"
#include "xnnpack/microkernel-utils.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pack.h"

// QD8-F32-QC4W GEMM packing tests.

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
  std::vector<uint8_t> packed_weights(g * round_up(nc, nr) * (sizeof(float) + round_up_po2(kc, kr * sr) / 2));
  auto a = xnn_qs8_qc4w_packing_params{ 0, 0x8 };
  xnn_pack_qs8_qc4w_gemm_goi_w(g, nc, kc, nr, kr, sr,
    k.data(), b.data(), /*scale=*/nullptr, packed_weights.data(), /*extra_bytes=*/0, /*params=*/&a);

  const std::vector<uint8_t> expected = {
    // 1 bias.
    0x00, 0x00, 0x00, 0x00,
    0x40, 0x51, 0x62, 0x73, 0xC8, 0xD9, 0xEA, 0xFB,
  };
  EXPECT_EQ(expected, packed_weights);
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
  std::vector<uint8_t> packed_weights(g * round_up(nc, nr) * (sizeof(float) + round_up_po2(kc, kr * sr) / 2));
  auto a = xnn_qs8_qc4w_packing_params{ 0, 0x8 };
  xnn_pack_qs8_qc4w_gemm_gio_w(g, nc, kc, nr, kr, sr,/*k_stride=*/round_up_po2(nc, 2),
    k.data(), b.data(), /*scale=*/nullptr, packed_weights.data(), /*extra_bytes=*/0, /*params=*/&a);

  const std::vector<uint8_t> expected = {
    // 1 bias.
    0x00, 0x00, 0x00, 0x00,
    0x40, 0x51, 0x62, 0x73, 0xC8, 0xD9, 0xEA, 0xFB,
  };
  EXPECT_EQ(expected, packed_weights);
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
  std::vector<uint8_t> packed_weights(g * round_up(nc, nr) * (sizeof(float) + round_up_po2(kc, kr * sr) / 2));
  auto a = xnn_qs8_qc4w_packing_params{ 0, 0x8 };
  xnn_pack_qs8_qc4w_gemm_goi_w(g, nc, kc, nr, kr, sr,
    k.data(), b.data(), /*scale=*/nullptr, packed_weights.data(), /*extra_bytes=*/0, /*params=*/&a);

  const std::vector<uint8_t> expected = {
    // 2 bias.
    0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00,
    0x40, 0x51, 0x62, 0x73,
    0xC8, 0xD9, 0xEA, 0xFB,
  };
  EXPECT_EQ(expected, packed_weights);
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
  std::vector<uint8_t> packed_weights(g * round_up(nc, nr) * (sizeof(float) + round_up_po2(kc, kr * sr) / 2));
  auto a = xnn_qs8_qc4w_packing_params{ 0, 0x8 };
  xnn_pack_qs8_qc4uw_gemm_goi_w(g, nc, kc, nr, kr, sr,
    k.data(), b.data(), /*scale=*/nullptr, packed_weights.data(), /*extra_bytes=*/0, /*params=*/&a);

  const std::vector<uint8_t> expected = {
    // 2 bias.
    0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00,
    0xC8, 0xD9, 0xEA, 0xFB,
    0x40, 0x51, 0x62, 0x73,
  };
  EXPECT_EQ(expected, packed_weights);
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
  std::vector<uint8_t> packed_weights(g * round_up(nc, nr) * (sizeof(float) + round_up_po2(kc, kr * sr) / 2));
  auto a = xnn_qs8_qc4w_packing_params{ 0, 0x8 };
  xnn_pack_qs8_qc4w_gemm_gio_w(g, nc, kc, nr, kr, sr, /*k_stride=*/nc,
    k.data(), b.data(), /*scale=*/nullptr, packed_weights.data(), /*extra_bytes=*/0, /*params=*/&a);

  const std::vector<uint8_t> expected = {
    // 2 bias.
    0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00,
    0x40, 0x51, 0x62, 0x73,
    0xC8, 0xD9, 0xEA, 0xFB,
  };
  EXPECT_EQ(expected, packed_weights);
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
  std::vector<uint16_t> scale(nc * k_num_blocks, 0);
  std::fill(scale.begin(), scale.end(), math_cvt_bf16_fp32(853.6010));
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
    /*channel_stride=*/ nr,
    /*stride=*/ stride,
    /*substride=*/ stride,
    /*num_blocks=*/k_num_blocks,
    /*block_stride=*/block_stride,
    /*stride_offset=*/ 0,
    /*scale=*/ scale.data(),
    /*packed_weight=*/ packed_weights.data() + start_offset);

  const std::vector<uint8_t> expected = {
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
  EXPECT_EQ(expected, packed_weights);
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
  std::vector<uint16_t> scale(nc * k_num_blocks, 0);
  std::fill(scale.begin(), scale.end(), math_cvt_bf16_fp32(853.6010));

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
    /*channels_subtile=*/nr,
    /*stride=*/stride,
    /*substride=*/stride,
    /*num_blocks=*/k_num_blocks,
    /*block_stride=*/block_stride,
    /*stride_offset=*/0,
    /*scale=*/scale.data(),
    /*packed_w=*/packed_weights.data() + start_offset);

  const std::vector<uint8_t> expected = {
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

  EXPECT_EQ(expected, packed_weights);
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
  std::vector<uint16_t> scale(nc * k_num_blocks, 0);
  std::fill(scale.begin(), scale.end(), math_cvt_bf16_fp32(853.6010));


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
    /*channel_stride=*/ nr,
    /*stride=*/ stride,
    /*substride=*/ stride,
    /*num_blocks=*/k_num_blocks,
    /*block_stride=*/block_stride,
    /*stride_offset=*/ 0,
    /*scale=*/ scale.data(),
    /*packed_weight=*/ packed_weights.data() + start_offset);


  const std::vector<uint8_t> expected = {
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
  EXPECT_EQ(expected, packed_weights);
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
  std::vector<uint16_t> scale(nc * k_num_blocks, 0);
  std::fill(scale.begin(), scale.end(), math_cvt_bf16_fp32(853.6010));


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
    /*channel_subtile=*/nr,
    /*stride=*/stride,
    /*substride=*/stride,
    /*num_blocks=*/k_num_blocks,
    /*block_stride=*/block_stride,
    /*stride_offset=*/0,
    /*scale=*/scale.data(),
    /*packed_w=*/packed_weights.data() + start_offset);

  const std::vector<uint8_t> expected = {
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
  EXPECT_EQ(expected, packed_weights);
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
  std::vector<float> packed_weights(g * round_up(nc, nr) * (1 + round_up_po2(kc, kr * sr)));
  xnn_pack_f32_gemm_gio_w(g, nc, kc, nr, kr, sr, nc,
    k.data(), b.data(), /*scale=*/nullptr, packed_weights.data(), /*extra_bytes=*/0, /*params=*/nullptr);

  const std::vector<float> expected = {
    0.0f,
    2.0f, 4.f,
    1.0f,
    3.0f, 5.0f,
  };
  EXPECT_EQ(expected, packed_weights);
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
  std::vector<float> packed_weights(g * round_up(nc, nr) * (1 + round_up_po2(kc, kr * sr)));
  xnn_pack_f32_gemm_gio_w(g, nc, kc, nr, kr, sr, nc,
    k.data(), b.data(), /*scale=*/nullptr, packed_weights.data(), /*extra_bytes=*/0, /*params=*/nullptr);

  const std::vector<float> expected = {
    0.0f, 1.0f,
    3.0f, 6.0f, 4.0f, 7.0f,
    9.0f, 0.0f, 10.0f, 0.0f,
    2.0f, 0.0f,
    5.0f, 8.0f, 0.0f, 0.0f,
    11.0f, 0.0f, 0.0f, 0.0f,
  };
  EXPECT_EQ(expected, packed_weights);
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
  std::vector<float> packed_weights(g * round_up(nc, nr) * (1 + round_up_po2(kc, kr * sr)));
  xnn_pack_f32_gemm_gio_w(g, nc, kc, nr, kr, sr, nc,
    k.data(), b.data(), /*scale=*/nullptr, packed_weights.data(), /*extra_bytes=*/0, /*params=*/nullptr);

  const std::vector<float> expected = {
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
  EXPECT_EQ(expected, packed_weights);
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
  std::vector<float> packed_weights(g * round_up(nc, nr) * (1 + round_up_po2(kc, kr * sr)));
  xnn_pack_f32_gemm_gio_w(g, nc, kc, nr, kr, sr, nc,
    k.data(), b.data(), /*scale=*/nullptr, packed_weights.data(), /*extra_bytes=*/0, /*params=*/nullptr);

  const std::vector<float> expected = {
    // Group 1.
    0.0f, 1.0f,
    9.0f, 12.0f, 10.0f, 13.0f,
    15.0f, 0.0f, 16.0f, 0.0f,
    2.0f, 0.0f,
    11.0f, 14.0f, 0.0f, 0.0f,
    17.0f, 0.0f, 0.0f, 0.0f,
    // Group 2.
    3.0f, 4.0f,
    18.0f, 21.0f, 19.0f, 22.0f,
    24.0f, 0.0f, 25.0f, 0.0f,
    5.0f, 0.0f,
    20.0f, 23.0f, 0.0f, 0.0f,
    26.0f, 0.0f, 0.0f, 0.0f,
    // Group 3.
    6.0f, 7.0f,
    27.0f, 30.0f, 28.0f, 31.0f,
    33.0f, 0.0f, 34.0f, 0.0f,
    8.0f, 0.0f,
    29.0f, 32.0f, 0.0f, 0.0f,
    35.0f, 0.0f, 0.0f, 0.0f,
  };
  EXPECT_EQ(expected, packed_weights);
}

// DWCONV packing tests.

// Calculates the size (number of elements) of packed weights required for a multi-pass dwconv.
// Assume that bias and filter elements are of the same size, and extra_weights_byte is 0.
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
  return xnn_dwconv_multipass_weights_size(
    xnn_dwconv_multipass_tile_size(kernel_size, first_pass_tile, middle_pass_tile, last_pass_tile), channels,
    channel_tile, channel_subtile, channel_round,
    /*bias_element_size=*/1, /*log2_filter_element_size=*/0, /*extra_weights_byte=*/0);
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

// Weights count for QS8 is really the size of the weights, count is not accurate because bias is different size.
static size_t qs8_multipass_weights_count(
  size_t kernel_size,
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t channels,
  size_t channel_tile,
  size_t channel_subtile)
{
  return xnn_dwconv_multipass_weights_size(
    xnn_dwconv_multipass_tile_size(kernel_size, first_pass_tile, middle_pass_tile, last_pass_tile),
    channels, channel_tile, channel_subtile, channel_subtile,
    /*bias_element_size=*/4, /*log2_filter_element_size=*/0, /*extra_weights_byte=*/0);
}

static size_t qs8_multipass_weights_count(
  size_t kernel_size,
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t channels,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round)
{
  return xnn_dwconv_multipass_weights_size(
    xnn_dwconv_multipass_tile_size(kernel_size, first_pass_tile, middle_pass_tile, last_pass_tile),
    channels, channel_tile, channel_subtile, channel_round,
    /*bias_element_size=*/4, /*log2_filter_element_size=*/0, /*extra_weights_byte=*/0);
}

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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

TEST(PACK_QU8_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<uint8_t> k(c * h * w);  // k = [2, 3, 4, 5, 6, 7, 8, 9]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));

  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<uint8_t> expected = {
    // First pass, bias first.
    // 64516 + 0 - (2 + 3 + 4 + 5) * 127 = 62738 = 0xF512
    0x12, 0xF5, 0, 0,
    // 64516 + 1 - (6 + 7 + 8 + 9) * 127 = 60707 = 0xED23
    0x23, 0xED, 0, 0,
    2, 6,  // 2 weights, channels first, then columns.
    4, 8,
    3, 7,  // Last pass, 2 weights.
    5, 9,
    0, 0,  // Padding to last_pass_tile.
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once_channels_gt_cr) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint8_t> k(c * h * w);  // k = [5, 6, 7, 8, // first 2x2 kernel
                                    //      9, 10, 11, 12, // second 2x2 kernel
                                    //      13, 14, 15, 16, // third 2x2 kernel
                                    //      17, 18, 19, 20, // fourth 2x2 kernel
                                    //      21, 22, 23, 24, // fifth 2x2 kernel
                                    //      ]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<uint8_t> expected = {
    // First pass, bias first.
    // 64516 + 0 - (5 + 6 + 7 + 8) * 127 = 61214 = 0xEF1E
    0x1E, 0xEF, 0, 0, // bias
    // 64516 + 1 - (9 + 10 + 11 + 12) * 127 = 59183 = 0xE72F
    0x2F, 0xE7, 0, 0,
    5, 9, // 2 weights, 2 channels first, then columns
    7, 11,
    // Bias.
    // 64516 + 2 - (13 + 14 + 15 + 16) * 127 = 57152 = 0xDF40
    0x40, 0xDF, 0, 0,
    // 64516 + 3 - (17 + 18 + 19 + 20) * 127 = 55121 = 0xD751
    0x51, 0xD7, 0, 0,
    13, 17, // 2 weights, 2 channels first, then columns
    15, 19,
    // 64516 + 4 - (21 + 22 + 23 + 24) * 127 = 53090 = 0xCF62
    0x62, 0xCF, 0, 0,
    0, 0, 0, 0,
    21, 0, // 2 weights, 1 remainder channels first, then columns
    23, 0,
    // No middle pass.
    6, 10, // last pass, 2 weights, 2 channels first, then columns
    8, 12,
    0, 0,  // padding
    14, 18,
    16, 20,
    0, 0,  // padding
    22, 0,
    24, 0,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_MULTIPASS_DWCONV_GHW_W, one_middle_pass_tile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 1;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<uint8_t> k(c * h * w);  // k = [2, 3, // first 2x2 kernel
                                    //      4, 5,
                                    //      6, 7, // second 2x2 kernel
                                    //      8, 9]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<uint8_t> expected = {
    // First pass only has 1 element.
    // 64516 + 0 - (2 + 3 + 4 + 5) * 127 = 62738 = 0xF512
    0x12, 0xF5, 0, 0,
    // 64516 + 1 - (6 + 7 + 8 + 9) * 127 = 60707 = 0xED23
    0x23, 0xED, 0, 0,
    2, 6, // weights, 2 channels, 1 element.
    // Middle pass has 2 elements, columns first.
    4, 8,
    3, 7,
    // Last pass.
    5, 9,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_MULTIPASS_DWCONV_GHW_W, one_middle_pass_tile_channels_gt_cr) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint8_t> k(c * h * w);  // k = [5, 6, // first 2x2 kernel
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
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<uint8_t> expected = {
    // First pass only has 1 element, bias first.
    // 64516 + 0 - (5 + 6 + 7 + 8) * 127 = 61214 = 0xEF1E
    0x1E, 0xEF, 0, 0, // bias
    // 64516 + 1 - (9 + 10 + 11 + 12) * 127 = 59183 = 0xE72F
    0x2F, 0xE7, 0, 0,
    5, 9, // weights, 2 channels, 1 element.
    // Bias.
    // 64516 + 2 - (13 + 14 + 15 + 16) * 127 = 57152 = 0xDF40
    0x40, 0xDF, 0, 0,
    // 64516 + 3 - (17 + 18 + 19 + 20) * 127 = 55121 = 0xD751
    0x51, 0xD7, 0, 0,
    13, 17, // weights, 2 channels, 1 element.
    // Bias.
    // 64516 + 4 - (21 + 22 + 23 + 24) * 127 = 53090 = 0xCF62
    0x62, 0xCF, 0, 0,
    0, 0, 0, 0,
    21, 0, // weights, 1 remainder channel, 1 element.
    // Middle pass has 2 elements, channels first, then columns.
    7, 11,
    6, 10,
    15, 19,
    14, 18,
    // Middle pass, 1 remainder channel.
    23, 0,
    22, 0,
    // Last pass,
    8, 12,
    0, 0,  // padding
    16, 20,
    0, 0,  // padding
    24, 0,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_MULTIPASS_DWCONV_GHW_W, multiple_middle_pass_tile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 3;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<uint8_t> k(c * h * w);  // k = [2, 3, 4, // first 2x3 kernel
                                    //      5, 6, 7,
                                    //      8, 9, 10, // second 2x3 kernel
                                    //      11, 12, 13]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 96774);
  const std::vector<uint8_t> expected = {
    // First pass has 2 elements.
    // 96774 + 0 - (2 + 3 + 4 + 5 + 6 + 7) * 127 = 93345 = 0x16CA1
    0xA1, 0x6C, 0x01, 0,
    // 96774 + 1 - (8 + 9 + 10 + 11 + 12 + 13) * 127 = 88774 = 0x15AC6
    0xC6, 0x5A, 0x01, 0,
    2, 8, // 1 weight, 2 channels first, then columns
    // Middle pass 1 (2 elements per pass).
    5, 11,
    3, 9,
    // Middle pass 2 (2 elements per pass).
    6, 12,
    4, 10,
    // Last pass.
    7, 13,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_MULTIPASS_DWCONV_GHW_W, multiple_middle_pass_tile_channels_gt_cr) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 3;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint8_t> k(c * h * w);  // k = [5, 6, 7, // first 2x3 kernel
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
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);


  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 96774);
  const std::vector<uint8_t> expected = {
    // First pass only has 1 element, bias first.
    // 96774 + 0 - (5 + 6 + 7 + 8 + 9 + 10) * 127 = 91059 = 0x163B3
    0xB3, 0x63, 0x01, 0,
    // 96774 + 1 - (11 + 12 + 13 + 14 + 15 + 16) * 127 = 86488 = 0x151D8
    0xD8, 0x51, 0x01, 0,
    5, 11, // 1 weight, 2 channels, 2 elements.
    // 96774 + 2 - (17 + 18 + 19 + 20 + 21 + 22) * 127 = 81917 = 0x13FFD
    0xFD, 0x3F, 0x01, 0,
    // 96774 + 3 - (23 + 24 + 25 + 26 + 27 + 28) * 127 = 77346 = 0x12E22
    0x22, 0x2E, 0x01, 0,
    17, 23, // 1 weight, 2 channels, 2 elements.
    // 96774 + 4 - (29 + 30 + 31 + 32 + 33 + 34) * 127 = 72775 = 0x11C47
    0x47, 0x1C, 0x01, 0,
    0, 0, 0, 0,
    29, 0, // 1 weight, 1 remainder channel, 2 elements.
    // Middle pass has 2 elements, channels first, then columns.
    8, 14,
    6, 12,
    20, 26,
    18, 24,
    // 1 remainder channel.
    32, 0,
    30, 0,
    // Second middle pass, 2 elements.
    9, 15,
    7, 13,
    21, 27,
    19, 25,
    // 1 remainder channel.
    33, 0,
    31, 0,
    // Last pass
    10, 16,
    0, 0,  // padding
    22, 28,
    0, 0,  // padding
    // 1 remainder channel.
    34, 0,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once_channel_subtile) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 4;
  const size_t channel_subtile = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint8_t> k(c * h * w);  // k = [5, 6, // first 2x2 kernel
                                    //      7, 8,
                                    //      9, 10, // second 2x2 kernel
                                    //      11, 12,
                                    //      13, 14, // third 2x2 kernel
                                    //      15, 16,
                                    //      17, 18, // fourth 2x2 kernel
                                    //      19, 20,
                                    //      21, 22, // fifth 2x2 kernel
                                    //      23, 24 ]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<uint8_t> expected = {
    // First pass.
    // 64516 + 0 - (5 + 6 + 7 + 8) * 127 = 61214 = 0xEF1E
    0x1E, 0xEF, 0, 0, // bias
    // 64516 + 1 - (9 + 10 + 11 + 12) * 127 = 59183 = 0xE72F
    0x2F, 0xE7, 0, 0,
    // 64516 + 2 - (13 + 14 + 15 + 16) * 127 = 57152 = 0xDF40
    0x40, 0xDF, 0, 0,
    // 64516 + 3 - (17 + 18 + 19 + 20) * 127 = 55121 = 0xD751
    0x51, 0xD7, 0, 0,
    5, 9, 13, 17,  // 2 weights, 4 channels first, then columns
    7, 11, 15, 19,
    // Bias, 1 last channel, 1 padding up to channel_subtile
    // 64516 + 4 - (21 + 22 + 23 + 24) * 127 = 53090 = 0xCF62
    0x62, 0xCF, 0, 0,
    0, 0, 0, 0,
    21, 0, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    23, 0,
    // No middle pass.
    6, 10, 14, 18,  // last pass, 2 weights, 4 channels first
    8, 12, 16, 20,
    0, 0, 0, 0, // padding to last_pass_tile
    22, 0, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    24, 0,
    0, 0, // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_MULTIPASS_DWCONV_GHW_W, one_middle_pass_channel_subtile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 4;
  const size_t channel_subtile = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint8_t> k(c * h * w);  // k = [5, 6, // first 2x2 kernel
                                    //      7, 8,
                                    //      9, 10, // second 2x2 kernel
                                    //      11, 12,
                                    //      13, 14, // third 2x2 kernel
                                    //      15, 16,
                                    //      17, 18, // fourth 2x2 kernel
                                    //      19, 20,
                                    //      21, 22, // fifth 2x2 kernel
                                    //      23, 24 ]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<uint8_t> expected = {
    // First pass, bias, 4 channels.
    // 64516 + 0 - (5 + 6 + 7 + 8) * 127 = 61214 = 0xEF1E
    0x1E, 0xEF, 0, 0, // bias
    // 64516 + 1 - (9 + 10 + 11 + 12) * 127 = 59183 = 0xE72F
    0x2F, 0xE7, 0, 0,
    // 64516 + 2 - (13 + 14 + 15 + 16) * 127 = 57152 = 0xDF40
    0x40, 0xDF, 0, 0,
    // 64516 + 3 - (17 + 18 + 19 + 20) * 127 = 55121 = 0xD751
    0x51, 0xD7, 0, 0,
    5, 9, 13, 17,  // 1 weight, 4 channels first, then columns
    // Bias, 1 last channel, 1 padding up to channel_subtile.
    // 64516 + 4 - (21 + 22 + 23 + 24) * 127 = 53090 = 0xCF62
    0x62, 0xCF, 0, 0,
    0, 0, 0, 0,
    21, 0, // 1 weight, 1 last channel, 1 padding up to channel_subtile
    // 1 middle pass
    7, 11, 15, 19, // 2 weights, 4 channels first, then columns
    6, 10, 14, 18,
    23, 0, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    22, 0,
    // Last pass.
    8, 12, 16, 20,  // 1 weight, 4 channels first
    0, 0, 0, 0, // padding to last_pass_tile
    24, 0, // 1 weight, 1 last channel, 1 padding up to channel_subtile
    0, 0, // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once_channel_subtile_rounded) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 7;
  const size_t cr = 4;
  const size_t channel_subtile = 2;
  // c rounded to channel_subtile is 8, so we will have 2 channel_tile loops in the first and middle pass.

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4, 5, 6]
  std::vector<uint8_t> k(c * h * w);  // k = [7, 8, // first 2x2 kernel
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
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<uint8_t> expected = {
    // First pass, bias, 4 channels.
    // 64516 + 0 - (7 + 8 + 9 + 10) * 127 = 60198 = 0xEB26
    0x26, 0xEB, 0, 0,
    // 64516 + 1 - (11 + 12 + 13 + 14) * 127 = 58167 = 0xE337
    0x37, 0xE3, 0, 0,
    // 64516 + 2 - (15 + 16 + 17 + 18) * 127 = 56136 = 0xDB48
    0x48, 0xDB, 0, 0,
    // 64516 + 3 - (19 + 20 + 21 + 22) * 127 = 54105 = 0xD359
    0x59, 0xD3, 0, 0,
    7, 11, 15, 19,  // 2 weights, 4 channels first, then columns
    9, 13, 17, 21,
    // Bias, 3 remainder channels, 1 padding up to channel_Tile.
    // 64516 + 4 - (23 + 24 + 25 + 26) * 127 = 52074 = 0xCB6A
    0x6A, 0xCB, 0, 0,
    // 64516 + 5 - (27 + 28 + 29 + 30) * 127 = 50043 = 0xC37B
    0x7B, 0xC3, 0, 0,
    // 64516 + 6 - (31 + 32 + 33 + 34) * 127 = 48012 = 0xBB8C
    0x8C, 0xBB, 0, 0,
    0, 0, 0, 0,
    23, 27, 31, 0,  // 2 weights, 3 remainder channels, 1 padding up to channel_tile
    25, 29, 33, 0,
    // No middle pass.
    // Last pass.
    8, 12, 16, 20,  // last pass, 2 weights, 4 channels first
    10, 14, 18, 22,
    0, 0, 0, 0, // padding to last_pass_tile
    24, 28, // last pass, 2 weights, channel_subtile (2)
    26, 30,
    0, 0, // padding to last_pass_tile
    32, 0, // 1 remainder channel, 1 padding up to channel_subtile
    34, 0, // 1 remainder channel, 1 padding up to channel_subtile
    0, 0, // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once_channel_subtile_rounded_to_channel_round) {
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

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4, 5]
  std::vector<uint8_t> k(c * h * w);  // k = [6, 7, 8, 9,
                                    //      10, 11, 12, 13
                                    //      14, 15, 16, 17,
                                    //      18, 19, 20, 21,
                                    //      22, 23, 24, 25,
                                    //      26, 27, 28, 29,]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile, channel_round));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<uint8_t> expected = {
    // First pass, bias, 4 channels.
    // 64516 + 0 - (6 + 7 + 8 + 9) * 127 = 60706 = 0xED22
    0x22, 0xED, 0, 0,
    // 64516 + 1 - (10 + 11 + 12 + 13) * 127 = 58675 = 0xE533
    0x33, 0xE5, 0, 0,
    // 64516 + 2 - (14 + 15 + 16 + 17) * 127 = 56644 = 0xDD44
    0x44, 0xDD, 0, 0,
    // 64516 + 3 - (18 + 19 + 20 + 21) * 127 = 54613 = 0xD555
    0x55, 0xD5, 0, 0,
    6, 10, 14, 18, // 2 weights, 4 channels first, then columns
    8, 12, 16, 20,
    // Bias, 2 remainder channels, 2 padding up to channel_subtile.
    // 64516 + 4 - (22 + 23 + 24 + 25) * 127 = 52582 = 0xCD66
    0x66, 0xCD, 0, 0,
    // 64516 + 5 - (26 + 27 + 28 + 29) * 127 = 50551 = 0xC577
    0x77, 0xC5, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    22, 26, 0, 0, // 2 weights, 2 remainder channel, 2 padding up to channel_subtile
    24, 28, 0, 0,
    // No middle pass.
    7, 11, 15, 19, // last pass, 2 weights, 4 channels first.
    9, 13, 17, 21,
    0, 0, 0, 0, // padding to last_pass_tile
    23, 27, 0, 0, // 2 weights, channel_subtile (4)
    25, 29, 0, 0,
    0, 0, 0, 0, // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_MULTIPASS_DWCONV_GHW_W, one_middle_pass_channel_subtile_rounded) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 7;
  const size_t cr = 4;
  const size_t channel_subtile = 2;
  // c rounded to channel_subtile is 8, so we will have 2 channel_tile loops in first and middle pass.

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);   // b = [0, 1, 2, 3, 4, 5, 6]
  std::vector<uint8_t> k(c * h * w);  // k = [7, 8, // first 2x2 kernel
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
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<uint8_t> expected = {
    // First pass, bias, 4 channels.
    // 64516 + 0 - (7 + 8 + 9 + 10) * 127 = 60198 = 0xEB26
    0x26, 0xEB, 0, 0,
    // 64516 + 1 - (11 + 12 + 13 + 14) * 127 = 58167 = 0xE337
    0x37, 0xE3, 0, 0,
    // 64516 + 2 - (15 + 16 + 17 + 18) * 127 = 56136 = 0xDB48
    0x48, 0xDB, 0, 0,
    // 64516 + 3 - (19 + 20 + 21 + 22) * 127 = 54105 = 0xD359
    0x59, 0xD3, 0, 0,
    7, 11, 15, 19,  // 1 weight, 4 channels first, then columns
    // Bias, 3 remainder channels, 1 padding up to channel_Tile.
    // 64516 + 4 - (23 + 24 + 25 + 26) * 127 = 52074 = 0xCB6A
    0x6A, 0xCB, 0, 0,
    // 64516 + 5 - (27 + 28 + 29 + 30) * 127 = 50043 = 0xC37B
    0x7B, 0xC3, 0, 0,
    // 64516 + 6 - (31 + 32 + 33 + 34) * 127 = 48012 = 0xBB8C
    0x8C, 0xBB, 0, 0,
    0, 0, 0, 0,
    23, 27, 31, 0,  // 1 weight, 3 remainder channels, 1 padding up to channel_tile
    // 1 middle pass.
    9, 13, 17, 21, // 2 weights, 4 channels first
    8, 12, 16, 20,
    25, 29, 33, 0, // 3 remainder channels, 1 padding up to channel_tile
    24, 28, 32, 0,
    // Last pass.
    10, 14, 18, 22,  // last pass, 1 weight, 4 channels first
    0, 0, 0, 0, // padding to last_pass_tile
    26, 30, // 1 weight, channel_subtile,
    0, 0, // padding to last_pass_tile
    34, 0, // 1 weight, 1 remainder channel, 1 padding up to channel_subtile
    0, 0, // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<uint8_t> k(c * h * w);  // k = [2, 3,
                                    //      4, 5,
                                    //      6, 7,
                                    //      8, 9]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<uint8_t> expected = {
    // First pass, bias first.
    // 64516 + 0 - (2 + 4 + 6 + 8) * 127 = 61976 = 0xF218
    0x18, 0xF2, 0, 0,
    // 64516 + 1 - (3 + 5 + 7 + 9) * 127 = 61469 = 0xF01D
    0x1D, 0xF0, 0, 0,
    2, 3, // First pass, 2 weights, channels first, then columns
    6, 7,
    // No middle pass.
    4, 5, // Last pass, 2 weights
    8, 9,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once_channels_gt_cr) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint8_t> k(c * h * w);  // k = [5, 6, 7, 8, 9,
                                    //      10, 11, 12, 13, 14,
                                    //      15, 16, 17, 18, 19,
                                    //      20, 21, 22, 23, 24,
                                    //      ]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<uint8_t> expected = {
    // First pass.
    // 64516 + 0 - (5 + 10 + 15 + 20) * 127 = 58166 = 0xE336
    0x36, 0xE3, 0, 0,
    // 64516 + 1 - (6 + 11 + 16 + 21) * 127 = 57659 = 0xE13B
    0x3B, 0xE1, 0, 0,
    5, 6, // 2 weights, 2 channels first, then columns
    15, 16,
    // 64516 + 2 - (7 + 12 + 17 + 22) * 127 = 57152 = 0xDF40
    0x40, 0xDF, 0, 0,
    // 64516 + 3 - (8 + 13 + 18 + 23) * 127 = 56645 = 0xDD45
    0x45, 0xDD, 0, 0,
    7, 8, // 2 weights, 2 channels first, then columns
    17, 18,
    // Bias, 1 remainder channel, 1 padding up to channel_subtile.
    // 64516 + 4 - (9 + 14 + 19 + 24) * 127 = 56138 = 0xDB4A
    0x4A, 0xDB, 0, 0,
    0, 0, 0, 0,
    9, 0, // weights, 1 remainder channels first, then columns
    19, 0,
    // No middle pass.
    // Last pass, 2 weights
    10, 11,
    20, 21,
    0, 0,  // padding
    12, 13,
    22, 23,
    0, 0,  // padding
    14, 0,
    24, 0,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_MULTIPASS_DWCONV_HWG_W, one_middle_pass_tile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 1;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<uint8_t> k(c * h * w);  // k = [2, 3,
                                    //      4, 5,
                                    //      6, 7,
                                    //      8, 9]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<uint8_t> expected = {
    // First pass only has 1 element.
    // 64516 + 0 - (2 + 4 + 6 + 8) * 127 = 61976 = 0xF218
    0x18, 0xF2, 0, 0,
    // 64516 + 1 - (3 + 5 + 7 + 9) * 127 = 61469 = 0xF01D
    0x1D, 0xF0, 0, 0,
    2, 3, // weights, 2 channels, 1 element.
    // Middle pass has 2 elements, columns first.
    6, 7,
    4, 5,
    // Last pass.
    8, 9,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_MULTIPASS_DWCONV_HWG_W, one_middle_pass_tile_channels_gt_cr) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint8_t> k(c * h * w);  // k = [5, 6, 7, 8, 9,
                                    //      10, 11, 12, 13, 14,
                                    //      15, 16, 17, 18, 19,
                                    //      20, 21, 22, 23, 24,
                                    //      ]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<uint8_t> expected = {
    // First pass only has 1 element.
    // 64516 + 0 - (5 + 10 + 15 + 20) * 127 = 58166 = 0xE336
    0x36, 0xE3, 0, 0,
    // 64516 + 1 - (6 + 11 + 16 + 21) * 127 = 57659 = 0xE13B
    0x3B, 0xE1, 0, 0,
    5, 6, // weights, 2 channels, 1 element.
    // 64516 + 2 - (7 + 12 + 17 + 22) * 127 = 57152 = 0xDF40
    0x40, 0xDF, 0, 0,
    // 64516 + 3 - (8 + 13 + 18 + 23) * 127 = 56645 = 0xDD45
    0x45, 0xDD, 0, 0,
    7, 8, // weights, 2 channels, 1 element.
    // Bias, 1 remainder channel, 1 padding up to channel_subtile.
    // 64516 + 4 - (9 + 14 + 19 + 24) * 127 = 56138 = 0xDB4A
    0x4A, 0xDB, 0, 0,
    0, 0, 0, 0,
    9, 0, // weights, 1 remainder channel, 1 element.
    // Middle pass has 2 elements, channels first, then columns.
    15, 16,
    10, 11,
    17, 18,
    12, 13,
    // Middle pass, 1 remainder channel.
    19, 0,
    14, 0,
    // Last pass.
    20, 21,
    0, 0,  // padding
    22, 23,
    0, 0,  // padding
    24, 0,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_MULTIPASS_DWCONV_HWG_W, multiple_middle_pass_tile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 3;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<uint8_t> k(c * h * w);  // k = [2, 3,
                                    //      4, 5,
                                    //      6, 7,
                                    //      8, 9,
                                    //      10, 11,
                                    //      12, 13]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 96774);
  const std::vector<uint8_t> expected = {
    // First pass has 2 elements.
    // 96774 + 0 - (2 + 4 + 6 + 8 + 10 + 12) * 127 = 91440 = 0x16530
    0x30, 0x65, 0x01, 0,
    // 96774 + 1 - (3 + 5 + 7 + 9 + 11 + 13) * 127 = 90679 = 0x16237
    0x37, 0x62, 0x01, 0,
    2, 3, // 1 weight, 2 channels first, then columns
    // 2 passes of middle pass (2 elements per pass).
    8, 9,
    4, 5,
    10, 11,
    6, 7,
    // Last pass.
    12, 13,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_MULTIPASS_DWCONV_HWG_W, multiple_middle_pass_tile_channels_gt_cr) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 3;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint8_t> k(c * h * w);  // k = [5, 6, 7, 8, 9,
                                    //      10, 11, 12, 13, 14,
                                    //      15, 16, 17, 18, 19,
                                    //      20, 21, 22, 23, 24,
                                    //      25, 26, 27, 28, 29,
                                    //      30, 31, 32, 33, 34,
                                    //      ]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 96774);
  const std::vector<uint8_t> expected = {
    // First pass.
    // 96774 + 0 - (5 + 10 + 15 + 20 + 25 + 30) * 127 = 83439 = 0x145EF
    0xEF, 0x45, 0x01, 0,
    // 96774 + 1 - (6 + 11 + 16 + 21 + 26 + 31) * 127 = 82678 = 0x142F6
    0xF6, 0x42, 0x01, 0,
    5, 6, // weights, 2 channels, 2 elements.
    // 96774 + 2 - (7 + 12 + 17 + 22 + 27 + 32) * 127 = 81917 = 0x13FFD
    0xFD, 0x3F, 0x01, 0,
    // 96774 + 3 - (8 + 13 + 18 + 23 + 28 + 33) * 127 = 81156 = 0x13D04
    0x04, 0x3D, 0x01, 0,
    7, 8, // weights, 2 channels, 2 elements.
    // Bias, 1 remainder channel, 1 padding up to channel_subtile.
    // 96774 + 4 - (9 + 14 + 19 + 24 + 29 + 34) * 127 = 80395 = 0x13A0B
    0x0B, 0x3A, 0x01, 0,
    0, 0, 0, 0,
    9, 0, // weights, 1 remainder channel, 2 elements.
    // Middle pass has 2 elements, channels first, then columns.
    20, 21,
    10, 11,
    22, 23,
    12, 13,
    // 1 remainder channel.
    24, 0,
    14, 0,
    // Second middle pass, 2 elements.
    25, 26,
    15, 16,
    27, 28,
    17, 18,
    29, 0,
    // 1 remainder channel.
    19, 0,
    // Last pass.
    30, 31,
    0, 0,  // padding
    32, 33,
    0, 0,  // padding
    // 1 remainder channel.
    34, 0,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once_channel_subtile) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 4;
  const size_t channel_subtile = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint8_t> k(c * h * w);  // k = [5, 6, 7, 8, 9, // first channel
                                    //      10, 11, 12, 13, 14, // second channel
                                    //      15, 16, 17, 18, 19, // third channel
                                    //      20, 21, 22, 23, 24] // fourth channel
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<uint8_t> expected = {
    // First pass, bias 4 channels.
    // 64516 + 0 - (5 + 10 + 15 + 20) * 127 = 58166 = 0xE336
    0x36, 0xE3, 0, 0,
    // 64516 + 1 - (6 + 11 + 16 + 21) * 127 = 57659 = 0xE13B
    0x3B, 0xE1, 0, 0,
    // 64516 + 2 - (7 + 12 + 17 + 22) * 127 = 57152 = 0xDF40
    0x40, 0xDF, 0, 0,
    // 64516 + 3 - (8 + 13 + 18 + 23) * 127 = 56645 = 0xDD45
    0x45, 0xDD, 0, 0,
    5, 6, 7, 8, // first pass, 2 weights, 4 channels first, then columns
    15, 16, 17, 18,
    // Bias, 1 remainder channel, 1 padding up to channel_subtile.
    // 64516 + 4 - (9 + 14 + 19 + 24) * 127 = 56138 = 0xDB4A
    0x4A, 0xDB, 0, 0,
    0, 0, 0, 0,
    9, 0, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    19, 0,
    // No middle pass.
    10, 11, 12, 13, // last pass, 2 weights, 4 channels first.
    20, 21, 22, 23,
    0, 0, 0, 0, // padding to last_pass_tile
    14, 0, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    24, 0, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    0, 0,  // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_MULTIPASS_DWCONV_HWG_W, one_middle_pass_channel_subtile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 4;
  const size_t channel_subtile = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint8_t> k(c * h * w);  // k = [5, 6, 7, 8, 9,
                                    //      10, 11, 12, 13, 14,
                                    //      15, 16, 17, 18, 19,
                                    //      20, 21, 22, 23, 24]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<uint8_t> expected = {
    // First pass, bias 4 channels.
    // 64516 + 0 - (5 + 10 + 15 + 20) * 127 = 58166 = 0xE336
    0x36, 0xE3, 0, 0,
    // 64516 + 1 - (6 + 11 + 16 + 21) * 127 = 57659 = 0xE13B
    0x3B, 0xE1, 0, 0,
    // 64516 + 2 - (7 + 12 + 17 + 22) * 127 = 57152 = 0xDF40
    0x40, 0xDF, 0, 0,
    // 64516 + 3 - (8 + 13 + 18 + 23) * 127 = 56645 = 0xDD45
    0x45, 0xDD, 0, 0,
    5, 6, 7, 8, // first pass, 1 weight, 4 channels first, then columns
    // Bias, 1 remainder channel, 1 padding up to channel_subtile.
    // 64516 + 4 - (9 + 14 + 19 + 24) * 127 = 56138 = 0xDB4A
    0x4A, 0xDB, 0, 0,
    0, 0, 0, 0,
    9, 0, // 1 weight, 1 last channel, 1 padding up to channel_subtile
    // 1 middle pass.
    15, 16, 17, 18, // 2 weights, 4 channels first.
    10, 11, 12, 13,
    19, 0, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    14, 0,
    // Last pass.
    20, 21, 22, 23, // 1 weight, 4 channels first.
    0, 0, 0, 0, // padding to last_pass_tile
    24, 0, // 1 weight, 1 last channel, 1 padding up to channel_subtile
    0, 0,  // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once_channel_subtile_rounded) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 7;
  const size_t cr = 4;
  const size_t channel_subtile = 2;
  // c rounded to channel_subtile is 8, so we will have 2 channel_tile loops for first and middle pass.

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4, 5, 6]
  std::vector<uint8_t> k(c * h * w);  // k = [7, 8, 9, 10, 11, 12, 13,
                                    //      14, 15, 16, 17, 18, 19, 20,
                                    //      21, 22, 23, 24, 25, 26, 27,
                                    //      28, 29, 30, 31, 32, 33, 34]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<uint8_t> expected = {
    // First pass, bias, 4 channels.
    // 64516 + 0 - (7 + 14 + 21 + 28) * 127 = 55626 = 0xD94A
    0x4A, 0xD9, 0, 0,
    // 64516 + 1 - (8 + 15 + 22 + 29) * 127 = 55119 = 0xD74F
    0x4F, 0xD7, 0, 0,
    // 64516 + 2 - (9 + 16 + 23 + 30) * 127 = 54612 = 0xD554
    0x54, 0xD5, 0, 0,
    // 64516 + 3 - (10 + 17 + 24 + 31) * 127 = 54105 = 0xD359
    0x59, 0xD3, 0, 0,
    7, 8, 9, 10, // 2 weights, 4 channels first, then columns
    21, 22, 23, 24,
    // Bias, 3 remainder channels, 1 padding up to channel_subtile.
    // 64516 + 4 - (11 + 18 + 25 + 32) * 127 = 53598 = 0xD15E
    0x5E, 0xD1, 0, 0,
    // 64516 + 5 - (12 + 19 + 26 + 33) * 127 = 53091 = 0xCF63
    0x63, 0xCF, 0, 0,
    // 64516 + 6 - (13 + 20 + 27 + 34) * 127 = 52584 = 0xCD68
    0x68, 0xCD, 0, 0,
    0, 0, 0, 0,
    11, 12, 13, 0, // 2 weights, 3 remainder channel, 1 padding up to channel_subtile
    25, 26, 27, 0,
    // No middle pass.
    14, 15, 16, 17, // last pass, 2 weights, 4 channels first.
    28, 29, 30, 31,
    0, 0, 0, 0, // padding to last_pass_tile
    18, 19, // 2 weights, channel_subtile (2)
    32, 33,
    0, 0,  // padding to last_pass_tile
    20, 0, // 2 weights, 1 remainder channel, 1 padding up to channel_subtile
    34, 0,
    0, 0,  // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_MULTIPASS_DWCONV_HWG_W, one_middle_pass_channel_subtile_rounded) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 7;
  const size_t cr = 4;
  const size_t channel_subtile = 2;
  // c rounded to channel_subtile is 8, so we will have 2 channel_tile loops for first and middle pass.

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4, 5, 6]
  std::vector<uint8_t> k(c * h * w);  // k = [7, 8, 9, 10, 11, 12, 13,
                                    //      14, 15, 16, 17, 18, 19, 20,
                                    //      21, 22, 23, 24, 25, 26, 27,
                                    //      28, 29, 30, 31, 32, 33, 34]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<uint8_t> expected = {
    // First pass, bias, 4 channels.
    // 64516 + 0 - (7 + 14 + 21 + 28) * 127 = 55626 = 0xD94A
    0x4A, 0xD9, 0, 0,
    // 64516 + 1 - (8 + 15 + 22 + 29) * 127 = 55119 = 0xD74F
    0x4F, 0xD7, 0, 0,
    // 64516 + 2 - (9 + 16 + 23 + 30) * 127 = 54612 = 0xD554
    0x54, 0xD5, 0, 0,
    // 64516 + 3 - (10 + 17 + 24 + 31) * 127 = 54105 = 0xD359
    0x59, 0xD3, 0, 0,
    7, 8, 9, 10, // 1 weight, 4 channels first, then columns
    // Bias, 3 remainder channels, 1 padding up to channel_subtile.
    // 64516 + 4 - (11 + 18 + 25 + 32) * 127 = 53598 = 0xD15E
    0x5E, 0xD1, 0, 0,
    // 64516 + 5 - (12 + 19 + 26 + 33) * 127 = 53091 = 0xCF63
    0x63, 0xCF, 0, 0,
    // 64516 + 6 - (13 + 20 + 27 + 34) * 127 = 52584 = 0xCD68
    0x68, 0xCD, 0, 0,
    0, 0, 0, 0,
    11, 12, 13, 0, // 1 weight, 3 remainder channel, 1 padding up to channel_subtile
    // 1 middle pass.
    21, 22, 23, 24, // 2 weights, 4 channels first
    14, 15, 16, 17,
    25, 26, 27, 0, // 3 remainder channels first, 1 padding up to channel_tile
    18, 19, 20, 0,
    // Last pass.
    28, 29, 30, 31, // last pass, 1 weight, 4 channels first.
    0, 0, 0, 0, // padding to last_pass_tile
    32, 33, // last pass, 1 weight, channel_subtile (2)
    0, 0,  // padding to last_pass_tile
    34, 0, // last pass, 1 remainder channel, 1 padding up to channel_subtile
    0, 0,  // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QU8_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once_channel_subtile_rounded_to_channel_round) {
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

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4, 5]
  std::vector<uint8_t> k(c * h * w);  // k = [6, 7, 8, 9, 10, 11,
                                    //      12, 13, 14, 15, 16, 17,
                                    //      18, 19, 20, 21, 22, 23,
                                    //      24, 25, 26, 27, 28, 29,]
  std::iota(k.begin(), k.end(), static_cast<uint8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile, channel_round));

  xnn_qu8_packing_params params = {};
  params.input_zero_point = 127;
  params.kernel_zero_point = 127;
  xnn_pack_qu8_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);


  const int32_t bias_offset = h * w * params.input_zero_point * params.kernel_zero_point;
  ASSERT_EQ(bias_offset, 64516);
  const std::vector<uint8_t> expected = {
    // First pass, bias, 4 channels.
    // 64516 + 0 - (6 + 12 + 18 + 24) * 127 = 56896 = 0xDE40
    0x40, 0xDE, 0, 0,
    // 64516 + 1 - (7 + 13 + 19 + 25) * 127 = 56389 = 0xDC45
    0x45, 0xDC, 0, 0,
    // 64516 + 2 - (8 + 14 + 20 + 26) * 127 = 55882 = 0xDA4A
    0x4A, 0xDA, 0, 0,
    // 64516 + 3 - (9 + 15 + 21 + 27) * 127 = 55375 = 0xD84F
    0x4F, 0xD8, 0, 0,
    6, 7, 8, 9, // 2 weights, 4 channels first, then columns
    18, 19, 20, 21,
    // Bias, 2 remainder channels, 2 padding up to channel_subtile
    // 64516 + 4 - (10 + 16 + 22 + 28) * 127 = 54868 = 0xD654
    0x54, 0xD6, 0, 0,
    // 64516 + 5 - (11 + 17 + 23 + 29) * 127 = 54361 = 0xD459
    0x59, 0xD4, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    10, 11, 0, 0, // 2 weights, 2 remainder channel, 1 padding up to channel_subtile
    22, 23, 0, 0,
    // No middle pass.
    12, 13, 14, 15, // last pass, 2 weights, 4 channels first.
    24, 25, 26, 27,
    0, 0, 0, 0, // padding to last_pass_tile
    16, 17, 0, 0, // 2 weights, channel_subtile (4)
    28, 29, 0, 0,
    0, 0, 0, 0, // padding to last_pass_tile
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

TEST(PACK_QS8_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<int8_t> k(c * h * w);  // k = [2, 3, 4, 5, 6, 7, 8, 9]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));

  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // First pass, bias first.
    // 0 - (2 + 3 + 4 + 5) * 127 = -1778 = 0xFFFFF90E
    0x0E, 0xF9, 0xFF, 0xFF,
    // 1 - (6 + 7 + 8 + 9) * 127 = -3809 = 0xFFFFF11F
    0x1F, 0xF1, 0xFF, 0xFF,
    2, 6,  // 2 weights, channels first, then columns.
    4, 8,
    3, 7,  // Last pass, 2 weights.
    5, 9,
    0, 0,  // Padding to last_pass_tile.
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once_channels_gt_cr) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<int8_t> k(c * h * w);  // k = [5, 6, 7, 8, // first 2x2 kernel
                                    //      9, 10, 11, 12, // second 2x2 kernel
                                    //      13, 14, 15, 16, // third 2x2 kernel
                                    //      17, 18, 19, 20, // fourth 2x2 kernel
                                    //      21, 22, 23, 24, // fifth 2x2 kernel
                                    //      ]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // First pass, bias first.
    // 0 - (5 + 6 + 7 + 8) * 127 = -3302 = 0xFFFFF31A
    0x1A, 0xF3, 0xFF, 0xFF, // bias
    // 1 - (9 + 10 + 11 + 12) * 127 = -5333 = 0xFFFFEB2B
    0x2B, 0xEB, 0xFF, 0xFF,
    5, 9, // 2 weights, 2 channels first, then columns
    7, 11,
    // Bias.
    // 2 - (13 + 14 + 15 + 16) * 127 = -7364 = 0xFFFFE33C
    0x3C, 0xE3, 0xFF, 0xFF,
    // 3 - (17 + 18 + 19 + 20) * 127 = -9395 = 0xFFFFDB4D
    0x4D, 0xDB, 0xFF, 0xFF,
    13, 17, // 2 weights, 2 channels first, then columns
    15, 19,
    // 4 - (21 + 22 + 23 + 24) * 127 = -11426 = 0xFFFFD35E
    0x5E, 0xD3, 0xFF, 0xFF,
    0, 0, 0, 0,
    21, 0, // 2 weights, 1 remainder channels first, then columns
    23, 0,
    // No middle pass.
    6, 10, // last pass, 2 weights, 2 channels first, then columns
    8, 12,
    0, 0,  // padding
    14, 18,
    16, 20,
    0, 0,  // padding
    22, 0,
    24, 0,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_MULTIPASS_DWCONV_GHW_W, one_middle_pass_tile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 1;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<int8_t> k(c * h * w);  // k = [2, 3, // first 2x2 kernel
                                    //      4, 5,
                                    //      6, 7, // second 2x2 kernel
                                    //      8, 9]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // First pass only has 1 element.
    // 0 - (2 + 3 + 4 + 5) * 127 = -1778 = 0xFFFFF90E
    0x0E, 0xF9, 0xFF, 0xFF,
    // 1 - (6 + 7 + 8 + 9) * 127 = -3809 = 0xFFFFF11F
    0x1F, 0xF1, 0xFF, 0xFF,
    2, 6, // weights, 2 channels, 1 element.
    // Middle pass has 2 elements, columns first.
    4, 8,
    3, 7,
    // Last pass.
    5, 9,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_MULTIPASS_DWCONV_GHW_W, one_middle_pass_tile_channels_gt_cr) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<int8_t> k(c * h * w);  // k = [5, 6, // first 2x2 kernel
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
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // First pass only has 1 element, bias first.
    // 0 - (5 + 6 + 7 + 8) * 127 = -3302 = 0xFFFFF31A
    0x1A, 0xF3, 0xFF, 0xFF, // bias
    // 1 - (9 + 10 + 11 + 12) * 127 = -5333 = 0xFFFFEB2B
    0x2B, 0xEB, 0xFF, 0xFF,
    5, 9, // weights, 2 channels, 1 element.
    // Bias.
    // 2 - (13 + 14 + 15 + 16) * 127 = -7364 = 0xFFFFE33C
    0x3C, 0xE3, 0xFF, 0xFF,
    // 3 - (17 + 18 + 19 + 20) * 127 = -9395 = 0xFFFFDB4D
    0x4D, 0xDB, 0xFF, 0xFF,
    13, 17, // weights, 2 channels, 1 element.
    // Bias.
    // 4 - (21 + 22 + 23 + 24) * 127 = -11426 = 0xFFFFD35E
    0x5E, 0xD3, 0xFF, 0xFF,
    0, 0, 0, 0,
    21, 0, // weights, 1 remainder channel, 1 element.
    // Middle pass has 2 elements, channels first, then columns.
    7, 11,
    6, 10,
    15, 19,
    14, 18,
    // Middle pass, 1 remainder channel.
    23, 0,
    22, 0,
    // Last pass,
    8, 12,
    0, 0,  // padding
    16, 20,
    0, 0,  // padding
    24, 0,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_MULTIPASS_DWCONV_GHW_W, multiple_middle_pass_tile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 3;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<int8_t> k(c * h * w);  // k = [2, 3, 4, // first 2x3 kernel
                                    //      5, 6, 7,
                                    //      8, 9, 10, // second 2x3 kernel
                                    //      11, 12, 13]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // First pass has 2 elements.
    // 0 - (2 + 3 + 4 + 5 + 6 + 7) * 127 = -3429 = 0xFFFFF29B
    0x9B, 0xF2, 0xFF, 0xFF,
    // 1 - (8 + 9 + 10 + 11 + 12 + 13) * 127 = -8000 = 0xFFFFE0C0
    0xC0, 0xE0, 0xFF, 0xFF,
    2, 8, // 1 weight, 2 channels first, then columns
    // Middle pass 1 (2 elements per pass).
    5, 11,
    3, 9,
    // Middle pass 2 (2 elements per pass).
    6, 12,
    4, 10,
    // Last pass.
    7, 13,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_MULTIPASS_DWCONV_GHW_W, multiple_middle_pass_tile_channels_gt_cr) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 3;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<int8_t> k(c * h * w);  // k = [5, 6, 7, // first 2x3 kernel
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
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);


  const std::vector<uint8_t> expected = {
    // First pass only has 1 element, bias first.
    // 0 - (5 + 6 + 7 + 8 + 9 + 10) * 127 = -5715 = 0xFFFFE9AD
    0xAD, 0xE9, 0xFF, 0xFF,
    // 1 - (11 + 12 + 13 + 14 + 15 + 16) * 127 = -10286 = 0xFFFFD7D2
    0xD2, 0xD7, 0xFF, 0xFF,
    5, 11, // 1 weight, 2 channels, 2 elements.
    // 2 - (17 + 18 + 19 + 20 + 21 + 22) * 127 = -14857 = 0xFFFFC5F7
    0xF7, 0xC5, 0xFF, 0xFF,
    // 3 - (23 + 24 + 25 + 26 + 27 + 28) * 127 = -19428 = 0xFFFFB41C
    0x1C, 0xB4, 0xFF, 0xFF,
    17, 23, // 1 weight, 2 channels, 2 elements.
    // 4 - (29 + 30 + 31 + 32 + 33 + 34) * 127 = -23999 = 0xFFFFA241
    0x41, 0xA2, 0xFF, 0xFF,
    0, 0, 0, 0,
    29, 0, // 1 weight, 1 remainder channel, 2 elements.
    // Middle pass has 2 elements, channels first, then columns.
    8, 14,
    6, 12,
    20, 26,
    18, 24,
    // 1 remainder channel.
    32, 0,
    30, 0,
    // Second middle pass, 2 elements.
    9, 15,
    7, 13,
    21, 27,
    19, 25,
    // 1 remainder channel.
    33, 0,
    31, 0,
    // Last pass
    10, 16,
    0, 0,  // padding
    22, 28,
    0, 0,  // padding
    // 1 remainder channel.
    34, 0,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once_channel_subtile) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 4;
  const size_t channel_subtile = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<int8_t> k(c * h * w);  // k = [5, 6, // first 2x2 kernel
                                    //      7, 8,
                                    //      9, 10, // second 2x2 kernel
                                    //      11, 12,
                                    //      13, 14, // third 2x2 kernel
                                    //      15, 16,
                                    //      17, 18, // fourth 2x2 kernel
                                    //      19, 20,
                                    //      21, 22, // fifth 2x2 kernel
                                    //      23, 24 ]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // First pass.
    // 0 - (5 + 6 + 7 + 8) * 127 = -3302 = 0xFFFFF31A
    0x1A, 0xF3, 0xFF, 0xFF, // bias
    // 1 - (9 + 10 + 11 + 12) * 127 = -5333 = 0xFFFFEB2B
    0x2B, 0xEB, 0xFF, 0xFF,
    // 2 - (13 + 14 + 15 + 16) * 127 = -7364 = 0xFFFFE33C
    0x3C, 0xE3, 0xFF, 0xFF,
    // 3 - (17 + 18 + 19 + 20) * 127 = -9395 = 0xFFFFDB4D
    0x4D, 0xDB, 0xFF, 0xFF,
    5, 9, 13, 17,  // 2 weights, 4 channels first, then columns
    7, 11, 15, 19,
    // Bias, 1 last channel, 1 padding up to channel_subtile
    // 4 - (21 + 22 + 23 + 24) * 127 = -11426 = 0xFFFFD35E
    0x5E, 0xD3, 0xFF, 0xFF,
    0, 0, 0, 0,
    21, 0, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    23, 0,
    // No middle pass.
    6, 10, 14, 18,  // last pass, 2 weights, 4 channels first
    8, 12, 16, 20,
    0, 0, 0, 0, // padding to last_pass_tile
    22, 0, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    24, 0,
    0, 0, // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_MULTIPASS_DWCONV_GHW_W, one_middle_pass_channel_subtile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 4;
  const size_t channel_subtile = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<int8_t> k(c * h * w);  // k = [5, 6, // first 2x2 kernel
                                    //      7, 8,
                                    //      9, 10, // second 2x2 kernel
                                    //      11, 12,
                                    //      13, 14, // third 2x2 kernel
                                    //      15, 16,
                                    //      17, 18, // fourth 2x2 kernel
                                    //      19, 20,
                                    //      21, 22, // fifth 2x2 kernel
                                    //      23, 24 ]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // First pass, bias, 4 channels.
    // 0 - (5 + 6 + 7 + 8) * 127 = -3302 = 0xFFFFF31A
    0x1A, 0xF3, 0xFF, 0xFF, // bias
    // 1 - (9 + 10 + 11 + 12) * 127 = -5333 = 0xFFFFEB2B
    0x2B, 0xEB, 0xFF, 0xFF,
    // 2 - (13 + 14 + 15 + 16) * 127 = -7364 = 0xFFFFE33C
    0x3C, 0xE3, 0xFF, 0xFF,
    // 3 - (17 + 18 + 19 + 20) * 127 = -9395 = 0xFFFFDB4D
    0x4D, 0xDB, 0xFF, 0xFF,
    5, 9, 13, 17,  // 1 weight, 4 channels first, then columns
    // Bias, 1 last channel, 1 padding up to channel_subtile.
    // 4 - (21 + 22 + 23 + 24) * 127 = -11426 = 0xFFFFD35E
    0x5E, 0xD3, 0xFF, 0xFF,
    0, 0, 0, 0,
    21, 0, // 1 weight, 1 last channel, 1 padding up to channel_subtile
    // 1 middle pass
    7, 11, 15, 19, // 2 weights, 4 channels first, then columns
    6, 10, 14, 18,
    23, 0, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    22, 0,
    // Last pass.
    8, 12, 16, 20,  // 1 weight, 4 channels first
    0, 0, 0, 0, // padding to last_pass_tile
    24, 0, // 1 weight, 1 last channel, 1 padding up to channel_subtile
    0, 0, // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once_channel_subtile_rounded) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 7;
  const size_t cr = 4;
  const size_t channel_subtile = 2;
  // c rounded to channel_subtile is 8, so we will have 2 channel_tile loops in the first and middle pass.

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4, 5, 6]
  std::vector<int8_t> k(c * h * w);  // k = [7, 8, // first 2x2 kernel
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
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // First pass, bias, 4 channels.
    // 0 - (7 + 8 + 9 + 10) * 127 = -4318 = 0xFFFFEF22
    0x22, 0xEF, 0xFF, 0xFF,
    // 1 - (11 + 12 + 13 + 14) * 127 = -6349 = 0xFFFFE733
    0x33, 0xE7, 0xFF, 0xFF,
    // 2 - (15 + 16 + 17 + 18) * 127 = -8380 = 0xFFFFDF44
    0x44, 0xDF, 0xFF, 0xFF,
    // 3 - (19 + 20 + 21 + 22) * 127 = -10411 = 0xFFFFD755
    0x55, 0xD7, 0xFF, 0xFF,
    7, 11, 15, 19,  // 2 weights, 4 channels first, then columns
    9, 13, 17, 21,
    // Bias, 3 remainder channels, 1 padding up to channel_Tile.
    // 4 - (23 + 24 + 25 + 26) * 127 = -12442 = 0xFFFFCF66
    0x66, 0xCF, 0xFF, 0xFF,
    // 5 - (27 + 28 + 29 + 30) * 127 = -14473 = 0xFFFFC777
    0x77, 0xC7, 0xFF, 0xFF,
    // 6 - (31 + 32 + 33 + 34) * 127 = -16504 = 0xFFFFBF88
    0x88, 0xBF, 0xFF, 0xFF,
    0, 0, 0, 0,
    23, 27, 31, 0,  // 2 weights, 3 remainder channels, 1 padding up to channel_tile
    25, 29, 33, 0,
    // No middle pass.
    // Last pass.
    8, 12, 16, 20,  // last pass, 2 weights, 4 channels first
    10, 14, 18, 22,
    0, 0, 0, 0, // padding to last_pass_tile
    24, 28, // last pass, 2 weights, channel_subtile (2)
    26, 30,
    0, 0, // padding to last_pass_tile
    32, 0, // 1 remainder channel, 1 padding up to channel_subtile
    34, 0, // 1 remainder channel, 1 padding up to channel_subtile
    0, 0, // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once_channel_subtile_rounded_to_channel_round) {
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

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4, 5]
  std::vector<int8_t> k(c * h * w);  // k = [6, 7, 8, 9,
                                    //      10, 11, 12, 13
                                    //      14, 15, 16, 17,
                                    //      18, 19, 20, 21,
                                    //      22, 23, 24, 25,
                                    //      26, 27, 28, 29,]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile, channel_round));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // First pass, bias, 4 channels.
    // 0 - (6 + 7 + 8 + 9) * 127 = -3810 = 0xFFFFF11E
    0x1E, 0xF1, 0xFF, 0xFF,
    // 1 - (10 + 11 + 12 + 13) * 127 = -5841 = 0xFFFFE92F
    0x2F, 0xE9, 0xFF, 0xFF,
    // 2 - (14 + 15 + 16 + 17) * 127 = -7872 = 0xFFFFE140
    0x40, 0xE1, 0xFF, 0xFF,
    // 3 - (18 + 19 + 20 + 21) * 127 = -9903 = 0xFFFFD951
    0x51, 0xD9, 0xFF, 0xFF,
    6, 10, 14, 18, // 2 weights, 4 channels first, then columns
    8, 12, 16, 20,

    // Bias, 2 remainder channels, 2 padding up to channel_subtile.
    // 4 - (22 + 23 + 24 + 25) * 127 = -11934 = 0xFFFFD162
    0x62, 0xD1, 0xFF, 0xFF,
    // 5 - (26 + 27 + 28 + 29) * 127 = -13965 = 0xFFFFC973
    0x73, 0xC9, 0xFF, 0xFF,
    0, 0, 0, 0,
    0, 0, 0, 0,
    22, 26, 0, 0, // 2 weights, 2 remainder channel, 2 padding up to channel_subtile
    24, 28, 0, 0,
    // No middle pass.
    7, 11, 15, 19, // last pass, 2 weights, 4 channels first.
    9, 13, 17, 21,
    0, 0, 0, 0, // padding to last_pass_tile
    23, 27, 0, 0, // 2 weights, channel_subtile (4)
    25, 29, 0, 0,
    0, 0, 0, 0, // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_MULTIPASS_DWCONV_GHW_W, one_middle_pass_channel_subtile_rounded) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 7;
  const size_t cr = 4;
  const size_t channel_subtile = 2;
  // c rounded to channel_subtile is 8, so we will have 2 channel_tile loops in first and middle pass.

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1, 2, 3, 4, 5, 6]
  std::vector<int8_t> k(c * h * w);  // k = [7, 8, // first 2x2 kernel
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
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // First pass, bias, 4 channels.
    // 0 - (7 + 8 + 9 + 10) * 127 = -4318 = 0xFFFFEF22
    0x22, 0xEF, 0xFF, 0xFF,
    // 1 - (11 + 12 + 13 + 14) * 127 = -6349 = 0xFFFFE733
    0x33, 0xE7, 0xFF, 0xFF,
    // 2 - (15 + 16 + 17 + 18) * 127 = -8380 = 0xFFFFDF44
    0x44, 0xDF, 0xFF, 0xFF,
    // 3 - (19 + 20 + 21 + 22) * 127 = -10411 = 0xFFFFD755
    0x55, 0xD7, 0xFF, 0xFF,
    7, 11, 15, 19,  // 1 weight, 4 channels first, then columns
    // Bias, 3 remainder channels, 1 padding up to channel_Tile.
    // 4 - (23 + 24 + 25 + 26) * 127 = -12442 = 0xFFFFCF66
    0x66, 0xCF, 0xFF, 0xFF,
    // 5 - (27 + 28 + 29 + 30) * 127 = -14473 = 0xFFFFC777
    0x77, 0xC7, 0xFF, 0xFF,
    // 6 - (31 + 32 + 33 + 34) * 127 = -16504 = 0xFFFFBF88
    0x88, 0xBF, 0xFF, 0xFF,
    0, 0, 0, 0,
    23, 27, 31, 0,  // 1 weight, 3 remainder channels, 1 padding up to channel_tile
    // 1 middle pass.
    9, 13, 17, 21, // 2 weights, 4 channels first
    8, 12, 16, 20,
    25, 29, 33, 0, // 3 remainder channels, 1 padding up to channel_tile
    24, 28, 32, 0,
    // Last pass.
    10, 14, 18, 22,  // last pass, 1 weight, 4 channels first
    0, 0, 0, 0, // padding to last_pass_tile
    26, 30, // 1 weight, channel_subtile,
    0, 0, // padding to last_pass_tile
    34, 0, // 1 weight, 1 remainder channel, 1 padding up to channel_subtile
    0, 0, // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<int8_t> k(c * h * w);  // k = [2, 3,
                                    //      4, 5,
                                    //      6, 7,
                                    //      8, 9]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // First pass, bias first.
    // 0 - (2 + 4 + 6 + 8) * 127 = -2540 = 0xFFFFF614
    0x14, 0xF6, 0xFF, 0xFF,
    // 1 - (3 + 5 + 7 + 9) * 127 = -3047 = 0xFFFFF419
    0x19, 0xF4, 0xFF, 0xFF,
    2, 3, // First pass, 2 weights, channels first, then columns
    6, 7,
    // No middle pass.
    4, 5, // Last pass, 2 weights
    8, 9,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once_channels_gt_cr) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<int8_t> k(c * h * w);  // k = [5, 6, 7, 8, 9,
                                    //      10, 11, 12, 13, 14,
                                    //      15, 16, 17, 18, 19,
                                    //      20, 21, 22, 23, 24,
                                    //      ]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // First pass.
    // 0 - (5 + 10 + 15 + 20) * 127 = -6350 = 0xFFFFE732
    0x32, 0xE7, 0xFF, 0xFF,
    // 1 - (6 + 11 + 16 + 21) * 127 = -6857 = 0xFFFFE537
    0x37, 0xE5, 0xFF, 0xFF,
    5, 6, // 2 weights, 2 channels first, then columns
    15, 16,
    // 2 - (7 + 12 + 17 + 22) * 127 = -7364 = 0xFFFFE33C
    0x3C, 0xE3, 0xFF, 0xFF,
    // 3 - (8 + 13 + 18 + 23) * 127 = -7871 = 0xFFFFE141
    0x41, 0xE1, 0xFF, 0xFF,
    7, 8, // 2 weights, 2 channels first, then columns
    17, 18,
    // Bias, 1 remainder channel, 1 padding up to channel_subtile.
    // 4 - (9 + 14 + 19 + 24) * 127 = -8378 = 0xFFFFDF46
    0x46, 0xDF, 0xFF, 0xFF,
    0, 0, 0, 0,
    9, 0, // weights, 1 remainder channels first, then columns
    19, 0,
    // No middle pass.
    // Last pass, 2 weights
    10, 11,
    20, 21,
    0, 0,  // padding
    12, 13,
    22, 23,
    0, 0,  // padding
    14, 0,
    24, 0,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_MULTIPASS_DWCONV_HWG_W, one_middle_pass_tile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 1;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0.0f);  // b = [0, 1]
  std::vector<int8_t> k(c * h * w);  // k = [2, 3,
                                    //      4, 5,
                                    //      6, 7,
                                    //      8, 9]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // First pass only has 1 element.
    // 0 - (2 + 4 + 6 + 8) * 127 = -2540 = 0xFFFFF614
    0x14, 0xF6, 0xFF, 0xFF,
    // 1 - (3 + 5 + 7 + 9) * 127 = -3047 = 0xFFFFF419
    0x19, 0xF4, 0xFF, 0xFF,
    2, 3, // weights, 2 channels, 1 element.
    // Middle pass has 2 elements, columns first.
    6, 7,
    4, 5,
    // Last pass.
    8, 9,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_MULTIPASS_DWCONV_HWG_W, one_middle_pass_tile_channels_gt_cr) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<int8_t> k(c * h * w);  // k = [5, 6, 7, 8, 9,
                                    //      10, 11, 12, 13, 14,
                                    //      15, 16, 17, 18, 19,
                                    //      20, 21, 22, 23, 24,
                                    //      ]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // First pass only has 1 element.
    // 0 - (5 + 10 + 15 + 20) * 127 = -6350 = 0xFFFFE732
    0x32, 0xE7, 0xFF, 0xFF,
    // 1 - (6 + 11 + 16 + 21) * 127 = -6857 = 0xFFFFE537
    0x37, 0xE5, 0xFF, 0xFF,
    5, 6, // weights, 2 channels, 1 element.
    // 2 - (7 + 12 + 17 + 22) * 127 = -7364 = 0xFFFFE33C
    0x3C, 0xE3, 0xFF, 0xFF,
    // 3 - (8 + 13 + 18 + 23) * 127 = -7871 = 0xFFFFE141
    0x41, 0xE1, 0xFF, 0xFF,
    7, 8, // weights, 2 channels, 1 element.
    // Bias, 1 remainder channel, 1 padding up to channel_subtile.
    // 4 - (9 + 14 + 19 + 24) * 127 = -8378 = 0xFFFFDF46
    0x46, 0xDF, 0xFF, 0xFF,
    0, 0, 0, 0,
    9, 0, // weights, 1 remainder channel, 1 element.
    // Middle pass has 2 elements, channels first, then columns.
    15, 16,
    10, 11,
    17, 18,
    12, 13,
    // Middle pass, 1 remainder channel.
    19, 0,
    14, 0,
    // Last pass.
    20, 21,
    0, 0,  // padding
    22, 23,
    0, 0,  // padding
    24, 0,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_MULTIPASS_DWCONV_HWG_W, multiple_middle_pass_tile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 3;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<int8_t> k(c * h * w);  // k = [2, 3,
                                    //      4, 5,
                                    //      6, 7,
                                    //      8, 9,
                                    //      10, 11,
                                    //      12, 13]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // First pass has 2 elements.
    // 0 - (2 + 4 + 6 + 8 + 10 + 12) * 127 = -5334 = 0xFFFFEB2A
    0x2A, 0xEB, 0xFF, 0xFF,
    // 1 - (3 + 5 + 7 + 9 + 11 + 13) * 127 = -6095 = 0xFFFFE831
    0x31, 0xE8, 0xFF, 0xFF,
    2, 3, // 1 weight, 2 channels first, then columns
    // 2 passes of middle pass (2 elements per pass).
    8, 9,
    4, 5,
    10, 11,
    6, 7,
    // Last pass.
    12, 13,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_MULTIPASS_DWCONV_HWG_W, multiple_middle_pass_tile_channels_gt_cr) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 3;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<int8_t> k(c * h * w);  // k = [5, 6, 7, 8, 9,
                                    //      10, 11, 12, 13, 14,
                                    //      15, 16, 17, 18, 19,
                                    //      20, 21, 22, 23, 24,
                                    //      25, 26, 27, 28, 29,
                                    //      30, 31, 32, 33, 34,
                                    //      ]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, cr));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // First pass.
    // 0 - (5 + 10 + 15 + 20 + 25 + 30) * 127 = -13335 = 0xFFFFCBE9
    0xE9, 0xCB, 0xFF, 0xFF,
    // 1 - (6 + 11 + 16 + 21 + 26 + 31) * 127 = -14096 = 0xFFFFC8F0
    0xF0, 0xC8, 0xFF, 0xFF,
    5, 6, // weights, 2 channels, 2 elements.
    // 2 - (7 + 12 + 17 + 22 + 27 + 32) * 127 = -14857 = 0xFFFFC5F7
    0xF7, 0xC5, 0xFF, 0xFF,
    // 3 - (8 + 13 + 18 + 23 + 28 + 33) * 127 = -15618 = 0xFFFFC2FE
    0xFE, 0xC2, 0xFF, 0xFF,
    7, 8, // weights, 2 channels, 2 elements.
    // Bias, 1 remainder channel, 1 padding up to channel_subtile.
    // 4 - (9 + 14 + 19 + 24 + 29 + 34) * 127 = -16379 = 0xFFFFC005
    0x05, 0xC0, 0xFF, 0xFF,
    0, 0, 0, 0,
    9, 0, // weights, 1 remainder channel, 2 elements.
    // Middle pass has 2 elements, channels first, then columns.
    20, 21,
    10, 11,
    22, 23,
    12, 13,
    // 1 remainder channel.
    24, 0,
    14, 0,
    // Second middle pass, 2 elements.
    25, 26,
    15, 16,
    27, 28,
    17, 18,
    29, 0,
    // 1 remainder channel.
    19, 0,
    // Last pass.
    30, 31,
    0, 0,  // padding
    32, 33,
    0, 0,  // padding
    // 1 remainder channel.
    34, 0,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once_channel_subtile) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 4;
  const size_t channel_subtile = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<int8_t> k(c * h * w);  // k = [5, 6, 7, 8, 9, // first channel
                                    //      10, 11, 12, 13, 14, // second channel
                                    //      15, 16, 17, 18, 19, // third channel
                                    //      20, 21, 22, 23, 24] // fourth channel
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // First pass, bias 4 channels.
    // 0 - (5 + 10 + 15 + 20) * 127 = -6350 = 0xFFFFE732
    0x32, 0xE7, 0xFF, 0xFF,
    // 1 - (6 + 11 + 16 + 21) * 127 = -6857 = 0xFFFFE537
    0x37, 0xE5, 0xFF, 0xFF,
    // 2 - (7 + 12 + 17 + 22) * 127 = -7364 = 0xFFFFE33C
    0x3C, 0xE3, 0xFF, 0xFF,
    // 3 - (8 + 13 + 18 + 23) * 127 = -7871 = 0xFFFFE141
    0x41, 0xE1, 0xFF, 0xFF,
    5, 6, 7, 8, // first pass, 2 weights, 4 channels first, then columns
    15, 16, 17, 18,
    // Bias, 1 remainder channel, 1 padding up to channel_subtile.
    // 4 - (9 + 14 + 19 + 24) * 127 = -8378 = 0xFFFFDF46
    0x46, 0xDF, 0xFF, 0xFF,
    0, 0, 0, 0,
    9, 0, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    19, 0,
    // No middle pass.
    10, 11, 12, 13, // last pass, 2 weights, 4 channels first.
    20, 21, 22, 23,
    0, 0, 0, 0, // padding to last_pass_tile
    14, 0, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    24, 0, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    0, 0,  // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_MULTIPASS_DWCONV_HWG_W, one_middle_pass_channel_subtile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 4;
  const size_t channel_subtile = 2;

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<int8_t> k(c * h * w);  // k = [5, 6, 7, 8, 9,
                                    //      10, 11, 12, 13, 14,
                                    //      15, 16, 17, 18, 19,
                                    //      20, 21, 22, 23, 24]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // First pass, bias 4 channels.
    // 0 - (5 + 10 + 15 + 20) * 127 = -6350 = 0xFFFFE732
    0x32, 0xE7, 0xFF, 0xFF,
    // 1 - (6 + 11 + 16 + 21) * 127 = -6857 = 0xFFFFE537
    0x37, 0xE5, 0xFF, 0xFF,
    // 2 - (7 + 12 + 17 + 22) * 127 = -7364 = 0xFFFFE33C
    0x3C, 0xE3, 0xFF, 0xFF,
    // 3 - (8 + 13 + 18 + 23) * 127 = -7871 = 0xFFFFE141
    0x41, 0xE1, 0xFF, 0xFF,
    5, 6, 7, 8, // first pass, 1 weight, 4 channels first, then columns
    // Bias, 1 remainder channel, 1 padding up to channel_subtile.
    // 4 - (9 + 14 + 19 + 24) * 127 = -8378 = 0xFFFFDF46
    0x46, 0xDF, 0xFF, 0xFF,
    0, 0, 0, 0,
    9, 0, // 1 weight, 1 last channel, 1 padding up to channel_subtile
    // 1 middle pass.
    15, 16, 17, 18, // 2 weights, 4 channels first.
    10, 11, 12, 13,
    19, 0, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    14, 0,
    // Last pass.
    20, 21, 22, 23, // 1 weight, 4 channels first.
    0, 0, 0, 0, // padding to last_pass_tile
    24, 0, // 1 weight, 1 last channel, 1 padding up to channel_subtile
    0, 0,  // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once_channel_subtile_rounded) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 7;
  const size_t cr = 4;
  const size_t channel_subtile = 2;
  // c rounded to channel_subtile is 8, so we will have 2 channel_tile loops for first and middle pass.

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4, 5, 6]
  std::vector<int8_t> k(c * h * w);  // k = [7, 8, 9, 10, 11, 12, 13,
                                    //      14, 15, 16, 17, 18, 19, 20,
                                    //      21, 22, 23, 24, 25, 26, 27,
                                    //      28, 29, 30, 31, 32, 33, 34]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // First pass, bias, 4 channels.
    // 0 - (7 + 14 + 21 + 28) * 127 = -8890 = 0xFFFFDD46
    0x46, 0xDD, 0xFF, 0xFF,
    // 1 - (8 + 15 + 22 + 29) * 127 = -9397 = 0xFFFFDB4B
    0x4B, 0xDB, 0xFF, 0xFF,
    // 2 - (9 + 16 + 23 + 30) * 127 = -9904 = 0xFFFFD950
    0x50, 0xD9, 0xFF, 0xFF,
    // 3 - (10 + 17 + 24 + 31) * 127 = -10411 = 0xFFFFD755
    0x55, 0xD7, 0xFF, 0xFF,
    7, 8, 9, 10, // 2 weights, 4 channels first, then columns
    21, 22, 23, 24,
    // Bias, 3 remainder channels, 1 padding up to channel_subtile.
    // 4 - (11 + 18 + 25 + 32) * 127 = -10918 = 0xFFFFD55A
    0x5A, 0xD5, 0xFF, 0xFF,
    // 5 - (12 + 19 + 26 + 33) * 127 = -11425 = 0xFFFFD35F
    0x5F, 0xD3, 0xFF, 0xFF,
    // 6 - (13 + 20 + 27 + 34) * 127 = -11932 = 0xFFFFD164
    0x64, 0xD1, 0xFF, 0xFF,
    0, 0, 0, 0,
    11, 12, 13, 0, // 2 weights, 3 remainder channel, 1 padding up to channel_subtile
    25, 26, 27, 0,
    // No middle pass.
    14, 15, 16, 17, // last pass, 2 weights, 4 channels first.
    28, 29, 30, 31,
    0, 0, 0, 0, // padding to last_pass_tile
    18, 19, // 2 weights, channel_subtile (2)
    32, 33,
    0, 0,  // padding to last_pass_tile
    20, 0, // 2 weights, 1 remainder channel, 1 padding up to channel_subtile
    34, 0,
    0, 0,  // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_MULTIPASS_DWCONV_HWG_W, one_middle_pass_channel_subtile_rounded) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 7;
  const size_t cr = 4;
  const size_t channel_subtile = 2;
  // c rounded to channel_subtile is 8, so we will have 2 channel_tile loops for first and middle pass.

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4, 5, 6]
  std::vector<int8_t> k(c * h * w);  // k = [7, 8, 9, 10, 11, 12, 13,
                                    //      14, 15, 16, 17, 18, 19, 20,
                                    //      21, 22, 23, 24, 25, 26, 27,
                                    //      28, 29, 30, 31, 32, 33, 34]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);

  const std::vector<uint8_t> expected = {
    // First pass, bias, 4 channels.
    // 0 - (7 + 14 + 21 + 28) * 127 = -8890 = 0xFFFFDD46
    0x46, 0xDD, 0xFF, 0xFF,
    // 1 - (8 + 15 + 22 + 29) * 127 = -9397 = 0xFFFFDB4B
    0x4B, 0xDB, 0xFF, 0xFF,
    // 2 - (9 + 16 + 23 + 30) * 127 = -9904 = 0xFFFFD950
    0x50, 0xD9, 0xFF, 0xFF,
    // 3 - (10 + 17 + 24 + 31) * 127 = -10411 = 0xFFFFD755
    0x55, 0xD7, 0xFF, 0xFF,
    7, 8, 9, 10, // 1 weight, 4 channels first, then columns
    // Bias, 3 remainder channels, 1 padding up to channel_subtile.
    // 4 - (11 + 18 + 25 + 32) * 127 = -10918 = 0xFFFFD55A
    0x5A, 0xD5, 0xFF, 0xFF,
    // 5 - (12 + 19 + 26 + 33) * 127 = -11425 = 0xFFFFD35F
    0x5F, 0xD3, 0xFF, 0xFF,
    // 6 - (13 + 20 + 27 + 34) * 127 = -11932 = 0xFFFFD164
    0x64, 0xD1, 0xFF, 0xFF,
    0, 0, 0, 0,
    11, 12, 13, 0, // 1 weight, 3 remainder channel, 1 padding up to channel_subtile
    // 1 middle pass.
    21, 22, 23, 24, // 2 weights, 4 channels first
    14, 15, 16, 17,
    25, 26, 27, 0, // 3 remainder channels first, 1 padding up to channel_tile
    18, 19, 20, 0,
    // Last pass.
    28, 29, 30, 31, // last pass, 1 weight, 4 channels first.
    0, 0, 0, 0, // padding to last_pass_tile
    32, 33, // last pass, 1 weight, channel_subtile (2)
    0, 0,  // padding to last_pass_tile
    34, 0, // last pass, 1 remainder channel, 1 padding up to channel_subtile
    0, 0,  // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_QS8_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once_channel_subtile_rounded_to_channel_round) {
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

  std::vector<int32_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4, 5]
  std::vector<int8_t> k(c * h * w);  // k = [6, 7, 8, 9, 10, 11,
                                    //      12, 13, 14, 15, 16, 17,
                                    //      18, 19, 20, 21, 22, 23,
                                    //      24, 25, 26, 27, 28, 29,]
  std::iota(k.begin(), k.end(), static_cast<int8_t>(b.size()));
  std::vector<uint8_t> packed_weights(
    qs8_multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile, channel_round));

  xnn_qs8_packing_params params = {};
  params.input_zero_point = 127;
  xnn_pack_qs8_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      &params);


  const std::vector<uint8_t> expected = {
    // First pass, bias, 4 channels.
    // 0 - (6 + 12 + 18 + 24) * 127 = -7620 = 0xFFFFE23C
    0x3C, 0xE2, 0xFF, 0xFF,
    // 1 - (7 + 13 + 19 + 25) * 127 = -8127 = 0xFFFFE041
    0x41, 0xE0, 0xFF, 0xFF,
    // 2 - (8 + 14 + 20 + 26) * 127 = -8634 = 0xFFFFDE46
    0x46, 0xDE, 0xFF, 0xFF,
    // 3 - (9 + 15 + 21 + 27) * 127 = -9141 = 0xFFFFDC4B
    0x4B, 0xDC, 0xFF, 0xFF,
    6, 7, 8, 9, // 2 weights, 4 channels first, then columns
    18, 19, 20, 21,
    // Bias, 2 remainder channels, 2 padding up to channel_subtile
    // 4 - (10 + 16 + 22 + 28) * 127 = -9648 = 0xFFFFDA50
    0x50, 0xDA, 0xFF, 0xFF,
    // 5 - (11 + 17 + 23 + 29) * 127 = -10155 = 0xFFFFD855
    0x55, 0xD8, 0xFF, 0xFF,
    0, 0, 0, 0,
    0, 0, 0, 0,
    10, 11, 0, 0, // 2 weights, 2 remainder channel, 1 padding up to channel_subtile
    22, 23, 0, 0,
    // No middle pass.
    12, 13, 14, 15, // last pass, 2 weights, 4 channels first.
    24, 25, 26, 27,
    0, 0, 0, 0, // padding to last_pass_tile
    16, 17, 0, 0, // 2 weights, channel_subtile (4)
    28, 29, 0, 0,
    0, 0, 0, 0, // padding to last_pass_tile
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

TEST(PACK_F16_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<uint16_t> k(c * h * w);  // k = [2, 3, // first 2x2 kernel
                                    //      4, 5,
                                    //      6, 7, // second 2x2 kernel
                                    //      8, 9]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f16_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // First pass.
    0, 1,  // bias
    2, 6,  // 2 weights, channels first, then columns
    4, 8,
    // No middle pass.
    3, 7,  // last pass, 2 weights
    5, 9,
    0, 0,  // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once_channels_gt_cr) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint16_t> k(c * h * w);  // k = [5, 6, 7, 8, // first 2x2 kernel
                                    //      9, 10, 11, 12, // second 2x2 kernel
                                    //      13, 14, 15, 16, // third 2x2 kernel
                                    //      17, 18, 19, 20, // fourth 2x2 kernel
                                    //      21, 22, 23, 24, // fifth 2x2 kernel
                                    //      ]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f16_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // First pass.
    0, 1, // bias
    5, 9, // 2 weights, 2 channels first, then columns
    7, 11,
    2, 3, // bias
    13, 17, // 2 weights, 2 channels first, then columns
    15, 19,
    4, 0, // bias
    21, 0, // 2 weights, 1 remainder channels first, then columns
    23, 0,
    // No middle pass.
    6, 10, // last pass, 2 weights, 2 channels first, then columns
    8, 12,
    0, 0,  // padding
    14, 18,
    16, 20,
    0, 0,  // padding
    22, 0,
    24, 0,
    0, 0,  // padding

  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_MULTIPASS_DWCONV_GHW_W, one_middle_pass_tile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 1;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<uint16_t> k(c * h * w);  // k = [2, 3, // first 2x2 kernel
                                    //      4, 5,
                                    //      6, 7, // second 2x2 kernel
                                    //      8, 9]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f16_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // First pass only has 1 element.
    0, 1, // bias
    2, 6, // weights, 2 channels, 1 element.
    // Middle pass has 2 elements, columns first.
    4, 8,
    3, 7,
    // Last pass.
    5, 9,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_MULTIPASS_DWCONV_GHW_W, one_middle_pass_tile_channels_gt_cr) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint16_t> k(c * h * w);  // k = [5, 6, // first 2x2 kernel
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
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f16_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // First pass only has 1 element.
    0, 1, // bias
    5, 9, // weights, 2 channels, 1 element.
    2, 3, // bias
    13, 17, // weights, 2 channels, 1 element.
    4, 0, // bias
    21, 0, // weights, 1 remainder channel, 1 element.
    // Middle pass has 2 elements, channels first, then columns.
    7, 11,
    6, 10,
    15, 19,
    14, 18,
    // Middle pass, 1 remainder channel.
    23, 0,
    22, 0,
    // Last pass,
    8, 12,
    0, 0,  // padding
    16, 20,
    0, 0,  // padding
    24, 0,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_MULTIPASS_DWCONV_GHW_W, multiple_middle_pass_tile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 3;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<uint16_t> k(c * h * w);  // k = [2, 3, 4, // first 2x3 kernel
                                    //      5, 6, 7,
                                    //      8, 9, 10, // second 2x3 kernel
                                    //      11, 12, 13]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f16_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // First pass has 2 elements.
    0, 1, // bias
    2, 8, // 1 weight, 2 channels first, then columns
    // Middle pass 1 (2 elements per pass).
    5, 11,
    3, 9,
    // Middle pass 2 (2 elements per pass).
    6, 12,
    4, 10,
    // Last pass.
    7, 13,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_MULTIPASS_DWCONV_GHW_W, multiple_middle_pass_tile_channels_gt_cr) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 3;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint16_t> k(c * h * w);  // k = [5, 6, 7, // first 2x3 kernel
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
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f16_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);


  const std::vector<uint16_t> expected = {
    // First pass has 1 element.
    0, 1, // bias
    5, 11, // 1 weight, 2 channels, 2 elements.
    2, 3, // bias
    17, 23, // 1 weight, 2 channels, 2 elements.
    4, 0, // bias
    29, 0, // 1 weight, 1 remainder channel, 2 elements.
    // Middle pass has 2 elements, channels first, then columns.
    8, 14,
    6, 12,
    20, 26,
    18, 24,
    // 1 remainder channel.
    32, 0,
    30, 0,
    // Second middle pass, 2 elements.
    9, 15,
    7, 13,
    21, 27,
    19, 25,
    // 1 remainder channel.
    33, 0,
    31, 0,
    // Last pass
    10, 16,
    0, 0,  // padding
    22, 28,
    0, 0,  // padding
    // 1 remainder channel.
    34, 0,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once_channel_subtile) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 4;
  const size_t channel_subtile = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint16_t> k(c * h * w);  // k = [5, 6, // first 2x2 kernel
                                    //      7, 8,
                                    //      9, 10, // second 2x2 kernel
                                    //      11, 12,
                                    //      13, 14, // third 2x2 kernel
                                    //      15, 16,
                                    //      17, 18, // fourth 2x2 kernel
                                    //      19, 20,
                                    //      21, 22, // fifth 2x2 kernel
                                    //      23, 24 ]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f16_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // First pass.
    0, 1, 2, 3,  // bias, 4 channels
    5, 9, 13, 17,  // 2 weights, 4 channels first, then columns
    7, 11, 15, 19,
    4, 0, // bias, 1 last channel, 1 padding up to channel_subtile
    21, 0, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    23, 0,
    // No middle pass.
    6, 10, 14, 18,  // last pass, 2 weights, 4 channels first
    8, 12, 16, 20,
    0, 0, 0, 0, // padding to last_pass_tile
    22, 0, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    24, 0,
    0, 0, // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_MULTIPASS_DWCONV_GHW_W, one_middle_pass_channel_subtile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 4;
  const size_t channel_subtile = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint16_t> k(c * h * w);  // k = [5, 6, // first 2x2 kernel
                                    //      7, 8,
                                    //      9, 10, // second 2x2 kernel
                                    //      11, 12,
                                    //      13, 14, // third 2x2 kernel
                                    //      15, 16,
                                    //      17, 18, // fourth 2x2 kernel
                                    //      19, 20,
                                    //      21, 22, // fifth 2x2 kernel
                                    //      23, 24 ]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f16_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // First pass.
    0, 1, 2, 3,  // bias, 4 channels
    5, 9, 13, 17,  // 1 weight, 4 channels first, then columns
    4, 0, // bias, 1 last channel, 1 padding up to channel_subtile
    21, 0, // 1 weight, 1 last channel, 1 padding up to channel_subtile
    // 1 middle pass
    7, 11, 15, 19, // 2 weights, 4 channels first, then columns
    6, 10, 14, 18,
    23, 0, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    22, 0,
    // Last pass.
    8, 12, 16, 20,  // 1 weight, 4 channels first
    0, 0, 0, 0, // padding to last_pass_tile
    24, 0, // 1 weight, 1 last channel, 1 padding up to channel_subtile
    0, 0, // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once_channel_subtile_rounded) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 7;
  const size_t cr = 4;
  const size_t channel_subtile = 2;
  // c rounded to channel_subtile is 8, so we will have 2 channel_tile loops in the first and middle pass.

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4, 5, 6]
  std::vector<uint16_t> k(c * h * w);  // k = [7, 8, // first 2x2 kernel
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
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f16_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // First pass.
    0, 1, 2, 3,  // bias, 4 channels
    7, 11, 15, 19,  // 2 weights, 4 channels first, then columns
    9, 13, 17, 21,
    4, 5, 6, 0, // bias, 3 remainder channels, 1 padding up to channel_tile
    23, 27, 31, 0,  // 2 weights, 3 remainder channels, 1 padding up to channel_tile
    25, 29, 33, 0,
    // No middle pass.
    // Last pass.
    8, 12, 16, 20,  // last pass, 2 weights, 4 channels first
    10, 14, 18, 22,
    0, 0, 0, 0, // padding to last_pass_tile
    24, 28, // last pass, 2 weights, channel_subtile (2)
    26, 30,
    0, 0, // padding to last_pass_tile
    32, 0, // 1 remainder channel, 1 padding up to channel_subtile
    34, 0, // 1 remainder channel, 1 padding up to channel_subtile
    0, 0, // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once_channel_subtile_rounded_to_channel_round) {
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

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4, 5]
  std::vector<uint16_t> k(c * h * w);  // k = [6, 7, 8, 9,
                                    //      10, 11, 12, 13
                                    //      14, 15, 16, 17,
                                    //      18, 19, 20, 21,
                                    //      22, 23, 24, 25,
                                    //      26, 27, 28, 29,]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(
      h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile, channel_round));

  xnn_pack_f16_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // First pass.
    0, 1, 2, 3, // bias, 4 channels
    6, 10, 14, 18, // 2 weights, 4 channels first, then columns
    8, 12, 16, 20,
    4, 5, 0, 0, // bias, 2 remainder channels, 1 padding up to channel_subtile
    22, 26, 0, 0, // 2 weights, 2 remainder channel, 1 padding up to channel_subtile
    24, 28, 0, 0,
    // No middle pass.
    7, 11, 15, 19, // last pass, 2 weights, 4 channels first.
    9, 13, 17, 21,
    0, 0, 0, 0, // padding to last_pass_tile
    23, 27, 0, 0, // 2 weights, channel_subtile (4)
    25, 29, 0, 0,
    0, 0, 0, 0, // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_MULTIPASS_DWCONV_GHW_W, one_middle_pass_channel_subtile_rounded) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 7;
  const size_t cr = 4;
  const size_t channel_subtile = 2;
  // c rounded to channel_subtile is 8, so we will have 2 channel_tile loops in first and middle pass.

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4, 5, 6]
  std::vector<uint16_t> k(c * h * w);  // k = [7, 8, // first 2x2 kernel
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
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f16_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // First pass.
    0, 1, 2, 3,  // bias, 4 channels
    7, 11, 15, 19,  // 1 weight, 4 channels first, then columns
    4, 5, 6, 0, // bias, 4 channels, 1 padding up to channel_tile
    23, 27, 31, 0,  // 1 weight, 3 remainder channels, 1 padding up to channel_tile
    // 1 middle pass.
    9, 13, 17, 21, // 2 weights, 4 channels first
    8, 12, 16, 20,
    25, 29, 33, 0, // 3 remainder channels, 1 padding up to channel_tile
    24, 28, 32, 0,
    // Last pass.
    10, 14, 18, 22,  // last pass, 1 weight, 4 channels first
    0, 0, 0, 0, // padding to last_pass_tile
    26, 30, // 1 weight, channel_subtile,
    0, 0, // padding to last_pass_tile
    34, 0, // 1 weight, 1 remainder channel, 1 padding up to channel_subtile
    0, 0, // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<uint16_t> k(c * h * w);  // k = [2, 3,
                                    //      4, 5,
                                    //      6, 7,
                                    //      8, 9]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f16_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // First pass.
    0, 1, // bias
    2, 3, // First pass, 2 weights, channels first, then columns
    6, 7,
    // No middle pass.
    4, 5, // Last pass, 2 weights
    8, 9,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once_channels_gt_cr) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint16_t> k(c * h * w);  // k = [5, 6, 7, 8, 9,
                                    //      10, 11, 12, 13, 14,
                                    //      15, 16, 17, 18, 19,
                                    //      20, 21, 22, 23, 24,
                                    //      ]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f16_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // First pass.
    0, 1, // bias
    5, 6, // 2 weights, 2 channels first, then columns
    15, 16,
    2, 3, // bias
    7, 8, // weights, 2 channels first, then columns
    17, 18,
    4, 0, // bias
    9, 0, // weights, 1 remainder channels first, then columns
    19, 0,
    // No middle pass.
    // Last pass, 2 weights
    10, 11,
    20, 21,
    0, 0,  // padding
    12, 13,
    22, 23,
    0, 0,  // padding
    14, 0,
    24, 0,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_MULTIPASS_DWCONV_HWG_W, one_middle_pass_tile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 1;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<uint16_t> k(c * h * w);  // k = [2, 3,
                                    //      4, 5,
                                    //      6, 7,
                                    //      8, 9]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f16_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // First pass only has 1 element.
    0, 1, // bias
    2, 3, // weights, 2 channels, 1 element.
    // Middle pass has 2 elements, columns first.
    6, 7,
    4, 5,
    // Last pass.
    8, 9,
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_MULTIPASS_DWCONV_HWG_W, one_middle_pass_tile_channels_gt_cr) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint16_t> k(c * h * w);  // k = [5, 6, 7, 8, 9,
                                    //      10, 11, 12, 13, 14,
                                    //      15, 16, 17, 18, 19,
                                    //      20, 21, 22, 23, 24,
                                    //      ]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f16_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // First pass only has 1 element.
    0, 1, // bias
    5, 6, // weights, 2 channels, 1 element.
    2, 3, // bias
    7, 8, // weights, 2 channels, 1 element.
    4, 0, // bias
    9, 0, // weights, 1 remainder channel, 1 element.
    // Middle pass has 2 elements, channels first, then columns.
    15, 16,
    10, 11,
    17, 18,
    12, 13,
    // Middle pass, 1 remainder channel.
    19, 0,
    14, 0,
    // Last pass.
    20, 21,
    0, 0,  // padding
    22, 23,
    0, 0,  // padding
    24, 0,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_MULTIPASS_DWCONV_HWG_W, multiple_middle_pass_tile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 3;
  const size_t c = 2;
  const size_t cr = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1]
  std::vector<uint16_t> k(c * h * w);  // k = [2, 3,
                                    //      4, 5,
                                    //      6, 7,
                                    //      8, 9,
                                    //      10, 11,
                                    //      12, 13]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f16_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // First pass has 2 elements.
    0, 1, // bias
    2, 3, // 1 weight, 2 channels first, then columns
    // 2 passes of middle pass (2 elements per pass).
    8, 9,
    4, 5,
    10, 11,
    6, 7,
    // Last pass.
    12, 13,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_MULTIPASS_DWCONV_HWG_W, multiple_middle_pass_tile_channels_gt_cr) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 3;
  const size_t c = 5;
  const size_t cr = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint16_t> k(c * h * w);  // k = [5, 6, 7, 8, 9,
                                    //      10, 11, 12, 13, 14,
                                    //      15, 16, 17, 18, 19,
                                    //      20, 21, 22, 23, 24,
                                    //      25, 26, 27, 28, 29,
                                    //      30, 31, 32, 33, 34,
                                    //      ]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f16_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);


  const std::vector<uint16_t> expected = {
    // First pass has 1 element.
    0, 1, // bias
    5, 6, // weights, 2 channels, 2 elements.
    2, 3, // bias
    7, 8, // weights, 2 channels, 2 elements.
    4, 0, // bias
    9, 0, // weights, 1 remainder channel, 2 elements.
    // Middle pass has 2 elements, channels first, then columns.
    20, 21,
    10, 11,
    22, 23,
    12, 13,
    // 1 remainder channel.
    24, 0,
    14, 0,
    // Second middle pass, 2 elements.
    25, 26,
    15, 16,
    27, 28,
    17, 18,
    29, 0,
    // 1 remainder channel.
    19, 0,
    // Last pass.
    30, 31,
    0, 0,  // padding
    32, 33,
    0, 0,  // padding
    // 1 remainder channel.
    34, 0,
    0, 0,  // padding
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once_channel_subtile) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 4;
  const size_t channel_subtile = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint16_t> k(c * h * w);  // k = [5, 6, 7, 8, 9, // first channel
                                    //      10, 11, 12, 13, 14, // second channel
                                    //      15, 16, 17, 18, 19, // third channel
                                    //      20, 21, 22, 23, 24] // fourth channel
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f16_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // First pass.
    0, 1, 2, 3, // bias, 4 channels
    5, 6, 7, 8, // first pass, 2 weights, 4 channels first, then columns
    15, 16, 17, 18,
    4, 0, // bias, 1 last channel, 1 padding up to channel_subtile
    9, 0, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    19, 0,
    // No middle pass.
    10, 11, 12, 13, // last pass, 2 weights, 4 channels first.
    20, 21, 22, 23,
    0, 0, 0, 0, // padding to last_pass_tile
    14, 0, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    24, 0, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    0, 0,  // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_MULTIPASS_DWCONV_HWG_W, one_middle_pass_channel_subtile) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 5;
  const size_t cr = 4;
  const size_t channel_subtile = 2;

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4]
  std::vector<uint16_t> k(c * h * w);  // k = [5, 6, 7, 8, 9,
                                    //      10, 11, 12, 13, 14,
                                    //      15, 16, 17, 18, 19,
                                    //      20, 21, 22, 23, 24]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f16_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // First pass.
    0, 1, 2, 3, // bias, 4 channels
    5, 6, 7, 8, // first pass, 1 weight, 4 channels first, then columns
    4, 0, // bias, 1 last channel, 1 padding up to channel_subtile
    9, 0, // 1 weight, 1 last channel, 1 padding up to channel_subtile
    // 1 middle pass.
    15, 16, 17, 18, // 2 weights, 4 channels first.
    10, 11, 12, 13,
    19, 0, // 2 weights, 1 last channel, 1 padding up to channel_subtile
    14, 0,
    // Last pass.
    20, 21, 22, 23, // 1 weight, 4 channels first.
    0, 0, 0, 0, // padding to last_pass_tile
    24, 0, // 1 weight, 1 last channel, 1 padding up to channel_subtile
    0, 0,  // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once_channel_subtile_rounded) {
  const size_t first_pass_tile = 2;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 3;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 7;
  const size_t cr = 4;
  const size_t channel_subtile = 2;
  // c rounded to channel_subtile is 8, so we will have 2 channel_tile loops for first and middle pass.

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4, 5, 6]
  std::vector<uint16_t> k(c * h * w);  // k = [7, 8, 9, 10, 11, 12, 13,
                                    //      14, 15, 16, 17, 18, 19, 20,
                                    //      21, 22, 23, 24, 25, 26, 27,
                                    //      28, 29, 30, 31, 32, 33, 34]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f16_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // First pass.
    0, 1, 2, 3, // bias, 4 channels
    7, 8, 9, 10, // 2 weights, 4 channels first, then columns
    21, 22, 23, 24,
    4, 5, 6, 0, // bias, 3 remainder channels, 1 padding up to channel_subtile
    11, 12, 13, 0, // 2 weights, 3 remainder channel, 1 padding up to channel_subtile
    25, 26, 27, 0,
    // No middle pass.
    14, 15, 16, 17, // last pass, 2 weights, 4 channels first.
    28, 29, 30, 31,
    0, 0, 0, 0, // padding to last_pass_tile
    18, 19, // 2 weights, channel_subtile (2)
    32, 33,
    0, 0,  // padding to last_pass_tile
    20, 0, // 2 weights, 1 remainder channel, 1 padding up to channel_subtile
    34, 0,
    0, 0,  // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_MULTIPASS_DWCONV_HWG_W, one_middle_pass_channel_subtile_rounded) {
  const size_t first_pass_tile = 1;
  const size_t middle_pass_tile = 2;
  const size_t last_pass_tile = 2;
  const size_t h = 2;
  const size_t w = 2;
  const size_t c = 7;
  const size_t cr = 4;
  const size_t channel_subtile = 2;
  // c rounded to channel_subtile is 8, so we will have 2 channel_tile loops for first and middle pass.

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4, 5, 6]
  std::vector<uint16_t> k(c * h * w);  // k = [7, 8, 9, 10, 11, 12, 13,
                                    //      14, 15, 16, 17, 18, 19, 20,
                                    //      21, 22, 23, 24, 25, 26, 27,
                                    //      28, 29, 30, 31, 32, 33, 34]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f16_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // First pass.
    0, 1, 2, 3, // bias, 4 channels
    7, 8, 9, 10, // 1 weight, 4 channels first, then columns
    4, 5, 6, 0, // bias, 3 remainder channels, 1 padding up to channel_subtile
    11, 12, 13, 0, // 1 weight, 3 remainder channel, 1 padding up to channel_subtile
    // 1 middle pass.
    21, 22, 23, 24, // 2 weights, 4 channels first
    14, 15, 16, 17,
    25, 26, 27, 0, // 3 remainder channels first, 1 padding up to channel_tile
    18, 19, 20, 0,
    // Last pass.
    28, 29, 30, 31, // last pass, 1 weight, 4 channels first.
    0, 0, 0, 0, // padding to last_pass_tile
    32, 33, // last pass, 1 weight, channel_subtile (2)
    0, 0,  // padding to last_pass_tile
    34, 0, // last pass, 1 remainder channel, 1 padding up to channel_subtile
    0, 0,  // padding to last_pass_tile
  };
  EXPECT_EQ(expected, packed_weights);
}

TEST(PACK_F16_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once_channel_subtile_rounded_to_channel_round) {
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

  std::vector<uint16_t> b(c);
  std::iota(b.begin(), b.end(), 0);  // b = [0, 1, 2, 3, 4, 5]
  std::vector<uint16_t> k(c * h * w);  // k = [6, 7, 8, 9, 10, 11,
                                    //      12, 13, 14, 15, 16, 17,
                                    //      18, 19, 20, 21, 22, 23,
                                    //      24, 25, 26, 27, 28, 29,]
  std::iota(k.begin(), k.end(), static_cast<uint16_t>(b.size()));
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(
      h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile, channel_round));

  xnn_pack_f16_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<uint16_t> expected = {
    // First pass.
    0, 1, 2, 3, // bias, 4 channels
    6, 7, 8, 9, // 2 weights, 4 channels first, then columns
    18, 19, 20, 21,
    4, 5, 0, 0, // bias, 2 remainder channels, 1 padding up to channel_subtile
    10, 11, 0, 0, // 2 weights, 2 remainder channel, 1 padding up to channel_subtile
    22, 23, 0, 0,
    // No middle pass.
    12, 13, 14, 15, // last pass, 2 weights, 4 channels first.
    24, 25, 26, 27,
    0, 0, 0, 0, // padding to last_pass_tile
    16, 17, 0, 0, // 2 weights, channel_subtile (4)
    28, 29, 0, 0,
    0, 0, 0, 0, // padding to last_pass_tile
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      1,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

  xnn_pack_f32_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

  xnn_pack_f32_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

  xnn_pack_f32_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

  xnn_pack_f32_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

  xnn_pack_f32_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

  xnn_pack_f32_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

  xnn_pack_f32_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass.
    0.0f, 1.0f, 2.0f, 3.0f,  // bias, 4 channels
    5.0f, 9.0f, 13.0f, 17.0f,  // 2 weights, 4 channels first, then columns
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

  xnn_pack_f32_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass.
    0.0f, 1.0f, 2.0f, 3.0f,  // bias, 4 channels
    5.0f, 9.0f, 13.0f, 17.0f,  // 1 weight, 4 channels first, then columns
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

  xnn_pack_f32_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass.
    0.0f, 1.0f, 2.0f, 3.0f,  // bias, 4 channels
    7.0f, 11.0f, 15.0f, 19.0f,  // 2 weights, 4 channels first, then columns
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

  xnn_pack_f32_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

  xnn_pack_f32_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass.
    0.0f, 1.0f, 2.0f, 3.0f,  // bias, 4 channels
    7.0f, 11.0f, 15.0f, 19.0f,  // 1 weight, 4 channels first, then columns
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

  xnn_pack_f32_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

  xnn_pack_f32_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

  xnn_pack_f32_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

  xnn_pack_f32_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

  xnn_pack_f32_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

  xnn_pack_f32_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

  xnn_pack_f32_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

  xnn_pack_f32_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

  xnn_pack_f32_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

  xnn_pack_f32_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

  xnn_pack_f32_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
      0,
      0,
      h,
      w,
      c,
      cr,
      cr,
      cr,
      k.data(),
      b.data(),
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

namespace {
// Helper matcher to allow us to specify expected weights as float, while
// comparing it to packed fp16 values.
MATCHER(Fp16MatchFp32, "") {
  return std::get<0>(arg) == fp16_ieee_from_fp32_value(std::get<1>(arg));
}
}

TEST(PACK_F32_TO_F16_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once) {
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
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_to_f16_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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

  EXPECT_THAT(packed_weights, testing::Pointwise(Fp16MatchFp32(), expected));
}

TEST(PACK_F32_TO_F16_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once_channels_gt_cr) {
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
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_to_f16_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
  EXPECT_THAT(packed_weights, testing::Pointwise(Fp16MatchFp32(), expected));
}

TEST(PACK_F32_TO_F16_MULTIPASS_DWCONV_GHW_W, one_middle_pass_tile) {
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
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_to_f16_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
  EXPECT_THAT(packed_weights, testing::Pointwise(Fp16MatchFp32(), expected));
}

TEST(PACK_F32_TO_F16_MULTIPASS_DWCONV_GHW_W, one_middle_pass_tile_channels_gt_cr) {
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
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_to_f16_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
  EXPECT_THAT(packed_weights, testing::Pointwise(Fp16MatchFp32(), expected));
}

TEST(PACK_F32_TO_F16_MULTIPASS_DWCONV_GHW_W, multiple_middle_pass_tile) {
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
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_to_f16_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
  EXPECT_THAT(packed_weights, testing::Pointwise(Fp16MatchFp32(), expected));
}

TEST(PACK_F32_TO_F16_MULTIPASS_DWCONV_GHW_W, multiple_middle_pass_tile_channels_gt_cr) {
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
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_to_f16_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
  EXPECT_THAT(packed_weights, testing::Pointwise(Fp16MatchFp32(), expected));
}

TEST(PACK_F32_TO_F16_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once_channel_subtile) {
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
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f32_to_f16_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass.
    0.0f, 1.0f, 2.0f, 3.0f,  // bias, 4 channels
    5.0f, 9.0f, 13.0f, 17.0f,  // 2 weights, 4 channels first, then columns
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
  EXPECT_THAT(packed_weights, testing::Pointwise(Fp16MatchFp32(), expected));
}

TEST(PACK_F32_TO_F16_MULTIPASS_DWCONV_GHW_W, one_middle_pass_channel_subtile) {
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
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f32_to_f16_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass.
    0.0f, 1.0f, 2.0f, 3.0f,  // bias, 4 channels
    5.0f, 9.0f, 13.0f, 17.0f,  // 1 weight, 4 channels first, then columns
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
  EXPECT_THAT(packed_weights, testing::Pointwise(Fp16MatchFp32(), expected));
}

TEST(PACK_F32_TO_F16_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once_channel_subtile_rounded) {
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
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f32_to_f16_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass.
    0.0f, 1.0f, 2.0f, 3.0f,  // bias, 4 channels
    7.0f, 11.0f, 15.0f, 19.0f,  // 2 weights, 4 channels first, then columns
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
  EXPECT_THAT(packed_weights, testing::Pointwise(Fp16MatchFp32(), expected));
}

TEST(PACK_F32_TO_F16_MULTIPASS_DWCONV_GHW_W, first_pass_once_last_pass_once_channel_subtile_rounded_to_channel_round) {
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
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(
      h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile, channel_round));

  xnn_pack_f32_to_f16_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
  EXPECT_THAT(packed_weights, testing::Pointwise(Fp16MatchFp32(), expected));
}

TEST(PACK_F32_TO_F16_MULTIPASS_DWCONV_GHW_W, one_middle_pass_channel_subtile_rounded) {
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
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f32_to_f16_dwconv_ghw_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
      0,
      nullptr);

  const std::vector<float> expected = {
    // First pass.
    0.0f, 1.0f, 2.0f, 3.0f,  // bias, 4 channels
    7.0f, 11.0f, 15.0f, 19.0f,  // 1 weight, 4 channels first, then columns
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
  EXPECT_THAT(packed_weights, testing::Pointwise(Fp16MatchFp32(), expected));
}

TEST(PACK_F32_TO_F16_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once) {
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
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_to_f16_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
  EXPECT_THAT(packed_weights, testing::Pointwise(Fp16MatchFp32(), expected));
}

TEST(PACK_F32_TO_F16_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once_channels_gt_cr) {
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
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_to_f16_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
  EXPECT_THAT(packed_weights, testing::Pointwise(Fp16MatchFp32(), expected));
}

TEST(PACK_F32_TO_F16_MULTIPASS_DWCONV_HWG_W, one_middle_pass_tile) {
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
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_to_f16_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
  EXPECT_THAT(packed_weights, testing::Pointwise(Fp16MatchFp32(), expected));
}

TEST(PACK_F32_TO_F16_MULTIPASS_DWCONV_HWG_W, one_middle_pass_tile_channels_gt_cr) {
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
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_to_f16_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
  EXPECT_THAT(packed_weights, testing::Pointwise(Fp16MatchFp32(), expected));
}

TEST(PACK_F32_TO_F16_MULTIPASS_DWCONV_HWG_W, multiple_middle_pass_tile) {
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
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_to_f16_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
  EXPECT_THAT(packed_weights, testing::Pointwise(Fp16MatchFp32(), expected));
}

TEST(PACK_F32_TO_F16_MULTIPASS_DWCONV_HWG_W, multiple_middle_pass_tile_channels_gt_cr) {
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
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr));

  xnn_pack_f32_to_f16_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
  EXPECT_THAT(packed_weights, testing::Pointwise(Fp16MatchFp32(), expected));
}

TEST(PACK_F32_TO_F16_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once_channel_subtile) {
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
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f32_to_f16_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
  EXPECT_THAT(packed_weights, testing::Pointwise(Fp16MatchFp32(), expected));
}

TEST(PACK_F32_TO_F16_MULTIPASS_DWCONV_HWG_W, one_middle_pass_channel_subtile) {
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
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f32_to_f16_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
  EXPECT_THAT(packed_weights, testing::Pointwise(Fp16MatchFp32(), expected));
}

TEST(PACK_F32_TO_F16_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once_channel_subtile_rounded) {
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
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f32_to_f16_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
  EXPECT_THAT(packed_weights, testing::Pointwise(Fp16MatchFp32(), expected));
}

TEST(PACK_F32_TO_F16_MULTIPASS_DWCONV_HWG_W, one_middle_pass_channel_subtile_rounded) {
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
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile));

  xnn_pack_f32_to_f16_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
  EXPECT_THAT(packed_weights, testing::Pointwise(Fp16MatchFp32(), expected));
}

TEST(PACK_F32_TO_F16_MULTIPASS_DWCONV_HWG_W, first_pass_once_last_pass_once_channel_subtile_rounded_to_channel_round) {
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
  std::vector<uint16_t> packed_weights(
    multipass_weights_count(
      h * w, first_pass_tile, middle_pass_tile, last_pass_tile, c, cr, channel_subtile, channel_round));

  xnn_pack_f32_to_f16_dwconv_hwg_w(
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
      /*scale=*/nullptr,
      packed_weights.data(),
      0,
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
  EXPECT_THAT(packed_weights, testing::Pointwise(Fp16MatchFp32(), expected));
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
  std::iota(k.begin(), k.end(), 3.0f);
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
