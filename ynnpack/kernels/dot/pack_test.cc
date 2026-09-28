#include "ynnpack/kernels/dot/pack.h"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/type.h"

using ::testing::_;
using ::testing::ElementsAreArray;
using ::testing::Matcher;

namespace ynn {

MATCHER_P4(Int2x4Match, e0, e1, e2, e3, "") {
  return ExplainMatchResult(e0, arg.get(0), result_listener) &&
         ExplainMatchResult(e1, arg.get(1), result_listener) &&
         ExplainMatchResult(e2, arg.get(2), result_listener) &&
         ExplainMatchResult(e3, arg.get(3), result_listener);
}

MATCHER_P2(Int4x2Match, e0, e1, "") {
  return ExplainMatchResult(e0, arg.get(0), result_listener) &&
         ExplainMatchResult(e1, arg.get(1), result_listener);
}

// Generate an (optionally transposed) matrix using an `iota` function down
// the columns:
//
// [ 0   m    2m  ...]
// [ 1  m+1  2m+1 ...]
// [ 2  m+1  2m+1 ...]
// [ .   .    .    . ]
// [ .   .    .    . ]
// [ .   .    .    . ]
// [m-1 2m-1 3m-1 ...]
//
// If `transposed` is true, the result is the above, transposed.
//
template <typename T>
std::vector<T> iota_columns(bool transposed, size_t m, size_t n) {
  const size_t elem_count = type_info<T>::element_count();
  std::vector<T> result(ceil_div(m * n, elem_count));
  if (transposed) {
    for (size_t i = 0; i < m * n; ++i) {
      type_info<T>::set(result.data(), i, i);
    }
  } else {
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        type_info<T>::set(result.data(), i * n + j,
                          static_cast<int>(j * m + i));
      }
    }
  }
  return result;
}

class pack : public ::testing::TestWithParam<bool> {};

INSTANTIATE_TEST_SUITE_P(pack, pack, ::testing::Bool());

TEST_P(pack, tile_1x1) {
  const bool transpose = GetParam();

  const int elem_size = sizeof(int);
  const int tile_m = 1;
  const int tile_n = 1;

  packer p(transpose, elem_size * 8, tile_m, tile_n);

  const int m = 5;
  const int n = 3;

  std::vector<int> input = iota_columns<int>(transpose, m, n);
  std::vector<int> output(align_up(m, tile_m) * align_up(n, tile_n), -1);

  p.pack(m, n, /*input_stride=*/(transpose ? m : n) * elem_size,
         /*input=*/input.data(),
         /*output_stride=*/elem_size * tile_m * tile_n,
         /*output_block_stride=*/elem_size * align_up(m, tile_m) * tile_n,
         /*output=*/output.data());

  std::vector<Matcher<int>> expected = {
      // clang-format off
      0,
      1,
      2,
      3,
      4,

      5,
      6,
      7,
      8,
      9,

      10,
      11,
      12,
      13,
      14,
      // clang-format on
  };

  EXPECT_THAT(output, ElementsAreArray(expected));
}

TEST_P(pack, tile_1x2) {
  const bool transpose = GetParam();

  const int elem_size = sizeof(int);
  const int tile_m = 1;
  const int tile_n = 2;

  packer p(transpose, elem_size * 8, tile_m, tile_n);

  const int m = 5;
  const int n = 3;

  std::vector<int> input = iota_columns<int>(transpose, m, n);
  std::vector<int> output(align_up(m, tile_m) * align_up(n, tile_n), -1);

  p.pack(m, n, /*input_stride=*/(transpose ? m : n) * elem_size,
         /*input=*/input.data(),
         /*output_stride=*/elem_size * tile_m * tile_n,
         /*output_block_stride=*/elem_size * align_up(m, tile_m) * tile_n,
         /*output=*/output.data());

  std::vector<Matcher<int>> expected = {
      // clang-format off
      0, 5,
      1, 6,
      2, 7,
      3, 8,
      4, 9,

      10, _,
      11, _,
      12, _,
      13, _,
      14, _,
      // clang-format on
  };

  EXPECT_THAT(output, ElementsAreArray(expected));
}

TEST_P(pack, tile_1x4) {
  const bool transpose = GetParam();

  const int elem_size = sizeof(int);
  const int tile_m = 1;
  const int tile_n = 4;

  packer p(transpose, elem_size * 8, tile_m, tile_n);

  const int m = 5;
  const int n = 3;

  std::vector<int> input = iota_columns<int>(transpose, m, n);
  std::vector<int> output(align_up(m, tile_m) * align_up(n, tile_n), -1);

  p.pack(m, n, /*input_stride=*/(transpose ? m : n) * elem_size,
         /*input=*/input.data(),
         /*output_stride=*/elem_size * tile_m * tile_n,
         /*output_block_stride=*/elem_size * align_up(m, tile_m) * tile_n,
         /*output=*/output.data());

  std::vector<Matcher<int>> expected = {
      // clang-format off
      0, 5, 10, _,
      1, 6, 11, _,
      2, 7, 12, _,
      3, 8, 13, _,
      4, 9, 14, _,
      // clang-format on
  };

  EXPECT_THAT(output, ElementsAreArray(expected));
}

TEST_P(pack, tile_2x1) {
  const bool transpose = GetParam();

  const int elem_size = sizeof(int);
  const int tile_m = 2;
  const int tile_n = 1;

  packer p(transpose, elem_size * 8, tile_m, tile_n);

  const int m = 5;
  const int n = 3;

  std::vector<int> input = iota_columns<int>(transpose, m, n);
  std::vector<int> output(align_up(m, tile_m) * align_up(n, tile_n), -1);

  p.pack(m, n, /*input_stride=*/(transpose ? m : n) * elem_size,
         /*input=*/input.data(),
         /*output_stride=*/elem_size * tile_m * tile_n,
         /*output_block_stride=*/elem_size * align_up(m, tile_m) * tile_n,
         /*output=*/output.data());

  std::vector<Matcher<int>> expected = {
      // clang-format off
      0, 1,
      2, 3,
      4, 0,

      5, 6,
      7, 8,
      9, 0,

      10, 11,
      12, 13,
      14, 0,
      // clang-format on
  };
  EXPECT_THAT(output, ElementsAreArray(expected));
}

TEST_P(pack, tile_2x2) {
  const bool transpose = GetParam();

  const int elem_size = sizeof(int);
  const int tile_m = 2;
  const int tile_n = 2;

  packer p(transpose, elem_size * 8, tile_m, tile_n);

  const int m = 5;
  const int n = 3;

  std::vector<int> input = iota_columns<int>(transpose, m, n);
  std::vector<int> output(align_up(m, tile_m) * align_up(n, tile_n), -1);

  p.pack(m, n, /*input_stride=*/(transpose ? m : n) * elem_size,
         /*input=*/input.data(),
         /*output_stride=*/elem_size * tile_m * tile_n,
         /*output_block_stride=*/elem_size * align_up(m, tile_m) * tile_n,
         /*output=*/output.data());

  std::vector<Matcher<int>> expected = {
      // clang-format off
      0, 1, 5, 6,
      2, 3, 7, 8,
      4, 0, 9, 0,

      10, 11, _, _,
      12, 13, _, _,
      14, 0, _, _,
      // clang-format on
  };
  EXPECT_THAT(output, ElementsAreArray(expected));
}

TEST_P(pack, tile_2x4) {
  const bool transpose = GetParam();

  const int elem_size = sizeof(int);
  const int tile_m = 2;
  const int tile_n = 4;

  packer p(transpose, elem_size * 8, tile_m, tile_n);

  const int m = 5;
  const int n = 3;

  std::vector<int> input = iota_columns<int>(transpose, m, n);
  std::vector<int> output(align_up(m, tile_m) * align_up(n, tile_n), -1);

  p.pack(m, n, /*input_stride=*/(transpose ? m : n) * elem_size,
         /*input=*/input.data(),
         /*output_stride=*/elem_size * tile_m * tile_n,
         /*output_block_stride=*/elem_size * align_up(m, tile_m) * tile_n,
         /*output=*/output.data());

  std::vector<Matcher<int>> expected = {
      // clang-format off
      0, 1, 5, 6, 10, 11, _, _,
      2, 3, 7, 8, 12, 13, _, _,
      4, 0, 9, 0, 14, 0, _, _,
      // clang-format on
  };
  EXPECT_THAT(output, ElementsAreArray(expected));
}

TEST_P(pack, tile_4x2) {
  const bool transpose = GetParam();

  const int elem_size = sizeof(int);
  const int tile_m = 4;
  const int tile_n = 2;

  packer p(transpose, elem_size * 8, tile_m, tile_n);

  const int m = 5;
  const int n = 3;

  std::vector<int> input = iota_columns<int>(transpose, m, n);
  std::vector<int> output(align_up(m, tile_m) * align_up(n, tile_n), -1);

  p.pack(m, n, /*input_stride=*/(transpose ? m : n) * elem_size,
         /*input=*/input.data(),
         /*output_stride=*/elem_size * tile_m * tile_n,
         /*output_block_stride=*/elem_size * align_up(m, tile_m) * tile_n,
         /*output=*/output.data());

  std::vector<Matcher<int>> expected = {
      // clang-format off
      0, 1, 2, 3, 5, 6, 7, 8,
      4, 0, 0, 0, 9, 0, 0, 0,

      10, 11, 12, 13, _, _, _, _,
      14, 0, 0, 0, _, _, _, _,
      // clang-format on
  };
  EXPECT_THAT(output, ElementsAreArray(expected));
}

TEST(pack_subbyte, int2_transpose_false) {
  const bool transpose = false;
  const int element_size_bits = 2;
  const int tile_m = 4;
  const int tile_n = 4;

  packer p(transpose, element_size_bits, tile_m, tile_n);

  const int m = 8;
  const int n = 4;

  std::vector<int2x4> input = iota_columns<int2x4>(transpose, m, n);
  std::vector<int2x4> output(8);

  p.pack(m, n, /*input_stride=*/1,
         /*input=*/input.data(),
         /*output_stride=*/4,
         /*output_block_stride=*/8,
         /*output=*/output.data());

  std::vector<Matcher<int2x4>> expected = {
      Int2x4Match(0, 1, -2, -1), Int2x4Match(0, 1, -2, -1),
      Int2x4Match(0, 1, -2, -1), Int2x4Match(0, 1, -2, -1),
      Int2x4Match(0, 1, -2, -1), Int2x4Match(0, 1, -2, -1),
      Int2x4Match(0, 1, -2, -1), Int2x4Match(0, 1, -2, -1),
  };

  EXPECT_THAT(output, ElementsAreArray(expected));
}

TEST(pack_subbyte, int4_transpose_false) {
  const bool transpose = false;
  const int element_size_bits = 4;
  const int tile_m = 4;
  const int tile_n = 4;

  packer p(transpose, element_size_bits, tile_m, tile_n);

  const int m = 8;
  const int n = 4;

  std::vector<int4x2> input = iota_columns<int4x2>(transpose, m, n);
  std::vector<int4x2> output(16);

  p.pack(m, n, /*input_stride=*/2,
         /*input=*/input.data(),
         /*output_stride=*/8,
         /*output_block_stride=*/16,
         /*output=*/output.data());

  std::vector<Matcher<int4x2>> expected = {
      Int4x2Match(0, 1),   Int4x2Match(2, 3),   Int4x2Match(-8, -7),
      Int4x2Match(-6, -5), Int4x2Match(0, 1),   Int4x2Match(2, 3),
      Int4x2Match(-8, -7), Int4x2Match(-6, -5), Int4x2Match(4, 5),
      Int4x2Match(6, 7),   Int4x2Match(-4, -3), Int4x2Match(-2, -1),
      Int4x2Match(4, 5),   Int4x2Match(6, 7),   Int4x2Match(-4, -3),
      Int4x2Match(-2, -1),
  };

  EXPECT_THAT(output, ElementsAreArray(expected));
}

TEST(pack_subbyte, int2_non_aligned) {
  const bool transpose = false;
  const int element_size_bits = 2;
  const int tile_m = 4;
  const int tile_n = 4;

  packer p(transpose, element_size_bits, tile_m, tile_n);

  const int m = 5;
  const int n = 4;

  std::vector<int2x4> input = iota_columns<int2x4>(transpose, m, n);
  std::vector<int2x4> output(8);

  p.pack(m, n, /*input_stride=*/1,
         /*input=*/input.data(),
         /*output_stride=*/4,
         /*output_block_stride=*/8,
         /*output=*/output.data());

  std::vector<Matcher<int2x4>> expected = {
      Int2x4Match(0, 1, -2, -1), Int2x4Match(1, -2, -1, 0),
      Int2x4Match(-2, -1, 0, 1), Int2x4Match(-1, 0, 1, -2),
      Int2x4Match(0, _, _, _),   Int2x4Match(1, _, _, _),
      Int2x4Match(-2, _, _, _),  Int2x4Match(-1, _, _, _),
  };

  EXPECT_THAT(output, ElementsAreArray(expected));
}

TEST(pack_a, aligned_blocks) {
  const size_t m = 4;
  const size_t k = 6;
  const size_t block_m = 2;
  const size_t tile_k = 3;
  const size_t elem_size = sizeof(int);

  // Row-major M x K matrix:
  // row 0:  0,  1,  2,  3,  4,  5
  // row 1:  6,  7,  8,  9, 10, 11
  // row 2: 12, 13, 14, 15, 16, 17
  // row 3: 18, 19, 20, 21, 22, 23
  std::vector<int> input(m * k);
  std::iota(input.begin(), input.end(), 0);

  const size_t output_m_blocks = ceil_div(m, block_m);
  const size_t output_k_blocks = ceil_div(k, tile_k);
  std::vector<int> output(output_k_blocks * output_m_blocks * block_m * tile_k,
                          -1);

  const size_t output_mo_stride = block_m * tile_k * elem_size;
  const size_t output_ko_stride = output_m_blocks * output_mo_stride;

  packer p(/*transpose=*/true, elem_size * 8, tile_k, block_m);
  p.pack(k, m, k * elem_size, input.data(), output_ko_stride, output_mo_stride,
         output.data());

  // Expected layout: outer block_k_i (0..1), then block_m_i (0..1), then m_i
  // (0..1), then k_i (0..2).
  std::vector<int> expected = {
      // clang-format off
      // block_k_i = 0 (cols 0..2), block_m_i = 0 (rows 0..1)
      0, 1, 2,
      6, 7, 8,
      // block_k_i = 0 (cols 0..2), block_m_i = 1 (rows 2..3)
      12, 13, 14,
      18, 19, 20,
      // block_k_i = 1 (cols 3..5), block_m_i = 0 (rows 0..1)
      3, 4, 5,
      9, 10, 11,
      // block_k_i = 1 (cols 3..5), block_m_i = 1 (rows 2..3)
      15, 16, 17,
      21, 22, 23,
      // clang-format on
  };
  EXPECT_THAT(output, ElementsAreArray(expected));
}

TEST(pack_a, unaligned_padding) {
  const size_t m = 3;
  const size_t k = 5;
  const size_t block_m = 2;
  const size_t tile_k = 4;
  const size_t elem_size = sizeof(int16_t);

  // Row-major 3 x 5 matrix:
  // row 0:  1,  2,  3,  4,  5
  // row 1:  6,  7,  8,  9, 10
  // row 2: 11, 12, 13, 14, 15
  std::vector<int16_t> input(m * k);
  std::iota(input.begin(), input.end(), 1);

  const size_t output_m_blocks = ceil_div(m, block_m);  // 2
  const size_t output_k_blocks = ceil_div(k, tile_k);   // 2
  std::vector<int16_t> output(
      output_k_blocks * output_m_blocks * block_m * tile_k, -1);

  const size_t output_mo_stride = block_m * tile_k * elem_size;
  const size_t output_ko_stride = output_m_blocks * output_mo_stride;

  packer p(/*transpose=*/true, elem_size * 8, tile_k, block_m);
  p.pack(k, m, k * elem_size, input.data(), output_ko_stride, output_mo_stride,
         output.data());

  std::vector<int16_t> expected = {
      // clang-format off
      // block_k_i = 0 (cols 0..3), block_m_i = 0 (rows 0..1)
      1, 2, 3, 4,
      6, 7, 8, 9,
      // block_k_i = 0 (cols 0..3), block_m_i = 1 (row 2, row 3 padded with 0)
      11, 12, 13, 14,
      0, 0, 0, 0,
      // block_k_i = 1 (col 4, cols 5..7 padded with 0),
      // block_m_i = 0 (rows 0..1)
      5, 0, 0, 0,
      10, 0, 0, 0,
      // block_k_i = 1 (col 4, cols 5..7 padded with 0),
      // block_m_i = 1 (row 2, row 3 padded)
      15, 0, 0, 0,
      0, 0, 0, 0,
      // clang-format on
  };
  EXPECT_THAT(output, ElementsAreArray(expected));
}

TEST(pack_a, amx_tile_32x32_bf16) {
  const size_t m = 35;
  const size_t k = 50;
  const size_t block_m = 32;
  const size_t tile_k = 32;
  const size_t elem_size = sizeof(uint16_t);

  std::vector<uint16_t> input(m * k);
  for (size_t i = 0; i < m * k; ++i) {
    input[i] = static_cast<uint16_t>((i % 251) + 1);
  }

  const size_t output_m_blocks = ceil_div(m, block_m);  // 2
  const size_t output_k_blocks = ceil_div(k, tile_k);   // 2
  std::vector<uint16_t> output(
      output_k_blocks * output_m_blocks * block_m * tile_k, 0xFFFF);

  const size_t output_mo_stride = block_m * tile_k * elem_size;
  const size_t output_ko_stride = output_m_blocks * output_mo_stride;

  packer p(/*transpose=*/true, elem_size * 8, tile_k, block_m);
  p.pack(k, m, k * elem_size, input.data(), output_ko_stride, output_mo_stride,
         output.data());

  for (size_t ko = 0; ko < output_k_blocks; ++ko) {
    for (size_t mo = 0; mo < output_m_blocks; ++mo) {
      for (size_t mi = 0; mi < block_m; ++mi) {
        for (size_t ki = 0; ki < tile_k; ++ki) {
          const size_t global_m = mo * block_m + mi;
          const size_t global_k = ko * tile_k + ki;
          const size_t out_idx =
              ((ko * output_m_blocks + mo) * block_m + mi) * tile_k + ki;
          const uint16_t expected_val = (global_m < m && global_k < k)
                                            ? input[global_m * k + global_k]
                                            : 0;
          EXPECT_EQ(output[out_idx], expected_val);
        }
      }
    }
  }
}

}  // namespace ynn
