#include "ynnpack/kernels/dot/pack.h"

#include <cstddef>
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

}  // namespace ynn
