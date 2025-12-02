#include "ynnpack/kernels/dot/pack.h"

#include <cstddef>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/arithmetic.h"

using ::testing::_;
using ::testing::ElementsAreArray;
using ::testing::Matcher;

namespace ynn {

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
  std::vector<T> result(m * n);
  if (transposed) {
    std::iota(result.begin(), result.end(), 0);
  } else {
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        result[i * n + j] = j * m + i;
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

}  // namespace ynn
