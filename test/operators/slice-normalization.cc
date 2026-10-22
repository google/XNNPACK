// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <climits>
#include <cstddef>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/normalization.h"
#include "test/operators/slice-normalization-tester.h"

TEST(SLICE_NORMALIZATION_TEST, normalize_1d_full_slice) {
  SliceNormalizationTester()
      .input_shape({3})
      .offsets({0})
      .sizes({3})
      .expected_offsets({0})
      .expected_input_shape({3})
      .expected_output_shape({3})
      .Test();
}

TEST(SLICE_NORMALIZATION_TEST, normalize_6d_full_slice) {
  SliceNormalizationTester()
      .input_shape({3, 4, 5, 6, 7, 8})
      .offsets({0, 0, 0, 0, 0, 0})
      .sizes({3, 4, 5, 6, 7, 8})
      .expected_offsets({0})
      .expected_input_shape({20160})
      .expected_output_shape({20160})
      .Test();
}

TEST(SLICE_NORMALIZATION_TEST, no_normalization_1d) {
  SliceNormalizationTester()
      .input_shape({3})
      .offsets({0})
      .sizes({1})
      .expected_offsets({0})
      .expected_input_shape({3})
      .expected_output_shape({1})
      .Test();
}

TEST(SLICE_NORMALIZATION_TEST, no_normalization_3d) {
  SliceNormalizationTester()
      .input_shape({3, 4, 5})
      .offsets({0, 1, 0})
      .sizes({2, 2, 3})
      .expected_offsets({0, 1, 0})
      .expected_input_shape({3, 4, 5})
      .expected_output_shape({2, 2, 3})
      .Test();
}

TEST(SLICE_NORMALIZATION_TEST, normalize_2d) {
  SliceNormalizationTester()
      .input_shape({5, 4})
      .offsets({1, 0})
      .sizes({2, 4})
      .expected_offsets({4})
      .expected_input_shape({20})
      .expected_output_shape({8})
      .Test();
}

TEST(SLICE_NORMALIZATION_TEST, normalize_4d) {
  SliceNormalizationTester()
      .input_shape({5, 4, 6, 3})
      .offsets({2, 0, 2, 0})
      .sizes({2, 4, 3, 3})
      .expected_offsets({8, 6})
      .expected_input_shape({20, 18})
      .expected_output_shape({8, 9})
      .Test();
}

TEST(SLICE_NORMALIZATION_TEST, normalize_5d_continuous_run) {
  SliceNormalizationTester()
      .input_shape({5, 4, 6, 3})
      .offsets({2, 0, 0, 0})
      .sizes({2, 4, 6, 3})
      .expected_offsets({144})
      .expected_input_shape({360})
      .expected_output_shape({144})
      .Test();
}

TEST(SLICE_NORMALIZATION_TEST, normalize_6d) {
  SliceNormalizationTester()
      .input_shape({5, 4, 6, 3, 7, 8})
      .offsets({0, 0, 0, 0, 2, 0})
      .sizes({5, 2, 6, 3, 2, 8})
      .expected_offsets({0, 0, 16})
      .expected_input_shape({5, 72, 56})
      .expected_output_shape({5, 36, 16})
      .Test();
}

TEST(SLICE_NORMALIZATION_TEST, normalize_1d_remove_size_1) {
  SliceNormalizationTester()
      .input_shape({3})
      .offsets({1})
      .sizes({1})
      .expected_offsets({1})
      .expected_input_shape({3})
      .expected_output_shape({1})
      .Test();
}

TEST(SLICE_NORMALIZATION_TEST, normalize_2d_remove_size_1) {
  SliceNormalizationTester()
      .input_shape({3, 4})
      .offsets({1, 2})
      .sizes({1, 2})
      .expected_offsets({6})
      .expected_input_shape({12})
      .expected_output_shape({2})
      .Test();
}

TEST(SLICE_NORMALIZATION_TEST, normalize_2d_remove_size_1_offset_0) {
  SliceNormalizationTester()
      .input_shape({3, 3})
      .offsets({0, 0})
      .sizes({1, 2})
      .expected_offsets({0})
      .expected_input_shape({9})
      .expected_output_shape({2})
      .Test();
}

TEST(SLICE_NORMALIZATION_TEST, normalize_2d_remove_size_1_offset_1) {
  SliceNormalizationTester()
      .input_shape({3, 3})
      .offsets({1, 0})
      .sizes({1, 3})
      .expected_offsets({3})
      .expected_input_shape({9})
      .expected_output_shape({3})
      .Test();
}

TEST(SLICE_NORMALIZATION_TEST, normalize_2d_remove_size_1_multiple) {
  SliceNormalizationTester()
      .input_shape({3, 4})
      .offsets({1, 2})
      .sizes({1, 1})
      .expected_offsets({6})
      .expected_input_shape({12})
      .expected_output_shape({1})
      .Test();
}

TEST(SLICE_NORMALIZATION_TEST, normalize_3d_remove_size_1) {
  SliceNormalizationTester()
      .input_shape({3, 4, 5})
      .offsets({1, 1, 1})
      .sizes({2, 1, 2})
      .expected_offsets({1, 6})
      .expected_input_shape({3, 20})
      .expected_output_shape({2, 2})
      .Test();
}

TEST(SLICE_NORMALIZATION_TEST, normalize_3d_remove_size_1_outer) {
  SliceNormalizationTester()
      .input_shape({3, 4, 5})
      .offsets({1, 2, 1})
      .sizes({1, 2, 1})
      .expected_offsets({1 * 4 + 2, 1})
      .expected_input_shape({12, 5})
      .expected_output_shape({2, 1})
      .Test();
}

TEST(SLICE_NORMALIZATION_TEST, normalize_4d_remove_size_1) {
  SliceNormalizationTester()
      .input_shape({3, 4, 5, 6})
      .offsets({2, 1, 3, 4})
      .sizes({1, 2, 1, 2})
      .expected_offsets({2 * 4 + 1, 3 * 6 + 4})
      .expected_input_shape({12, 30})
      .expected_output_shape({2, 2})
      .Test();
}

TEST(SLICE_NORMALIZATION_TEST, normalize_4d_remove_size_1_full_slice) {
  SliceNormalizationTester()
      .input_shape({3, 4, 5, 6})
      .offsets({2, 3, 0, 4})
      .sizes({1, 1, 5, 2})
      .expected_offsets({2 * 4 * 5 + 3 * 5, 4})
      .expected_input_shape({3 * 4 * 5, 6})
      .expected_output_shape({5, 2})
      .Test();
}

TEST(SLICE_NORMALIZATION_TEST, normalize_4d_remove_size_1_contiguous) {
  SliceNormalizationTester()
      .input_shape({3, 4, 5, 6})
      .offsets({1, 2, 3, 4})
      .sizes({2, 1, 1, 2})
      .expected_offsets({1, 2 * 5 * 6 + 3 * 6 + 4})
      .expected_input_shape({3, 120})
      .expected_output_shape({2, 2})
      .Test();
}

TEST(SLICE_NORMALIZATION_TEST, normalize_6d_remove_size_1) {
  SliceNormalizationTester()
      .input_shape({3, 4, 5, 6, 7, 8})
      .offsets({1, 2, 3, 4, 5, 6})
      .sizes({2, 1, 1, 1, 1, 2})
      .expected_offsets({1, (2 * 5 * 6 * 7 * 8) + (3 * 6 * 7 * 8) +
                                (4 * 7 * 8) + (5 * 8) + 6})
      .expected_input_shape({3, 6720})
      .expected_output_shape({2, 2})
      .Test();
}

// Overflow regression tests: xnn_normalize_slice must return false when
// merging dimensions would overflow size_t.
TEST(SLICE_NORMALIZATION_TEST, overflow_merged_dim_returns_false) {
  // Two adjacent full-slice dims whose product overflows size_t.
  constexpr size_t kHalf = (SIZE_MAX / 2) + 2;
  size_t offsets[2] = {0, 0};
  size_t sizes[2] = {kHalf, kHalf};
  size_t input_shape[2] = {kHalf, kHalf};
  size_t normalized_offsets[XNN_MAX_TENSOR_DIMS];
  size_t normalized_input_shape[XNN_MAX_TENSOR_DIMS];
  size_t normalized_output_shape[XNN_MAX_TENSOR_DIMS];
  size_t num_normalized_dims = 0;
  EXPECT_FALSE(xnn_normalize_slice(
      2, offsets, sizes, input_shape,
      normalized_offsets, normalized_input_shape,
      normalized_output_shape, &num_normalized_dims));
}

TEST(SLICE_NORMALIZATION_TEST, overflow_size1_merge_returns_false) {
  // Inner dim has size 1 (triggers first-pass merge path) with overflow.
  constexpr size_t kHalf = (SIZE_MAX / 2) + 2;
  // Three dims: [kHalf, kHalf, 1] — sizes [kHalf, kHalf, 1].
  // The innermost size-1 dim triggers a merge of the first two.
  size_t offsets[3] = {0, 0, 0};
  size_t sizes[3] = {kHalf, kHalf, 1};
  size_t input_shape[3] = {kHalf, kHalf, 1};
  size_t normalized_offsets[XNN_MAX_TENSOR_DIMS];
  size_t normalized_input_shape[XNN_MAX_TENSOR_DIMS];
  size_t normalized_output_shape[XNN_MAX_TENSOR_DIMS];
  size_t num_normalized_dims = 0;
  EXPECT_FALSE(xnn_normalize_slice(
      3, offsets, sizes, input_shape,
      normalized_offsets, normalized_input_shape,
      normalized_output_shape, &num_normalized_dims));
}
