// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>
#include <xnnpack/normalization.h>

#include "slice-normalization-tester.h"
#include <gtest/gtest.h>

TEST(SLICE_NORMALIZATION_TEST, 1d_full_slice) {
  SliceNormalizationTester()
      .input_shape({3})
      .offsets({0})
      .sizes({3})
      .expected_offsets({0})
      .expected_input_shape({3})
      .expected_output_shape({3})
      .Test();
}

TEST(SLICE_NORMALIZATION_TEST, 6d_full_slice) {
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
      .sizes({1, 2, 3})
      .expected_offsets({0, 1, 0})
      .expected_input_shape({3, 4, 5})
      .expected_output_shape({1, 2, 3})
      .Test();
}

TEST(SLICE_NORMALIZATION_TEST, normalize_2d) {
  SliceNormalizationTester()
      .input_shape({5, 4})
      .offsets({1, 0})
      .sizes({2, 4})
      .expected_offsets({4})
      .expected_input_shape({12})
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

TEST(SLICE_NORMALIZATION_TEST, normalize_3d_remove_size_1) {
  SliceNormalizationTester()
      .input_shape({3, 4, 5})
      .offsets({1, 1, 1})
      .sizes({2, 1, 2})
      .expected_offsets({1, 5})
      .expected_input_shape({3, 20})
      .expected_output_shape({2, 2})
      .Test();
}

TEST(SLICE_NORMALIZATION_TEST, normalize_6d_remove_size_1) {
  SliceNormalizationTester()
      .input_shape({3, 4, 5, 6, 7, 8})
      .offsets({1, 1, 1, 1, 1, 1})
      .sizes({2, 1, 1, 1, 1, 2})
      .expected_offsets({1, 1680})
      .expected_input_shape({3, 6720})
      .expected_output_shape({2, 2})
      .Test();
}
