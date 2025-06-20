// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "test/operators/deconvolution-operator-tester.h"

constexpr size_t kUnstridedInputHeight = 8;
constexpr size_t kUnstridedInputWidth = 7;
constexpr size_t kStridedInputHeight = 6;
constexpr size_t kStridedInputWidth = 5;

// using DeconvolutionTestCase =
//     std::pair<const char*, std::vector<DeconvolutionOperatorTester>>;
struct DeconvolutionTestCase {
  const char* first;
  std::vector<DeconvolutionOperatorTester> second;
};

static std::vector<DeconvolutionTestCase> CreateDeconvolutionTests(
    const struct xnn_gemm_config* gemm_config) {
  std::vector<DeconvolutionTestCase> tests;
  if (!gemm_config) {
    return {};
  }

  /**************************** Future GEMM path ****************************/
  tests.push_back({"kernel_1x1", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"kernel_1x1_varying_input_width", {}});
  for (size_t input_height = kUnstridedInputHeight - 2;
       input_height <= kUnstridedInputHeight + 2; input_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(input_height, kUnstridedInputWidth)
            .kernel_size(1, 1)
            .group_input_channels(23)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_1x1_varying_input_height", {}});
  for (size_t input_width = kUnstridedInputWidth - 2;
       input_width <= kUnstridedInputWidth + 2; input_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, input_width)
            .kernel_size(1, 1)
            .group_input_channels(23)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_1x1_varying_input_channels", {}});
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .kernel_size(1, 1)
            .group_input_channels(input_channels)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_1x1_varying_output_channels", {}});
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2;
       output_channels *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .kernel_size(1, 1)
            .group_input_channels(23)
            .group_output_channels(output_channels));
  }

  tests.push_back({"kernel_1x1_with_input_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .input_pixel_stride(28));

  tests.push_back({"kernel_1x1_with_output_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .output_pixel_stride(gemm_config->nr * 2 + 13));

  tests.push_back({"kernel_1x1_with_qmin", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmin(128));

  tests.push_back({"kernel_1x1_with_qmax", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmax(128));

  tests.push_back({"kernel_1x1_without_bias", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .has_bias(false)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  /************************ Future GEMM path, grouped *************************/
  tests.push_back({"grouped_1x1", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"grouped_1x1_varying_input_width", {}});
  for (size_t input_height = kUnstridedInputHeight - 2;
       input_height <= kUnstridedInputHeight + 2; input_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(input_height, kUnstridedInputWidth)
            .kernel_size(1, 1)
            .groups(2)
            .group_input_channels(23)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_1x1_varying_input_height", {}});
  for (size_t input_width = kUnstridedInputWidth - 2;
       input_width <= kUnstridedInputWidth + 2; input_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, input_width)
            .kernel_size(1, 1)
            .groups(2)
            .group_input_channels(23)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_1x1_varying_input_channels", {}});
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .kernel_size(1, 1)
            .groups(2)
            .group_input_channels(input_channels)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_1x1_varying_output_channels", {}});
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2;
       output_channels *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .kernel_size(1, 1)
            .groups(2)
            .group_input_channels(23)
            .group_output_channels(output_channels));
  }

  tests.push_back({"grouped_1x1_with_input_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .input_pixel_stride(47));

  tests.push_back({"grouped_1x1_with_output_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr + 3)
          .output_pixel_stride(gemm_config->nr * 2 + 13));

  tests.push_back({"grouped_1x1_with_qmin", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmin(128));

  tests.push_back({"grouped_1x1_with_qmax", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmax(128));

  tests.push_back({"grouped_1x1_without_bias", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .has_bias(false)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  /************************ Future GEMM path, batched ************************/
  tests.push_back({"batched_1x1", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"batched_1x1_varying_input_width", {}});
  for (size_t input_height = kUnstridedInputHeight - 2;
       input_height <= kUnstridedInputHeight + 2; input_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(input_height, kUnstridedInputWidth)
            .kernel_size(1, 1)
            .group_input_channels(23)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_1x1_varying_input_height", {}});
  for (size_t input_width = kUnstridedInputWidth - 2;
       input_width <= kUnstridedInputWidth + 2; input_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, input_width)
            .kernel_size(1, 1)
            .group_input_channels(23)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_1x1_varying_input_channels", {}});
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .kernel_size(1, 1)
            .group_input_channels(input_channels)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_1x1_varying_output_channels", {}});
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2;
       output_channels *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .kernel_size(1, 1)
            .group_input_channels(23)
            .group_output_channels(output_channels));
  }

  tests.push_back({"batched_1x1_with_input_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .input_pixel_stride(28));

  tests.push_back({"batched_1x1_with_output_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .output_pixel_stride(gemm_config->nr * 2 + 13));

  tests.push_back({"batched_1x1_with_qmin", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmin(128));

  tests.push_back({"batched_1x1_with_qmax", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmax(128));

  tests.push_back({"batched_1x1_without_bias", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .has_bias(false)
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  /******************** Future GEMM path, batched, grouped ********************/
  tests.push_back({"batched_grouped_1x1", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"batched_grouped_1x1_varying_input_width", {}});
  for (size_t input_height = kUnstridedInputHeight - 2;
       input_height <= kUnstridedInputHeight + 2; input_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(input_height, kUnstridedInputWidth)
            .kernel_size(1, 1)
            .groups(2)
            .group_input_channels(23)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_1x1_varying_input_height", {}});
  for (size_t input_width = kUnstridedInputWidth - 2;
       input_width <= kUnstridedInputWidth + 2; input_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, input_width)
            .kernel_size(1, 1)
            .groups(2)
            .group_input_channels(23)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_1x1_varying_input_channels", {}});
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .kernel_size(1, 1)
            .groups(2)
            .group_input_channels(input_channels)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_1x1_varying_output_channels", {}});
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2;
       output_channels *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .kernel_size(1, 1)
            .groups(2)
            .group_input_channels(23)
            .group_output_channels(output_channels));
  }

  tests.push_back({"batched_grouped_1x1_with_input_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .input_pixel_stride(47));

  tests.push_back({"batched_grouped_1x1_with_output_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr + 3)
          .output_pixel_stride(gemm_config->nr * 2 + 13));

  tests.push_back({"batched_grouped_1x1_with_qmin", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmin(128));

  tests.push_back({"batched_grouped_1x1_with_qmax", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmax(128));

  tests.push_back({"batched_grouped_1x1_without_bias", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .has_bias(false)
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_size(1, 1)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  /**************************** CONV path ****************************/
  tests.push_back({"kernel_3x3", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"Kx3", {}});
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding_width(1)
            .kernel_size(kernel_height, 3)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_3xK", {}});
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding_height(1)
            .kernel_size(3, kernel_width)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_3x3_varying_height_padding", {}});
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      tests.back().second.push_back(
          DeconvolutionOperatorTester()
              .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
              .padding_width(1)
              .padding_top(padding_top)
              .padding_bottom(padding_bottom)
              .kernel_size(3, 3)
              .group_input_channels(15)
              .group_output_channels(gemm_config->nr * 2 + 3));
    }
  }

  tests.push_back({"kernel_3x3_varying_width_padding", {}});
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      tests.back().second.push_back(
          DeconvolutionOperatorTester()
              .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
              .padding_height(1)
              .padding_left(padding_left)
              .padding_right(padding_right)
              .kernel_size(3, 3)
              .group_input_channels(15)
              .group_output_channels(gemm_config->nr * 2 + 3));
    }
  }

  tests.push_back({"kernel_3x3_varying_height_adjustment", {}});
  for (size_t adjustment_height = 1; adjustment_height <= 2;
       adjustment_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .stride_height(adjustment_height + 1)
            .adjustment_height(adjustment_height)
            .kernel_size(3, 3)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_3x3_varying_width_adjustment", {}});
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .stride_width(adjustment_width + 1)
            .adjustment_width(adjustment_width)
            .kernel_size(3, 3)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_3x3_varying_input_height", {}});
  for (size_t input_height = kUnstridedInputHeight - 2;
       input_height <= kUnstridedInputHeight + 2; input_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(input_height, kUnstridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_3x3_varying_input_width", {}});
  for (size_t input_width = kUnstridedInputWidth - 2;
       input_width <= kUnstridedInputWidth + 2; input_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, input_width)
            .padding(1)
            .kernel_size(3, 3)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_3x3_varying_input_channels", {}});
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .group_input_channels(input_channels)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_3x3_varying_output_channels", {}});
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2;
       output_channels *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .group_input_channels(23)
            .group_output_channels(output_channels));
  }

  tests.push_back({"kernel_3x3_with_height_dilation", {}});
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .dilation_height(dilation_height)
            .group_input_channels(23)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_3x3_with_width_dilation", {}});
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .dilation_width(dilation_width)
            .group_input_channels(23)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_3x3_with_height_dilation_and_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .dilation_height(3)
          .stride_height(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"kernel_3x3_with_width_dilation_and_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .dilation_width(3)
          .stride_width(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"kernel_3x3_with_input_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .input_pixel_stride(28));

  tests.push_back({"kernel_3x3_with_output_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .output_pixel_stride(gemm_config->nr * 2 + 13));

  tests.push_back({"kernel_3x3_with_qmin", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmin(128));

  tests.push_back({"kernel_3x3_with_qmax", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmax(128));

  tests.push_back({"kernel_3x3_without_bias", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .has_bias(false)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"weights_cache_3x3", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .use_weights_cache(true));

  /**************************** CONV path, grouped ****************************/
  tests.push_back({"grouped_3x3", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"grouped_Kx3", {}});
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding_width(1)
            .kernel_size(kernel_height, 3)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_3xK", {}});
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding_height(1)
            .kernel_size(3, kernel_width)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_3x3_varying_height_padding", {}});
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      tests.back().second.push_back(
          DeconvolutionOperatorTester()
              .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
              .padding_width(1)
              .padding_top(padding_top)
              .padding_bottom(padding_bottom)
              .kernel_size(3, 3)
              .groups(2)
              .group_input_channels(15)
              .group_output_channels(gemm_config->nr * 2 + 3));
    }
  }

  tests.push_back({"grouped_3x3_varying_width_padding", {}});
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      tests.back().second.push_back(
          DeconvolutionOperatorTester()
              .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
              .padding_height(1)
              .padding_left(padding_left)
              .padding_right(padding_right)
              .kernel_size(3, 3)
              .groups(2)
              .group_input_channels(15)
              .group_output_channels(gemm_config->nr * 2 + 3));
    }
  }

  tests.push_back({"grouped_3x3_varying_height_adjustment", {}});
  for (size_t adjustment_height = 1; adjustment_height <= 2;
       adjustment_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .stride_height(adjustment_height + 1)
            .adjustment_height(adjustment_height)
            .kernel_size(3, 3)
            .groups(2)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_3x3_varying_width_adjustment", {}});
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .stride_width(adjustment_width + 1)
            .adjustment_width(adjustment_width)
            .kernel_size(3, 3)
            .groups(2)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_3x3_varying_input_height", {}});
  for (size_t input_height = kUnstridedInputHeight - 2;
       input_height <= kUnstridedInputHeight + 2; input_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(input_height, kUnstridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .groups(2)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_3x3_varying_input_width", {}});
  for (size_t input_width = kUnstridedInputWidth - 2;
       input_width <= kUnstridedInputWidth + 2; input_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, input_width)
            .padding(1)
            .kernel_size(3, 3)
            .groups(2)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_3x3_varying_input_channels", {}});
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .groups(2)
            .group_input_channels(input_channels)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_3x3_varying_output_channels", {}});
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2;
       output_channels *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .groups(2)
            .group_input_channels(23)
            .group_output_channels(output_channels));
  }

  tests.push_back({"grouped_3x3_with_height_dilation", {}});
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .dilation_height(dilation_height)
            .groups(2)
            .group_input_channels(23)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_3x3_with_width_dilation", {}});
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .dilation_width(dilation_width)
            .groups(2)
            .group_input_channels(23)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_3x3_with_height_dilation_and_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .dilation_height(3)
          .stride_height(2)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"grouped_3x3_with_width_dilation_and_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .dilation_width(3)
          .stride_width(2)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"grouped_3x3_with_input_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .input_pixel_stride(47));

  tests.push_back({"grouped_3x3_with_output_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr + 3)
          .output_pixel_stride(gemm_config->nr * 2 + 13));

  tests.push_back({"grouped_3x3_with_qmin", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmin(128));

  tests.push_back({"grouped_3x3_with_qmax", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmax(128));

  tests.push_back({"grouped_3x3_without_bias", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .has_bias(false)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"weights_cache_grouped_3x3", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .use_weights_cache(true));

  /**************************** CONV path, batched ****************************/
  tests.push_back({"batched_3x3", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"batched_Kx3", {}});
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding_width(1)
            .kernel_size(kernel_height, 3)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_3xK", {}});
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding_height(1)
            .kernel_size(3, kernel_width)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_3x3_varying_height_padding", {}});
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      tests.back().second.push_back(
          DeconvolutionOperatorTester()
              .batch_size(2)
              .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
              .padding_width(1)
              .padding_top(padding_top)
              .padding_bottom(padding_bottom)
              .kernel_size(3, 3)
              .group_input_channels(15)
              .group_output_channels(gemm_config->nr * 2 + 3));
    }
  }

  tests.push_back({"batched_3x3_varying_width_padding", {}});
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      tests.back().second.push_back(
          DeconvolutionOperatorTester()
              .batch_size(2)
              .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
              .padding_height(1)
              .padding_left(padding_left)
              .padding_right(padding_right)
              .kernel_size(3, 3)
              .group_input_channels(15)
              .group_output_channels(gemm_config->nr * 2 + 3));
    }
  }

  tests.push_back({"batched_3x3_varying_height_adjustment", {}});
  for (size_t adjustment_height = 1; adjustment_height <= 2;
       adjustment_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .stride_height(adjustment_height + 1)
            .adjustment_height(adjustment_height)
            .kernel_size(3, 3)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_3x3_varying_width_adjustment", {}});
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .stride_width(adjustment_width + 1)
            .adjustment_width(adjustment_width)
            .kernel_size(3, 3)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_3x3_varying_input_height", {}});
  for (size_t input_height = kUnstridedInputHeight - 2;
       input_height <= kUnstridedInputHeight + 2; input_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(input_height, kUnstridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_3x3_varying_input_width", {}});
  for (size_t input_width = kUnstridedInputWidth - 2;
       input_width <= kUnstridedInputWidth + 2; input_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, input_width)
            .padding(1)
            .kernel_size(3, 3)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_3x3_varying_input_channels", {}});
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .group_input_channels(input_channels)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_3x3_varying_output_channels", {}});
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2;
       output_channels *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .group_input_channels(23)
            .group_output_channels(output_channels));
  }

  tests.push_back({"batched_3x3_with_height_dilation", {}});
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .dilation_height(dilation_height)
            .group_input_channels(23)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_3x3_with_width_dilation", {}});
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .dilation_width(dilation_width)
            .group_input_channels(23)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_3x3_with_height_dilation_and_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .dilation_height(3)
          .stride_height(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"batched_3x3_with_width_dilation_and_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .dilation_width(3)
          .stride_width(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"batched_3x3_with_input_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .input_pixel_stride(28));

  tests.push_back({"batched_3x3_with_output_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .output_pixel_stride(gemm_config->nr * 2 + 13));

  tests.push_back({"batched_3x3_with_qmin", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmin(128));

  tests.push_back({"batched_3x3_with_qmax", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmax(128));

  tests.push_back({"batched_3x3_without_bias", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .has_bias(false)
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"weights_cache_batched_3x3", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .use_weights_cache(true));

  /*********************** CONV path, grouped, batched ************************/
  tests.push_back({"batched_grouped_3x3", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"batched_grouped_Kx3", {}});
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding_width(1)
            .kernel_size(kernel_height, 3)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_3xK", {}});
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding_height(1)
            .kernel_size(3, kernel_width)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_3x3_varying_height_padding", {}});
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      tests.back().second.push_back(
          DeconvolutionOperatorTester()
              .batch_size(2)
              .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
              .padding_width(1)
              .padding_top(padding_top)
              .padding_bottom(padding_bottom)
              .kernel_size(3, 3)
              .groups(2)
              .group_input_channels(15)
              .group_output_channels(gemm_config->nr * 2 + 3));
    }
  }

  tests.push_back({"batched_grouped_3x3_varying_width_padding", {}});
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      tests.back().second.push_back(
          DeconvolutionOperatorTester()
              .batch_size(2)
              .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
              .padding_height(1)
              .padding_left(padding_left)
              .padding_right(padding_right)
              .kernel_size(3, 3)
              .groups(2)
              .group_input_channels(15)
              .group_output_channels(gemm_config->nr * 2 + 3));
    }
  }

  tests.push_back({"batched_grouped_3x3_varying_height_adjustment", {}});
  for (size_t adjustment_height = 1; adjustment_height <= 2;
       adjustment_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .stride_height(adjustment_height + 1)
            .adjustment_height(adjustment_height)
            .kernel_size(3, 3)
            .groups(2)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_3x3_varying_width_adjustment", {}});
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .stride_width(adjustment_width + 1)
            .adjustment_width(adjustment_width)
            .kernel_size(3, 3)
            .groups(2)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_3x3_varying_input_height", {}});
  for (size_t input_height = kUnstridedInputHeight - 2;
       input_height <= kUnstridedInputHeight + 2; input_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(input_height, kUnstridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .groups(2)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_3x3_varying_input_width", {}});
  for (size_t input_width = kUnstridedInputWidth - 2;
       input_width <= kUnstridedInputWidth + 2; input_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, input_width)
            .padding(1)
            .kernel_size(3, 3)
            .groups(2)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_3x3_varying_input_channels", {}});
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .groups(2)
            .group_input_channels(input_channels)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_3x3_varying_output_channels", {}});
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2;
       output_channels *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .groups(2)
            .group_input_channels(23)
            .group_output_channels(output_channels));
  }

  tests.push_back({"batched_grouped_3x3_with_height_dilation", {}});
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .dilation_height(dilation_height)
            .groups(2)
            .group_input_channels(23)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_3x3_with_width_dilation", {}});
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .dilation_width(dilation_width)
            .groups(2)
            .group_input_channels(23)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_3x3_with_height_dilation_and_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .dilation_height(3)
          .stride_width(2)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"batched_grouped_3x3_with_width_dilation_and_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .dilation_width(3)
          .stride_width(2)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"batched_grouped_3x3_with_input_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .input_pixel_stride(47));

  tests.push_back({"batched_grouped_3x3_with_output_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr + 3)
          .output_pixel_stride(gemm_config->nr * 2 + 13));

  tests.push_back({"batched_grouped_3x3_with_qmin", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmin(128));

  tests.push_back({"batched_grouped_3x3_with_qmax", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmax(128));

  tests.push_back({"batched_grouped_3x3_without_bias", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .has_bias(false)
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"weights_cache_batched_grouped_3x3", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .use_weights_cache(true));

  /************************** SUBCONV2D/IGEMM path ****************************/
  tests.push_back({"kernel_3x3s2", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .group_input_channels(15)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"Kx3s2", {}});
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding_width(1)
            .kernel_size(kernel_height, 3)
            .stride(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_3xKs2", {}});
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding_height(1)
            .kernel_size(3, kernel_width)
            .stride(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_3x3sSx1", {}});
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .padding_width(1)
            .kernel_size(3, 3)
            .stride_height(stride_height)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_3x3s1xS", {}});
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .padding_width(1)
            .kernel_size(3, 3)
            .stride_width(stride_width)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_3x3s2_varying_height_padding", {}});
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      tests.back().second.push_back(
          DeconvolutionOperatorTester()
              .input_size(kStridedInputHeight, kStridedInputWidth)
              .padding_width(1)
              .padding_top(padding_top)
              .padding_bottom(padding_bottom)
              .kernel_size(3, 3)
              .stride(2)
              .group_input_channels(15)
              .group_output_channels(gemm_config->nr * 2 + 3));
    }
  }

  tests.push_back({"kernel_3x3s2_varying_width_padding", {}});
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      tests.back().second.push_back(
          DeconvolutionOperatorTester()
              .input_size(kStridedInputHeight, kStridedInputWidth)
              .padding_height(1)
              .padding_left(padding_left)
              .padding_right(padding_right)
              .kernel_size(3, 3)
              .stride(2)
              .group_input_channels(15)
              .group_output_channels(gemm_config->nr * 2 + 3));
    }
  }

  tests.push_back({"kernel_3x3s2_varying_height_adjustment", {}});
  for (size_t adjustment_height = 0; adjustment_height <= 1;
       adjustment_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .adjustment_height(adjustment_height)
            .kernel_size(3, 3)
            .stride(2)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_3x3s2_varying_width_adjustment", {}});
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .adjustment_width(adjustment_width)
            .kernel_size(3, 3)
            .stride(2)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_3x3s2_varying_input_height", {}});
  for (size_t input_height = kStridedInputHeight - 2;
       input_height <= kStridedInputHeight + 2; input_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(input_height, kStridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .stride(2)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_3x3s2_varying_input_width", {}});
  for (size_t input_width = kStridedInputWidth - 2;
       input_width <= kStridedInputWidth + 2; input_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .stride(2)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_3x3s2_varying_input_channels", {}});
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .stride(2)
            .group_input_channels(input_channels)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_3x3s2_varying_output_channels", {}});
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2;
       output_channels *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .stride(2)
            .group_input_channels(23)
            .group_output_channels(output_channels));
  }

  tests.push_back({"kernel_3x3s2_with_input_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .input_pixel_stride(28));

  tests.push_back({"kernel_3x3s2_with_output_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .output_pixel_stride(gemm_config->nr * 2 + 13));

  tests.push_back({"kernel_3x3s2_with_qmin", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmin(128));

  tests.push_back({"kernel_3x3s2_with_qmax", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmax(128));

  tests.push_back({"kernel_3x3s2_without_bias", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .has_bias(false)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"weights_cache_3x3s2", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .group_input_channels(15)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .use_weights_cache(true));

  /********************** SUBCONV2D/IGEMM path, grouped **********************/
  tests.push_back({"grouped_3x3s2", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"grouped_Kx3s2", {}});
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding_width(1)
            .kernel_size(kernel_height, 3)
            .stride(2)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_3xKs2", {}});
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding_height(1)
            .kernel_size(3, kernel_width)
            .stride(2)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_3x3sSx1", {}});
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .padding_width(1)
            .kernel_size(3, 3)
            .stride_height(stride_height)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_3x3s1xS", {}});
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .padding_width(1)
            .kernel_size(3, 3)
            .stride_width(stride_width)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_3x3s2_varying_height_padding", {}});
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      tests.back().second.push_back(
          DeconvolutionOperatorTester()
              .input_size(kStridedInputHeight, kStridedInputWidth)
              .padding_width(1)
              .padding_top(padding_top)
              .padding_bottom(padding_bottom)
              .kernel_size(3, 3)
              .stride(2)
              .groups(2)
              .group_input_channels(17)
              .group_output_channels(gemm_config->nr * 2 + 3));
    }
  }

  tests.push_back({"grouped_3x3s2_varying_width_padding", {}});
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      tests.back().second.push_back(
          DeconvolutionOperatorTester()
              .input_size(kStridedInputHeight, kStridedInputWidth)
              .padding_height(1)
              .padding_left(padding_left)
              .padding_right(padding_right)
              .kernel_size(3, 3)
              .stride(2)
              .groups(2)
              .group_input_channels(17)
              .group_output_channels(gemm_config->nr * 2 + 3));
    }
  }

  tests.push_back({"grouped_3x3s2_varying_height_adjustment", {}});
  for (size_t adjustment_height = 0; adjustment_height <= 1;
       adjustment_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .adjustment_height(adjustment_height)
            .kernel_size(3, 3)
            .stride(2)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_3x3s2_varying_width_adjustment", {}});
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .adjustment_width(adjustment_width)
            .kernel_size(3, 3)
            .stride(2)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_3x3s2_varying_input_height", {}});
  for (size_t input_height = kStridedInputHeight - 2;
       input_height <= kStridedInputHeight + 2; input_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(input_height, kStridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .stride(2)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_3x3s2_varying_input_width", {}});
  for (size_t input_width = kStridedInputWidth - 2;
       input_width <= kStridedInputWidth + 2; input_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .stride(2)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_3x3s2_varying_input_channels", {}});
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .stride(2)
            .groups(2)
            .group_input_channels(input_channels)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_3x3s2_varying_output_channels", {}});
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2;
       output_channels *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .stride(2)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(output_channels));
  }

  tests.push_back({"grouped_3x3s2_with_input_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .input_pixel_stride(37));

  tests.push_back({"grouped_3x3s2_with_output_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr + 3)
          .output_pixel_stride(gemm_config->nr * 2 + 13));

  tests.push_back({"grouped_3x3s2_with_qmin", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmin(128));

  tests.push_back({"grouped_3x3s2_with_qmax", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmax(128));

  tests.push_back({"grouped_3x3s2_without_bias", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .has_bias(false)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"weights_cache_grouped_3x3s2", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .use_weights_cache(true));

  /********************** SUBCONV2D/IGEMM path, batched ***********************/
  tests.push_back({"batched_3x3s2", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .group_input_channels(15)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"batched_Kx3s2", {}});
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding_width(1)
            .kernel_size(kernel_height, 3)
            .stride(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_3xKs2", {}});
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding_height(1)
            .kernel_size(3, kernel_width)
            .stride(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_3x3sSx1", {}});
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .padding_width(1)
            .kernel_size(3, 3)
            .stride_height(stride_height)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_3x3s1xS", {}});
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .padding_width(1)
            .kernel_size(3, 3)
            .stride_width(stride_width)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_3x3s2_varying_height_padding", {}});
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      tests.back().second.push_back(
          DeconvolutionOperatorTester()
              .batch_size(2)
              .input_size(kStridedInputHeight, kStridedInputWidth)
              .padding_width(1)
              .padding_top(padding_top)
              .padding_bottom(padding_bottom)
              .kernel_size(3, 3)
              .stride(2)
              .group_input_channels(15)
              .group_output_channels(gemm_config->nr * 2 + 3));
    }
  }

  tests.push_back({"batched_3x3s2_varying_width_padding", {}});
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      tests.back().second.push_back(
          DeconvolutionOperatorTester()
              .batch_size(2)
              .input_size(kStridedInputHeight, kStridedInputWidth)
              .padding_height(1)
              .padding_left(padding_left)
              .padding_right(padding_right)
              .kernel_size(3, 3)
              .stride(2)
              .group_input_channels(15)
              .group_output_channels(gemm_config->nr * 2 + 3));
    }
  }

  tests.push_back({"batched_3x3s2_varying_height_adjustment", {}});
  for (size_t adjustment_height = 0; adjustment_height <= 1;
       adjustment_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .adjustment_height(adjustment_height)
            .kernel_size(3, 3)
            .stride(2)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_3x3s2_varying_width_adjustment", {}});
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .adjustment_width(adjustment_width)
            .kernel_size(3, 3)
            .stride(2)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_3x3s2_varying_input_height", {}});
  for (size_t input_height = kStridedInputHeight - 2;
       input_height <= kStridedInputHeight + 2; input_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(input_height, kStridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .stride(2)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_3x3s2_varying_input_width", {}});
  for (size_t input_width = kStridedInputWidth - 2;
       input_width <= kStridedInputWidth + 2; input_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .stride(2)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_3x3s2_varying_input_channels", {}});
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .stride(2)
            .group_input_channels(input_channels)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_3x3s2_varying_output_channels", {}});
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2;
       output_channels *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .stride(2)
            .group_input_channels(23)
            .group_output_channels(output_channels));
  }

  tests.push_back({"batched_3x3s2_with_input_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .input_pixel_stride(28));

  tests.push_back({"batched_3x3s2_with_output_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .output_pixel_stride(gemm_config->nr * 2 + 13));

  tests.push_back({"batched_3x3s2_with_qmin", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmin(128));

  tests.push_back({"batched_3x3s2_with_qmax", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmax(128));

  tests.push_back({"batched_3x3s2_without_bias", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .has_bias(false)
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"weights_cache_batched_3x3s2", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .group_input_channels(15)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .use_weights_cache(true));

  /****************** SUBCONV2D/IGEMM path, grouped, batched ******************/
  tests.push_back({"batched_grouped_3x3s2", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"batched_grouped_Kx3s2", {}});
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding_width(1)
            .kernel_size(kernel_height, 3)
            .stride(2)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_3xKs2", {}});
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding_height(1)
            .kernel_size(3, kernel_width)
            .stride(2)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_3x3sSx1", {}});
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .padding_width(1)
            .kernel_size(3, 3)
            .stride_height(stride_height)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_3x3s1xS", {}});
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .padding_width(1)
            .kernel_size(3, 3)
            .stride_width(stride_width)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_3x3s2_varying_height_padding", {}});
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      tests.back().second.push_back(
          DeconvolutionOperatorTester()
              .batch_size(2)
              .input_size(kStridedInputHeight, kStridedInputWidth)
              .padding_width(1)
              .padding_top(padding_top)
              .padding_bottom(padding_bottom)
              .kernel_size(3, 3)
              .stride(2)
              .groups(2)
              .group_input_channels(17)
              .group_output_channels(gemm_config->nr * 2 + 3));
    }
  }

  tests.push_back({"batched_grouped_3x3s2_varying_width_padding", {}});
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      tests.back().second.push_back(
          DeconvolutionOperatorTester()
              .batch_size(2)
              .input_size(kStridedInputHeight, kStridedInputWidth)
              .padding_height(1)
              .padding_left(padding_left)
              .padding_right(padding_right)
              .kernel_size(3, 3)
              .stride(2)
              .groups(2)
              .group_input_channels(17)
              .group_output_channels(gemm_config->nr * 2 + 3));
    }
  }

  tests.push_back({"batched_grouped_3x3s2_varying_height_adjustment", {}});
  for (size_t adjustment_height = 0; adjustment_height <= 1;
       adjustment_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .adjustment_height(adjustment_height)
            .kernel_size(3, 3)
            .stride(2)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_3x3s2_varying_width_adjustment", {}});
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .adjustment_width(adjustment_width)
            .kernel_size(3, 3)
            .stride(2)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_3x3s2_varying_input_height", {}});
  for (size_t input_height = kStridedInputHeight - 2;
       input_height <= kStridedInputHeight + 2; input_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(input_height, kStridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .stride(2)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_3x3s2_varying_input_width", {}});
  for (size_t input_width = kStridedInputWidth - 2;
       input_width <= kStridedInputWidth + 2; input_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .stride(2)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_3x3s2_varying_input_channels", {}});
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .stride(2)
            .groups(2)
            .group_input_channels(input_channels)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_3x3s2_varying_output_channels", {}});
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2;
       output_channels *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .padding(1)
            .kernel_size(3, 3)
            .stride(2)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(output_channels));
  }

  tests.push_back({"batched_grouped_3x3s2_with_input_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .input_pixel_stride(37));

  tests.push_back({"batched_grouped_3x3s2_with_output_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr + 3)
          .output_pixel_stride(gemm_config->nr * 2 + 13));

  tests.push_back({"batched_grouped_3x3s2_with_qmin", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmin(128));

  tests.push_back({"batched_grouped_3x3s2_with_qmax", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmax(128));

  tests.push_back({"batched_grouped_3x3s2_without_bias", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .has_bias(false)
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"weights_cache_batched_grouped_3x3s2", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .padding(1)
          .kernel_size(3, 3)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .use_weights_cache(true));

  /*************************** SUBCONV2D/GEMM path ****************************/
  tests.push_back({"kernel_2x2s2", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .group_input_channels(15)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"Kx2sKx2", {}});
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .kernel_size(kernel_height, 2)
            .stride(kernel_height, 2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_2xKs2xK", {}});
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .kernel_size(2, kernel_width)
            .stride(2, kernel_width)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_2x2s2_height_adjustment", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .adjustment_height(1)
          .kernel_size(2, 2)
          .stride(2)
          .group_input_channels(15)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"kernel_2x2s2_width_adjustment", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .adjustment_width(1)
          .kernel_size(2, 2)
          .stride(2)
          .group_input_channels(15)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"kernel_2x2s2_varying_input_height", {}});
  for (size_t input_height = kStridedInputHeight - 2;
       input_height <= kStridedInputHeight + 2; input_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(input_height, kStridedInputWidth)
            .kernel_size(2, 2)
            .stride(2)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_2x2s2_varying_input_width", {}});
  for (size_t input_width = kStridedInputWidth - 2;
       input_width <= kStridedInputWidth + 2; input_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .kernel_size(2, 2)
            .stride(2)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_2x2s2_varying_input_channels", {}});
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .kernel_size(2, 2)
            .stride(2)
            .group_input_channels(input_channels)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"kernel_2x2s2_varying_output_channels", {}});
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2;
       output_channels *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .kernel_size(2, 2)
            .stride(2)
            .group_input_channels(23)
            .group_output_channels(output_channels));
  }

  tests.push_back({"kernel_2x2s2_with_input_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .input_pixel_stride(28));

  tests.push_back({"kernel_2x2s2_with_output_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .output_pixel_stride(gemm_config->nr * 2 + 13));

  tests.push_back({"kernel_2x2s2_with_qmin", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmin(128));

  tests.push_back({"kernel_2x2s2_with_qmax", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmax(128));

  tests.push_back({"kernel_2x2s2_without_bias", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .has_bias(false)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"weights_cache_2x2s2", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .group_input_channels(15)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .use_weights_cache(true));

  /*********************** SUBCONV2D/GEMM path, grouped ***********************/
  tests.push_back({"grouped_2x2s2", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"grouped_Kx2sKx2", {}});
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .kernel_size(kernel_height, 2)
            .stride(kernel_height, 2)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_2xKs2xK", {}});
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .kernel_size(2, kernel_width)
            .stride(2, kernel_width)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_2x2s2_height_adjustment", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .adjustment_height(1)
          .kernel_size(2, 2)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"grouped_2x2s2_width_adjustment", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .adjustment_width(1)
          .kernel_size(2, 2)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"grouped_2x2s2_varying_input_height", {}});
  for (size_t input_height = kStridedInputHeight - 2;
       input_height <= kStridedInputHeight + 2; input_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(input_height, kStridedInputWidth)
            .kernel_size(2, 2)
            .stride(2)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_2x2s2_varying_input_width", {}});
  for (size_t input_width = kStridedInputWidth - 2;
       input_width <= kStridedInputWidth + 2; input_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .kernel_size(2, 2)
            .stride(2)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_2x2s2_varying_input_channels", {}});
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .kernel_size(2, 2)
            .stride(2)
            .groups(2)
            .group_input_channels(input_channels)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"grouped_2x2s2_varying_output_channels", {}});
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2;
       output_channels *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .kernel_size(2, 2)
            .stride(2)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(output_channels));
  }

  tests.push_back({"grouped_2x2s2_with_input_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .input_pixel_stride(37));

  tests.push_back({"grouped_2x2s2_with_output_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr + 3)
          .output_pixel_stride(gemm_config->nr * 2 + 13));

  tests.push_back({"grouped_2x2s2_with_qmin", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmin(128));

  tests.push_back({"grouped_2x2s2_with_qmax", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmax(128));

  tests.push_back({"grouped_2x2s2_without_bias", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .has_bias(false)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"weights_cache_grouped_2x2s2", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .use_weights_cache(true));

  /********************** SUBCONV2D/GEMM path, batched ************************/
  tests.push_back({"batched_2x2s2", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .group_input_channels(15)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"batched_Kx2sKx2", {}});
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .kernel_size(kernel_height, 2)
            .stride(kernel_height, 2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_2xKs2xK", {}});
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .kernel_size(2, kernel_width)
            .stride(2, kernel_width)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_2x2s2_height_adjustment", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .adjustment_height(1)
          .kernel_size(2, 2)
          .stride(2)
          .group_input_channels(15)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"batched_2x2s2_width_adjustment", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .adjustment_width(1)
          .kernel_size(2, 2)
          .stride(2)
          .group_input_channels(15)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"batched_2x2s2_varying_input_height", {}});
  for (size_t input_height = kStridedInputHeight - 2;
       input_height <= kStridedInputHeight + 2; input_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(input_height, kStridedInputWidth)
            .kernel_size(2, 2)
            .stride(2)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_2x2s2_varying_input_width", {}});
  for (size_t input_width = kStridedInputWidth - 2;
       input_width <= kStridedInputWidth + 2; input_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .kernel_size(2, 2)
            .stride(2)
            .group_input_channels(15)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_2x2s2_varying_input_channels", {}});
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .kernel_size(2, 2)
            .stride(2)
            .group_input_channels(input_channels)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_2x2s2_varying_output_channels", {}});
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2;
       output_channels *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .kernel_size(2, 2)
            .stride(2)
            .group_input_channels(23)
            .group_output_channels(output_channels));
  }

  tests.push_back({"batched_2x2s2_with_input_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .input_pixel_stride(28));

  tests.push_back({"batched_2x2s2_with_output_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .output_pixel_stride(gemm_config->nr * 2 + 13));

  tests.push_back({"batched_2x2s2_with_qmin", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmin(128));

  tests.push_back({"batched_2x2s2_with_qmax", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmax(128));

  tests.push_back({"batched_2x2s2_without_bias", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .has_bias(false)
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .group_input_channels(23)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"weights_cache_batched_2x2s2", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .group_input_channels(15)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .use_weights_cache(true));

  /****************** SUBCONV2D/GEMM path, grouped, batched *******************/
  tests.push_back({"batched_grouped_2x2s2", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"batched_grouped_Kx2sKx2", {}});
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .kernel_size(kernel_height, 2)
            .stride(kernel_height, 2)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_2xKs2xK", {}});
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .kernel_size(2, kernel_width)
            .stride(2, kernel_width)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_2x2s2_height_adjustment", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .adjustment_height(1)
          .kernel_size(2, 2)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"batched_grouped_2x2s2_width_adjustment", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .adjustment_width(1)
          .kernel_size(2, 2)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"batched_grouped_2x2s2_varying_input_height", {}});
  for (size_t input_height = kStridedInputHeight - 2;
       input_height <= kStridedInputHeight + 2; input_height++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(input_height, kStridedInputWidth)
            .kernel_size(2, 2)
            .stride(2)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_2x2s2_varying_input_width", {}});
  for (size_t input_width = kStridedInputWidth - 2;
       input_width <= kStridedInputWidth + 2; input_width++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .kernel_size(2, 2)
            .stride(2)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_2x2s2_varying_input_channels", {}});
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .kernel_size(2, 2)
            .stride(2)
            .groups(2)
            .group_input_channels(input_channels)
            .group_output_channels(gemm_config->nr * 2 + 3));
  }

  tests.push_back({"batched_grouped_2x2s2_varying_output_channels", {}});
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2;
       output_channels *= 2) {
    tests.back().second.push_back(
        DeconvolutionOperatorTester()
            .batch_size(2)
            .input_size(kStridedInputHeight, kStridedInputWidth)
            .kernel_size(2, 2)
            .stride(2)
            .groups(2)
            .group_input_channels(17)
            .group_output_channels(output_channels));
  }

  tests.push_back({"batched_grouped_2x2s2_with_input_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .input_pixel_stride(37));

  tests.push_back({"batched_grouped_2x2s2_with_output_stride", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr + 3)
          .output_pixel_stride(gemm_config->nr * 2 + 13));

  tests.push_back({"batched_grouped_2x2s2_with_qmin", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmin(128));

  tests.push_back({"batched_grouped_2x2s2_with_qmax", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .qmax(128));

  tests.push_back({"batched_grouped_2x2s2_without_bias", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .has_bias(false)
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3));

  tests.push_back({"weights_cache_batched_grouped_2x2s2", {}});
  tests.back().second.push_back(
      DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(gemm_config->nr * 2 + 3)
          .use_weights_cache(true));
  return tests;
}
static const DeconvolutionTestCase kDeconvolutionSetupTests[] = {
    /**************************** CONV path, setup ****************************/
    {"kernel_3x3_setup_changing_batch",
     {DeconvolutionOperatorTester()
          .batch_size(2)
          .next_batch_size(5)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .kernel_height(3)
          .kernel_width(5)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_3x3_setup_changing_height",
     {DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .next_input_height(kUnstridedInputHeight + 3)
          .kernel_height(3)
          .kernel_width(5)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_3x3_setup_changing_width",
     {DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
          .next_input_width(kUnstridedInputWidth + 3)
          .kernel_height(3)
          .kernel_width(5)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},

    /********************** SUBCONV2D/IGEMM path, setup ***********************/
    {"kernel_3x3s2_setup_changing_batch",
     {DeconvolutionOperatorTester()
          .batch_size(2)
          .next_batch_size(5)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_height(3)
          .kernel_width(5)
          .stride(2)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_3x3s2_setup_changing_height",
     {DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .next_input_height(kStridedInputHeight + 3)
          .kernel_height(3)
          .kernel_width(5)
          .stride(2)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_3x3s2_setup_changing_width",
     {DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .next_input_width(kStridedInputWidth + 3)
          .kernel_height(3)
          .kernel_width(5)
          .stride(2)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},

    /********************** SUBCONV2D/GEMM path, setup ************************/
    {"kernel_2x2s2_setup_changing_batch",
     {DeconvolutionOperatorTester()
          .batch_size(2)
          .next_batch_size(5)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .kernel_size(2, 2)
          .stride(2)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_2x2s2_setup_changing_height",
     {DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .next_input_height(kStridedInputHeight + 3)
          .kernel_size(2, 2)
          .stride(2)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_2x2s2_setup_changing_width",
     {DeconvolutionOperatorTester()
          .batch_size(2)
          .input_size(kStridedInputHeight, kStridedInputWidth)
          .next_input_width(kStridedInputWidth + 3)
          .kernel_size(2, 2)
          .stride(2)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
};

#define CREATE_DECONVOLUTION_TESTS(test_suite_name, gemm_config, test_fn) \
  GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(test_suite_name);         \
  using test_suite_name = testing::TestWithParam<DeconvolutionTestCase>;  \
  TEST_P(test_suite_name, DeconvolutionTest) {                            \
    const DeconvolutionTestCase& test_case = GetParam();                  \
    for (const DeconvolutionOperatorTester& tester : test_case.second) {  \
      tester.test_fn();                                                   \
    }                                                                     \
  }                                                                       \
  INSTANTIATE_TEST_SUITE_P(                                               \
      test_suite_name, test_suite_name,                                   \
      testing::ValuesIn(CreateDeconvolutionTests(gemm_config)),           \
      [](const testing::TestParamInfo<DeconvolutionTestCase>& info) {     \
        return info.param.first;                                          \
      });

#define CREATE_DECONVOLUTION_SETUP_TESTS(test_suite_name, setup_test_fn)  \
  GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SETUP_##test_suite_name); \
  using SETUP_##test_suite_name =                                         \
      testing::TestWithParam<DeconvolutionTestCase>;                      \
  TEST_P(SETUP_##test_suite_name, DeconvolutionSetupTest) {               \
    const DeconvolutionTestCase& test_case = GetParam();                  \
    for (const DeconvolutionOperatorTester& tester : test_case.second) {  \
      tester.setup_test_fn();                                             \
    }                                                                     \
  }                                                                       \
  INSTANTIATE_TEST_SUITE_P(                                               \
      SETUP_##test_suite_name, SETUP_##test_suite_name,                   \
      testing::ValuesIn<DeconvolutionTestCase>(kDeconvolutionSetupTests), \
      [](const testing::TestParamInfo<DeconvolutionTestCase>& info) {     \
        return info.param.first;                                          \
      });

CREATE_DECONVOLUTION_TESTS(DECONVOLUTION_NHWC_QC8,
                           xnn_init_qs8_qc8w_gemm_config(), TestQC8)
CREATE_DECONVOLUTION_SETUP_TESTS(DECONVOLUTION_NHWC_QC8, TestSetupQS8)

CREATE_DECONVOLUTION_TESTS(DECONVOLUTION_NHWC_QU8, xnn_init_qu8_gemm_config(),
                           TestQU8)
CREATE_DECONVOLUTION_SETUP_TESTS(DECONVOLUTION_NHWC_QU8, TestSetupQU8)

CREATE_DECONVOLUTION_TESTS(DECONVOLUTION_NHWC_F16, xnn_init_f16_gemm_config(),
                           TestF16)
CREATE_DECONVOLUTION_SETUP_TESTS(DECONVOLUTION_NHWC_F16, TestSetupF16)

CREATE_DECONVOLUTION_TESTS(DECONVOLUTION_NHWC_F32, xnn_init_f32_igemm_config(),
                           TestF32)
CREATE_DECONVOLUTION_SETUP_TESTS(DECONVOLUTION_NHWC_F32, TestSetupF32)
