// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/config.h>

#include "deconvolution-operator-tester.h"


constexpr size_t kUnstridedInputHeight = 8;
constexpr size_t kUnstridedInputWidth = 7;
constexpr size_t kStridedInputHeight = 6;
constexpr size_t kStridedInputWidth = 5;


/**************************** Future GEMM path ****************************/

TEST(DECONVOLUTION_NHWC_QS8, kernel_1x1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_1x1_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_1x1_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_1x1_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_1x1_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_1x1_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_1x1_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_1x1_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_1x1_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_1x1_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

/**************************** Future GEMM path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_QS8, grouped_1x1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_1x1_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_1x1_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_1x1_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_1x1_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_1x1_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_1x1_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_1x1_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_1x1_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_1x1_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

/**************************** Future GEMM path, batched ****************************/

TEST(DECONVOLUTION_NHWC_QS8, batched_1x1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_1x1_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_1x1_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_1x1_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_1x1_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_1x1_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_1x1_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_1x1_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_1x1_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_1x1_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

/**************************** Future GEMM path, batched, grouped ****************************/

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_1x1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_1x1_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_1x1_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_1x1_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_1x1_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_1x1_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_1x1_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_1x1_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_1x1_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_1x1_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

/**************************** CONV path ****************************/

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, Kx3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 1; adjustment_height <= 2; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_height(adjustment_height + 1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_width(adjustment_width + 1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3_with_height_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3_with_width_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3_with_height_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_height(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3_with_width_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, weights_cache_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQS8();
}

/**************************** CONV path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_Kx3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .groups(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .groups(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 1; adjustment_height <= 2; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_height(adjustment_height + 1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_width(adjustment_width + 1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3_with_height_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3_with_width_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3_with_height_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_height(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3_with_width_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, weights_cache_grouped_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQS8();
}

/**************************** CONV path, batched ****************************/

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_Kx3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 1; adjustment_height <= 2; adjustment_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_height(adjustment_height + 1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_width(adjustment_width + 1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3_with_height_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3_with_width_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3_with_height_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_height(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3_with_width_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, weights_cache_batched_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQS8();
}

/**************************** CONV path, grouped, batched ****************************/

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_Kx3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .groups(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .groups(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 1; adjustment_height <= 2; adjustment_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_height(adjustment_height + 1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_width(adjustment_width + 1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3_with_height_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3_with_width_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3_with_height_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_width(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3_with_width_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, weights_cache_batched_grouped_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQS8();
}

/**************************** CONV path, setup ****************************/

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3_setup_changing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .next_batch_size(5)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_height(3)
    .kernel_width(5)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3_setup_changing_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .next_input_height(kUnstridedInputHeight + 3)
    .kernel_height(3)
    .kernel_width(5)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3_setup_changing_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .next_input_width(kUnstridedInputWidth + 3)
    .kernel_height(3)
    .kernel_width(5)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupQS8();
}

/**************************** SUBCONV2D/IGEMM path ****************************/

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, Kx3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3xKs2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3sSx1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_height(stride_height)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3s1xS) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_width(stride_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3s2_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .stride(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3s2_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .stride(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3s2_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 0; adjustment_height <= 1; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3s2_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, weights_cache_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQS8();
}

/**************************** SUBCONV2D/IGEMM path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_Kx3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3xKs2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3sSx1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_height(stride_height)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3s1xS) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_width(stride_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3s2_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .stride(2)
        .groups(2)
        .group_input_channels(17)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3s2_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .stride(2)
        .groups(2)
        .group_input_channels(17)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3s2_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 0; adjustment_height <= 1; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3s2_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_3x3s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, weights_cache_grouped_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQS8();
}

/**************************** SUBCONV2D/IGEMM path, batched ****************************/

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_Kx3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3xKs2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3sSx1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_height(stride_height)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3s1xS) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_width(stride_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3s2_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .stride(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3s2_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .stride(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3s2_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 0; adjustment_height <= 1; adjustment_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3s2_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_3x3s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, weights_cache_batched_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQS8();
}

/**************************** SUBCONV2D/IGEMM path, grouped, batched ****************************/

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_Kx3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3xKs2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3sSx1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_height(stride_height)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3s1xS) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_width(stride_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3s2_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
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
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3s2_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
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
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3s2_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 0; adjustment_height <= 1; adjustment_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3s2_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_3x3s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, weights_cache_batched_grouped_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQS8();
}

/**************************** SUBCONV2D/IGEMM path, setup ****************************/

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3s2_setup_changing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .next_batch_size(5)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_height(3)
    .kernel_width(5)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3s2_setup_changing_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .next_input_height(kStridedInputHeight + 3)
    .kernel_height(3)
    .kernel_width(5)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_3x3s2_setup_changing_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .next_input_width(kStridedInputWidth + 3)
    .kernel_height(3)
    .kernel_width(5)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupQS8();
}

/**************************** SUBCONV2D/GEMM path ****************************/

TEST(DECONVOLUTION_NHWC_QS8, kernel_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, Kx2sKx2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_2xKs2xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_2x2s2_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_2x2s2_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_2x2s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_2x2s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_2x2s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_2x2s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_2x2s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_2x2s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_2x2s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_2x2s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_2x2s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, weights_cache_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQS8();
}

/**************************** SUBCONV2D/GEMM path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_QS8, grouped_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_Kx2sKx2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_2xKs2xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_2x2s2_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_2x2s2_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_2x2s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_2x2s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_2x2s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_2x2s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_2x2s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_2x2s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_2x2s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_2x2s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, grouped_2x2s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, weights_cache_grouped_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQS8();
}

/**************************** SUBCONV2D/GEMM path, batched ****************************/

TEST(DECONVOLUTION_NHWC_QS8, batched_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_Kx2sKx2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_2xKs2xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_2x2s2_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_2x2s2_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_2x2s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_2x2s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_2x2s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_2x2s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_2x2s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_2x2s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_2x2s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_2x2s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_2x2s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, weights_cache_batched_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQS8();
}

/**************************** SUBCONV2D/GEMM path, grouped, batched ****************************/

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_Kx2sKx2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_2xKs2xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_2x2s2_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_2x2s2_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_2x2s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_2x2s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_2x2s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_2x2s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQS8();
  }
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_2x2s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_2x2s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_2x2s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_2x2s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, batched_grouped_2x2s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, weights_cache_batched_grouped_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQS8();
}

/**************************** SUBCONV2D/GEMM path, setup ****************************/

TEST(DECONVOLUTION_NHWC_QS8, kernel_2x2s2_setup_changing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .next_batch_size(5)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_2x2s2_setup_changing_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .next_input_height(kStridedInputHeight + 3)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupQS8();
}

TEST(DECONVOLUTION_NHWC_QS8, kernel_2x2s2_setup_changing_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .next_input_width(kStridedInputWidth + 3)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupQS8();
}

/**************************** Future GEMM path ****************************/

TEST(DECONVOLUTION_NHWC_QU8, kernel_1x1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_1x1_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_1x1_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_1x1_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_1x1_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_1x1_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_1x1_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_1x1_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_1x1_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_1x1_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

/**************************** Future GEMM path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_QU8, grouped_1x1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_1x1_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_1x1_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_1x1_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_1x1_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_1x1_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_1x1_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_1x1_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_1x1_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_1x1_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

/**************************** Future GEMM path, batched ****************************/

TEST(DECONVOLUTION_NHWC_QU8, batched_1x1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_1x1_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_1x1_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_1x1_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_1x1_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_1x1_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_1x1_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_1x1_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_1x1_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_1x1_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

/**************************** Future GEMM path, batched, grouped ****************************/

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_1x1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_1x1_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_1x1_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_1x1_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_1x1_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_1x1_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_1x1_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_1x1_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_1x1_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_1x1_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

/**************************** CONV path ****************************/

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, Kx3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQU8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQU8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 1; adjustment_height <= 2; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_height(adjustment_height + 1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_width(adjustment_width + 1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3_with_height_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3_with_width_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3_with_height_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_height(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3_with_width_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, weights_cache_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQU8();
}

/**************************** CONV path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_Kx3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .groups(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQU8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .groups(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQU8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 1; adjustment_height <= 2; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_height(adjustment_height + 1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_width(adjustment_width + 1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3_with_height_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3_with_width_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3_with_height_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_height(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3_with_width_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, weights_cache_grouped_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQU8();
}

/**************************** CONV path, batched ****************************/

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_Kx3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQU8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQU8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 1; adjustment_height <= 2; adjustment_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_height(adjustment_height + 1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_width(adjustment_width + 1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3_with_height_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3_with_width_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3_with_height_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_height(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3_with_width_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, weights_cache_batched_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQU8();
}

/**************************** CONV path, grouped, batched ****************************/

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_Kx3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .groups(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQU8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .groups(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQU8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 1; adjustment_height <= 2; adjustment_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_height(adjustment_height + 1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_width(adjustment_width + 1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3_with_height_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3_with_width_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3_with_height_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_width(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3_with_width_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, weights_cache_batched_grouped_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQU8();
}

/**************************** CONV path, setup ****************************/

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3_setup_changing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .next_batch_size(5)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_height(3)
    .kernel_width(5)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3_setup_changing_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .next_input_height(kUnstridedInputHeight + 3)
    .kernel_height(3)
    .kernel_width(5)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3_setup_changing_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .next_input_width(kUnstridedInputWidth + 3)
    .kernel_height(3)
    .kernel_width(5)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupQU8();
}

/**************************** SUBCONV2D/IGEMM path ****************************/

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, Kx3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3xKs2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3sSx1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_height(stride_height)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3s1xS) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_width(stride_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3s2_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .stride(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQU8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3s2_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .stride(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQU8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3s2_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 0; adjustment_height <= 1; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3s2_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, weights_cache_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQU8();
}

/**************************** SUBCONV2D/IGEMM path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_Kx3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3xKs2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3sSx1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_height(stride_height)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3s1xS) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_width(stride_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3s2_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .stride(2)
        .groups(2)
        .group_input_channels(17)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQU8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3s2_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .stride(2)
        .groups(2)
        .group_input_channels(17)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQU8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3s2_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 0; adjustment_height <= 1; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3s2_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_3x3s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, weights_cache_grouped_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQU8();
}

/**************************** SUBCONV2D/IGEMM path, batched ****************************/

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_Kx3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3xKs2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3sSx1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_height(stride_height)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3s1xS) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_width(stride_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3s2_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .stride(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQU8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3s2_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .stride(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQU8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3s2_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 0; adjustment_height <= 1; adjustment_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3s2_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_3x3s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, weights_cache_batched_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQU8();
}

/**************************** SUBCONV2D/IGEMM path, grouped, batched ****************************/

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_Kx3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3xKs2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3sSx1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_height(stride_height)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3s1xS) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_width(stride_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3s2_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
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
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQU8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3s2_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
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
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestQU8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3s2_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 0; adjustment_height <= 1; adjustment_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3s2_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_3x3s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, weights_cache_batched_grouped_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQU8();
}

/**************************** SUBCONV2D/IGEMM path, setup ****************************/

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3s2_setup_changing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .next_batch_size(5)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_height(3)
    .kernel_width(5)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3s2_setup_changing_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .next_input_height(kStridedInputHeight + 3)
    .kernel_height(3)
    .kernel_width(5)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_3x3s2_setup_changing_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .next_input_width(kStridedInputWidth + 3)
    .kernel_height(3)
    .kernel_width(5)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupQU8();
}

/**************************** SUBCONV2D/GEMM path ****************************/

TEST(DECONVOLUTION_NHWC_QU8, kernel_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, Kx2sKx2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_2xKs2xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_2x2s2_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_2x2s2_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_2x2s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_2x2s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_2x2s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_2x2s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_2x2s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_2x2s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_2x2s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_2x2s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_2x2s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, weights_cache_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQU8();
}

/**************************** SUBCONV2D/GEMM path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_QU8, grouped_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_Kx2sKx2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_2xKs2xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_2x2s2_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_2x2s2_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_2x2s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_2x2s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_2x2s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_2x2s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_2x2s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_2x2s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_2x2s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_2x2s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, grouped_2x2s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, weights_cache_grouped_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQU8();
}

/**************************** SUBCONV2D/GEMM path, batched ****************************/

TEST(DECONVOLUTION_NHWC_QU8, batched_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_Kx2sKx2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_2xKs2xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_2x2s2_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_2x2s2_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_2x2s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_2x2s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_2x2s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_2x2s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_2x2s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_2x2s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_2x2s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_2x2s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_2x2s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, weights_cache_batched_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQU8();
}

/**************************** SUBCONV2D/GEMM path, grouped, batched ****************************/

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_Kx2sKx2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_2xKs2xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_2x2s2_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_2x2s2_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_2x2s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_2x2s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_2x2s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_2x2s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQU8();
  }
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_2x2s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_2x2s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_2x2s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_2x2s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, batched_grouped_2x2s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, weights_cache_batched_grouped_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestQU8();
}

/**************************** SUBCONV2D/GEMM path, setup ****************************/

TEST(DECONVOLUTION_NHWC_QU8, kernel_2x2s2_setup_changing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .next_batch_size(5)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_2x2s2_setup_changing_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .next_input_height(kStridedInputHeight + 3)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupQU8();
}

TEST(DECONVOLUTION_NHWC_QU8, kernel_2x2s2_setup_changing_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .next_input_width(kStridedInputWidth + 3)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupQU8();
}

/**************************** Future GEMM path ****************************/

TEST(DECONVOLUTION_NHWC_F16, kernel_1x1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_1x1_with_fp32_weights) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .weights_type(DeconvolutionOperatorTester::WeightsType::FP32)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_1x1_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_1x1_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_1x1_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_1x1_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_1x1_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_1x1_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_1x1_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_1x1_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_1x1_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

#if (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_JIT // TODO(b/287020333)
TEST(DECONVOLUTION_NHWC_F16, jit_1x1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_jit(true)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, jit_1x1_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .use_jit(true)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, jit_1x1_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .use_jit(true)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, jit_1x1_with_relu) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .activation(DeconvolutionOperatorTester::Activation::Relu)
    .use_jit(true)
    .iterations(3)
    .TestF16();
}
#endif  // (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_JIT

/**************************** Future GEMM path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_F16, grouped_1x1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_1x1_with_fp32_weights) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .weights_type(DeconvolutionOperatorTester::WeightsType::FP32)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_1x1_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_1x1_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_1x1_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_1x1_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_1x1_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_1x1_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_1x1_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_1x1_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_1x1_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

/**************************** Future GEMM path, batched ****************************/

TEST(DECONVOLUTION_NHWC_F16, batched_1x1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_1x1_with_fp32_weights) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .weights_type(DeconvolutionOperatorTester::WeightsType::FP32)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_1x1_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_1x1_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_1x1_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_1x1_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_1x1_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_1x1_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_1x1_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_1x1_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_1x1_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

/**************************** Future GEMM path, batched, grouped ****************************/

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_1x1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_1x1_with_fp32_weights) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .weights_type(DeconvolutionOperatorTester::WeightsType::FP32)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_1x1_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_1x1_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_1x1_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_1x1_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_1x1_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_1x1_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_1x1_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_1x1_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_1x1_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

/**************************** CONV path ****************************/

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3_with_fp32_weights) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .weights_type(DeconvolutionOperatorTester::WeightsType::FP32)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, Kx3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t adjustment_height = 1; adjustment_height <= 2; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_height(adjustment_height + 1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_width(adjustment_width + 1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3_with_height_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3_with_width_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3_with_height_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_height(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3_with_width_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, weights_cache_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF16();
}

#if (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_JIT // TODO(b/287020333)
TEST(DECONVOLUTION_NHWC_F16, jit_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_jit(true)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, jit_3x3_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .use_jit(true)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, jit_3x3_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .use_jit(true)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, jit_3x3_with_relu) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .activation(DeconvolutionOperatorTester::Activation::Relu)
    .use_jit(true)
    .iterations(3)
    .TestF16();
}
#endif  // (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_JIT

/**************************** CONV path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3_with_fp32_weights) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .weights_type(DeconvolutionOperatorTester::WeightsType::FP32)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_Kx3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .groups(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .groups(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t adjustment_height = 1; adjustment_height <= 2; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_height(adjustment_height + 1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_width(adjustment_width + 1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3_with_height_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3_with_width_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3_with_height_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_height(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3_with_width_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, weights_cache_grouped_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF16();
}

/**************************** CONV path, batched ****************************/

TEST(DECONVOLUTION_NHWC_F16, batched_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3_with_fp32_weights) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .weights_type(DeconvolutionOperatorTester::WeightsType::FP32)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_Kx3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t adjustment_height = 1; adjustment_height <= 2; adjustment_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_height(adjustment_height + 1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_width(adjustment_width + 1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3_with_height_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3_with_width_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3_with_height_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_height(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3_with_width_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, weights_cache_batched_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF16();
}

/**************************** CONV path, grouped, batched ****************************/

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3_with_fp32_weights) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .weights_type(DeconvolutionOperatorTester::WeightsType::FP32)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_Kx3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .groups(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .groups(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t adjustment_height = 1; adjustment_height <= 2; adjustment_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_height(adjustment_height + 1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_width(adjustment_width + 1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3_with_height_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3_with_width_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3_with_height_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_width(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3_with_width_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, weights_cache_batched_grouped_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF16();
}

/**************************** CONV path, setup ****************************/

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3_setup_changing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .next_batch_size(5)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_height(3)
    .kernel_width(5)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3_setup_changing_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .next_input_height(kUnstridedInputHeight + 3)
    .kernel_height(3)
    .kernel_width(5)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3_setup_changing_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .next_input_width(kUnstridedInputWidth + 3)
    .kernel_height(3)
    .kernel_width(5)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupF16();
}

/**************************** SUBCONV2D/IGEMM path ****************************/

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3s2_with_fp32_weights) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .weights_type(DeconvolutionOperatorTester::WeightsType::FP32)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, Kx3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3xKs2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3sSx1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_height(stride_height)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3s1xS) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_width(stride_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3s2_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .stride(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3s2_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .stride(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3s2_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t adjustment_height = 0; adjustment_height <= 1; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3s2_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, weights_cache_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF16();
}

#if (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_JIT // TODO(b/287020333)
TEST(DECONVOLUTION_NHWC_F16, jit_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_jit(true)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, jit_3x3s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .use_jit(true)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, jit_3x3s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .use_jit(true)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, jit_3x3s2_with_relu) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .activation(DeconvolutionOperatorTester::Activation::Relu)
    .use_jit(true)
    .iterations(3)
    .TestF16();
}
#endif  // (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_JIT

/**************************** SUBCONV2D/IGEMM path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3s2_with_fp32_weights) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .weights_type(DeconvolutionOperatorTester::WeightsType::FP32)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_Kx3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3xKs2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3sSx1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_height(stride_height)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3s1xS) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_width(stride_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3s2_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .stride(2)
        .groups(2)
        .group_input_channels(17)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3s2_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .stride(2)
        .groups(2)
        .group_input_channels(17)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3s2_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t adjustment_height = 0; adjustment_height <= 1; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3s2_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_3x3s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, weights_cache_grouped_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF16();
}

/**************************** SUBCONV2D/IGEMM path, batched ****************************/

TEST(DECONVOLUTION_NHWC_F16, batched_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3s2_with_fp32_weights) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .weights_type(DeconvolutionOperatorTester::WeightsType::FP32)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_Kx3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3xKs2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3sSx1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_height(stride_height)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3s1xS) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_width(stride_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3s2_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .stride(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3s2_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .stride(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3s2_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t adjustment_height = 0; adjustment_height <= 1; adjustment_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3s2_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_3x3s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, weights_cache_batched_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF16();
}

/**************************** SUBCONV2D/IGEMM path, grouped, batched ****************************/

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3s2_with_fp32_weights) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .weights_type(DeconvolutionOperatorTester::WeightsType::FP32)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_Kx3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3xKs2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3sSx1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_height(stride_height)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3s1xS) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_width(stride_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3s2_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
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
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3s2_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
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
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3s2_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t adjustment_height = 0; adjustment_height <= 1; adjustment_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3s2_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_3x3s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, weights_cache_batched_grouped_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF16();
}

/**************************** SUBCONV2D/IGEMM path, setup ****************************/


TEST(DECONVOLUTION_NHWC_F16, kernel_3x3s2_setup_changing_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .next_input_height(kStridedInputHeight + 3)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_3x3s2_setup_changing_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .next_input_width(kStridedInputWidth + 3)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupF16();
}

/**************************** SUBCONV2D/GEMM path ****************************/

TEST(DECONVOLUTION_NHWC_F16, kernel_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_2x2s2_with_fp32_weights) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .weights_type(DeconvolutionOperatorTester::WeightsType::FP32)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, Kx2sKx2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_2xKs2xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_2x2s2_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_2x2s2_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_2x2s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_2x2s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_2x2s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_2x2s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, kernel_2x2s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_2x2s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_2x2s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_2x2s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_2x2s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, weights_cache_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF16();
}

#if (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_JIT // TODO(b/287020333)
TEST(DECONVOLUTION_NHWC_F16, jit_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_jit(true)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, jit_2x2s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .use_jit(true)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, jit_2x2s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .use_jit(true)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, jit_2x2s2_with_relu) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .activation(DeconvolutionOperatorTester::Activation::Relu)
    .use_jit(true)
    .iterations(3)
    .TestF16();
}
#endif  // (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_JIT

/**************************** SUBCONV2D/GEMM path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_F16, grouped_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_2x2s2_with_fp32_weights) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .weights_type(DeconvolutionOperatorTester::WeightsType::FP32)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_Kx2sKx2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_2xKs2xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_2x2s2_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_2x2s2_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_2x2s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_2x2s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_2x2s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_2x2s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, grouped_2x2s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_2x2s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_2x2s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_2x2s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, grouped_2x2s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, weights_cache_grouped_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF16();
}

/**************************** SUBCONV2D/GEMM path, batched ****************************/

TEST(DECONVOLUTION_NHWC_F16, batched_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_2x2s2_with_fp32_weights) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .weights_type(DeconvolutionOperatorTester::WeightsType::FP32)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_Kx2sKx2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_2xKs2xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_2x2s2_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_2x2s2_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_2x2s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_2x2s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_2x2s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_2x2s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_2x2s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_2x2s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_2x2s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_2x2s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_2x2s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, weights_cache_batched_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF16();
}

/**************************** SUBCONV2D/GEMM path, grouped, batched ****************************/

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_2x2s2_with_fp32_weights) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .weights_type(DeconvolutionOperatorTester::WeightsType::FP32)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_Kx2sKx2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_2xKs2xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_2x2s2_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_2x2s2_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_2x2s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_2x2s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_2x2s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_2x2s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF16();
  }
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_2x2s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_2x2s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_2x2s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_2x2s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, batched_grouped_2x2s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF16();
}

TEST(DECONVOLUTION_NHWC_F16, weights_cache_batched_grouped_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF16();
}

// /**************************** SUBCONV2D/GEMM path, setup ****************************/

TEST(DECONVOLUTION_NHWC_F16, kernel_2x2s2_setup_changing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .next_batch_size(5)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_2x2s2_setup_changing_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .next_input_height(kStridedInputHeight + 3)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupF16();
}

TEST(DECONVOLUTION_NHWC_F16, kernel_2x2s2_setup_changing_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .next_input_width(kStridedInputWidth + 3)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupF16();
}

/**************************** Future GEMM path ****************************/

TEST(DECONVOLUTION_NHWC_F32, kernel_1x1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_1x1_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_1x1_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_1x1_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_1x1_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_1x1_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_1x1_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_1x1_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_1x1_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_1x1_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

#if !XNN_ARCH_WASM && XNN_ENABLE_JIT //  TODO(b/290880274)
TEST(DECONVOLUTION_NHWC_F32, jit_1x1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_jit(true)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, jit_1x1_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .use_jit(true)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, jit_1x1_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .use_jit(true)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, jit_1x1_with_relu) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .activation(DeconvolutionOperatorTester::Activation::Relu)
    .use_jit(true)
    .iterations(3)
    .TestF32();
}
#endif  // !XNN_ARCH_WASM && XNN_ENABLE_JIT

/**************************** Future GEMM path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_F32, grouped_1x1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_1x1_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_1x1_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_1x1_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_1x1_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_1x1_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_1x1_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_1x1_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_1x1_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_1x1_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

/**************************** Future GEMM path, batched ****************************/

TEST(DECONVOLUTION_NHWC_F32, batched_1x1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_1x1_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_1x1_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_1x1_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_1x1_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_1x1_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_1x1_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_1x1_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_1x1_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_1x1_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

/**************************** Future GEMM path, batched, grouped ****************************/

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_1x1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_1x1_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_1x1_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_1x1_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_1x1_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_1x1_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_1x1_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_1x1_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_1x1_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_1x1_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

/**************************** CONV path ****************************/

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, Kx3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 1; adjustment_height <= 2; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_height(adjustment_height + 1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_width(adjustment_width + 1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3_with_height_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3_with_width_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3_with_height_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_height(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3_with_width_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, weights_cache_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF32();
}

#if !XNN_ARCH_WASM && XNN_ENABLE_JIT //  TODO(b/290880274)
TEST(DECONVOLUTION_NHWC_F32, jit_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_jit(true)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, jit_3x3_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .use_jit(true)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, jit_3x3_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .use_jit(true)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, jit_3x3_with_relu) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .activation(DeconvolutionOperatorTester::Activation::Relu)
    .use_jit(true)
    .iterations(3)
    .TestF32();
}
#endif //  !XNN_ARCH_WASM && XNN_ENABLE_JIT

/**************************** CONV path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_Kx3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .groups(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .groups(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 1; adjustment_height <= 2; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_height(adjustment_height + 1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_width(adjustment_width + 1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_with_height_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_with_width_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_with_height_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_height(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_with_width_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, weights_cache_grouped_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF32();
}

/**************************** CONV path, batched ****************************/

TEST(DECONVOLUTION_NHWC_F32, batched_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_Kx3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 1; adjustment_height <= 2; adjustment_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_height(adjustment_height + 1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_width(adjustment_width + 1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_with_height_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_with_width_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_with_height_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_height(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_with_width_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, weights_cache_batched_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF32();
}


/**************************** CONV path, grouped, batched ****************************/

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_Kx3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .groups(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .groups(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 1; adjustment_height <= 2; adjustment_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_height(adjustment_height + 1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_width(adjustment_width + 1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_with_height_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_with_width_dilation) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_with_height_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_width(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_with_width_dilation_and_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, weights_cache_batched_grouped_3x3) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF32();
}

/**************************** CONV path, setup ****************************/

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3_setup_changing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .next_batch_size(5)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_height(3)
    .kernel_width(5)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3_setup_changing_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .next_input_height(kUnstridedInputHeight + 3)
    .kernel_height(3)
    .kernel_width(5)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3_setup_changing_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .next_input_width(kUnstridedInputWidth + 3)
    .kernel_height(3)
    .kernel_width(5)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupF32();
}

/**************************** SUBCONV2D/IGEMM path ****************************/

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, Kx3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3xKs2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3sSx1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_height(stride_height)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3s1xS) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_width(stride_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3s2_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .stride(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3s2_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .stride(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3s2_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 0; adjustment_height <= 1; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3s2_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, weights_cache_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, stress_weights_cache_5x5s4) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(5, 5)
    .stride(4)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(60)  // Higher number of iterations to write more weights.
    .StressWeightsCacheTestF32();
}

#if (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_JIT // TODO(b/287020333)
TEST(DECONVOLUTION_NHWC_F32, jit_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_jit(true)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, jit_3x3s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .use_jit(true)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, jit_3x3s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .use_jit(true)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, jit_3x3s2_with_relu) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .activation(DeconvolutionOperatorTester::Activation::Relu)
    .use_jit(true)
    .iterations(3)
    .TestF32();
}
#endif  // (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_JIT

/**************************** SUBCONV2D/IGEMM path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_Kx3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3xKs2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3sSx1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_height(stride_height)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s1xS) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_width(stride_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .stride(2)
        .groups(2)
        .group_input_channels(17)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .stride(2)
        .groups(2)
        .group_input_channels(17)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 0; adjustment_height <= 1; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, weights_cache_grouped_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF32();
}

/**************************** SUBCONV2D/IGEMM path, batched ****************************/

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_Kx3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3xKs2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3sSx1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_height(stride_height)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s1xS) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_width(stride_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .stride(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .batch_size(2)
        .input_size(kStridedInputHeight, kStridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .stride(2)
        .group_input_channels(15)
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 0; adjustment_height <= 1; adjustment_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, weights_cache_batched_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF32();
}

/**************************** SUBCONV2D/IGEMM path, grouped, batched ****************************/

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_Kx3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3xKs2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3sSx1) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_height(stride_height)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s1xS) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_width(stride_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_varying_height_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
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
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_varying_width_padding) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
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
        .group_output_channels(gemm_config->nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_varying_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_height = 0; adjustment_height <= 1; adjustment_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_varying_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, weights_cache_batched_grouped_3x3s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF32();
}

/**************************** SUBCONV2D/IGEMM path, setup ****************************/

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3s2_setup_changing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .next_batch_size(5)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3s2_setup_changing_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .next_input_height(kStridedInputHeight + 3)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_3x3s2_setup_changing_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .next_input_width(kStridedInputWidth + 3)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupF32();
}

/**************************** SUBCONV2D/GEMM path ****************************/

TEST(DECONVOLUTION_NHWC_F32, kernel_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, Kx2sKx2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_2xKs2xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_2x2s2_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_2x2s2_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_2x2s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_2x2s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_2x2s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_2x2s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, kernel_2x2s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_2x2s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_2x2s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_2x2s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_2x2s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, weights_cache_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF32();
}

#if (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_JIT // TODO(b/287020333)
TEST(DECONVOLUTION_NHWC_F32, jit_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_jit(true)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, jit_2x2s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .use_jit(true)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, jit_2x2s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .use_jit(true)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, jit_2x2s2_with_relu) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .activation(DeconvolutionOperatorTester::Activation::Relu)
    .use_jit(true)
    .iterations(3)
    .TestF32();
}
#endif  // (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_JIT

/**************************** SUBCONV2D/GEMM path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_Kx2sKx2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2xKs2xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, weights_cache_grouped_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF32();
}

/**************************** SUBCONV2D/GEMM path, batched ****************************/

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_Kx2sKx2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_2xKs2xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, weights_cache_batched_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF32();
}

/**************************** SUBCONV2D/GEMM path, grouped, batched ****************************/

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_Kx2sKx2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2xKs2xK) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2_height_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2_width_adjustment) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(1)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2_varying_input_height) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2_varying_input_width) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2_varying_input_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(gemm_config->nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2_varying_output_channels) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  for (size_t output_channels = 1; output_channels <= gemm_config->nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2_with_input_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2_with_output_stride) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr + 3)
    .output_pixel_stride(gemm_config->nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2_with_qmin) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2_with_qmax) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2_without_bias) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, weights_cache_batched_grouped_2x2s2) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  ASSERT_NE(gemm_config, nullptr);
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(gemm_config->nr * 2 + 3)
    .use_weights_cache(true)
    .iterations(3)
    .TestF32();
}

/**************************** SUBCONV2D/GEMM path, setup ****************************/

TEST(DECONVOLUTION_NHWC_F32, kernel_2x2s2_setup_changing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .next_batch_size(5)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_2x2s2_setup_changing_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .next_input_height(kStridedInputHeight + 3)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupF32();
}

TEST(DECONVOLUTION_NHWC_F32, kernel_2x2s2_setup_changing_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .next_input_width(kStridedInputWidth + 3)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupF32();
}
