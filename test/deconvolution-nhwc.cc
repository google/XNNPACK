// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/params.h>

#include "deconvolution-operator-tester.h"


constexpr size_t kUnstridedInputHeight = 8;
constexpr size_t kUnstridedInputWidth = 7;
constexpr size_t kStridedInputHeight = 6;
constexpr size_t kStridedInputWidth = 5;


/**************************** Future GEMM path ****************************/

TEST(DECONVOLUTION_NHWC_Q8, 1x1) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 1x1_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 1x1_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 1x1_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 1x1_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.q8.gemm.nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 1x1_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 1x1_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .output_pixel_stride(xnn_params.q8.gemm.nr * 2 + 13)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 1x1_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 1x1_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 1x1_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

/**************************** Future GEMM path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_Q8, grouped_1x1) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_1x1_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_1x1_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_1x1_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_1x1_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.q8.gemm.nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_1x1_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_1x1_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr + 3)
    .output_pixel_stride(xnn_params.q8.gemm.nr * 2 + 13)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_1x1_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_1x1_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_1x1_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

/**************************** Future GEMM path, batched ****************************/

TEST(DECONVOLUTION_NHWC_Q8, batched_1x1) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_1x1_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_1x1_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_1x1_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_1x1_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.q8.gemm.nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_1x1_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_1x1_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .output_pixel_stride(xnn_params.q8.gemm.nr * 2 + 13)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_1x1_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_1x1_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_1x1_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

/**************************** Future GEMM path, batched, grouped ****************************/

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_1x1) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_1x1_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_1x1_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_1x1_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_1x1_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.q8.gemm.nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_1x1_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_1x1_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr + 3)
    .output_pixel_stride(xnn_params.q8.gemm.nr * 2 + 13)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_1x1_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_1x1_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_1x1_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

/**************************** CONV path ****************************/

TEST(DECONVOLUTION_NHWC_Q8, 3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, Kx3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3xK) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3_varying_height_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .group_input_channels(15)
        .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3_varying_width_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .group_input_channels(15)
        .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3_varying_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_height = 1; adjustment_height <= 2; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_height(adjustment_height + 1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3_varying_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_width(adjustment_width + 1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3_varying_input_height_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3_varying_input_width_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.q8.gemm.nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3_with_height_dilation) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .group_input_channels(23)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3_with_width_dilation) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .group_input_channels(23)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3_with_height_dilation_and_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_height(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3_with_width_dilation_and_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .output_pixel_stride(xnn_params.q8.gemm.nr * 2 + 13)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

/**************************** CONV path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_Kx3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3xK) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3_varying_height_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3_varying_width_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3_varying_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_height = 1; adjustment_height <= 2; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_height(adjustment_height + 1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3_varying_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_width(adjustment_width + 1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3_varying_input_height_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3_varying_input_width_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.q8.gemm.nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3_with_height_dilation) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3_with_width_dilation) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3_with_height_dilation_and_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_height(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3_with_width_dilation_and_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr + 3)
    .output_pixel_stride(xnn_params.q8.gemm.nr * 2 + 13)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

/**************************** CONV path, batched ****************************/

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_Kx3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3xK) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3_varying_height_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3_varying_width_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3_varying_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_height = 1; adjustment_height <= 2; adjustment_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_height(adjustment_height + 1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3_varying_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_width(adjustment_width + 1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3_varying_input_height_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3_varying_input_width_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.q8.gemm.nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3_with_height_dilation) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .group_input_channels(23)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3_with_width_dilation) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .group_input_channels(23)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3_with_height_dilation_and_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_height(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3_with_width_dilation_and_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .output_pixel_stride(xnn_params.q8.gemm.nr * 2 + 13)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

/**************************** CONV path, grouped, batched ****************************/

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_Kx3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3xK) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3_varying_height_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3_varying_width_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3_varying_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3_varying_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3_varying_input_height_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3_varying_input_width_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.q8.gemm.nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3_with_height_dilation) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3_with_width_dilation) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3_with_height_dilation_and_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_width(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3_with_width_dilation_and_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr + 3)
    .output_pixel_stride(xnn_params.q8.gemm.nr * 2 + 13)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

/**************************** CONV path, setup ****************************/

TEST(DECONVOLUTION_NHWC_Q8, 3x3_setup_changing_batch) {
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
    .TestSetupQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3_setup_changing_height) {
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
    .TestSetupQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3_setup_changing_width) {
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
    .TestSetupQ8();
}

/**************************** SUBCONV2D/IGEMM path ****************************/

TEST(DECONVOLUTION_NHWC_Q8, 3x3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, Kx3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3xKs2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3sSx1) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_height(stride_height)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3s1xS) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_width(stride_width)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3s2_varying_height_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3s2_varying_width_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3s2_varying_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_height = 0; adjustment_height <= 1; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3s2_varying_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3s2_varying_input_height_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3s2_varying_input_width_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3s2_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3s2_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.q8.gemm.nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3s2_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3s2_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .output_pixel_stride(xnn_params.q8.gemm.nr * 2 + 13)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3s2_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3s2_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3s2_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

/**************************** SUBCONV2D/IGEMM path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_Kx3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3xKs2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3sSx1) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_height(stride_height)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3s1xS) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_width(stride_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3s2_varying_height_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3s2_varying_width_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3s2_varying_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_height = 0; adjustment_height <= 1; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3s2_varying_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3s2_varying_input_height_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3s2_varying_input_width_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3s2_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3s2_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.q8.gemm.nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3s2_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3s2_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr + 3)
    .output_pixel_stride(xnn_params.q8.gemm.nr * 2 + 13)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3s2_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3s2_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_3x3s2_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

/**************************** SUBCONV2D/IGEMM path, batched ****************************/

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_Kx3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3xKs2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3sSx1) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_height(stride_height)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3s1xS) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_width(stride_width)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3s2_varying_height_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3s2_varying_width_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3s2_varying_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_height = 0; adjustment_height <= 1; adjustment_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3s2_varying_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3s2_varying_input_height_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3s2_varying_input_width_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3s2_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3s2_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.q8.gemm.nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3s2_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3s2_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .output_pixel_stride(xnn_params.q8.gemm.nr * 2 + 13)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3s2_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3s2_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_3x3s2_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

/**************************** SUBCONV2D/IGEMM path, grouped, batched ****************************/

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_Kx3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3xKs2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3sSx1) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3s1xS) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3s2_varying_height_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3s2_varying_width_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3s2_varying_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3s2_varying_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3s2_varying_input_height_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3s2_varying_input_width_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3s2_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3s2_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.q8.gemm.nr * 2; output_channels *= 2) {
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
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3s2_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3s2_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr + 3)
    .output_pixel_stride(xnn_params.q8.gemm.nr * 2 + 13)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3s2_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3s2_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_3x3s2_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

/**************************** SUBCONV2D/IGEMM path, setup ****************************/

TEST(DECONVOLUTION_NHWC_Q8, 3x3s2_setup_changing_batch) {
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
    .TestSetupQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3s2_setup_changing_height) {
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
    .TestSetupQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 3x3s2_setup_changing_width) {
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
    .TestSetupQ8();
}

/**************************** SUBCONV2D/GEMM path ****************************/

TEST(DECONVOLUTION_NHWC_Q8, 2x2s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, Kx2sKx2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 2xKs2xK) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 2x2s2_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(1)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 2x2s2_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(1)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 2x2s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 2x2s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 2x2s2_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 2x2s2_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.q8.gemm.nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, 2x2s2_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 2x2s2_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .output_pixel_stride(xnn_params.q8.gemm.nr * 2 + 13)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 2x2s2_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 2x2s2_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 2x2s2_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

/**************************** SUBCONV2D/GEMM path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_Q8, grouped_2x2s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_Kx2sKx2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_2xKs2xK) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_2x2s2_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(1)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_2x2s2_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(1)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_2x2s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_2x2s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_2x2s2_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_2x2s2_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.q8.gemm.nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_2x2s2_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_2x2s2_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr + 3)
    .output_pixel_stride(xnn_params.q8.gemm.nr * 2 + 13)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_2x2s2_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_2x2s2_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, grouped_2x2s2_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

/**************************** SUBCONV2D/GEMM path, batched ****************************/

TEST(DECONVOLUTION_NHWC_Q8, batched_2x2s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_Kx2sKx2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_2xKs2xK) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_2x2s2_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(1)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_2x2s2_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(1)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_2x2s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_2x2s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_2x2s2_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_2x2s2_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.q8.gemm.nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_2x2s2_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_2x2s2_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .output_pixel_stride(xnn_params.q8.gemm.nr * 2 + 13)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_2x2s2_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_2x2s2_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_2x2s2_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

/**************************** SUBCONV2D/GEMM path, grouped, batched ****************************/

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_2x2s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_Kx2sKx2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_2xKs2xK) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(3)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_2x2s2_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(1)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_2x2s2_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(1)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_2x2s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_2x2s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_2x2s2_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_2x2s2_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.q8.gemm.nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestQ8();
  }
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_2x2s2_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_2x2s2_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr + 3)
    .output_pixel_stride(xnn_params.q8.gemm.nr * 2 + 13)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_2x2s2_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_2x2s2_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, batched_grouped_2x2s2_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.q8.gemm.nr * 2 + 3)
    .iterations(3)
    .TestQ8();
}

/**************************** SUBCONV2D/GEMM path, setup ****************************/

TEST(DECONVOLUTION_NHWC_Q8, 2x2s2_setup_changing_batch) {
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
    .TestSetupQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 2x2s2_setup_changing_height) {
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
    .TestSetupQ8();
}

TEST(DECONVOLUTION_NHWC_Q8, 2x2s2_setup_changing_width) {
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
    .TestSetupQ8();
}

/**************************** Future GEMM path ****************************/

TEST(DECONVOLUTION_NHWC_F32, 1x1) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, 1x1_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 1x1_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 1x1_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 1x1_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.f32.gemm.nr * 2; output_channels *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 1x1_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, 1x1_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .output_pixel_stride(xnn_params.f32.gemm.nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, 1x1_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, 1x1_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, 1x1_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

/**************************** Future GEMM path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_F32, grouped_1x1) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_1x1_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_1x1_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_1x1_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_1x1_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.f32.gemm.nr * 2; output_channels *= 2) {
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
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_1x1_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr + 3)
    .output_pixel_stride(xnn_params.f32.gemm.nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_1x1_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_1x1_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_1x1_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

/**************************** Future GEMM path, batched ****************************/

TEST(DECONVOLUTION_NHWC_F32, batched_1x1) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_1x1_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_1x1_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_1x1_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_1x1_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.f32.gemm.nr * 2; output_channels *= 2) {
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
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_1x1_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .output_pixel_stride(xnn_params.f32.gemm.nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_1x1_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_1x1_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_1x1_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

/**************************** Future GEMM path, batched, grouped ****************************/

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_1x1) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_1x1_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_1x1_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_1x1_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_1x1_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.f32.gemm.nr * 2; output_channels *= 2) {
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
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_1x1_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr + 3)
    .output_pixel_stride(xnn_params.f32.gemm.nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_1x1_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_1x1_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_1x1_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

/**************************** CONV path ****************************/

TEST(DECONVOLUTION_NHWC_F32, 3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, Kx3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3xK) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3_varying_height_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t padding_top = 0; padding_top <= 2; padding_top++) {
    for (size_t padding_bottom = 0; padding_bottom <= 2; padding_bottom++) {
      DeconvolutionOperatorTester()
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_width(1)
        .padding_top(padding_top)
        .padding_bottom(padding_bottom)
        .kernel_size(3, 3)
        .group_input_channels(15)
        .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3_varying_width_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t padding_left = 0; padding_left <= 2; padding_left++) {
    for (size_t padding_right = 0; padding_right <= 2; padding_right++) {
      DeconvolutionOperatorTester()
        .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
        .padding_height(1)
        .padding_left(padding_left)
        .padding_right(padding_right)
        .kernel_size(3, 3)
        .group_input_channels(15)
        .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3_varying_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_height = 1; adjustment_height <= 2; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_height(adjustment_height + 1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3_varying_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_width(adjustment_width + 1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3_varying_input_height_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3_varying_input_width_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.f32.gemm.nr * 2; output_channels *= 2) {
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

TEST(DECONVOLUTION_NHWC_F32, 3x3_with_height_dilation) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .group_input_channels(23)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3_with_width_dilation) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .group_input_channels(23)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3_with_height_dilation_and_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_height(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, 3x3_with_width_dilation_and_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, 3x3_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, 3x3_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .output_pixel_stride(xnn_params.f32.gemm.nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, 3x3_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, 3x3_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, 3x3_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

/**************************** CONV path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_Kx3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3xK) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_varying_height_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_varying_width_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_varying_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_height = 1; adjustment_height <= 2; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_height(adjustment_height + 1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_varying_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_width(adjustment_width + 1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_varying_input_height_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kUnstridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_varying_input_width_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, input_width)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.f32.gemm.nr * 2; output_channels *= 2) {
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
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_with_width_dilation) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_with_height_dilation_and_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_height(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_with_width_dilation_and_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr + 3)
    .output_pixel_stride(xnn_params.f32.gemm.nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

/**************************** CONV path, batched ****************************/

TEST(DECONVOLUTION_NHWC_F32, batched_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_Kx3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3xK) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_varying_height_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_varying_width_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_varying_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_height = 1; adjustment_height <= 2; adjustment_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_height(adjustment_height + 1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_varying_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_width = 1; adjustment_width <= 2; adjustment_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .stride_width(adjustment_width + 1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_varying_input_height_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_varying_input_width_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.f32.gemm.nr * 2; output_channels *= 2) {
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
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .group_input_channels(23)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_with_width_dilation) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .group_input_channels(23)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_with_height_dilation_and_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_height(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_with_width_dilation_and_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .output_pixel_stride(xnn_params.f32.gemm.nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

/**************************** CONV path, grouped, batched ****************************/

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_Kx3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 1; kernel_height <= 4; kernel_height *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3xK) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 1; kernel_width <= 4; kernel_width *= 2) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_varying_height_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_varying_width_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_varying_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_varying_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_varying_input_height_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kUnstridedInputHeight - 2; input_height <= kUnstridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kUnstridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_varying_input_width_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kUnstridedInputWidth - 2; input_width <= kUnstridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, input_width)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.f32.gemm.nr * 2; output_channels *= 2) {
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
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t dilation_height = 2; dilation_height <= 3; dilation_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_height(dilation_height)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_with_width_dilation) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t dilation_width = 2; dilation_width <= 3; dilation_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .dilation_width(dilation_width)
      .groups(2)
      .group_input_channels(23)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_with_height_dilation_and_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_height(3)
    .stride_width(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_with_width_dilation_and_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .dilation_width(3)
    .stride_width(2)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .input_pixel_stride(47)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr + 3)
    .output_pixel_stride(xnn_params.f32.gemm.nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kUnstridedInputHeight, kUnstridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

/**************************** CONV path, setup ****************************/

TEST(DECONVOLUTION_NHWC_F32, 3x3_setup_changing_batch) {
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

TEST(DECONVOLUTION_NHWC_F32, 3x3_setup_changing_height) {
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

TEST(DECONVOLUTION_NHWC_F32, 3x3_setup_changing_width) {
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

TEST(DECONVOLUTION_NHWC_F32, 3x3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, Kx3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3xKs2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3sSx1) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_height(stride_height)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3s1xS) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_width(stride_width)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3s2_varying_height_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3s2_varying_width_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3s2_varying_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_height = 0; adjustment_height <= 1; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3s2_varying_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3s2_varying_input_height_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3s2_varying_input_width_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3s2_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 3x3s2_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.f32.gemm.nr * 2; output_channels *= 2) {
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

TEST(DECONVOLUTION_NHWC_F32, 3x3s2_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, 3x3s2_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .output_pixel_stride(xnn_params.f32.gemm.nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, 3x3s2_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, 3x3s2_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, 3x3s2_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

/**************************** SUBCONV2D/IGEMM path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_Kx3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3xKs2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3sSx1) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_height(stride_height)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s1xS) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_width(stride_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_varying_height_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_varying_width_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_varying_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_height = 0; adjustment_height <= 1; adjustment_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_varying_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_varying_input_height_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_varying_input_width_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.f32.gemm.nr * 2; output_channels *= 2) {
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
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr + 3)
    .output_pixel_stride(xnn_params.f32.gemm.nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_3x3s2_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

/**************************** SUBCONV2D/IGEMM path, batched ****************************/

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_Kx3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3xKs2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3sSx1) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t stride_height = 2; stride_height <= 3; stride_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_height(stride_height)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s1xS) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t stride_width = 2; stride_width <= 3; stride_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .padding_width(1)
      .kernel_size(3, 3)
      .stride_width(stride_width)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_varying_height_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_varying_width_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_varying_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_height = 0; adjustment_height <= 1; adjustment_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_height(adjustment_height)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_varying_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t adjustment_width = 0; adjustment_width <= 1; adjustment_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .adjustment_width(adjustment_width)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_varying_input_height_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_varying_input_width_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.f32.gemm.nr * 2; output_channels *= 2) {
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
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .output_pixel_stride(xnn_params.f32.gemm.nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_3x3s2_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

/**************************** SUBCONV2D/IGEMM path, grouped, batched ****************************/

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_Kx3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 2; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_width(1)
      .kernel_size(kernel_height, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3xKs2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 2; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_height(1)
      .kernel_size(3, kernel_width)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3sSx1) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s1xS) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_varying_height_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_varying_width_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
        .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_varying_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_varying_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
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
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_varying_input_height_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_varying_input_width_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding_tf_same(true)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .padding(1)
      .kernel_size(3, 3)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.f32.gemm.nr * 2; output_channels *= 2) {
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
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr + 3)
    .output_pixel_stride(xnn_params.f32.gemm.nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_3x3s2_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .padding(1)
    .kernel_size(3, 3)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

/**************************** SUBCONV2D/IGEMM path, setup ****************************/

TEST(DECONVOLUTION_NHWC_F32, 3x3s2_setup_changing_batch) {
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

TEST(DECONVOLUTION_NHWC_F32, 3x3s2_setup_changing_height) {
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

TEST(DECONVOLUTION_NHWC_F32, 3x3s2_setup_changing_width) {
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

TEST(DECONVOLUTION_NHWC_F32, 2x2s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, Kx2sKx2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 2xKs2xK) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 2x2s2_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(1)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, 2x2s2_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(1)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, 2x2s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 2x2s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 2x2s2_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, 2x2s2_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.f32.gemm.nr * 2; output_channels *= 2) {
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

TEST(DECONVOLUTION_NHWC_F32, 2x2s2_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, 2x2s2_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .output_pixel_stride(xnn_params.f32.gemm.nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, 2x2s2_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, 2x2s2_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, 2x2s2_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

/**************************** SUBCONV2D/GEMM path, grouped ****************************/

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_Kx2sKx2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2xKs2xK) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(1)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(1)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.f32.gemm.nr * 2; output_channels *= 2) {
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
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr + 3)
    .output_pixel_stride(xnn_params.f32.gemm.nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, grouped_2x2s2_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

/**************************** SUBCONV2D/GEMM path, batched ****************************/

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_Kx2sKx2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_2xKs2xK) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(1)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(15)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(1)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(15)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.f32.gemm.nr * 2; output_channels *= 2) {
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
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .input_pixel_stride(28)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .output_pixel_stride(xnn_params.f32.gemm.nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_2x2s2_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .group_input_channels(23)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

/**************************** SUBCONV2D/GEMM path, grouped, batched ****************************/

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_Kx2sKx2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_height = 3; kernel_height <= 5; kernel_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(kernel_height, 2)
      .stride(kernel_height, 2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2xKs2xK) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t kernel_width = 3; kernel_width <= 5; kernel_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, kernel_width)
      .stride(2, kernel_width)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(3)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2_height_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_height(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(1)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2_width_adjustment) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .adjustment_width(1)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(1)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = kStridedInputHeight - 2; input_height <= kStridedInputHeight + 2; input_height++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = kStridedInputWidth - 2; input_width <= kStridedInputWidth + 2; input_width++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 14; input_channels <= 20; input_channels++) {
    DeconvolutionOperatorTester()
      .batch_size(2)
      .input_size(kStridedInputHeight, kStridedInputWidth)
      .kernel_size(2, 2)
      .stride(2)
      .groups(2)
      .group_input_channels(input_channels)
      .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
      .iterations(1)
      .TestF32();
  }
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels <= xnn_params.f32.gemm.nr * 2; output_channels *= 2) {
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
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .input_pixel_stride(37)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr + 3)
    .output_pixel_stride(xnn_params.f32.gemm.nr * 2 + 13)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(DECONVOLUTION_NHWC_F32, batched_grouped_2x2s2_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  DeconvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(kStridedInputHeight, kStridedInputWidth)
    .kernel_size(2, 2)
    .stride(2)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(xnn_params.f32.gemm.nr * 2 + 3)
    .iterations(3)
    .TestF32();
}

/**************************** SUBCONV2D/GEMM path, setup ****************************/

TEST(DECONVOLUTION_NHWC_F32, 2x2s2_setup_changing_batch) {
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

TEST(DECONVOLUTION_NHWC_F32, 2x2s2_setup_changing_height) {
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

TEST(DECONVOLUTION_NHWC_F32, 2x2s2_setup_changing_width) {
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
