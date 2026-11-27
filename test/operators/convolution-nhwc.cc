// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "include/xnnpack.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/kai-dwconv.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/params.h"
#include "test/operators/convolution-operator-tester.h"

namespace {

class GokiPackerTracker {
 public:
  explicit GokiPackerTracker(const xnn_gemm_config* config)
      : config_(const_cast<xnn_gemm_config*>(config)),
        original_(config_->pack_igemm_goki) {
    assert(active_ == nullptr);
    active_ = this;
    config_->pack_igemm_goki = Track;
  }

  ~GokiPackerTracker() {
    config_->pack_igemm_goki = original_;
    active_ = nullptr;
  }

  size_t calls() const { return calls_; }

 private:
  static void Track(size_t groups, size_t channels, size_t kernel_size,
                    size_t depth, size_t nr, size_t kr, size_t sr,
                    const void* kernel, const void* bias, const void* scale,
                    void* packed, size_t extra_bytes, const void* params) {
    ++active_->calls_;
    active_->original_(groups, channels, kernel_size, depth, nr, kr, sr, kernel,
                       bias, scale, packed, extra_bytes, params);
  }

  inline static GokiPackerTracker* active_ = nullptr;
  xnn_gemm_config* config_;
  xnn_pack_conv_goki_w_fn original_;
  size_t calls_ = 0;
};

template <bool fp16, bool converted_weights = false>
void TestPackedFloatGoki() {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr));
  const xnn_gemm_config* config =
      fp16 ? xnn_init_pf16_gemm_config() : xnn_init_pf32_gemm_config();
  if (config == nullptr) {
    GTEST_SKIP();
  }
  ASSERT_NE(nullptr, config->pack_igemm_goki);
  constexpr size_t expected_packer_calls = fp16 && converted_weights ? 0 : 1;
  using Value = std::conditional_t<fp16, xnn_float16, float>;
  using Weight =
      std::conditional_t<fp16 != converted_weights, xnn_float16, float>;
  const auto convert = [](float value) -> Value {
    if constexpr (fp16) {
      return xnn_float16_from_float(value);
    } else {
      return value;
    }
  };
  const auto convert_weight = [](float value) -> Weight {
    if constexpr (fp16 != converted_weights) {
      return xnn_float16_from_float(value);
    } else {
      return value;
    }
  };
  const size_t shapes[][4] = {{1, 3, 3, 1},   {1, 3, 3, 15},  {1, 33, 9, 17},
                              {2, 33, 3, 17}, {1, 64, 9, 64}, {2, 64, 9, 64}};
  for (const auto& shape : shapes) {
    const size_t groups = shape[0];
    const size_t channels = shape[1];
    const size_t kernel_size = shape[2];
    const size_t depth = shape[3];
    const size_t kernel_height = kernel_size == 9 ? 3 : 1;
    const size_t kernel_width = 3;
    for (const bool has_bias : {false, true}) {
      SCOPED_TRACE(testing::Message()
                   << groups << '/' << channels << '/' << kernel_size << '/'
                   << depth << "/bias:" << has_bias);
      std::vector<Weight> kernel(groups * channels * kernel_size * depth);
      std::vector<Weight> bias(groups * channels, convert_weight(0.5f));
      for (size_t channel = 0; channel < groups * channels; ++channel) {
        const int sign = static_cast<int>(channel % 3) - 1;
        std::fill_n(kernel.data() + channel * kernel_size * depth,
                    kernel_size * depth, convert_weight(sign * 0.25f));
      }
      const size_t input_height = kernel_height + 1;
      const size_t input_width = kernel_width + 2;
      std::vector<Value> input(input_height * input_width * groups * depth +
                                   XNN_EXTRA_BYTES / sizeof(Value),
                               convert(0.5f));
      std::vector<Value> output(6 * groups * channels);
      GokiPackerTracker tracker(config);
      xnn_operator_t raw_operation = nullptr;
      xnn_status status;
      const uint32_t flags =
          XNN_FLAG_INLINE_LHS_PACKING |
          ((fp16 && converted_weights) ? XNN_FLAG_FP32_STATIC_WEIGHTS : 0);
      if constexpr (fp16) {
        status = xnn_create_convolution2d_nhwc_pf16(
            0, 0, 0, 0, kernel_height, kernel_width, 1, 1, 1, 1, groups, depth,
            channels, groups * depth, groups * channels, kernel.data(),
            has_bias ? bias.data() : nullptr,
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(), flags, nullptr,
            &raw_operation);
      } else {
        if constexpr (converted_weights) {
          status = xnn_create_convolution2d_nhwc_pf32_f16(
              0, 0, 0, 0, kernel_height, kernel_width, 1, 1, 1, 1, groups,
              depth, channels, groups * depth, groups * channels, kernel.data(),
              has_bias ? bias.data() : nullptr,
              -std::numeric_limits<float>::infinity(),
              std::numeric_limits<float>::infinity(), flags, nullptr,
              &raw_operation);
        } else {
          status = xnn_create_convolution2d_nhwc_pf32(
              0, 0, 0, 0, kernel_height, kernel_width, 1, 1, 1, 1, groups,
              depth, channels, groups * depth, groups * channels, kernel.data(),
              has_bias ? bias.data() : nullptr,
              -std::numeric_limits<float>::infinity(),
              std::numeric_limits<float>::infinity(), flags, nullptr,
              &raw_operation);
        }
      }
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> operation(
          raw_operation, xnn_delete_operator);
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, operation);
      EXPECT_EQ(expected_packer_calls, tracker.calls());
      size_t workspace_size = 0;
      size_t output_height = 0;
      size_t output_width = 0;
      const auto reshape = fp16 ? xnn_reshape_convolution2d_nhwc_pf16
                                : xnn_reshape_convolution2d_nhwc_pf32;
      ASSERT_EQ(
          xnn_status_success,
          reshape(operation.get(), 1, input_height, input_width,
                  &workspace_size, &output_height, &output_width, nullptr));
      ASSERT_EQ(2, output_height);
      ASSERT_EQ(3, output_width);
      xnnpack::Buffer<uint8_t> workspace(workspace_size);
      if constexpr (fp16) {
        ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_pf16(
                                          operation.get(), workspace.data(),
                                          input.data(), output.data()));
      } else {
        ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_pf32(
                                          operation.get(), workspace.data(),
                                          input.data(), output.data()));
      }
      ASSERT_EQ(xnn_status_success, xnn_run_operator(operation.get(), nullptr));
      EXPECT_EQ(expected_packer_calls, tracker.calls());
      for (size_t index = 0; index < output.size(); ++index) {
        const int sign =
            static_cast<int>((index % (groups * channels)) % 3) - 1;
        const float expected =
            sign * 0.125f * (kernel_size * depth) + (has_bias ? 0.5f : 0.0f);
        if constexpr (fp16) {
          EXPECT_EQ(expected, xnn_float16_to_float(output[index]));
        } else {
          EXPECT_EQ(expected, output[index]);
        }
      }
    }
  }
}

using ConvolutionTestCase =
    std::pair<const char*, std::vector<ConvolutionOperatorTester>>;

static const ConvolutionTestCase kConvolutionTests[] = {
    {"kernel_1x1",
     {ConvolutionOperatorTester()
          .input_size(27, 37)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(19)}},
    {"kernel_1x1_with_qmin",
     {ConvolutionOperatorTester()
          .input_size(27, 37)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(19)
          .qmin(128)}},
    {"kernel_1x1_with_qmax",
     {ConvolutionOperatorTester()
          .input_size(27, 37)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(19)
          .qmax(128)}},
    {"kernel_1x1_with_input_stride",
     {ConvolutionOperatorTester()
          .input_size(27, 37)
          .kernel_size(1, 1)
          .input_channel_stride(28)
          .group_input_channels(23)
          .group_output_channels(19)}},
    {"kernel_1x1_with_output_stride",
     {ConvolutionOperatorTester()
          .input_size(27, 37)
          .kernel_size(1, 1)
          .output_channel_stride(29)
          .group_input_channels(23)
          .group_output_channels(19)}},
    {"kernel_1x1_without_bias",
     {ConvolutionOperatorTester()
          .has_bias(false)
          .input_size(13, 14)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(19)}},
    {"kernel_1x1_with_batch",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_size(13, 14)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(19)}},
    {"grouped_1x1",
     {ConvolutionOperatorTester()
          .input_size(24, 25)
          .kernel_size(1, 1)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(19)}},
    {"grouped_1x1_with_qmin",
     {ConvolutionOperatorTester()
          .input_size(24, 25)
          .kernel_size(1, 1)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(19)
          .qmin(128)}},
    {"grouped_1x1_with_qmax",
     {ConvolutionOperatorTester()
          .input_size(24, 25)
          .kernel_size(1, 1)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(19)
          .qmax(128)}},
    {"grouped_1x1_with_input_stride",
     {ConvolutionOperatorTester()
          .input_size(24, 25)
          .kernel_size(1, 1)
          .groups(2)
          .input_channel_stride(37)
          .group_input_channels(17)
          .group_output_channels(19)}},
    {"grouped_1x1_with_output_stride",
     {ConvolutionOperatorTester()
          .input_size(24, 25)
          .kernel_size(1, 1)
          .groups(2)
          .output_channel_stride(41)
          .group_input_channels(17)
          .group_output_channels(19)}},
    {"grouped_1x1_without_bias",
     {ConvolutionOperatorTester()
          .has_bias(false)
          .input_size(24, 25)
          .kernel_size(1, 1)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(19)}},
    {"grouped_1x1_with_batch",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_size(24, 25)
          .kernel_size(1, 1)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(19)}},
    {"kernel_1x3",
     {ConvolutionOperatorTester()
          .input_size(20, 19)
          .padding_width(1)
          .kernel_size(1, 3)
          .group_input_channels(17)
          .group_output_channels(15)}},
    {"grouped_1x3",
     {ConvolutionOperatorTester()
          .input_size(20, 19)
          .padding_width(1)
          .kernel_size(1, 3)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(15)}},
    {"kernel_3x1",
     {ConvolutionOperatorTester()
          .input_size(19, 20)
          .padding_height(1)
          .kernel_size(3, 1)
          .group_input_channels(17)
          .group_output_channels(15)}},
    {"grouped_3x1",
     {ConvolutionOperatorTester()
          .input_size(19, 20)
          .padding_height(1)
          .kernel_size(3, 1)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(15)}},
    {"kernel_3x3",
     {ConvolutionOperatorTester()
          .input_size(13, 12)
          .padding(1)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_3x3_without_padding",
     {ConvolutionOperatorTester()
          .input_size(13, 12)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_3x3_with_left_padding",
     {ConvolutionOperatorTester()
          .input_size(13, 12)
          .padding_left(1)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_3x3_with_right_padding",
     {ConvolutionOperatorTester()
          .input_size(13, 12)
          .padding_right(1)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_3x3_with_top_padding",
     {ConvolutionOperatorTester()
          .input_size(13, 12)
          .padding_top(1)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_3x3_with_bottom_padding",
     {ConvolutionOperatorTester()
          .input_size(13, 12)
          .padding_bottom(1)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_3x3_with_input_stride",
     {ConvolutionOperatorTester()
          .input_size(13, 12)
          .padding(1)
          .kernel_size(3, 3)
          .input_channel_stride(22)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_3x3_with_output_stride",
     {ConvolutionOperatorTester()
          .input_size(13, 12)
          .padding(1)
          .kernel_size(3, 3)
          .output_channel_stride(23)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_3x3_without_bias",
     {ConvolutionOperatorTester()
          .has_bias(false)
          .input_size(10, 9)
          .padding(1)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_3x3_with_batch",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_size(10, 9)
          .padding(1)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"grouped_3x3",
     {ConvolutionOperatorTester()
          .input_size(10, 11)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(14)
          .group_output_channels(13)}},
    {"grouped_3x3_transient_indirection",
     {ConvolutionOperatorTester()
          .input_size(10, 11)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(14)
          .group_output_channels(13)
          .transient_indirection_buffer(true)}},
    {"grouped_3x3_without_padding",
     {ConvolutionOperatorTester()
          .input_size(13, 12)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"grouped_3x3_with_left_padding",
     {ConvolutionOperatorTester()
          .input_size(10, 11)
          .padding_left(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(14)
          .group_output_channels(13)}},
    {"grouped_3x3_with_right_padding",
     {ConvolutionOperatorTester()
          .input_size(10, 11)
          .padding_right(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(14)
          .group_output_channels(13)}},
    {"grouped_3x3_with_top_padding",
     {ConvolutionOperatorTester()
          .input_size(10, 11)
          .padding_top(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(14)
          .group_output_channels(13)}},
    {"grouped_3x3_with_bottom_padding",
     {ConvolutionOperatorTester()
          .input_size(10, 11)
          .padding_bottom(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(14)
          .group_output_channels(13)}},
    {"grouped_3x3_with_input_stride",
     {ConvolutionOperatorTester()
          .input_size(10, 11)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .input_channel_stride(29)
          .group_input_channels(14)
          .group_output_channels(13)}},
    {"grouped_3x3_with_output_stride",
     {ConvolutionOperatorTester()
          .input_size(10, 11)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .output_channel_stride(31)
          .group_input_channels(14)
          .group_output_channels(13)}},
    {"grouped_3x3_without_bias",
     {ConvolutionOperatorTester()
          .has_bias(false)
          .input_size(10, 11)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(14)
          .group_output_channels(13)}},
    {"grouped_3x3_with_batch",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_size(10, 11)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(14)
          .group_output_channels(13)}},
    {"kernel_3x3s2",
     {ConvolutionOperatorTester()
          .input_size(14, 13)
          .padding(1)
          .kernel_size(3, 3)
          .subsampling(2)
          .group_input_channels(27)
          .group_output_channels(19)}},
    {"kernel_3x3s1x2",
     {ConvolutionOperatorTester()
          .input_size(14, 13)
          .padding(1)
          .kernel_size(3, 3)
          .subsampling(1, 2)
          .group_input_channels(27)
          .group_output_channels(19)}},
    {"kernel_3x3s2x1",
     {ConvolutionOperatorTester()
          .input_size(14, 13)
          .padding(1)
          .kernel_size(3, 3)
          .subsampling(2, 1)
          .group_input_channels(27)
          .group_output_channels(19)}},
    {"kernel_3x3d2",
     {ConvolutionOperatorTester()
          .input_size(14, 13)
          .padding(2)
          .kernel_size(3, 3)
          .dilation(2)
          .group_input_channels(27)
          .group_output_channels(19)}},
    {"kernel_3x3d1x2",
     {ConvolutionOperatorTester()
          .input_size(14, 13)
          .padding(1, 2)
          .kernel_size(3, 3)
          .dilation(1, 2)
          .group_input_channels(27)
          .group_output_channels(19)}},
    {"kernel_3x3d2x1",
     {ConvolutionOperatorTester()
          .input_size(14, 13)
          .padding(2, 1)
          .kernel_size(3, 3)
          .dilation(2, 1)
          .group_input_channels(27)
          .group_output_channels(19)}},
    {"depthwise_2x5",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(2, 5)
          .groups(27)}},
    {"depthwise_2x5_multithreaded",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(2, 5)
          .groups(27)
          .multithreaded(true)}},
    {"depthwise_2x5s5x6",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(2, 5)
          .subsampling(5, 6)
          .groups(27)}},
    {"depthwise_3x3",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .groups(27)}},
    {"depthwise_3x3_transient_indirection",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .groups(27)
          .input_channel_stride(28)
          .transient_indirection_buffer(true)}},
    {"depthwise_3x3_without_bias",
     {ConvolutionOperatorTester()
          .has_bias(false)
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .groups(27)}},
    {"depthwise_3x3s2",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .subsampling(2)
          .groups(27)}},
    {"depthwise_3x3s1x2",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .subsampling(1, 2)
          .groups(27)}},
    {"depthwise_3x3s2x1",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .subsampling(2, 1)
          .groups(27)}},
    {"depthwise_3x3d2",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .dilation(2)
          .groups(27)}},
    {"depthwise_3x3d1x2",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .dilation(1, 2)
          .groups(27)}},
    {"depthwise_3x3d2x1",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .dilation(2, 1)
          .groups(27)}},
    /* Tests GEMM microkernel with weights_cache.  */
    {"weights_cache_1x1",
     {ConvolutionOperatorTester()
          .input_size(27, 37)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(19)
          .use_weights_cache(true)}},
    /* Tests IGEMM microkernel with weights cache.  */
    {"weights_cache_3x3",
     {ConvolutionOperatorTester()
          .input_size(13, 12)
          .padding(1)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(17)
          .use_weights_cache(true)}},
    /* Tests vmulcaddc microkernel with weights cache.  */
    {"weights_cache_depthwise_1x1",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .kernel_size(1, 1)
          .groups(24)
          .use_weights_cache(true)}},
    /* Tests dwconv microkernel with weights cache.  */
    {"weights_cache_depthwise_2x2d2",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(2, 2)
          .dilation(2)
          .groups(27)
          .use_weights_cache(true)}},
    {"kernel_3x3s2_with_tf_same_padding",
     {ConvolutionOperatorTester()
          .input_size(13, 13)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2)
          .group_input_channels(27)
          .group_output_channels(19),
      ConvolutionOperatorTester()
          .input_size(13, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2)
          .group_input_channels(27)
          .group_output_channels(19),
      ConvolutionOperatorTester()
          .input_size(14, 13)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2)
          .group_input_channels(27)
          .group_output_channels(19),
      ConvolutionOperatorTester()
          .input_size(14, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2)
          .group_input_channels(27)
          .group_output_channels(19)}},
    {"kernel_3x3s1x2_with_tf_same_padding",
     {ConvolutionOperatorTester()
          .input_size(13, 13)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(1, 2)
          .group_input_channels(27)
          .group_output_channels(19),
      ConvolutionOperatorTester()
          .input_size(13, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(1, 2)
          .group_input_channels(27)
          .group_output_channels(19),
      ConvolutionOperatorTester()
          .input_size(14, 13)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(1, 2)
          .group_input_channels(27)
          .group_output_channels(19),
      ConvolutionOperatorTester()
          .input_size(14, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(1, 2)
          .group_input_channels(27)
          .group_output_channels(19)}},
    {"kernel_3x3s2x1_with_tf_same_padding",
     {ConvolutionOperatorTester()
          .input_size(13, 13)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2, 1)
          .group_input_channels(27)
          .group_output_channels(19),
      ConvolutionOperatorTester()
          .input_size(13, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2, 1)
          .group_input_channels(27)
          .group_output_channels(19),
      ConvolutionOperatorTester()
          .input_size(14, 13)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2, 1)
          .group_input_channels(27)
          .group_output_channels(19),
      ConvolutionOperatorTester()
          .input_size(14, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2, 1)
          .group_input_channels(27)
          .group_output_channels(19)}},
    {"depthwise_3x3s2_with_tf_same_padding",
     {ConvolutionOperatorTester()
          .input_size(14, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2)
          .groups(27),
      ConvolutionOperatorTester()
          .input_size(14, 15)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2)
          .groups(27),
      ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2)
          .groups(27),
      ConvolutionOperatorTester()
          .input_size(15, 15)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2)
          .groups(27)}},
    {"depthwise_3x3s1x2_with_tf_same_padding",
     {ConvolutionOperatorTester()
          .input_size(14, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(1, 2)
          .groups(27),
      ConvolutionOperatorTester()
          .input_size(14, 15)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(1, 2)
          .groups(27),
      ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(1, 2)
          .groups(27),
      ConvolutionOperatorTester()
          .input_size(15, 15)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(1, 2)
          .groups(27)}},
    {"depthwise_3x3s2x1_with_tf_same_padding",
     {ConvolutionOperatorTester()
          .input_size(14, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2, 1)
          .groups(27),
      ConvolutionOperatorTester()
          .input_size(14, 15)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2, 1)
          .groups(27),
      ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2, 1)
          .groups(27),
      ConvolutionOperatorTester()
          .input_size(15, 15)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2, 1)
          .groups(27)}},
};

static const ConvolutionTestCase kDepthwiseConvolutionTests[] = {
    {"kernel_1x1",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .kernel_size(1, 1)
          .groups(24)}},
    {"kernel_1x1_with_depth_multiplier",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .kernel_size(1, 1)
          .groups(24)
          .group_output_channels(3)}},
    {"kernel_1x1_without_bias",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .has_bias(false)
          .input_size(15, 14)
          .kernel_size(1, 1)
          .groups(24)}},
    {"kernel_2x2",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(2, 2)
          .groups(24)}},
    {"kernel_2x2_with_depth_multiplier",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(2, 2)
          .groups(24)
          .group_output_channels(3)}},
    {"kernel_2x2_without_bias",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .has_bias(false)
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(2, 2)
          .groups(24)}},
    {"kernel_3x3",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .groups(24)}},
    {"kernel_3x3_with_depth_multiplier",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .groups(24)
          .group_output_channels(3)}},
    {"kernel_3x3_without_bias",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .has_bias(false)
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .groups(24)}},
    {"kernel_5x5",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .padding(2, 2)
          .kernel_size(5, 5)
          .groups(24)}},
    {"kernel_5x5_with_depth_multiplier",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .padding(2, 2)
          .kernel_size(5, 5)
          .groups(24)
          .group_output_channels(3)}},
    {"kernel_5x5_without_bias",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .has_bias(false)
          .input_size(15, 14)
          .padding(2, 2)
          .kernel_size(5, 5)
          .groups(24)}},
    {"kernel_7x7",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .padding(3, 3)
          .kernel_size(7, 7)
          .groups(24)}},
    {"kernel_7x7_without_bias",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .has_bias(false)
          .input_size(15, 14)
          .padding(3, 3)
          .kernel_size(7, 7)
          .groups(24)}},
    /* Tests dwconv microkernel with weights cache.  */
    {"weights_cache_1x1",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .kernel_size(1, 1)
          .groups(24)
          .use_weights_cache(true)}},
    /* Tests dwconv microkernel with non 1x1 kernel (dwconv_hwg packing). */
    {"weights_cache_2x2",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(2, 2)
          .groups(24)
          .use_weights_cache(true)}},
    {"kernel_3x3s2_with_tf_same_padding",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(14, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .groups(24),
      ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(14, 15)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .groups(24),
      ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .groups(24),
      ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 15)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .groups(24)}},
};

static const ConvolutionTestCase kConvolutionSetupTests[] = {
    {"setup_changing_input_buffer",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .kernel_height(5)
          .kernel_width(3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_changing_input_buffer_grouped",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .kernel_height(5)
          .kernel_width(3)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_changing_input_buffer_depthwise",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .kernel_height(3)
          .kernel_width(3)
          .groups(19)
          .group_input_channels(1)
          .group_output_channels(1)}},
    {"setup_increasing_batch",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .next_batch_size(5)
          .input_height(8)
          .input_width(8)
          .kernel_height(5)
          .kernel_width(3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_increasing_batch_grouped",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .next_batch_size(5)
          .input_height(8)
          .input_width(8)
          .kernel_height(5)
          .kernel_width(3)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_increasing_batch_depthwise",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .next_batch_size(5)
          .input_height(8)
          .input_width(8)
          .kernel_height(3)
          .kernel_width(3)
          .groups(19)
          .group_input_channels(1)
          .group_output_channels(1)}},
    {"setup_decreasing_batch",
     {ConvolutionOperatorTester()
          .batch_size(5)
          .next_batch_size(3)
          .input_height(8)
          .input_width(8)
          .kernel_height(5)
          .kernel_width(3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_decreasing_batch_grouped",
     {ConvolutionOperatorTester()
          .batch_size(5)
          .next_batch_size(3)
          .input_height(8)
          .input_width(8)
          .kernel_height(5)
          .kernel_width(3)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_decreasing_batch_depthwise",
     {ConvolutionOperatorTester()
          .batch_size(5)
          .next_batch_size(3)
          .input_height(8)
          .input_width(8)
          .kernel_height(3)
          .kernel_width(3)
          .groups(19)
          .group_input_channels(1)
          .group_output_channels(1)}},
    {"setup_changing_height_1",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_height(9)
          .kernel_height(5)
          .kernel_width(3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_changing_height_2",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_height(7)
          .kernel_height(5)
          .kernel_width(3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_changing_height_grouped_1",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_height(9)
          .kernel_height(5)
          .kernel_width(3)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_changing_height_grouped_2",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_height(7)
          .kernel_height(5)
          .kernel_width(3)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_changing_height_depthwise_1",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_height(9)
          .kernel_height(3)
          .kernel_width(3)
          .groups(19)
          .group_input_channels(1)
          .group_output_channels(1)}},
    {"setup_changing_height_depthwise_2",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_height(7)
          .kernel_height(3)
          .kernel_width(3)
          .groups(19)
          .group_input_channels(1)
          .group_output_channels(1)}},
    {"setup_changing_width_1",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_width(9)
          .kernel_height(5)
          .kernel_width(3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_changing_width_2",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_width(7)
          .kernel_height(5)
          .kernel_width(3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_changing_width_grouped_1",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_width(9)
          .kernel_height(5)
          .kernel_width(3)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_changing_width_grouped_2",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_width(7)
          .kernel_height(5)
          .kernel_width(3)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_changing_width_depthwise_1",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_width(9)
          .kernel_height(3)
          .kernel_width(3)
          .groups(19)
          .group_input_channels(1)
          .group_output_channels(1)}},
    {"setup_changing_width_depthwise_2",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_width(7)
          .kernel_height(3)
          .kernel_width(3)
          .groups(19)
          .group_input_channels(1)
          .group_output_channels(1)}},
    {"setup_swap_height_and_width",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(9)
          .input_width(8)
          .next_input_height(8)
          .next_input_width(9)
          .kernel_height(5)
          .kernel_width(3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_swap_height_and_width_grouped",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(9)
          .input_width(8)
          .next_input_height(8)
          .next_input_width(9)
          .kernel_height(5)
          .kernel_width(3)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_swap_height_and_width_depthwise",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(9)
          .input_width(8)
          .next_input_height(8)
          .next_input_width(9)
          .kernel_height(3)
          .kernel_width(3)
          .groups(19)
          .group_input_channels(1)
          .group_output_channels(1)}}};

#define CREATE_CONVOLUTION_TESTS(test_suite_name, test_fn)                \
  using test_suite_name = testing::TestWithParam<ConvolutionTestCase>;    \
  TEST_P(test_suite_name, ConvolutionTest) {                              \
    const ConvolutionTestCase& test_case = GetParam();                    \
    for (const ConvolutionOperatorTester& tester : test_case.second) {    \
      tester.test_fn();                                                   \
    }                                                                     \
  }                                                                       \
  INSTANTIATE_TEST_SUITE_P(                                               \
      test_suite_name, test_suite_name,                                   \
      testing::ValuesIn<ConvolutionTestCase>(kConvolutionTests),          \
      [](const testing::TestParamInfo<ConvolutionTestCase>& info) {       \
        return info.param.first;                                          \
      });                                                                 \
  INSTANTIATE_TEST_SUITE_P(                                               \
      DEPTHWISE_##test_suite_name, test_suite_name,                       \
      testing::ValuesIn<ConvolutionTestCase>(kDepthwiseConvolutionTests), \
      [](const testing::TestParamInfo<ConvolutionTestCase>& info) {       \
        return info.param.first;                                          \
      });

#define CREATE_CONVOLUTION_SETUP_TESTS(test_suite_name, setup_test_fn)         \
  using SETUP_##test_suite_name = testing::TestWithParam<ConvolutionTestCase>; \
  TEST_P(SETUP_##test_suite_name, ConvolutionSetupTest) {                      \
    const ConvolutionTestCase& test_case = GetParam();                         \
    for (const ConvolutionOperatorTester& tester : test_case.second) {         \
      tester.setup_test_fn();                                                  \
    }                                                                          \
  }                                                                            \
  INSTANTIATE_TEST_SUITE_P(                                                    \
      SETUP_##test_suite_name, SETUP_##test_suite_name,                        \
      testing::ValuesIn<ConvolutionTestCase>(kConvolutionSetupTests),          \
      [](const testing::TestParamInfo<ConvolutionTestCase>& info) {            \
        return info.param.first;                                               \
      });

CREATE_CONVOLUTION_TESTS(CONVOLUTION_NHWC_QC8, TestNHWCxQC8)
CREATE_CONVOLUTION_SETUP_TESTS(CONVOLUTION_NHWC_QC8, TestSetupNHWCxQC8)

CREATE_CONVOLUTION_TESTS(CONVOLUTION_NHWC_PQS8_QS8_QC8W, TestNHWCxPQC8)
CREATE_CONVOLUTION_SETUP_TESTS(CONVOLUTION_NHWC_PQS8_QS8_QC8W,
                               TestSetupNHWCxPQC8)

CREATE_CONVOLUTION_TESTS(CONVOLUTION_NHWC_QD8_F16_QC8W, TestNHWCxQD8F16QC8W)

CREATE_CONVOLUTION_TESTS(CONVOLUTION_NHWC_QD8_F32_QC8W, TestNHWCxQD8F32QC8W)

// Installs counting hooks that track how many aligned allocations are still
// live, so that a zero buffer the operator forgets to release is observable,
// and restores the previous allocator when it goes out of scope, so a failed
// assertion cannot leave the hooks installed.
class CountingAllocatorGuard {
 public:
  CountingAllocatorGuard() : saved_allocator_(xnn_params.allocator) {
    xnn_params.allocator.context = this;
    xnn_params.allocator.aligned_allocate = AlignedAllocate;
    xnn_params.allocator.aligned_deallocate = AlignedDeallocate;
  }
  ~CountingAllocatorGuard() { xnn_params.allocator = saved_allocator_; }

  // Non-copyable/non-movable because xnn_params.allocator.context holds
  // `this`.
  CountingAllocatorGuard(const CountingAllocatorGuard&) = delete;
  CountingAllocatorGuard& operator=(const CountingAllocatorGuard&) = delete;

  size_t live_aligned_allocations() const { return live_aligned_allocations_; }

 private:
  static void* AlignedAllocate(void* context, size_t alignment, size_t size) {
    auto* self = static_cast<CountingAllocatorGuard*>(context);
    void* pointer = self->saved_allocator_.aligned_allocate(
        self->saved_allocator_.context, alignment, size);
    if (pointer != nullptr) {
      self->live_aligned_allocations_++;
    }
    return pointer;
  }
  static void AlignedDeallocate(void* context, void* pointer) {
    auto* self = static_cast<CountingAllocatorGuard*>(context);
    if (pointer != nullptr) {
      self->live_aligned_allocations_--;
    }
    self->saved_allocator_.aligned_deallocate(self->saved_allocator_.context,
                                              pointer);
  }

  const struct xnn_allocator saved_allocator_;
  size_t live_aligned_allocations_ = 0;
};

// The dynamically quantized reshape grows convolution_op->zero_buffers and
// bumps valid_batch_size before it delegates to the shared reshape, which can
// still reject the shape and leave xnn_operator::batch_size behind. Deleting
// the operator has to release every zero buffer that valid_batch_size covers.
TEST(CONVOLUTION_NHWC_QD8_F32_QC8W,
     zero_buffers_released_after_failed_reshape) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  const size_t input_channels = 8;
  const size_t output_channels = 8;
  const size_t kernel_height = 3;
  const size_t kernel_width = 3;
  std::vector<int8_t> kernel(
      output_channels * kernel_height * kernel_width * input_channels, 0);
  std::vector<float> bias(output_channels, 0.0f);
  std::vector<float> kernel_scale(output_channels, 1.0f);

  const CountingAllocatorGuard allocator_guard;

  xnn_operator_t convolution_op = nullptr;
  enum xnn_status status = xnn_create_convolution2d_nhwc_qd8_f32_qc8w(
      /*input_padding_top=*/1, /*input_padding_right=*/1,
      /*input_padding_bottom=*/1, /*input_padding_left=*/1, kernel_height,
      kernel_width, /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1, /*groups=*/1, input_channels,
      output_channels,
      /*input_channel_stride=*/input_channels,
      /*output_channel_stride=*/output_channels, kernel_scale.data(),
      kernel.data(), bias.data(), /*output_min=*/-1.0e+30f,
      /*output_max=*/1.0e+30f, /*flags=*/0, /*weights_cache=*/nullptr,
      &convolution_op);

  if (status == xnn_status_success) {
    size_t workspace_size = 0;
    size_t output_height = 0;
    size_t output_width = 0;
    ASSERT_EQ(xnn_status_success,
              xnn_reshape_convolution2d_nhwc_qd8_f32_qc8w(
                  convolution_op, /*batch_size=*/1, /*input_height=*/4,
                  /*input_width=*/4, &workspace_size, &output_height,
                  &output_width, /*threadpool=*/nullptr));

    // Rejected for the zero input height, but only after the zero buffers for
    // the eight batches have been allocated.
    ASSERT_EQ(xnn_status_invalid_parameter,
              xnn_reshape_convolution2d_nhwc_qd8_f32_qc8w(
                  convolution_op, /*batch_size=*/8, /*input_height=*/0,
                  /*input_width=*/4, &workspace_size, &output_height,
                  &output_width, /*threadpool=*/nullptr));

    ASSERT_EQ(xnn_status_success, xnn_delete_operator(convolution_op));
  }

  ASSERT_EQ(xnn_status_success, status);
  EXPECT_EQ(allocator_guard.live_aligned_allocations(), 0);
}

CREATE_CONVOLUTION_TESTS(CONVOLUTION_NHWC_QS8, TestNHWCxQS8)
CREATE_CONVOLUTION_SETUP_TESTS(CONVOLUTION_NHWC_QS8, TestSetupNHWCxQS8)

CREATE_CONVOLUTION_TESTS(CONVOLUTION_NHWC_QU8, TestNHWCxQU8)
CREATE_CONVOLUTION_SETUP_TESTS(CONVOLUTION_NHWC_QU8, TestSetupNHWCxQU8)

CREATE_CONVOLUTION_TESTS(CONVOLUTION_NHWC_F32, TestNHWCxF32)
CREATE_CONVOLUTION_SETUP_TESTS(CONVOLUTION_NHWC_F32, TestSetupNHWCxF32)

CREATE_CONVOLUTION_TESTS(CONVOLUTION_NHWC_F16, TestNHWCxF16)
CREATE_CONVOLUTION_SETUP_TESTS(CONVOLUTION_NHWC_F16, TestSetupNHWCxF16)

TEST(CONVOLUTION_NHWC_PF16, native_goki_uses_configured_packer) {
  TestPackedFloatGoki<true>();
}

TEST(CONVOLUTION_NHWC_PF32, native_goki_uses_configured_packer) {
  TestPackedFloatGoki<false>();
}

TEST(CONVOLUTION_NHWC_PF16, fp32_static_weights_preserve_conversion) {
  TestPackedFloatGoki<true, true>();
}

TEST(CONVOLUTION_NHWC_PF32, f16_weights_preserve_conversion) {
  TestPackedFloatGoki<false, true>();
}

TEST(CONVOLUTION_NHWC, conversion_buffer_size_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  constexpr size_t group_input_channels =
      std::numeric_limits<size_t>::max() / sizeof(float) + 17;
  constexpr size_t group_output_channels = 1;
  const std::array<uint16_t, 17> kernel{};
  xnn_operator_t convolution_op = nullptr;

  EXPECT_EQ(
      xnn_status_invalid_parameter,
      xnn_create_convolution2d_nhwc_f32_f16(
          0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, group_input_channels,
          group_output_channels, group_input_channels, group_output_channels,
          kernel.data(), nullptr, -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::infinity(), 0, nullptr, &convolution_op));
  EXPECT_EQ(nullptr, convolution_op);
}

TEST(CONVOLUTION_NHWC, fp32_static_bias_size_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  constexpr size_t group_input_channels = 1;
  constexpr size_t group_output_channels =
      std::numeric_limits<size_t>::max() / sizeof(uint16_t) + 9;
  const std::array<uint16_t, 1> kernel{};
  const std::array<float, 17> bias{};
  xnn_operator_t convolution_op = nullptr;

  EXPECT_EQ(
      xnn_status_invalid_parameter,
      xnn_create_convolution2d_nhwc_f16(
          0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, group_input_channels,
          group_output_channels, group_input_channels, group_output_channels,
          kernel.data(), bias.data(), -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::infinity(), XNN_FLAG_FP32_STATIC_BIASES,
          nullptr, &convolution_op));
  EXPECT_EQ(nullptr, convolution_op);
}
#if XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI && XNN_ENABLE_ARM_SME2
TEST(CONVOLUTION_NHWC_F32, depthwise_3x3_kleidiai) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr));
  if (xnn_init_kai_f32_dwconv_config() == nullptr) {
    GTEST_SKIP() << "KleidiAI F32 DWConv requires SME2";
  }
  const size_t channel_tile =
      xnn_f32_dwconv_minmax_ukernel_9pvc__neonsme2_get_channel_tile();
  const size_t groups = channel_tile + 1;
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(3, 3)
      .groups(groups)
      .input_channel_stride(groups)
      .output_channel_stride(groups)
      .TestNHWCxF32(
          /*expected_workspace_size=*/0,
          /*expected_microkernel_type=*/xnn_microkernel_type_kai_dwconv);
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(3, 3)
      .groups(groups)
      .input_channel_stride(groups + 1)
      .output_channel_stride(groups)
      .TestNHWCxF32(
          /*expected_workspace_size=*/SIZE_MAX,
          /*expected_microkernel_type=*/xnn_microkernel_type_dwconv);
}
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI && XNN_ENABLE_ARM_SME2

TEST(CONVOLUTION_NHWC_F32, unioutput_1x1) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_1x1_with_qmin) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .qmin(128)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_1x1_with_qmax) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .qmax(128)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_1x1_with_input_stride) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .input_channel_stride(28)
      .group_input_channels(23)
      .group_output_channels(19)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_1x1_with_output_stride) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .output_channel_stride(29)
      .group_input_channels(23)
      .group_output_channels(19)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_grouped_1x1) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(13)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_grouped_1x1_with_qmin) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(13)
      .qmin(128)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_grouped_1x1_with_qmax) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(13)
      .qmax(128)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_grouped_1x1_with_input_stride) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .groups(2)
      .input_channel_stride(37)
      .group_input_channels(17)
      .group_output_channels(13)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_grouped_1x1_with_output_stride) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .groups(2)
      .output_channel_stride(41)
      .group_input_channels(17)
      .group_output_channels(13)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_3x3) {
  ConvolutionOperatorTester()
      .input_size(3, 3)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(23)
      .group_output_channels(19)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_3x3_with_qmin) {
  ConvolutionOperatorTester()
      .input_size(3, 3)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(23)
      .group_output_channels(19)
      .qmin(128)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_3x3_with_qmax) {
  ConvolutionOperatorTester()
      .input_size(3, 3)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(23)
      .group_output_channels(19)
      .qmax(128)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_3x3_with_input_stride) {
  ConvolutionOperatorTester()
      .input_size(3, 3)
      .padding(1)
      .kernel_size(3, 3)
      .input_channel_stride(28)
      .group_input_channels(23)
      .group_output_channels(19)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_3x3_with_output_stride) {
  ConvolutionOperatorTester()
      .input_size(3, 3)
      .padding(1)
      .kernel_size(3, 3)
      .output_channel_stride(29)
      .group_input_channels(23)
      .group_output_channels(19)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_grouped_3x3) {
  ConvolutionOperatorTester()
      .input_size(3, 3)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(13)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_grouped_3x3_with_qmin) {
  ConvolutionOperatorTester()
      .input_size(3, 3)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(13)
      .qmin(128)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_grouped_3x3_with_qmax) {
  ConvolutionOperatorTester()
      .input_size(3, 3)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(13)
      .qmax(128)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_grouped_3x3_with_input_stride) {
  ConvolutionOperatorTester()
      .input_size(3, 3)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .input_channel_stride(37)
      .group_input_channels(17)
      .group_output_channels(13)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_grouped_3x3_with_output_stride) {
  ConvolutionOperatorTester()
      .input_size(3, 3)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .output_channel_stride(41)
      .group_input_channels(17)
      .group_output_channels(13)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_1x1) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .kernel_size(1, 1)
      .groups(24)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_1x1_without_bias) {
  ConvolutionOperatorTester()
      .has_bias(false)
      .input_size(15, 14)
      .kernel_size(1, 1)
      .groups(24)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .groups(24)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2_without_bias) {
  ConvolutionOperatorTester()
      .has_bias(false)
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .groups(24)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2s2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .subsampling(2)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2s1x2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .subsampling(1, 2)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2s2x1) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .subsampling(2, 1)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2d2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .dilation(2)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2d1x2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .dilation(1, 2)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2d2x1) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .dilation(2, 1)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5_without_bias) {
  ConvolutionOperatorTester()
      .has_bias(false)
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5s2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .subsampling(2)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5s1x2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .subsampling(1, 2)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5s2x1) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .subsampling(2, 1)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5d2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .dilation(2)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5d1x2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .dilation(1, 2)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5d2x1) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .dilation(2, 1)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F16, grouped_1x1_with_fp32_weights) {
  ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .input_size(24, 25)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(19)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_1x1) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_1x1_with_qmin) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .qmin(128)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_1x1_with_qmax) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .qmax(128)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_1x1_with_input_stride) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .input_channel_stride(28)
      .group_input_channels(23)
      .group_output_channels(19)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_1x1_with_output_stride) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .output_channel_stride(29)
      .group_input_channels(23)
      .group_output_channels(19)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_grouped_1x1) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(13)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_grouped_1x1_with_qmin) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(13)
      .qmin(128)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_grouped_1x1_with_qmax) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(13)
      .qmax(128)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_grouped_1x1_with_input_stride) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .groups(2)
      .input_channel_stride(37)
      .group_input_channels(17)
      .group_output_channels(13)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_grouped_1x1_with_output_stride) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .groups(2)
      .output_channel_stride(41)
      .group_input_channels(17)
      .group_output_channels(13)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, kernel_3x3_with_fp32_weights) {
  ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .input_size(13, 12)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(17)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, grouped_3x3_with_fp32_weights) {
  ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .input_size(10, 11)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(14)
      .group_output_channels(13)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_3x3) {
  ConvolutionOperatorTester()
      .input_size(3, 3)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(23)
      .group_output_channels(19)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_3x3_with_qmin) {
  ConvolutionOperatorTester()
      .input_size(3, 3)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(23)
      .group_output_channels(19)
      .qmin(128)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_3x3_with_qmax) {
  ConvolutionOperatorTester()
      .input_size(3, 3)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(23)
      .group_output_channels(19)
      .qmax(128)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_3x3_with_input_stride) {
  ConvolutionOperatorTester()
      .input_size(3, 3)
      .padding(1)
      .kernel_size(3, 3)
      .input_channel_stride(28)
      .group_input_channels(23)
      .group_output_channels(19)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_3x3_with_output_stride) {
  ConvolutionOperatorTester()
      .input_size(3, 3)
      .padding(1)
      .kernel_size(3, 3)
      .output_channel_stride(29)
      .group_input_channels(23)
      .group_output_channels(19)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_grouped_3x3) {
  ConvolutionOperatorTester()
      .input_size(3, 3)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(13)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_grouped_3x3_with_qmin) {
  ConvolutionOperatorTester()
      .input_size(3, 3)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(13)
      .qmin(128)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_grouped_3x3_with_qmax) {
  ConvolutionOperatorTester()
      .input_size(3, 3)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(13)
      .qmax(128)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_grouped_3x3_with_input_stride) {
  ConvolutionOperatorTester()
      .input_size(3, 3)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .input_channel_stride(37)
      .group_input_channels(17)
      .group_output_channels(13)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_grouped_3x3_with_output_stride) {
  ConvolutionOperatorTester()
      .input_size(3, 3)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .output_channel_stride(41)
      .group_input_channels(17)
      .group_output_channels(13)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_1x1) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .kernel_size(1, 1)
      .groups(24)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_1x1_with_fp32_weights) {
  ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .input_size(15, 14)
      .kernel_size(1, 1)
      .groups(24)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_1x1_without_bias) {
  ConvolutionOperatorTester()
      .has_bias(false)
      .input_size(15, 14)
      .kernel_size(1, 1)
      .groups(24)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .groups(24)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2_without_bias) {
  ConvolutionOperatorTester()
      .has_bias(false)
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .groups(24)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2s2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .subsampling(2)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2s1x2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .subsampling(1, 2)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2s2x1) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .subsampling(2, 1)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2d2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .dilation(2)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2d1x2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .dilation(1, 2)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2d2x1) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .dilation(2, 1)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_3x3_with_fp32_weights) {
  ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(3, 3)
      .groups(24)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5_without_bias) {
  ConvolutionOperatorTester()
      .has_bias(false)
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5s2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .subsampling(2)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5s1x2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .subsampling(1, 2)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5s2x1) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .subsampling(2, 1)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5d2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .dilation(2)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5d1x2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .dilation(1, 2)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5d2x1) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .dilation(2, 1)
      .groups(27)
      .TestNHWCxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16, kernel_1x1_with_fp32_weights) {
  ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .depthwise_layout(true)
      .input_size(15, 14)
      .kernel_size(1, 1)
      .groups(24)
      .TestNHWCxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16,
     kernel_1x1_with_depth_multiplier_with_fp32_weights) {
  ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .depthwise_layout(true)
      .input_size(15, 14)
      .kernel_size(1, 1)
      .groups(24)
      .group_output_channels(3)
      .TestNHWCxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16, kernel_3x3_weight_fp32_weights) {
  ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .depthwise_layout(true)
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(3, 3)
      .groups(24)
      .TestNHWCxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16,
     kernel_3x3_with_depth_multiplier_with_fp32_weights) {
  ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .depthwise_layout(true)
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(3, 3)
      .groups(24)
      .group_output_channels(3)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_QS8, reject_scale_buffer_size_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr));
  const int8_t kernel[1] = {0};
  const int32_t bias[1] = {0};
  xnn_operator_t convolution_op = nullptr;
  const xnn_status status = xnn_create_convolution2d_nhwc_qs8(
      0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1.0f, 1.0f, kernel,
      bias, 0, 1.0f, -128, 127, 0, nullptr, &convolution_op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(status, xnn_status_success);
  xnn_delete_operator(convolution_op);

  const size_t overflowing_group_output_channels =
      std::numeric_limits<size_t>::max() / sizeof(float) + 1;
  convolution_op = nullptr;
  EXPECT_EQ(
      xnn_status_invalid_parameter,
      xnn_create_convolution2d_nhwc_qs8(
          0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
          overflowing_group_output_channels, 1, 1, 0, 1.0f, 1.0f, kernel,
          bias, 0, 1.0f, -128, 127, 0, nullptr, &convolution_op));

  const size_t overflowing_output_channels =
      std::numeric_limits<size_t>::max() / 2 + 1;
  convolution_op = nullptr;
  EXPECT_EQ(
      xnn_status_invalid_parameter,
      xnn_create_convolution2d_nhwc_qs8(
          0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 1,
          overflowing_output_channels, 2, 2, 0, 1.0f, 1.0f, kernel, bias,
          0, 1.0f, -128, 127, 0, nullptr, &convolution_op));
}

TEST(CONVOLUTION_NHWC_QS8_QC8W, reject_scale_buffer_size_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr));
  const int8_t kernel[4] = {0};
  const int32_t bias[4] = {0};
  const float kernel_scale[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  xnn_operator_t convolution_op = nullptr;
  const xnn_status status = xnn_create_convolution2d_nhwc_qs8_qc8w(
      0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 0, 1.0f, kernel_scale,
      kernel, bias, 0, 1.0f, -128, 127, 0, nullptr, &convolution_op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(status, xnn_status_success);
  xnn_delete_operator(convolution_op);

  const size_t overflowing_output_channels =
      std::numeric_limits<size_t>::max() / 2 + 1;
  convolution_op = nullptr;
  EXPECT_EQ(
      xnn_status_invalid_parameter,
      xnn_create_convolution2d_nhwc_qs8_qc8w(
          0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 1,
          overflowing_output_channels, 2, 2, 0, 1.0f, kernel_scale, kernel,
          bias, 0, 1.0f, -128, 127, 0, nullptr, &convolution_op));
}

TEST(CONVOLUTION_NHWC_PQS8_QS8_QS8, reject_scale_buffer_size_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr));
  const int8_t kernel[1] = {0};
  const int32_t bias[1] = {0};
  xnn_operator_t convolution_op = nullptr;
  const xnn_status status = xnn_create_convolution2d_nhwc_pqs8_qs8_qs8(
      0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1.0f, 1.0f, kernel,
      bias, 0, 1.0f, -128, 127, 0, nullptr, &convolution_op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(status, xnn_status_success);
  xnn_delete_operator(convolution_op);

  const size_t overflowing_group_output_channels =
      std::numeric_limits<size_t>::max() / sizeof(float) + 1;
  convolution_op = nullptr;
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_create_convolution2d_nhwc_pqs8_qs8_qs8(
                0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                overflowing_group_output_channels, 1, 1, 0, 1.0f, 1.0f,
                kernel, bias, 0, 1.0f, -128, 127, 0, nullptr,
                &convolution_op));

  const size_t overflowing_output_channels =
      std::numeric_limits<size_t>::max() / 2 + 1;
  convolution_op = nullptr;
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_create_convolution2d_nhwc_pqs8_qs8_qs8(
                0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 1,
                overflowing_output_channels, 2, 2, 0, 1.0f, 1.0f, kernel,
                bias, 0, 1.0f, -128, 127, 0, nullptr, &convolution_op));
}

TEST(CONVOLUTION_NHWC_F32, input_channels_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr));
  constexpr uint32_t groups = std::numeric_limits<uint32_t>::max();
  constexpr size_t group_input_channels =
      std::numeric_limits<size_t>::max() / groups + 1;
  const float kernel[1] = {0.0f};
  const float bias[1] = {0.0f};
  xnn_operator_t convolution_op = nullptr;

  EXPECT_EQ(
      xnn_status_invalid_parameter,
      xnn_create_convolution2d_nhwc_f32(
          0, 0, 0, 0, 1, 1, 1, 1, 1, 1, groups, group_input_channels, 1,
          group_input_channels, 1, kernel, bias,
          -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::infinity(), 0, nullptr,
          &convolution_op));
  EXPECT_EQ(nullptr, convolution_op);
}

TEST(CONVOLUTION_NHWC_F32, output_channels_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr));
  constexpr uint32_t groups = std::numeric_limits<uint32_t>::max();
  constexpr size_t group_output_channels =
      std::numeric_limits<size_t>::max() / groups + 1;
  const float kernel[1] = {0.0f};
  const float bias[1] = {0.0f};
  xnn_operator_t convolution_op = nullptr;

  EXPECT_EQ(
      xnn_status_invalid_parameter,
      xnn_create_convolution2d_nhwc_f32(
          0, 0, 0, 0, 1, 1, 1, 1, 1, 1, groups, 1, group_output_channels,
          1, group_output_channels, kernel, bias,
          -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::infinity(), 0, nullptr,
          &convolution_op));
  EXPECT_EQ(nullptr, convolution_op);
}

TEST(CONVOLUTION_NHWC_F32, batch_stride_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr));

  const float kernel[1] = {1.0f};
  const float bias[1] = {0.0f};
  xnn_operator_t convolution_op = nullptr;

  ASSERT_EQ(
      xnn_status_success,
      xnn_create_convolution2d_nhwc_f32(
          0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          kernel, bias,
          -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::infinity(), 0, nullptr,
          &convolution_op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      convolution_op, xnn_delete_operator);
  size_t workspace_size = 0;
  const size_t overflow_batch = (SIZE_MAX / 4) + 1;
  EXPECT_EQ(
      xnn_status_out_of_memory,
      xnn_reshape_convolution2d_nhwc_f32(
          convolution_op, overflow_batch, 1, 1, &workspace_size,
          nullptr, nullptr, nullptr));
}

TEST(DWCONV_CONFIG, valid_configs_and_dwconv_dispatch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr));

  const xnn_dwconv_config* f32_config = xnn_init_f32_dwconv_config();
  const xnn_dwconv_config* qs8_qc8w_config = xnn_init_qs8_qc8w_dwconv_config();
  const xnn_dwconv_config* qs8_config = xnn_init_qs8_dwconv_config();
  const xnn_dwconv_config* qu8_config = xnn_init_qu8_dwconv_config();
  ASSERT_NE(nullptr, f32_config);
  ASSERT_NE(nullptr, qs8_qc8w_config);
  ASSERT_NE(nullptr, qs8_config);
  ASSERT_NE(nullptr, qu8_config);

  for (size_t i = 0; i < XNN_MAX_F32_DWCONV_UKERNELS; ++i) {
    if (f32_config[i].primary_tile != 0) {
      EXPECT_NE(nullptr, f32_config[i].minmax);
      EXPECT_NE(nullptr, f32_config[i].init.f32);
      EXPECT_GT(f32_config[i].channel_tile, 0);
    }
  }
  for (size_t i = 0; i < XNN_MAX_QC8_DWCONV_UKERNELS; ++i) {
    if (qs8_qc8w_config[i].primary_tile != 0) {
      EXPECT_NE(nullptr, qs8_qc8w_config[i].minmax);
      EXPECT_NE(nullptr, qs8_qc8w_config[i].init.qs8_qc8w);
      EXPECT_GT(qs8_qc8w_config[i].channel_tile, 0);
    }
  }
  EXPECT_EQ(9, qs8_config[0].primary_tile);
  EXPECT_EQ(25, qs8_config[1].primary_tile);
  for (size_t i = 0; i < XNN_MAX_QS8_DWCONV_UKERNELS; ++i) {
    if (qs8_config[i].primary_tile != 0) {
      EXPECT_NE(nullptr, qs8_config[i].minmax);
      EXPECT_NE(nullptr, qs8_config[i].init.qs8);
      EXPECT_GT(qs8_config[i].channel_tile, 0);
    }
  }
  EXPECT_EQ(9, qu8_config[0].primary_tile);
  EXPECT_EQ(25, qu8_config[1].primary_tile);
  for (size_t i = 0; i < XNN_MAX_QU8_DWCONV_UKERNELS; ++i) {
    if (qu8_config[i].primary_tile != 0) {
      EXPECT_NE(nullptr, qu8_config[i].minmax);
      EXPECT_NE(nullptr, qu8_config[i].init.qu8);
      EXPECT_GT(qu8_config[i].channel_tile, 0);
    }
  }

  // Ensure init_qu8_dwconv_config does not overwrite qs8_dwconv_config and
  // init_qs8_dwconv_config uses per-tensor qs8 init rather than qs8_qc8w init.
  EXPECT_NE(qs8_config[0].minmax, qu8_config[0].minmax);
  EXPECT_NE(qs8_config[1].minmax, qu8_config[1].minmax);
  EXPECT_NE(reinterpret_cast<uintptr_t>(qs8_config[0].init.qs8),
            reinterpret_cast<uintptr_t>(qs8_qc8w_config[1].init.qs8_qc8w));

  const xnn_dwconv_config* f16_config = xnn_init_f16_dwconv_config();
  if (f16_config != nullptr) {
    EXPECT_NE(0, f16_config[0].primary_tile);
    for (size_t i = 0; i < XNN_MAX_F16_DWCONV_UKERNELS; ++i) {
      if (f16_config[i].primary_tile != 0) {
        EXPECT_NE(nullptr, f16_config[i].minmax);
        EXPECT_NE(nullptr, f16_config[i].init.f16);
        EXPECT_GT(f16_config[i].channel_tile, 0);
      }
    }
  }

  // Verify 3x3 and 5x5 depthwise convolutions dispatch dwconv rather than
  // falling back to igemm.
  constexpr uint32_t kGroups = 8;
  const std::vector<uint8_t> qu8_kernel(5 * 5 * kGroups, 128);
  const std::vector<int8_t> qs8_kernel(5 * 5 * kGroups, 0);
  const std::vector<int32_t> q_bias(kGroups, 0);
  const std::vector<float> qc8_scale(kGroups, 1.0f);
  for (uint32_t k : {3u, 5u}) {
    xnn_operator_t qu8_op = nullptr;
    ASSERT_EQ(
        xnn_status_success,
        xnn_create_convolution2d_nhwc_qu8(
            0, 0, 0, 0, k, k, 1, 1, 1, 1, kGroups, 1, 1, kGroups, kGroups, 128,
            1.0f, 128, 1.0f, qu8_kernel.data(), q_bias.data(), 128, 1.0f, 0,
            255, XNN_FLAG_DEPTHWISE_CONVOLUTION, nullptr, &qu8_op));
    std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_qu8_op(
        qu8_op, xnn_delete_operator);
    EXPECT_EQ(xnn_microkernel_type_dwconv, qu8_op->ukernel.type);

    xnn_operator_t qc8_op = nullptr;
    ASSERT_EQ(
        xnn_status_success,
        xnn_create_convolution2d_nhwc_qs8_qc8w(
            0, 0, 0, 0, k, k, 1, 1, 1, 1, kGroups, 1, 1, kGroups, kGroups, 0,
            1.0f, qc8_scale.data(), qs8_kernel.data(), q_bias.data(), 0, 1.0f,
            -128, 127, XNN_FLAG_DEPTHWISE_CONVOLUTION, nullptr, &qc8_op));
    std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_qc8_op(
        qc8_op, xnn_delete_operator);
    EXPECT_EQ(xnn_microkernel_type_dwconv, qc8_op->ukernel.type);
  }
}
}  // namespace
