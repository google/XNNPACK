// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/indirection.h"
#include "xnnpack/math.h"
#include "xnnpack/operator-utils.h"
#include "xnnpack/operator.h"

namespace xnnpack {
namespace {


// Constant index to refer to zero buffer.
static constexpr uintptr_t kZero = SIZE_MAX;

class IndirectionTester {
 public:
  IndirectionTester& input_height(size_t input_height) {
    input_height_ = input_height;
    return *this;
  }

  IndirectionTester& input_width(size_t input_width) {
    input_width_ = input_width;
    return *this;
  }

  IndirectionTester& kernel_height(size_t kernel_height) {
    kernel_height_ = kernel_height;
    return *this;
  }

  IndirectionTester& kernel_width(size_t kernel_width) {
    kernel_width_ = kernel_width;
    return *this;
  }

  IndirectionTester& padding_height(size_t padding_height) {
    padding_height_ = padding_height;
    return *this;
  }

  IndirectionTester& padding_width(size_t padding_width) {
    padding_width_ = padding_width;
    return *this;
  }

  IndirectionTester& subsampling(size_t subsampling) {
    subsampling_ = subsampling;
    return *this;
  }

  IndirectionTester& dilation(size_t dilation) {
    dilation_ = dilation;
    return *this;
  }

  IndirectionTester& channels(size_t channels) {
    channels_ = channels;
    return *this;
  }

  IndirectionTester& primary_tile(size_t primary_tile) {
    primary_tile_ = primary_tile;
    return *this;
  }

  IndirectionTester& channel_tile(size_t channel_tile) {
    channel_tile_ = channel_tile;
    return *this;
  }

  IndirectionTester& expected_indices(std::vector<size_t> expected_indices) {
    expected_indices_ = expected_indices;
    return *this;
  }

  void Test() {
    IndirectionInit();
    EXPECT_EQ(indirection_buffer_.size(), expected_indices_.size());
    for (size_t i = 0; i < expected_indices_.size(); i++) {
      EXPECT_EQ(indirection_buffer_[i], &input_[expected_indices_[i]])
          << "i: " << i << ", input_index:" << expected_indices_[i];
    }
  }

  void TestCompressed() {
    IndirectionCompressedInit();
    EXPECT_EQ(indirection_buffer_.size(), expected_indices_.size());
    for (size_t i = 0; i < expected_indices_.size(); i++) {
      if (expected_indices_[i] == kZero) {
        EXPECT_EQ(indirection_buffer_[i], zero_buffer_.data())
            << "i: " << i << ", zero_buffer: " << zero_buffer_.data();
      } else {
        EXPECT_EQ(indirection_buffer_[i], &input_[expected_indices_[i]])
            << "i: " << i << ", input_index:" << expected_indices_[i];
      }
    }
  }

 private:
  void IndirectionInit() {
    const size_t kernel_size = kernel_height_ * kernel_width_;
    const size_t output_height = xnn_compute_convolution_output_dimension(
        input_height_ + padding_height_, kernel_height_, dilation_, subsampling_);
    const size_t output_width = xnn_compute_convolution_output_dimension(
        input_width_ + padding_width_, kernel_width_, dilation_, subsampling_);
    const size_t step_width = dilation_ == 1 ? min(subsampling_, kernel_width_) : kernel_width_;
    const size_t step_height =
        kernel_size + (output_width - 1) * step_width * kernel_height_;

    input_ = std::vector<float>(channels_ * input_height_ * input_width_);
    std::iota(input_.begin(), input_.end(), 0.0f);
    zero_buffer_ = std::vector<float>(channels_);

    const size_t num_indirection_elements = (primary_tile_ - kernel_size) + output_height * step_height;
    indirection_buffer_ = std::vector<const float*>(num_indirection_elements);
    xnn_operator op = {};
    op.indirection_buffer = reinterpret_cast<const void**>(indirection_buffer_.data());
    op.input = input_.data();
    op.input_pixel_stride = channels_;
    op.zero_buffer = zero_buffer_.data();
    op.input_height = input_height_;
    op.input_width = input_width_;
    op.output_height = output_height;
    op.output_width = output_width;
    op.kernel_height = kernel_height_;
    op.kernel_width = kernel_width_;
    op.stride_height = subsampling_;
    op.stride_width = subsampling_;
    op.dilation_height = dilation_;
    op.dilation_width = dilation_;
    op.padding_top = padding_height_ / 2;
    op.padding_left = padding_width_ / 2;
    xnn_indirection_init_dwconv2d(
      /*output_y_start=*/0, /*output_y_end=*/output_height,
      op.indirection_buffer,
      op.input,
      op.input_pixel_stride << /*log2_input_element_size=*/2,
      op.zero_buffer,
      op.input_height, op.input_width,
      op.output_height, op.output_width,
      op.kernel_height, op.kernel_width,
      op.stride_height, op.stride_width,
      op.dilation_height, op.dilation_width,
      op.padding_top, op.padding_left,
      step_height, step_width, primary_tile_);
  }

  void IndirectionCompressedInit() {
    const size_t kernel_size = kernel_height_ * kernel_width_;
    const size_t output_height = xnn_compute_convolution_output_dimension(
        input_height_ + padding_height_, kernel_height_, dilation_, subsampling_);
    const size_t output_width = xnn_compute_convolution_output_dimension(
        input_width_ + padding_width_, kernel_width_, dilation_, subsampling_);
    const size_t step_width = dilation_ == 1 ? min(subsampling_, kernel_width_) : kernel_width_;
    const size_t step_height =
        kernel_size + (output_width - 1) * step_width * kernel_height_;

    input_ = std::vector<float>(channels_ * input_height_ * input_width_);
    std::iota(input_.begin(), input_.end(), 0);
    zero_buffer_ = std::vector<float>(channels_);

    const size_t indirect_top_height = divide_round_up(padding_height_ / 2, subsampling_);
    const size_t indirect_bot_height = divide_round_up(padding_height_ / 2, subsampling_);
    const size_t indirection_buffer_output_height = (indirect_top_height + indirect_bot_height + 1);

    const size_t num_indirection_elements = (primary_tile_ - kernel_size) + indirection_buffer_output_height * step_height;
    indirection_buffer_ = std::vector<const float*>(num_indirection_elements);
    xnn_operator op = {};
    op.indirection_buffer = reinterpret_cast<const void**>(indirection_buffer_.data());
    op.input = input_.data();
    op.input_pixel_stride = channels_;
    op.zero_buffer = zero_buffer_.data();
    op.input_height = input_height_;
    op.input_width = input_width_;
    op.output_height = output_height;
    op.output_width = output_width;
    op.kernel_height = kernel_height_;
    op.kernel_width = kernel_width_;
    op.stride_height = subsampling_;
    op.stride_width = subsampling_;
    op.dilation_height = dilation_;
    op.dilation_width = dilation_;
    op.padding_top = padding_height_ / 2;
    op.padding_left = padding_width_ / 2;
    xnn_indirection_init_dwconv2d_compressed(
      /*output_y_start=*/0, /*output_y_end=*/output_height,
      op.indirection_buffer,
      op.input,
      op.input_pixel_stride << /*log2_input_element_size=*/2,
      op.zero_buffer,
      op.input_height, op.input_width,
      op.output_height, op.output_width,
      op.kernel_height, op.kernel_width,
      op.stride_height, op.stride_width,
      op.dilation_height, op.dilation_width,
      op.padding_top, op.padding_left,
      step_height, step_width,
      indirect_top_height,
      indirect_bot_height,
      primary_tile_);
  }

  // Set by tests using setter functions.
  size_t input_height_;
  size_t input_width_;
  size_t kernel_height_;
  size_t kernel_width_;
  size_t padding_height_ = 0;
  size_t padding_width_ = 0;
  size_t subsampling_ = 1;
  size_t dilation_ = 1;
  size_t channels_ = 1;
  size_t primary_tile_;
  size_t channel_tile_ = 1;
  std::vector<size_t> expected_indices_;

  // Initialized by IndirectionInit.
  std::vector<const float*> indirection_buffer_;
  std::vector<float> input_;
  std::vector<float> zero_buffer_;
};

TEST(INDIRECTION, input3x3_kernel1x1) {
  IndirectionTester()
      .input_height(3)
      .input_width(3)
      .kernel_height(1)
      .kernel_width(1)
      .primary_tile(1)
      .expected_indices({0, 1, 2, 3, 4, 5, 6, 7, 8})
      .Test();
}

TEST(INDIRECTION, input3x3_kernel2x2) {
  IndirectionTester()
      .input_height(3)
      .input_width(3)
      .kernel_height(2)
      .kernel_width(2)
      .primary_tile(4)
      // input:  kernel:
      // 0 1 2   a b
      // 3 4 5   c d
      // 6 7 8
      .expected_indices({
        // For each output row, column major, and compress pointers within a single output row.
        0, 3, 1, 4, 2, 5,
        3, 6, 4, 7, 5, 8,
      })
      .Test();
}

TEST(INDIRECTION, input3x3_kernel1x1_subsampling2) {
  IndirectionTester()
      .input_height(3)
      .input_width(3)
      .kernel_height(1)
      .kernel_width(1)
      .subsampling(2)
      .primary_tile(1)
      // input:  kernel:
      // 0 1 2   a
      // 3 4 5
      // 6 7 8
      .expected_indices({
        0, 2,
        6, 8,
      })
      .Test();
}

TEST(INDIRECTION, input4x4_kernel2x2_subsampling2) {
  IndirectionTester()
      .input_height(4)
      .input_width(4)
      .kernel_height(2)
      .kernel_width(2)
      .subsampling(2)
      .primary_tile(4)
      // input:       kernel:
      // 0  1  2  3   a b
      // 4  5  6  7   c d
      // 8  9  10 11
      // 12 13 14 15
      .expected_indices({
        0, 4, 1, 5, 2, 6, 3, 7,
        8, 12, 9, 13, 10, 14, 11, 15,
      })
      .Test();
}

TEST(INDIRECTION, input4x4_kernel2x1_primarytile4) {
  IndirectionTester()
      .input_height(4)
      .input_width(4)
      .kernel_height(2)
      .kernel_width(1)
      .primary_tile(4)
      // input:       kernel:
      // 0  1  2  3   a
      // 4  5  6  7   b
      // 8  9  10 11
      // 12 13 14 15
      .expected_indices({
        0, 4, 1, 5, 2, 6, 3, 7,
        4, 8, 5, 9, 6, 10, 7, 11,
        8, 12, 9, 13, 10, 14, 11, 15,
        // 4 - (2 x 2) extra elements.
        15, 15,
      })
      .Test();
}

TEST(INDIRECTION, input4x4_kernel1x2_primarytile4_subsampling2) {
  IndirectionTester()
      .input_height(4)
      .input_width(4)
      .kernel_height(1)
      .kernel_width(2)
      .primary_tile(4)
      .subsampling(2)
      // input:       kernel:
      // 0  1  2  3   a b
      // 4  5  6  7
      // 8  9  10 11
      // 12 13 14 15
      .expected_indices({
        0, 1, 2, 3, 8, 9, 10, 11,
        // primary_tile - kernel_size (4 - 2) extra elements, set to last input pixel.
        11, 11,
      })
      .Test();
}

TEST(INDIRECTION, input4x4_kernel2x1_primarytile4_subsampling2) {
  IndirectionTester()
      .input_height(4)
      .input_width(4)
      .kernel_height(2)
      .kernel_width(1)
      .primary_tile(4)
      .subsampling(2)
      // input:       kernel:  output:
      // 0  1  2  3   a        A B
      // 4  5  6  7   b        C D
      // 8  9  10 11
      // 12 13 14 15
      .expected_indices({
        0, 4, 2, 6,
        8, 12, 10, 14,
        // primary_tile - kernel_size (4 - 2) extra elements, set to last input pixel.
        14, 14
      })
      .Test();
}

TEST(INDIRECTION_COMPRESSED, input3x3_kernel1x1) {
  IndirectionTester()
      .input_height(3)
      .input_width(3)
      .kernel_height(1)
      .kernel_width(1)
      .primary_tile(1)
      .expected_indices({0, 1, 2})
      .TestCompressed();
}

TEST(INDIRECTION_COMPRESSED, input3x3_kernel2x2_padding2x2) {
  IndirectionTester()
      .input_height(3)
      .input_width(3)
      .kernel_height(2)
      .kernel_width(2)
      .padding_height(2)
      .padding_width(2)
      .primary_tile(4)
      // input:     kernel:
      // 0 0 0 0 0  a b
      // 0 0 1 2 0  c d
      // 0 3 4 5 0
      // 0 6 7 8 0
      // 0 0 0 0 0
      .expected_indices({
        // Top section.
        kZero, kZero, kZero, 0, kZero, 1, kZero, 2, kZero, kZero,
        // Compressed rows
        kZero, kZero, 0, 3, 1, 4, 2, 5, kZero, kZero,
        // Compressed rows.
        kZero, kZero, 6, kZero, 7, kZero, 8, kZero, kZero, kZero,
      })
      .TestCompressed();
}

TEST(INDIRECTION_COMPRESSED, input2x2_kernel2x2_padding2x2_subsampling2) {
  IndirectionTester()
      .input_height(2)
      .input_width(2)
      .kernel_height(2)
      .kernel_width(2)
      .padding_height(2)
      .padding_width(2)
      .primary_tile(4)
      .subsampling(2)
      // input:   kernel:
      // 0 0 0 0  a b
      // 0 0 1 0  c d
      // 0 2 3 0
      // 0 0 0 0
      .expected_indices({
        // Top section.
        kZero, kZero, kZero, 0, kZero, 1, kZero, kZero,
        // We don't actually need a compressed row, but it is easier to assume that there is always 1 compressed row,
        // and this is the same as the bottom section.
        kZero, kZero, 2, kZero, 3, kZero, kZero, kZero,
        // Bottom section is just all 0.
        kZero, kZero, 2, kZero, 3, kZero, kZero, kZero,
      })
      .TestCompressed();
}
}  // namespace
}  // namespace xnnpack
