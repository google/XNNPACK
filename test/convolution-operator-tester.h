// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "convolution-test-helpers.h"
#include <fp16/fp16.h>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/cache.h>
#include <xnnpack/allocator.h>


class ConvolutionOperatorTester {
 public:
  enum class WeightsType {
    Default,
    FP32,
  };

  inline ConvolutionOperatorTester& padding_tf_same(bool padding_same) {
    if (padding_same) {
      assert(padding_top() == 0);
      assert(padding_left() == 0);
      assert(padding_bottom() == 0);
      assert(padding_right() == 0);
    }
    this->padding_tf_same_ = padding_same;
    return *this;
  }

  inline bool padding_tf_same() const {
    return this->padding_tf_same_;
  }

  inline ConvolutionOperatorTester& padding(uint32_t padding) {
    assert(!padding_tf_same());
    this->padding_top_ = padding;
    this->padding_right_ = padding;
    this->padding_bottom_ = padding;
    this->padding_left_ = padding;
    return *this;
  }

  inline ConvolutionOperatorTester& padding(uint32_t padding_height, uint32_t padding_width) {
    assert(!padding_tf_same());
    this->padding_top_ = padding_height;
    this->padding_right_ = padding_width;
    this->padding_bottom_ = padding_height;
    this->padding_left_ = padding_width;
    return *this;
  }

  inline ConvolutionOperatorTester& padding_height(uint32_t padding_height) {
    assert(!padding_tf_same());
    this->padding_top_ = padding_height;
    this->padding_bottom_ = padding_height;
    return *this;
  }

  inline ConvolutionOperatorTester& padding_width(uint32_t padding_width) {
    assert(!padding_tf_same());
    this->padding_right_ = padding_width;
    this->padding_left_ = padding_width;
    return *this;
  }

  inline ConvolutionOperatorTester& padding_top(uint32_t padding_top) {
    assert(!padding_tf_same());
    this->padding_top_ = padding_top;
    return *this;
  }

  inline uint32_t padding_top() const {
    if (padding_tf_same()) {
      const uint32_t total_padding_height =
        (output_height() - 1) * subsampling_height() + dilated_kernel_height() - input_height();
      return total_padding_height / 2;
    } else {
      return this->padding_top_;
    }
  }

  inline ConvolutionOperatorTester& padding_left(uint32_t padding_left) {
    assert(!padding_tf_same());
    this->padding_left_ = padding_left;
    return *this;
  }

  inline uint32_t padding_left() const {
    if (padding_tf_same()) {
      const uint32_t total_padding_width =
        (output_width() - 1) * subsampling_width() + dilated_kernel_width() - input_width();
      return total_padding_width / 2;
    } else {
      return this->padding_left_;
    }
  }

  inline ConvolutionOperatorTester& padding_bottom(uint32_t padding_bottom) {
    assert(!padding_tf_same());
    this->padding_bottom_ = padding_bottom;
    return *this;
  }

  inline uint32_t padding_bottom() const {
    if (padding_tf_same()) {
      const uint32_t total_padding_height =
        (output_height() - 1) * subsampling_height() + dilated_kernel_height() - input_height();
      return total_padding_height - total_padding_height / 2;
    } else {
      return this->padding_bottom_;
    }
  }

  inline ConvolutionOperatorTester& padding_right(uint32_t padding_right) {
    assert(!padding_tf_same());
    this->padding_right_ = padding_right;
    return *this;
  }

  inline uint32_t padding_right() const {
    if (padding_tf_same()) {
      const uint32_t total_padding_width =
        (output_width() - 1) * subsampling_width() + dilated_kernel_width() - input_width();
      return total_padding_width - total_padding_width / 2;
    } else {
      return this->padding_right_;
    }
  }

  inline ConvolutionOperatorTester& input_size(uint32_t input_height, uint32_t input_width) {
    assert(input_height >= 1);
    assert(input_width >= 1);
    this->input_height_ = input_height;
    this->input_width_ = input_width;
    return *this;
  }

  inline ConvolutionOperatorTester& input_height(uint32_t input_height) {
    assert(input_height >= 1);
    this->input_height_ = input_height;
    return *this;
  }

  inline uint32_t input_height() const {
    return this->input_height_;
  }

  inline ConvolutionOperatorTester& input_width(uint32_t input_width) {
    assert(input_width >= 1);
    this->input_width_ = input_width;
    return *this;
  }

  inline uint32_t input_width() const {
    return this->input_width_;
  }

  inline ConvolutionOperatorTester& groups(uint32_t groups) {
    assert(groups >= 1);
    this->groups_ = groups;
    return *this;
  }

  inline uint32_t groups() const {
    return this->groups_;
  }

  inline ConvolutionOperatorTester& group_input_channels(size_t group_input_channels) {
    assert(group_input_channels >= 1);
    this->group_input_channels_ = group_input_channels;
    return *this;
  }

  inline size_t group_input_channels() const {
    return this->group_input_channels_;
  }

  inline ConvolutionOperatorTester& group_output_channels(size_t group_output_channels) {
    assert(group_output_channels >= 1);
    this->group_output_channels_ = group_output_channels;
    return *this;
  }

  inline size_t group_output_channels() const {
    return this->group_output_channels_;
  }

  inline ConvolutionOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size >= 1);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline ConvolutionOperatorTester& kernel_size(uint32_t kernel_size) {
    assert(kernel_size >= 1);
    this->kernel_height_ = kernel_size;
    this->kernel_width_ = kernel_size;
    return *this;
  }

  inline ConvolutionOperatorTester& kernel_size(uint32_t kernel_height, uint32_t kernel_width) {
    assert(kernel_height >= 1);
    assert(kernel_width >= 1);
    this->kernel_height_ = kernel_height;
    this->kernel_width_ = kernel_width;
    return *this;
  }

  inline ConvolutionOperatorTester& transient_indirection_buffer(bool use_transient_buffer) {
    this->transient_indirection_buffer_ = use_transient_buffer;
    return *this;
  }

  inline bool transient_indirection_buffer() const {
    return this->transient_indirection_buffer_;
  }

  inline ConvolutionOperatorTester& kernel_height(uint32_t kernel_height) {
    assert(kernel_height >= 1);
    this->kernel_height_ = kernel_height;
    return *this;
  }

  inline uint32_t kernel_height() const {
    return this->kernel_height_;
  }

  inline ConvolutionOperatorTester& kernel_width(uint32_t kernel_width) {
    assert(kernel_width >= 1);
    this->kernel_width_ = kernel_width;
    return *this;
  }

  inline uint32_t kernel_width() const {
    return this->kernel_width_;
  }

  inline ConvolutionOperatorTester& dilation(uint32_t dilation) {
    assert(dilation >= 1);
    this->dilation_height_ = dilation;
    this->dilation_width_ = dilation;
    return *this;
  }

  inline ConvolutionOperatorTester& dilation(uint32_t dilation_height, uint32_t dilation_width) {
    assert(dilation_height >= 1);
    assert(dilation_width >= 1);
    this->dilation_height_ = dilation_height;
    this->dilation_width_ = dilation_width;
    return *this;
  }

  inline ConvolutionOperatorTester& dilation_height(uint32_t dilation_height) {
    assert(dilation_height >= 1);
    this->dilation_height_ = dilation_height;
    return *this;
  }

  inline uint32_t dilation_height() const {
    return this->dilation_height_;
  }

  inline ConvolutionOperatorTester& dilation_width(uint32_t dilation_width) {
    assert(dilation_width >= 1);
    this->dilation_width_ = dilation_width;
    return *this;
  }

  inline uint32_t dilation_width() const {
    return this->dilation_width_;
  }

  inline ConvolutionOperatorTester& subsampling(uint32_t subsampling) {
    assert(subsampling >= 1);
    this->subsampling_height_ = subsampling;
    this->subsampling_width_ = subsampling;
    return *this;
  }

  inline ConvolutionOperatorTester& subsampling(uint32_t subsampling_height, uint32_t subsampling_width) {
    assert(subsampling_height >= 1);
    assert(subsampling_width >= 1);
    this->subsampling_height_ = subsampling_height;
    this->subsampling_width_ = subsampling_width;
    return *this;
  }

  inline ConvolutionOperatorTester& subsampling_height(uint32_t subsampling_height) {
    assert(subsampling_height >= 1);
    this->subsampling_height_ = subsampling_height;
    return *this;
  }

  inline uint32_t subsampling_height() const {
    return this->subsampling_height_;
  }

  inline ConvolutionOperatorTester& subsampling_width(uint32_t subsampling_width) {
    assert(subsampling_width >= 1);
    this->subsampling_width_ = subsampling_width;
    return *this;
  }

  inline uint32_t subsampling_width() const {
    return this->subsampling_width_;
  }

  inline ConvolutionOperatorTester& input_channel_stride(size_t input_channel_stride) {
    assert(input_channel_stride >= 1);
    this->input_channel_stride_ = input_channel_stride;
    return *this;
  }

  inline size_t input_channel_stride() const {
    if (this->input_channel_stride_ == 0) {
      return group_input_channels() * groups();
    } else {
      assert(this->input_channel_stride_ >= group_input_channels() * groups());
      return this->input_channel_stride_;
    }
  }

  inline ConvolutionOperatorTester& output_channel_stride(size_t output_channel_stride) {
    assert(output_channel_stride >= 1);
    this->output_channel_stride_ = output_channel_stride;
    return *this;
  }

  inline size_t output_channel_stride() const {
    if (this->output_channel_stride_ == 0) {
      return group_output_channels() * groups();
    } else {
      assert(this->output_channel_stride_ >= group_output_channels() * groups());
      return this->output_channel_stride_;
    }
  }

  inline uint32_t dilated_kernel_height() const {
    return (kernel_height() - 1) * dilation_height() + 1;
  }

  inline uint32_t dilated_kernel_width() const {
    return (kernel_width() - 1) * dilation_width() + 1;
  }

  inline size_t output_height() const {
    if (padding_tf_same()) {
      return (input_height() + subsampling_height() - 1) / subsampling_height();
    } else {
      const size_t padded_input_height = padding_top() + input_height() + padding_bottom();
      if (padded_input_height <= dilated_kernel_height()) {
        return 1;
      } else {
        return (padded_input_height - dilated_kernel_height()) / subsampling_height() + 1;
      }
    }
  }

  inline size_t output_width() const {
    if (padding_tf_same()) {
      return (input_width() + subsampling_width() - 1) / subsampling_width();
    } else {
      const size_t padded_input_width = padding_left() + input_width() + padding_right();
      if (padded_input_width <= dilated_kernel_width()) {
        return 1;
      } else {
        return (padded_input_width - dilated_kernel_width()) / subsampling_width() + 1;
      }
    }
  }

  inline ConvolutionOperatorTester& next_input_size(uint32_t next_input_height, uint32_t next_input_width) {
    assert(next_input_height >= 1);
    assert(next_input_width >= 1);
    this->next_input_height_ = next_input_height;
    this->next_input_width_ = next_input_width;
    return *this;
  }

  inline ConvolutionOperatorTester& next_input_height(uint32_t next_input_height) {
    assert(next_input_height >= 1);
    this->next_input_height_ = next_input_height;
    return *this;
  }

  inline uint32_t next_input_height() const {
    if (this->next_input_height_ == 0) {
      return input_height();
    } else {
      return this->next_input_height_;
    }
  }

  inline ConvolutionOperatorTester& next_input_width(uint32_t next_input_width) {
    assert(next_input_width >= 1);
    this->next_input_width_ = next_input_width;
    return *this;
  }

  inline uint32_t next_input_width() const {
    if (this->next_input_width_ == 0) {
      return input_width();
    } else {
      return this->next_input_width_;
    }
  }

  inline size_t next_output_height() const {
    const size_t padded_input_height = padding_top() + next_input_height() + padding_bottom();
    if (padded_input_height <= dilated_kernel_height()) {
      return 1;
    } else {
      return (padded_input_height - dilated_kernel_height()) / subsampling_height() + 1;
    }
  }

  inline size_t next_output_width() const {
    const size_t padded_input_width = padding_left() + next_input_width() + padding_right();
    if (padded_input_width <= dilated_kernel_width()) {
      return 1;
    } else {
      return (padded_input_width - dilated_kernel_width()) / subsampling_width() + 1;
    }
  }

  inline ConvolutionOperatorTester& next_batch_size(size_t next_batch_size) {
    assert(next_batch_size >= 1);
    this->next_batch_size_ = next_batch_size;
    return *this;
  }

  inline size_t next_batch_size() const {
    if (this->next_batch_size_ == 0) {
      return batch_size();
    } else {
      return this->next_batch_size_;
    }
  }

  inline ConvolutionOperatorTester& sparsity(float sparsity) {
    this->sparsity_ = sparsity;
    return *this;
  }

  inline float sparsity() const {
    return this->sparsity_;
  }

  inline ConvolutionOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline ConvolutionOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline ConvolutionOperatorTester& force_nhwc_input(bool force_nhwc_input) {
    this->force_nhwc_input_ = force_nhwc_input;
    return *this;
  }

  inline bool force_nhwc_input() const {
    return this->force_nhwc_input_;
  }

  inline ConvolutionOperatorTester& depthwise_layout(bool depthwise_layout) {
    this->depthwise_layout_ = depthwise_layout;
    return *this;
  }

  inline bool depthwise_layout() const {
    return this->depthwise_layout_;
  }

  inline ConvolutionOperatorTester& has_bias(bool has_bias) {
    this->has_bias_ = has_bias;
    return *this;
  }

  inline bool has_bias() const {
    return this->has_bias_;
  }

  inline ConvolutionOperatorTester& weights_type(WeightsType weights_type) {
    this->weights_type_ = weights_type;
    return *this;
  }

  inline WeightsType weights_type() const {
    return this->weights_type_;
  }

  inline ConvolutionOperatorTester& multithreaded(size_t multithreaded) {
    this->multithreaded_ = multithreaded;
    return *this;
  }

  inline size_t multithreaded() const {
    return this->multithreaded_;
  }

  size_t num_threads() const {
    // Do not spin up excessive number of threads for tests.
    return multithreaded() ? 5 : 1;
  }

  inline ConvolutionOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

#if XNN_PLATFORM_JIT
  inline ConvolutionOperatorTester& use_jit(bool use_jit) {
    this->use_jit_ = use_jit;
    return *this;
  }

  inline bool use_jit() const {
    return this->use_jit_;
  }
#endif

  inline ConvolutionOperatorTester& use_weights_cache(bool use_weights_cache) {
    this->use_weights_cache_ = use_weights_cache;
    return *this;
  }

  inline bool use_weights_cache() const {
    return this->use_weights_cache_;
  }

  void TestNHWCxQC8() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    std::uniform_int_distribution<int32_t> w8dist(
      -std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) +
      batch_size() * ((input_height() * input_width() - 1) * input_channel_stride() + groups() * group_input_channels()));
    std::vector<int8_t> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<int32_t> bias(groups() * group_output_channels());
    std::vector<int8_t> output(batch_size() * ((output_height() * output_width() - 1) * output_channel_stride() + groups() * group_output_channels()));
    std::vector<int32_t> accumulators(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<double> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<float> requantization_scales(groups() * group_output_channels());

    const int8_t input_zero_point = -1;
    const int8_t output_zero_point = -1;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      // Compute reference results, without renormalization.
      if (depthwise_layout()) {
        ASSERT_EQ(group_input_channels(), 1);
        xnnpack::compute_depthwise_convolution_qs8_reference_results(
          batch_size(),
          output_height(),
          output_width(),
          input_height(),
          input_width(),
          padding_top(),
          padding_right(),
          padding_bottom(),
          padding_left(),
          kernel_height(),
          kernel_width(),
          subsampling_height(),
          subsampling_width(),
          dilation_height(),
          dilation_width(),
          groups(),
          group_output_channels(),
          input_channel_stride(),
          input_zero_point,
          input,
          kernel,
          accumulators,
          has_bias(),
          bias);
      } else {
        xnnpack::compute_convolution_qs8_reference_results(
          batch_size(),
          output_height(),
          output_width(),
          input_height(),
          input_width(),
          padding_top(),
          padding_right(),
          padding_bottom(),
          padding_left(),
          kernel_height(),
          kernel_width(),
          subsampling_height(),
          subsampling_width(),
          dilation_height(),
          dilation_width(),
          groups(),
          group_input_channels(),
          group_output_channels(),
          input_channel_stride(),
          input_zero_point,
          input,
          kernel,
          accumulators,
          has_bias(),
          bias);
      }

      // Compute renormalization parameters.
      for (size_t c = 0; c < groups() * group_output_channels(); c++) {
        int32_t accumulated_min = accumulators[c];
        int32_t accumulated_max = accumulators[c];
        for (size_t px = 0; px < batch_size() * output_height() * output_width(); px++) {
          accumulated_min = std::min(accumulated_min, accumulators[px * groups() * group_output_channels() + c]);
          accumulated_max = std::max(accumulated_max, accumulators[px * groups() * group_output_channels() + c]);
        }

        float requantization_scale = 0x1.0p-32f;
        if (accumulated_max != 0) {
          requantization_scale = std::max(requantization_scale,
            float(int32_t(std::numeric_limits<int8_t>::max()) - int32_t(output_zero_point)) / float(accumulated_max));
        }
        if (accumulated_min != 0) {
          requantization_scale = std::max(requantization_scale,
            float(int32_t(std::numeric_limits<int8_t>::min()) - int32_t(output_zero_point)) / float(accumulated_min));
        }
        requantization_scale = std::min(requantization_scale, 0x1.FFFFFEp-1f);

        requantization_scales[c] = requantization_scale;
      }

      // Renormalize reference results.
      for (size_t c = 0; c < groups() * group_output_channels(); c++) {
        for (size_t px = 0; px < batch_size() * output_height() * output_width(); px++) {
          output_ref[px * groups() * group_output_channels() + c] = double(int32_t(output_zero_point)) +
            double(accumulators[px * groups() * group_output_channels() + c]) * double(requantization_scales[c]);
        }
      }
      std::transform(output_ref.cbegin(), output_ref.cend(), output_ref.begin(),
        [this](double x) -> double {
          return std::max<double>(std::min<double>(x, double(qmax() - 0x80)), double(qmin() - 0x80));
        });

      // Create, setup, run, and destroy Convolution operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convolution_op = nullptr;
      struct xnn_internal_weights_cache* internal_weights_cache = nullptr;
      std::unique_ptr<xnn_weights_cache_provider, decltype(&xnn_delete_weights_cache)> auto_weights_cache(
        nullptr, xnn_delete_weights_cache);
      if (use_weights_cache()) {
        xnn_weights_cache_t weights_cache = nullptr;
        xnn_create_weights_cache(&weights_cache);
        auto_weights_cache.reset(weights_cache);
        if (weights_cache) {
          internal_weights_cache = (struct xnn_internal_weights_cache*) weights_cache->context;
        }
      }

      uint32_t flags = 0;
      if (depthwise_layout()) {
        flags |= XNN_FLAG_DEPTHWISE_CONVOLUTION;
      }
      if (padding_tf_same()) {
        flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
      }
      if (transient_indirection_buffer()) {
        flags |= XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER;
      }
      xnn_status status = xnn_create_convolution2d_nhwc_qs8_qc8w(
          padding_tf_same() ? 0 : padding_top(), padding_tf_same() ? 0 : padding_right(),
          padding_tf_same() ? 0 : padding_bottom(), padding_tf_same() ? 0 : padding_left(),
          kernel_height(), kernel_width(),
          subsampling_height(), subsampling_width(),
          dilation_height(), dilation_width(),
          groups(), group_input_channels(), group_output_channels(),
          input_channel_stride(), output_channel_stride(),
          input_zero_point, 1.0f /* input scale */, requantization_scales.data(),
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_zero_point, 1.0f /* output scale */, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80),
          flags,
          /*code_cache=*/nullptr,
          auto_weights_cache.get(),
          &convolution_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, convolution_op);
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete convolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convolution_op(convolution_op, xnn_delete_operator);
      size_t workspace_size = SIZE_MAX;
      size_t workspace_alignment = SIZE_MAX;
      ASSERT_EQ(
        xnn_status_success, xnn_reshape_convolution2d_nhwc_qs8_qc8w(
                              convolution_op, batch_size(), input_height(), input_width(),
                              &workspace_size, &workspace_alignment,
                              /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                              auto_threadpool.get()));
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
      std::iota(workspace.begin(), workspace.end(), 0);
      if (transient_indirection_buffer()) {
        ASSERT_NE(workspace_size, 0);
        ASSERT_NE(workspace_size, SIZE_MAX);
        ASSERT_EQ(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
        ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_qs8_qc8w(convolution_op, workspace.data(), input.data(), output.data()));
      } else {
        ASSERT_NE(workspace_size, SIZE_MAX);
        ASSERT_NE(workspace_alignment, SIZE_MAX);
        ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_qs8_qc8w(convolution_op, workspace.data(), input.data(), output.data()));
      }
      ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op, auto_threadpool.get()));

      // Verify results.
      VerifyNHWCxQC8(output, output_ref);

      if (use_weights_cache()) {
        xnn_operator_t convolution_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;

        xnn_status status = xnn_create_convolution2d_nhwc_qs8_qc8w(
            padding_tf_same() ? 0 : padding_top(), padding_tf_same() ? 0 : padding_right(),
            padding_tf_same() ? 0 : padding_bottom(), padding_tf_same() ? 0 : padding_left(),
            kernel_height(), kernel_width(),
            subsampling_height(), subsampling_width(),
            dilation_height(), dilation_width(),
            groups(), group_input_channels(), group_output_channels(),
            input_channel_stride(), output_channel_stride(),
            input_zero_point, 1.0f /* input scale */, requantization_scales.data(),
            kernel.data(), has_bias() ? bias.data() : nullptr,
            output_zero_point, 1.0f /* output scale */, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80),
            flags,
            /*code_cache=*/nullptr,
            auto_weights_cache.get(),
            &convolution_op2);
        (void) status;
        ASSERT_NE(nullptr, convolution_op2);

        // Smart pointer to automatically delete convolution_op2.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convolution_op(convolution_op2, xnn_delete_operator);
        std::vector<int8_t> output2(output.size(), INT8_C(0xA5));
        size_t workspace_size = SIZE_MAX;
        size_t workspace_alignment = SIZE_MAX;
        ASSERT_EQ(
          xnn_status_success,
          xnn_reshape_convolution2d_nhwc_qs8_qc8w(
            convolution_op2, batch_size(), input_height(), input_width(),
            &workspace_size, &workspace_alignment,
            /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
            auto_threadpool.get()));
        std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
        std::iota(workspace.begin(), workspace.end(), 0);
        if (transient_indirection_buffer()) {
          ASSERT_NE(workspace_size, 0);
          ASSERT_NE(workspace_size, SIZE_MAX);
          ASSERT_EQ(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
          ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_qs8_qc8w(convolution_op2, workspace.data(), input.data(), output2.data()));
        } else {
          ASSERT_NE(workspace_size, SIZE_MAX);
          ASSERT_NE(workspace_alignment, SIZE_MAX);
          ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_qs8_qc8w(convolution_op2, workspace.data(), input.data(), output2.data()));
        }
        ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op2, auto_threadpool.get()));

        VerifyNHWCxQC8(output2, output_ref);
        VerifyWeightsCache(*internal_weights_cache, old_weights_cache_size);
      }
    }
  }

  void VerifyNHWCxQC8(const std::vector<int8_t> &output,
                      const std::vector<double> &output_ref) const {
    for (size_t i = 0; i < batch_size(); i++) {
      for (size_t y = 0; y < output_height(); y++) {
        for (size_t x = 0; x < output_width(); x++) {
          for (size_t g = 0; g < groups(); g++) {
            for (size_t c = 0; c < group_output_channels(); c++) {
              EXPECT_LE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c]), int32_t(qmax() - 0x80))
                << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              EXPECT_GE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c]), int32_t(qmin() - 0x80))
                << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              EXPECT_NEAR(
                  output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c],
                  double(output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c]),
                  0.9)
                << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
            }
          }
        }
      }
    }
  }

  void TestNHWCxQD8F16QC8W() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(-1.f, 1.f);
    std::uniform_real_distribution<float> f32idist(0.1f, 1.0f);
    std::uniform_int_distribution<int32_t> w8dist(
        std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    // Weights typically have a Gaussian distrubution centred on zero. A
    // standard deviation of 40 means that > 99.99% of values fall in the
    // -127->127 range. However, the reduce the chance of overflow, we constrain
    // the weights further.
    std::normal_distribution<float> normal_dist{0.f, 10.f};

    std::vector<int8_t> input(
        XNN_EXTRA_BYTES / sizeof(int8_t) +
        batch_size() *
            ((input_height() * input_width() - 1) * input_channel_stride() +
             groups() * group_input_channels()));
    std::vector<int8_t> kernel(groups() * group_output_channels() *
                               kernel_height() * kernel_width() *
                               group_input_channels());
    std::vector<float> bias(groups() * group_output_channels());
    std::vector<uint16_t> output(
        batch_size() *
        ((output_height() * output_width() - 1) * output_channel_stride() +
         groups() * group_output_channels()));
    std::vector<float> output_ref(batch_size() * output_height() *
                                  output_width() * groups() *
                                  group_output_channels());
    std::vector<xnn_qd8_quantization_params> quantization_params(batch_size());
    std::vector<float> kernel_scale(groups() * group_output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)>
          auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }
      std::generate(input.begin(), input.end(), [&]() { return w8dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return (int8_t) normal_dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::generate(kernel_scale.begin(), kernel_scale.end(),
                    [&]() { return f32idist(rng); });
      std::generate(
          quantization_params.begin(), quantization_params.end(), [&]() {
            return xnn_qd8_quantization_params{w8dist(rng), f32idist(rng)};
          });
      std::fill(output.begin(), output.end(), UINT16_C(0xDEAD));

      // Compute reference results.
      xnnpack::compute_convolution_qd8_f32_qc8w_reference_results(
          batch_size(), output_height(), output_width(), input_height(),
          input_width(), padding_top(), padding_right(), padding_bottom(),
          padding_left(), kernel_height(), kernel_width(), subsampling_height(),
          subsampling_width(), dilation_height(), dilation_width(), groups(),
          group_input_channels(), group_output_channels(),
          input_channel_stride(), input, kernel, kernel_scale,
          quantization_params, output_ref, has_bias(), bias);

      const float output_min = -std::numeric_limits<float>::infinity();
      const float output_max = std::numeric_limits<float>::infinity();

      // Clamp reference results.
      for (size_t i = 0; i < output_ref.size(); ++i) {
        output_ref[i] = std::max(std::min(output_ref[i], output_max), output_min);
        output_ref[i] = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_ref[i]));
      }


      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }
      // Create, setup, run, and destroy Convolution operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convolution_op = nullptr;
      struct xnn_internal_weights_cache* internal_weights_cache = nullptr;
      std::unique_ptr<xnn_weights_cache_provider, decltype(&xnn_delete_weights_cache)> auto_weights_cache(
        nullptr, xnn_delete_weights_cache);
      if (use_weights_cache()) {
        xnn_weights_cache_t weights_cache = nullptr;
        xnn_create_weights_cache(&weights_cache);
        auto_weights_cache.reset(weights_cache);
        if (weights_cache) {
          internal_weights_cache = (struct xnn_internal_weights_cache*) weights_cache->context;
        }
      }

      uint32_t flags = 0;
      if (padding_tf_same()) {
        flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
      }
      if (transient_indirection_buffer()) {
        flags |= XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER;
      }
      xnn_status status = xnn_create_convolution2d_nhwc_qd8_f16_qc8w(
          padding_tf_same() ? 0 : padding_top(),
          padding_tf_same() ? 0 : padding_right(),
          padding_tf_same() ? 0 : padding_bottom(),
          padding_tf_same() ? 0 : padding_left(), kernel_height(),
          kernel_width(), subsampling_height(), subsampling_width(),
          dilation_height(), dilation_width(), groups(), group_input_channels(),
          group_output_channels(), input_channel_stride(),
          output_channel_stride(), kernel_scale.data(), kernel.data(),
          has_bias() ? bias.data() : nullptr, output_min, output_max, flags,
          /*code_cache=*/nullptr, auto_weights_cache.get(), &convolution_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, convolution_op);
      if (use_weights_cache()) {
        ASSERT_EQ(
            xnn_status_success,
            xnn_finalize_weights_cache(
                auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete convolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
          auto_convolution_op(convolution_op, xnn_delete_operator);

      size_t workspace_size = SIZE_MAX;
      size_t workspace_alignment = SIZE_MAX;
      ASSERT_EQ(xnn_status_success,
                xnn_reshape_convolution2d_nhwc_qd8_f16_qc8w(
                    convolution_op, batch_size(), input_height(), input_width(),
                    &workspace_size, &workspace_alignment,
                    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                    auto_threadpool.get()));
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>>
          workspace(workspace_size);
      std::iota(workspace.begin(), workspace.end(), 0);
      if (transient_indirection_buffer()) {
        ASSERT_NE(workspace_size, 0);
        ASSERT_NE(workspace_size, SIZE_MAX);
        ASSERT_EQ(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
        ASSERT_EQ(
            xnn_status_success,
            xnn_setup_convolution2d_nhwc_qd8_f16_qc8w(
                convolution_op, workspace.data(), input.data(), output.data(),
                reinterpret_cast<const struct xnn_dynamic_quantization_params*>(
                    quantization_params.data())));
      } else {
        ASSERT_NE(workspace_size, SIZE_MAX);
        ASSERT_NE(workspace_alignment, SIZE_MAX);
        ASSERT_EQ(
            xnn_status_success,
            xnn_setup_convolution2d_nhwc_qd8_f16_qc8w(
                convolution_op, workspace.data(), input.data(), output.data(),
                reinterpret_cast<const struct xnn_dynamic_quantization_params*>(
                    quantization_params.data())));
      }
      ASSERT_EQ(xnn_status_success,
                xnn_run_operator(convolution_op, auto_threadpool.get()));

      VerifyNHWCxF16(output, output_ref, output_min, output_max, batch_size(), output_height(), output_width());

      if (use_weights_cache()) {
        xnn_operator_t convolution_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;

        ASSERT_EQ(xnn_status_success,
                  xnn_create_convolution2d_nhwc_qd8_f16_qc8w(
                      padding_tf_same() ? 0 : padding_top(),
                      padding_tf_same() ? 0 : padding_right(),
                      padding_tf_same() ? 0 : padding_bottom(),
                      padding_tf_same() ? 0 : padding_left(), kernel_height(),
                      kernel_width(), subsampling_height(), subsampling_width(),
                      dilation_height(), dilation_width(), groups(),
                      group_input_channels(), group_output_channels(),
                      input_channel_stride(), output_channel_stride(),
                      kernel_scale.data(), kernel.data(),
                      has_bias() ? bias.data() : nullptr, output_min,
                      output_max, flags, /*code_cache=*/nullptr,
                      auto_weights_cache.get(), &convolution_op2));
        ASSERT_NE(nullptr, convolution_op2);

        // Smart pointer to automatically delete convolution_op.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
            auto_convolution_op(convolution_op2, xnn_delete_operator);

        std::vector<uint16_t> output2(output.size(), UINT16_C(0xDEAD));
        size_t workspace_size = SIZE_MAX;
        size_t workspace_alignment = SIZE_MAX;
        ASSERT_EQ(xnn_status_success,
                  xnn_reshape_convolution2d_nhwc_qd8_f16_qc8w(
                      convolution_op2, batch_size(), input_height(),
                      input_width(), &workspace_size, &workspace_alignment,
                      /*output_height_out=*/nullptr,
                      /*output_width_out=*/nullptr, auto_threadpool.get()));
        std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>>
            workspace(workspace_size);
        std::iota(workspace.begin(), workspace.end(), 0);
        if (transient_indirection_buffer()) {
          ASSERT_NE(workspace_size, 0);
          ASSERT_NE(workspace_size, SIZE_MAX);
          ASSERT_EQ(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
          ASSERT_EQ(xnn_status_success,
                    xnn_setup_convolution2d_nhwc_qd8_f16_qc8w(
                        convolution_op2, workspace.data(), input.data(),
                        output2.data(),
                        reinterpret_cast<
                            const struct xnn_dynamic_quantization_params*>(
                            quantization_params.data())));
        } else {
          ASSERT_NE(workspace_size, SIZE_MAX);
          ASSERT_NE(workspace_alignment, SIZE_MAX);
          ASSERT_EQ(xnn_status_success,
                    xnn_setup_convolution2d_nhwc_qd8_f16_qc8w(
                        convolution_op2, workspace.data(), input.data(),
                        output2.data(),
                        reinterpret_cast<
                            const struct xnn_dynamic_quantization_params*>(
                            quantization_params.data())));
        }
        ASSERT_EQ(xnn_status_success,
                  xnn_run_operator(convolution_op2, auto_threadpool.get()));

        VerifyNHWCxF16(output, output_ref, output_min, output_max, batch_size(), output_height(), output_width());
        VerifyWeightsCache(*internal_weights_cache, old_weights_cache_size);
      }
    }
  }

  void TestNHWCxQD8F32QC8W() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(-1.f, 1.f);
    std::uniform_real_distribution<float> f32idist(0.5f, 2.0f);
    std::uniform_int_distribution<int32_t> w8dist(
        std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input(
        XNN_EXTRA_BYTES / sizeof(int8_t) +
        batch_size() *
            ((input_height() * input_width() - 1) * input_channel_stride() +
             groups() * group_input_channels()));
    std::vector<int8_t> kernel(groups() * group_output_channels() *
                               kernel_height() * kernel_width() *
                               group_input_channels());
    std::vector<float> bias(groups() * group_output_channels());
    std::vector<float> output(
        batch_size() *
        ((output_height() * output_width() - 1) * output_channel_stride() +
         groups() * group_output_channels()));
    std::vector<float> output_ref(batch_size() * output_height() *
                                  output_width() * groups() *
                                  group_output_channels());
    std::vector<xnn_qd8_quantization_params> quantization_params(batch_size());
    std::vector<float> kernel_scale(groups() * group_output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)>
          auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }
      std::generate(input.begin(), input.end(), [&]() { return w8dist(rng); });
      std::generate(kernel.begin(), kernel.end(),
                    [&]() { return w8dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::generate(kernel_scale.begin(), kernel_scale.end(),
                    [&]() { return f32idist(rng); });
      std::generate(
          quantization_params.begin(), quantization_params.end(), [&]() {
            return xnn_qd8_quantization_params{w8dist(rng), f32idist(rng)};
          });
      std::fill(output.begin(), output.end(), nanf(""));

      // Compute reference results.
      xnnpack::compute_convolution_qd8_f32_qc8w_reference_results(
          batch_size(), output_height(), output_width(), input_height(),
          input_width(), padding_top(), padding_right(), padding_bottom(),
          padding_left(), kernel_height(), kernel_width(), subsampling_height(),
          subsampling_width(), dilation_height(), dilation_width(), groups(),
          group_input_channels(), group_output_channels(),
          input_channel_stride(), input, kernel, kernel_scale,
          quantization_params, output_ref, has_bias(), bias);

      const float output_min = -std::numeric_limits<float>::infinity();
      const float output_max = std::numeric_limits<float>::infinity();

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }
      // Create, setup, run, and destroy Convolution operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convolution_op = nullptr;
      struct xnn_internal_weights_cache* internal_weights_cache = nullptr;
      std::unique_ptr<xnn_weights_cache_provider, decltype(&xnn_delete_weights_cache)> auto_weights_cache(
        nullptr, xnn_delete_weights_cache);
      if (use_weights_cache()) {
        xnn_weights_cache_t weights_cache = nullptr;
        xnn_create_weights_cache(&weights_cache);
        auto_weights_cache.reset(weights_cache);
        if (weights_cache) {
          internal_weights_cache = (struct xnn_internal_weights_cache*) weights_cache->context;
        }
      }

      uint32_t flags = 0;
      if (padding_tf_same()) {
        flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
      }
      if (transient_indirection_buffer()) {
        flags |= XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER;
      }
      xnn_status status = xnn_create_convolution2d_nhwc_qd8_f32_qc8w(
          padding_tf_same() ? 0 : padding_top(),
          padding_tf_same() ? 0 : padding_right(),
          padding_tf_same() ? 0 : padding_bottom(),
          padding_tf_same() ? 0 : padding_left(), kernel_height(),
          kernel_width(), subsampling_height(), subsampling_width(),
          dilation_height(), dilation_width(), groups(), group_input_channels(),
          group_output_channels(), input_channel_stride(),
          output_channel_stride(), kernel_scale.data(), kernel.data(),
          has_bias() ? bias.data() : nullptr, output_min, output_max, flags,
          /*code_cache=*/nullptr, auto_weights_cache.get(), &convolution_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, convolution_op);
      if (use_weights_cache()) {
        ASSERT_EQ(
            xnn_status_success,
            xnn_finalize_weights_cache(
                auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete convolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
          auto_convolution_op(convolution_op, xnn_delete_operator);

      size_t workspace_size = SIZE_MAX;
      size_t workspace_alignment = SIZE_MAX;
      ASSERT_EQ(xnn_status_success,
                xnn_reshape_convolution2d_nhwc_qd8_f32_qc8w(
                    convolution_op, batch_size(), input_height(), input_width(),
                    &workspace_size, &workspace_alignment,
                    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                    auto_threadpool.get()));
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>>
          workspace(workspace_size);
      std::iota(workspace.begin(), workspace.end(), 0);
      if (transient_indirection_buffer()) {
        ASSERT_NE(workspace_size, 0);
        ASSERT_NE(workspace_size, SIZE_MAX);
        ASSERT_EQ(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
        ASSERT_EQ(
            xnn_status_success,
            xnn_setup_convolution2d_nhwc_qd8_f32_qc8w(
                convolution_op, workspace.data(), input.data(), output.data(),
                reinterpret_cast<const struct xnn_dynamic_quantization_params*>(
                    quantization_params.data())));
      } else {
        ASSERT_NE(workspace_size, SIZE_MAX);
        ASSERT_NE(workspace_alignment, SIZE_MAX);
        ASSERT_EQ(
            xnn_status_success,
            xnn_setup_convolution2d_nhwc_qd8_f32_qc8w(
                convolution_op, workspace.data(), input.data(), output.data(),
                reinterpret_cast<const struct xnn_dynamic_quantization_params*>(
                    quantization_params.data())));
      }
      ASSERT_EQ(xnn_status_success,
                xnn_run_operator(convolution_op, auto_threadpool.get()));

      VerifyNHWCxF32(output, output_ref, output_min, output_max);

      if (use_weights_cache()) {
        xnn_operator_t convolution_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;

        ASSERT_EQ(xnn_status_success,
                  xnn_create_convolution2d_nhwc_qd8_f32_qc8w(
                      padding_tf_same() ? 0 : padding_top(),
                      padding_tf_same() ? 0 : padding_right(),
                      padding_tf_same() ? 0 : padding_bottom(),
                      padding_tf_same() ? 0 : padding_left(), kernel_height(),
                      kernel_width(), subsampling_height(), subsampling_width(),
                      dilation_height(), dilation_width(), groups(),
                      group_input_channels(), group_output_channels(),
                      input_channel_stride(), output_channel_stride(),
                      kernel_scale.data(), kernel.data(),
                      has_bias() ? bias.data() : nullptr, output_min,
                      output_max, flags, /*code_cache=*/nullptr,
                      auto_weights_cache.get(), &convolution_op2));
        ASSERT_NE(nullptr, convolution_op2);

        // Smart pointer to automatically delete convolution_op.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
            auto_convolution_op(convolution_op2, xnn_delete_operator);

        std::vector<float> output2(output.size(), nanf(""));
        size_t workspace_size = SIZE_MAX;
        size_t workspace_alignment = SIZE_MAX;
        ASSERT_EQ(xnn_status_success,
                  xnn_reshape_convolution2d_nhwc_qd8_f32_qc8w(
                      convolution_op2, batch_size(), input_height(),
                      input_width(), &workspace_size, &workspace_alignment,
                      /*output_height_out=*/nullptr,
                      /*output_width_out=*/nullptr, auto_threadpool.get()));
        std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>>
            workspace(workspace_size);
        std::iota(workspace.begin(), workspace.end(), 0);
        if (transient_indirection_buffer()) {
          ASSERT_NE(workspace_size, 0);
          ASSERT_NE(workspace_size, SIZE_MAX);
          ASSERT_EQ(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
          ASSERT_EQ(xnn_status_success,
                    xnn_setup_convolution2d_nhwc_qd8_f32_qc8w(
                        convolution_op2, workspace.data(), input.data(),
                        output2.data(),
                        reinterpret_cast<
                            const struct xnn_dynamic_quantization_params*>(
                            quantization_params.data())));
        } else {
          ASSERT_NE(workspace_size, SIZE_MAX);
          ASSERT_NE(workspace_alignment, SIZE_MAX);
          ASSERT_EQ(xnn_status_success,
                    xnn_setup_convolution2d_nhwc_qd8_f32_qc8w(
                        convolution_op2, workspace.data(), input.data(),
                        output2.data(),
                        reinterpret_cast<
                            const struct xnn_dynamic_quantization_params*>(
                            quantization_params.data())));
        }
        ASSERT_EQ(xnn_status_success,
                  xnn_run_operator(convolution_op2, auto_threadpool.get()));

        VerifyNHWCxF32(output2, output_ref, output_min, output_max);
        VerifyWeightsCache(*internal_weights_cache, old_weights_cache_size);
      }
    }
  }

  void TestNHWCxQS8() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    std::uniform_int_distribution<int32_t> w8dist(
      -std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) +
      batch_size() * ((input_height() * input_width() - 1) * input_channel_stride() + groups() * group_input_channels()));
    std::vector<int8_t> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<int32_t> bias(groups() * group_output_channels());
    std::vector<int8_t> output(batch_size() * ((output_height() * output_width() - 1) * output_channel_stride() + groups() * group_output_channels()));
    std::vector<int32_t> accumulators(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<double> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());

    const int8_t input_zero_point = -1;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      // Compute reference results, without renormalization.
      if (depthwise_layout()) {
        ASSERT_EQ(group_input_channels(), 1);
        xnnpack::compute_depthwise_convolution_qs8_reference_results(
          batch_size(),
          output_height(),
          output_width(),
          input_height(),
          input_width(),
          padding_top(),
          padding_right(),
          padding_bottom(),
          padding_left(),
          kernel_height(),
          kernel_width(),
          subsampling_height(),
          subsampling_width(),
          dilation_height(),
          dilation_width(),
          groups(),
          group_output_channels(),
          input_channel_stride(),
          input_zero_point,
          input,
          kernel,
          accumulators,
          has_bias(),
          bias);
      } else {
        xnnpack::compute_convolution_qs8_reference_results(
          batch_size(),
          output_height(),
          output_width(),
          input_height(),
          input_width(),
          padding_top(),
          padding_right(),
          padding_bottom(),
          padding_left(),
          kernel_height(),
          kernel_width(),
          subsampling_height(),
          subsampling_width(),
          dilation_height(),
          dilation_width(),
          groups(),
          group_input_channels(),
          group_output_channels(),
          input_channel_stride(),
          input_zero_point,
          input,
          kernel,
          accumulators,
          has_bias(),
          bias);
      }

      // Compute renormalization parameters.
      const int32_t accumulated_min = *std::min_element(accumulators.cbegin(), accumulators.cend());
      const int32_t accumulated_max = *std::max_element(accumulators.cbegin(), accumulators.cend());

      const double output_scale = double(uint32_t(accumulated_max - accumulated_min)) / 255.0;
      const int8_t output_zero_point = int8_t(std::max(std::min(
        lrint(-0.5 - 0.5 * double(accumulated_min + accumulated_max) / output_scale),
        long(std::numeric_limits<int8_t>::max())), long(std::numeric_limits<int8_t>::min())));

      // Renormalize reference results.
      std::transform(accumulators.cbegin(), accumulators.cend(), output_ref.begin(),
        [this, output_scale, output_zero_point](int32_t x) -> double {
          return std::max<double>(std::min<double>(double(x) / output_scale, double(qmax() - 0x80) - output_zero_point), double(qmin() - 0x80) - output_zero_point);
        });

      // Create, setup, run, and destroy Convolution operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convolution_op = nullptr;
      struct xnn_internal_weights_cache* internal_weights_cache = nullptr;
      std::unique_ptr<xnn_weights_cache_provider, decltype(&xnn_delete_weights_cache)> auto_weights_cache(
        nullptr, xnn_delete_weights_cache);
      if (use_weights_cache()) {
        xnn_weights_cache_t weights_cache = nullptr;
        xnn_create_weights_cache(&weights_cache);
        auto_weights_cache.reset(weights_cache);
        if (weights_cache) {
          internal_weights_cache = (struct xnn_internal_weights_cache*) weights_cache->context;
        }
      }

      uint32_t flags = 0;
      if (depthwise_layout()) {
        flags |= XNN_FLAG_DEPTHWISE_CONVOLUTION;
      }
      if (padding_tf_same()) {
        flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
      }
      if (transient_indirection_buffer()) {
        flags |= XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER;
      }
      xnn_status status = xnn_create_convolution2d_nhwc_qs8(
          padding_tf_same() ? 0 : padding_top(), padding_tf_same() ? 0 : padding_right(),
          padding_tf_same() ? 0 : padding_bottom(), padding_tf_same() ? 0 : padding_left(),
          kernel_height(), kernel_width(),
          subsampling_height(), subsampling_width(),
          dilation_height(), dilation_width(),
          groups(), group_input_channels(), group_output_channels(),
          input_channel_stride(), output_channel_stride(),
          input_zero_point, 1.0f /* input scale */, 1.0f /* kernel scale */,
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_zero_point, output_scale, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80),
          flags,
          /*code_cache=*/nullptr,
          auto_weights_cache.get(),
          &convolution_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, convolution_op);
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete convolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convolution_op(convolution_op, xnn_delete_operator);

      size_t workspace_size = SIZE_MAX;
      size_t workspace_alignment = SIZE_MAX;
      ASSERT_EQ(
        xnn_status_success, xnn_reshape_convolution2d_nhwc_qs8(
                              convolution_op, batch_size(), input_height(), input_width(),
                              &workspace_size, &workspace_alignment,
                              /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                              auto_threadpool.get()));
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
      std::iota(workspace.begin(), workspace.end(), 0);
      if (transient_indirection_buffer()) {
        ASSERT_NE(workspace_size, 0);
        ASSERT_NE(workspace_size, SIZE_MAX);
        ASSERT_EQ(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
        ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_qs8(convolution_op, workspace.data(), input.data(), output.data()));
      } else {
        ASSERT_NE(workspace_size, SIZE_MAX);
        ASSERT_NE(workspace_alignment, SIZE_MAX);
        ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_qs8(convolution_op, workspace.data(), input.data(), output.data()));
      }
      ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op, auto_threadpool.get()));

      VerifyNHWCxQS8(output, output_ref, output_zero_point);

      if (use_weights_cache()) {
        xnn_operator_t convolution_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;

        ASSERT_EQ(
            xnn_status_success,
            xnn_create_convolution2d_nhwc_qs8(
                padding_tf_same() ? 0 : padding_top(),
                padding_tf_same() ? 0 : padding_right(),
                padding_tf_same() ? 0 : padding_bottom(),
                padding_tf_same() ? 0 : padding_left(), kernel_height(),
                kernel_width(), subsampling_height(), subsampling_width(),
                dilation_height(), dilation_width(), groups(),
                group_input_channels(), group_output_channels(),
                input_channel_stride(), output_channel_stride(),
                input_zero_point, 1.0f /* input scale */,
                1.0f /* kernel scale */, kernel.data(),
                has_bias() ? bias.data() : nullptr, output_zero_point,
                output_scale, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80),
                flags,
                /*code_cache=*/nullptr, auto_weights_cache.get(),
                &convolution_op2));
        ASSERT_NE(nullptr, convolution_op2);

        // Smart pointer to automatically delete convolution_op.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
            auto_convolution_op(convolution_op2, xnn_delete_operator);

        std::vector<int8_t> output2(output.size(), INT8_C(0xA5));
        size_t workspace_size = SIZE_MAX;
        size_t workspace_alignment = SIZE_MAX;
        ASSERT_EQ(
          xnn_status_success,
          xnn_reshape_convolution2d_nhwc_qs8(
            convolution_op2, batch_size(), input_height(), input_width(),
            &workspace_size, &workspace_alignment,
            /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
            auto_threadpool.get()));
        std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
        std::iota(workspace.begin(), workspace.end(), 0);
        if (transient_indirection_buffer()) {
          ASSERT_NE(workspace_size, 0);
          ASSERT_NE(workspace_size, SIZE_MAX);
          ASSERT_EQ(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
          ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_qs8(convolution_op2, workspace.data(), input.data(), output2.data()));
        } else {
          ASSERT_NE(workspace_size, SIZE_MAX);
          ASSERT_NE(workspace_alignment, SIZE_MAX);
          ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_qs8(convolution_op2, workspace.data(), input.data(), output2.data()));
        }
        ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op2, auto_threadpool.get()));

        VerifyNHWCxQS8(output2, output_ref, output_zero_point);
        VerifyWeightsCache(*internal_weights_cache, old_weights_cache_size);
      }
    }
  }

  void VerifyNHWCxQS8(const std::vector<int8_t> &output,
                      const std::vector<double> &output_ref,
                      const int8_t output_zero_point) const {
    for (size_t i = 0; i < batch_size(); i++) {
      for (size_t y = 0; y < output_height(); y++) {
        for (size_t x = 0; x < output_width(); x++) {
          for (size_t g = 0; g < groups(); g++) {
            for (size_t c = 0; c < group_output_channels(); c++) {
              EXPECT_LE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c]), int32_t(qmax() - 0x80))
                << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              EXPECT_GE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c]), int32_t(qmin() - 0x80))
                << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              EXPECT_NEAR(
                  output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c],
                  double(output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c]) - double(output_zero_point),
                  0.9)
                << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
            }
          }
        }
      }
    }
  }

  void TestNHWCxQU8() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) +
      batch_size() * ((input_height() * input_width() - 1) * input_channel_stride() + groups() * group_input_channels()));
    std::vector<uint8_t> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<int32_t> bias(groups() * group_output_channels());
    std::vector<uint8_t> output(batch_size() * ((output_height() * output_width() - 1) * output_channel_stride() + groups() * group_output_channels()));
    std::vector<int32_t> accumulators(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<double> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());

    const uint8_t input_zero_point = 127;
    const uint8_t kernel_zero_point = 127;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return u8dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
      std::fill(output.begin(), output.end(), UINT8_C(0xA5));

      // Compute reference results, without renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  accumulators[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(accumulators.begin(), accumulators.end(), 0);
      }
      if (depthwise_layout()) {
        ASSERT_EQ(group_input_channels(), 1);
        xnnpack::compute_depthwise_convolution_qu8_reference_results(
            batch_size(),
            output_height(),
            output_width(),
            input_height(),
            input_width(),
            padding_top(),
            padding_right(),
            padding_bottom(),
            padding_left(),
            kernel_height(),
            kernel_width(),
            subsampling_height(),
            subsampling_width(),
            dilation_height(),
            dilation_width(),
            groups(),
            group_output_channels(),
            input_channel_stride(),
            input_zero_point,
            kernel_zero_point,
            input,
            kernel,
            accumulators,
            has_bias(),
            bias);
      } else {
        xnnpack::compute_convolution_qu8_reference_results(
            batch_size(),
            output_height(),
            output_width(),
            input_height(),
            input_width(),
            padding_top(),
            padding_right(),
            padding_bottom(),
            padding_left(),
            kernel_height(),
            kernel_width(),
            subsampling_height(),
            subsampling_width(),
            dilation_height(),
            dilation_width(),
            groups(),
            group_input_channels(),
            group_output_channels(),
            input_channel_stride(),
            input_zero_point,
            kernel_zero_point,
            input,
            kernel,
            accumulators,
            has_bias(),
            bias);
      }

      // Compute renormalization parameters.
      const int32_t accumulated_min = *std::min_element(accumulators.cbegin(), accumulators.cend());
      const int32_t accumulated_max = *std::max_element(accumulators.cbegin(), accumulators.cend());

      const double output_scale = double(uint32_t(accumulated_max - accumulated_min)) / 255.0;
      const uint8_t output_zero_point = uint8_t(std::max(std::min(
        lrint(127.5 - 0.5 * double(accumulated_min + accumulated_max) / output_scale),
        long(std::numeric_limits<uint8_t>::max())), long(std::numeric_limits<uint8_t>::min())));

      // Renormalize reference results.
      std::transform(accumulators.cbegin(), accumulators.cend(), output_ref.begin(),
        [this, output_scale, output_zero_point](int32_t x) -> double {
          return std::max<double>(std::min<double>(double(x) / output_scale, double(qmax()) - output_zero_point), double(qmin()) - output_zero_point);
        });

      // Create, setup, run, and destroy Convolution operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convolution_op = nullptr;
      struct xnn_internal_weights_cache* internal_weights_cache = nullptr;
      std::unique_ptr<xnn_weights_cache_provider, decltype(&xnn_delete_weights_cache)> auto_weights_cache(
        nullptr, xnn_delete_weights_cache);
      if (use_weights_cache()) {
        xnn_weights_cache_t weights_cache = nullptr;
        xnn_create_weights_cache(&weights_cache);
        auto_weights_cache.reset(weights_cache);
        if (weights_cache) {
          internal_weights_cache = (struct xnn_internal_weights_cache*) weights_cache->context;
        }
      }

      uint32_t flags = 0;
      if (depthwise_layout()) {
        flags |= XNN_FLAG_DEPTHWISE_CONVOLUTION;
      }
      if (padding_tf_same()) {
        flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
      }
      if (transient_indirection_buffer()) {
        flags |= XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER;
      }
      xnn_status status = xnn_create_convolution2d_nhwc_qu8(
          padding_tf_same() ? 0 : padding_top(), padding_tf_same() ? 0 : padding_right(),
          padding_tf_same() ? 0 : padding_bottom(), padding_tf_same() ? 0 : padding_left(),
          kernel_height(), kernel_width(),
          subsampling_height(), subsampling_width(),
          dilation_height(), dilation_width(),
          groups(), group_input_channels(), group_output_channels(),
          input_channel_stride(), output_channel_stride(),
          input_zero_point, 1.0f /* input scale */,
          kernel_zero_point, 1.0f /* kernel scale */,
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_zero_point, output_scale, qmin(), qmax(),
          flags,
          /*code_cache=*/nullptr,
          auto_weights_cache.get(),
          &convolution_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, convolution_op);
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete convolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convolution_op(convolution_op, xnn_delete_operator);

      size_t workspace_size = SIZE_MAX;
      size_t workspace_alignment = SIZE_MAX;
      ASSERT_EQ(
        xnn_status_success, xnn_reshape_convolution2d_nhwc_qu8(
                              convolution_op, batch_size(), input_height(), input_width(),
                              &workspace_size, &workspace_alignment,
                              /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                              auto_threadpool.get()));
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
      std::iota(workspace.begin(), workspace.end(), 0);
      if (transient_indirection_buffer()) {
        ASSERT_NE(workspace_size, 0);
        ASSERT_NE(workspace_size, SIZE_MAX);
        ASSERT_EQ(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
        ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_qu8(convolution_op, workspace.data(), input.data(), output.data()));
      } else {
        ASSERT_NE(workspace_size, SIZE_MAX);
        ASSERT_NE(workspace_alignment, SIZE_MAX);
        ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_qu8(convolution_op, workspace.data(), input.data(), output.data()));
      }
      ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op, auto_threadpool.get()));

      // Verify results.
      VerifyNHWCxQU8(output, output_ref, output_zero_point);

      if (use_weights_cache()) {
        xnn_operator_t convolution_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;

        ASSERT_EQ(
            xnn_status_success,
            xnn_create_convolution2d_nhwc_qu8(
                padding_tf_same() ? 0 : padding_top(),
                padding_tf_same() ? 0 : padding_right(),
                padding_tf_same() ? 0 : padding_bottom(),
                padding_tf_same() ? 0 : padding_left(), kernel_height(),
                kernel_width(), subsampling_height(), subsampling_width(),
                dilation_height(), dilation_width(), groups(),
                group_input_channels(), group_output_channels(),
                input_channel_stride(), output_channel_stride(),
                input_zero_point, 1.0f /* input scale */, kernel_zero_point,
                1.0f /* kernel scale */, kernel.data(),
                has_bias() ? bias.data() : nullptr, output_zero_point,
                output_scale, qmin(), qmax(),
                flags,
                /*code_cache=*/nullptr, auto_weights_cache.get(),
                &convolution_op2));
        ASSERT_NE(nullptr, convolution_op2);

        // Smart pointer to automatically delete convolution_op2.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
            auto_convolution_op2(convolution_op2, xnn_delete_operator);
        std::vector<uint8_t> output2(output.size(), UINT8_C(0xA5));

        size_t workspace_size = SIZE_MAX;
        size_t workspace_alignment = SIZE_MAX;
        ASSERT_EQ(
          xnn_status_success,
          xnn_reshape_convolution2d_nhwc_qu8(
            convolution_op2, batch_size(), input_height(), input_width(),
            &workspace_size, &workspace_alignment,
            /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
            auto_threadpool.get()));
        std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
        std::iota(workspace.begin(), workspace.end(), 0);
        if (transient_indirection_buffer()) {
          ASSERT_NE(workspace_size, 0);
          ASSERT_NE(workspace_size, SIZE_MAX);
          ASSERT_EQ(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
          ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_qu8(convolution_op2, workspace.data(), input.data(), output2.data()));
        } else {
          ASSERT_NE(workspace_size, SIZE_MAX);
          ASSERT_NE(workspace_alignment, SIZE_MAX);
          ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_qu8(convolution_op2, workspace.data(), input.data(), output2.data()));
        }
        ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op2, auto_threadpool.get()));

        // Verify results.
        VerifyNHWCxQU8(output2, output_ref, output_zero_point);
        VerifyWeightsCache(*internal_weights_cache, old_weights_cache_size);
      }
    }
  }

  void VerifyNHWCxQU8(const std::vector<uint8_t> &output,
                      const std::vector<double> &output_ref,
                      const uint8_t output_zero_point) const {
    for (size_t i = 0; i < batch_size(); i++) {
      for (size_t y = 0; y < output_height(); y++) {
        for (size_t x = 0; x < output_width(); x++) {
          for (size_t g = 0; g < groups(); g++) {
            for (size_t c = 0; c < group_output_channels(); c++) {
              EXPECT_LE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c]), int32_t(qmax()))
                << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              EXPECT_GE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c]), int32_t(qmin()))
                << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              EXPECT_NEAR(
                  output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c],
                  double(output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c]) - double(output_zero_point),
                  0.9)
                << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
            }
          }
        }
      }
    }
  }

  void TestNHWCxF32() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      batch_size() * ((input_height() * input_width() - 1) * input_channel_stride() + groups() * group_input_channels()));
    std::vector<float> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<float> bias(groups() * group_output_channels());
    std::vector<float> output(batch_size() * ((output_height() * output_width() - 1) * output_channel_stride() + groups() * group_output_channels()));
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), nanf(""));

      // Compute reference results, without clamping.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      }
      if (depthwise_layout()) {
        ASSERT_EQ(group_input_channels(), 1);

        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t ky = 0; ky < kernel_height(); ky++) {
                const size_t iy = oy * subsampling_height() + ky * dilation_height() - padding_top();
                if (iy < input_height()) {
                  for (size_t kx = 0; kx < kernel_width(); kx++) {
                    const size_t ix = ox * subsampling_width() + kx * dilation_width() - padding_left();
                    if (ix < input_width()) {
                      for (size_t g = 0; g < groups(); g++) {
                        for (size_t oc = 0; oc < group_output_channels(); oc++) {
                          output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            input[((i * input_height() + iy) * input_width() + ix) * input_channel_stride() + g] *
                            kernel[((ky * kernel_width() + kx) * groups() + g) * group_output_channels() + oc];
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t ky = 0; ky < kernel_height(); ky++) {
                const size_t iy = oy * subsampling_height() + ky * dilation_height() - padding_top();
                if (iy < input_height()) {
                  for (size_t kx = 0; kx < kernel_width(); kx++) {
                    const size_t ix = ox * subsampling_width() + kx * dilation_width() - padding_left();
                    if (ix < input_width()) {
                      for (size_t g = 0; g < groups(); g++) {
                        for (size_t oc = 0; oc < group_output_channels(); oc++) {
                          for (size_t ic = 0; ic < group_input_channels(); ic++) {
                            output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                              input[((i * input_height() + iy) * input_width() + ix) * input_channel_stride() + g * group_input_channels() + ic] *
                              kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic];
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());

      const float output_min = accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      const float output_max = accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, run, and destroy Convolution operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convolution_op = nullptr;

      std::unique_ptr<xnn_code_cache, decltype(&xnn_release_code_cache)> auto_code_cache(
          nullptr, xnn_release_code_cache);

      #if XNN_PLATFORM_JIT
        xnn_code_cache code_cache;
        if (use_jit()) {
          xnn_init_code_cache(&code_cache);
          auto_code_cache.reset(&code_cache);
        }
      #endif
      struct xnn_internal_weights_cache* internal_weights_cache = nullptr;
      std::unique_ptr<xnn_weights_cache_provider, decltype(&xnn_delete_weights_cache)> auto_weights_cache(
        nullptr, xnn_delete_weights_cache);
      if (use_weights_cache()) {
        xnn_weights_cache_t weights_cache = nullptr;
        xnn_create_weights_cache(&weights_cache);
        auto_weights_cache.reset(weights_cache);
        if (weights_cache) {
          internal_weights_cache = (struct xnn_internal_weights_cache*) weights_cache->context;
        }
      }

      uint32_t flags = 0;
      if (depthwise_layout()) {
        flags |= XNN_FLAG_DEPTHWISE_CONVOLUTION;
      }
      if (padding_tf_same()) {
        flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
      }
      if (transient_indirection_buffer()) {
        flags |= XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER;
      }
      xnn_status status = xnn_create_convolution2d_nhwc_f32(
          padding_tf_same() ? 0 : padding_top(), padding_tf_same() ? 0 : padding_right(),
          padding_tf_same() ? 0 : padding_bottom(), padding_tf_same() ? 0 : padding_left(),
          kernel_height(), kernel_width(),
          subsampling_height(), subsampling_width(),
          dilation_height(), dilation_width(),
          groups(), group_input_channels(), group_output_channels(),
          input_channel_stride(), output_channel_stride(),
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_min, output_max,
          flags,
          auto_code_cache.get(),
          auto_weights_cache.get(),
          &convolution_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, convolution_op);
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete convolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convolution_op(convolution_op, xnn_delete_operator);

      #if XNN_PLATFORM_JIT
        if (use_jit()) {
          // Check that we actually generated code.
          ASSERT_GT(code_cache.cache.code.size, 0);
          xnn_finalize_code_memory(&code_cache.cache.code);
        }
      #endif

        size_t workspace_size = SIZE_MAX;
        size_t workspace_alignment = SIZE_MAX;
        ASSERT_EQ(
          xnn_status_success,
          xnn_reshape_convolution2d_nhwc_f32(
            convolution_op, batch_size(), input_height(), input_width(),
            &workspace_size, &workspace_alignment,
            /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
            auto_threadpool.get()));
        std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
        std::iota(workspace.begin(), workspace.end(), 0);
        if (transient_indirection_buffer()) {
          ASSERT_NE(workspace_size, 0);
          ASSERT_NE(workspace_size, SIZE_MAX);
          ASSERT_EQ(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
          ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_f32(convolution_op, workspace.data(), input.data(), output.data()));
        } else {
          ASSERT_NE(workspace_size, SIZE_MAX);
          ASSERT_NE(workspace_alignment, SIZE_MAX);
          ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_f32(convolution_op, workspace.data(), input.data(), output.data()));
        }
        ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op, auto_threadpool.get()));

        VerifyNHWCxF32(output, output_ref, output_min, output_max);

        if (use_weights_cache()) {
          // We already finalized the code cache, so create a new code cache if we are testing JIT.
          std::unique_ptr<xnn_code_cache, decltype(&xnn_release_code_cache)> auto_inner_code_cache(
            nullptr, xnn_release_code_cache);
        #if XNN_PLATFORM_JIT
          xnn_code_cache inner_code_cache;
          if (use_jit()) {
            xnn_init_code_cache(&inner_code_cache);
            auto_inner_code_cache.reset(&inner_code_cache);
          }
        #endif
        // To test weights cache, we create the operator with the same parameters, and setup with a different output.
        xnn_operator_t convolution_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;

        ASSERT_EQ(xnn_status_success, xnn_create_convolution2d_nhwc_f32(
            padding_tf_same() ? 0 : padding_top(), padding_tf_same() ? 0 : padding_right(),
            padding_tf_same() ? 0 : padding_bottom(), padding_tf_same() ? 0 : padding_left(),
            kernel_height(), kernel_width(),
            subsampling_height(), subsampling_width(),
            dilation_height(), dilation_width(),
            groups(), group_input_channels(), group_output_channels(),
            input_channel_stride(), output_channel_stride(),
            kernel.data(), has_bias() ? bias.data() : nullptr,
            output_min, output_max,
            flags,
            auto_inner_code_cache.get(), auto_weights_cache.get(),
            &convolution_op2));

        ASSERT_NE(nullptr, convolution_op2);

        #if XNN_PLATFORM_JIT
          if (use_jit()) {
            // Check that we actually generated code.
            ASSERT_GT(inner_code_cache.cache.code.size, 0);
            xnn_finalize_code_memory(&inner_code_cache.cache.code);
          }
        #endif

        std::vector<float> output2(output.size(), nanf(""));
        size_t workspace_size = SIZE_MAX;
        size_t workspace_alignment = SIZE_MAX;
        ASSERT_EQ(
          xnn_status_success,
          xnn_reshape_convolution2d_nhwc_f32(
            convolution_op2, batch_size(), input_height(), input_width(),
            &workspace_size, &workspace_alignment,
            /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
            auto_threadpool.get()));
        std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
        std::iota(workspace.begin(), workspace.end(), 0);
        if (transient_indirection_buffer()) {
          ASSERT_NE(workspace_size, 0);
          ASSERT_NE(workspace_size, SIZE_MAX);
          ASSERT_EQ(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
          ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_f32(convolution_op2, workspace.data(), input.data(), output2.data()));
        } else {
          ASSERT_NE(workspace_size, SIZE_MAX);
          ASSERT_NE(workspace_alignment, SIZE_MAX);
          ASSERT_EQ(workspace_alignment, 1);
          ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_f32(convolution_op2, workspace.data(), input.data(), output2.data()));
        }
        ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op2, auto_threadpool.get()));

        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convolution_op2(convolution_op2, xnn_delete_operator);
        VerifyWeightsCache(*internal_weights_cache, old_weights_cache_size);

        VerifyNHWCxF32(output2, output_ref, output_min, output_max);
      }
    }
  }

  void VerifyNHWCxF32(const std::vector<float>& output, const std::vector<float>& output_ref, const float output_min, const float output_max) const {
    for (size_t i = 0; i < batch_size(); i++) {
      for (size_t y = 0; y < output_height(); y++) {
        for (size_t x = 0; x < output_width(); x++) {
          for (size_t g = 0; g < groups(); g++) {
            for (size_t c = 0; c < group_output_channels(); c++) {
              const float tolerance = std::max(1.0e-4f, 1.0e-4f * std::abs(output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c]));
              EXPECT_GE(output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c], output_min)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              EXPECT_LE(output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c], output_max)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              EXPECT_NEAR(
                  output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c],
                  output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c],
                  tolerance) << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
            }
          }
        }
      }
    }
  }

  void TestNHWCxF16() const {
    switch (weights_type()) {
      case WeightsType::Default:
        break;
      case WeightsType::FP32:
        break;
      default:
        GTEST_FAIL() << "unexpected weights type";
    }

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);

    std::vector<uint16_t> input(XNN_EXTRA_BYTES / sizeof(uint16_t) +
      batch_size() * ((input_height() * input_width() - 1) * input_channel_stride() + groups() * group_input_channels()));
    std::vector<uint16_t> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<float> kernel_as_float(kernel.size());
    std::vector<uint16_t> bias(groups() * group_output_channels());
    std::vector<float> bias_as_float(bias.size());
    std::vector<uint16_t> output(batch_size() * ((output_height() * output_width() - 1) * output_channel_stride() + groups() * group_output_channels()));
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::transform(kernel.cbegin(), kernel.cend(), kernel_as_float.begin(), fp16_ieee_to_fp32_value);
      std::generate(bias.begin(), bias.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::transform(bias.cbegin(), bias.cend(), bias_as_float.begin(), fp16_ieee_to_fp32_value);
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);


      // Compute reference results, without clamping.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    fp16_ieee_to_fp32_value(bias[g * group_output_channels() + oc]);
                }
              }
            }
          }
        }
      } else {
        std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      }
      if (depthwise_layout()) {
        ASSERT_EQ(group_input_channels(), 1);

        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t ky = 0; ky < kernel_height(); ky++) {
                const size_t iy = oy * subsampling_height() + ky * dilation_height() - padding_top();
                if (iy < input_height()) {
                  for (size_t kx = 0; kx < kernel_width(); kx++) {
                    const size_t ix = ox * subsampling_width() + kx * dilation_width() - padding_left();
                    if (ix < input_width()) {
                      for (size_t g = 0; g < groups(); g++) {
                        for (size_t oc = 0; oc < group_output_channels(); oc++) {
                          output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            fp16_ieee_to_fp32_value(input[((i * input_height() + iy) * input_width() + ix) * input_channel_stride() + g]) *
                            fp16_ieee_to_fp32_value(kernel[((ky * kernel_width() + kx) * groups() + g) * group_output_channels() + oc]);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t ky = 0; ky < kernel_height(); ky++) {
                const size_t iy = oy * subsampling_height() + ky * dilation_height() - padding_top();
                if (iy < input_height()) {
                  for (size_t kx = 0; kx < kernel_width(); kx++) {
                    const size_t ix = ox * subsampling_width() + kx * dilation_width() - padding_left();
                    if (ix < input_width()) {
                      for (size_t g = 0; g < groups(); g++) {
                        for (size_t oc = 0; oc < group_output_channels(); oc++) {
                          for (size_t ic = 0; ic < group_input_channels(); ic++) {
                            output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                              fp16_ieee_to_fp32_value(input[((i * input_height() + iy) * input_width() + ix) * input_channel_stride() + g * group_input_channels() + ic]) *
                              fp16_ieee_to_fp32_value(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]);
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float scaled_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_min + accumulated_range / 255.0f * float(qmin())));
      const float scaled_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_max - accumulated_range / 255.0f * float(255 - qmax())));
      const float output_min = scaled_min == scaled_max ? -std::numeric_limits<float>::infinity() : scaled_min;
      const float output_max = scaled_min == scaled_max ? +std::numeric_limits<float>::infinity() : scaled_max;

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, run, and destroy Convolution operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convolution_op = nullptr;
      std::unique_ptr<xnn_code_cache, decltype(&xnn_release_code_cache)> auto_code_cache(
          nullptr, xnn_release_code_cache);
      #if XNN_PLATFORM_JIT
        xnn_code_cache code_cache;
        if (use_jit()) {
          xnn_init_code_cache(&code_cache);
          auto_code_cache.reset(&code_cache);
        }
      #endif
      struct xnn_internal_weights_cache* internal_weights_cache = nullptr;
      std::unique_ptr<xnn_weights_cache_provider, decltype(&xnn_delete_weights_cache)> auto_weights_cache(
        nullptr, xnn_delete_weights_cache);
      if (use_weights_cache()) {
        xnn_weights_cache_t weights_cache = nullptr;
        xnn_create_weights_cache(&weights_cache);
        auto_weights_cache.reset(weights_cache);
        if (weights_cache) {
          internal_weights_cache = (struct xnn_internal_weights_cache*) weights_cache->context;
        }
      }

      const void* kernel_data = kernel.data();
      const void* bias_data = bias.data();
      if (weights_type() == WeightsType::FP32) {
        kernel_data = kernel_as_float.data();
        bias_data = bias_as_float.data();
      }
      uint32_t flags = 0;
      if (depthwise_layout()) {
        flags |= XNN_FLAG_DEPTHWISE_CONVOLUTION;
      }
      if (padding_tf_same()) {
        flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
      }
      if (transient_indirection_buffer()) {
        flags |= XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER;
      }
      if (weights_type() == WeightsType::FP32) {
        flags |= XNN_FLAG_FP32_STATIC_WEIGHTS;
      }
      xnn_status status = xnn_create_convolution2d_nhwc_f16(
          padding_tf_same() ? 0 : padding_top(), padding_tf_same() ? 0 : padding_right(),
          padding_tf_same() ? 0 : padding_bottom(), padding_tf_same() ? 0 : padding_left(),
          kernel_height(), kernel_width(),
          subsampling_height(), subsampling_width(),
          dilation_height(), dilation_width(),
          groups(), group_input_channels(), group_output_channels(),
          input_channel_stride(), output_channel_stride(),
          kernel_data, has_bias() ? bias_data : nullptr,
          output_min, output_max,
          flags,
          auto_code_cache.get(),
          auto_weights_cache.get(),
          &convolution_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, convolution_op);
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete convolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convolution_op(convolution_op, xnn_delete_operator);

      #if XNN_PLATFORM_JIT
        if (use_jit()) {
          // Check that we actually generated code.
          ASSERT_GT(code_cache.cache.code.size, 0);
          xnn_finalize_code_memory(&code_cache.cache.code);
        }
      #endif

        size_t workspace_size = SIZE_MAX;
        size_t workspace_alignment = SIZE_MAX;
        ASSERT_EQ(
          xnn_status_success,
          xnn_reshape_convolution2d_nhwc_f16(
            convolution_op, batch_size(), input_height(), input_width(),
            &workspace_size, &workspace_alignment,
            /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
            auto_threadpool.get()));
        std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
        std::iota(workspace.begin(), workspace.end(), 0);
        if (transient_indirection_buffer()) {
          ASSERT_NE(workspace_size, 0);
          ASSERT_NE(workspace_size, SIZE_MAX);
          ASSERT_EQ(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
          ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_f16(convolution_op, workspace.data(), input.data(), output.data()));
        } else {
          ASSERT_NE(workspace_size, SIZE_MAX);
          ASSERT_NE(workspace_alignment, SIZE_MAX);
          ASSERT_EQ(workspace_alignment, 1);
          ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_f16(convolution_op, workspace.data(), input.data(), output.data()));
        }
        ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op, auto_threadpool.get()));

        VerifyNHWCxF16(output, output_ref, output_min, output_max, batch_size(), output_height(), output_width());

        if (use_weights_cache()) {
          // We already finalized the code cache, so create a new code cache if we are testing JIT.
          std::unique_ptr<xnn_code_cache, decltype(&xnn_release_code_cache)> auto_inner_code_cache(
            nullptr, xnn_release_code_cache);
        #if XNN_PLATFORM_JIT
          xnn_code_cache inner_code_cache;
          if (use_jit()) {
            xnn_init_code_cache(&inner_code_cache);
            auto_inner_code_cache.reset(&inner_code_cache);
          }
        #endif
        xnn_operator_t convolution_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;
        ASSERT_EQ(xnn_status_success, xnn_create_convolution2d_nhwc_f16(
            padding_tf_same() ? 0 : padding_top(), padding_tf_same() ? 0 : padding_right(),
            padding_tf_same() ? 0 : padding_bottom(), padding_tf_same() ? 0 : padding_left(),
            kernel_height(), kernel_width(),
            subsampling_height(), subsampling_width(),
            dilation_height(), dilation_width(),
            groups(), group_input_channels(), group_output_channels(),
            input_channel_stride(), output_channel_stride(),
            kernel_data, has_bias() ? bias_data : nullptr,
            output_min, output_max,
            flags,
            auto_inner_code_cache.get(), auto_weights_cache.get(),
            &convolution_op2));
        ASSERT_NE(nullptr, convolution_op2);

        // Smart pointer to automatically delete convolution_op.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convolution_op(convolution_op2, xnn_delete_operator);

        #if XNN_PLATFORM_JIT
          if (use_jit()) {
            // Check that we actually generated code.
            ASSERT_GT(inner_code_cache.cache.code.size, 0);
            xnn_finalize_code_memory(&inner_code_cache.cache.code);
          }
        #endif

        std::vector<uint16_t> output2(output.size(), UINT16_C(0x7E00) /* NaN */);
        size_t workspace_size = SIZE_MAX;
        size_t workspace_alignment = SIZE_MAX;
        ASSERT_EQ(
          xnn_status_success,
          xnn_reshape_convolution2d_nhwc_f16(
            convolution_op2, batch_size(), input_height(), input_width(),
            &workspace_size, &workspace_alignment,
            /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
            auto_threadpool.get()));
        std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
        std::iota(workspace.begin(), workspace.end(), 0);
        if (transient_indirection_buffer()) {
          ASSERT_NE(workspace_size, 0);
          ASSERT_NE(workspace_size, SIZE_MAX);
          ASSERT_EQ(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
          ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_f16(convolution_op2, workspace.data(), input.data(), output2.data()));
        } else {
          ASSERT_NE(workspace_size, SIZE_MAX);
          ASSERT_NE(workspace_alignment, SIZE_MAX);
          ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_f16(convolution_op2, workspace.data(), input.data(), output2.data()));
        }
        ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op2, auto_threadpool.get()));

        VerifyNHWCxF16(output, output_ref, output_min, output_max, batch_size(), output_height(), output_width());
        VerifyWeightsCache(*internal_weights_cache, old_weights_cache_size);
      }
    }
  }

  void VerifyNHWCxF16(const std::vector<uint16_t> &output,
                      const std::vector<float> &output_ref,
                      const float output_min, const float output_max,
                      size_t bs, size_t oh, size_t ow) const {
    for (size_t i = 0; i < bs; i++) {
      for (size_t y = 0; y < oh; y++) {
        for (size_t x = 0; x < ow; x++) {
          for (size_t g = 0; g < groups(); g++) {
            for (size_t c = 0; c < group_output_channels(); c++) {
              // FP16 saturates, it's the nature of the beast. If both reference and
              // actual are infinity, then consider the output to be correct.
              const bool reference_infinity = std::isinf(output_ref[(((i * oh + y) * ow + x) * groups() + g) * group_output_channels() + c]);
              const bool actual_infinity = std::isinf(fp16_ieee_to_fp32_value(output[((i * oh + y) * ow + x) * output_channel_stride() + g * group_output_channels() + c]));
              const float tolerance = std::max(1.0e-4f, std::abs(output_ref[(((i * oh + y) * ow + x) * groups() + g) * group_output_channels() + c]) * 2.0e-2f);
              if (reference_infinity && actual_infinity) {
                continue;
              }
              EXPECT_GE(fp16_ieee_to_fp32_value(output[((i * oh + y) * ow + x) * output_channel_stride() + g * group_output_channels() + c]), output_min)
                << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              EXPECT_LE(fp16_ieee_to_fp32_value(output[((i * oh + y) * ow + x) * output_channel_stride() + g * group_output_channels() + c]), output_max)
                << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              EXPECT_NEAR(output_ref[(((i * oh + y) * ow + x) * groups() + g) * group_output_channels() + c], fp16_ieee_to_fp32_value(output[((i * oh + y) * ow + x) * output_channel_stride() + g * group_output_channels() + c]), tolerance)
               << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
            }
          }
        }
      }
    }
  }

  void TestNCHWxF32() {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);
    std::uniform_real_distribution<float> pdist;

    std::vector<float> input(2 * XNN_EXTRA_BYTES / sizeof(float) +
      ((batch_size() - 1) * input_channel_stride() + groups() * group_input_channels()) * input_height() * input_width());
    std::vector<float> kernel(
      groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<float> bias(groups() * group_output_channels());
    std::vector<float> output(
      ((batch_size() - 1) * output_channel_stride() + groups() * group_output_channels()) * output_height() * output_width());
    std::vector<float> output_ref(batch_size() * groups() * group_output_channels() * output_height() * output_width());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
      for (float& k : kernel) {
        if (pdist(rng) <= sparsity()) {
          k = 0.0f;
        }
      }
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), nanf(""));

      // Compute reference results, without clamping.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  output_ref[(((i * groups() + g) * group_output_channels() + oc) * output_height() + oy) * output_width() + ox] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      }
      if (force_nhwc_input()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t ky = 0; ky < kernel_height(); ky++) {
                const size_t iy = oy * subsampling_height() + ky * dilation_height() - padding_top();
                if (iy < input_height()) {
                  for (size_t kx = 0; kx < kernel_width(); kx++) {
                    const size_t ix = ox * subsampling_width() + kx * dilation_width() - padding_left();
                    if (ix < input_width()) {
                      for (size_t g = 0; g < groups(); g++) {
                        for (size_t oc = 0; oc < group_output_channels(); oc++) {
                          for (size_t ic = 0; ic < group_input_channels(); ic++) {
                            output_ref[(((i * groups() + g) * group_output_channels() + oc) * output_height() + oy) * output_width() + ox] +=
                              input[((((i * input_height() + iy) * input_width() + ix) * groups() + g) * group_input_channels() + ic)] *
                              kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic];
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else if (depthwise_layout()) {
        ASSERT_EQ(group_input_channels(), 1);

        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t ky = 0; ky < kernel_height(); ky++) {
                const size_t iy = oy * subsampling_height() + ky * dilation_height() - padding_top();
                if (iy < input_height()) {
                  for (size_t kx = 0; kx < kernel_width(); kx++) {
                    const size_t ix = ox * subsampling_width() + kx * dilation_width() - padding_left();
                    if (ix < input_width()) {
                      for (size_t g = 0; g < groups(); g++) {
                        for (size_t oc = 0; oc < group_output_channels(); oc++) {
                          output_ref[(((i * groups() + g) * group_output_channels() + oc) * output_height() + oy) * output_width() + ox] +=
                            input[((i * input_channel_stride() + g) * input_height() + iy) * input_width() + ix] *
                            kernel[((ky * kernel_width() + kx) * groups() + g) * group_output_channels() + oc];
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t ky = 0; ky < kernel_height(); ky++) {
                const size_t iy = oy * subsampling_height() + ky * dilation_height() - padding_top();
                if (iy < input_height()) {
                  for (size_t kx = 0; kx < kernel_width(); kx++) {
                    const size_t ix = ox * subsampling_width() + kx * dilation_width() - padding_left();
                    if (ix < input_width()) {
                      for (size_t g = 0; g < groups(); g++) {
                        for (size_t oc = 0; oc < group_output_channels(); oc++) {
                          for (size_t ic = 0; ic < group_input_channels(); ic++) {
                            output_ref[(((i * groups() + g) * group_output_channels() + oc) * output_height() + oy) * output_width() + ox] +=
                              input[((i * input_channel_stride() + g * group_input_channels() + ic) * input_height() + iy) * input_width() + ix] *
                              kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic];
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());

      const float output_min = qmin() == 0 ? -std::numeric_limits<float>::infinity() :
        accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      const float output_max = qmax() == 255 ? std::numeric_limits<float>::infinity() :
        accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, run, and destroy Convolution operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convolution_op = nullptr;
      struct xnn_internal_weights_cache* internal_weights_cache = nullptr;
      std::unique_ptr<xnn_weights_cache_provider, decltype(&xnn_delete_weights_cache)> auto_weights_cache(
        nullptr, xnn_delete_weights_cache);
      if (use_weights_cache()) {
        xnn_weights_cache_t weights_cache = nullptr;
        xnn_create_weights_cache(&weights_cache);
        auto_weights_cache.reset(weights_cache);
        if (weights_cache) {
          internal_weights_cache = (struct xnn_internal_weights_cache*) weights_cache->context;
        }
      }

      xnn_status status = xnn_create_convolution2d_nchw_f32(
          padding_top(), padding_right(), padding_bottom(), padding_left(),
          kernel_height(), kernel_width(),
          subsampling_height(), subsampling_width(),
          dilation_height(), dilation_width(),
          groups(), group_input_channels(), group_output_channels(),
          input_channel_stride(), output_channel_stride(),
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_min, output_max,
          (depthwise_layout() ? XNN_FLAG_DEPTHWISE_CONVOLUTION : 0) | (force_nhwc_input() ? XNN_FLAG_INPUT_NHWC : 0),
          nullptr, auto_weights_cache.get(),
          &convolution_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, convolution_op);
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete convolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convolution_op(convolution_op, xnn_delete_operator);

      ASSERT_EQ(
        xnn_status_success, xnn_reshape_convolution2d_nchw_f32(
                              convolution_op, batch_size(), input_height(), input_width(),
                              /*output_height_out=*/nullptr, /*output_width_out=*/nullptr, auto_threadpool.get()));
      ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nchw_f32(convolution_op, input.data(), output.data()));
      ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op, auto_threadpool.get()));

      VerifyNCHWxF32(output, output_ref, output_min, output_max);

      if (use_weights_cache()) {
        xnn_operator_t convolution_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;
        ASSERT_EQ(
            xnn_status_success,
            xnn_create_convolution2d_nchw_f32(
                padding_top(), padding_right(), padding_bottom(),
                padding_left(), kernel_height(), kernel_width(),
                subsampling_height(), subsampling_width(), dilation_height(),
                dilation_width(), groups(), group_input_channels(),
                group_output_channels(), input_channel_stride(),
                output_channel_stride(), kernel.data(),
                has_bias() ? bias.data() : nullptr, output_min, output_max,
                (depthwise_layout() ? XNN_FLAG_DEPTHWISE_CONVOLUTION : 0) |
                    (force_nhwc_input() ? XNN_FLAG_INPUT_NHWC : 0),
                /*code_cache=*/nullptr, auto_weights_cache.get(),
                &convolution_op2));
        ASSERT_NE(nullptr, convolution_op2);

        // Smart pointer to automatically delete convolution_op2.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convolution_op(convolution_op2, xnn_delete_operator);
        std::vector<float> output2(output.size(), nanf(""));

        ASSERT_EQ(
          xnn_status_success,
          xnn_reshape_convolution2d_nchw_f32(
            convolution_op2, batch_size(), input_height(), input_width(),
            /*output_height_out=*/nullptr, /*output_width_out=*/nullptr, auto_threadpool.get()));
        ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nchw_f32(convolution_op2, input.data(), output2.data()));
        ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op2, auto_threadpool.get()));

        VerifyNCHWxF32(output2, output_ref, output_min, output_max);
        if (IsSpmm()) {
          VerifyWeightsCacheUnused(*internal_weights_cache);
        } else {
          VerifyWeightsCache(*internal_weights_cache, old_weights_cache_size);
        }
      }
    }
  }

  void VerifyNCHWxF32(const std::vector<float> &output,
                      const std::vector<float> &output_ref,
                      const float output_min, const float output_max) const {
    for (size_t i = 0; i < batch_size(); i++) {
      for (size_t y = 0; y < output_height(); y++) {
        for (size_t x = 0; x < output_width(); x++) {
          for (size_t g = 0; g < groups(); g++) {
            for (size_t c = 0; c < group_output_channels(); c++) {
              EXPECT_GE(output[((i * output_channel_stride() + g * group_output_channels() + c) * output_height() + y) * output_width() + x], output_min)
                << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c << ", image = " << i;
              EXPECT_LE(output[((i * output_channel_stride() + g * group_output_channels() + c) * output_height() + y) * output_width() + x], output_max)
                << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c << ", image = " << i;
              EXPECT_NEAR(
                  output_ref[(((i * groups() + g) * group_output_channels() + c) * output_height() + y) * output_width() + x],
                  output[((i * output_channel_stride() + g * group_output_channels() + c) * output_height() + y) * output_width() + x],
                  1.0e-4 * std::abs(output_ref[(((i * groups() + g) * group_output_channels() + c) * output_height() + y) * output_width() + x]))
                << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c << ", image = " << i;
            }
          }
        }
      }
    }
  }

  void TestNCHWxF16() {
    switch (weights_type()) {
      case WeightsType::Default:
        break;
      case WeightsType::FP32:
        break;
      default:
        GTEST_FAIL() << "unexpected weights type";
    }

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);
    std::uniform_real_distribution<float> pdist;

    std::vector<uint16_t> input(2 * XNN_EXTRA_BYTES / sizeof(uint16_t) +
      ((batch_size() - 1) * input_channel_stride() + groups() * group_input_channels()) * input_height() * input_width());
    std::vector<uint16_t> kernel(
      groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<float> kernel_as_float(kernel.size());
    std::vector<uint16_t> bias(groups() * group_output_channels());
    std::vector<float> bias_as_float(bias.size());
    std::vector<uint16_t> output(
      ((batch_size() - 1) * output_channel_stride() + groups() * group_output_channels()) * output_height() * output_width());
    std::vector<float> output_ref(batch_size() * groups() * group_output_channels() * output_height() * output_width());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      for (uint16_t& k : kernel) {
        if (pdist(rng) <= sparsity()) {
          k = 0;
        }
      }
      std::transform(kernel.cbegin(), kernel.cend(), kernel_as_float.begin(), fp16_ieee_to_fp32_value);
      std::generate(bias.begin(), bias.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::transform(bias.cbegin(), bias.cend(), bias_as_float.begin(), fp16_ieee_to_fp32_value);
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results, without clamping.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  output_ref[(((i * groups() + g) * group_output_channels() + oc) * output_height() + oy) * output_width() + ox] =
                    fp16_ieee_to_fp32_value(bias[g * group_output_channels() + oc]);
                }
              }
            }
          }
        }
      } else {
        std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      }
      if (force_nhwc_input()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t ky = 0; ky < kernel_height(); ky++) {
                const size_t iy = oy * subsampling_height() + ky * dilation_height() - padding_top();
                if (iy < input_height()) {
                  for (size_t kx = 0; kx < kernel_width(); kx++) {
                    const size_t ix = ox * subsampling_width() + kx * dilation_width() - padding_left();
                    if (ix < input_width()) {
                      for (size_t g = 0; g < groups(); g++) {
                        for (size_t oc = 0; oc < group_output_channels(); oc++) {
                          for (size_t ic = 0; ic < group_input_channels(); ic++) {
                            output_ref[(((i * groups() + g) * group_output_channels() + oc) * output_height() + oy) * output_width() + ox] +=
                              fp16_ieee_to_fp32_value(input[((((i * input_height() + iy) * input_width() + ix) * groups() + g) * group_input_channels() + ic)]) *
                              fp16_ieee_to_fp32_value(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]);
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else if (depthwise_layout()) {
        ASSERT_EQ(group_input_channels(), 1);

        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t ky = 0; ky < kernel_height(); ky++) {
                const size_t iy = oy * subsampling_height() + ky * dilation_height() - padding_top();
                if (iy < input_height()) {
                  for (size_t kx = 0; kx < kernel_width(); kx++) {
                    const size_t ix = ox * subsampling_width() + kx * dilation_width() - padding_left();
                    if (ix < input_width()) {
                      for (size_t g = 0; g < groups(); g++) {
                        for (size_t oc = 0; oc < group_output_channels(); oc++) {
                          output_ref[(((i * groups() + g) * group_output_channels() + oc) * output_height() + oy) * output_width() + ox] +=
                            fp16_ieee_to_fp32_value(input[((i * input_channel_stride() + g) * input_height() + iy) * input_width() + ix]) *
                            fp16_ieee_to_fp32_value(kernel[((ky * kernel_width() + kx) * groups() + g) * group_output_channels() + oc]);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t ky = 0; ky < kernel_height(); ky++) {
                const size_t iy = oy * subsampling_height() + ky * dilation_height() - padding_top();
                if (iy < input_height()) {
                  for (size_t kx = 0; kx < kernel_width(); kx++) {
                    const size_t ix = ox * subsampling_width() + kx * dilation_width() - padding_left();
                    if (ix < input_width()) {
                      for (size_t g = 0; g < groups(); g++) {
                        for (size_t oc = 0; oc < group_output_channels(); oc++) {
                          for (size_t ic = 0; ic < group_input_channels(); ic++) {
                            output_ref[(((i * groups() + g) * group_output_channels() + oc) * output_height() + oy) * output_width() + ox] +=
                              fp16_ieee_to_fp32_value(input[((i * input_channel_stride() + g * group_input_channels() + ic) * input_height() + iy) * input_width() + ix]) *
                              fp16_ieee_to_fp32_value(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]);
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float scaled_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_min + accumulated_range / 255.0f * float(qmin())));
      const float scaled_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_max - accumulated_range / 255.0f * float(255 - qmax())));
      const float output_min = scaled_min == scaled_max ? -std::numeric_limits<float>::infinity() : scaled_min;
      const float output_max = scaled_min == scaled_max ? +std::numeric_limits<float>::infinity() : scaled_max;

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, run, and destroy Convolution operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convolution_op = nullptr;
      struct xnn_internal_weights_cache* internal_weights_cache = nullptr;
      std::unique_ptr<xnn_weights_cache_provider, decltype(&xnn_delete_weights_cache)> auto_weights_cache(
        nullptr, xnn_delete_weights_cache);
      if (use_weights_cache()) {
        xnn_weights_cache_t weights_cache = nullptr;
        xnn_create_weights_cache(&weights_cache);
        auto_weights_cache.reset(weights_cache);
        if (weights_cache) {
          internal_weights_cache = (struct xnn_internal_weights_cache*) weights_cache->context;
        }
      }

      const void* kernel_data = kernel.data();
      const void* bias_data = bias.data();
      if (weights_type() == WeightsType::FP32) {
        kernel_data = kernel_as_float.data();
        bias_data = bias_as_float.data();
      }
      uint32_t flags = 0;
      if (depthwise_layout()) {
        flags |= XNN_FLAG_DEPTHWISE_CONVOLUTION;
      }
      if (force_nhwc_input()) {
        flags |= XNN_FLAG_INPUT_NHWC;
      }
      if (weights_type() == WeightsType::FP32) {
        flags |= XNN_FLAG_FP32_STATIC_WEIGHTS;
      }
      xnn_status status = xnn_create_convolution2d_nchw_f16(
          padding_top(), padding_right(), padding_bottom(), padding_left(),
          kernel_height(), kernel_width(),
          subsampling_height(), subsampling_width(),
          dilation_height(), dilation_width(),
          groups(), group_input_channels(), group_output_channels(),
          input_channel_stride(), output_channel_stride(),
          kernel_data, has_bias() ? bias_data : nullptr,
          output_min, output_max,
          flags,
          /*code_cache=*/nullptr, auto_weights_cache.get(),
          &convolution_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, convolution_op);
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete convolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convolution_op(convolution_op, xnn_delete_operator);

      ASSERT_EQ(
        xnn_status_success, xnn_reshape_convolution2d_nchw_f16(
                              convolution_op, batch_size(), input_height(), input_width(),
                              /*output_height_out=*/nullptr, /*output_width_out=*/nullptr, auto_threadpool.get()));
      ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nchw_f16(convolution_op, input.data(), output.data()));
      ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op, auto_threadpool.get()));

      VerifyNCHWxF16(output, output_ref, output_min, output_max);

      if (use_weights_cache()) {
        xnn_operator_t convolution_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;
        ASSERT_EQ(
            xnn_status_success,

            xnn_create_convolution2d_nchw_f16(
                padding_top(), padding_right(), padding_bottom(), padding_left(),
                kernel_height(), kernel_width(),
                subsampling_height(), subsampling_width(),
                dilation_height(), dilation_width(),
                groups(), group_input_channels(), group_output_channels(),
                input_channel_stride(), output_channel_stride(),
                kernel_data, has_bias() ? bias_data : nullptr,
                output_min, output_max,
                flags,
                /*code_cache=*/nullptr, auto_weights_cache.get(),
                &convolution_op2));

        ASSERT_NE(nullptr, convolution_op2);

        // Smart pointer to automatically delete convolution_op2.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convolution_op(convolution_op2, xnn_delete_operator);
        std::vector<uint16_t> output2(output.size(), UINT16_C(0x7E00) /* NaN */);

        ASSERT_EQ(
          xnn_status_success,
          xnn_reshape_convolution2d_nchw_f16(
            convolution_op2, batch_size(), input_height(), input_width(),
            /*output_height_out=*/nullptr, /*output_width_out=*/nullptr, auto_threadpool.get()));
        ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nchw_f16(convolution_op2, input.data(), output2.data()));
        ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op2, auto_threadpool.get()));

        VerifyNCHWxF16(output2, output_ref, output_min, output_max);
        if (IsSpmm()) {
          VerifyWeightsCacheUnused(*internal_weights_cache);
        } else {
          VerifyWeightsCache(*internal_weights_cache, old_weights_cache_size);
        }
      }
    }
  }

  void VerifyNCHWxF16(const std::vector<uint16_t> &output,
                      const std::vector<float> &output_ref,
                      const float output_min, const float output_max) const {
    for (size_t i = 0; i < batch_size(); i++) {
      for (size_t y = 0; y < output_height(); y++) {
        for (size_t x = 0; x < output_width(); x++) {
          for (size_t g = 0; g < groups(); g++) {
            for (size_t c = 0; c < group_output_channels(); c++) {
              EXPECT_GE(fp16_ieee_to_fp32_value(output[((i * output_channel_stride() + g * group_output_channels() + c) * output_height() + y) * output_width() + x]), output_min)
                << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c << ", image = " << i;
              EXPECT_LE(fp16_ieee_to_fp32_value(output[((i * output_channel_stride() + g * group_output_channels() + c) * output_height() + y) * output_width() + x]), output_max)
                << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c << ", image = " << i;
              EXPECT_NEAR(output_ref[(((i * groups() + g) * group_output_channels() + c) * output_height() + y) * output_width() + x], fp16_ieee_to_fp32_value(output[((i * output_channel_stride() + g * group_output_channels() + c) * output_height() + y) * output_width() + x]), std::max(1.0e-4f, std::abs(output_ref[(((i * groups() + g) * group_output_channels() + c) * output_height() + y) * output_width() + x]) * 1.0e-2f))
                << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c << ", image = " << i;
            }
          }
        }
      }
    }
  }

  void TestSetupNHWCxQC8() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    ASSERT_FALSE(depthwise_layout());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    std::uniform_int_distribution<int32_t> w8dist(
      -std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) + std::max(
      batch_size() * ((input_height() * input_width() - 1) * input_channel_stride() + groups() * group_input_channels()),
      next_batch_size() * ((next_input_height() * next_input_width() - 1) * input_channel_stride() + groups() * group_input_channels())));
    std::vector<int8_t> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<int32_t> bias(groups() * group_output_channels());
    std::vector<int8_t> output(std::max(
      batch_size() * ((output_height() * output_width() - 1) * output_channel_stride() + groups() * group_output_channels()),
      next_batch_size() * ((next_output_height() * next_output_width() - 1) * output_channel_stride() + groups() * group_output_channels())));
    std::vector<int32_t> accumulators(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<double> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<float> requantization_scales(groups() * group_output_channels());
    std::vector<int32_t> next_accumulators(next_batch_size() * next_output_height() * next_output_width() * groups() * group_output_channels());
    std::vector<double> next_output_ref(next_batch_size() * next_output_height() * next_output_width() * groups() * group_output_channels());
    std::vector<float> next_requantization_scales(groups() * group_output_channels());

    const int8_t input_zero_point = -1;
    const int8_t output_zero_point = -1;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      // Compute reference results, without renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  accumulators[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(accumulators.begin(), accumulators.end(), 0);
      }
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t iy = oy * subsampling_height() + ky * dilation_height() - padding_top();
              if (iy < input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t ix = ox * subsampling_width() + kx * dilation_width() - padding_left();
                  if (ix < input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          accumulators[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            (int32_t(input[((i * input_height() + iy) * input_width() + ix) * input_channel_stride() + g * group_input_channels() + ic]) - int32_t(input_zero_point)) *
                            int32_t(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Compute renormalization parameters.
      for (size_t c = 0; c < groups() * group_output_channels(); c++) {
        int32_t accumulated_min = accumulators[c];
        int32_t accumulated_max = accumulators[c];
        for (size_t px = 0; px < batch_size() * output_height() * output_width(); px++) {
          accumulated_min = std::min(accumulated_min, accumulators[px * groups() * group_output_channels() + c]);
          accumulated_max = std::max(accumulated_max, accumulators[px * groups() * group_output_channels() + c]);
        }

        float requantization_scale = 0x1.0p-32f;
        if (accumulated_max != 0) {
          requantization_scale = std::max(requantization_scale,
            float(int32_t(std::numeric_limits<int8_t>::max()) - int32_t(output_zero_point)) / float(accumulated_max));
        }
        if (accumulated_min != 0) {
          requantization_scale = std::max(requantization_scale,
            float(int32_t(std::numeric_limits<int8_t>::min()) - int32_t(output_zero_point)) / float(accumulated_min));
        }
        requantization_scale = std::min(requantization_scale, 0x1.FFFFFEp-1f);

        requantization_scales[c] = requantization_scale;
      }

      // Renormalize reference results.
      for (size_t c = 0; c < groups() * group_output_channels(); c++) {
        for (size_t px = 0; px < batch_size() * output_height() * output_width(); px++) {
          output_ref[px * groups() * group_output_channels() + c] = double(int32_t(output_zero_point)) +
            double(accumulators[px * groups() * group_output_channels() + c]) * double(requantization_scales[c]);
        }
      }
      std::transform(output_ref.cbegin(), output_ref.cend(), output_ref.begin(),
        [this](double x) -> double {
          return std::max<double>(std::min<double>(x, double(qmax() - 0x80)), double(qmin() - 0x80));
        });

      // Create, setup, and run Convolution operator once.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convolution_op = nullptr;

      xnn_status status = xnn_create_convolution2d_nhwc_qs8_qc8w(
          padding_top(), padding_right(), padding_bottom(), padding_left(),
          kernel_height(), kernel_width(),
          subsampling_height(), subsampling_width(),
          dilation_height(), dilation_width(),
          groups(), group_input_channels(), group_output_channels(),
          input_channel_stride(), output_channel_stride(),
          input_zero_point, 1.0f /* input scale */, requantization_scales.data(),
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_zero_point, 1.0f /* output scale */, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80),
          0, nullptr, nullptr, &convolution_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, convolution_op);

      // Smart pointer to automatically delete convolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convolution_op(convolution_op, xnn_delete_operator);

      size_t workspace_size = SIZE_MAX;
      size_t workspace_alignment = SIZE_MAX;
      ASSERT_EQ(
        xnn_status_success, xnn_reshape_convolution2d_nhwc_qs8_qc8w(
                              convolution_op, batch_size(), input_height(), input_width(),
                              &workspace_size, &workspace_alignment,
                              /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                              auto_threadpool.get()));
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
      std::iota(workspace.begin(), workspace.end(), 0);
      ASSERT_NE(workspace_size, SIZE_MAX);
      ASSERT_NE(workspace_alignment, SIZE_MAX);
      ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_qs8_qc8w(convolution_op, workspace.data(), input.data(), output.data()));
      ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op, auto_threadpool.get()));

      // Verify results of the first run.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                EXPECT_LE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c]), int32_t(qmax() - 0x80))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_GE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c]), int32_t(qmin() - 0x80))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_NEAR(
                    output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c],
                    double(output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c]),
                    0.9)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              }
            }
          }
        }
      }

      // Re-generate data for the second run.
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      // Compute reference results for the second run, including renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < next_batch_size(); i++) {
          for (size_t oy = 0; oy < next_output_height(); oy++) {
            for (size_t ox = 0; ox < next_output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  next_accumulators[(((i * next_output_height() + oy) * next_output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(next_accumulators.begin(), next_accumulators.end(), 0);
      }
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t oy = 0; oy < next_output_height(); oy++) {
          for (size_t ox = 0; ox < next_output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t iy = oy * subsampling_height() + ky * dilation_height() - padding_top();
              if (iy < next_input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t ix = ox * subsampling_width() + kx * dilation_width() - padding_left();
                  if (ix < next_input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          next_accumulators[(((i * next_output_height() + oy) * next_output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            (int32_t(input[((i * next_input_height() + iy) * next_input_width() + ix) * input_channel_stride() + g * group_input_channels() + ic]) - int32_t(input_zero_point)) *
                            int32_t(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      for (size_t c = 0; c < groups() * group_output_channels(); c++) {
        for (size_t px = 0; px < next_batch_size() * next_output_height() * next_output_width(); px++) {
          next_output_ref[px * groups() * group_output_channels() + c] = double(int32_t(output_zero_point)) +
            double(next_accumulators[px * groups() * group_output_channels() + c]) * double(requantization_scales[c]);
        }
      }
      std::transform(next_output_ref.cbegin(), next_output_ref.cend(), next_output_ref.begin(),
        [this](double x) -> double {
          return std::max<double>(std::min<double>(x, double(qmax() - 0x80)), double(qmin() - 0x80));
        });

      // Setup and run Convolution operator the second time, and destroy the operator.
      workspace_size = SIZE_MAX;
      workspace_alignment = SIZE_MAX;
      ASSERT_EQ(
        xnn_status_success, xnn_reshape_convolution2d_nhwc_qs8_qc8w(
                              convolution_op, next_batch_size(), next_input_height(), next_input_width(),
                              &workspace_size, &workspace_alignment,
                              /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                              auto_threadpool.get()));
      ASSERT_NE(workspace_size, SIZE_MAX);
      ASSERT_NE(workspace_alignment, SIZE_MAX);
      ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_qs8_qc8w(convolution_op, workspace.data(), input.data(), output.data()));
      ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op, auto_threadpool.get()));

      // Verify results of the second run.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t y = 0; y < next_output_height(); y++) {
          for (size_t x = 0; x < next_output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                EXPECT_LE(int32_t(output[((i * next_output_height() + y) * next_output_width() + x) * output_channel_stride() + g * group_output_channels() + c]), int32_t(qmax() - 0x80))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_GE(int32_t(output[((i * next_output_height() + y) * next_output_width() + x) * output_channel_stride() + g * group_output_channels() + c]), int32_t(qmin() - 0x80))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_NEAR(
                    next_output_ref[(((i * next_output_height() + y) * next_output_width() + x) * groups() + g) * group_output_channels() + c],
                    double(output[((i * next_output_height() + y) * next_output_width() + x) * output_channel_stride() + g * group_output_channels() + c]),
                    0.9)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              }
            }
          }
        }
      }
    }
  }

  void TestSetupNHWCxQS8() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    ASSERT_FALSE(depthwise_layout());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    std::uniform_int_distribution<int32_t> w8dist(
      -std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) + std::max(
      batch_size() * ((input_height() * input_width() - 1) * input_channel_stride() + groups() * group_input_channels()),
      next_batch_size() * ((next_input_height() * next_input_width() - 1) * input_channel_stride() + groups() * group_input_channels())));
    std::vector<int8_t> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<int32_t> bias(groups() * group_output_channels());
    std::vector<int8_t> output(std::max(
      batch_size() * ((output_height() * output_width() - 1) * output_channel_stride() + groups() * group_output_channels()),
      next_batch_size() * ((next_output_height() * next_output_width() - 1) * output_channel_stride() + groups() * group_output_channels())));
    std::vector<int32_t> accumulators(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<double> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<int32_t> next_accumulators(next_batch_size() * next_output_height() * next_output_width() * groups() * group_output_channels());
    std::vector<double> next_output_ref(next_batch_size() * next_output_height() * next_output_width() * groups() * group_output_channels());

    const int8_t input_zero_point = -1;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      // Compute reference results, without renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  accumulators[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(accumulators.begin(), accumulators.end(), 0);
      }
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t iy = oy * subsampling_height() + ky * dilation_height() - padding_top();
              if (iy < input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t ix = ox * subsampling_width() + kx * dilation_width() - padding_left();
                  if (ix < input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          accumulators[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            (int32_t(input[((i * input_height() + iy) * input_width() + ix) * input_channel_stride() + g * group_input_channels() + ic]) - int32_t(input_zero_point)) *
                            int32_t(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Compute renormalization parameters.
      const int32_t accumulated_min = *std::min_element(accumulators.cbegin(), accumulators.cend());
      const int32_t accumulated_max = *std::max_element(accumulators.cbegin(), accumulators.cend());

      const double output_scale = double(uint32_t(accumulated_max - accumulated_min)) / 255.0;
      const int8_t output_zero_point = int8_t(std::max(std::min(
        lrint(-0.5 - 0.5 * double(accumulated_min + accumulated_max) / output_scale),
        long(std::numeric_limits<int8_t>::max())), long(std::numeric_limits<int8_t>::min())));

      // Renormalize reference results.
      std::transform(accumulators.cbegin(), accumulators.cend(), output_ref.begin(),
        [this, output_scale, output_zero_point](int32_t x) -> double {
          return std::max<double>(std::min<double>(double(x) / output_scale, double(qmax() - 0x80) - output_zero_point), double(qmin() - 0x80) - output_zero_point);
        });

      // Create, setup, and run Convolution operator once.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convolution_op = nullptr;

      xnn_status status = xnn_create_convolution2d_nhwc_qs8(
          padding_top(), padding_right(), padding_bottom(), padding_left(),
          kernel_height(), kernel_width(),
          subsampling_height(), subsampling_width(),
          dilation_height(), dilation_width(),
          groups(), group_input_channels(), group_output_channels(),
          input_channel_stride(), output_channel_stride(),
          input_zero_point, 1.0f /* input scale */, 1.0f /* kernel scale */,
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_zero_point, output_scale, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80),
          0, nullptr, nullptr, &convolution_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, convolution_op);

      // Smart pointer to automatically delete convolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convolution_op(convolution_op, xnn_delete_operator);

      size_t workspace_size = SIZE_MAX;
      size_t workspace_alignment = SIZE_MAX;
      ASSERT_EQ(
        xnn_status_success, xnn_reshape_convolution2d_nhwc_qs8(
                              convolution_op, batch_size(), input_height(), input_width(),
                              &workspace_size, &workspace_alignment,
                              /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                              auto_threadpool.get()));
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
      std::iota(workspace.begin(), workspace.end(), 0);
      ASSERT_NE(workspace_size, SIZE_MAX);
      ASSERT_NE(workspace_alignment, SIZE_MAX);
      ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_qs8(convolution_op, workspace.data(), input.data(), output.data()));
      ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op, auto_threadpool.get()));

      // Verify results of the first run.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                EXPECT_LE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c]), int32_t(qmax() - 0x80))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_GE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c]), int32_t(qmin() - 0x80))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_NEAR(
                    output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c],
                    double(output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c]) - double(output_zero_point),
                    0.9)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              }
            }
          }
        }
      }

      // Re-generate data for the second run.
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      // Compute reference results for the second run, including renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < next_batch_size(); i++) {
          for (size_t oy = 0; oy < next_output_height(); oy++) {
            for (size_t ox = 0; ox < next_output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  next_accumulators[(((i * next_output_height() + oy) * next_output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(next_accumulators.begin(), next_accumulators.end(), 0);
      }
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t oy = 0; oy < next_output_height(); oy++) {
          for (size_t ox = 0; ox < next_output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t iy = oy * subsampling_height() + ky * dilation_height() - padding_top();
              if (iy < next_input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t ix = ox * subsampling_width() + kx * dilation_width() - padding_left();
                  if (ix < next_input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          next_accumulators[(((i * next_output_height() + oy) * next_output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            (int32_t(input[((i * next_input_height() + iy) * next_input_width() + ix) * input_channel_stride() + g * group_input_channels() + ic]) - int32_t(input_zero_point)) *
                            int32_t(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      std::transform(next_accumulators.cbegin(), next_accumulators.cend(), next_output_ref.begin(),
        [this, output_scale, output_zero_point](int32_t x) -> double {
          return std::max<double>(std::min<double>(double(x) / output_scale, double(qmax() - 0x80) - output_zero_point), double(qmin() - 0x80) - output_zero_point);
        });

      workspace_size = SIZE_MAX;
      workspace_alignment = SIZE_MAX;
      // Setup and run Convolution operator the second time, and destroy the operator.
      ASSERT_EQ(
        xnn_status_success, xnn_reshape_convolution2d_nhwc_qs8(
                              convolution_op, next_batch_size(), next_input_height(), next_input_width(),
                              &workspace_size, &workspace_alignment,
                              /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                              auto_threadpool.get()));
      ASSERT_NE(workspace_size, SIZE_MAX);
      ASSERT_NE(workspace_alignment, SIZE_MAX);
      ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_qs8(convolution_op, workspace.data(), input.data(), output.data()));
      ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op, auto_threadpool.get()));

      // Verify results of the second run.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t y = 0; y < next_output_height(); y++) {
          for (size_t x = 0; x < next_output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                EXPECT_LE(int32_t(output[((i * next_output_height() + y) * next_output_width() + x) * output_channel_stride() + g * group_output_channels() + c]), int32_t(qmax() - 0x80))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_GE(int32_t(output[((i * next_output_height() + y) * next_output_width() + x) * output_channel_stride() + g * group_output_channels() + c]), int32_t(qmin() - 0x80))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_NEAR(
                    next_output_ref[(((i * next_output_height() + y) * next_output_width() + x) * groups() + g) * group_output_channels() + c],
                    double(output[((i * next_output_height() + y) * next_output_width() + x) * output_channel_stride() + g * group_output_channels() + c]) - double(output_zero_point),
                    0.9)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              }
            }
          }
        }
      }
    }
  }

  void TestSetupNHWCxQU8() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    ASSERT_FALSE(depthwise_layout());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) + std::max(
      batch_size() * ((input_height() * input_width() - 1) * input_channel_stride() + groups() * group_input_channels()),
      next_batch_size() * ((next_input_height() * next_input_width() - 1) * input_channel_stride() + groups() * group_input_channels())));
    std::vector<uint8_t> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<int32_t> bias(groups() * group_output_channels());
    std::vector<uint8_t> output(std::max(
      batch_size() * ((output_height() * output_width() - 1) * output_channel_stride() + groups() * group_output_channels()),
      next_batch_size() * ((next_output_height() * next_output_width() - 1) * output_channel_stride() + groups() * group_output_channels())));
    std::vector<int32_t> accumulators(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<double> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<int32_t> next_accumulators(next_batch_size() * next_output_height() * next_output_width() * groups() * group_output_channels());
    std::vector<double> next_output_ref(next_batch_size() * next_output_height() * next_output_width() * groups() * group_output_channels());

    const uint8_t input_zero_point = 127;
    const uint8_t kernel_zero_point = 127;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return u8dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
      std::fill(output.begin(), output.end(), UINT8_C(0xA5));

      // Compute reference results, without renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  accumulators[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(accumulators.begin(), accumulators.end(), 0);
      }
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t iy = oy * subsampling_height() + ky * dilation_height() - padding_top();
              if (iy < input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t ix = ox * subsampling_width() + kx * dilation_width() - padding_left();
                  if (ix < input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          accumulators[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            (int32_t(input[((i * input_height() + iy) * input_width() + ix) * input_channel_stride() + g * group_input_channels() + ic]) - int32_t(input_zero_point)) *
                            (int32_t(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]) - int32_t(kernel_zero_point));
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Compute renormalization parameters.
      const int32_t accumulated_min = *std::min_element(accumulators.cbegin(), accumulators.cend());
      const int32_t accumulated_max = *std::max_element(accumulators.cbegin(), accumulators.cend());

      const double output_scale = double(uint32_t(accumulated_max - accumulated_min)) / 255.0;
      const uint8_t output_zero_point = uint8_t(std::max(std::min(
        lrint(127.5 - 0.5 * double(accumulated_min + accumulated_max) / output_scale),
        long(std::numeric_limits<uint8_t>::max())), long(std::numeric_limits<uint8_t>::min())));

      // Renormalize reference results.
      std::transform(accumulators.cbegin(), accumulators.cend(), output_ref.begin(),
        [this, output_scale, output_zero_point](int32_t x) -> double {
          return std::max<double>(std::min<double>(double(x) / output_scale, double(qmax()) - output_zero_point), double(qmin()) - output_zero_point);
        });

      // Create, setup, and run Convolution operator once.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convolution_op = nullptr;

      xnn_status status = xnn_create_convolution2d_nhwc_qu8(
          padding_top(), padding_right(), padding_bottom(), padding_left(),
          kernel_height(), kernel_width(),
          subsampling_height(), subsampling_width(),
          dilation_height(), dilation_width(),
          groups(), group_input_channels(), group_output_channels(),
          input_channel_stride(), output_channel_stride(),
          input_zero_point, 1.0f /* input scale */,
          kernel_zero_point, 1.0f /* kernel scale */,
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_zero_point, output_scale, qmin(), qmax(),
          0, nullptr, nullptr, &convolution_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, convolution_op);

      // Smart pointer to automatically delete convolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convolution_op(convolution_op, xnn_delete_operator);

      size_t workspace_size = SIZE_MAX;
      size_t workspace_alignment = SIZE_MAX;
      ASSERT_EQ(
        xnn_status_success, xnn_reshape_convolution2d_nhwc_qu8(
                              convolution_op, batch_size(), input_height(), input_width(),
                              &workspace_size, &workspace_alignment,
                              /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                              auto_threadpool.get()));
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
      std::iota(workspace.begin(), workspace.end(), 0);
      ASSERT_NE(workspace_size, SIZE_MAX);
      ASSERT_NE(workspace_alignment, SIZE_MAX);
      ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_qu8(convolution_op, workspace.data(), input.data(), output.data()));
      ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op, auto_threadpool.get()));

      // Verify results of the first run.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                EXPECT_LE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c]), int32_t(qmax()))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_GE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c]), int32_t(qmin()))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_NEAR(
                    output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c],
                    double(output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c]) - double(output_zero_point),
                    0.9)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              }
            }
          }
        }
      }

      // Re-generate data for the second run.
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::fill(output.begin(), output.end(), 0xA5);

      // Compute reference results for the second run, including renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < next_batch_size(); i++) {
          for (size_t oy = 0; oy < next_output_height(); oy++) {
            for (size_t ox = 0; ox < next_output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  next_accumulators[(((i * next_output_height() + oy) * next_output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(next_accumulators.begin(), next_accumulators.end(), 0);
      }
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t oy = 0; oy < next_output_height(); oy++) {
          for (size_t ox = 0; ox < next_output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t iy = oy * subsampling_height() + ky * dilation_height() - padding_top();
              if (iy < next_input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t ix = ox * subsampling_width() + kx * dilation_width() - padding_left();
                  if (ix < next_input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          next_accumulators[(((i * next_output_height() + oy) * next_output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            (int32_t(input[((i * next_input_height() + iy) * next_input_width() + ix) * input_channel_stride() + g * group_input_channels() + ic]) - int32_t(input_zero_point)) *
                            (int32_t(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]) - int32_t(kernel_zero_point));
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      std::transform(next_accumulators.cbegin(), next_accumulators.cend(), next_output_ref.begin(),
        [this, output_scale, output_zero_point](int32_t x) -> double {
          return std::max<double>(std::min<double>(double(x) / output_scale, double(qmax()) - output_zero_point), double(qmin()) - output_zero_point);
        });

      workspace_size = SIZE_MAX;
      workspace_alignment = SIZE_MAX;
      // Setup and run Convolution operator the second time, and destroy the operator.
      ASSERT_EQ(
        xnn_status_success, xnn_reshape_convolution2d_nhwc_qu8(
                              convolution_op, next_batch_size(), next_input_height(), next_input_width(),
                              &workspace_size, &workspace_alignment,
                              /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                              auto_threadpool.get()));
      ASSERT_NE(workspace_size, SIZE_MAX);
      ASSERT_NE(workspace_alignment, SIZE_MAX);
      ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_qu8(convolution_op, workspace.data(), input.data(), output.data()));
      ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op, auto_threadpool.get()));

      // Verify results of the second run.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t y = 0; y < next_output_height(); y++) {
          for (size_t x = 0; x < next_output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                EXPECT_LE(int32_t(output[((i * next_output_height() + y) * next_output_width() + x) * output_channel_stride() + g * group_output_channels() + c]), int32_t(qmax()))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_GE(int32_t(output[((i * next_output_height() + y) * next_output_width() + x) * output_channel_stride() + g * group_output_channels() + c]), int32_t(qmin()))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_NEAR(
                    next_output_ref[(((i * next_output_height() + y) * next_output_width() + x) * groups() + g) * group_output_channels() + c],
                    double(output[((i * next_output_height() + y) * next_output_width() + x) * output_channel_stride() + g * group_output_channels() + c]) - double(output_zero_point),
                    0.9)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              }
            }
          }
        }
      }
    }
  }

  void TestSetupNHWCxF16() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    ASSERT_FALSE(depthwise_layout());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);

    std::vector<uint16_t> input(XNN_EXTRA_BYTES / sizeof(uint16_t) + std::max(
      batch_size() * ((input_height() * input_width() - 1) * input_channel_stride() + groups() * group_input_channels()),
      next_batch_size() * ((next_input_height() * next_input_width() - 1) * input_channel_stride() + groups() * group_input_channels())));
    std::vector<uint16_t> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<uint16_t> bias(groups() * group_output_channels());
    std::vector<uint16_t> output(std::max(
      batch_size() * ((output_height() * output_width() - 1) * output_channel_stride() + groups() * group_output_channels()),
      next_batch_size() * ((next_output_height() * next_output_width() - 1) * output_channel_stride() + groups() * group_output_channels())));
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<float> next_output_ref(next_batch_size() * next_output_height() * next_output_width() * groups() * group_output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(bias.begin(), bias.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results, without clamping.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    fp16_ieee_to_fp32_value(bias[g * group_output_channels() + oc]);
                }
              }
            }
          }
        }
      } else {
        std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      }
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t iy = oy * subsampling_height() + ky * dilation_height() - padding_top();
              if (iy < input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t ix = ox * subsampling_width() + kx * dilation_width() - padding_left();
                  if (ix < input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            fp16_ieee_to_fp32_value(input[((i * input_height() + iy) * input_width() + ix) * input_channel_stride() + g * group_input_channels() + ic]) *
                            fp16_ieee_to_fp32_value(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float scaled_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_min + accumulated_range / 255.0f * float(qmin())));
      const float scaled_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_max - accumulated_range / 255.0f * float(255 - qmax())));
      const float output_min = scaled_min == scaled_max ? -std::numeric_limits<float>::infinity() : scaled_min;
      const float output_max = scaled_min == scaled_max ? +std::numeric_limits<float>::infinity() : scaled_max;

      for (float& output_value : output_ref) {
        output_value = std::min(std::max(output_value, output_min), output_max);
      }

      // Create, setup, and run Convolution operator once.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convolution_op = nullptr;

      xnn_status status = xnn_create_convolution2d_nhwc_f16(
          padding_top(), padding_right(), padding_bottom(), padding_left(),
          kernel_height(), kernel_width(),
          subsampling_height(), subsampling_width(),
          dilation_height(), dilation_width(),
          groups(), group_input_channels(), group_output_channels(),
          input_channel_stride(), output_channel_stride(),
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_min, output_max,
          0, nullptr, nullptr, &convolution_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, convolution_op);

      // Smart pointer to automatically delete convolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convolution_op(convolution_op, xnn_delete_operator);

      size_t workspace_size = SIZE_MAX;
      size_t workspace_alignment = SIZE_MAX;
      ASSERT_EQ(
        xnn_status_success, xnn_reshape_convolution2d_nhwc_f16(
                              convolution_op, batch_size(), input_height(), input_width(),
                              &workspace_size, &workspace_alignment,
                              /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                              auto_threadpool.get()));
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
      std::iota(workspace.begin(), workspace.end(), 0);
      ASSERT_NE(workspace_size, SIZE_MAX);
      ASSERT_NE(workspace_alignment, SIZE_MAX);
      ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_f16(convolution_op, workspace.data(), input.data(), output.data()));
      ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op, auto_threadpool.get()));

      // Verify results of the first run.
      VerifyNHWCxF16(output, output_ref, output_min, output_max, batch_size(), output_height(), output_width());

      // Re-generate data for the second run.
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results for the second run, including clamping.
      if (has_bias()) {
        for (size_t i = 0; i < next_batch_size(); i++) {
          for (size_t oy = 0; oy < next_output_height(); oy++) {
            for (size_t ox = 0; ox < next_output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  next_output_ref[(((i * next_output_height() + oy) * next_output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    fp16_ieee_to_fp32_value(bias[g * group_output_channels() + oc]);
                }
              }
            }
          }
        }
      } else {
        std::fill(next_output_ref.begin(), next_output_ref.end(), 0.0f);
      }
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t oy = 0; oy < next_output_height(); oy++) {
          for (size_t ox = 0; ox < next_output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t iy = oy * subsampling_height() + ky * dilation_height() - padding_top();
              if (iy < next_input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t ix = ox * subsampling_width() + kx * dilation_width() - padding_left();
                  if (ix < next_input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          next_output_ref[(((i * next_output_height() + oy) * next_output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            fp16_ieee_to_fp32_value(input[((i * next_input_height() + iy) * next_input_width() + ix) * input_channel_stride() + g * group_input_channels() + ic]) *
                            fp16_ieee_to_fp32_value(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      for (float& value : next_output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      workspace_size = SIZE_MAX;
      workspace_alignment = SIZE_MAX;
      // Setup and run Convolution operator the second time, and destroy the operator.
      ASSERT_EQ(
        xnn_status_success, xnn_reshape_convolution2d_nhwc_f16(
                              convolution_op, next_batch_size(), next_input_height(), next_input_width(),
                              &workspace_size, &workspace_alignment,
                              /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                              auto_threadpool.get()));
      ASSERT_NE(workspace_size, SIZE_MAX);
      ASSERT_NE(workspace_alignment, SIZE_MAX);
      ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_f16(convolution_op, workspace.data(), input.data(), output.data()));
      ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op, auto_threadpool.get()));

      // Verify results of the second run.
      VerifyNHWCxF16(output, next_output_ref, output_min, output_max, next_batch_size(), next_output_height(), next_output_width());
    }
  }

  void TestSetupNHWCxF32() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    ASSERT_FALSE(depthwise_layout());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) + std::max(
      batch_size() * ((input_height() * input_width() - 1) * input_channel_stride() + groups() * group_input_channels()),
      next_batch_size() * ((next_input_height() * next_input_width() - 1) * input_channel_stride() + groups() * group_input_channels())));
    std::vector<float> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<float> bias(groups() * group_output_channels());
    std::vector<float> output(std::max(
      batch_size() * ((output_height() * output_width() - 1) * output_channel_stride() + groups() * group_output_channels()),
      next_batch_size() * ((next_output_height() * next_output_width() - 1) * output_channel_stride() + groups() * group_output_channels())));
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<float> next_output_ref(next_batch_size() * next_output_height() * next_output_width() * groups() * group_output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), nanf(""));

      // Compute reference results, without clamping.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      }
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t iy = oy * subsampling_height() + ky * dilation_height() - padding_top();
              if (iy < input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t ix = ox * subsampling_width() + kx * dilation_width() - padding_left();
                  if (ix < input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            input[((i * input_height() + iy) * input_width() + ix) * input_channel_stride() + g * group_input_channels() + ic] *
                            kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic];
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());

      const float output_min = accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      const float output_max = accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, and run Convolution operator once.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convolution_op = nullptr;

      xnn_status status = xnn_create_convolution2d_nhwc_f32(
          padding_top(), padding_right(), padding_bottom(), padding_left(),
          kernel_height(), kernel_width(),
          subsampling_height(), subsampling_width(),
          dilation_height(), dilation_width(),
          groups(), group_input_channels(), group_output_channels(),
          input_channel_stride(), output_channel_stride(),
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_min, output_max,
          0, nullptr, nullptr, &convolution_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, convolution_op);

      // Smart pointer to automatically delete convolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convolution_op(convolution_op, xnn_delete_operator);

      size_t workspace_size = SIZE_MAX;
      size_t workspace_alignment = SIZE_MAX;
      ASSERT_EQ(
        xnn_status_success, xnn_reshape_convolution2d_nhwc_f32(
                              convolution_op, batch_size(), input_height(), input_width(),
                              &workspace_size, &workspace_alignment,
                              /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                              auto_threadpool.get()));
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
      std::iota(workspace.begin(), workspace.end(), 0);
      ASSERT_NE(workspace_size, SIZE_MAX);
      ASSERT_NE(workspace_alignment, SIZE_MAX);
      ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_f32(convolution_op, workspace.data(), input.data(), output.data()));
      ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op, auto_threadpool.get()));

      // Verify results of the first run.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                EXPECT_GE(output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c], output_min)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_LE(output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c], output_max)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_NEAR(
                    output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c],
                    output[((i * output_height() + y) * output_width() + x) * output_channel_stride() + g * group_output_channels() + c],
                    1.0e-4 * std::abs(output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c]))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              }
            }
          }
        }
      }

      // Re-generate data for the second run.
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), nanf(""));

      // Compute reference results for the second run, including clamping.
      if (has_bias()) {
        for (size_t i = 0; i < next_batch_size(); i++) {
          for (size_t oy = 0; oy < next_output_height(); oy++) {
            for (size_t ox = 0; ox < next_output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  next_output_ref[(((i * next_output_height() + oy) * next_output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(next_output_ref.begin(), next_output_ref.end(), 0.0f);
      }
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t oy = 0; oy < next_output_height(); oy++) {
          for (size_t ox = 0; ox < next_output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t iy = oy * subsampling_height() + ky * dilation_height() - padding_top();
              if (iy < next_input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t ix = ox * subsampling_width() + kx * dilation_width() - padding_left();
                  if (ix < next_input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          next_output_ref[(((i * next_output_height() + oy) * next_output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            input[((i * next_input_height() + iy) * next_input_width() + ix) * input_channel_stride() + g * group_input_channels() + ic] *
                            kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic];
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      for (float& value : next_output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      workspace_size = SIZE_MAX;
      workspace_alignment = SIZE_MAX;
      // Setup and run Convolution operator the second time, and destroy the operator.
      ASSERT_EQ(
        xnn_status_success, xnn_reshape_convolution2d_nhwc_f32(
                              convolution_op, next_batch_size(), next_input_height(), next_input_width(),
                              &workspace_size, &workspace_alignment,
                              /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                              auto_threadpool.get()));
      ASSERT_NE(workspace_size, SIZE_MAX);
      ASSERT_NE(workspace_alignment, SIZE_MAX);
      ASSERT_EQ(xnn_status_success, xnn_setup_convolution2d_nhwc_f32(convolution_op, workspace.data(), input.data(), output.data()));
      ASSERT_EQ(xnn_status_success, xnn_run_operator(convolution_op, auto_threadpool.get()));

      // Verify results of the second run.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t y = 0; y < next_output_height(); y++) {
          for (size_t x = 0; x < next_output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                EXPECT_GE(output[((i * next_output_height() + y) * next_output_width() + x) * output_channel_stride() + g * group_output_channels() + c], output_min)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_LE(output[((i * next_output_height() + y) * next_output_width() + x) * output_channel_stride() + g * group_output_channels() + c], output_max)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_NEAR(
                    next_output_ref[(((i * next_output_height() + y) * next_output_width() + x) * groups() + g) * group_output_channels() + c],
                    output[((i * next_output_height() + y) * next_output_width() + x) * output_channel_stride() + g * group_output_channels() + c],
                    1.0e-4 * std::abs(next_output_ref[(((i * next_output_height() + y) * next_output_width() + x) * groups() + g) * group_output_channels() + c]))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              }
            }
          }
        }
      }
    }
  }

  void VerifyWeightsCache(const xnn_internal_weights_cache &weights_cache, size_t old_size) const {
    ASSERT_EQ(weights_cache.cache.hits, 1);
    // Ensure that we did not write more weights to the cache because it was a
    // cache hit.
    ASSERT_EQ(old_size, weights_cache.cache.weights.size);
  };

  void VerifyWeightsCacheUnused(const xnn_internal_weights_cache &weights_cache) const {
    ASSERT_EQ(weights_cache.cache.hits, 0);
    ASSERT_EQ(0, weights_cache.cache.weights.size);
  }

  bool IsSpmm() const {
    const bool is_1x1 = kernel_width() == 1 && kernel_height() == 1 &&
        subsampling_height() == 1 && subsampling_width() == 1;
    const bool any_padding = (padding_left() | padding_top() | padding_right() | padding_bottom()) != 0;
    return is_1x1 && !any_padding && !force_nhwc_input() && groups() == 1;
  }

 private:
  uint32_t padding_top_{0};
  uint32_t padding_right_{0};
  uint32_t padding_bottom_{0};
  uint32_t padding_left_{0};
  bool padding_tf_same_{false};
  size_t input_height_{1};
  size_t input_width_{1};
  uint32_t groups_{1};
  size_t group_input_channels_{1};
  size_t input_channel_stride_{0};
  size_t group_output_channels_{1};
  size_t output_channel_stride_{0};
  size_t batch_size_{1};
  uint32_t kernel_height_{1};
  uint32_t kernel_width_{1};
  uint32_t dilation_height_{1};
  uint32_t dilation_width_{1};
  uint32_t subsampling_height_{1};
  uint32_t subsampling_width_{1};
  size_t next_input_height_{0};
  size_t next_input_width_{0};
  size_t next_batch_size_{0};
  float sparsity_{0.0f};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  bool depthwise_layout_{false};
  bool force_nhwc_input_{false};
  bool has_bias_{true};
  WeightsType weights_type_{WeightsType::Default};
  bool multithreaded_{false};
  size_t iterations_{1};
#if XNN_PLATFORM_JIT
  bool use_jit_{false};
#endif
  bool use_weights_cache_{false};
  bool transient_indirection_buffer_{false};
};
