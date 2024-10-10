// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"
#include "replicable_random_device.h"

class ReduceWindowMicrokernelTester {
 public:
  enum class OpType {
    Sum,
  };

  ReduceWindowMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  size_t batch_size() const {
    return this->batch_size_;
  }

 ReduceWindowMicrokernelTester& init_value(float init_value) {
    assert(init_value != 0);
    this->init_value_ = init_value;
    return *this;
  }

  float init_value() const {
    return this->init_value_;
  }

 ReduceWindowMicrokernelTester& padding(int64_t* padding) {
    assert(padding != 0);
    this->padding_[0] = padding[0];
    this->padding_[1] = padding[1];
    return *this;
  }

  int64_t padding_high() const {
    return this->padding_[0];
  }

  int64_t padding_low() const {
    return this->padding_[1];
  }

 ReduceWindowMicrokernelTester& base_dilation(int64_t base_dilation) {
    assert(base_dilation != 0);
    this->base_dilation_ = base_dilation;
    return *this;
  }

  int64_t base_dilation() const {
    return this->base_dilation_;
  }

 ReduceWindowMicrokernelTester& window_dilation(int64_t window_dilation) {
    assert(window_dilation != 0);
    this->window_dilation_ = window_dilation;
    return *this;
  }

  int64_t window_dilation() const {
    return this->window_dilation_;
  }

 ReduceWindowMicrokernelTester& window_dimension(int64_t window_dimension) {
    assert(window_dimension != 0);
    this->window_dimension_ = window_dimension;
    return *this;
  }

  int64_t window_dimension() const {
    return this->window_dimension_;
  }

 ReduceWindowMicrokernelTester& window_stride(int64_t window_stride) {
    assert(window_stride != 0);
    this->window_stride_ = window_stride;
    return *this;
  }

  int64_t window_stride() const {
    return this->window_stride_;
  }

  ReduceWindowMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_rw_ukernel_fn reduce_window, OpType op_type, xnn_init_f32_default_params_fn init_params = nullptr) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;

    std::vector<float> input(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });

      int64_t padding[2] = {padding_high(), padding_low()};

      // Compute reference results.
      int64_t size = batch_size();

      int64_t padded_size = size + (size - 1) * (base_dilation() - 1) + padding[0] + padding[1];
      int64_t output_size = (padded_size - (window_dimension() - 1) * window_dilation() - 1) / window_stride() + 1;

      float output_ref[output_size];

      for (int64_t i = 0; i < output_size; i++) {
              float sum = init_value();
              for (int64_t k = 0; k < window_dimension(); k++) {
                  int64_t window_row = i * window_stride() + k * window_dilation();
                  if (window_row < padding[0] || 
                      window_row >= padded_size - padding[1] || 
                      (window_row - padding[0]) % base_dilation() != 0) {
                      sum += init_value();
                      continue;
                  }
                  window_row = (window_row - padding[0]) / base_dilation();
                  sum += input[window_row]; 
              }
              output_ref[i] = sum;   
      }

      // Prepare parameters.
      xnn_f32_default_params params;
      if (init_params != nullptr) {
        init_params(&params);
      }

      // Call optimized micro-kernel.
      float output[output_size];
      reduce_window(batch_size() * sizeof(float), input.data(), init_value(), padding, base_dilation(),
              window_dilation(), window_dimension(), window_stride(), output, init_params != nullptr ? &params : nullptr);

      // Verify results.
      switch (op_type) {
        case OpType::Sum:
        for (size_t i = 0; i < output_size; i++){
          EXPECT_EQ(output[i], output_ref[i])<< "with batch " << batch_size();
        }
        break;
      }
    }
  }

 private:
  size_t batch_size_{1};
  size_t iterations_{15};
  float init_value_{0.0};
  int64_t padding_[2] = {1,1};
  int64_t base_dilation_{1};
  int64_t window_dimension_{1};
  int64_t window_dilation_{1};
  int64_t window_stride_{1};
};
