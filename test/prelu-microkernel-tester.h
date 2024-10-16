// Copyright 2019 Google LLC
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
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/buffer.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "replicable_random_device.h"

class PReLUMicrokernelTester {
 public:
  PReLUMicrokernelTester& rows(size_t rows) {
    assert(rows != 0);
    this->rows_ = rows;
    return *this;
  }

  size_t rows() const {
    return this->rows_;
  }

  PReLUMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  size_t channels() const {
    return this->channels_;
  }

  PReLUMicrokernelTester& input_stride(size_t input_stride) {
    assert(input_stride != 0);
    this->input_stride_ = input_stride;
    return *this;
  }

  size_t input_stride() const {
    if (this->input_stride_ == 0) {
      return channels();
    } else {
      assert(this->input_stride_ >= channels());
      return this->input_stride_;
    }
  }

  PReLUMicrokernelTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return channels();
    } else {
      assert(this->output_stride_ >= channels());
      return this->output_stride_;
    }
  }

  PReLUMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  bool inplace() const {
    return this->inplace_;
  }

  PReLUMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f16_prelu_ukernel_fn prelu) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> w32dist(0.25f, 0.75f);

    xnnpack::Buffer<xnn_float16> x(channels() + (rows() - 1) * input_stride() + XNN_EXTRA_BYTES / sizeof(xnn_float16));
    xnnpack::Buffer<xnn_float16, XNN_ALLOCATION_ALIGNMENT> w(
        channels() + XNN_EXTRA_BYTES / sizeof(xnn_float16));
    xnnpack::Buffer<xnn_float16> y(channels() + (rows() - 1) * output_stride() + XNN_EXTRA_BYTES / sizeof(xnn_float16));
    xnnpack::Buffer<float> y_ref(channels() * rows());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), [&]() { return f32dist(rng); });
      std::generate(w.begin(), w.end(), [&]() { return w32dist(rng); });
      if (inplace()) {
        std::generate(y.begin(), y.end(), [&]() { return f32dist(rng); });
      }
      const xnn_float16* x_data = inplace() ? y.data() : x.data();

      // Compute reference results, without clamping.
      for (size_t n = 0; n < rows(); n++) {
        for (size_t c = 0; c < channels(); c++) {
          const float x_value = x_data[n * input_stride() + c];
          y_ref[n * channels() + c] = std::signbit(x_value) ?
              float(xnn_float16(x_value * w[c])) : x_value;  // What is going on here?
        }
      }

      // Call optimized micro-kernel.
      prelu(rows(), channels() * sizeof(xnn_float16),
        x_data, input_stride() * sizeof(xnn_float16),
        w.data(),
        y.data(), output_stride() * sizeof(xnn_float16));

      // Verify results.
      for (size_t n = 0; n < rows(); n++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_EQ(y[n * output_stride() + c], y_ref[n * channels() + c])
            << "at row " << n << " / " << rows()
            << ", channel " << c << " / " << channels();
        }
      }
    }
  }

  void Test(xnn_f32_prelu_ukernel_fn prelu) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> w32dist(0.25f, 0.75f);

    xnnpack::Buffer<float> x(channels() + (rows() - 1) * input_stride() + XNN_EXTRA_BYTES / sizeof(float));
    xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> w(channels() +
                                               XNN_EXTRA_BYTES / sizeof(float));
    xnnpack::Buffer<float> y(channels() + (rows() - 1) * output_stride() + XNN_EXTRA_BYTES / sizeof(float));
    xnnpack::Buffer<float> y_ref(channels() * rows());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), [&]() { return f32dist(rng); });
      std::generate(w.begin(), w.end(), [&]() { return w32dist(rng); });
      if (inplace()) {
        std::generate(y.begin(), y.end(), [&]() { return f32dist(rng); });
      }
      const float* x_data = inplace() ? y.data() : x.data();

      // Compute reference results, without clamping.
      for (size_t n = 0; n < rows(); n++) {
        for (size_t c = 0; c < channels(); c++) {
          const float x_value = x_data[n * input_stride() + c];
          y_ref[n * channels() + c] = std::signbit(x_value) ? x_value * w[c] : x_value;
        }
      }

      // Call optimized micro-kernel.
      prelu(rows(), channels() * sizeof(float),
        x_data, input_stride() * sizeof(float),
        w.data(),
        y.data(), output_stride() * sizeof(float));

      // Verify results.
      for (size_t n = 0; n < rows(); n++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_EQ(y[n * output_stride() + c], y_ref[n * channels() + c])
            << "at row " << n << " / " << rows()
            << ", channel " << c << " / " << channels();
        }
      }
    }
  }

 private:
  size_t rows_{1};
  size_t channels_{1};
  size_t input_stride_{0};
  size_t output_stride_{0};
  bool inplace_{false};
  size_t iterations_{15};
};
