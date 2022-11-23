// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include <xnnpack.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/params.h>

class TransposeMicrokernelTester {
 public:
  inline TransposeMicrokernelTester& element_size(size_t element_size) {
    assert(element_size != 0);
    this->element_size_ = element_size;
    return *this;
  }

  inline size_t element_size() const { return this->element_size_; }

  inline TransposeMicrokernelTester& block_height(size_t block_height) {
    assert(block_height != 0);
    this->block_height_ = block_height;
    return *this;
  }

  inline size_t block_height() const { return this->block_height_; }

  inline TransposeMicrokernelTester& block_width(size_t block_width) {
    assert(block_width != 0);
    this->block_width_ = block_width;
    return *this;
  }

  inline size_t block_width() const { return this->block_width_; }

  inline TransposeMicrokernelTester& input_stride(size_t input_stride) {
    this->input_stride_ = input_stride;
    return *this;
  }

  inline size_t input_stride() const { return this->input_stride_; }

  inline TransposeMicrokernelTester& output_stride(size_t output_stride) {
    this->output_stride_ = output_stride;
    return *this;
  }

  inline size_t output_stride() const { return this->output_stride_; }

  inline TransposeMicrokernelTester& input_element_stride(size_t input_element_stride) {
    assert(input_element_stride >=  element_size_);
    this->input_element_stride_ = input_element_stride;
    return *this;
  }

  inline size_t input_element_stride() const {
    if (input_element_stride_ == 0) {
      return element_size_;
    } else {
      return input_element_stride_;
    }
  }

  inline TransposeMicrokernelTester& output_element_stride(size_t output_element_stride) {
    assert(output_element_stride >=  element_size_);
    this->output_element_stride_ = output_element_stride;
    return *this;
  }

  inline size_t output_element_stride() const {
    if (output_element_stride_ == 0) {
      return element_size_;
    } else {
      return output_element_stride_;
    }
  }

  inline TransposeMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const { return this->iterations_; }

  void Test(xnn_transposev_ukernel_fn transpose) const {
    std::vector<uint8_t> input(input_stride() * block_height() * input_element_stride() + XNN_EXTRA_BYTES);
    std::vector<uint8_t> output(output_stride() * block_width() * output_element_stride());
    std::iota(input.begin(), input.end(), 0);
    std::fill(output.begin(), output.end(), UINT8_C(0xA5));

    // Call optimized micro-kernel.
    transpose(input.data(),
              output.data(),
              input_stride() * input_element_stride(),
              output_stride() * output_element_stride(),
              input_element_stride(),
              output_element_stride(),
              element_size(),
              block_width(),
              block_height());

    // Verify results.
    for (size_t c = 0; c < block_width(); c++) {
      for (size_t r = 0; r < block_height(); r++) {
        EXPECT_EQ(std::memcmp(&input[input_element_stride() * (c+ r * input_stride())],
                              &output[output_element_stride() * (r + c * output_stride())],
                              element_size()), 0)
            << "at row " << r << " / " << block_height()
            << ", at column " << c << " / " << block_width();
      }
    }
  }

  void Test(xnn_x64_transposec_ukernel_fn transpose, const xnn_init_x64_transpose_params_fn init_params = nullptr) const {
    std::vector<uint64_t> input(input_stride() * output_stride() + XNN_EXTRA_BYTES / sizeof(uint64_t));
    std::vector<uint64_t> output(input_stride() * output_stride());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::iota(input.begin(), input.end(), 0);
      std::fill(output.begin(), output.end(), UINT64_C(0xBADC0FFEE0DDF00D));

      union xnn_x64_transpose_params params;
      if (init_params != nullptr) {
        init_params(&params);
      }

      // Call optimized micro-kernel.
      transpose(input.data(),
                output.data(),
                input_stride() * sizeof(uint64_t),
                output_stride() * sizeof(uint64_t),
                block_width(),
                block_height(),
                &params);

      // Verify results.
      for (size_t c = 0; c < block_width(); c++) {
        for (size_t r = 0; r < block_height(); r++) {
          EXPECT_EQ(input[c + r * input_stride()], output[r + c * output_stride()])
              << "at row " << r << " / " << block_height()
              << ", at column " << c << " / " << block_width();
        }
      }
    }
  }

  void Test(xnn_x32_transposec_ukernel_fn transpose, const xnn_init_x32_transpose_params_fn init_params = nullptr) const {
    std::vector<uint32_t> input(input_stride() * output_stride() + XNN_EXTRA_BYTES / sizeof(uint32_t));
    std::vector<uint32_t> output(input_stride() * output_stride());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::iota(input.begin(), input.end(), 0);
      std::fill(output.begin(), output.end(), UINT32_C(0xDEADBEEF));

      union xnn_x32_transpose_params params;
      if (init_params != nullptr) {
        init_params(&params);
      }

      // Call optimized micro-kernel.
      transpose(input.data(),
                output.data(),
                input_stride() * sizeof(uint32_t),
                output_stride() * sizeof(uint32_t),
                block_width(),
                block_height(),
                &params);

      // Verify results.
      for (size_t c = 0; c < block_width(); c++) {
        for (size_t r = 0; r < block_height(); r++) {
          EXPECT_EQ(input[c + r * input_stride()], output[r + c * output_stride()])
              << "at row " << r << " / " << block_height()
              << ", at column " << c << " / " << block_width();
        }
      }
    }
  }
  void Test(xnn_x24_transposec_ukernel_fn transpose, const xnn_init_x24_transpose_params_fn init_params = nullptr) const {
    std::vector<uint8_t> input(input_stride() * output_stride() * element_size() + XNN_EXTRA_BYTES);
    std::vector<uint8_t> output(input_stride() * output_stride() * element_size());
    std::iota(input.begin(), input.end(), 0);
    std::fill(output.begin(), output.end(), UINT8_C(0xA5));

    union xnn_x24_transpose_params params;
    if (init_params != nullptr) {
      init_params(&params);
    }
    // Call optimized micro-kernel.
    transpose(input.data(),
              output.data(),
              input_stride() * element_size(),
              output_stride() * element_size(),
              block_width(),
              block_height(),
              &params);

    // Verify results.
    for (size_t c = 0; c < block_width(); c++) {
      for (size_t r = 0; r < block_height(); r++) {
        EXPECT_EQ(std::memcmp(&input[element_size() * (c+ r * input_stride())],
                              &output[element_size() * (r + c * output_stride())],
                              element_size()), 0)
            << "at row " << r << " / " << block_height()
            << ", at column " << c << " / " << block_width();
      }
    }
  }

  void Test(xnn_x16_transposec_ukernel_fn transpose, const xnn_init_x16_transpose_params_fn init_params = nullptr) const {
    std::vector<uint16_t> input(input_stride() * output_stride() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> output(input_stride() * output_stride());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::iota(input.begin(), input.end(), 0);
      std::fill(output.begin(), output.end(), UINT16_C(0xDEAD));

      union xnn_x16_transpose_params params;
      if (init_params != nullptr) {
        init_params(&params);
      }
      // Call optimized micro-kernel.
      transpose(input.data(),
                output.data(),
                input_stride() * sizeof(uint16_t),
                output_stride() * sizeof(uint16_t),
                block_width(),
                block_height(),
                &params);

      // Verify results.
      for (size_t c = 0; c < block_width(); c++) {
        for (size_t r = 0; r < block_height(); r++) {
          ASSERT_EQ(input[c + r * input_stride()], output[r + c * output_stride()])
              << "at row " << r << " / " << block_height()
              << ", at column " << c << " / " << block_width();
        }
      }
    }
  }

  void Test(xnn_x8_transposec_ukernel_fn transpose, const xnn_init_x8_transpose_params_fn init_params = nullptr) const {
    std::vector<uint8_t> input(input_stride() * output_stride() + XNN_EXTRA_BYTES);
    std::vector<uint8_t> output(input_stride() * output_stride());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::iota(input.begin(), input.end(), 0);
      std::fill(output.begin(), output.end(), UINT8_C(0xA5));

      union xnn_x8_transpose_params params;
      if (init_params != nullptr) {
        init_params(&params);
      }
      // Call optimized micro-kernel.
      transpose(input.data(),
                output.data(),
                input_stride() * sizeof(uint8_t),
                output_stride() * sizeof(uint8_t),
                block_width(),
                block_height(),
                &params);

      // Verify results.
      for (size_t c = 0; c < block_width(); c++) {
        for (size_t r = 0; r < block_height(); r++) {
          ASSERT_EQ((int)input[c + r * input_stride()], (int)output[r + c * output_stride()])
              << "at row " << r << " / " << block_height()
              << ", at column " << c << " / " << block_width();
        }
      }
    }
  }

 private:
  size_t element_size_ = 1;
  size_t input_stride_ = 1;
  size_t output_stride_ = 1;
  size_t input_element_stride_ = 0;
  size_t output_element_stride_ = 0;
  size_t block_height_ = 1;
  size_t block_width_ = 1;
  size_t iterations_ = 15;
};
