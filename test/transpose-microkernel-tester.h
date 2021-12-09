// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <xnnpack.h>
#include <xnnpack/params.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <gtest/gtest.h>


class TransposeMicrokernelTester {
 public:
  inline TransposeMicrokernelTester& height(size_t height) {
    assert(height != 0);
    this->height_ = height;
    return *this;
  }

  inline size_t height() const { return this->height_; }

  inline TransposeMicrokernelTester& width(size_t width) {
    assert(width != 0);
    this->width_ = width;
    return *this;
  }

  inline size_t width() const { return this->width_; }

  inline TransposeMicrokernelTester& h_start(size_t h_start) {
    this->h_start_ = h_start;
    return *this;
  }

  inline size_t h_start() const { return this->h_start_; }

  inline TransposeMicrokernelTester& w_start(size_t w_start) {
    this->w_start_ = w_start;
    return *this;
  }

  inline size_t w_start() const { return this->w_start_; }

  inline TransposeMicrokernelTester& h_end(size_t h_end) {
    this->h_end_ = h_end;
    return *this;
  }

  inline size_t h_end() const { return this->h_end_; }

  inline TransposeMicrokernelTester& w_end(size_t w_end) {
    this->w_end_ = w_end;
    return *this;
  }

  inline size_t w_end() const { return this->w_end_; }

  inline TransposeMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const { return this->iterations_; }

  void Test(xnn_x32_transpose_ukernel_function transpose) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u32rng = std::bind(std::uniform_int_distribution<uint32_t>(), rng);

    std::vector<uint32_t> input(height() * width() + XNN_EXTRA_BYTES / sizeof(uint32_t));
    size_t output_height = w_end() - w_start();
    size_t output_width = h_end() - h_start();
    std::vector<uint32_t> output(output_height * output_width);
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u32rng));
      std::fill(output.begin(), output.end(), 0);

      // Call optimized micro-kernel.
      transpose(input.data() + w_start() + h_start() * width(),
                output.data(),
                width() * sizeof(uint32_t),
                output_width * sizeof(uint32_t),
                (w_end() - w_start()) * sizeof(uint32_t),
                (h_end() - h_start()));

      // Verify results.
      for (size_t r = w_start(); r < w_end(); r++) {
        for (size_t c = h_start(); c < h_end(); c++) {
          EXPECT_EQ(input[r + c * width()], output[c - h_start() + (r - w_start()) * output_width])
              << "at row " << r << " / " << width()
              << ", at column " << c << " / " << height();
        }
      }
    }
  }

 private:
  size_t h_start_ = 1;
  size_t h_end_ = 1;
  size_t w_start_ = 1;
  size_t w_end_ = 1;
  size_t height_ = 1;
  size_t width_ = 1;
  size_t iterations_ = 15;
};
