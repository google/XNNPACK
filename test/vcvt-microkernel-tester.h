// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <fp16.h>

#include <xnnpack.h>
#include <xnnpack/params.h>


class VCvtMicrokernelTester {
 public:
  inline VCvtMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline VCvtMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f16_f32_vcvt_ukernel_function vcvt) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto distribution = std::uniform_real_distribution<float>(-100.0f, 100.0f);
    auto f32rng = std::bind(distribution, std::ref(rng));
    auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

    std::vector<uint16_t> input(batch_size() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<float> output(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f16rng));
      std::fill(output.begin(), output.end(), nanf(""));

      // Call optimized micro-kernel.
      vcvt(batch_size() * sizeof(float), input.data(), output.data(), nullptr /* params */);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_EQ(fp32_to_bits(output[i]), fp32_to_bits(fp16_ieee_to_fp32_value(input[i])))
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = 0x" << std::hex << std::setw(4) << std::setfill('0') << input[i];
      }
    }
  }

  void Test(xnn_f32_f16_vcvt_ukernel_function vcvt) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto distribution = std::uniform_real_distribution<float>(-100.0f, 100.0f);
    auto f32rng = std::bind(distribution, std::ref(rng));

    std::vector<float> input(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<uint16_t> output(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), UINT16_C(0x7E));

      // Call optimized micro-kernel.
      vcvt(batch_size() * sizeof(uint16_t), input.data(), output.data(), nullptr /* params */);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_EQ(output[i], fp16_ieee_from_fp32_value(input[i]))
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(input[i])
          << " (" << input[i] << ")";
      }
    }
  }

 private:
  size_t batch_size_ = 1;
  size_t iterations_ = 15;
};
