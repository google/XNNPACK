// Copyright 2020 Google LLC
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
#include <functional>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/params.h>



class IBilinearCHWMicrokernelTester {
 public:
  inline IBilinearCHWMicrokernelTester& pixels(uint32_t pixels) {
    assert(pixels >= 1);
    this->pixels_ = pixels;
    return *this;
  }

  inline uint32_t pixels() const {
    return this->pixels_;
  }

  inline IBilinearCHWMicrokernelTester& input_offset(uint32_t input_offset) {
    this->input_offset_ = input_offset;
    return *this;
  }

  inline uint32_t input_offset() const {
    return this->input_offset_;
  }

  inline IBilinearCHWMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_ibilinear_chw_ukernel_function ibilinear) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), rng);

    std::vector<const float*> indirection(pixels() * 4);
    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) + indirection.size());
    std::vector<float, AlignedAllocator<float, 64>> packed_horizontal_weights(pixels());
    std::vector<float, AlignedAllocator<float, 64>> packed_vertical_weights(pixels());
    std::vector<float> output(pixels());
    std::vector<float> output_ref(pixels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::generate(packed_horizontal_weights.begin(), packed_horizontal_weights.end(), std::ref(f32rng));
      std::generate(packed_vertical_weights.begin(), packed_vertical_weights.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), nanf(""));

      auto input_pixels = indirection.size();
      for (size_t i = 0; i < input_pixels; i++) {
        indirection[i] = input.data() + i - input_offset();
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);

      // Compute reference results.
      for (size_t i = 0; i < pixels(); i++) {
        const float alpha_h = packed_horizontal_weights[i];
        const float alpha_v = packed_vertical_weights[i];
        // `c * pixels() + i` because the output is NCHW.
        output_ref[i] =
          // `c * indirection.size()` because the input is NCHW.
          indirection[i * 4 + 0][input_offset()] * (1.0f - alpha_h) * (1.0f - alpha_v) +
          indirection[i * 4 + 1][input_offset()] * alpha_h * (1.0f - alpha_v) +
          indirection[i * 4 + 2][input_offset()] * (1.0f - alpha_h) * alpha_v +
          indirection[i * 4 + 3][input_offset()] * alpha_h * alpha_v;
      }

      // Call optimized micro-kernel.
      ibilinear(
        pixels(), indirection.data(), input_offset() * sizeof(float),
        packed_horizontal_weights.data(), packed_vertical_weights.data(),
        output.data());

      // Verify results.
      for (size_t i = 0; i < pixels(); i++) {
        ASSERT_NEAR(output_ref[i], output[i], std::abs(output_ref[i]) * 1.0e-4)
            << "i = " << i;
      }
    }
  }

 private:
  uint32_t pixels_{1};
  uint32_t input_offset_{0};
  size_t iterations_{3};
};
