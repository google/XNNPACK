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
#include <functional>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/params.h>


class IBilinearMicrokernelTester {
 public:
  inline IBilinearMicrokernelTester& pixels(uint32_t pixels) {
    assert(pixels >= 1);
    this->pixels_ = pixels;
    return *this;
  }

  inline uint32_t pixels() const {
    return this->pixels_;
  }

  inline IBilinearMicrokernelTester& channels(uint32_t channels) {
    assert(channels >= 1);
    this->channels_ = channels;
    return *this;
  }

  inline uint32_t channels() const {
    return this->channels_;
  }

  inline IBilinearMicrokernelTester& input_offset(uint32_t input_offset) {
    this->input_offset_ = input_offset;
    return *this;
  }

  inline uint32_t input_offset() const {
    return this->input_offset_;
  }

  inline IBilinearMicrokernelTester& output_stride(uint32_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  inline uint32_t output_stride() const {
    if (this->output_stride_ == 0) {
      return channels();
    } else {
      assert(this->output_stride_ >= channels());
      return this->output_stride_;
    }
  }

  inline IBilinearMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_ibilinear_ukernel_function ibilinear) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), rng);

    std::vector<const float*> indirection(pixels() * 4);
    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) + indirection.size() * channels());
    std::vector<float, AlignedAllocator<float, 64>> packed_weights(pixels() * 2);
    std::vector<float> output((pixels() - 1) * output_stride() + channels());
    std::vector<float> output_ref(pixels() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::generate(packed_weights.begin(), packed_weights.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), nanf(""));

      for (size_t i = 0; i < indirection.size(); i++) {
        indirection[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);

      // Compute reference results.
      for (size_t i = 0; i < pixels(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const float alpha_h = packed_weights[i * 2 + 0];
          const float alpha_v = packed_weights[i * 2 + 1];
          output_ref[i * channels() + c] =
            indirection[i * 4 + 0][c + input_offset()] * (1.0f - alpha_h) * (1.0f - alpha_v) +
            indirection[i * 4 + 1][c + input_offset()] * alpha_h * (1.0f - alpha_v) +
            indirection[i * 4 + 2][c + input_offset()] * (1.0f - alpha_h) * alpha_v +
            indirection[i * 4 + 3][c + input_offset()] * alpha_h * alpha_v;
        }
      }

      // Call optimized micro-kernel.
      ibilinear(
        pixels(), channels() * sizeof(float),
        indirection.data(), input_offset() * sizeof(float),
        packed_weights.data(), output.data(),
        (output_stride() - channels()) * sizeof(float));

      // Verify results.
      for (size_t i = 0; i < pixels(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_NEAR(
              output_ref[i * channels() + c],
              output[i * output_stride() + c],
              std::abs(output_ref[i * channels() + c]) * 1.0e-4)
            << "i = " << i << ", channel = " << c;
        }
      }
    }
  }

 private:
  uint32_t channels_{1};
  uint32_t pixels_{1};
  uint32_t output_stride_{0};
  uint32_t input_offset_{0};
  size_t iterations_{3};
};
