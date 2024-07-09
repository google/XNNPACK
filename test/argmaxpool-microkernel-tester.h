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
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/microfnptr.h"
#include "replicable_random_device.h"

class ArgMaxPoolMicrokernelTester {
 public:
  enum class Variant {
    Native,
    Scalar,
  };

  ArgMaxPoolMicrokernelTester& output_pixels(size_t output_pixels) {
    assert(output_pixels != 0);
    this->output_pixels_ = output_pixels;
    return *this;
  }

  size_t output_pixels() const {
    return this->output_pixels_;
  }

  ArgMaxPoolMicrokernelTester& step(size_t step) {
    assert(step != 0);
    this->step_ = step;
    return *this;
  }

  size_t step() const {
    return this->step_;
  }

  ArgMaxPoolMicrokernelTester& input_offset(size_t input_offset) {
    assert(input_offset != 0);
    this->input_offset_ = input_offset;
    return *this;
  }

  size_t input_offset() const {
    return this->input_offset_;
  }

  ArgMaxPoolMicrokernelTester& pooling_elements(size_t pooling_elements) {
    assert(pooling_elements != 0);
    this->pooling_elements_ = pooling_elements;
    return *this;
  }

  size_t pooling_elements() const {
    return this->pooling_elements_;
  }

  size_t packed_pooling_elements() const {
    if (pooling_elements() <= primary_pooling_tile()) {
      return primary_pooling_tile();
    } else {
      return (pooling_elements() - primary_pooling_tile()) % incremental_pooling_tile() == 0 ? pooling_elements() : ((pooling_elements() - primary_pooling_tile()) / incremental_pooling_tile() + 1) * incremental_pooling_tile() + primary_pooling_tile();
    }
  }

  ArgMaxPoolMicrokernelTester& pooling_tile(size_t primary_tile) {
    assert(primary_tile != 0);
    this->primary_pooling_tile_ = primary_tile;
    this->incremental_pooling_tile_ = 0;
    return *this;
  }

  ArgMaxPoolMicrokernelTester& pooling_tile(size_t primary_tile, size_t incremental_tile) {
    assert(primary_tile != 0);
    this->primary_pooling_tile_ = primary_tile;
    this->incremental_pooling_tile_ = incremental_tile;
    return *this;
  }

  ArgMaxPoolMicrokernelTester& primary_pooling_tile(size_t primary_pooling_tile) {
    assert(primary_pooling_tile != 0);
    this->primary_pooling_tile_ = primary_pooling_tile;
    return *this;
  }

  size_t primary_pooling_tile() const {
    return this->primary_pooling_tile_;
  }

  ArgMaxPoolMicrokernelTester& incremental_pooling_tile(size_t incremental_pooling_tile) {
    assert(incremental_pooling_tile != 0);
    this->incremental_pooling_tile_ = incremental_pooling_tile;
    return *this;
  }

  size_t incremental_pooling_tile() const {
    return this->incremental_pooling_tile_;
  }

  ArgMaxPoolMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  size_t channels() const {
    return this->channels_;
  }

  ArgMaxPoolMicrokernelTester& output_stride(size_t output_stride) {
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

  ArgMaxPoolMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_argmaxpool_unipass_ukernel_fn argmaxpool, Variant variant = Variant::Native) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;

    std::vector<const float*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      ((output_pixels() - 1) * step() + pooling_elements()) * channels());
    std::vector<float> output((output_pixels() - 1) * output_stride() + channels());
    std::vector<uint32_t> index(output_pixels() * channels());
    std::vector<float> output_ref(output_pixels() * channels());
    std::vector<uint32_t> index_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), nanf(""));

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);

      // Compute reference results, without clamping.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float max_value = indirect_input[x * step()][c + input_offset()];
          uint32_t max_index = 0;
          for (size_t p = 0; p < pooling_elements(); p++) {
            const float value = indirect_input[x * step() + p][c + input_offset()];
            if (value > max_value) {
              max_value = value;
              max_index = p;
            }
          }
          output_ref[x * channels() + c] = max_value;
          index_ref[x * channels() + c] = max_index;
        }
      }

      // Call optimized micro-kernel.
      argmaxpool(output_pixels(), pooling_elements(), channels(),
        indirect_input.data(), input_offset() * sizeof(float), output.data(), index.data(),
        step() * sizeof(void*),
        (output_stride() - channels()) * sizeof(float));

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_EQ(output_ref[x * channels() + c], output[x * output_stride() + c])
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_EQ(
              indirect_input[x * step() + index_ref[x * channels() + c]][c + input_offset()],
              indirect_input[x * step() + index[x * channels() + c]][c + input_offset()])
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_EQ(index_ref[x * channels() + c], index[x * channels() + c])
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
        }
      }
    }
  }

  void Test(xnn_f32_argmaxpool_multipass_ukernel_fn argmaxpool, Variant variant = Variant::Native) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;

    std::vector<const float*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      ((output_pixels() - 1) * step() + pooling_elements()) * channels());
    std::vector<float> output((output_pixels() - 1) * output_stride() + channels());
    std::vector<uint32_t> index(output_pixels() * channels());
    std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> index_buffer(
      channels() + XNN_EXTRA_BYTES / sizeof(uint32_t));
    std::vector<float, AlignedAllocator<float, 64>> output_buffer(
      channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output_ref(output_pixels() * channels());
    std::vector<uint32_t> index_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), nanf(""));

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);

      // Compute reference results, without clamping.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float max_value = indirect_input[x * step()][c + input_offset()];
          uint32_t max_index = 0;
          for (size_t p = 0; p < pooling_elements(); p++) {
            const float value = indirect_input[x * step() + p][c + input_offset()];
            if (value > max_value) {
              max_value = value;
              max_index = p;
            }
          }
          output_ref[x * channels() + c] = max_value;
          index_ref[x * channels() + c] = max_index;
        }
      }

      // Call optimized micro-kernel.
      argmaxpool(output_pixels(), pooling_elements(), channels(),
        indirect_input.data(), input_offset() * sizeof(float),
        output_buffer.data(), index_buffer.data(),
        output.data(), index.data(),
        (step() - (packed_pooling_elements() - incremental_pooling_tile())) * sizeof(void*),
        (output_stride() - channels()) * sizeof(float));

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_EQ(output_ref[x * channels() + c], output[x * output_stride() + c])
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_EQ(
              indirect_input[x * step() + index_ref[x * channels() + c]][c + input_offset()],
              indirect_input[x * step() + index[x * channels() + c]][c + input_offset()])
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_EQ(index_ref[x * channels() + c], index[x * channels() + c])
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
        }
      }
    }
  }

 private:
  size_t output_pixels_{1};
  size_t pooling_elements_{1};
  size_t channels_{1};
  size_t input_offset_{0};
  size_t step_{1};
  size_t primary_pooling_tile_{1};
  size_t incremental_pooling_tile_{1};
  size_t output_stride_{0};
  size_t iterations_{3};
};
